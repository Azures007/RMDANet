"""Guarded Adaptation (GA).

Online collapse detection for pseudo-label interventions.  Two families of
signals are monitored every epoch:

  * pseudo-label statistics: class-distribution entropy, teacher confidence,
    high-confidence coverage (cheap, accumulated inside the training epoch);
  * probe signals: source-domain accuracy (catastrophic-forgetting probe) and
    target prediction flip rate between epochs (temporal stability probe).

When a sustained degradation is detected the guard

  1. rolls the model back to the last healthy-epoch snapshot,
  2. disables the interventions (RGS weight and reliability operators),
  3. lets plain TAPDA finish the remaining epochs.

Worst case therefore equals the plain TAPDA baseline (no-harm property), while
healthy runs keep the original gains.
"""

from __future__ import annotations

import math

import torch

_MONITOR = {
    "hist": None,
    "conf_sum": 0.0,
    "n": 0,
    "high": 0,
}


def guard_reset(num_classes=None, device=None):
    if num_classes is None:
        return
    _MONITOR["hist"] = torch.zeros(int(num_classes), device=device)
    _MONITOR["conf_sum"] = 0.0
    _MONITOR["n"] = 0
    _MONITOR["high"] = 0


@torch.no_grad()
def guard_accumulate(prob, labels, high_conf=0.9):
    if _MONITOR["hist"] is None:
        return
    p = prob.detach().float().ravel()
    y = labels.detach().long().ravel()
    ones = torch.ones_like(y, dtype=torch.float)
    _MONITOR["hist"].index_add_(0, y, ones)
    _MONITOR["conf_sum"] += float(p.sum().item())
    _MONITOR["n"] += int(p.numel())
    _MONITOR["high"] += int((p >= high_conf).sum().item())


def guard_stats():
    n = max(1, _MONITOR["n"])
    hist = _MONITOR["hist"].clamp_min(0) if _MONITOR["hist"] is not None else None
    if hist is None or float(hist.sum().item()) <= 0:
        return None
    frac = hist / hist.sum().clamp_min(1.0)
    entropy = float(-(frac * frac.clamp_min(1e-8).log()).sum().item())
    max_entropy = math.log(max(2, len(frac)))
    return {
        "coverage": _MONITOR["high"] / n,
        "conf_mean": _MONITOR["conf_sum"] / n,
        "entropy": entropy / max_entropy,
    }


@torch.no_grad()
def probe_source_acc(G, F1, src_loader, device, max_batches=10):
    """Accuracy of the student on a small source batch (source labels are given)."""
    was_training = G.training
    G.eval()
    F1.eval()
    correct = 0
    total = 0
    for i, (data, label) in enumerate(src_loader):
        if i >= max_batches:
            break
        data = data.to(device)
        label = label.to(device)
        feats = G(data)
        feats = feats[0] if isinstance(feats, tuple) else feats
        out = F1(feats)
        logits = out[0] if isinstance(out, tuple) else out
        pred = logits.argmax(dim=1)
        correct += int((pred == label).sum().item())
        total += int(label.numel())
    if was_training:
        G.train()
        F1.train()
    return correct / max(1, total)


@torch.no_grad()
def probe_target_preds(G, F1, tar_loader, device, max_batches=8):
    """Predictions on a small target batch keyed by target id (order independent)."""
    was_training = F1.training
    F1.eval()
    out = {}
    for i, batch in enumerate(tar_loader):
        if i >= max_batches:
            break
        data, ids = batch[0], batch[2]
        data = data.to(device)
        feats = G(data)
        feats = feats[0] if isinstance(feats, tuple) else feats
        logits = F1(feats)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = logits.argmax(dim=1).cpu()
        for idx, p in zip(ids.cpu().tolist(), pred.tolist()):
            out[int(idx)] = int(p)
    if was_training:
        F1.train()
    return out


class Guard:
    def __init__(self, num_classes, device, start_epoch=15, patience=2,
                 entropy_drop=0.25, conf_drop=0.03, coverage_drop=0.50,
                 source_drop=0.05, flip_threshold=0.35,
                 keep=3, enabled=True):
        self.num_classes = int(num_classes)
        self.device = device
        self.start_epoch = int(start_epoch)
        self.patience = int(patience)
        self.entropy_drop = float(entropy_drop)
        self.conf_drop = float(conf_drop)
        self.coverage_drop = float(coverage_drop)
        self.source_drop = float(source_drop)
        self.flip_threshold = float(flip_threshold)
        self.keep = int(keep)
        self.enabled = bool(enabled)

        self.ema_entropy = None
        self.ema_conf = None
        self.ema_coverage = None
        self.best_source_acc = None
        self.prev_preds = None
        self.bad_streak = 0
        self.snapshots = []
        self.events = []
        self.log = []

    def _snapshot(self, epoch, models):
        state = {
            name: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            for name, model in models.items()
        }
        self.snapshots.append((epoch, state))
        if len(self.snapshots) > self.keep:
            self.snapshots.pop(0)

    def _update_ema(self, stats):
        a = 0.8
        if self.ema_entropy is None:
            self.ema_entropy = stats["entropy"]
            self.ema_conf = stats["conf_mean"]
            self.ema_coverage = stats["coverage"]
            return
        self.ema_entropy = a * self.ema_entropy + (1 - a) * stats["entropy"]
        self.ema_conf = a * self.ema_conf + (1 - a) * stats["conf_mean"]
        self.ema_coverage = a * self.ema_coverage + (1 - a) * stats["coverage"]

    def step(self, epoch, models, source_acc=None, tar_preds=None):
        if not self.enabled:
            return None
        stats = guard_stats()
        if stats is None:
            return None

        flip = None
        if tar_preds is not None and self.prev_preds is not None:
            keys = set(tar_preds) & set(self.prev_preds)
            if keys:
                flips = sum(1 for k in keys if tar_preds[k] != self.prev_preds[k])
                flip = flips / len(keys)
        if tar_preds is not None:
            self.prev_preds = tar_preds

        if source_acc is not None:
            if self.best_source_acc is None or source_acc > self.best_source_acc:
                self.best_source_acc = source_acc

        self.log.append({
            "epoch": epoch,
            **stats,
            "source_acc": source_acc,
            "flip": flip,
        })

        if epoch < self.start_epoch:
            self._snapshot(epoch, models)
            self._update_ema(stats)
            return None

        reasons = []
        if self.ema_entropy is not None and stats["entropy"] < self.ema_entropy * (1 - self.entropy_drop):
            reasons.append("entropy_drop")
        if self.ema_conf is not None and stats["conf_mean"] < self.ema_conf - self.conf_drop:
            reasons.append("conf_drop")
        if (self.ema_coverage is not None and self.ema_coverage > 0
                and stats["coverage"] < self.ema_coverage * self.coverage_drop):
            reasons.append("coverage_drop")
        if (source_acc is not None and self.best_source_acc is not None
                and source_acc < self.best_source_acc - self.source_drop):
            reasons.append("source_forgetting")
        if flip is not None and flip > self.flip_threshold:
            reasons.append("prediction_unstable")

        if reasons:
            self.bad_streak += 1
        else:
            self.bad_streak = 0

        if epoch % 5 == 0:
            print(
                f"[guard] ep={epoch} cov={stats['coverage']:.3f} conf={stats['conf_mean']:.4f} "
                f"H={stats['entropy']:.3f} src={'-' if source_acc is None else f'{source_acc:.4f}'} "
                f"flip={'-' if flip is None else f'{flip:.3f}'} bad={self.bad_streak}"
            )

        if self.bad_streak >= self.patience and self.snapshots:
            restore_epoch, _ = self.snapshots[-1]
            info = {"epoch": epoch, "restore": restore_epoch, "reasons": reasons,
                    "source_acc": source_acc, "flip": flip}
            self.events.append(info)
            self.enabled = False
            self.prev_preds = None
            return info

        if self.bad_streak == 0:
            self._snapshot(epoch, models)
            self._update_ema(stats)
        return None

    @staticmethod
    def load_state(state, models):
        for name, module in models.items():
            module.load_state_dict(state[name], strict=True)
