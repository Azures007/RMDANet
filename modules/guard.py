"""Guarded Adaptation (GA), v3.

Monitors two families of signals every epoch:

  * pseudo-label statistics accumulated inside the training epoch
    (class-distribution entropy, teacher confidence, coverage);
  * probe signals measured on a FIXED cached batch (so they are comparable
    across epochs): class-balanced source accuracy (catastrophic-forgetting
    probe) and target prediction flip rate (temporal-stability probe).

On a sustained degradation the guard does NOT restore old weights (that proved
unstable).  It simply switches the interventions off for the remaining epochs:
RGS weight -> 0, reliability operators -> off, and SSTI -> off for the final
evaluation.  The run then continues as plain TAPDA, so the worst case equals
the plain TAPDA baseline (no-harm property).
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


def _forward_logits(G, F1, data):
    feats = G(data)
    feats = feats[0] if isinstance(feats, tuple) else feats
    out = F1(feats)
    return out[0] if isinstance(out, tuple) else out


def build_probe_cache(src_loader, tar_loader, n_src=6, n_tar=6):
    """Freeze a small, fixed subset of source/target batches for epoch probes."""
    cache = {"src": [], "tar": []}
    for i, (data, label) in enumerate(src_loader):
        if i >= n_src:
            break
        cache["src"].append((data.detach().cpu(), label.detach().cpu()))
    for i, batch in enumerate(tar_loader):
        if i >= n_tar:
            break
        cache["tar"].append((batch[0].detach().cpu(), batch[2].detach().cpu()))
    return cache


@torch.no_grad()
def probe_cached(G, F1, cache, device):
    """Return (class-balanced source accuracy, {target_id: prediction})."""
    was_training = G.training
    G.eval()
    F1.eval()

    cls_correct = torch.zeros(256)
    cls_total = torch.zeros(256)
    for data, label in cache["src"]:
        data = data.to(device)
        label = label.to(device)
        pred = _forward_logits(G, F1, data).argmax(dim=1)
        for c in label.unique().tolist():
            mask = label.eq(int(c))
            cls_correct[int(c)] += float((pred[mask] == label[mask]).sum().item())
            cls_total[int(c)] += float(mask.sum().item())
    seen = cls_total > 0
    source_acc = float((cls_correct[seen] / cls_total[seen]).mean().item()) if seen.any() else 0.0

    preds = {}
    for data, ids in cache["tar"]:
        data = data.to(device)
        pred = _forward_logits(G, F1, data).argmax(dim=1).cpu()
        for idx, p in zip(ids.tolist(), pred.tolist()):
            preds[int(idx)] = int(p)

    if was_training:
        G.train()
        F1.train()
    return source_acc, preds


class Guard:
    def __init__(self, num_classes, device, start_epoch=15, patience=2,
                 entropy_drop=0.25, conf_drop=0.03, coverage_drop=0.50,
                 source_drop=0.03, flip_threshold=0.25, enabled=True):
        self.start_epoch = int(start_epoch)
        self.patience = int(patience)
        self.entropy_drop = float(entropy_drop)
        self.conf_drop = float(conf_drop)
        self.coverage_drop = float(coverage_drop)
        self.source_drop = float(source_drop)
        self.flip_threshold = float(flip_threshold)
        self.enabled = bool(enabled)

        self.ema_entropy = None
        self.ema_conf = None
        self.ema_coverage = None
        self.best_source_acc = None
        self.prev_preds = None
        self.bad_streak = 0
        self.events = []
        self.log = []

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

    def step(self, epoch, source_acc=None, tar_preds=None):
        if not self.enabled:
            return None
        stats = guard_stats()
        if stats is None:
            return None

        flip = None
        if tar_preds is not None and self.prev_preds is not None:
            keys = set(tar_preds) & set(self.prev_preds)
            if keys:
                flip = sum(1 for k in keys if tar_preds[k] != self.prev_preds[k]) / len(keys)
        if tar_preds is not None:
            self.prev_preds = tar_preds

        if source_acc is not None:
            if self.best_source_acc is None or source_acc > self.best_source_acc:
                self.best_source_acc = source_acc

        self.log.append({"epoch": epoch, **stats, "source_acc": source_acc, "flip": flip})

        if epoch < self.start_epoch:
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

        self.bad_streak = self.bad_streak + 1 if reasons else 0

        if epoch % 5 == 0:
            print(
                f"[guard] ep={epoch} cov={stats['coverage']:.3f} conf={stats['conf_mean']:.4f} "
                f"H={stats['entropy']:.3f} src={'-' if source_acc is None else f'{source_acc:.4f}'} "
                f"flip={'-' if flip is None else f'{flip:.3f}'} bad={self.bad_streak}"
            )

        if self.bad_streak >= self.patience:
            info = {"epoch": epoch, "reasons": reasons,
                    "source_acc": source_acc, "flip": flip}
            self.events.append(info)
            self.enabled = False
            self.prev_preds = None
            return info

        if self.bad_streak == 0:
            self._update_ema(stats)
        return None
