"""Guarded Adaptation (GA).

Online collapse detection for pseudo-label interventions.  The guard watches
label-free training statistics (pseudo-label class distribution entropy, teacher
confidence, pseudo-label coverage), compares them with their own running EMA,
and when a sustained degradation is detected it

  1. rolls the model back to the last healthy-epoch snapshot,
  2. disables the interventions (RGS weight and reliability operators),
  3. lets plain TAPDA finish the remaining epochs.

Worst case therefore equals the plain TAPDA baseline (no-harm property), while
healthy runs keep the original gains.
"""

from __future__ import annotations

import copy
import math

import torch

_MONITOR = {
    "hist": None,
    "conf_sum": 0.0,
    "n": 0,
    "high": 0,
    "num_classes": 0,
}


def guard_reset(num_classes=None, device=None):
    """Start a new epoch of statistics collection."""
    if num_classes is None:
        return
    _MONITOR["hist"] = torch.zeros(int(num_classes), device=device)
    _MONITOR["conf_sum"] = 0.0
    _MONITOR["n"] = 0
    _MONITOR["high"] = 0
    _MONITOR["num_classes"] = int(num_classes)


@torch.no_grad()
def guard_accumulate(prob, labels, high_conf=0.9):
    """Accumulate one batch of teacher pseudo-label statistics."""
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
        "frac": frac.detach().cpu().tolist(),
    }


class Guard:
    def __init__(self, num_classes, device, start_epoch=15, patience=2,
                 entropy_drop=0.25, conf_drop=0.03, coverage_drop=0.50,
                 keep=3, enabled=True):
        self.num_classes = int(num_classes)
        self.device = device
        self.start_epoch = int(start_epoch)
        self.patience = int(patience)
        self.entropy_drop = float(entropy_drop)
        self.conf_drop = float(conf_drop)
        self.coverage_drop = float(coverage_drop)
        self.keep = int(keep)
        self.enabled = bool(enabled)

        self.ema_entropy = None
        self.ema_conf = None
        self.ema_coverage = None
        self.bad_streak = 0
        self.snapshots = []          # list of (epoch, state_dicts)
        self.events = []
        self.last_stats = None

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

    def step(self, epoch, models):
        if not self.enabled:
            return None
        stats = guard_stats()
        if stats is None:
            return None
        self.last_stats = stats
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

        if reasons:
            self.bad_streak += 1
        else:
            self.bad_streak = 0

        if epoch % 5 == 0:
            print(
                f"[guard] ep={epoch} cov={stats['coverage']:.3f} conf={stats['conf_mean']:.4f} "
                f"H={stats['entropy']:.3f} bad={self.bad_streak}"
            )

        if self.bad_streak >= self.patience and self.snapshots:
            restore_epoch, _ = self.snapshots[-1]
            info = {"epoch": epoch, "restore": restore_epoch, "reasons": reasons,
                    "stats": stats}
            self.events.append(info)
            self.enabled = False
            return info

        if self.bad_streak == 0:
            self._snapshot(epoch, models)
            self._update_ema(stats)
        return None

    @staticmethod
    def load_state(state, models):
        for name, module in models.items():
            module.load_state_dict(state[name], strict=True)
