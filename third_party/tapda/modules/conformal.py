"""Source-calibrated conformal threshold (candidate 2).

We calibrate a single confidence threshold on the labelled source domain so that
the error rate of the retained (pseudo-label) subset is <= epsilon.  The
threshold is then applied to the target domain.  Only the gate threshold is
touched: the M1 loss form and the reliability operators are unchanged.
"""

from __future__ import annotations

import torch


class SourceConformal:
    def __init__(self, num_bins=20, device=None, default_threshold=0.95,
                 min_kept=200, ema=0.9):
        self.num_bins = int(num_bins)
        self.device = device
        self.default_threshold = float(default_threshold)
        self.min_kept = int(min_kept)
        self.ema = float(ema)
        self.ok = torch.zeros(self.num_bins, device=device)
        self.bad = torch.zeros(self.num_bins, device=device)

    @torch.no_grad()
    def update(self, confidence, correct):
        confidence = confidence.detach().float().ravel().clamp(0.0, 1.0)
        correct = correct.detach().float().ravel()
        index = (confidence * (self.num_bins - 1)).round().long()
        ok_batch = torch.zeros(self.num_bins, device=self.device)
        bad_batch = torch.zeros(self.num_bins, device=self.device)
        ok_batch.index_add_(0, index, correct)
        bad_batch.index_add_(0, index, 1.0 - correct)
        self.ok = self.ok + ok_batch
        self.bad = self.bad + bad_batch

    @torch.no_grad()
    def threshold(self, epsilon):
        total_ok = float(self.ok.sum().item())
        if total_ok < self.min_kept:
            return self.default_threshold
        edges = torch.linspace(0.0, 1.0, self.num_bins, device=self.device)
        best = self.default_threshold
        for bin_index in range(self.num_bins - 1, -1, -1):
            kept_ok = float(self.ok[bin_index:].sum().item())
            kept_bad = float(self.bad[bin_index:].sum().item())
            kept = kept_ok + kept_bad
            if kept < self.min_kept:
                continue
            error = kept_bad / max(kept, 1e-6)
            if error <= float(epsilon):
                best = float(edges[bin_index].item())
                break
        return max(best, 0.5)
