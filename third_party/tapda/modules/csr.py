"""Class-Starvation Revival (CSR).

Pseudo-label self-training systematically starves low-frequency classes: the teacher
never assigns them high confidence, so they never receive supervision, so they are
never predicted.  CSR diagnoses the starved classes from label-free statistics and
revives them with targeted threshold relaxation and a per-class quota floor.

Diagnosis (no target labels required):
  * pi_t[c]  - target class prior estimated with BBSE from the source confusion matrix
               and the running target prediction histogram,
  * yhat[c]  - running target prediction frequency,
  * starved[c] = yhat[c] / pi_t[c] < starvation_ratio.

Revival:
  * threshold for the starved class is relaxed by threshold_multiplier,
  * at least `quota` samples of the starved class are kept per batch.
"""

from __future__ import annotations

import numpy as np
import torch


class CSRState:
    def __init__(self, num_classes, device, source_loader=None, G=None, F1=None,
                 ema=0.9, starvation_ratio=0.3, threshold_multiplier=0.8, quota=2):
        self.num_classes = int(num_classes)
        self.device = device
        self.source_loader = source_loader
        self.G = G
        self.F1 = F1
        self.ema = float(ema)
        self.starvation_ratio = float(starvation_ratio)
        self.threshold_multiplier = float(threshold_multiplier)
        self.quota = int(quota)

        self.pred_hist = torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)
        self.prior = None
        self.confusion = None
        self.starved = None

    @torch.no_grad()
    def update(self, pseudo_labels):
        labels = pseudo_labels.detach().long()
        counts = torch.zeros(self.num_classes, device=self.device)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
        counts = counts / counts.sum().clamp_min(1.0)
        self.pred_hist = self.ema * self.pred_hist + (1.0 - self.ema) * counts

    @torch.no_grad()
    def _source_confusion(self):
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
        for index, (data, target) in enumerate(self.source_loader):
            if index >= 100:
                break
            features, _ = self.G(data.to(self.device))
            output = self.F1(features)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.dim() == 1:
                output = output.unsqueeze(0)
            np.add.at(
                confusion,
                (target.cpu().numpy(), output.argmax(dim=1).cpu().numpy()),
                1.0,
            )
        return confusion

    @torch.no_grad()
    def ensure_ready(self):
        if self.prior is not None:
            return
        if self.source_loader is None or self.G is None or self.F1 is None:
            self.prior = torch.full((self.num_classes,), 1.0 / self.num_classes, device=self.device)
            return
        g_was_training = self.G.training
        f_was_training = self.F1.training
        self.G.eval()
        self.F1.eval()
        try:
            confusion = self._source_confusion()
        finally:
            self.G.train(g_was_training)
            self.F1.train(f_was_training)
        row_sum = confusion.sum(axis=1, keepdims=True)
        row_sum = confusion.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        conditional = confusion / row_sum
        histogram = self.pred_hist.detach().cpu().numpy().astype(np.float64)
        try:
            prior = np.linalg.lstsq(conditional, histogram, rcond=None)[0]
        except np.linalg.LinAlgError:
            prior = histogram
        prior = np.clip(prior, 1e-4, None)
        prior = prior / prior.sum()
        self.prior = torch.from_numpy(prior).float().to(self.device)
        ratio = self.pred_hist / self.prior.clamp_min(1e-8)
        self.starved = ratio < self.starvation_ratio

    def threshold_multiplier_vector(self):
        multiplier = torch.ones(self.num_classes, device=self.device)
        if self.starved is not None and bool(self.starved.any()):
            multiplier[self.starved] = self.threshold_multiplier
        return multiplier

    def starved_mask(self):
        if self.starved is None:
            return torch.zeros(self.num_classes, dtype=torch.bool, device=self.device)
        return self.starved

    def apply_quota(self, confident, labels):
        """Ensure at least `quota` selected samples for every starved class."""
        if self.quota <= 0 or self.starved is None or not bool(self.starved.any()):
            return confident
        for class_id in torch.nonzero(self.starved, as_tuple=False).flatten().tolist():
            class_mask = labels.eq(int(class_id))
            if not bool(class_mask.any()):
                continue
            selected = int((class_mask & confident).sum().item())
            if selected >= self.quota:
                continue
            index = torch.nonzero(class_mask, as_tuple=False).flatten()
            need = min(self.quota, int(index.numel()))
            confident[index[:need]] = True
        return confident
