"""Theory-driven supervision budget (candidate 1).

Per-class statistics (all label-free except the source accuracy):
  prior   pi_c   : EMA of the pseudo-label histogram,
  accuracy acc_c : EMA of the source-domain per-class accuracy,
  confidence bar_c: EMA of the teacher confidence of the class,
  variance var_c : EMA of the distance to the class prototype.

Budget score:  score_c = pi_c * acc_c * bar_c / (1 + var_c), mean-normalised.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TheoryBudget:
    def __init__(self, num_classes, feature_dim=152, ema=0.9, device=None):
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.ema = float(ema)
        self.device = device
        self.prior = torch.full((self.num_classes,), 1.0 / max(1, self.num_classes), device=device)
        self.accuracy = torch.zeros(self.num_classes, device=device)
        self.confidence = torch.zeros(self.num_classes, device=device)
        self.variance = torch.ones(self.num_classes, device=device)
        self.count = torch.zeros(self.num_classes, device=device)
        self.last_budget = None

    @torch.no_grad()
    def update(self, pseudo_labels, pseudo_probability, target_features, source_labels,
               source_logits):
        labels = pseudo_labels.long()
        probability = pseudo_probability.float().ravel()
        features = F.normalize(target_features.detach().float(), dim=1)
        ones = torch.ones_like(labels, dtype=torch.float)

        class_count = torch.zeros(self.num_classes, device=self.device)
        class_count.index_add_(0, labels, ones)
        observed = class_count > 0

        confidence_sum = torch.zeros(self.num_classes, device=self.device)
        confidence_sum.index_add_(0, labels, probability)
        mean_confidence = torch.where(
            observed, confidence_sum / class_count.clamp_min(1.0), self.confidence
        )
        self.confidence = self.ema * self.confidence + (1.0 - self.ema) * mean_confidence

        batch_prior = class_count / class_count.sum().clamp_min(1.0)
        self.prior = self.ema * self.prior + (1.0 - self.ema) * batch_prior
        self.prior = self.prior / self.prior.sum().clamp_min(1e-8)

        source_labels = source_labels.long()
        prediction = source_logits.detach().argmax(dim=1)
        correct = prediction.eq(source_labels).float()
        accuracy_sum = torch.zeros(self.num_classes, device=self.device)
        accuracy_sum.index_add_(0, source_labels, correct)
        accuracy_count = torch.zeros(self.num_classes, device=self.device)
        accuracy_count.index_add_(0, source_labels, torch.ones_like(correct))
        observed_accuracy = accuracy_count > 0
        batch_accuracy = torch.where(
            observed_accuracy, accuracy_sum / accuracy_count.clamp_min(1.0), self.accuracy
        )
        self.accuracy = self.ema * self.accuracy + (1.0 - self.ema) * batch_accuracy

        feature_sum = torch.zeros(self.num_classes, self.feature_dim, device=self.device)
        feature_sum.index_add_(0, labels, features)
        prototype = feature_sum / class_count.clamp_min(1.0).unsqueeze(1)
        prototype = F.normalize(prototype, dim=1)
        distance = (1.0 - (features * prototype[labels]).sum(dim=1)).clamp_min(0.0)
        distance_sum = torch.zeros(self.num_classes, device=self.device)
        distance_sum.index_add_(0, labels, distance)
        batch_variance = torch.where(
            observed, distance_sum / class_count.clamp_min(1.0), self.variance
        )
        self.variance = self.ema * self.variance + (1.0 - self.ema) * batch_variance
        self.count = self.ema * self.count + (1.0 - self.ema) * class_count

        valid = (self.count > 20) & observed_accuracy
        if int(valid.sum().item()) < 2:
            return None
        score = (
            self.prior
            * self.accuracy.clamp(0.0, 1.0)
            * self.confidence.clamp(0.0, 1.0)
            / (1.0 + self.variance.clamp_min(0.0))
        )
        score = torch.where(valid, score, torch.full_like(score, score[valid].mean()))
        score = score / score.mean().clamp_min(1e-6)
        self.last_budget = score
        return score
