"""Transferability-Aware Supervision (TAS).

The labelled source domain is used to estimate a per-class transferability score:
  * source per-class accuracy (how learnable the class is),
  * source/target prototype cosine similarity (how aligned the class features are),
  * domain-classifier separability (how easily a domain discriminator can tell
    the class's source and target samples apart; lower is more transferable).

The score drives per-class confidence thresholds and per-class supervision
weights, replacing the dataset-specific hand-tuned knobs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TASState:
    def __init__(self, num_classes, feature_dim=152, device=None, ema=0.9):
        self.num_classes = int(num_classes)
        self.device = device
        self.feature_dim = int(feature_dim)
        self.ema = float(ema)

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        ).to(device)
        self.domain_optimizer = torch.optim.Adam(self.domain_classifier.parameters(), lr=1e-3)

        self.source_sum = torch.zeros(self.num_classes, self.feature_dim, device=device)
        self.source_count = torch.zeros(self.num_classes, device=device)
        self.target_sum = torch.zeros(self.num_classes, self.feature_dim, device=device)
        self.target_count = torch.zeros(self.num_classes, device=device)
        self.source_correct = torch.zeros(self.num_classes, device=device)
        self.source_seen = torch.zeros(self.num_classes, device=device)
        self.separability = torch.zeros(self.num_classes, device=device)
        self.last_transferability = None

    @torch.no_grad()
    def _update_statistics(self, source_features, source_labels, target_features,
                           target_labels, target_confidence, source_logits):
        source_features = F.normalize(source_features.detach().float(), dim=1)
        target_features = F.normalize(target_features.detach().float(), dim=1)
        labels = source_labels.long()
        ones = torch.ones_like(labels, dtype=torch.float)

        self.source_sum.index_add_(0, labels, source_features)
        self.source_count.index_add_(0, labels, ones)

        confident = target_confidence >= 0.5
        if bool(confident.any()):
            kept_labels = target_labels[confident].long()
            kept_features = target_features[confident]
            self.target_sum.index_add_(0, kept_labels, kept_features)
            self.target_count.index_add_(0, kept_labels, torch.ones_like(kept_labels, dtype=torch.float))

        prediction = source_logits.detach().argmax(dim=1)
        correct = prediction.eq(labels).float()
        self.source_correct.index_add_(0, labels, correct)
        self.source_seen.index_add_(0, labels, ones)

    def _update_domain_classifier(self, source_features, target_features):
        source_features = F.normalize(source_features.detach().float(), dim=1)
        target_features = F.normalize(target_features.detach().float(), dim=1)
        # The caller runs inside torch.no_grad(); re-enable grad for this local step only.
        with torch.enable_grad():
            logits_source = self.domain_classifier(source_features)
            logits_target = self.domain_classifier(target_features)
            loss = F.binary_cross_entropy_with_logits(
                logits_source, torch.zeros_like(logits_source)
            ) + F.binary_cross_entropy_with_logits(logits_target, torch.ones_like(logits_target))
            self.domain_optimizer.zero_grad()
            loss.backward()
            self.domain_optimizer.step()
        return logits_source.detach(), logits_target.detach()

    @torch.no_grad()
    def _update_separability(self, logits_source, logits_target, source_labels, target_labels,
                             target_confidence):
        probability_source = torch.sigmoid(logits_source).ravel()
        probability_target = torch.sigmoid(logits_target).ravel()
        labels_source = source_labels.long()
        confident = target_confidence >= 0.5
        labels_target = target_labels[confident].long()
        probability_target = probability_target[confident]
        for class_id in range(self.num_classes):
            mask_source = labels_source.eq(class_id)
            mask_target = labels_target.eq(class_id)
            if int(mask_source.sum().item()) < 2 or int(mask_target.sum().item()) < 2:
                continue
            gap = (probability_target[mask_target].mean() - probability_source[mask_source].mean()).abs()
            self.separability[class_id] = (
                self.ema * self.separability[class_id] + (1.0 - self.ema) * gap.clamp(max=1.0)
            )

    def update_and_transferability(self, source_features, source_labels, target_features,
                                   target_labels, target_confidence, source_logits):
        self._update_statistics(
            source_features, source_labels, target_features, target_labels,
            target_confidence, source_logits,
        )
        logits_source, logits_target = self._update_domain_classifier(source_features, target_features)
        self._update_separability(
            logits_source, logits_target, source_labels, target_labels, target_confidence
        )

        source_prototype = F.normalize(
            self.source_sum / self.source_count.clamp_min(1.0).unsqueeze(1), dim=1
        )
        target_prototype = F.normalize(
            self.target_sum / self.target_count.clamp_min(1.0).unsqueeze(1), dim=1
        )
        similarity = (source_prototype * target_prototype).sum(dim=1).clamp(min=0.0)
        accuracy = (self.source_correct / self.source_seen.clamp_min(1.0)).clamp(0.0, 1.0)
        transferability = accuracy * similarity * (1.0 - 0.5 * self.separability.clamp(0.0, 1.0))

        valid = (self.source_count > 5) & (self.target_count > 5)
        if int(valid.sum().item()) < 2:
            return None
        transferability = torch.where(
            valid, transferability, torch.full_like(transferability, transferability[valid].mean())
        )
        transferability = transferability / transferability.mean().clamp_min(1e-6)
        self.last_transferability = transferability
        return transferability
