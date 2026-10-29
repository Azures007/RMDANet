"""Literature-recipe models: SSSE backbone + two residual classifiers.

This module is used ONLY by the ``lit-recipe`` branch (trainer_lit.py).  It
follows the training recipe of the 2024-2026 HSI-UDA papers (MSDA / TAPDA):

* backbone  : 3D-2D hybrid spectral-spatial encoder (HybridSN / SSSE style)
* heads     : two independent residual classifiers (adversarial discrepancy)
* input     : per-band z-score normalization (see ``input_normalization: band``)
* training  : SGD momentum 0.9, 180 labels per class, no validation split

The original v7.1 pipeline (DualReliabilityEncoder, PRAT, memory pseudo
labels, ...) is untouched and remains on the ``main`` branch.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSSE(nn.Module):
    """3D-2D hybrid spectral-spatial feature extractor (HybridSN / SSSE style).

    Input : [B, bands, H, W]  (per-band z-scored patches)
    Output: [B, feature_dim] global spectral-spatial descriptor
    """

    def __init__(self, in_channels: int, patch_size: int, feature_dim: int = 128) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        spectral_remain = int(in_channels) - 12  # 7 + 5 + 3 = 15 with padding-free 3D convs
        spatial_remain = int(patch_size) - 6
        if spectral_remain < 1 or spatial_remain < 1:
            raise ValueError(
                f"SSSE needs bands>12 and patch_size>6, got bands={in_channels}, "
                f"patch={patch_size}"
            )
        flattened = 32 * spectral_remain
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(flattened, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B, 1, bands, H, W]
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        batch, channels, depth, height, width = x.shape
        x = x.reshape(batch, channels * depth, height, width)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.pool(x).flatten(1)
        return self.projection(x)


class ResClassifier(nn.Module):
    """Residual MLP classifier head (TAPDA / MSDA style)."""

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = nn.Linear(feature_dim, 256)
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.scale = nn.Linear(128, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.block1(features) + self.residual(features)
        h = self.block2(h)
        return self.scale(h)


class EMATeacher(nn.Module):
    """Exponential moving average copy of (backbone, classifier)."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module, decay: float = 0.99) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        self.classifier = copy.deepcopy(classifier)
        self.decay = float(decay)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def update(self, backbone: nn.Module, classifier: nn.Module) -> None:
        decay = self.decay
        for target, source in zip(self.backbone.parameters(), backbone.parameters()):
            target.mul_(decay).add_(source.detach(), alpha=1.0 - decay)
        for target, source in zip(self.classifier.parameters(), classifier.parameters()):
            target.mul_(decay).add_(source.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.classifier(self.backbone(x))


def batch_nuclear_norm(probabilities: torch.Tensor) -> torch.Tensor:
    """Batch nuclear-norm maximization loss (Cui et al., CVPR 2020)."""
    singular_values = torch.linalg.svdvals(probabilities.float())
    nuclear = singular_values.sum()
    return -nuclear / max(1.0, float(probabilities.size(0)))


def discrepancy_l1(probabilities_a: torch.Tensor, probabilities_b: torch.Tensor) -> torch.Tensor:
    """L1 discrepancy between the two classifier predictions (MCD style)."""
    return torch.mean(torch.abs(probabilities_a - probabilities_b))
