"""Adapter that plugs the v3.0 DualReliabilityEncoder into TAPDA's pipeline.

TAPDA expects a backbone ``G`` returning ``(features, projections)`` with
``features`` of dimension 152 (so that its ``ResClassifier`` can be reused
unchanged) and ``projections`` of dimension 128 (SupCon projection space).

The v3.0 encoder emits L2-normalised 160-d fused features, so this adapter
projects them to 152 dimensions for the dual classifiers and adds TAPDA's
projection head on top.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.reliability_model import DualReliabilityEncoder


class ReliabilityEncoderContrastive(nn.Module):
    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        n_classes: int,
        hide_dim: int = 512,
        projection_dim: int = 128,
        aspp_mode: str = "adaptive",
        pooling_mode: str = "guided",
        spectral_fusion_mode: str = "reliable",
    ) -> None:
        super().__init__()
        # features kept at 152 dims so that TAPDA's ResClassifier stays valid
        self.encoder = DualReliabilityEncoder(
            bands=input_channels,
            num_classes=n_classes,
            projection_dim=152,
            aspp_mode=aspp_mode,
            pooling_mode=pooling_mode,
            spectral_fusion_mode=spectral_fusion_mode,
            backbone="ours",
        )
        self.inter_size = 152
        self.projection_head = nn.Sequential(
            nn.Linear(self.inter_size, hide_dim),
            nn.BatchNorm1d(hide_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hide_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor):
        features = self.encoder(x).features
        projections = F.normalize(self.projection_head(features), dim=1)
        return features, projections
