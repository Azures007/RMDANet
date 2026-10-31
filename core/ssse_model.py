"""SSSE backbone adapter for the v3.0 trainer (E experiment).

Wraps TAPDA's SSSE feature extractor so that the v3.0 training pipeline
(PRAT + reliability memory + CCA/LMMD + guard) can run on top of a strong
backbone.  The adapter returns the same ``ModelOutput`` structure as
``DualReliabilityEncoder``; auxiliary fields that the SSSE backbone does not
produce are filled with neutral placeholders so that the existing losses stay
well-defined.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CosineClassifier, ModelOutput
from ssse_backbone import SSSE_Contrastive


class SSSEAdapter(nn.Module):
    def __init__(
        self,
        bands: int,
        num_classes: int,
        projection_dim: int = 128,
        patch_size: int | None = None,
        enable_dual_head: bool = False,
        amplitude_channel: bool = False,
        amplitude_input: bool = False,
        backbone: str = "ssse",
        **_unused,
    ) -> None:
        super().__init__()
        self.bands = int(bands) - (1 if amplitude_input else 0)
        self.num_classes = int(num_classes)
        self.projection_dim = int(projection_dim)
        self.patch_size = int(patch_size) if patch_size is not None else None
        self.enable_dual_head = bool(enable_dual_head)
        self.ssse: SSSE_Contrastive | None = None
        self.feature_projection = nn.Linear(152, self.projection_dim)
        self.classifier = CosineClassifier(self.projection_dim, self.num_classes)
        self.classifier2 = (
            CosineClassifier(self.projection_dim, self.num_classes)
            if self.enable_dual_head else None
        )
        if self.patch_size is not None:
            # Build eagerly so that the EMA teacher deepcopy sees the full parameter set.
            self._build_ssse(self.patch_size, torch.device("cpu"))

    def _build_ssse(self, patch_size: int, device: torch.device) -> None:
        self.ssse = SSSE_Contrastive(
            input_channels=self.bands,
            patch_size=int(patch_size),
            n_classes=self.num_classes,
        ).to(device)
        self.patch_size = int(patch_size)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.ndim != 4:
            raise ValueError("Expected input shape [batch, bands, height, width].")
        if self.ssse is None:
            self._build_ssse(x.shape[-1], x.device)
        features_raw, _projections = self.ssse(x)
        features = F.normalize(self.feature_projection(features_raw), dim=1)
        logits = self.classifier(features)
        batch = features.size(0)
        device = features.device
        scale_weights = torch.full((batch, 3), 1.0 / 3.0, device=device, dtype=features.dtype)
        center_weights = torch.full((batch, 3), 1.0 / 3.0, device=device, dtype=features.dtype)
        attention_map = torch.zeros(
            (batch, 1, x.size(-2), x.size(-1)), device=device, dtype=features.dtype
        )
        spectral_reliability = torch.ones((batch,), device=device, dtype=features.dtype)
        logits2 = self.classifier2(features) if self.classifier2 is not None else None
        return ModelOutput(
            logits=logits,
            features=features,
            scale_weights=scale_weights,
            center_weights=center_weights,
            attention_map=attention_map,
            context_logits=logits,
            spectral_logits=logits,
            spectral_reliability=spectral_reliability,
            logits2=logits2,
        )
