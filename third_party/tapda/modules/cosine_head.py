"""v3.0 CosineClassifier adapted to TAPDA's classifier interface.

TAPDA builds ``ResClassifier`` heads and calls ``F(features)``:
  * training mode -> ``(logits, embedding)``
  * eval mode     -> ``logits``

ResClassifier implements this by overriding ``nn.Module.training`` in its
constructor, so ``.train()`` / ``.eval()`` toggle the return signature.  This
head reproduces exactly that behaviour (otherwise evaluation would receive a
tuple and fail when adding the two heads' logits).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifierHead(nn.Module):
    def __init__(
        self,
        class_num: int,
        feature_dim: int = 152,
        training: bool = True,
        initial_scale: float = 12.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(class_num, feature_dim))
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale), dtype=torch.float32))
        nn.init.normal_(self.weight, std=0.02)
        # Overrides nn.Module.training on purpose, mirroring ResClassifier.
        self.training = bool(training)

    def forward(self, features: torch.Tensor):
        normalised = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)
        scale = self.log_scale.exp().clamp(4.0, 30.0)
        logits = scale * (normalised @ weight.t())
        if self.training:
            return logits, normalised
        return logits
