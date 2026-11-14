from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    for group in range(min(channels, preferred), 0, -1):
        if channels % group == 0:
            return group
    return 1


class ConvNormAct3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class LocalSpectralSpatialEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvNormAct3D(1, 24, (7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1))
        self.stage1 = ConvNormAct3D(24, 40, (5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1))
        self.stage2 = ConvNormAct3D(40, 64, (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.residual = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(64), 64),
            nn.SiLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(64), 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stage2(self.stage1(self.stem(x.unsqueeze(1))))
        return F.silu(features + self.residual(features), inplace=True)


# ---------------------------------------------------------------------------
# Mamba (selective state-space) backbone variants for the HSI patch encoder.
#
# The whole family is selected at runtime via the HSI_BACKBONE environment
# variable, so the released golden configuration continues to use the original
# 3-D convolutional encoder unless explicitly requested.
#
#   HSI_BACKBONE=conv3d       (default) original LocalSpectralSpatialEncoder
#   HSI_BACKBONE=mamba        per-pixel spectra -> spatial token Mamba (v1)
#   HSI_BACKBONE=mamba_spectral  spatial pixels -> spectral sequence Mamba (v2)
# ---------------------------------------------------------------------------
_BACKBONE = os.environ.get("HSI_BACKBONE", "conv3d").lower()


def _selective_scan(u, delta, A, B, C, D):
    """Minimal pure-PyTorch selective scan (S6).

    u:      (B, L, d_inner)
    delta:  (B, L, d_inner)
    A:      (d_inner, d_state)
    B:      (B, L, d_state)
    C:      (B, L, d_state)
    D:      (d_inner,)
    returns y: (B, L, d_inner)
    """
    batch, length, d_inner = u.shape
    d_state = A.shape[1]
    h = u.new_zeros((batch, d_inner, d_state))
    ys = []
    for t in range(length):
        dt = delta[:, t, :]                       # (B, d_inner)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))   # (B, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B[:, t, None, :]            # (B, d_inner, d_state)
        h = dA * h + dB * u[:, t, :, None]
        y = torch.einsum("bds,bs->bd", h, C[:, t, :]) + D * u[:, t, :]
        ys.append(y)
    return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    """A single Mamba (S6) block operating on a token sequence."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        with torch.no_grad():
            nn.init.uniform_(self.dt_proj.bias, -2.0, 0.0)
            nn.init.uniform_(self.dt_proj.weight, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d_model)
        batch, length, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)                         # (B, L, d_inner)
        u = u.transpose(1, 2)                              # (B, d_inner, L)
        u = self.conv1d(u)[..., :length]
        u = F.silu(u).transpose(1, 2)                      # (B, L, d_inner)

        x_dbl = self.x_proj(u)                             # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        dt = F.softplus(dt + self.dt_proj.bias)
        A = -torch.exp(self.A_log.float())
        y = _selective_scan(u, dt, A, B, C, self.D)
        y = y * F.silu(z)
        return self.out_proj(y)


class _TokenPositionalEncoding(nn.Module):
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)


class MambaLocalEncoder(nn.Module):
    """Mamba v1: per-pixel spectra are embedded and mixed as spatial tokens.

    Input:  (B, bands, H, W)  (H = W = patch size)
    Output: (B, 64, 1, H, W)  -- same interface as LocalSpectralSpatialEncoder.
    """

    def __init__(self, bands: int, d_model: int = 64, depth: int = 4, d_state: int = 16):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(bands),
            nn.Linear(bands, d_model),
        )
        self.pos = _TokenPositionalEncoding(7 * 7, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state=d_state) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(64), 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, bands, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, bands)
        tokens = self.in_proj(tokens)                       # (B, HW, d_model)
        tokens = tokens + self.pos.pos[:, : height * width]
        for block in self.blocks:
            tokens = tokens + block(tokens)
        tokens = self.norm(tokens)
        feature_map = tokens.transpose(1, 2).reshape(batch, -1, height, width)
        feature_map = self.out_proj(feature_map)
        return feature_map.unsqueeze(2)                     # (B, 64, 1, H, W)


class MambaSpectralEncoder(nn.Module):
    """Mamba v2: each pixel is a token, the spectral axis is the SSM sequence.

    Input:  (B, bands, H, W)
    Output: (B, 64, 1, H, W)
    """

    def __init__(self, bands: int, d_model: int = 64, depth: int = 3, d_state: int = 16):
        super().__init__()
        self.band_proj = nn.Linear(1, d_model)
        self.pos = _TokenPositionalEncoding(bands, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state=d_state) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(64), 64),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, bands, height, width = x.shape
        # (B, H, W, bands) -> batch the pixels, sequence = bands
        tokens = x.permute(0, 2, 3, 1).reshape(batch * height * width, bands, 1)
        tokens = self.band_proj(tokens)                     # (B*HW, bands, d_model)
        tokens = tokens + self.pos.pos[:, :bands]
        for block in self.blocks:
            tokens = tokens + block(tokens)
        tokens = self.norm(tokens)
        feature = tokens.mean(dim=1).reshape(batch, height, width, -1)  # (B,H,W,d_model)
        feature = feature.permute(0, 3, 1, 2)
        feature = self.spatial_proj(feature)
        return feature.unsqueeze(2)



class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.08) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout2d(dropout)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(_groups(out_channels), out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.silu(self.norm1(self.conv1(x)), inplace=True)
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual, inplace=True)


class _ASPPBranch(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        padding = 0 if kernel_size == 1 else dilation
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ScaleSelectiveMicroASPP(nn.Module):
    """Four receptive fields with selectable ablation behavior."""

    def __init__(
        self,
        in_channels: int = 128,
        branch_channels: int = 32,
        mode: str = "adaptive",
    ) -> None:
        super().__init__()
        if mode not in {"single", "equal", "adaptive"}:
            raise ValueError("MicroASPP mode must be single, equal or adaptive.")
        self.mode = mode
        specs = [(1, 1), (3, 1), (3, 2), (3, 3)]
        self.branches = nn.ModuleList(
            [_ASPPBranch(in_channels, branch_channels, kernel, dilation) for kernel, dilation in specs]
        )
        descriptor_dim = branch_channels * len(self.branches)
        hidden = max(24, descriptor_dim // 4)
        self.selector = nn.Sequential(
            nn.Linear(descriptor_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, len(self.branches)),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(descriptor_dim, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(in_channels), in_channels),
        )
        self.residual_logit = nn.Parameter(torch.tensor(-2.1972246))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        branch_features = [branch(x) for branch in self.branches]
        descriptors = torch.cat(
            [F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in branch_features],
            dim=1,
        )
        if self.mode == "adaptive":
            weights = torch.softmax(self.selector(descriptors), dim=1)
        elif self.mode == "equal":
            weights = descriptors.new_full((descriptors.size(0), len(self.branches)), 1.0 / len(self.branches))
        else:
            weights = descriptors.new_zeros((descriptors.size(0), len(self.branches)))
            weights[:, 1] = 1.0  # the standard 3x3 branch
        weighted = [
            feature * weights[:, index].view(-1, 1, 1, 1)
            for index, feature in enumerate(branch_features)
        ]
        multi_scale = self.fuse(torch.cat(weighted, dim=1))
        residual_weight = torch.sigmoid(self.residual_logit)
        return F.silu(x + residual_weight * multi_scale, inplace=True), weights


class CenterGuidedPooling(nn.Module):
    """Combine global, center and center-guided contextual descriptors.

    The center pixel owns the patch label.  A learned three-way gate keeps global
    context while preventing a large surrounding class from overwhelming the
    center semantics at boundaries.
    """

    def __init__(
        self,
        channels: int,
        center_size: int = 3,
        attention_temperature: float = 0.12,
        mode: str = "guided",
        gate_init: tuple[float, float, float] = (0.38, 0.47, 0.15),
        residual: bool = False,
        alpha_logit: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"gap", "center", "guided"}:
            raise ValueError("Pooling mode must be gap, center or guided.")
        self.mode = mode
        self.channels = int(channels)
        self.center_size = int(center_size)
        self.temperature = max(float(attention_temperature), 1e-4)
        self.residual = bool(residual)
        hidden = max(32, channels // 3)
        self.gate = nn.Sequential(
            nn.Linear(channels * 4, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.gate[-1].weight)
        initial = torch.tensor(gate_init, dtype=torch.float32)
        nn.init.constant_(self.gate[-1].bias, 0.0)
        with torch.no_grad():
            self.gate[-1].bias.copy_(initial.log())
        # Residual center guidance: pooled = gap + alpha * (guided - gap),
        # alpha = sigmoid(learnable logit), initialised near zero so the module
        # starts from the stable GAP baseline and only deviates when helpful.
        self._alpha_logit = nn.Parameter(torch.tensor(float(alpha_logit))) if self.residual else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, channels, height, width = x.shape
        global_descriptor = F.adaptive_avg_pool2d(x, 1).flatten(1)
        crop_h = min(self.center_size, height)
        crop_w = min(self.center_size, width)
        row0 = max(0, (height - crop_h) // 2)
        col0 = max(0, (width - crop_w) // 2)
        center_crop = x[:, :, row0 : row0 + crop_h, col0 : col0 + crop_w]
        center_descriptor = F.adaptive_avg_pool2d(center_crop, 1).flatten(1)

        center_query = F.normalize(center_descriptor, dim=1)
        spatial_tokens = x.flatten(2).transpose(1, 2)
        normalized_tokens = F.normalize(spatial_tokens, dim=2)
        similarity = torch.sum(normalized_tokens * center_query[:, None, :], dim=2)
        attention = torch.softmax(similarity / self.temperature, dim=1)
        attended_descriptor = torch.sum(spatial_tokens * attention[:, :, None], dim=1)

        discrepancy = (center_descriptor - global_descriptor).abs()
        gate_input = torch.cat(
            [global_descriptor, center_descriptor, attended_descriptor, discrepancy], dim=1
        )
        if self.mode == "guided":
            weights = torch.softmax(self.gate(gate_input), dim=1)
        elif self.mode == "gap":
            weights = global_descriptor.new_zeros((batch, 3))
            weights[:, 0] = 1.0
        else:
            weights = global_descriptor.new_zeros((batch, 3))
            weights[:, 1] = 1.0
        pooled = (
            weights[:, 0:1] * global_descriptor
            + weights[:, 1:2] * center_descriptor
            + weights[:, 2:3] * attended_descriptor
        )
        if self.residual and self.mode == "guided":
            alpha = torch.sigmoid(self._alpha_logit)
            pooled = global_descriptor + alpha * (pooled - global_descriptor)
        attention_map = attention.view(batch, height, width)
        return pooled, weights, attention_map


class CenterSpectralShapeEncoder(nn.Module):
    def __init__(self, bands: int, projection_dim: int, amplitude: bool = False) -> None:
        super().__init__()
        self.bands = int(bands)
        self.amplitude = bool(amplitude)
        self.raw_projection = nn.Sequential(
            nn.LayerNorm(self.bands),
            nn.Linear(self.bands, 128, bias=False),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
        )
        self.shape_encoder = nn.Sequential(
            nn.Conv1d(3, 48, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(_groups(48), 48),
            nn.SiLU(inplace=True),
            nn.Conv1d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(96), 96),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        # Per-pixel albedo is destroyed by the LayerNorm above and by the
        # (spectrum - mean) / std standardization below.  When enabled, the
        # log spectral norm is injected here so it bypasses both.
        self.amplitude_head = (
            nn.Sequential(nn.Linear(1, 32), nn.SiLU(inplace=True))
            if self.amplitude else None
        )
        self.fuse = nn.Sequential(
            nn.Linear(224 + (32 if self.amplitude else 0), 192, bias=False),
            nn.LayerNorm(192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(192, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        spectrum = x[:, :, height // 2, width // 2].float()
        mean = spectrum.mean(dim=1, keepdim=True)
        std = spectrum.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        standardized = (spectrum - mean) / std
        first = F.pad(standardized[:, 1:] - standardized[:, :-1], (0, 1))
        second = F.pad(first[:, 1:] - first[:, :-1], (0, 1))
        shape = self.shape_encoder(torch.stack([standardized, first, second], dim=1)).flatten(1)
        raw = self.raw_projection(spectrum)
        parts = [raw, shape]
        if self.amplitude_head is not None:
            log_amp = torch.log(spectrum.norm(dim=1, keepdim=True).clamp_min(1e-6))
            parts.append(self.amplitude_head(log_amp))
        return F.normalize(self.fuse(torch.cat(parts, dim=1)), dim=1)


class CosineClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, initial_scale: float = 12.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale), dtype=torch.float32))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)
        scale = self.log_scale.exp().clamp(4.0, 30.0)
        return scale * (features @ weight.t())


class SpectralReliabilityFusion(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        max_gate: float = 0.16,
        agreement_threshold: float = 0.20,
        agreement_temperature: float = 0.10,
        rescue_strength: float = 0.55,
        rescue_confidence: float = 0.55,
        mode: str = "reliable",
        fusion_alpha: float = 1.0,
        fusion_delta: float = -0.2,
        fusion_tau_a: float = 0.20,
        fusion_t_a: float = 0.10,
        fusion_t_w: float = 0.15,
        fusion_w_min: float = 0.10,
        fusion_w_max: float = 0.90,
    ) -> None:
        super().__init__()
        if mode not in {"off", "average", "reliable", "bidirectional"}:
            raise ValueError(
                "Spectral fusion mode must be off, average, reliable or bidirectional."
            )
        self.mode = mode
        self.max_gate = float(max_gate)
        self.agreement_threshold = float(agreement_threshold)
        self.agreement_temperature = max(float(agreement_temperature), 1e-4)
        self.rescue_strength = float(max(0.0, min(1.0, rescue_strength)))
        self.rescue_confidence = float(max(0.0, min(1.0, rescue_confidence)))
        self.fusion_alpha = float(fusion_alpha)
        self.fusion_delta = float(fusion_delta)
        self.fusion_tau_a = float(fusion_tau_a)
        self.fusion_t_a = max(float(fusion_t_a), 1e-4)
        self.fusion_t_w = max(float(fusion_t_w), 1e-4)
        self.fusion_w_min = float(fusion_w_min)
        self.fusion_w_max = float(fusion_w_max)
        hidden = max(32, feature_dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, feature_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        init_gate = 0.12 / self.max_gate
        nn.init.constant_(self.gate[-1].bias, math.log(init_gate / (1.0 - init_gate)))

    @staticmethod
    def _branch_reliability(probability: torch.Tensor) -> torch.Tensor:
        """Top-1 minus top-2 margin of a branch's softmax, in [0, 1]."""
        top2 = torch.topk(probability, k=2, dim=1).values
        return (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)

    def _bidirectional_forward(
        self,
        context_features: torch.Tensor,
        spectral_features: torch.Tensor,
        context_logits: torch.Tensor,
        spectral_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_probability = F.softmax(context_logits.detach().float(), dim=1)
        spectral_probability = F.softmax(spectral_logits.detach().float(), dim=1)
        r_c = self._branch_reliability(context_probability)
        r_s = self._branch_reliability(spectral_probability)

        cosine = torch.sum(
            F.normalize(context_features, dim=1) * F.normalize(spectral_features, dim=1), dim=1
        )
        agreement = torch.sigmoid((cosine - self.fusion_tau_a) / self.fusion_t_a)

        base = torch.sigmoid(
            self.fusion_alpha * (r_s - r_c) / self.fusion_t_w + self.fusion_delta
        )
        w_s = (0.5 + agreement * (base - 0.5)).clamp(self.fusion_w_min, self.fusion_w_max)

        weight = w_s[:, None].expand_as(context_features)
        fused = F.normalize((1.0 - weight) * context_features + weight * spectral_features, dim=1)
        return fused, weight, w_s

    def forward(
        self,
        context_features: torch.Tensor,
        spectral_features: torch.Tensor,
        context_logits: torch.Tensor,
        spectral_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "off":
            zero = context_features.new_zeros(context_features.size(0))
            return context_features, context_features.new_zeros(context_features.shape), zero
        if self.mode == "average":
            fused = F.normalize(0.5 * context_features + 0.5 * spectral_features, dim=1)
            reliability = context_features.new_ones(context_features.size(0))
            gate = context_features.new_full(context_features.shape, 0.5)
            return fused, gate, reliability
        if self.mode == "bidirectional":
            return self._bidirectional_forward(
                context_features, spectral_features, context_logits, spectral_logits
            )

        difference = (context_features - spectral_features).abs()
        raw_gate = self.max_gate * torch.sigmoid(
            self.gate(torch.cat([context_features, spectral_features, difference], dim=1))
        )
        cosine = torch.sum(
            F.normalize(context_features, dim=1) * F.normalize(spectral_features, dim=1), dim=1
        )
        representation_agreement = torch.sigmoid(
            (cosine - self.agreement_threshold) / self.agreement_temperature
        )
        context_probability = F.softmax(context_logits.detach().float(), dim=1)
        spectral_probability = F.softmax(spectral_logits.detach().float(), dim=1)
        semantic_agreement = context_probability.argmax(dim=1).eq(
            spectral_probability.argmax(dim=1)
        ).float()

        # Rescue path: 允许自信的光谱分支在语义不一致时纠正上下文错误
        # 条件：两分支预测不一致 & 光谱置信度 >= rescue_confidence & 光谱置信度 > 上下文
        spectral_confidence = spectral_probability.max(dim=1).values
        context_confidence = context_probability.max(dim=1).values
        spectral_advantage = (spectral_confidence - context_confidence).clamp(0.0, 1.0)
        confidence_pass = (spectral_confidence >= self.rescue_confidence).float()
        rescue = (1.0 - semantic_agreement) * spectral_advantage * confidence_pass
        semantic_gate = 0.05 + 0.95 * semantic_agreement + self.rescue_strength * rescue

        reliability = (0.35 + 0.65 * representation_agreement) * semantic_gate
        gate = raw_gate * reliability[:, None]
        fused = F.normalize(context_features + gate * (spectral_features - context_features), dim=1)
        return fused, gate, reliability


@dataclass
class ModelOutput:
    logits: torch.Tensor
    features: torch.Tensor
    scale_weights: torch.Tensor
    center_weights: torch.Tensor
    attention_map: torch.Tensor
    context_logits: torch.Tensor
    spectral_logits: torch.Tensor
    spectral_reliability: torch.Tensor
    logits2: torch.Tensor | None = None



class DualReliabilityEncoder(nn.Module):
    def __init__(
        self,
        bands: int,
        num_classes: int,
        projection_dim: int = 160,
        projection_norm: str = "batch",
        aspp_mode: str = "adaptive",
        pooling_mode: str = "guided",
        spectral_fusion_mode: str = "reliable",
        amplitude_channel: bool = False,
        use_spectral_branch: bool = True,
        enable_dual_head: bool = False,
        pool_center_size: int = 3,
        pool_temperature: float = 0.12,
        pool_gate_init: tuple[float, float, float] = (0.38, 0.47, 0.15),
        pool_residual: bool = False,
        pool_alpha_logit: float = 0.0,
        fusion_alpha: float = 1.0,
        fusion_delta: float = -0.2,
        fusion_tau_a: float = 0.20,
        fusion_t_a: float = 0.10,
        fusion_t_w: float = 0.15,
        fusion_w_min: float = 0.10,
        fusion_w_max: float = 0.90,
    ) -> None:
        super().__init__()
        self.enable_dual_head = bool(enable_dual_head)
        self.bands = int(bands)
        self.num_classes = int(num_classes)
        self.projection_dim = int(projection_dim)
        self.aspp_mode = aspp_mode
        self.pooling_mode = pooling_mode
        self.spectral_fusion_mode = spectral_fusion_mode
        self.pool_gate_init = tuple(gate_init_value for gate_init_value in pool_gate_init)
        if _BACKBONE == "mamba":
            self.local3d = MambaLocalEncoder(bands)
        elif _BACKBONE == "mamba_spectral":
            self.local3d = MambaSpectralEncoder(bands)
        else:
            self.local3d = LocalSpectralSpatialEncoder()
        self.semantic2d = nn.Sequential(
            ResidualBlock2D(64, 96, dropout=0.06),
            ResidualBlock2D(96, 128, dropout=0.08),
            ResidualBlock2D(128, 128, dropout=0.08),
        )
        self.micro_aspp = ScaleSelectiveMicroASPP(128, 32, mode=aspp_mode)
        self.pool = CenterGuidedPooling(
            128,
            center_size=pool_center_size,
            attention_temperature=pool_temperature,
            mode=pooling_mode,
            gate_init=pool_gate_init,
            residual=pool_residual,
            alpha_logit=pool_alpha_logit,
        )
        if projection_norm == "batch":
            norm: nn.Module = nn.BatchNorm1d(256)
        elif projection_norm == "layer":
            norm = nn.LayerNorm(256)
        else:
            raise ValueError("projection_norm must be 'layer' or 'batch'.")
        self.projection = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            norm,
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(256, self.projection_dim),
        )
        self.amplitude_channel = bool(amplitude_channel)
        # DRSE (Dual-branch Reliability Spectral-spatial Encoder) = spatial branch
        # + spectral branch + reliability fusion.  ``use_spectral_branch=False``
        # removes the spectral branch and the fusion, i.e. it disables DRSE as a
        # whole (single-branch baseline for the ablation study).
        self.use_spectral_branch = bool(use_spectral_branch)
        if self.use_spectral_branch:
            self.spectral_encoder = CenterSpectralShapeEncoder(
                self.bands, self.projection_dim, amplitude=self.amplitude_channel
            )
        else:
            self.spectral_encoder = None
        self.classifier = CosineClassifier(self.projection_dim, self.num_classes, initial_scale=12.0)
        self.classifier2 = (CosineClassifier(self.projection_dim, self.num_classes, initial_scale=12.0)
                            if self.enable_dual_head else None)
        if self.use_spectral_branch:
            self.spectral_fusion = SpectralReliabilityFusion(
                self.projection_dim,
                mode=spectral_fusion_mode,
                fusion_alpha=fusion_alpha,
                fusion_delta=fusion_delta,
                fusion_tau_a=fusion_tau_a,
                fusion_t_a=fusion_t_a,
                fusion_t_w=fusion_t_w,
                fusion_w_min=fusion_w_min,
                fusion_w_max=fusion_w_max,
            )
        else:
            self.spectral_fusion = None
        self.reset_parameters()
        self._restore_safe_initialization()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _restore_safe_initialization(self) -> None:
        nn.init.zeros_(self.micro_aspp.selector[-1].weight)
        nn.init.zeros_(self.micro_aspp.selector[-1].bias)
        nn.init.zeros_(self.pool.gate[-1].weight)
        initial = torch.tensor(self.pool_gate_init, dtype=self.pool.gate[-1].bias.dtype)
        with torch.no_grad():
            self.pool.gate[-1].bias.copy_(initial.log())
        if self.spectral_fusion is not None:
            nn.init.zeros_(self.spectral_fusion.gate[-1].weight)
            init_gate = 0.12 / self.spectral_fusion.max_gate
            nn.init.constant_(
                self.spectral_fusion.gate[-1].bias,
                math.log(init_gate / (1.0 - init_gate)),
            )

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.ndim != 4:
            raise ValueError("Expected input shape [batch, bands, height, width].")
        features3d = self.local3d(x)
        spatial = features3d.mean(dim=2)
        spatial = self.semantic2d(spatial)
        spatial, scale_weights = self.micro_aspp(spatial)
        pooled, center_weights, attention_map = self.pool(spatial)
        context_features = F.normalize(self.projection(pooled), dim=1)
        context_logits = self.classifier(context_features)
        if self.spectral_encoder is not None:
            spectral_features = self.spectral_encoder(x)
            spectral_logits = self.classifier(spectral_features)
            features, _, spectral_reliability = self.spectral_fusion(
                context_features, spectral_features, context_logits, spectral_logits
            )
            logits = self.classifier(features)
        else:
            # Single-branch (DRSE disabled): no spectral branch, no fusion.
            features = context_features
            logits = context_logits
            spectral_logits = None
            spectral_reliability = torch.ones(
                context_features.size(0), device=context_features.device, dtype=context_features.dtype
            )
        logits2 = self.classifier2(features) if self.classifier2 is not None else None
        return ModelOutput(
            logits=logits,
            logits2=logits2,
            features=features,
            scale_weights=scale_weights,
            center_weights=center_weights,
            attention_map=attention_map,
            context_logits=context_logits,
            spectral_logits=spectral_logits,
            spectral_reliability=spectral_reliability,
        )
