from __future__ import annotations

import math
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
    ) -> None:
        super().__init__()
        if mode not in {"gap", "center", "guided"}:
            raise ValueError("Pooling mode must be gap, center or guided.")
        self.mode = mode
        self.channels = int(channels)
        self.center_size = int(center_size)
        self.temperature = max(float(attention_temperature), 1e-4)
        hidden = max(32, channels // 3)
        self.gate = nn.Sequential(
            nn.Linear(channels * 4, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.gate[-1].weight)
        initial = torch.tensor([0.38, 0.47, 0.15], dtype=torch.float32)
        nn.init.constant_(self.gate[-1].bias, 0.0)
        with torch.no_grad():
            self.gate[-1].bias.copy_(initial.log())

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
    ) -> None:
        super().__init__()
        if mode not in {"off", "average", "reliable"}:
            raise ValueError("Spectral fusion mode must be off, average or reliable.")
        self.mode = mode
        self.max_gate = float(max_gate)
        self.agreement_threshold = float(agreement_threshold)
        self.agreement_temperature = max(float(agreement_temperature), 1e-4)
        self.rescue_strength = float(max(0.0, min(1.0, rescue_strength)))
        self.rescue_confidence = float(max(0.0, min(1.0, rescue_confidence)))
        hidden = max(32, feature_dim // 2)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, feature_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        init_gate = 0.12 / self.max_gate
        nn.init.constant_(self.gate[-1].bias, math.log(init_gate / (1.0 - init_gate)))

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



class DCRNChannelAttention(nn.Module):
    """Channel attention from DCRN (SCLUDA, TGRS 2023)."""

    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global max via amax: same value as AdaptiveMaxPool2d(1) but with a
        # deterministic backward pass (required by torch.use_deterministic_algorithms).
        global_max = torch.amax(x, dim=(2, 3), keepdim=True)
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(global_max))


class DCRNSpatialAttention(nn.Module):
    """Spatial attention from DCRN (SCLUDA, TGRS 2023)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class DCRNLocalStem(nn.Module):
    """DCRN convolutional stem (SCLUDA reference backbone), spatial output.

    Input  : [B, bands, H, W]
    Output : [B, 288, H, W]  (192 spectral + 96 spatial channels, DCRN attention)
    The rest of the dual-reliability pipeline (ASPP, centre pooling, PRAT,
    memory pseudo-labels, CCA/CORAL/LMMD, ...) is unchanged.
    """

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.feature_dim = int(input_channels)
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.conv4 = nn.Conv3d(
            24, 192, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True
        )
        self.bn4 = nn.BatchNorm3d(192)
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.conv7 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.bn7 = nn.BatchNorm3d(96)
        self.conv8 = nn.Conv3d(24, 96, kernel_size=1)
        self.inter_size = 192 + 96
        self.ca = DCRNChannelAttention(self.inter_size)
        self.sa = DCRNSpatialAttention()
        self.activation = nn.ReLU(inplace=True)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B, 1, bands, H, W]
        x1 = self.activation(self.bn1(self.conv1(x)))
        residual = x1
        x1 = self.activation(self.bn2(self.conv2(x1)))
        x1 = self.conv3(x1)
        x1 = self.activation(self.bn3(residual + x1))
        x1 = self.activation(self.bn4(self.conv4(x1)))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        x2 = self.activation(self.bn5(self.conv5(x)))
        residual2 = self.conv8(x2)
        x2 = self.activation(self.bn6(self.conv6(x2)))
        x2 = self.conv7(x2)
        x2 = self.activation(self.bn7(residual2 + x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
        features = torch.cat((x1, x2), dim=1)
        features = self.ca(features) * features
        return self.sa(features) * features


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
        amplitude_input: bool = False,
        backbone: str = "ours",
        enable_dual_head: bool = False,
    ) -> None:
        super().__init__()
        self.enable_dual_head = bool(enable_dual_head)
        # When amplitude_input is on the input cube carries one extra trailing
        # channel (per-pixel log magnitude); the spectral encoders keep seeing
        # only the real spectral bands.
        self.amplitude_input = bool(amplitude_input)
        self.bands = int(bands) - (1 if self.amplitude_input else 0)
        self.backbone = str(backbone)
        self.num_classes = int(num_classes)
        self.projection_dim = int(projection_dim)
        self.aspp_mode = aspp_mode
        self.pooling_mode = pooling_mode
        self.spectral_fusion_mode = spectral_fusion_mode
        if self.backbone == "dcrn":
            # Only the local feature extractor is replaced (SCLUDA/DCRN stem);
            # ASPP, centre pooling, spectral branch and all losses are unchanged.
            self.local3d = DCRNLocalStem(self.bands)
            self.semantic2d = nn.Sequential(
                nn.Conv2d(288, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
            )
        else:
            self.local3d = LocalSpectralSpatialEncoder()
            self.semantic2d = nn.Sequential(
                ResidualBlock2D(64, 96, dropout=0.06),
                ResidualBlock2D(96, 128, dropout=0.08),
                ResidualBlock2D(128, 128, dropout=0.08),
            )
        self.micro_aspp = ScaleSelectiveMicroASPP(128, 32, mode=aspp_mode)
        self.pool = CenterGuidedPooling(
            128, center_size=3, attention_temperature=0.12, mode=pooling_mode
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
        self.spectral_encoder = CenterSpectralShapeEncoder(
            self.bands, self.projection_dim, amplitude=self.amplitude_channel
        )
        self.amplitude_encoder: nn.Module | None = None
        self.amplitude_gate: nn.Parameter | None = None
        if self.amplitude_input:
            self.amplitude_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.GroupNorm(4, 16),
                nn.SiLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, self.projection_dim),
            )
            nn.init.zeros_(self.amplitude_encoder[-1].weight)
            nn.init.zeros_(self.amplitude_encoder[-1].bias)
            self.amplitude_gate = nn.Parameter(torch.zeros(1))
        self.classifier = CosineClassifier(self.projection_dim, self.num_classes, initial_scale=12.0)
        self.classifier2 = (CosineClassifier(self.projection_dim, self.num_classes, initial_scale=12.0)
                            if self.enable_dual_head else None)
        self.spectral_fusion = SpectralReliabilityFusion(
            self.projection_dim, mode=spectral_fusion_mode
        )
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
        initial = torch.tensor([0.38, 0.47, 0.15], dtype=self.pool.gate[-1].bias.dtype)
        with torch.no_grad():
            self.pool.gate[-1].bias.copy_(initial.log())
        nn.init.zeros_(self.spectral_fusion.gate[-1].weight)
        init_gate = 0.12 / self.spectral_fusion.max_gate
        nn.init.constant_(
            self.spectral_fusion.gate[-1].bias,
            math.log(init_gate / (1.0 - init_gate)),
        )
        if self.amplitude_encoder is not None:
            # Start from "no amplitude contribution" and let the gate open it.
            nn.init.zeros_(self.amplitude_encoder[-1].weight)
            nn.init.zeros_(self.amplitude_encoder[-1].bias)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        if x.ndim != 4:
            raise ValueError("Expected input shape [batch, bands, height, width].")
        if self.amplitude_input:
            if x.size(1) != self.bands + 1:
                raise ValueError(
                    f"amplitude_input expects {self.bands + 1} channels, got {x.size(1)}."
                )
            spectral_x = x[:, :-1]
            amplitude_x = x[:, -1:]
        else:
            spectral_x = x
            amplitude_x = None
        if self.backbone == "dcrn":
            spatial = self.local3d(spectral_x)
            spatial = self.semantic2d(spatial)
            scale_weights = None
        else:
            features3d = self.local3d(spectral_x)
            spatial = features3d.mean(dim=2)
            spatial = self.semantic2d(spatial)
            scale_weights = None
        spatial, scale_weights = self.micro_aspp(spatial)
        pooled, center_weights, attention_map = self.pool(spatial)
        context_features = F.normalize(self.projection(pooled), dim=1)
        spectral_features = self.spectral_encoder(spectral_x)
        context_logits = self.classifier(context_features)
        spectral_logits = self.classifier(spectral_features)
        features, _, spectral_reliability = self.spectral_fusion(
            context_features, spectral_features, context_logits, spectral_logits
        )
        if amplitude_x is not None and self.amplitude_encoder is not None:
            amplitude_features = self.amplitude_encoder(amplitude_x)
            features = F.normalize(
                features + torch.tanh(self.amplitude_gate) * amplitude_features, dim=1
            )
        logits = self.classifier(features)
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
