from __future__ import annotations

import torch
import torch.nn as nn


class PhaseReliabilityResidualAmplitudeTransfer(nn.Module):
    """Phase-reliability-guided residual amplitude transfer.

    Target labels are never used.  Target log-amplitude statistics are maintained
    with a robust EMA reference.  Cross-band phase coherence, source spectral
    energy and source-target amplitude distance jointly gate each frequency.
    """

    def __init__(
        self,
        bands: int,
        patch_size: int,
        ema_momentum: float = 0.96,
        low_frequency_ratio: float = 0.36,
        amplitude_temperature: float = 0.80,
        energy_temperature: float = 0.65,
        delta_clip: float = 1.25,
        mad_clip: float = 3.0,
        reference_update_interval: int = 4,
        mode: str = "full",
    ) -> None:
        super().__init__()
        if mode not in {"fixed", "phase", "phase_energy", "full", "replace"}:
            raise ValueError("PRAT mode must be fixed, phase, phase_energy, full or replace.")
        self.mode = mode
        self.bands = int(bands)
        self.patch_size = int(patch_size)
        self.ema_momentum = float(ema_momentum)
        self.amplitude_temperature = max(float(amplitude_temperature), 1e-4)
        self.energy_temperature = max(float(energy_temperature), 1e-4)
        self.delta_clip = float(delta_clip)
        self.mad_clip = float(mad_clip)
        self.reference_update_interval = max(1, int(reference_update_interval))

        mask = self._build_low_frequency_mask(self.patch_size, low_frequency_ratio)
        self.register_buffer("low_frequency_mask", mask[None, None], persistent=True)
        self.register_buffer(
            "target_log_amplitude_reference",
            torch.zeros(1, self.bands, self.patch_size, self.patch_size),
            persistent=True,
        )
        self.register_buffer("reference_initialized", torch.tensor(False), persistent=True)
        self.register_buffer("reference_updates", torch.tensor(0, dtype=torch.long), persistent=True)

    @staticmethod
    def _build_low_frequency_mask(size: int, ratio: float) -> torch.Tensor:
        ratio = float(min(max(ratio, 0.05), 0.95))
        axis = torch.arange(size, dtype=torch.float32)
        center = (size - 1) / 2.0
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        radius = torch.sqrt((yy - center) ** 2 + (xx - center) ** 2)
        sigma = max(1.0, ratio * size * 0.55)
        mask = torch.exp(-0.5 * (radius / sigma) ** 2)
        mask = torch.fft.ifftshift(mask)
        return mask.clamp(0.0, 1.0)

    @staticmethod
    def _lower_median(values: torch.Tensor, dim: int) -> torch.Tensor:
        sorted_values = torch.sort(values, dim=dim).values
        index = max(0, (values.size(dim) - 1) // 2)
        return sorted_values.select(dim, index)

    @torch.no_grad()
    def _robust_batch_reference(self, batch_log_amplitude: torch.Tensor) -> torch.Tensor:
        # CUDA median with indices is incompatible with strict deterministic mode.
        # The reference is gradient-free and only [1,C,H,W] returns to the device.
        original_device = batch_log_amplitude.device
        deterministic_cuda = bool(
            batch_log_amplitude.is_cuda and torch.are_deterministic_algorithms_enabled()
        )
        work = batch_log_amplitude.detach().float()
        if deterministic_cuda:
            work = work.cpu()
        median = self._lower_median(work, dim=0).unsqueeze(0)
        absolute_deviation = (work - median).abs()
        mad = self._lower_median(absolute_deviation, dim=0).unsqueeze(0).clamp_min(1e-4)
        lower = median - self.mad_clip * mad
        upper = median + self.mad_clip * mad
        winsorized = torch.minimum(torch.maximum(work, lower), upper)
        reference = winsorized.mean(dim=0, keepdim=True)
        return reference.to(original_device, non_blocking=False)

    @torch.no_grad()
    def _update_target_reference(self, batch_log_amplitude: torch.Tensor) -> None:
        update_index = int(self.reference_updates.item())
        self.reference_updates.add_(1)
        if self.reference_initialized.item() and update_index % self.reference_update_interval != 0:
            return
        reference = self._robust_batch_reference(batch_log_amplitude)
        if not bool(self.reference_initialized.item()):
            self.target_log_amplitude_reference.copy_(reference)
            self.reference_initialized.fill_(True)
        else:
            self.target_log_amplitude_reference.mul_(self.ema_momentum).add_(
                reference, alpha=1.0 - self.ema_momentum
            )

    @staticmethod
    def _phase_coherence(source_phase: torch.Tensor) -> torch.Tensor:
        # Circular coherence across adjacent spectral bands at each frequency.
        if source_phase.size(1) <= 1:
            return torch.ones_like(source_phase)
        phase_difference = source_phase[:, 1:] - source_phase[:, :-1]
        complex_difference = torch.polar(torch.ones_like(phase_difference), phase_difference)
        adjacent = complex_difference.mean(dim=1, keepdim=True).abs().clamp(0.0, 1.0)
        return adjacent.expand_as(source_phase)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        strength: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if source.shape[1:] != target.shape[1:]:
            raise ValueError(
                f"Source and target patch shapes must match, got {source.shape} and {target.shape}."
            )
        if source.size(1) != self.bands:
            raise ValueError(f"Expected {self.bands} bands, received {source.size(1)}.")
        if source.shape[-2:] != (self.patch_size, self.patch_size):
            raise ValueError(
                f"Expected {self.patch_size}x{self.patch_size} patches, got {source.shape[-2:]}."
            )

        source_float = source.float()
        target_float = target.float()
        source_fft = torch.fft.fft2(source_float, dim=(-2, -1))
        target_fft = torch.fft.fft2(target_float, dim=(-2, -1))
        source_amplitude = source_fft.abs().clamp_min(1e-6)
        target_amplitude = target_fft.abs().clamp_min(1e-6)
        source_log_amplitude = source_amplitude.log()
        target_log_amplitude = target_amplitude.log()
        source_phase = torch.angle(source_fft)

        self._update_target_reference(target_log_amplitude.detach())
        target_reference = self.target_log_amplitude_reference.to(source_log_amplitude)
        amplitude_residual = (target_reference - source_log_amplitude).clamp(
            -self.delta_clip, self.delta_clip
        )

        phase_reliability = self._phase_coherence(source_phase)
        energy_center = source_log_amplitude.mean(dim=(-2, -1), keepdim=True)
        energy_scale = source_log_amplitude.std(
            dim=(-2, -1), keepdim=True, unbiased=False
        ).clamp_min(1e-4)
        energy_reliability = torch.sigmoid(
            (source_log_amplitude - energy_center) / (self.energy_temperature * energy_scale)
        )
        amplitude_safety = torch.exp(
            -amplitude_residual.abs() / self.amplitude_temperature
        )
        if self.mode == "fixed":
            reliability = torch.ones_like(phase_reliability)
        elif self.mode == "phase":
            reliability = phase_reliability
        elif self.mode == "phase_energy":
            reliability = phase_reliability * (0.20 + 0.80 * energy_reliability)
        else:
            reliability = (
                phase_reliability * (0.20 + 0.80 * energy_reliability) * amplitude_safety
            )
        reliability = reliability.clamp(0.0, 1.0)
        low_frequency = self.low_frequency_mask.to(reliability)
        if self.mode == "replace":
            transfer_gate = (low_frequency >= 0.50).to(reliability).expand_as(reliability)
            new_log_amplitude = (
                (1.0 - transfer_gate) * source_log_amplitude
                + transfer_gate * target_reference.expand_as(source_log_amplitude)
            )
        else:
            transfer_gate = (
                float(strength) * low_frequency * reliability
            ).clamp(0.0, 1.0)
            new_log_amplitude = source_log_amplitude + transfer_gate * amplitude_residual
        new_amplitude = new_log_amplitude.exp()
        reconstructed_fft = torch.polar(new_amplitude, source_phase)
        transferred = torch.fft.ifft2(reconstructed_fft, dim=(-2, -1)).real
        transferred = transferred.to(dtype=source.dtype)

        sample_reliability = reliability.mean(dim=(1, 2, 3))
        sample_transfer = (transfer_gate * amplitude_residual.abs()).mean(dim=(1, 2, 3))
        metadata = {
            "frequency_reliability": reliability.detach(),
            "sample_reliability": sample_reliability.detach(),
            "sample_transfer": sample_transfer.detach(),
            "mean_transfer": sample_transfer.mean().detach(),
        }
        return transferred, metadata
