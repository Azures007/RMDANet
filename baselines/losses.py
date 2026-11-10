"""Adaptation losses for the DAN / DANN / JAN baselines."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# multi-kernel MMD (used by DAN and JAN)
# --------------------------------------------------------------------------- #
class MultipleKernelMMD(nn.Module):
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5, fix_sigma=None):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        total = torch.cat([source, target], dim=0)
        n = total.size(0)
        diff = total.unsqueeze(0) - total.unsqueeze(1)
        l2 = (diff ** 2).sum(-1)
        if self.fix_sigma:
            bandwidth = torch.tensor(float(self.fix_sigma), device=l2.device)
        else:
            bandwidth = l2.detach().sum() / (n * n - n)
        bandwidth = bandwidth.clamp_min(1e-6) / (self.kernel_mul ** (self.kernel_num // 2))
        kernels = sum(
            torch.exp(-l2 / (bandwidth * (self.kernel_mul ** i)).clamp_min(1e-6) / 2.0)
            for i in range(self.kernel_num)
        )
        return kernels

    def forward(self, source, target):
        batch = min(source.size(0), target.size(0))
        source, target = source[:batch], target[:batch]
        kernels = self.gaussian_kernel(source, target)
        n = batch
        return (
            kernels[:n, :n].mean()
            + kernels[n:, n:].mean()
            - kernels[:n, n:].mean()
            - kernels[n:, :n].mean()
        ).clamp_min(0.0)


class DANLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.mmd = MultipleKernelMMD()

    def forward(self, out_s, out_t, p: float):
        return self.weight * self.mmd(out_s.features, out_t.features)


# --------------------------------------------------------------------------- #
# DANN: gradient reversal + domain discriminator
# --------------------------------------------------------------------------- #
class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class DANNLoss(nn.Module):
    def __init__(self, feature_dim: int, device, domain_weight: float = 1.0):
        super().__init__()
        self.disc = DomainDiscriminator(feature_dim).to(device)
        self.disc_optim = torch.optim.AdamW(self.disc.parameters(), lr=1e-3, weight_decay=1e-4)
        self.domain_weight = domain_weight

    def forward(self, out_s, out_t, p: float):
        lambd = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0
        fs = _GradientReversal.apply(out_s.features, float(lambd))
        ft = _GradientReversal.apply(out_t.features, float(lambd))
        logits = self.disc(torch.cat([fs, ft], dim=0))
        labels = torch.cat([torch.zeros(fs.size(0), device=fs.device),
                            torch.ones(ft.size(0), device=fs.device)])
        return self.domain_weight * F.binary_cross_entropy_with_logits(logits, labels)

    @torch.no_grad()
    def post_step(self):
        self.disc_optim.step()
        self.disc_optim.zero_grad(set_to_none=True)


# --------------------------------------------------------------------------- #
# JAN: joint MMD over (features, class probabilities)
# --------------------------------------------------------------------------- #
class JANLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.mmd = MultipleKernelMMD()

    @staticmethod
    def _joint(out):
        prob = F.softmax(out.logits, dim=1)
        return torch.cat([out.features, prob], dim=1)

    def forward(self, out_s, out_t, p: float):
        return self.weight * self.mmd(self._joint(out_s), self._joint(out_t))
