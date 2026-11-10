"""Adaptation losses for the baseline methods.

All losses share the same interface:
    forward(out_s, out_t, p, ys=None)
where out_s / out_t are model outputs for a source / target batch, p is the
normalised training progress in [0, 1] and ys are the source labels.

The MTLDA / CACL / UIPCL / PCDM / VCMoE entries are re-implementations of the
core losses described in their papers, adapted to the identical data protocol
(180 source labels per class, 50x40 schedule, patch 7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# multi-kernel MMD (DAN / JAN)
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

    def forward(self, out_s, out_t, p, ys=None):
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

    def forward(self, out_s, out_t, p, ys=None):
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

    def forward(self, out_s, out_t, p, ys=None):
        return self.weight * self.mmd(self._joint(out_s), self._joint(out_t))


# --------------------------------------------------------------------------- #
# supervised contrastive loss (used by CACL / UIPCL / PCDM)
# --------------------------------------------------------------------------- #
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, weights=None):
        z = F.normalize(features, dim=1)
        sim = z @ z.t() / self.temperature
        n = z.size(0)
        eye = torch.eye(n, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e4)
        labels = labels[:, None]
        pos = (labels == labels.t()) & (~eye)
        pos_count = pos.sum(1).clamp_min(1).float()
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        loss = -(log_prob * pos.float()).sum(1) / pos_count
        if weights is not None:
            loss = loss * weights
        return loss.mean()


# --------------------------------------------------------------------------- #
# MTLDA: classification + domain discrimination + entropy minimisation
# --------------------------------------------------------------------------- #
class MTLDALoss(nn.Module):
    def __init__(self, feature_dim: int, device,
                 domain_weight: float = 0.5, entropy_weight: float = 0.05):
        super().__init__()
        self.adv = DANNLoss(feature_dim, device, domain_weight=domain_weight)
        self.entropy_weight = entropy_weight

    def forward(self, out_s, out_t, p, ys=None):
        adv = self.adv(out_s, out_t, p)
        prob = F.softmax(out_t.logits, dim=1).clamp_min(1e-8)
        entropy = -(prob * prob.log()).sum(1).mean()
        return adv + self.entropy_weight * entropy

    def post_step(self):
        self.adv.post_step()


# --------------------------------------------------------------------------- #
# CACL: class-aware contrastive learning
# --------------------------------------------------------------------------- #
class CACLLoss(nn.Module):
    def __init__(self, contrast_weight: float = 0.5,
                 pseudo_weight: float = 0.3, confidence: float = 0.9):
        super().__init__()
        self.contrast = SupConLoss()
        self.contrast_weight = contrast_weight
        self.pseudo_weight = pseudo_weight
        self.confidence = confidence

    def forward(self, out_s, out_t, p, ys=None):
        loss = out_s.logits.new_zeros(())
        if ys is not None:
            loss = loss + self.contrast_weight * self.contrast(out_s.features, ys)
        prob = F.softmax(out_t.logits, dim=1)
        conf, pseudo = prob.max(1)
        mask = conf >= self.confidence
        if mask.any():
            loss = loss + self.pseudo_weight * self.contrast(
                out_t.features[mask], pseudo[mask], weights=conf[mask]
            )
        return loss


# --------------------------------------------------------------------------- #
# UIPCL: uncertainty-informed prototype contrastive learning
# --------------------------------------------------------------------------- #
class UIPCLLoss(nn.Module):
    def __init__(self, num_classes: int, prototype_weight: float = 0.5,
                 pseudo_weight: float = 0.5, temperature: float = 0.07,
                 momentum: float = 0.9):
        super().__init__()
        self.num_classes = num_classes
        self.prototype_weight = prototype_weight
        self.pseudo_weight = pseudo_weight
        self.temperature = temperature
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, 0))

    def _update_prototypes(self, out_s, ys):
        feat = F.normalize(out_s.features.detach(), dim=1)
        dim = feat.size(1)
        if self.prototypes.numel() == 0 or self.prototypes.size(1) != dim:
            self.prototypes = feat.new_zeros(self.num_classes, dim)
        for c in ys.unique():
            mask = ys == c
            mean = feat[mask].mean(0)
            self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * mean

    def forward(self, out_s, out_t, p, ys=None):
        if ys is not None:
            self._update_prototypes(out_s, ys)
        prob = F.softmax(out_t.logits, dim=1).clamp_min(1e-8)
        conf, pseudo = prob.max(1)
        # uncertainty weight: 1 - normalised entropy
        entropy = -(prob * prob.log()).sum(1) / torch.log(torch.tensor(float(self.num_classes),
                                                                       device=prob.device))
        weight = (conf * (1.0 - entropy)).clamp(0.0, 1.0)
        z = F.normalize(out_t.features, dim=1)
        proto = F.normalize(self.prototypes, dim=1)
        if proto.size(0) == 0 or proto.abs().sum() == 0:
            return out_t.logits.new_zeros(())
        logits = z @ proto.t() / self.temperature
        proto_loss = F.cross_entropy(logits, pseudo, reduction="none")
        proto_loss = (proto_loss * weight).mean()
        pseudo_loss = F.cross_entropy(out_t.logits, pseudo, reduction="none")
        pseudo_loss = (pseudo_loss * weight).mean()
        return self.prototype_weight * proto_loss + self.pseudo_weight * pseudo_loss


# --------------------------------------------------------------------------- #
# PCDM-UDA: pseudo-class distribution guided matching
# --------------------------------------------------------------------------- #
class PCDMLoss(nn.Module):
    def __init__(self, num_classes: int, prior_weight: float = 0.5,
                 contrast_weight: float = 0.3, confidence: float = 0.9):
        super().__init__()
        self.num_classes = num_classes
        self.prior_weight = prior_weight
        self.contrast_weight = contrast_weight
        self.confidence = confidence
        self.register_buffer("source_prior", torch.full((num_classes,), 1.0 / num_classes))
        self.contrast = SupConLoss()

    def forward(self, out_s, out_t, p, ys=None):
        if ys is not None:
            counts = torch.bincount(ys, minlength=self.num_classes).float()
            prior = (counts / counts.sum().clamp_min(1.0)).to(self.source_prior.device)
            self.source_prior = 0.9 * self.source_prior + 0.1 * prior
        prob = F.softmax(out_t.logits, dim=1)
        target_prior = prob.mean(0)
        prior_loss = F.kl_div(torch.log(target_prior.clamp_min(1e-8)),
                              self.source_prior.clamp_min(1e-8), reduction="sum")
        conf, pseudo = prob.max(1)
        mask = conf >= self.confidence
        contrast_loss = out_t.logits.new_zeros(())
        if mask.any():
            contrast_loss = self.contrast(out_t.features[mask], pseudo[mask], weights=conf[mask])
        return self.prior_weight * prior_loss + self.contrast_weight * contrast_loss


# --------------------------------------------------------------------------- #
# VCMoE: variational contrastive mixture-of-experts (approximation)
# --------------------------------------------------------------------------- #
class VCMoELoss(nn.Module):
    def __init__(self, feature_dim: int, device, adv_weight: float = 0.3,
                 contrast_weight: float = 0.3, diversity_weight: float = 0.1):
        super().__init__()
        self.adv = DANNLoss(feature_dim, device, domain_weight=adv_weight)
        self.contrast = SupConLoss()
        self.contrast_weight = contrast_weight
        self.diversity_weight = diversity_weight

    def forward(self, out_s, out_t, p, ys=None):
        loss = self.adv(out_s, out_t, p)
        if ys is not None:
            loss = loss + self.contrast_weight * self.contrast(out_s.features, ys)
        prob = F.softmax(out_t.logits, dim=1).clamp_min(1e-8)
        conf, pseudo = prob.max(1)
        mask = conf >= 0.9
        if mask.any():
            loss = loss + self.contrast_weight * self.contrast(
                out_t.features[mask], pseudo[mask], weights=conf[mask]
            )
        # variational diversity: push expert/feature directions apart
        z = F.normalize(out_t.features, dim=1)
        gram = z @ z.t()
        off = gram - torch.diag_embed(torch.diagonal(gram))
        loss = loss + self.diversity_weight * off.abs().mean()
        return loss

    def post_step(self):
        self.adv.post_step()
