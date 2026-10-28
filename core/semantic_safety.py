"""Source-GT anchored, epoch-lagged semantic safety controller.

No target ground truth, target coordinates of labels, or validation target labels
are used here. Target evidence can only cautiously attenuate future augmentation.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


def true_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    pos = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.detach().clone()
    other.scatter_(1, labels[:, None], -torch.inf)
    return pos - other.max(dim=1).values


@torch.no_grad()
def judge_augmentation(clean_logits, augmented_logits, labels, tolerance=0.30):
    """GT-safe decision: teacher correct on both, limited relative margin loss."""
    m0 = true_margin(clean_logits, labels)
    m1 = true_margin(augmented_logits, labels)
    safe = ((clean_logits.argmax(1) == labels) &
            (augmented_logits.argmax(1) == labels) &
            (m1 >= m0 - tolerance * m0.abs().clamp_min(1.0)))
    return safe.detach(), (m0 - m1).detach()


def weighted_ce(logits, labels, weights, label_smoothing=0.0, class_weights=None):
    """Zero gradient if no safe evidence; no dilution via epsilon denominator."""
    per = F.cross_entropy(logits.float(), labels, weight=class_weights,
                          label_smoothing=label_smoothing, reduction='none')
    w = weights.detach().float().clamp_min(0)
    if bool((w.sum() <= 0).item()):
        return logits.sum() * 0.0
    if class_weights is not None:
        denominator = (w * class_weights[labels].float()).sum().clamp_min(1e-8)
    else:
        denominator = w.sum().clamp_min(1e-8)
    return (per * w).sum() / denominator


def class_basis_from_centers(centers: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """Orthonormal class-mean-difference directions, rank <= classes-1."""
    centers = centers.float()
    labels = labels.long()
    means=[]
    for c in range(num_classes):
        if bool((labels == c).any().item()):
            means.append(centers[labels == c].mean(0))
    if len(means) < 2:
        return centers.new_zeros((centers.size(-1), 0))
    centered = torch.stack(means) - torch.stack(means).mean(0)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    if singular.numel() == 0 or singular[0] < 1e-9:
        return centers.new_zeros((centers.size(-1), 0))
    rank = int((singular > singular[0] * 1e-5).sum().item())
    return vh[:min(rank, len(means)-1)].T.contiguous()


@torch.no_grad()
def transport(source: torch.Tensor, target: torch.Tensor, basis: torch.Tensor,
              per_class_scale: torch.Tensor, labels: torch.Tensor,
              mode='projected', max_ratio=0.12):
    """Bounded batch mean source->target spectral shift, per-class feedback scaling."""
    h,w=source.shape[-2:]
    src_center=source[:,:,h//2,w//2].float()
    tgt_center=target[:,:,target.shape[-2]//2,target.shape[-1]//2].float()
    delta=tgt_center.mean(0)-src_center.mean(0)
    if mode == 'projected' and basis.numel():
        q=basis.to(delta.device, delta.dtype)
        delta = delta - q @ (q.T @ delta)
    reference=src_center.norm(dim=1).median().clamp_min(1e-6)
    max_norm = max_ratio * reference
    delta=delta * (max_norm / delta.norm().clamp_min(1e-8)).clamp(max=1)
    scales=per_class_scale[labels].to(source.device).view(-1,1,1,1)
    pixel_valid=(source.float().abs().sum(dim=1, keepdim=True) > 1e-10)
    augmented=source.float()+scales * delta.reshape(1,-1,1,1)*pixel_valid
    shift=(scales[:,0,0,0].mean()*delta.norm()/reference).item()
    return augmented.to(source.dtype), float(shift)


@dataclass
class FeedbackDiagnostics:
    prat_pass: float = 0.0
    cnt_pass: float = 0.0
    cnt_strength: float = 0.0
    target_support: float = 0.0


class SafetyFeedback:
    """Separate PRAT/CNT class caps, committed only at epoch boundary.

    Target pseudo evidence is deliberately not interpreted as target correctness;
    it may only mildly attenuate (never increase) both augmentation strengths.
    """
    def __init__(self, num_classes, device, min_source=8, target_support=20,
                 ema_decay=0.8, min_scale=0.5):
        self.n=num_classes;self.device=device
        self.min_source=min_source;self.min_target=target_support
        self.decay=ema_decay;self.min_scale=min_scale
        self.scale_cnt=torch.ones(num_classes,device=device)
        self.scale_prat=torch.ones(num_classes,device=device)
        self.source_pass_cnt=torch.ones(num_classes,device=device)
        self.source_pass_prat=torch.ones(num_classes,device=device)
        self.target_factor=torch.ones(num_classes,device=device)
        self.reset()

    def reset(self):
        self.cnt_safe=torch.zeros(self.n,device=self.device)
        self.cnt_total=torch.zeros(self.n,device=self.device)
        self.prat_safe=torch.zeros(self.n,device=self.device)
        self.prat_total=torch.zeros(self.n,device=self.device)
        self.target=torch.zeros(self.n,device=self.device)
        self.target_consensus=torch.zeros(self.n,device=self.device)

    @torch.no_grad()
    def observe_source(self, labels, passed, branch='cnt'):
        if branch not in ('cnt','prat'):raise ValueError(branch)
        total=getattr(self,branch+'_total')
        good=getattr(self,branch+'_safe')
        for c in range(self.n):
            mask=labels == c
            total[c] += mask.sum()
            good[c] += passed[mask].float().sum()

    @torch.no_grad()
    def observe_target(self, pseudo_labels, stable_mask, agreement):
        for c in range(self.n):
            mask = (pseudo_labels == c) & stable_mask
            self.target[c] += mask.sum()
            self.target_consensus[c] += (mask & agreement).sum()

    @torch.no_grad()
    def observe_target_independent(self, predicted_labels, spectral_agreement):
        """Track perturbation agreement for ALL target samples, not stable_mask.

        This measures input-perturbation robustness but cannot verify semantic
        correctness. The target evidence only attenuates augmentation strength.
        """
        score = spectral_agreement.detach().float().clamp(0, 1)
        for c in range(self.n):
            mask = predicted_labels.eq(c)
            self.target[c] += mask.sum()
            self.target_consensus[c] += score[mask].sum()

    @torch.no_grad()
    def end_epoch(self, use_target=True):
        if use_target:
            confident=self.target >= self.min_target
            agreement=self.target_consensus/self.target.clamp_min(1)
            self.target_factor.fill_(1.0)
            self.target_factor[confident]=(0.70+0.30*agreement[confident]).clamp(.70,1.)
        else:
            self.target_factor.fill_(1.0)
        for branch in ('cnt','prat'):
            total=getattr(self,branch+'_total')
            good=getattr(self,branch+'_safe')
            ratio=getattr(self,'source_pass_'+branch)
            sufficient=total >= self.min_source
            observed=good/total.clamp_min(1)
            ratio[sufficient]=self.decay*ratio[sufficient]+(1-self.decay)*observed[sufficient]
            setattr(self,'scale_'+branch,((0.4+0.6*ratio).clamp(self.min_scale,1.0)*self.target_factor).detach())
        result={
            'safety_scale_cnt_mean':float(self.scale_cnt.mean().item()),
            'safety_scale_prat_mean':float(self.scale_prat.mean().item()),
            'source_safe_cnt_mean':float(self.source_pass_cnt.mean().item()),
            'source_safe_prat_mean':float(self.source_pass_prat.mean().item()),
            'target_supported_classes':int((self.target>=self.min_target).sum().item()),
            'target_agreement_mean':float((self.target_consensus/self.target.clamp_min(1)).mean().item()),
        }
        self.reset()
        return result
