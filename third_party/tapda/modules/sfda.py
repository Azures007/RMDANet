"""Source-Free Domain Adaptation (SFDA) pipeline for HSI.

Stage 1 trains the encoder and both classifiers on the labelled source domain only.
Stage 2 discards the source data and adapts on the unlabelled target scene with:
  * batch-norm statistics adaptation (the model stays in train mode),
  * reliability-gated pseudo-label supervision (RGS),
  * information maximisation on the batch-averaged prediction,
  * confidence weighted cross entropy on the gated subset.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def source_pretrain(G, F1, F2, source_loader, epochs, lr, device):
    parameters = list(G.parameters()) + list(F1.parameters()) + list(F2.parameters())
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    G.train()
    F1.train()
    F2.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        batches = 0
        for data, target in source_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long()
            features, _ = G(data)
            logits_1 = F1(features)[0]
            logits_2 = F2(features)[0]
            loss = criterion(logits_1, target) + criterion(logits_2, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        print(f"[sfda] source epoch {epoch}/{epochs} loss={total / max(1, batches):.4f}")
    return float(total / max(1, batches))


def sfda_adapt(G, F1, F2, target_loader, epochs, lr, tau, im_weight, entropy_weight=0.05):
    parameters = list(G.parameters()) + list(F1.parameters()) + list(F2.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    G.train()
    F1.train()
    F2.train()
    for epoch in range(1, epochs + 1):
        total_ce, total_im, batches, covered = 0.0, 0.0, 0, 0
        for item in target_loader:
            data = item[0].to(next(G.parameters()).device, non_blocking=True)
            features, _ = G(data)
            logits = F1(features)[0]
            probability = torch.softmax(logits.float(), dim=1).clamp_min(1e-8)
            confidence, pseudo_label = probability.max(dim=1)
            keep = confidence >= tau
            if not bool(keep.any()):
                continue
            weighted_ce = (
                F.cross_entropy(logits[keep], pseudo_label[keep], reduction="none")
                * confidence[keep]
            ).sum() / confidence[keep].sum().clamp_min(1e-6)
            entropy = -(probability[keep] * probability[keep].log()).sum(dim=1).mean()
            mean_probability = probability.mean(dim=0)
            information_max = -(mean_probability * mean_probability.log()).sum()
            loss = weighted_ce + im_weight * information_max + entropy_weight * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_ce += float(weighted_ce.item())
            total_im += float(information_max.item())
            batches += 1
            covered += int(keep.sum().item())
        if batches == 0:
            break
        print(
            f"[sfda] target epoch {epoch}/{epochs} ce={total_ce / batches:.4f} "
            f"im={total_im / batches:.4f} covered={covered / max(1, batches):.1f}/batch"
        )
    return float(total_ce / max(1, batches))


def run_sfda_pipeline(G, F1, F2, source_loader, target_loader, device, num_classes,
                      source_epochs, target_epochs, lr, tau, im_weight):
    print(
        f"[sfda] stage1: source-only training for {source_epochs} epochs "
        f"(num_classes={num_classes}, device={device})"
    )
    source_pretrain(G, F1, F2, source_loader, source_epochs, lr, device)
    print(f"[sfda] stage2: source-free adaptation on target for {target_epochs} epochs (tau={tau})")
    sfda_adapt(G, F1, F2, target_loader, target_epochs, lr * 0.1, tau, im_weight)
