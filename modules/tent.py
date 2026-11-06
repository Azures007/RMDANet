"""TENT-style test-time adaptation for TAPDA.

Freeze every weight; enable only the affine parameters of BatchNorm layers and
minimise the entropy of the ensemble prediction on the *unlabeled* target
loader.  BN running statistics are updated (module in train mode) while the
convolutional weights stay frozen.

    loss = -sum_c p_c log p_c ,  p = softmax(0.5*(F1(feat)+F2(feat)))

No target label, no pseudo-label, no graph prior.
"""

from __future__ import annotations


def tent_adapt(G, F1, F2, loader, device, args):
    import torch
    import torch.nn as nn

    steps = int(getattr(args, "tent_steps", 10))
    lr = float(getattr(args, "tent_lr", 1e-3))
    if steps <= 0:
        return

    bn_params = []
    for net in (G, F1, F2):
        net.eval()
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                    bn_params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad_(True)
                    bn_params.append(m.bias)
    if not bn_params:
        print("[tent] no BatchNorm affine parameters found, skipped")
        return

    opt = torch.optim.Adam(bn_params, lr=lr)
    it = iter(loader)
    step = 0
    last_ent = None
    while step < steps:
        try:
            data, _ = next(it)
        except StopIteration:
            it = iter(loader)
            data, _ = next(it)
        data = data.to(device).float()
        feat, _ = G(data)
        l1 = F1(feat)
        l2 = F2(feat)
        if isinstance(l1, tuple):
            l1 = l1[0]
        if isinstance(l2, tuple):
            l2 = l2[0]
        logits = 0.5 * (l1 + l2)
        p = torch.softmax(logits, dim=1)
        ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
        opt.zero_grad()
        ent.backward()
        opt.step()
        last_ent = float(ent.detach())
        step += 1

    for net in (G, F1, F2):
        net.eval()
    print(f"[tent] BN affine adapted for {steps} steps, final entropy {last_ent:.4f}")
