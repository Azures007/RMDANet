"""Target Masked Spectral Adapter with source anchoring (TMSA, simplified).

Freeze the TAPDA backbone, attach a tiny low-rank residual adapter on the
penultimate features, and adapt it on the *unlabeled* target scene with a
masked-spectral reconstruction objective:

    x_masked = x with a contiguous band block zeroed
    feat = G(x_masked) ;  feat' = feat + adapter(feat)
    rec  = decoder(feat') for the masked bands
    loss = MSE(rec, x[masked bands]) + anchor * ||adapter||^2

The decoder is thrown away after adaptation; only the adapter is used at
inference.  No target label, no pseudo-label, no entropy minimisation.
"""

from __future__ import annotations

import random

import numpy as np


class Adapter:
    def __init__(self, dim, rank, device):
        import torch
        import torch.nn as nn

        class _A(nn.Module):
            def __init__(self):
                super().__init__()
                self.down = nn.Linear(dim, rank)
                self.up = nn.Linear(rank, dim)

            def forward(self, x):
                import torch as _t
                return x + self.up(_t.relu(self.down(x)))

        self.net = _A().to(device)

    def __call__(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()


class AdaptedG:
    def __init__(self, G, adapter):
        self.G = G
        self.adapter = adapter

    def __call__(self, x):
        feat, proj = self.G(x)
        return self.adapter(feat), proj

    def eval(self):
        self.G.eval()

    def train(self):
        self.G.train()


def adapt(G, loader, device, args, steps=None):
    """Adapt a low-rank adapter on the unlabeled target loader; returns AdaptedG."""
    import torch
    import torch.nn as nn

    rank = int(getattr(args, "tmsa_rank", 16))
    lr = float(getattr(args, "tmsa_lr", 1e-3))
    n_steps = int(steps if steps is not None else getattr(args, "tmsa_steps", 30))
    k_bands = int(getattr(args, "tmsa_mask", 8))
    anchor = float(getattr(args, "tmsa_anchor", 1e-4))

    # probe the feature dimension
    G.eval()
    with torch.no_grad():
        for data, _ in loader:
            feat, _ = G(data[:1].to(device))
            dim = feat.shape[1]
            break

    adapter = Adapter(dim, rank, device)
    decoder = nn.Linear(dim, int(data.shape[1])).to(device)
    params = list(adapter.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    loader_iter = iter(loader)
    step = 0
    while step < n_steps:
        try:
            data, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            data, _ = next(loader_iter)
        x = data.to(device).float()
        nb = x.shape[1]
        k = min(k_bands, max(1, nb - 1))
        start = random.randint(0, nb - k)
        idx = list(range(start, start + k))
        x_in = x.clone()
        x_in[:, idx] = 0.0
        with torch.no_grad():
            feat, _ = G(x_in)
        feat = adapter(feat)
        rec = decoder(feat)[:, idx]
        hw = x.shape[-1] // 2
        target = x[:, idx, hw, hw].detach()
        loss = ((rec - target) ** 2).mean() + anchor * sum((p ** 2).sum() for p in adapter.parameters())
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
    print(f"[tmsa] adapter adapted for {n_steps} steps (rank={rank}, "
          f"final recon loss {float(loss):.4f})")
    return AdaptedG(G, adapter)
