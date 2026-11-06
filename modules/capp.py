"""Covariance-Aware Prototype Posterior (CAPP).

TAPDA represents each source class only by a prototype mean (and uses cosine
similarity in PATS).  It never uses the *within-class geometry* of the source
embeddings.  CAPP estimates a shrunk class-conditional covariance on the source
domain, evaluates the Mahalanobis energy of each target embedding, and fuses the
resulting geometric posterior with the TAPDA logits:

    logits' = (1 - alpha) * logits + alpha * log p_cov

Nothing is fed back into training and no target label is used.
"""

from __future__ import annotations

import numpy as np


def build_stats(G, loader, num_classes, device, shrink=0.2):
    """Source class means and shrunk inverse covariances from the embedding."""
    import torch

    G.eval()
    feats, labels = [], []
    with torch.no_grad():
        for data, label in loader:
            out, _ = G(data.to(device))
            feats.append(out.detach().cpu())
            labels.append(label.detach().cpu())
    Z = torch.cat(feats).numpy().astype(np.float64)
    y = torch.cat(labels).numpy().astype(np.int64)
    if y.min() == 1 and y.max() == num_classes:
        y = y - 1

    mus, invs = [], []
    d = Z.shape[1]
    for c in range(int(num_classes)):
        Zc = Z[y == c]
        if len(Zc) < 2:
            mu = np.zeros(d)
            cov = np.eye(d)
        else:
            mu = Zc.mean(axis=0)
            cov = np.cov(Zc, rowvar=False)
            cov = np.atleast_2d(cov)
        cov = (1.0 - float(shrink)) * cov + float(shrink) * (np.trace(cov) / d) * np.eye(d)
        cov = cov + 1e-6 * np.eye(d)
        mus.append(mu)
        invs.append(np.linalg.pinv(cov))
    print(f"[capp] source stats built: {len(Z)} samples, dim={d}, shrink={shrink}")
    return {"mu": np.stack(mus), "inv": np.stack(invs)}


def capp_log_probs(feats, stats, temperature=1.0):
    """log p_cov(c|x) from Mahalanobis energies (numpy/torch tensor in, torch out)."""
    import torch

    Z = feats.detach().cpu().numpy().astype(np.float64)
    mu, inv = stats["mu"], stats["inv"]
    energies = []
    for c in range(len(mu)):
        diff = Z - mu[c]
        energies.append(np.einsum("ij,jk,ik->i", diff, inv[c], diff))
    E = np.stack(energies, axis=1)
    logits = -E / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return torch.from_numpy(np.log(p + 1e-12)).float()
