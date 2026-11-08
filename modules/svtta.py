"""SV-TTA: self-verifying test-time adaptation.

Label-free verification of a candidate correction:
  * spatial consistency of the predicted label map (fraction of 8-neighbours
    sharing the centre label) -- a correction that destroys the spatial
    structure of the scene is rejected;
  * class-prior plausibility (chi-square distance to the BBSE-style target
    prior estimated from the base predictions);
  * mean entropy (small weight, TENT can lower entropy artificially).

The verification is used by RATC to accept/reject a candidate correction and
to fall back to the untouched prediction, closing the loop
   RATC proposes -> verification (SSTI side) accepts/rejects -> SSTI refines.
"""

from __future__ import annotations

import numpy as np


def predict_labels(G, F1, F2, loader, device):
    """argmax of the F1+F2 ensemble over a loader (no labels used)."""
    import torch

    out_labs = []
    with torch.no_grad():
        for batch in loader:
            data = batch[0] if isinstance(batch, (tuple, list)) else batch
            data = data.to(device)
            out, _ = G(data)
            l1 = F1(out)
            l2 = F2(out)
            if isinstance(l1, (tuple, list)):
                l1 = l1[0]
            if isinstance(l2, (tuple, list)):
                l2 = l2[0]
            out_labs.append((l1 + l2).argmax(dim=1).cpu().numpy())
    return np.concatenate(out_labs, 0)


def spatial_consistency(labels, coords, shape):
    labels = np.asarray(labels, dtype=np.int64)
    rows = np.asarray(coords[0], dtype=np.int64)
    cols = np.asarray(coords[1], dtype=np.int64)
    height, width = int(shape[0]), int(shape[1])
    if len(rows) != len(labels):
        return float("nan")
    grid = np.full((height, width), -1, dtype=np.int64)
    grid[rows, cols] = labels
    agree = 0
    total = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = rows + dr, cols + dc
            inside = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
            vi = np.nonzero(inside)[0]
            if len(vi) == 0:
                continue
            ni = grid[nr[vi], nc[vi]]
            keep = ni >= 0
            vi, ni = vi[keep], ni[keep]
            if len(vi) == 0:
                continue
            agree += int((labels[vi] == ni).sum())
            total += int(len(vi))
    return float(agree / max(1, total))


def prior_plausibility(labels, num_classes, reference=None):
    hist = np.bincount(np.asarray(labels, dtype=np.int64),
                       minlength=int(num_classes)).astype(np.float64)
    hist = hist / max(1.0, hist.sum())
    if reference is None:
        return 0.0
    ref = np.asarray(reference, dtype=np.float64)
    if ref.sum() <= 0:
        return 0.0
    ref = ref / ref.sum()
    # symmetric chi-square style distance, lower is better
    denom = np.maximum(ref + hist, 1e-8)
    return float(np.sum((hist - ref) ** 2 / denom))


def verification_score(labels, coords, shape, num_classes, reference_prior,
                       entropy=None, weights=(1.0, 0.5, 0.1)):
    cons = spatial_consistency(labels, coords, shape)
    prior = prior_plausibility(labels, num_classes, reference_prior)
    ent = 0.0 if entropy is None else float(entropy)
    w_cons, w_prior, w_ent = weights
    score = w_cons * cons - w_prior * prior - w_ent * ent
    return float(score), {'consistency': cons, 'prior_dist': prior, 'entropy': ent}
