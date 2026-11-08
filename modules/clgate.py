"""CLG: per-pixel consensus label gate.

RATC/SSTI apply one correction to the whole scene, but a correction is rarely
uniformly beneficial: in Pavia the MV-LCR branch changes many pixels that the
base model already classified correctly, while in Houston TENT destroys class
boundaries.  CLG keeps a corrected label only when at least K independent
views (F2, source spectral classifier, spatial neighbourhood, EMA teacher)
agree with it; otherwise the base prediction is restored.  The gated map is
then passed through SSTI again so the refinement and the gate form a loop.
"""

from __future__ import annotations

import numpy as np


def _spatial_majority(labels, coords, shape):
    rows = np.asarray(coords[0], dtype=np.int64)
    cols = np.asarray(coords[1], dtype=np.int64)
    height, width = int(shape[0]), int(shape[1])
    num_classes = int(labels.max()) + 1 if len(labels) else 1
    grid = np.full((height, width), -1, dtype=np.int64)
    grid[rows, cols] = labels
    votes = np.zeros((len(labels), num_classes), dtype=np.float32)
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
            np.add.at(votes, (vi, ni), 1.0)
    return votes.argmax(axis=1)


def consensus_gate(base_labels, corrected_labels, views, coords, shape,
                   min_votes=2, use_spatial=True):
    """views: list of label arrays (same length as base_labels)."""
    base = np.asarray(base_labels, dtype=np.int64)
    corr = np.asarray(corrected_labels, dtype=np.int64)
    n = len(base)
    if len(corr) != n:
        return corr, 0.0

    all_views = []
    for v in views:
        v = np.asarray(v)
        if v.shape[0] == n:
            all_views.append(v.astype(np.int64))
    if use_spatial and coords is not None and shape is not None:
        all_views.append(_spatial_majority(base, coords, shape))

    if not all_views:
        return corr, 0.0

    support = np.zeros(n, dtype=np.int64)
    for v in all_views:
        support += (v == corr).astype(np.int64)

    accept = support >= int(min_votes)
    gated = np.where(accept, corr, base)
    changed = float((gated != corr).mean())
    return gated, changed
