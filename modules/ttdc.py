"""Temporal Trajectory Disagreement Correction (TTDC).

Idea
----
TAPDA only reports the final-epoch predictions, but the model produces a
prediction trajectory during training.  A pixel whose predicted class keeps
switching across training is *unstable*, and single-step confidence does not
capture this (teacher confidence 0.945 vs accuracy 79.25%).

TTDC collects the target predictions at a few fixed epochs, builds a class
frequency prior per pixel, and at evaluation adds a small log-prior term to the
classifier logits:

    logits'(x) = logits(x) + gamma * log(freq(x) + eps)

Stable pixels are unaffected (their frequency is concentrated), while unstable
pixels are gently pulled towards the classes they visited during training.
Nothing is fed back into training; target labels are never used.
"""

from __future__ import annotations

import numpy as np


def collect_predictions(G, F1, F2, loader, device, mode='ensemble'):
    """One eval pass over the target loader -> predicted class per pixel."""
    import torch

    was_training = G.training
    G.eval()
    F1.eval()
    F2.eval()
    preds = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            feats, _ = G(data)
            logits1 = F1(feats)
            logits2 = F2(feats)
            if isinstance(logits1, tuple):
                logits1 = logits1[0]
            if isinstance(logits2, tuple):
                logits2 = logits2[0]
            logits = 0.5 * (logits1 + logits2) if mode == 'ensemble' else logits1
            preds.append(logits.argmax(dim=1).cpu().numpy())
    if was_training:
        G.train()
        F1.train()
        F2.train()
    return np.concatenate(preds)


def build_bias(pred_list, num_classes, gamma=0.5):
    """Log class-frequency prior from the collected trajectory predictions."""
    n = len(pred_list[0])
    counts = np.zeros((n, int(num_classes)), dtype=np.float32)
    rows = np.arange(n)
    for pred in pred_list:
        counts[rows, pred.astype(np.int64)] += 1.0
    freq = counts / max(1, len(pred_list))
    bias = float(gamma) * np.log(freq + 1e-6)
    print(f"[ttdc] trajectory bias built from {len(pred_list)} checkpoints, "
          f"gamma={gamma}, mean stability={float((freq.max(axis=1)).mean()):.3f}")
    return bias.astype(np.float32)
