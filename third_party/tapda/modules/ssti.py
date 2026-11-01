"""Source-Structured Transductive Inference (SSTI). See docs for the design."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def build_transition_prior(label_map, num_classes, smoothing=0.1):
    label_map = np.asarray(label_map).astype(np.int64)
    height, width = label_map.shape
    padded = np.pad(label_map, 1, mode="constant", constant_values=0)
    counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbour = padded[1 + dr:1 + dr + height, 1 + dc:1 + dc + width]
            valid = (label_map > 0) & (neighbour > 0)
            np.add.at(counts, (label_map[valid] - 1, neighbour[valid] - 1), 1.0)
    counts += smoothing
    return counts / counts.sum(axis=1, keepdims=True)


def source_confusion(G, F1, source_loader, num_classes, device, max_batches=100):
    G.eval()
    F1.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    with torch.no_grad():
        for index, (data, target) in enumerate(source_loader):
            if index >= max_batches:
                break
            features, _ = G(data.to(device))
            logits = F1(features)[0]
            np.add.at(
                confusion,
                (target.cpu().numpy(), logits.argmax(dim=1).cpu().numpy()),
                1.0,
            )
    return confusion


def build_context(source_label_map, target_label_map, G, F1, source_loader,
                  num_classes, device, prior_strength=1.0, smoothing=0.1):
    transition = build_transition_prior(source_label_map, num_classes, smoothing)
    confusion = source_confusion(G, F1, source_loader, num_classes, device)
    row_sum = confusion.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    conditional = confusion / row_sum
    source_map = np.asarray(source_label_map).astype(np.int64)
    histogram = np.array([(source_map == c).sum() for c in range(1, num_classes + 1)], dtype=np.float64)
    source_prior = (histogram + 1.0) / (histogram.sum() + num_classes)
    coords = np.nonzero(np.asarray(target_label_map) > 0)
    return {
        "transition": torch.from_numpy(transition).float().to(device),
        "confusion": torch.from_numpy(conditional).float(),
        "source_prior": torch.from_numpy(source_prior).float(),
        "coords": coords,
        "shape": np.asarray(target_label_map).shape,
        "num_classes": num_classes,
        "prior_strength": float(prior_strength),
    }


def target_prior_bias(context, target_predictions):
    num_classes = context["num_classes"]
    histogram = np.zeros(num_classes, dtype=np.float64)
    np.add.at(histogram, target_predictions, 1.0)
    histogram = histogram / max(histogram.sum(), 1.0)
    confusion = context["confusion"].cpu().numpy()
    try:
        estimated = np.linalg.lstsq(confusion, histogram, rcond=None)[0]
    except np.linalg.LinAlgError:
        estimated = histogram
    estimated = np.clip(estimated, 0.0, None)
    if estimated.sum() <= 0:
        estimated = histogram.copy()
    estimated = (estimated + 1e-3) / (estimated.sum() + 1e-3 * num_classes)
    source_prior = context["source_prior"].cpu().numpy()
    bias = np.log(estimated / np.clip(source_prior, 1e-8, None))
    bias = bias - bias.mean()
    return torch.from_numpy(bias).float().to(context["transition"].device)


def refine_labels(logits, context, labels, alpha=0.3, beta=0.5, iterations=3,
                  temperature=1.0, device=None):
    device = device if device is not None else context["transition"].device
    num_classes = context["num_classes"]
    rows, cols = context["coords"]
    height, width = context["shape"]
    log_probability = F.log_softmax(logits.to(device) / float(temperature), dim=1)

    initial = log_probability.argmax(dim=1).cpu().numpy()
    bias = target_prior_bias(context, initial)
    log_transition = torch.log(context["transition"].clamp_min(1e-8))

    grid = np.full((height, width), -1, dtype=np.int64)
    grid[rows, cols] = initial
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for _ in range(max(1, int(iterations))):
        neighbour_score = torch.zeros((len(rows), num_classes), device=device)
        for dr, dc in offsets:
            neighbour_rows = rows + dr
            neighbour_cols = cols + dc
            inside = (
                (neighbour_rows >= 0) & (neighbour_rows < height)
                & (neighbour_cols >= 0) & (neighbour_cols < width)
            )
            valid_index = np.nonzero(inside)[0]
            if len(valid_index) == 0:
                continue
            neighbour_labels = grid[neighbour_rows[inside], neighbour_cols[inside]]
            keep = neighbour_labels >= 0
            valid_index = valid_index[keep]
            neighbour_labels = neighbour_labels[keep]
            if len(valid_index) == 0:
                continue
            neighbour_score[valid_index] += log_transition[:, neighbour_labels].t()
        score = log_probability + alpha * bias.unsqueeze(0) + beta * neighbour_score
        refined = score.argmax(dim=1).cpu().numpy()
        grid[rows, cols] = refined

    return np.asarray(grid[rows, cols], dtype=np.int64)
