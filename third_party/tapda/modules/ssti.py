"""Source-Structured Transductive Inference (SSTI). See docs for the design."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _split_logits(output):
    """Models in this codebase return either a tensor or a (logits, features) tuple."""
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.dim() == 1:
        output = output.unsqueeze(0)
    return output


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
            logits = _split_logits(F1(features))
            np.add.at(
                confusion,
                (target.cpu().numpy(), logits.argmax(dim=1).cpu().numpy()),
                1.0,
            )
    return confusion


def build_context(source_label_map, target_label_map, num_classes, device,
                  source_loader=None, G=None, F1=None, smoothing=0.1):
    transition = build_transition_prior(source_label_map, num_classes, smoothing)
    source_map = np.asarray(source_label_map).astype(np.int64)
    histogram = np.array(
        [(source_map == c).sum() for c in range(1, num_classes + 1)], dtype=np.float64
    )
    source_prior = (histogram + 1.0) / (histogram.sum() + num_classes)
    coords = np.nonzero(np.asarray(target_label_map) > 0)
    return {
        "transition": torch.from_numpy(transition).float().to(device),
        "confusion": None,
        "source_prior": torch.from_numpy(source_prior).float(),
        "coords": coords,
        "shape": np.asarray(target_label_map).shape,
        "num_classes": num_classes,
        "source_loader": source_loader,
        "G": G,
        "F1": F1,
        "device": device,
    }


def ensure_ready(context):
    """Compute the source confusion matrix lazily, i.e. after the model is trained."""
    if context["confusion"] is not None:
        return context
    confusion = source_confusion(
        context["G"], context["F1"], context["source_loader"],
        context["num_classes"], context["device"],
    )
    row_sum = confusion.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    context["confusion"] = torch.from_numpy(confusion / row_sum).float()
    return context


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
    return torch.from_numpy(bias).float().to(context["device"])


def refine_labels(logits, context, labels, alpha=0.1, beta=0.1, iterations=3,
                  temperature=1.0, confidence_threshold=0.95, device=None):
    context = ensure_ready(context)
    device = device if device is not None else context["device"]
    num_classes = context["num_classes"]
    rows, cols = context["coords"]
    height, width = context["shape"]
    log_probability = F.log_softmax(logits.to(device) / float(temperature), dim=1)

    initial = log_probability.argmax(dim=1).cpu().numpy()
    maximum = log_probability.max(dim=1).values
    editable = (maximum < float(confidence_threshold)).cpu().numpy()
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
        # Only the uncertain pixels may change; confident predictions are frozen.
        refined = np.where(editable, refined, initial)
        grid[rows, cols] = refined

    return np.asarray(grid[rows, cols], dtype=np.int64)
