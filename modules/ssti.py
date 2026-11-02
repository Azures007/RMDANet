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


def source_only_copy(G, F1, source_loader, num_classes, device, epochs=20, lr=0.01):
    """Deep-copy the encoder/classifier and retrain them on the source labels only.

    This uses exactly the labelled source budget already spent by the running
    experiment (e.g. 180 labels/class), so the predicted structure map is not
    contaminated by the target-domain adaptation.
    """
    import copy as _copy
    G_copy = _copy.deepcopy(G).to(device)
    F1_copy = _copy.deepcopy(F1).to(device)
    parameters = list(G_copy.parameters()) + list(F1_copy.parameters())
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    G_copy.train()
    F1_copy.train()
    # The caller runs inside torch.no_grad(); re-enable grad for this local training.
    with torch.enable_grad():
        for _ in range(max(1, int(epochs))):
            for data, target in source_loader:
                data = data.to(device)
                target = target.to(device).long()
                features, _ = G_copy(data)
                output = F1_copy(features)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    G_copy.eval()
    F1_copy.eval()
    return G_copy, F1_copy


def predict_source_map(G, F1, source_image, halfwidth, num_classes, device,
                       max_rows_per_batch=1):
    """Predict every pixel of the source image with the trained source model.

    Only the source image (no labels) and the trained model are used, so this is
    compatible with the limited-label protocol (e.g. 180 labels/class).
    """
    import torch as _torch

    source_image = np.asarray(source_image, dtype=np.float32)
    height, width, _ = source_image.shape
    patch_size = 2 * int(halfwidth) + 1
    padded = np.pad(source_image, ((halfwidth, halfwidth), (halfwidth, halfwidth), (0, 0)),
                    mode="constant")
    label_map = np.zeros((height, width), dtype=np.int64)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    g_was_training, f_was_training = G.training, F1.training
    G.eval()
    F1.eval()
    rows_per_batch = max(1, min(int(max_rows_per_batch), max(1, 512 // max(1, width))))
    try:
        with _torch.no_grad():
            for row_start in range(0, height, rows_per_batch):
                row_end = min(row_start + rows_per_batch, height)
                patches = []
                for row in range(row_start, row_end):
                    block = padded[row:row + patch_size, :, :]
                    windows = np.lib.stride_tricks.sliding_window_view(
                        block, (patch_size, patch_size), axis=(0, 1)
                    )
                    row_patches = np.transpose(windows, (1, 3, 4, 0, 2)).reshape(
                        width, patch_size, patch_size, source_image.shape[2]
                    )
                    patches.append(row_patches)
                batch = np.concatenate(patches, axis=0)
                tensor = _torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device)
                features, _ = G(tensor)
                output = F1(features)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                probability = _torch.softmax(output.float(), dim=1) if not isinstance(output, (tuple, list)) else None
                if probability is None:
                    probability = _torch.softmax(output.float(), dim=1)
                confidence = probability.max(dim=1).values.cpu().numpy().reshape(row_end - row_start, width)
                prediction = output.argmax(dim=1).cpu().numpy().reshape(row_end - row_start, width)
                label_map[row_start:row_end] = prediction
                confidence_map[row_start:row_end] = confidence
    finally:
        G.train(g_was_training)
        F1.train(f_was_training)
    return label_map, confidence_map


def build_transition_prior(label_map, num_classes, smoothing=0.1, valid_mask=None):
    label_map = np.asarray(label_map).astype(np.int64)
    height, width = label_map.shape
    padded = np.pad(label_map, 1, mode="constant", constant_values=0)
    if valid_mask is not None:
        padded_mask = np.pad(np.asarray(valid_mask, dtype=np.uint8), 1, mode="constant")
    else:
        padded_mask = None
    counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbour = padded[1 + dr:1 + dr + height, 1 + dc:1 + dc + width]
            valid = (label_map > 0) & (neighbour > 0)
            if padded_mask is not None:
                neighbour_mask = padded_mask[1 + dr:1 + dr + height, 1 + dc:1 + dc + width]
                valid = valid & (padded_mask[1 + dr:1 + dr + height, 1 + dc:1 + dc + width] > 0)
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
                  source_loader=None, G=None, F1=None, smoothing=0.1, coords=None,
                  structure="gt", source_image=None, halfwidth=0, pred_confidence=0.9,
                  source_only_epochs=20, source_only_lr=0.01):
    if structure == "predicted":
        if source_image is None:
            raise ValueError("structure='predicted' requires the source image")
    if structure == "potts":
        # Label-free isotropic prior: no source statistics at all.
        identity = np.eye(num_classes, dtype=np.float64)
        off = (1.0 - identity) / max(1, num_classes - 1)
        transition = (1.0 - 0.1) * identity + 0.1 * off
        structural_map = None
    elif structure == "predicted":
        structural_map = None
    else:
        structural_map = np.asarray(source_label_map)
    if structure != "potts":
        transition = None
        if structural_map is not None:
            transition = build_transition_prior(structural_map, num_classes, smoothing)
    source_prior = None
    if structural_map is not None:
        source_map = np.asarray(structural_map).astype(np.int64)
        histogram = np.array(
            [(source_map == c).sum() for c in range(1, num_classes + 1)], dtype=np.float64
        )
        source_prior = (histogram + 1.0) / (histogram.sum() + num_classes)
    if coords is None:
        coords = np.nonzero(np.asarray(target_label_map) > 0)
    else:
        coords = (np.asarray(coords[0], dtype=np.int64), np.asarray(coords[1], dtype=np.int64))
    return {
        "transition": (torch.from_numpy(transition).float().to(device)
                       if transition is not None else None),
        "confusion": None,
        "source_prior": (torch.from_numpy(source_prior).float()
                         if source_prior is not None else None),
        "structure": structure,
        "source_image": source_image,
        "halfwidth": int(halfwidth),
        "source_label_map": (None if structure == "predicted"
                             else np.asarray(source_label_map)),
        "smoothing": float(smoothing),
        "pred_confidence": float(pred_confidence),
        "source_only_epochs": int(source_only_epochs),
        "source_only_lr": float(source_only_lr),
        "coords": coords,
        "shape": np.asarray(target_label_map).shape,
        "target_map": np.asarray(target_label_map).astype(np.int64),
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
    if context.get("structure") == "predicted" and context["transition"] is None:
        G_structure, F1_structure = source_only_copy(
            context["G"], context["F1"], context["source_loader"],
            context["num_classes"], context["device"],
            epochs=context.get("source_only_epochs", 20),
            lr=context.get("source_only_lr", 0.01),
        )
        structural_map, confidence_map = predict_source_map(
            G_structure, F1_structure, context["source_image"],
            context["halfwidth"], context["num_classes"], context["device"],
        )
        mask = confidence_map >= context.get("pred_confidence", 0.9)
        if int(mask.sum()) < 50:
            mask = np.ones_like(mask, dtype=bool)
        transition = build_transition_prior(
            structural_map, context["num_classes"], context["smoothing"],
            valid_mask=mask,
        )
        context["transition"] = torch.from_numpy(transition).float().to(context["device"])
        # The labelled source budget is uniform (e.g. 180 labels/class), so the
        # source class prior is taken as uniform instead of a GT histogram.
        context["source_prior"] = torch.full(
            (context["num_classes"],), 1.0 / context["num_classes"]
        )
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
    if len(initial) != len(rows):
        raise ValueError(
            f"SSTI: {len(initial)} predictions but {len(rows)} coordinates"
        )
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
