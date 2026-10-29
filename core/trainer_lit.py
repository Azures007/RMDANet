"""Literature training recipe (lit-recipe branch) — v2.

TAPDA/MSDA-style UDA trainer with the discriminative terms added:

* backbone      : SSSE (3D-2D hybrid) + two residual classifiers
* alignment     : classifier discrepancy (MCD style, adversarial)
* discriminative: source supervised contrastive (SupCon) + target class-top-k
                  contrastive on EMA-teacher pseudo labels
* target pseudo : EMA teacher, confidence ramp 0.75 -> 0.85, weight ramp
* class balance : batch nuclear norm (BNM), configurable per dataset
* optimizer     : SGD momentum 0.9, constant lr (literature setting)
* protocol      : 180 labels/class, no validation, final epoch, no target GT

The original v7.1 pipeline stays untouched on the ``main`` branch.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_lit import (
    EMATeacher,
    ResClassifier,
    SSSE,
    SupConLoss,
    batch_nuclear_norm,
    discrepancy_l1,
)
from trainer import build_loaders, build_parser, load_config_into_args, make_grad_scaler, set_seed


def build_lit_parser():
    parser = build_parser()
    parser.add_argument("--lit_feature_dim", type=int, default=128)
    parser.add_argument("--lit_ema_decay", type=float, default=0.99)
    parser.add_argument("--lit_tau_start", type=float, default=0.75)
    parser.add_argument("--lit_tau_end", type=float, default=0.85)
    parser.add_argument("--lit_pseudo_start", type=int, default=10)
    parser.add_argument("--lit_pseudo_ramp", type=int, default=15)
    parser.add_argument("--lit_pseudo_weight", type=float, default=0.20)
    parser.add_argument("--lit_discrepancy_weight", type=float, default=0.20)
    parser.add_argument("--lit_discrepancy_ramp", type=int, default=10)
    parser.add_argument("--lit_bnm_weight", type=float, default=0.0)
    parser.add_argument("--lit_bnm_start", type=int, default=5)
    parser.add_argument("--lit_bnm_ramp", type=int, default=10)
    parser.add_argument("--lit_supcon_weight", type=float, default=0.5)
    parser.add_argument("--lit_supcon_temperature", type=float, default=0.1)
    parser.add_argument("--lit_target_supcon_weight", type=float, default=0.2)
    parser.add_argument("--lit_target_supcon_start", type=int, default=10)
    parser.add_argument("--lit_target_supcon_ramp", type=int, default=15)
    parser.add_argument("--lit_target_topk", type=int, default=8)
    return parser


def _linear_ramp(epoch: int, start: int, span: int) -> float:
    if epoch < start:
        return 0.0
    if span <= 0:
        return 1.0
    return min(1.0, (epoch - start + 1) / float(span))


def _tau(epoch: int, args) -> float:
    start, end = float(args.lit_tau_start), float(args.lit_tau_end)
    progress = min(1.0, max(0.0, epoch / max(1.0, args.epochs / 2.0)))
    return start + (end - start) * progress


def _select_target_pseudo(teacher_prob: torch.Tensor, tau: float, topk: int):
    confidence, pseudo_label = teacher_prob.max(dim=1)
    keep = confidence >= tau
    if not bool(keep.any()):
        return None, None
    indices = torch.nonzero(keep, as_tuple=False).flatten()
    labels = pseudo_label[indices]
    scores = confidence[indices]
    selected = []
    for class_id in torch.unique(labels).tolist():
        class_mask = labels.eq(int(class_id))
        class_indices = indices[class_mask]
        class_scores = scores[class_mask]
        if class_indices.numel() > topk:
            order = torch.topk(class_scores, k=topk, largest=True).indices
            class_indices = class_indices[order]
        selected.append(class_indices)
    if not selected:
        return None, None
    selected_indices = torch.cat(selected)
    return selected_indices, pseudo_label[selected_indices]


def train(args) -> dict:
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    set_seed(args.seed, bool(getattr(args, "deterministic", False)))

    run_dir = Path(args.save_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[lit-recipe v2] TAPDA/MSDA-style trainer (+ SupCon)")
    print(
        f"[lit-recipe] classes={args.num_classes} patch={args.patch_size} "
        f"batch={args.batch_size} epochs={args.epochs} spc={args.source_samples_per_class} "
        f"norm={getattr(args, 'input_normalization', 'band')} "
        f"opt={getattr(args, 'optimizer_type', 'sgd')} lr={args.lr} "
        f"band_align={args.band_alignment_strength}"
    )
    loaders = build_loaders(args, device)
    num_classes = int(loaders["num_classes"])
    bands = int(loaders["bands"])

    feature_dim = int(args.lit_feature_dim)
    backbone = SSSE(bands, args.patch_size, feature_dim=feature_dim).to(device)
    f1 = ResClassifier(feature_dim, num_classes).to(device)
    f2 = ResClassifier(feature_dim, num_classes).to(device)
    projection = nn.Sequential(
        nn.Linear(feature_dim, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128),
    ).to(device)
    teacher = EMATeacher(backbone, f1, decay=float(args.lit_ema_decay)).to(device)
    supcon = SupConLoss(temperature=float(args.lit_supcon_temperature)).to(device)

    use_sgd = str(getattr(args, "optimizer_type", "sgd")).lower() == "sgd"
    if use_sgd:
        optimizer_g = torch.optim.SGD(
            list(backbone.parameters()) + list(projection.parameters()),
            lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
        )
        optimizer_c = torch.optim.SGD(
            list(f1.parameters()) + list(f2.parameters()),
            lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
        )
    else:
        optimizer_g = torch.optim.AdamW(
            list(backbone.parameters()) + list(projection.parameters()),
            lr=args.lr, weight_decay=args.weight_decay,
        )
        optimizer_c = torch.optim.AdamW(
            list(f1.parameters()) + list(f2.parameters()),
            lr=args.lr, weight_decay=args.weight_decay,
        )
    scaler = make_grad_scaler(bool(args.amp) and device.type == "cuda")

    source_loader = loaders["source_loader"]
    target_loader = loaders["target_loader"]
    class_weights = loaders["class_weights"]
    steps_per_epoch = len(source_loader)
    use_amp = bool(args.amp) and device.type == "cuda"

    target_iterator = iter(target_loader)
    start_time = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        backbone.train()
        f1.train()
        f2.train()
        projection.train()
        eta = float(args.lit_discrepancy_weight) * _linear_ramp(
            epoch, 1, int(args.lit_discrepancy_ramp)
        )
        pseudo_weight = float(args.lit_pseudo_weight) * _linear_ramp(
            epoch, int(args.lit_pseudo_start), int(args.lit_pseudo_ramp)
        )
        bnm_weight = float(args.lit_bnm_weight) * _linear_ramp(
            epoch, int(args.lit_bnm_start), int(args.lit_bnm_ramp)
        )
        supcon_weight = float(args.lit_supcon_weight)
        target_supcon_weight = float(args.lit_target_supcon_weight) * _linear_ramp(
            epoch, int(args.lit_target_supcon_start), int(args.lit_target_supcon_ramp)
        )
        tau = _tau(epoch, args)
        totals = {"ce": 0.0, "dis": 0.0, "pseudo": 0.0, "bnm": 0.0,
                  "supcon": 0.0, "tsupcon": 0.0, "acc": 0.0, "sel": 0.0}
        batches = 0

        for source_x, source_y in source_loader:
            try:
                target_x, _ = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                target_x, _ = next(target_iterator)
            source_x = source_x.to(device, non_blocking=True)
            source_y = source_y.to(device, non_blocking=True)
            target_x = target_x.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_prob = F.softmax(teacher(target_x).float(), dim=1)
                selected_indices, pseudo_labels = _select_target_pseudo(
                    teacher_prob, tau, int(args.lit_target_topk)
                )

            # ---- classifiers: source CE (+ pseudo/BNM/SupCon) - eta * discrepancy
            with torch.autocast(device_type=device.type, enabled=use_amp):
                source_features = backbone(source_x)
                logits_s1 = f1(source_features)
                logits_s2 = f2(source_features)
                ce_s = 0.5 * (
                    F.cross_entropy(logits_s1, source_y, weight=class_weights,
                                    label_smoothing=args.label_smoothing)
                    + F.cross_entropy(logits_s2, source_y, weight=class_weights,
                                      label_smoothing=args.label_smoothing)
                )
                target_features = backbone(target_x)
                logits_t1 = f1(target_features)
                logits_t2 = f2(target_features)
                prob_t1 = F.softmax(logits_t1.float(), dim=1)
                prob_t2 = F.softmax(logits_t2.float(), dim=1)
                discrepancy = discrepancy_l1(prob_t1, prob_t2)
            pseudo_term = torch.zeros((), device=device)
            if pseudo_weight > 0.0 and selected_indices is not None:
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    pseudo_term = 0.5 * (
                        F.cross_entropy(logits_t1[selected_indices], pseudo_labels)
                        + F.cross_entropy(logits_t2[selected_indices], pseudo_labels)
                    )
            bnm_term = torch.zeros((), device=device)
            if bnm_weight > 0.0:
                bnm_term = batch_nuclear_norm(torch.cat([prob_t1, prob_t2], dim=0))
            supcon_source = torch.zeros((), device=device)
            if supcon_weight > 0.0:
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    supcon_source = supcon(projection(source_features.detach()), source_y)

            optimizer_c.zero_grad()
            scaler.scale(
                ce_s - eta * discrepancy
                + pseudo_weight * pseudo_term
                + bnm_weight * bnm_term
                + supcon_weight * supcon_source
            ).backward()
            scaler.unscale_(optimizer_c)
            torch.nn.utils.clip_grad_norm_(
                list(f1.parameters()) + list(f2.parameters()), args.grad_clip
            )
            scaler.step(optimizer_c)
            scaler.update()

            # ---- generator: source CE + discrepancy + pseudo + SupCon (+ target SupCon)
            optimizer_g.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                source_features2 = backbone(source_x)
                ce_s2 = 0.5 * (
                    F.cross_entropy(f1(source_features2), source_y,
                                    weight=class_weights, label_smoothing=args.label_smoothing)
                    + F.cross_entropy(f2(source_features2), source_y,
                                      weight=class_weights, label_smoothing=args.label_smoothing)
                )
                target_features2 = backbone(target_x)
                prob_g1 = F.softmax(f1(target_features2).float(), dim=1)
                prob_g2 = F.softmax(f2(target_features2).float(), dim=1)
                generator_loss = ce_s2 + eta * discrepancy_l1(prob_g1, prob_g2)
                supcon_g = torch.zeros((), device=device)
                if supcon_weight > 0.0:
                    supcon_g = supcon(projection(source_features2), source_y)
                    generator_loss = generator_loss + supcon_weight * supcon_g
                target_supcon_term = torch.zeros((), device=device)
                if target_supcon_weight > 0.0 and selected_indices is not None:
                    target_subset = target_features2[selected_indices]
                    if target_subset.size(0) >= 2:
                        target_supcon_term = supcon(
                            projection(target_subset), pseudo_labels
                        )
                        generator_loss = generator_loss + target_supcon_weight * target_supcon_term
                if pseudo_weight > 0.0 and selected_indices is not None:
                    generator_loss = generator_loss + pseudo_weight * 0.5 * (
                        F.cross_entropy(f1(target_features2)[selected_indices], pseudo_labels)
                        + F.cross_entropy(f2(target_features2)[selected_indices], pseudo_labels)
                    )
                if bnm_weight > 0.0:
                    generator_loss = generator_loss + bnm_weight * batch_nuclear_norm(
                        torch.cat([prob_g1, prob_g2], dim=0)
                    )
            scaler.scale(generator_loss).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(projection.parameters()), args.grad_clip
            )
            scaler.step(optimizer_g)
            scaler.update()

            if pseudo_weight > 0.0:
                teacher.update(backbone, f1)

            with torch.no_grad():
                accuracy = float(
                    (logits_s1.detach().argmax(dim=1) == source_y).float().mean().item()
                )
            totals["ce"] += float(ce_s.detach().item())
            totals["dis"] += float(discrepancy.detach().item())
            totals["pseudo"] += float(pseudo_term.detach().item())
            totals["bnm"] += float(bnm_term.detach().item())
            totals["supcon"] += float(supcon_g.detach().item())
            totals["tsupcon"] += float(target_supcon_term.detach().item())
            totals["acc"] += accuracy
            totals["sel"] += (
                float(selected_indices.numel()) if selected_indices is not None else 0.0
            )
            batches += 1

        if epoch % int(args.log_interval) == 0 or epoch == int(args.epochs):
            elapsed = (time.time() - start_time) / 60.0
            count = max(1, batches)
            print(
                f"lit-epoch {epoch:03d}/{args.epochs} "
                f"ce={totals['ce']/count:.4f} dis={totals['dis']/count:.4f} "
                f"pseudo={totals['pseudo']/count:.4f} bnm={totals['bnm']/count:.4f} "
                f"sup={totals['supcon']/count:.4f} tsup={totals['tsupcon']/count:.4f} "
                f"sel={totals['sel']/count:.1f} src_acc={100.0*totals['acc']/count:.2f} "
                f"tau={tau:.3f} eta={eta:.3f} wp={pseudo_weight:.3f} "
                f"wts={target_supcon_weight:.3f} steps={steps_per_epoch} time={elapsed:.1f}m"
            )

    checkpoint_path = run_dir / "selected_model.pt"
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in backbone.state_dict().items()},
            "f1": {k: v.detach().cpu() for k, v in f1.state_dict().items()},
            "f2": {k: v.detach().cpu() for k, v in f2.state_dict().items()},
            "args": vars(args),
            "selection": {"main_best_epoch": int(args.epochs)},
            "recipe": "lit",
        },
        checkpoint_path,
    )
    print("checkpoint=", checkpoint_path)
    print("Final target benchmark was skipped; target ground truth was never opened.")
    return {"checkpoint": str(checkpoint_path), "recipe": "lit"}


def main(argv=None) -> None:
    parser = build_lit_parser()
    args = parser.parse_args(argv)
    if args.config:
        args = load_config_into_args(args, args.config)
    train(args)


if __name__ == "__main__":
    main()
