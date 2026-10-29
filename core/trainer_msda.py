"""Faithful MSDA recipe trainer (imports the official components directly).

The official implementation lives in an external checkout:
    MSDA_DIR=/root/autodl-tmp/project      (Li-ZK/MSDA-2024)
We import ``HybridSN.SSSE_test``, ``basenet.ResClassifier``,
``modules.masking.Masking_new``, ``modules.mcc.MinimumClassConfusionLoss`` and
``modules.teacher.EMATeacher`` from it, and re-implement the training loop of
``up2pc.py`` inside our repository so that the protocol stays under our control.

Recipe (as in the official code):
  source 180/class, per-band z-score input, patch 9, batch 32, SGD lr 0.1,
  momentum 0.9, weight decay 5e-4, 50 epochs x one pass over the source loader,
  masked image consistency (MIC), MCC loss, EMA teacher pseudo labels,
  two classifiers with adversarial discrepancy (Step A/B/C), no validation.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer import build_loaders, build_parser, load_config_into_args, set_seed


def load_official(msda_dir: str | None = None):
    """Import the official MSDA building blocks from the external checkout."""
    directory = msda_dir or os.environ.get("MSDA_DIR", "/root/autodl-tmp/project")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from HybridSN import SSSE_test  # type: ignore
    from basenet import ResClassifier  # type: ignore
    from modules.masking import Masking_new  # type: ignore
    from modules.mcc import MinimumClassConfusionLoss  # type: ignore
    from modules.teacher import EMATeacher  # type: ignore
    from utils import weights_init  # type: ignore

    return {
        "SSSE_test": SSSE_test,
        "ResClassifier": ResClassifier,
        "weights_init": weights_init,
        "Masking_new": Masking_new,
        "MinimumClassConfusionLoss": MinimumClassConfusionLoss,
        "EMATeacher": EMATeacher,
        "dir": directory,
    }


def build_msda_parser():
    parser = build_parser()
    parser.add_argument("--msda_dir", type=str, default="")
    parser.add_argument("--mask_block_size", type=int, default=32)
    parser.add_argument("--mask_ratio_a", type=float, default=0.7)
    parser.add_argument("--mask_ratio_b", type=float, default=0.7)
    parser.add_argument("--spectral_mask", action="store_true", default=True)
    parser.add_argument("--spatial_mask", action="store_true", default=True)
    parser.add_argument("--spectral_block_size", type=int, default=9)
    parser.add_argument("--spatial_block_size", type=int, default=4)
    parser.add_argument("--mask_color_jitter_s", type=float, default=0.2)
    parser.add_argument("--mask_color_jitter_p", type=float, default=0.2)
    parser.add_argument("--mask_blur", action="store_true", default=True)
    parser.add_argument("--mcc_temperature", type=float, default=9.0)
    parser.add_argument("--teacher_alpha", type=float, default=0.9)
    parser.add_argument("--num_k", type=int, default=3)
    parser.add_argument("--mic_weight", type=float, default=0.5)
    return parser


def _masked_consistency(classifier, backbone, masked_input, pseudo_label, pseudo_prob, weight):
    logits, _ = classifier(backbone(masked_input))
    ce = F.cross_entropy(logits, pseudo_label, reduction="none")
    return (pseudo_prob * ce).mean() * weight


def train(args) -> dict:
    official = load_official(args.msda_dir or None)
    print(f"[msda-recipe] official code: {official['dir']}")
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    set_seed(args.seed, bool(getattr(args, "deterministic", False)))

    run_dir = Path(args.save_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[msda-recipe] classes={args.num_classes} patch={args.patch_size} "
        f"batch={args.batch_size} epochs={args.epochs} spc={args.source_samples_per_class} "
        f"norm={getattr(args, 'input_normalization', 'band')} lr={args.lr}"
    )

    loaders = build_loaders(args, device)
    num_classes = int(loaders["num_classes"])
    bands = int(loaders["bands"])

    backbone = official["SSSE_test"](
        input_channels=bands, patch_size=args.patch_size, n_classes=num_classes
    ).to(device)
    f1 = official["ResClassifier"](num_classes, training=True).to(device)
    f2 = official["ResClassifier"](num_classes, training=True).to(device)
    f1.apply(official["weights_init"])
    f2.apply(official["weights_init"])
    teacher = official["EMATeacher"](
        backbone, f1, alpha=float(args.teacher_alpha), pseudo_label_weight="prob"
    ).to(device)
    mcc_loss = official["MinimumClassConfusionLoss"](temperature=float(args.mcc_temperature))
    masking_a = official["Masking_new"](
        block_size=args.mask_block_size, ratio=args.mask_ratio_a,
        color_jitter_s=args.mask_color_jitter_s, color_jitter_p=args.mask_color_jitter_p,
        blur=bool(args.mask_blur), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        spectral_mask=args.spectral_mask, spatial_mask=args.spatial_mask,
        spectral_block_size=args.spectral_block_size, spatial_block_size=args.spatial_block_size,
    )
    masking_b = official["Masking_new"](
        block_size=args.mask_block_size, ratio=args.mask_ratio_b,
        color_jitter_s=args.mask_color_jitter_s, color_jitter_p=args.mask_color_jitter_p,
        blur=bool(args.mask_blur), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        spectral_mask=args.spectral_mask, spatial_mask=args.spatial_mask,
        spectral_block_size=args.spectral_block_size, spatial_block_size=args.spatial_block_size,
    )

    optimizer_g = torch.optim.SGD(
        backbone.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    optimizer_f = torch.optim.SGD(
        list(f1.parameters()) + list(f2.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )

    source_loader = loaders["source_loader"]
    target_loader = loaders["target_loader"]
    class_weights = loaders["class_weights"]
    num_iter = len(source_loader)
    target_iterator = iter(target_loader)
    start_time = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        backbone.train()
        f1.train()
        f2.train()
        totals = {"ce": 0.0, "mic": 0.0, "mcc": 0.0, "dis": 0.0, "acc": 0.0}
        batches = 0
        for step, (source_x, source_y) in enumerate(source_loader):
            try:
                target_x, _ = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                target_x, _ = next(target_iterator)
            source_x = source_x.to(device, non_blocking=True)
            source_y = source_y.to(device, non_blocking=True)
            target_x = target_x.to(device, non_blocking=True)
            src_bs = source_x.size(0)
            data = torch.cat((source_x, target_x), 0)

            masked_a = masking_a(target_x)
            masked_b = masking_b(target_x)
            teacher.update_weights(backbone, f1, epoch * num_iter + step)
            with torch.no_grad():
                pseudo_label, pseudo_prob = teacher(target_x)

            # ---- Step A: all networks minimise source CE + MIC + MCC
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = backbone(data)
            out1 = f1(output)
            out2 = f2(output)
            logits_s1 = out1[0][:src_bs]
            logits_s2 = out2[0][:src_bs]
            prob_t1 = F.softmax(out1[0][src_bs:].float(), dim=1)
            prob_t2 = F.softmax(out2[0][src_bs:].float(), dim=1)
            loss1 = F.cross_entropy(logits_s1, source_y, weight=class_weights)
            loss2 = F.cross_entropy(logits_s2, source_y, weight=class_weights)
            mic_a = _masked_consistency(
                f1, backbone, masked_a, pseudo_label, pseudo_prob, float(args.mic_weight)
            )
            mcc_a = mcc_loss(prob_t1) + mcc_loss(prob_t2)
            (loss1 + loss2 + mic_a + mcc_a).backward()
            optimizer_g.step()
            optimizer_f.step()

            # ---- Step B: classifiers maximise the discrepancy (adversarial)
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = backbone(data)
            out1 = f1(output)
            out2 = f2(output)
            logits_s1 = out1[0][:src_bs]
            logits_s2 = out2[0][:src_bs]
            prob_t1 = F.softmax(out1[0][src_bs:].float(), dim=1)
            prob_t2 = F.softmax(out2[0][src_bs:].float(), dim=1)
            loss1 = F.cross_entropy(logits_s1, source_y, weight=class_weights)
            loss2 = F.cross_entropy(logits_s2, source_y, weight=class_weights)
            mic_b = _masked_consistency(
                f1, backbone, masked_b, pseudo_label, pseudo_prob, float(args.mic_weight)
            )
            mcc_b = mcc_loss(prob_t1) + mcc_loss(prob_t2)
            discrepancy = torch.mean(torch.abs(prob_t1 - prob_t2))
            classifier_loss = (
                loss1 + loss2 - discrepancy + mic_b + mcc_b
            )
            classifier_loss.backward()
            optimizer_f.step()

            # ---- Step C: generator minimises the discrepancy (num_k times)
            generator_loss_value = torch.zeros((), device=device)
            for _ in range(int(args.num_k)):
                optimizer_g.zero_grad()
                output = backbone(data)
                out1 = f1(output)
                out2 = f2(output)
                prob_t1 = F.softmax(out1[0][src_bs:].float(), dim=1)
                prob_t2 = F.softmax(out2[0][src_bs:].float(), dim=1)
                discrepancy = torch.mean(torch.abs(prob_t1 - prob_t2))
                generator_loss = (
                    discrepancy + mcc_loss(prob_t1) + mcc_loss(prob_t2)
                )
                generator_loss.backward()
                optimizer_g.step()
                generator_loss_value = generator_loss.detach()

            with torch.no_grad():
                accuracy = float((logits_s1.detach().argmax(dim=1) == source_y).float().mean())
            totals["ce"] += float((loss1 + loss2).detach().item())
            totals["mic"] += float(mic_a.detach().item())
            totals["mcc"] += float(mcc_a.detach().item())
            totals["dis"] += float(discrepancy.detach().item())
            totals["acc"] += accuracy
            batches += 1

        if epoch % int(args.log_interval) == 0 or epoch == int(args.epochs):
            elapsed = (time.time() - start_time) / 60.0
            count = max(1, batches)
            print(
                f"msda-epoch {epoch:03d}/{args.epochs} "
                f"ce={totals['ce']/count:.4f} mic={totals['mic']/count:.4f} "
                f"mcc={totals['mcc']/count:.4f} dis={totals['dis']/count:.4f} "
                f"src_acc={100.0*totals['acc']/count:.2f} steps={num_iter} time={elapsed:.1f}m"
            )

    checkpoint_path = run_dir / "selected_model.pt"
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in backbone.state_dict().items()},
            "f1": {k: v.detach().cpu() for k, v in f1.state_dict().items()},
            "f2": {k: v.detach().cpu() for k, v in f2.state_dict().items()},
            "args": vars(args),
            "selection": {"main_best_epoch": int(args.epochs)},
            "recipe": "msda",
        },
        checkpoint_path,
    )
    print("checkpoint=", checkpoint_path)
    print("Final target benchmark was skipped; target ground truth was never opened.")
    return {"checkpoint": str(checkpoint_path), "recipe": "msda"}


def main(argv=None) -> None:
    parser = build_msda_parser()
    args = parser.parse_args(argv)
    if args.config:
        args = load_config_into_args(args, args.config)
    train(args)


if __name__ == "__main__":
    main()
