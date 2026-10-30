"""Patch TAPDA's up2pc.py with our reliability-weighted pseudo-label CE.

TAPDA selects reliable target samples (PATS) and uses them ONLY for contrastive
distillation; there is no explicit pseudo-label classification loss on the
target domain.  This patch adds exactly that missing supervision path:

    L_rel = sum_i w_i * CE(student_logits_i, teacher_label_i) / sum_i w_i
    w_i   = teacher_prob_i * [teacher_prob_i >= min_prob] * class_cap_mask

Control flags (default: disabled, i.e. original TAPDA):
    --reliab_pseudo_ce_weight   (float, default 0.0)
    --reliab_pseudo_min_prob    (float, default 0.95)
    --reliab_pseudo_class_cap   (int,   default 8)

Usage:
    python tools/patch_tapda_reliab.py --tapda-dir /root/autodl-tmp/tapda
"""
from __future__ import annotations

import argparse
from pathlib import Path

ARGS_BLOCK = """
# ---- reliability-weighted pseudo-label supervision (our add-on) ----
parser.add_argument('--reliab_pseudo_ce_weight', default=0.0, type=float,
                    help='weight of reliability-weighted pseudo-label CE (0 = original TAPDA)')
parser.add_argument('--reliab_pseudo_min_prob', default=0.95, type=float,
                    help='teacher confidence threshold for pseudo-label supervision')
parser.add_argument('--reliab_pseudo_class_cap', default=8, type=int,
                    help='max pseudo-labelled samples per class per batch (0 = no cap)')
"""

LOSS_BLOCK = """
        # ---- our contribution: explicit reliability-weighted pseudo-label CE ----
        if getattr(args, 'reliab_pseudo_ce_weight', 0.0) > 0.0:
            with torch.no_grad():
                rel_labels = F1_pseudo_label_t.detach()
                rel_weights = F1_pseudo_prob_t.detach() * (
                    F1_pseudo_prob_t.detach() >= args.reliab_pseudo_min_prob).float()
                if args.reliab_pseudo_class_cap > 0:
                    for class_idx in range(num_classes):
                        class_mask = (rel_labels == class_idx) & (rel_weights > 0)
                        if int(class_mask.sum()) > args.reliab_pseudo_class_cap:
                            class_weights = rel_weights[class_mask]
                            threshold = torch.topk(
                                class_weights, args.reliab_pseudo_class_cap).values.min()
                            drop = class_mask & (rel_weights < threshold)
                            rel_weights = torch.where(
                                drop, torch.zeros_like(rel_weights), rel_weights)
            if float(rel_weights.sum()) > 0.0:
                pseudo_ce = F.cross_entropy(
                    output_t1, rel_labels.long(), reduction='none')
                reliability_loss = (pseudo_ce * rel_weights).sum() / rel_weights.sum().clamp_min(1.0)
                all_loss = all_loss + args.reliab_pseudo_ce_weight * reliability_loss
"""

ARGS_ANCHOR = "args = parser.parse_args()"
LOSS_ANCHOR = "        all_loss = loss1 + loss2 + bnm_loss_a + contrastive_loss"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapda-dir", default="/root/autodl-tmp/tapda")
    options = parser.parse_args()

    path = Path(options.tapda_dir) / "up2pc.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf8")
    changes = []

    if "reliab_pseudo_ce_weight" not in text:
        if ARGS_ANCHOR not in text:
            raise RuntimeError("argparse anchor not found")
        text = text.replace(ARGS_ANCHOR, ARGS_BLOCK + "\n" + ARGS_ANCHOR, 1)
        changes.append("added CLI flags")
    else:
        changes.append("CLI flags already present")

    if "reliability_loss = (pseudo_ce * rel_weights).sum()" not in text:
        if LOSS_ANCHOR not in text:
            raise RuntimeError("loss anchor not found")
        text = text.replace(LOSS_ANCHOR, LOSS_ANCHOR + "\n" + LOSS_BLOCK, 1)
        changes.append("added reliability-weighted pseudo CE")
    else:
        changes.append("loss block already present")

    path.write_text(text, encoding="utf8")
    print("patched:", path)
    for item in changes:
        print(" -", item)

    # sanity check
    check = path.read_text(encoding="utf8")
    assert "reliab_pseudo_ce_weight" in check
    assert "reliability_loss" in check
    print("OK: flags + loss inserted; run with --reliab_pseudo_ce_weight 0.10 to enable.")


if __name__ == "__main__":
    main()
