"""Patch TAPDA's up2pc.py with our reliability-weighted pseudo-label CE.

TAPDA selects reliable target samples (PATS) and uses them ONLY for contrastive
distillation; there is no explicit pseudo-label classification loss on the
target domain.  This patch adds that missing supervision path:

    L_rel = sum_i w_i * CE(student_logits_i, teacher_label_i) / sum_i w_i
    w_i   = teacher_prob_i * [teacher_prob_i >= min_prob] * class_cap_mask

Control flags (default: disabled, i.e. original TAPDA):
    --reliab_pseudo_ce_weight   (float, default 0.0)
    --reliab_pseudo_min_prob    (float, default 0.95)
    --reliab_pseudo_class_cap   (int,   default 8)

The TAPDA code builds its CLI inside ``build_parser()`` and later calls
``build_parser().parse_args()``, so the flags are injected before the
``return parser`` statement of that function.  Idempotent.

Usage:
    python tools/patch_tapda_reliab.py --tapda-dir /root/autodl-tmp/tapda
"""
from __future__ import annotations

import argparse
import re
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
                pseudo_ce = F.cross_entropy(output_t1, rel_labels.long(), reduction='none')
                reliability_loss = (pseudo_ce * rel_weights).sum() / rel_weights.sum().clamp_min(1.0)
                all_loss = all_loss + args.reliab_pseudo_ce_weight * reliability_loss
"""

FUNC_PATTERN = re.compile(r"^def build_parser\(", re.M)
RETURN_PATTERN = re.compile(r"^([ \t]*)return parser\b", re.M)
LOSS_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)all_loss\s*=\s*loss1\s*\+\s*loss2\s*\+\s*bnm_loss_a\s*\+\s*contrastive_loss",
    re.M,
)


def _insert_args(text: str) -> str:
    func = FUNC_PATTERN.search(text)
    if func is None:
        raise RuntimeError(
            "def build_parser(...) not found; run: grep -n 'def build_parser' up2pc.py"
        )
    body_start = func.end()
    next_top = re.search(r"^(def |class )", text[body_start:], re.M)
    body_end = body_start + (next_top.start() if next_top else len(text))
    body = text[body_start:body_end]
    ret = RETURN_PATTERN.search(body)
    if ret is None:
        raise RuntimeError(
            "return parser not found inside build_parser; run: grep -n 'return parser' up2pc.py"
        )
    position = body_start + ret.start()
    return text[:position] + ARGS_BLOCK + text[position:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapda-dir", default="/root/autodl-tmp/tapda")
    options = parser.parse_args()

    path = Path(options.tapda_dir) / "up2pc.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf8")
    changes = []

    if "reliab_pseudo_ce_weight" in text:
        changes.append("CLI flags already present")
    else:
        text = _insert_args(text)
        changes.append("added CLI flags inside build_parser()")

    if "reliability_loss = (pseudo_ce * rel_weights).sum()" in text:
        changes.append("loss block already present")
    else:
        match = LOSS_PATTERN.search(text)
        if match is None:
            raise RuntimeError(
                "loss anchor 'all_loss = loss1 + loss2 + bnm_loss_a + contrastive_loss' "
                "not found; run: grep -n 'all_loss =' up2pc.py"
            )
        end = match.end()
        text = text[:end] + "\n" + LOSS_BLOCK.lstrip("\n") + text[end:]
        changes.append("added reliability-weighted pseudo CE")

    path.write_text(text, encoding="utf8")
    print("patched:", path)
    for item in changes:
        print(" -", item)

    check = path.read_text(encoding="utf8")
    assert "reliab_pseudo_ce_weight" in check
    assert "reliability_loss" in check
    print("OK: flags + loss inserted; enable with --reliab_pseudo_ce_weight 0.10")


if __name__ == "__main__":
    main()
