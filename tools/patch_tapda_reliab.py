"""Patch TAPDA's up2pc.py with our reliability-weighted pseudo-label CE.

TAPDA selects reliable target samples (PATS) and uses them ONLY for contrastive
distillation; there is no explicit pseudo-label classification loss on the
target domain.  This patch adds that missing supervision path:

    L_rel = sum_i w_i * CE(student_logits_i, teacher_label_i) / sum_i w_i
    w_i   = teacher_prob_i * [teacher_prob_i >= min_prob] * class_cap_mask

with a warmup schedule to reduce trajectory variance:

    effective_weight(ep) = weight * clamp((ep - start) / max(ramp, 1), 0, 1)

Control flags (default: disabled, i.e. original TAPDA):
    --reliab_pseudo_ce_weight    (float, default 0.0)
    --reliab_pseudo_min_prob     (float, default 0.95)
    --reliab_pseudo_class_cap    (int,   default 8)
    --reliab_pseudo_start_epoch  (int,   default 10)
    --reliab_pseudo_ramp_epochs  (int,   default 10)

The TAPDA code builds its CLI inside ``build_parser()``, so the flags are
injected before the ``return parser`` statement of that function.  The loss is
inserted after ``all_loss = loss1 + loss2 + bnm_loss_a + contrastive_loss``.
Idempotent and upgrade-safe (an older block without warmup is replaced).

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
    parser.add_argument('--reliab_pseudo_start_epoch', default=10, type=int,
                        help='epoch at which the pseudo-label CE starts (warmup)')
    parser.add_argument('--reliab_pseudo_ramp_epochs', default=10, type=int,
                        help='epochs over which the pseudo-label CE ramps up')
"""

LOSS_BLOCK = """
        # ---- our contribution: explicit reliability-weighted pseudo-label CE ----
        if getattr(args, 'reliab_pseudo_ce_weight', 0.0) > 0.0:
            _rel_start = float(getattr(args, 'reliab_pseudo_start_epoch', 10))
            _rel_ramp = max(1.0, float(getattr(args, 'reliab_pseudo_ramp_epochs', 10)))
            _rel_scale = min(1.0, max(0.0, (float(ep) - _rel_start) / _rel_ramp))
            if _rel_scale > 0.0:
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
                    all_loss = all_loss + args.reliab_pseudo_ce_weight * _rel_scale * reliability_loss
"""

FUNC_PATTERN = re.compile(r"^def build_parser\(", re.M)
RETURN_PATTERN = re.compile(r"^([ \t]*)return parser\b", re.M)
LOSS_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)all_loss\s*=\s*loss1\s*\+\s*loss2\s*\+\s*bnm_loss_a\s*\+\s*contrastive_loss",
    re.M,
)
OLD_BLOCK_PATTERN = re.compile(
    r"\n[ \t]*# ---- our contribution: explicit reliability-weighted pseudo-label CE ----.*?"
    r"all_loss = all_loss \+ args\.reliab_pseudo_ce_weight \* reliability_loss\n",
    re.S,
)


def _insert_args(text: str) -> str:
    func = FUNC_PATTERN.search(text)
    if func is None:
        raise RuntimeError("def build_parser(...) not found; grep -n 'def build_parser' up2pc.py")
    body_start = func.end()
    next_top = re.search(r"^(def |class )", text[body_start:], re.M)
    body_end = body_start + (next_top.start() if next_top else len(text))
    ret = RETURN_PATTERN.search(text[body_start:body_end])
    if ret is None:
        raise RuntimeError("return parser not found inside build_parser")
    position = body_start + ret.start()
    return text[:position] + ARGS_BLOCK + text[position:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapda-dir", default="/root/autodl-tmp/tapda")
    parser.add_argument("--script", default="up2pc.py",
                        help="entry script to patch (up2pc.py / houston13218.py / ...)")
    options = parser.parse_args()

    path = Path(options.tapda_dir) / options.script
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf8")
    changes = []

    # ---- CLI flags ---------------------------------------------------------
    if "reliab_pseudo_start_epoch" in text:
        changes.append("CLI flags already up to date")
    else:
        if "reliab_pseudo_ce_weight" in text:
            text = re.sub(
                r"[ \t]*# ---- reliability-weighted pseudo-label supervision \(our add-on\) ----.*?"
                r"parser\.add_argument\('--reliab_pseudo_class_cap'[^\n]*\n[^\n]*\n",
                ARGS_BLOCK.lstrip("\n"),
                text,
                count=1,
                flags=re.S,
            )
            changes.append("upgraded CLI flags (warmup args)")
        else:
            text = _insert_args(text)
            changes.append("added CLI flags inside build_parser()")

    # ---- loss block (replace old version, then insert if missing) ----------
    if "reliab_pseudo_start_epoch" in text and "_rel_scale" in text:
        changes.append("loss block already up to date")
    else:
        old = OLD_BLOCK_PATTERN.search(text)
        if old is not None:
            text = text[: old.start()] + LOSS_BLOCK + text[old.end():]
            changes.append("replaced older loss block with warmup version")
        else:
            match = LOSS_PATTERN.search(text)
            if match is None:
                raise RuntimeError(
                    "loss anchor 'all_loss = loss1 + loss2 + bnm_loss_a + contrastive_loss' "
                    "not found; grep -n 'all_loss =' up2pc.py"
                )
            end = match.end()
            text = text[:end] + "\n" + LOSS_BLOCK.lstrip("\n") + text[end:]
            changes.append("added reliability-weighted pseudo CE (with warmup)")

    path.write_text(text, encoding="utf8")
    print("patched:", path)
    for item in changes:
        print(" -", item)

    check = path.read_text(encoding="utf8")
    for token in ("reliab_pseudo_ce_weight", "reliability_loss", "_rel_scale"):
        assert token in check, token
    print("OK: warmup version installed; enable with --reliab_pseudo_ce_weight 0.05")


if __name__ == "__main__":
    main()
