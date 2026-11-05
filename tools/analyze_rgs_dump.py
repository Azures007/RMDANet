"""Offline analysis of RGS diagnostics dumps.

Reads the .npz files written by --m1_dump and reports, per epoch:
  * N_pre / N_post (PATS-selected) / N_excluded and exclusion ratio
  * teacher pseudo-label and student accuracy on PATS vs excluded samples
  * teacher confidence statistics per group
  * per-class coverage of PATS / excluded / RGS
  * |RGS ∩ excluded| (expected 0 with the current implementation)

Target labels are used ONLY here, for analysis, never during training.

Usage:
  python tools/analyze_rgs_dump.py --dataset pavia --dump-dir /path/to/dump \
      --data-root /root/autodl-tmp/tapda_repro/datasets
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util.utils as utils  # noqa: E402

CONFIGS = {
    "pavia": dict(
        hw=4, num_classes=7,
        src=("Pavia/paviaU.mat", "Pavia/paviaU_gt_7.mat"),
        tar=("Pavia/pavia.mat", "Pavia/pavia_gt_7.mat"),
        loader="load_data_pavia",
    ),
    "houston": dict(
        hw=3, num_classes=7,
        src=("Houston/Houston13.mat", "Houston/Houston13_7gt.mat"),
        tar=("Houston/Houston18.mat", "Houston/Houston18_7gt.mat"),
        loader="load_data_houston",
    ),
    "yrd": dict(
        hw=1, num_classes=7,
        src=("YRD/yrd_nc12.mat", "YRD/yrd_nc12_7gt.mat"),
        tar=("YRD/yrd_nc13.mat", "YRD/yrd_nc13_7gt.mat"),
        loader="load_data_YRD",
    ),
    "hyrank": dict(
        hw=1, num_classes=12,
        src=("HyRANK/Dioni.mat", "HyRANK/Dioni_gt_out68.mat"),
        tar=("HyRANK/Loukia.mat", "HyRANK/Loukia_gt_out68.mat"),
        loader="load_data_hyrank",
    ),
}


def load_target_labels(dataset: str, data_root: str) -> np.ndarray:
    cfg = CONFIGS[dataset]
    fn = getattr(utils, cfg["loader"])
    _dt, lt = fn(os.path.join(data_root, cfg["tar"][0]), os.path.join(data_root, cfg["tar"][1]))
    out = utils.get_all_data(_dt, lt, cfg["hw"])
    test_y = out[2]
    return np.asarray(test_y).astype(np.int64).ravel()


def summarize(npz_path: str, gt: np.ndarray, num_classes: int) -> str:
    data = np.load(npz_path)
    ids = data["id"].astype(np.int64)
    teacher = data["teacher"].astype(np.int64)
    student = data["student"].astype(np.int64)
    conf = data["conf"].astype(np.float32)
    in_pats = data["in_pats"].astype(bool)
    in_rgs = data["in_rgs"].astype(bool)

    # keep the last occurrence of each target id (a sample may appear in several batches)
    order = {}
    for pos, sid in enumerate(ids.tolist()):
        order[sid] = pos
    keep = np.array([order[s] for s in order], dtype=np.int64)

    ids, teacher, student, conf = ids[keep], teacher[keep], student[keep], conf[keep]
    in_pats, in_rgs = in_pats[keep], in_rgs[keep]
    labels = gt[ids]

    n_pre = len(ids)
    n_post = int(in_pats.sum())
    n_excl = n_pre - n_post
    ratio = n_excl / max(1, n_pre)

    def acc(mask_pred, mask_sel):
        if mask_sel.sum() == 0:
            return float("nan")
        return float((mask_pred[mask_sel] == labels[mask_sel]).mean() * 100.0)

    lines = []
    lines.append(f"# RGS diagnostics: {os.path.basename(npz_path)}")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| N_pre（目标样本总数） | {n_pre} |")
    lines.append(f"| N_post（PATS 选中） | {n_post} |")
    lines.append(f"| N_excluded（被排除） | {n_excl} |")
    lines.append(f"| 排除比例 R_excluded | {ratio*100:.2f}% |")
    lines.append(f"| 被排除样本数占比 | {n_excl / max(1,n_pre)*100:.2f}% |")
    lines.append("")
    lines.append("| 组别 | 样本数 | 教师伪标签准确率 | 学生准确率 | 平均置信度 |")
    lines.append("|---|---|---|---|---|")
    groups = {
        "全部": np.ones(n_pre, dtype=bool),
        "PATS 保留": in_pats,
        "PATS 排除": ~in_pats,
        "RGS 监督": in_rgs,
        "RGS ∩ 排除": in_rgs & ~in_pats,
    }
    for gname, gmask in groups.items():
        gmean = float(conf[gmask].mean()) if gmask.sum() else float("nan")
        lines.append(
            f"| {gname} | {int(gmask.sum())} | {acc(teacher, gmask):.2f} | "
            f"{acc(student, gmask):.2f} | {gmean:.4f} |"
        )

    lines.append("")
    lines.append("| 类别 | 总数 | PATS 数 | 排除数 | 排除样本教师 acc | RGS 数 |")
    lines.append("|---|---|---|---|---|---|")
    for c in range(num_classes):
        cm = labels == c
        if cm.sum() == 0:
            continue
        ce = cm & ~in_pats
        lines.append(
            f"| {c} | {int(cm.sum())} | {int((cm & in_pats).sum())} | {int(ce.sum())} | "
            f"{acc(teacher, ce):.2f} | {int((cm & in_rgs).sum())} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"
    )
    saved = os.path.join(args.dump_dir, "target_labels.npy")
    if os.path.exists(saved):
        gt = np.load(saved).astype(np.int64).ravel()
        print(f"[analyze] labels loaded from {saved} (aligned with the training order)")
    else:
        print("[analyze] WARNING: target_labels.npy not found, recomputing labels "
              "(order may not match the training permutation)")
        gt = load_target_labels(args.dataset, data_root)
    files = sorted(glob.glob(os.path.join(args.dump_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz found under {args.dump_dir}")

    reports = [summarize(f, gt, CONFIGS[args.dataset]["num_classes"]) for f in files]
    text = "\n\n".join(reports)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
