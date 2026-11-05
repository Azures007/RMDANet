"""Validate whether the LSCS signals can certify target pseudo-labels.

Uses an existing Houston dump (teacher/student predictions, confidences) and the
target labels *for analysis only*, to answer two questions before training:

1. Is there any signal that separates correct from wrong pseudo-labels?
   (ROC-AUC of teacher confidence / teacher-student agreement)
2. Is the corrective candidate set (teacher == F2 != F1) large enough and is the
   teacher actually right more often than the student there?

Usage:
  python tools/validate_lscs.py --dataset houston \
      --dump-dir /root/autodl-tmp/diag/houston
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="houston")
    ap.add_argument("--dump-dir", required=True)
    args = ap.parse_args()

    labels_path = os.path.join(args.dump_dir, "target_labels.npy")
    if not os.path.exists(labels_path):
        raise SystemExit(f"missing {labels_path}")
    gt = np.load(labels_path).astype(np.int64).ravel()

    for f in sorted(glob.glob(os.path.join(args.dump_dir, "*.npz"))):
        d = np.load(f)
        ids = d["id"].astype(np.int64)
        teacher = d["teacher"].astype(np.int64)
        student = d["student"].astype(np.int64)
        conf = d["conf"].astype(np.float32)

        # deduplicate by target id (keep last)
        pos = {}
        for i, sid in enumerate(ids.tolist()):
            pos[sid] = i
        keep = np.array([pos[s] for s in pos], dtype=np.int64)
        ids, teacher, student, conf = ids[keep], teacher[keep], student[keep], conf[keep]
        y = gt[ids]

        correct = (teacher == y)
        print(f"\n===== {os.path.basename(f)} ({len(ids)} samples) =====")
        print(f"teacher acc = {correct.mean()*100:.2f}%   student acc = {(student == y).mean()*100:.2f}%")

        # 1) signal AUC for predicting teacher correctness
        try:
            from sklearn.metrics import roc_auc_score
            print(f"AUC(confidence)            = {roc_auc_score(correct, conf):.4f}")
            print(f"AUC(teacher-student agree) = {roc_auc_score(correct, (teacher == student).astype(int)):.4f}")
        except Exception as exc:
            print("AUC unavailable:", exc)

        # 2) corrective candidate analysis: teacher == student vs disagreement
        agree = teacher == student
        print(f"\nteacher-student agreement rate = {agree.mean()*100:.2f}%")
        for name, mask in [("agreed", agree), ("disagreed", ~agree)]:
            if mask.sum() == 0:
                continue
            print(f"  [{name}] n={int(mask.sum())}  teacher acc={correct[mask].mean()*100:.2f}%  "
                  f"student acc={(student[mask] == y[mask]).mean()*100:.2f}%")

        # 3) mock 2-vs-1 correction using teacher vs student (F2 unavailable in dump)
        # upper bound of accuracy if we replace student preds on disagreements by teacher preds
        fixed = student.copy()
        fixed[~agree] = teacher[~agree]
        print(f"\n[oracle upper bound] replace student by teacher on disagreements -> "
              f"{(fixed == y).mean()*100:.2f}% (student {(student == y).mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
