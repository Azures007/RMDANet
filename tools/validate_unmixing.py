"""Offline validation of the spectral-mixture (unmixing) prior.

Question
--------
Can a physics-based reliability signal (unmixing residual / abundance purity),
computed WITHOUT target labels, separate correct from wrong pseudo-labels, and
is there a theoretical accuracy gain from trusting only "pure" target pixels?

Usage
-----
    python tools/validate_unmixing.py --dataset houston \
        --data-root /root/autodl-tmp/tapda_repro/datasets [--max-pixels 20000]

Target labels are used for analysis only (metrics), never for endmember
estimation (endmembers come from the source scene, which has labels).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util.utils as utils  # noqa: E402

CONFIGS = {
    "houston": dict(
        loader="load_data_houston", num_classes=7,
        src=("Houston/Houston13.mat", "Houston/Houston13_7gt.mat"),
        tar=("Houston/Houston18.mat", "Houston/Houston18_7gt.mat"),
    ),
    "pavia": dict(
        loader="load_data_pavia", num_classes=7,
        src=("Pavia/paviaU.mat", "Pavia/paviaU_gt_7.mat"),
        tar=("Pavia/pavia.mat", "Pavia/pavia_gt_7.mat"),
    ),
    "yrd": dict(
        loader="load_data_YRD", num_classes=7,
        src=("YRD/yrd_nc12.mat", "YRD/yrd_nc12_7gt.mat"),
        tar=("YRD/yrd_nc13.mat", "YRD/yrd_nc13_7gt.mat"),
    ),
}


def spectra_of(cube):
    h, w, b = cube.shape
    return cube.reshape(-1, b).astype(np.float64), h, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="houston", choices=sorted(CONFIGS))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--max-pixels", type=int, default=20000)
    args = ap.parse_args()

    cfg = CONFIGS[args.dataset]
    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"
    )
    fn = getattr(utils, cfg["loader"])

    src_cube, src_gt = fn(os.path.join(root, cfg["src"][0]), os.path.join(root, cfg["src"][1]))
    tar_cube, tar_gt = fn(os.path.join(root, cfg["tar"][0]), os.path.join(root, cfg["tar"][1]))

    src_cube = np.asarray(src_cube, dtype=np.float64)
    tar_cube = np.asarray(tar_cube, dtype=np.float64)
    src_gt = np.asarray(src_gt).astype(np.int64)
    tar_gt = np.asarray(tar_gt).astype(np.int64)

    Xs, hs, ws = spectra_of(src_cube)
    Xt, ht, wt = spectra_of(tar_cube)
    ys = src_gt.reshape(-1)
    yt = tar_gt.reshape(-1)
    C = cfg["num_classes"]

    # ---- endmembers from SOURCE (labels available) ----
    endmembers = np.zeros((Xs.shape[1], C))
    for c in range(C):
        mask = ys == (c + 1)
        if mask.sum() == 0:
            raise SystemExit(f"source has no class {c + 1}")
        endmembers[:, c] = np.median(Xs[mask], axis=0)

    # ---- target labeled pixels (GT used for metrics only) ----
    idx = np.nonzero(yt > 0)[0]
    rng = np.random.default_rng(0)
    if args.max_pixels and len(idx) > args.max_pixels:
        idx = rng.choice(idx, size=args.max_pixels, replace=False)
    X = Xt[idx]
    y = yt[idx] - 1
    print(f"[unmix] target pixels: {len(idx)}  bands: {X.shape[1]}  classes: {C}")

    from scipy.optimize import nnls

    residual = np.zeros(len(idx))
    purity = np.zeros(len(idx))
    pred = np.zeros(len(idx), dtype=np.int64)
    for i in range(len(idx)):
        x = X[i]
        a, r = nnls(endmembers, x)
        s = a.sum()
        residual[i] = r / max(1e-8, np.linalg.norm(x))
        purity[i] = a.max() / max(1e-8, s)
        pred[i] = int(np.argmax(a))

    correct = pred == y
    acc = correct.mean() * 100

    def auc(score, label):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(label, score)

    print("\n===== 解混先验有效性 =====")
    print(f"解混分类准确率（仅丰度 argmax）: {acc:.2f}%")
    print(f"AUC(残差 -> 预测正确)           : {auc(residual, correct):.4f}")
    print(f"AUC(丰度纯度 -> 预测正确)       : {auc(purity, correct):.4f}")

    qs = np.quantile(residual, [0.0, 0.25, 0.5, 0.75, 1.0])
    print("\n| 残差分位 | 像素数 | 解混分类准确率 |")
    print("|---|---|---|")
    for k in range(4):
        lo, hi = qs[k], qs[k + 1]
        mask = (residual >= lo) & (residual <= hi)
        print(f"| Q{k+1} [{lo:.4f},{hi:.4f}] | {int(mask.sum())} | {correct[mask].mean()*100:.2f}% |")

    pure = purity >= 0.9
    mixed = purity < 0.6
    print("\n| 组别 | 像素数 | 解混分类准确率 |")
    print("|---|---|---|")
    for name, mask in [("纯像元 (purity>=0.9)", pure), ("混合像元 (purity<0.6)", mixed)]:
        if mask.sum():
            print(f"| {name} | {int(mask.sum())} | {correct[mask].mean()*100:.2f}% |")

    # theoretical oracle: supervise only pixels whose unmixing prediction is
    # "pure and confident"; what accuracy does that subset reach?
    for pth in [0.7, 0.8, 0.9]:
        mask = purity >= pth
        if mask.sum():
            print(f"[oracle] purity>={pth}: 覆盖 {mask.sum()}/{len(idx)} "
                  f"({mask.mean()*100:.1f}%) 子集准确率 {correct[mask].mean()*100:.2f}%")

    print("\n[结论提示] 若 AUC(残差/纯度) > 0.6 且纯像元准确率显著高于整体，"
          "则解混先验可作为可靠性认证信号；否则该方向不可行。")


if __name__ == "__main__":
    main()
