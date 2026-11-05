"""Pre-check for Fourier-domain target-statistics adaptation (candidate 2).

Question
--------
Does the input-level domain gap between the source and target scenes have enough
"low-frequency amplitude" difference for FDA-style re-parameterization to help?
And does such a transformation actually move the source statistics towards the
target statistics (i.e. is the lever real)?

The script never uses target labels.  It reports:

  1. per-band mean/std distance between source and target;
  2. low-frequency amplitude ratio (target/source) per band;
  3. a simulation of FDA: source phase + target low-frequency amplitude, then
     the remaining statistics distance;
  4. a source-side class-separability proxy (Fisher ratio) before/after to make
     sure the transformation does not destroy the class structure.

Decision rule (printed at the end)
    * amplitude gap large (median |log ratio| > 0.15)  -> lever exists;
    * post-FDA statistics distance reduced by >= 25%   -> transformation works;
    * Fisher ratio drop <= 10%                          -> class structure kept.
Only if all three hold is it worth implementing the training-time FDA module.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util.utils as utils  # noqa: E402

CONFIGS = {
    "houston": dict(loader="load_data_houston",
                    src=("Houston/Houston13.mat", "Houston/Houston13_7gt.mat"),
                    tar=("Houston/Houston18.mat", "Houston/Houston18_7gt.mat")),
    "pavia": dict(loader="load_data_pavia",
                  src=("Pavia/paviaU.mat", "Pavia/paviaU_gt_7.mat"),
                  tar=("Pavia/pavia.mat", "Pavia/pavia_gt_7.mat")),
    "yrd": dict(loader="load_data_YRD",
                src=("YRD/yrd_nc12.mat", "YRD/yrd_nc12_7gt.mat"),
                tar=("YRD/yrd_nc13.mat", "YRD/yrd_nc13_7gt.mat")),
}


def load(cfg, root):
    fn = getattr(utils, cfg["loader"])
    s_cube, s_gt = fn(os.path.join(root, cfg["src"][0]), os.path.join(root, cfg["src"][1]))
    t_cube, t_gt = fn(os.path.join(root, cfg["tar"][0]), os.path.join(root, cfg["tar"][1]))
    return (np.asarray(s_cube, np.float64), np.asarray(s_gt).astype(np.int64),
            np.asarray(t_cube, np.float64), np.asarray(t_gt).astype(np.int64))


def band_stats(cube, mask):
    """Per-band mean/std over valid pixels."""
    flat = cube.reshape(-1, cube.shape[-1])
    m = mask.reshape(-1)
    x = flat[m]
    return x.mean(axis=0), x.std(axis=0)


def low_freq_amplitude(cube, radius_frac=0.1):
    """Per-band mean magnitude of the low-frequency 2-D FFT components."""
    h, w, b = cube.shape
    out = np.zeros(b)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    mask = r <= radius_frac
    for i in range(b):
        spec = np.fft.fft2(cube[..., i])
        out[i] = np.abs(spec)[mask].mean()
    return out


def apply_lowfreq_matching(src, tar, radius_frac=0.1):
    """Source phase + target low-frequency amplitude (per band)."""
    h, w, b = src.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    low = r <= radius_frac
    out = np.empty_like(src)
    for i in range(b):
        fs = np.fft.fft2(src[..., i])
        ft = np.fft.fft2(tar[..., i])
        a_s = np.abs(fs)
        a_t = np.abs(ft)
        scale = a_t[low].mean() / max(1e-8, a_s[low].mean())
        fs_scaled = np.where(low, fs * scale, fs)
        out[..., i] = np.real(np.fft.ifft2(fs_scaled))
    return out


def fisher_ratio(cube, gt, num_classes, max_pixels=20000, seed=0):
    """Class-separability proxy on raw spectra (higher = better)."""
    flat = cube.reshape(-1, cube.shape[-1])
    y = gt.reshape(-1)
    idx = np.nonzero(y > 0)[0]
    rng = np.random.default_rng(seed)
    if len(idx) > max_pixels:
        idx = rng.choice(idx, size=max_pixels, replace=False)
    X, yy = flat[idx], y[idx]
    mu = X.mean(axis=0)
    within, between = 0.0, 0.0
    for c in range(1, num_classes + 1):
        m = yy == c
        if m.sum() < 2:
            continue
        Xc = X[m]
        within += ((Xc - Xc.mean(axis=0)) ** 2).sum()
        between += m.sum() * ((Xc.mean(axis=0) - mu) ** 2).sum()
    return between / max(1e-8, within)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="houston", choices=sorted(CONFIGS))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--num-classes", type=int, default=7)
    ap.add_argument("--radius", type=float, default=0.1)
    args = ap.parse_args()

    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    cfg = CONFIGS[args.dataset]
    s_cube, s_gt, t_cube, t_gt = load(cfg, root)
    print(f"[fda-check] source {s_cube.shape}  target {t_cube.shape}")

    ms, ss = band_stats(s_cube, s_gt > 0)
    mt, st = band_stats(t_cube, t_gt > 0)
    eps = 1e-8

    def rel(a, b):
        return np.abs(a - b) / (np.abs(a) + eps)

    d_mean = float(np.mean(rel(ms, mt)) * 100)
    d_std = float(np.mean(rel(ss, st)) * 100)
    cos_mean = float(np.dot(ms, mt) / (np.linalg.norm(ms) * np.linalg.norm(mt) + eps))

    a_s = low_freq_amplitude(s_cube, args.radius)
    a_t = low_freq_amplitude(t_cube, args.radius)
    log_ratio = np.log((a_t + eps) / (a_s + eps))
    amp_gap = float(np.median(np.abs(log_ratio)))
    big_gap_bands = int((np.abs(log_ratio) > np.log(1.2)).sum())

    fr_before = fisher_ratio(s_cube, s_gt, args.num_classes)
    s_fda = apply_lowfreq_matching(s_cube, t_cube, args.radius)
    mf, sf_ = band_stats(s_fda, s_gt > 0)
    d_mean_after = float(np.mean(rel(mf, mt)) * 100)
    d_std_after = float(np.mean(rel(sf_, st)) * 100)
    fr_after = fisher_ratio(s_fda, s_gt, args.num_classes)

    red_mean = 1.0 - d_mean_after / max(eps, d_mean)
    red_std = 1.0 - d_std_after / max(eps, d_std)
    fr_drop = 1.0 - fr_after / max(eps, fr_before)

    print("\n===== 输入统计差异（源 vs 目标）=====")
    print(f"每波段均值相对差（平均） : {d_mean:.2f}%")
    print(f"每波段方差相对差（平均） : {d_std:.2f}%")
    print(f"平均光谱余弦相似度       : {cos_mean:.4f}")
    print(f"低频振幅 log 比中位数    : {amp_gap:.4f}（{big_gap_bands}/{len(a_s)} 个波段偏差>20%）")

    print("\n===== FDA 模拟（源相位 + 目标低频振幅）=====")
    print(f"变换后均值相对差 : {d_mean_after:.2f}%（下降 {red_mean*100:.1f}%）")
    print(f"变换后方差相对差 : {d_std_after:.2f}%（下降 {red_std*100:.1f}%）")
    print(f"Fisher 比 before/after: {fr_before:.3f} / {fr_after:.3f}（下降 {fr_drop*100:.1f}%）")

    lever = amp_gap > 0.15
    works = (red_mean >= 0.25) or (red_std >= 0.25)
    keeps = fr_drop <= 0.10
    print("\n===== 判定 =====")
    print(f"低频振幅差异足够（>0.15）      : {'是' if lever else '否'}")
    print(f"FDA 后统计距离下降>=25%        : {'是' if works else '否'}")
    print(f"类别可分性损失<=10%            : {'是' if keeps else '否'}")
    print(f"\n=> 候选 2 {'可行，建议实现训练期 FDA' if (lever and works and keeps) else '不可行/收益有限，建议放弃'}")


if __name__ == "__main__":
    main()
