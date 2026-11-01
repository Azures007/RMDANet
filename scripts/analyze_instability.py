#!/usr/bin/env python3
"""Mine historical TAPDA runs: which early-training indicator predicts a bad run?

Scans runs/*/log_*/ directories, extracts early-epoch indicators from the training
log and the final target OA from the result JSON, then reports:
  * Pearson / Spearman correlation between each indicator and final OA
  * separation between the best and worst quartile of runs
  * per-dataset breakdown

Usage:
    cd ~/autodl-tmp/RMDANet/third_party/tapda
    python ../../scripts/analyze_instability.py --root runs --out ../../results_analysis
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

TRAIN_RE = re.compile(
    r"Train Ep:\s*(?P<ep>\d+)\s*\[[^\]]*\]\s*Loss1:\s*(?P<loss1>[0-9.]+).*?"
    r"Loss2:\s*(?P<loss2>[0-9.]+).*?Dis:\s*(?P<dis>[0-9.]+)"
    r"(?:.*?ConS:\s*(?P<cons>[0-9.]+))?(?:.*?ConT:\s*(?P<cont>[0-9.]+))?"
)
INST_RE = re.compile(
    r"\[instability\]\s*ep=(?P<ep>\d+)\s*agree=(?P<agree>[0-9.]+)\s*conf=(?P<conf>[0-9.]+)"
    r"\s*score=(?P<score>[0-9.]+)"
)
DIAG_RE = re.compile(
    r"\[diag\]\s*ep=(?P<ep>\d+).*?conf=(?P<conf>[0-9.]+)\+-(?P<std>[0-9.]+)"
    r".*?gini_all=(?P<gini>[0-9.]+)"
)
OURS_RE = re.compile(r"\[ours\]\s*ep=(?P<ep>\d+)\s*pats_selected=(?P<sel>\d+).*?mean_prob=(?P<prob>[0-9.]+)")
OA_JSON_RE = re.compile(r'"acc_eval"\s*:\s*([0-9.]+)')
OA_LOG_RE = re.compile(r"Eval\(ensemble\):\s*([0-9.]+)")


def read_text(path: str, limit: int = 600_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def mean_upto(values: list[tuple[int, float]], upper: int) -> float | None:
    picked = [value for epoch, value in values if epoch <= upper]
    return float(np.mean(picked)) if picked else None


def find_oa(run_dir: str) -> float | None:
    for name in sorted(glob.glob(os.path.join(run_dir, "dataset_*_result.json"))):
        try:
            payload = json.load(open(name, encoding="utf-8"))
        except Exception:
            continue
        for key in ("acc_eval", "OA", "oa"):
            if key in payload:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    pass
    for name in glob.glob(os.path.join(run_dir, "*.log")) + glob.glob(os.path.join(run_dir, "*.txt")):
        text = read_text(name)
        matches = OA_LOG_RE.findall(text)
        if matches:
            return float(matches[-1])
    return None


def dataset_of(path: str) -> str:
    lowered = path.lower()
    if "houston" in lowered:
        return "Houston"
    if "nc12" in lowered or "yrd" in lowered:
        return "YRD"
    if "up_to_pc" in lowered or "pavia" in lowered:
        return "Pavia"
    return "other"


def collect(root: str) -> list[dict]:
    records = []
    for run_dir in sorted(glob.glob(os.path.join(root, "**", "log_*"), recursive=True)):
        if not os.path.isdir(run_dir):
            continue
        text = ""
        for name in sorted(glob.glob(os.path.join(run_dir, "*.log")))[:2]:
            text += read_text(name)
        if not text:
            continue
        loss1, loss2, dis, cons, cont = [], [], [], [], []
        agree, conf, inst, gini, sel = [], [], [], [], []
        for match in TRAIN_RE.finditer(text):
            epoch = int(match.group("ep"))
            loss1.append((epoch, float(match.group("loss1"))))
            loss2.append((epoch, float(match.group("loss2"))))
            dis.append((epoch, float(match.group("dis"))))
            if match.group("cons"):
                cons.append((epoch, float(match.group("cons"))))
            if match.group("cont"):
                cont.append((epoch, float(match.group("cont"))))
        for match in INST_RE.finditer(text):
            epoch = int(match.group("ep"))
            agree.append((epoch, float(match.group("agree"))))
            conf.append((epoch, float(match.group("conf"))))
            inst.append((epoch, float(match.group("score"))))
        for match in DIAG_RE.finditer(text):
            epoch = int(match.group("ep"))
            conf.append((epoch, float(match.group("conf"))))
            gini.append((epoch, float(match.group("gini"))))
        for match in OURS_RE.finditer(text):
            sel.append((int(match.group("ep")), float(match.group("sel"))))
        oa = find_oa(run_dir)
        if oa is None:
            continue
        record = {
            "dir": run_dir,
            "dataset": dataset_of(run_dir),
            "OA": oa,
            "loss1_3": mean_upto(loss1, 3),
            "loss2_3": mean_upto(loss2, 3),
            "dis_3": mean_upto(dis, 3),
            "cons_3": mean_upto(cons, 3),
            "cont_3": mean_upto(cont, 3),
            "loss1_5": mean_upto(loss1, 5),
            "dis_5": mean_upto(dis, 5),
            "agree_5": mean_upto(agree, 5),
            "conf_5": mean_upto(conf, 5),
            "inst_5": mean_upto(inst, 5),
            "gini_5": mean_upto(gini, 5),
            "sel_5": mean_upto(sel, 5),
        }
        records.append(record)
    return records


def ranks(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    rank = np.empty_like(order, dtype=float)
    rank[order] = np.arange(len(values), dtype=float)
    return rank


def pearson(x: np.ndarray, y: np.ndarray):
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray):
    if len(x) < 3:
        return float("nan")
    return pearson(ranks(x), ranks(y))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs")
    parser.add_argument("--out", default="results_analysis")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    records = collect(args.root)
    if not records:
        raise SystemExit(f"no usable runs under {args.root}")

    print(f"collected {len(records)} runs")
    names = [key for key in records[0] if key not in ("dir", "dataset", "OA")]
    datasets = sorted({record["dataset"] for record in records})
    print("datasets:", {name: sum(1 for r in records if r["dataset"] == name) for name in datasets})

    rows = []
    for name in names:
        usable = [record for record in records if record[name] is not None]
        if len(usable) < 5:
            continue
        oa = np.array([record["OA"] for record in usable], dtype=float)
        metric = np.array([record[name] for record in usable], dtype=float)
        per_dataset = {}
        for dataset in datasets:
            subset = [record for record in usable if record["dataset"] == dataset]
            if len(subset) >= 5:
                d_oa = np.array([record["OA"] for record in subset])
                d_metric = np.array([record[name] for record in subset])
                per_dataset[dataset] = round(spearman(d_metric, d_oa), 3)
        # quartile separation in z-scored OA (pooled, per-dataset standardised)
        z_oa = np.zeros(len(usable))
        for dataset in datasets:
            index = [i for i, record in enumerate(usable) if record["dataset"] == dataset]
            if not index:
                continue
            values = oa[index]
            if np.std(values) < 1e-9:
                continue
            z_oa[index] = (values - values.mean()) / values.std()
        order = np.argsort(metric)
        low = order[: max(1, len(order) // 4)]
        high = order[-max(1, len(order) // 4):]
        rows.append({
            "indicator": name,
            "n": len(usable),
            "spearman": round(spearman(metric, oa), 3),
            "pearson": round(pearson(metric, oa), 3),
            "zOA_low_quartile": round(float(z_oa[low].mean()), 3),
            "zOA_high_quartile": round(float(z_oa[high].mean()), 3),
            "quartile_gap": round(float(z_oa[high].mean() - z_oa[low].mean()), 3),
            "per_dataset_spearman": per_dataset,
        })

    rows.sort(key=lambda row: abs(row["quartile_gap"]), reverse=True)
    out_csv = os.path.join(args.out, "instability_predictors.csv")
    with open(out_csv, "w", encoding="utf-8") as handle:
        header = ["indicator", "n", "spearman", "pearson", "zOA_low_quartile", "zOA_high_quartile", "quartile_gap", "per_dataset_spearman"]
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(
                ",".join(str(row[key]) for key in header[:-1]) + "," + str(row["per_dataset_spearman"]) + "\n"
            )
    print()
    print(f"{'indicator':12s} {'n':>4s} {'spearman':>9s} {'gap(zOA)':>9s}  per-dataset")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['indicator']:12s} {row['n']:4d} {row['spearman']:9.3f} {row['quartile_gap']:9.3f}  {row['per_dataset_spearman']}"
        )
    print()
    print(f"table written to {out_csv}")
    print("interpretation: |quartile_gap| > 0.5 means the indicator separates good and bad runs by")
    print("at least half a standard deviation of OA - usable for early intervention.")


if __name__ == "__main__":
    main()
