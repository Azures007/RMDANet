#!/usr/bin/env python3
import re
import numpy as np
from pathlib import Path

PROJECT = Path("/root/autodl-tmp/pavia-test/RMDANet_v5.1")

DATASETS = {
    "Pavia": {
        "log": PROJECT / "results_pavia_delloss_seed1236" / "run.log",
        "classes": ["Asphalt", "Meadows", "Gravel", "Trees",
                    "Painted", "BareSoil", "Bitumen"],
    },
    "HyRANK": {
        "log": PROJECT / "results_hyrank_final_seed1237" / "run.log",
        "classes": ["DenseUrban", "Mineral", "NonIrrigated", "FruitTrees",
                    "OliveGroves", "Coniferous", "DenseScle", "SparseScle",
                    "SparselyVeg", "RocksSand", "Water", "CoastalWater"],
    },
    "Houston": {
        "log": PROJECT / "results_houston_final_seed1238" / "run.log",
        "classes": ["GrassHealthy", "GrassStressed", "Trees", "Water",
                    "Residential", "NonResidential", "Road"],
    },
}

for name, info in DATASETS.items():
    log = info["log"]
    if not log.exists():
        print(f"\n[{name}] 日志不存在: {log}")
        continue
    text = log.read_text(errors="replace")
    print(f"\n{'='*60}")
    print(f"  {name}  ({log.parent.name})")
    print(f"{'='*60}")
    # 提取 OA/AA/Kappa
    for m in re.finditer(r"OA=([\d.]+)\s+AA=([\d.]+)\s+Kappa=([\d.]+)", text):
        oa, aa, kp = m.group(1), m.group(2), m.group(3)
    print(f"  OA={oa}  AA={aa}  Kappa={kp}")
    # 提取混淆矩阵
    cm = re.search(r"Final target benchmark\s*\n(\[\[.*?\]\])", text, re.S)
    if cm:
        matrix = np.array(eval(cm.group(1)))
        row_sum = matrix.sum(axis=1)
        acc = np.divide(matrix.diagonal(), row_sum,
                        out=np.zeros_like(row_sum, dtype=float),
                        where=row_sum > 0) * 100
        print(f"\n  {'Class':<22} {'Acc':>8}  {'Correct':>8} {'Total':>8}")
        print(f"  {'-'*50}")
        for i, cname in enumerate(info["classes"]):
            if i < len(acc):
                print(f"  {cname:<22} {acc[i]:>7.2f}%  {int(matrix[i,i]):>8} {int(row_sum[i]):>8}")

print("\n完成。")
