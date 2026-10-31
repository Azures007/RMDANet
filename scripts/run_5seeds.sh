#!/usr/bin/env bash
# 5-seed final runs for the three TAPDA-based datasets (paper table protocol).
# Usage: bash scripts/run_5seeds.sh [OUT_ROOT]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/results_final_5seed}"
mkdir -p "$OUT"
cd "$ROOT/third_party/tapda"

export PAVIA="${PAVIA_DATA_ROOT:-/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets}"
export OTHER="${DATA_ROOT:-/root/autodl-tmp/Datasets}"

run() {
  local name="$1"; shift
  echo "===== ${name} =====" | tee -a "$OUT/${name}.log"
  "$@" 2>&1 | tee -a "$OUT/${name}.log" | grep -E "Eval\(ensemble\)"
}

# Pavia: M1
run pavia_m1_5seed python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight 0.05
# Houston: M1 + class quota
run houston_m1quota_5seed python -u houston13218.py --gpu 0 --num_seeds 5 \
  --reliab_ce_weight 0.05 --target_class_quota_mode confusion_ema
# YRD: plain baseline
run yrd_base_5seed python -u nc122nc13.py --gpu 0 --num_seeds 5

echo "===== DONE -> $OUT ====="
