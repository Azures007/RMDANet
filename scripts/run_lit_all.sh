#!/usr/bin/env bash
# Run the literature recipe (lit-recipe branch) on all three datasets.
# Usage: bash scripts/run_lit_all.sh [SEED] [OUT_ROOT]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-1237}"
OUT_ROOT="${2:-$ROOT/results_lit}"
WORKERS="${WORKERS:-4}"

PAVIA_DATA="${PAVIA_DATA_ROOT:-/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets}"
OTHER_DATA="${DATA_ROOT:-/root/autodl-tmp/Datasets}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

cd "$ROOT"

run_one() {
  local ds="$1" data_root="$2" data_path="$3"
  local run="$OUT_ROOT/$ds/lit_epochs50_seed${SEED}"
  echo "==================== ${ds^^} (lit-recipe) ===================="
  python -u scripts/run_lit.py "$ds" "$SEED" \
    --data-root "$data_root" --out-root "$OUT_ROOT" --workers "$WORKERS"
  python -u scripts/final_eval_lit.py \
    --checkpoint "$run/selected_model.pt" \
    --data-path "$data_path" \
    --output "$run/target_final_once.json" --confirm-final-eval
  echo "########## ${ds^^} RESULT ##########"
  python -u scripts/report_metrics.py "$run/target_final_once.json"
}

run_one pavia "$PAVIA_DATA" "$PAVIA_DATA/Pavia"
run_one hyrank "$OTHER_DATA" "$OTHER_DATA/HyRANK"
run_one houston "$OTHER_DATA" "$OTHER_DATA/Houston"

echo "########## ALL DONE ##########"
