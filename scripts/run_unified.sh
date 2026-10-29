#!/usr/bin/env bash
# Unified protocol v2: identical modules / schedule / epochs for all datasets.
# (120 epochs, 20 optimizer steps/epoch, pseudo start 40 + warm 24, LMMD start 40,
#  CCA + CORAL + LMMD + prototype alignment all enabled.)
# Usage: bash scripts/run_unified.sh [MODE] [SEED] [OUT_ROOT]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-C0}"
SEED="${2:-1237}"
OUT_ROOT="${3:-$ROOT/results_v71/unified}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-120}"

PAVIA_DATA="${PAVIA_DATA_ROOT:-/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets}"
OTHER_DATA="${DATA_ROOT:-/root/autodl-tmp/Datasets}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

cd "$ROOT"

run_one() {
  local ds="$1" data_root="$2" data_path="$3"
  local run="$OUT_ROOT/$ds/${MODE}_epochs${EPOCHS}_seed${SEED}"
  echo "==================== ${ds^^} (${EPOCHS} epochs) ===================="
  python -u scripts/run_v71.py "$ds" "$MODE" "$SEED" \
    --data-root "$data_root" --epochs "$EPOCHS" --workers "$WORKERS" --out-root "$OUT_ROOT"
  python -u scripts/final_eval_locked.py \
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
