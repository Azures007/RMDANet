#!/usr/bin/env bash
# Unified protocol runner: identical modules / schedule / epochs for all datasets.
# Epochs are taken from each dataset config unless the EPOCHS env var is set.
# Usage: bash scripts/run_unified.sh [MODE] [SEED] [OUT_ROOT]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-C0}"
SEED="${2:-1237}"
OUT_ROOT="${3:-$ROOT/results_v71/unified}"
WORKERS="${WORKERS:-4}"
EPOCHS_OVERRIDE="${EPOCHS:-}"

PAVIA_DATA="${PAVIA_DATA_ROOT:-/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets}"
OTHER_DATA="${DATA_ROOT:-/root/autodl-tmp/Datasets}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

cd "$ROOT"

config_epochs() {
  python -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1],encoding='utf8'))['training']['epochs'])" \
    "$ROOT/datasets/$1/config.yaml"
}

run_one() {
  local ds="$1" data_root="$2" data_path="$3"
  local epochs
  epochs="${EPOCHS_OVERRIDE:-$(config_epochs "$ds")}"
  local run="$OUT_ROOT/$ds/${MODE}_epochs${epochs}_seed${SEED}"
  echo "==================== ${ds^^} (${epochs} epochs) ===================="
  python -u scripts/run_v71.py "$ds" "$MODE" "$SEED" \
    --data-root "$data_root" --epochs "$epochs" --workers "$WORKERS" --out-root "$OUT_ROOT"
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
