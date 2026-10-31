#!/usr/bin/env bash
# Same-protocol MSDA baseline (Pavia / Houston / HyRANK), 2 seeds each, with locked eval.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/results_comparison/msda}"
mkdir -p "$OUT"
cd "$ROOT"

export PAVIA="${PAVIA_DATA_ROOT:-/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets}"
export OTHER="${DATA_ROOT:-/root/autodl-tmp/Datasets}"

run_dataset() {
  local ds="$1" seeds="$2" data_root="$3" data_path="$4"
  for seed in $seeds; do
    echo "===== MSDA ${ds} seed=${seed} ====="
    python -u scripts/run_msda.py "$ds" "$seed" --data-root "$data_root" --out-root "$OUT" \
      2>&1 | tee "$OUT/${ds}_${seed}.log" | tail -3
    local ckpt
    ckpt=$(find "$OUT/$ds" -path "*seed${seed}*" -name selected_model.pt | head -1)
    if [ -n "$ckpt" ]; then
      python -u scripts/final_eval_msda.py --checkpoint "$ckpt" --data-path "$data_path" \
        --output "$(dirname "$ckpt")/target_final_once.json" --confirm-final-eval \
        2>&1 | tee -a "$OUT/${ds}_${seed}.log" | grep -E "OA|Kappa"
    fi
  done
}

run_dataset pavia "1726 1040" "$PAVIA" "$PAVIA/Pavia"
run_dataset houston "1599 1364" "$OTHER" "$OTHER/Houston"
run_dataset hyrank "1726 1040" "$OTHER" "$OTHER/HyRANK"

echo "===== MSDA DONE -> $OUT ====="
