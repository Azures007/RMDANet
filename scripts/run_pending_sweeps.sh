#!/usr/bin/env bash
# One-shot runner for all pending week-1 sweeps (C group, M2 pool, Pavia M1 sweep, YRD variants).
# Usage: bash scripts/run_pending_sweeps.sh [OUT_DIR]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/results_sweeps/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
cd "$ROOT/third_party/tapda"
ln -sfn /root/autodl-tmp/Datasets/YRD_extract/YRD datasets/YRD 2>/dev/null || true
echo "logs -> $OUT"

run() {
  local tag="$1"; shift
  echo "===== $tag =====" | tee -a "$OUT/$tag.log"
  "$@" 2>&1 | tee -a "$OUT/$tag.log" | grep -E "Eval\(ensemble\)"
}

# --- C group (second-mechanism candidates) ---
run C1_maturity_gate      python -u up2pc.py --gpu 0 --num_seeds 2 --reliab_ce_weight 0.05 --m1_maturity_gate
run C2_class_cap4         python -u up2pc.py --gpu 0 --num_seeds 2 --reliab_ce_weight 0.05 --m1_class_cap 4
run C3_maturity_cap4      python -u up2pc.py --gpu 0 --num_seeds 2 --reliab_ce_weight 0.05 --m1_maturity_gate --m1_class_cap 4
run S4_cnt_shift          python -u up2pc.py --gpu 0 --num_seeds 2 --reliab_ce_weight 0.05 --cnt_shift_weight 0.1

# --- M2 pool (new mechanisms) ---
run M2S_soft_label        python -u up2pc.py --gpu 0 --num_seeds 2 --reliab_ce_weight 0.05 --m1_soft_label
run M2C_class_adaptive    python -u up2pc.py --gpu 0 --num_seeds 2 --reliab_ce_weight 0.05 --m1_class_adaptive 0.05
run M2T_trim              python -u up2pc.py --gpu 0 --num_seeds 2 --reliab_ce_weight 0.05 --m1_trim_ratio 0.2

# --- Pavia M1 hyperparameter sweep ---
for pair in "0.08 0.95" "0.10 0.95" "0.05 0.97" "0.05 0.93"; do
  w="${pair% *}"; p="${pair#* }"
  run "M1_w${w}_p${p}" python -u up2pc.py --gpu 0 --num_seeds 2 --reliab_ce_weight "$w" --reliab_ce_min_prob "$p"
done

# --- YRD variants (long-tail) ---
for pair in "0.05 0.85" "0.05 0.80" "0.08 0.90"; do
  w="${pair% *}"; p="${pair#* }"
  run "YRD_w${w}_p${p}" python -u nc122nc13.py --gpu 0 --num_seeds 2 --reliab_ce_weight "$w" --reliab_ce_min_prob "$p"
done
run YRD_balance_inverse   python -u nc122nc13.py --gpu 0 --num_seeds 2 --reliab_ce_weight 0.05 --reliab_ce_balance inverse

echo "===== ALL SWEEPS DONE -> $OUT ====="
