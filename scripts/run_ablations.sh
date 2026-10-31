#!/usr/bin/env bash
# Final ablation + sensitivity matrix, run AFTER the best configuration is locked.
# Usage:
#   BEST_W=0.05 BEST_P=0.95 BEST_EXTRA="--m1_class_quota 4" bash scripts/run_ablations.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT_DIR:-$ROOT/results_ablations/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
cd "$ROOT/third_party/tapda"
BEST_W="${BEST_W:-0.05}"
BEST_P="${BEST_P:-0.95}"
BEST_EXTRA="${BEST_EXTRA:-}"
echo "best: w=$BEST_W p=$BEST_P extra=$BEST_EXTRA -> $OUT"

run() {
  local tag="$1"; shift
  echo "===== $tag =====" | tee -a "$OUT/$tag.log"
  "$@" 2>&1 | tee -a "$OUT/$tag.log" | grep -E "Eval\(ensemble\)"
}

# --- weight ablation (M1 off / 0.03 / best / 0.10) ---
run W_off     python -u up2pc.py --gpu 0 --num_seeds 5
run W_003     python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight 0.03 --reliab_ce_min_prob "$BEST_P"
run W_best    python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$BEST_P" $BEST_EXTRA
run W_010     python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight 0.10 --reliab_ce_min_prob "$BEST_P"

# --- threshold sensitivity ---
for p in 0.90 0.93 0.97; do
  run "P_${p}" python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$p"
done

# --- gate ablations (each on top of the best M1) ---
run G_maturity  python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$BEST_P" --m1_maturity_gate
run G_cap4      python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$BEST_P" --m1_class_cap 4
run G_quota4    python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$BEST_P" --m1_class_quota 4
run G_adapt     python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$BEST_P" --m1_class_adaptive 0.05
run G_soft      python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$BEST_P" --m1_soft_label
run G_trim      python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$BEST_P" --m1_trim_ratio 0.2
run G_temp      python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$BEST_P" --m1_teacher_temperature 2.0
run G_margin    python -u up2pc.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$BEST_P" --m1_margin_gate 0.02

# --- Houston class quota (dataset-specific switch) ---
run H_noquota   python -u houston13218.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$BEST_P"
run H_quota     python -u houston13218.py --gpu 0 --num_seeds 5 --reliab_ce_weight "$BEST_W" --reliab_ce_min_prob "$BEST_P" --target_class_quota_mode confusion_ema

echo "===== ABLATIONS DONE -> $OUT ====="
