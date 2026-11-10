#!/usr/bin/env bash
set -euo pipefail
DS=${1:-houston}
SEED=${2:-1237}
ROWS=${3:-"base drse drse_rftm full"}

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT="results_ablation_clean/$DS"
for ROW in $ROWS; do
  RUN="$OUT/${ROW}_seed${SEED}"
  if [ -f "$RUN/target_final_once.json" ]; then
    echo "[skip] $ROW already evaluated"
    continue
  fi
  rm -rf "$RUN"
  python -u ablate_4rows.py --dataset "$DS" --row "$ROW" --seed "$SEED" \
    --gpu 0 --num_workers 4 2>&1 | tee "/root/autodl-tmp/ablation_clean_${DS}_${ROW}.log"

  if [ "$DS" = "pavia" ]; then
    DATA=/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets/Pavia
  elif [ "$DS" = "houston" ]; then
    DATA=/root/autodl-tmp/Datasets/Houston
  else
    DATA=/root/autodl-tmp/Datasets/HyRANK
  fi

  python -u scripts/final_eval_locked.py \
    --checkpoint "$RUN/selected_model.pt" \
    --data-path "$DATA" \
    --output "$RUN/target_final_once.json" \
    --confirm-final-eval
  python -u scripts/report_metrics.py "$RUN/target_final_once.json"
done

echo "==== summary: $DS ===="
python - "$OUT" <<'PY'
import json, glob, sys
out = sys.argv[1]
for p in sorted(glob.glob(f"{out}/*/target_final_once.json")):
    m = json.load(open(p))["metrics"]
    print(p.split("/")[-2], f"OA={m['OA']:.2f} AA={m['AA']:.2f} Kappa={m['Kappa']:.4f}")
PY
