#!/usr/bin/env bash
# True DRSE-off baseline for the module ablation.
#
# DRSE = Dual-branch Reliability Spectral-spatial Encoder.  ``--abl-drse 0``
# removes the spectral branch and the reliability fusion, leaving a single
# spatial branch; it is the only new row needed (all DRSE-on rows are unchanged
# by the switch and keep their existing results).
set -euo pipefail
cd "$(dirname "$0")"
SEED=${1:-1237}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PAVIA_DATA=/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets/Pavia
HOUSTON_DATA=/root/autodl-tmp/Datasets/Houston
HYRANK_DATA=/root/autodl-tmp/Datasets/HyRANK

train_eval() {  # dataset tag alpha data_path
  local ds=$1 tag=$2 alpha=$3 data=$4
  export DRSE_ALPHA_LOGIT="$alpha"
  python -u ablate_4rows.py --dataset "$ds" --seed "$SEED" --gpu 0 --num_workers 4 \
    --abl-drse 0 --abl-prat 0 --abl-pool gap --abl-fusion off --abl-rtml 0 \
    --tag "$tag"
  local RUN="results_ablation_clean/$ds/${tag}_seed${SEED}"
  python -u scripts/final_eval_locked.py --checkpoint "$RUN/selected_model.pt" \
    --data-path "$data" --output "$RUN/target_final_once.json" --confirm-final-eval
}

# alpha logit only affects RCGP, which is off here; kept identical to the
# dataset-level configuration for reproducibility.
train_eval houston houston_drseoff -4 "$HOUSTON_DATA"
train_eval pavia   pavia_drseoff   -4 "$PAVIA_DATA"
train_eval hyrank  hyrank_drseoff  -3.5 "$HYRANK_DATA"

echo
echo "================ DRSE-OFF SUMMARY ================"
python - "$SEED" <<'PY'
import json
import sys

seed = sys.argv[1]
for ds in ("houston", "pavia", "hyrank"):
    path = f"results_ablation_clean/{ds}/{ds}_drseoff_seed{seed}/target_final_once.json"
    m = json.load(open(path))["metrics"]
    print(f"{ds:8s} DRSE-off  OA={m['OA']:6.2f} AA={m['AA']:6.2f} Kappa={m['Kappa']:.4f}")
PY
