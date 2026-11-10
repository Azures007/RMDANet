#!/usr/bin/env bash
# Final three-dataset module ablation for the clean v7.0 build.
# Usage: bash run_final_ablation.sh [seed]
set -euo pipefail
cd "$(dirname "$0")"
SEED=${1:-1237}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# DRSE residual center-guided pooling is baked into core/model.py and configured
# per dataset in datasets/*/config.yaml (pool_center_size/pool_temperature/...).

PAVIA_DATA=/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets/Pavia
HOUSTON_DATA=/root/autodl-tmp/Datasets/Houston
HYRANK_DATA=/root/autodl-tmp/Datasets/HyRANK

train_row() {  # dataset tag prat pool fusion rtml
  local ds=$1 tag=$2 prat=$3 pool=$4 fusion=$5 rtml=$6
  python -u ablate_4rows.py --dataset "$ds" --seed "$SEED" --gpu 0 --num_workers 4 \
    --abl-prat "$prat" --abl-pool "$pool" --abl-fusion "$fusion" --abl-rtml "$rtml" \
    --tag "$tag"
}

eval_row() {  # dataset tag data_path
  local ds=$1 tag=$2 data=$3
  local RUN="results_ablation_clean/$ds/${tag}_seed${SEED}"
  python -u eval_tuned.py --checkpoint "$RUN/selected_model.pt" --data-path "$data" \
    --output "$RUN/target_final_once.json" --confirm-final-eval
}

run_dataset() {  # dataset data_path alpha extra_fusion_args rows...
  local ds=$1 data=$2 alpha=$3 extra=$4; shift 4
  export DRSE_ALPHA_LOGIT="$alpha"
  for spec in "$@"; do
    set -- $spec
    local tag=$1
    train_row "$ds" "${ds}_${tag}" "$2" "$3" "$4" "$5" $extra
    eval_row "$ds" "${ds}_${tag}" "$data"
  done
}

# Final protocol: bidirectional spectral reliability fusion on DRSE rows,
# residual center-guided pooling, per-dataset alpha (see datasets/*/config.yaml).
# Houston and Pavia: alpha logit -4 (residual strength 0.018)
run_dataset houston "$HOUSTON_DATA" -4 "" \
  "base 0 gap off 0" \
  "drse 0 guided bidirectional 0" \
  "drse_rftm 1 guided bidirectional 0" \
  "full 1 guided bidirectional 1"

run_dataset pavia "$PAVIA_DATA" -4 "" \
  "base 0 gap off 0" \
  "drse 0 guided bidirectional 0" \
  "drse_rftm 1 guided bidirectional 0" \
  "full 1 guided bidirectional 1"

# HyRANK: fusion disabled by the label-free rule (target spectral-branch
# confidence 0.599 < 0.65), alpha logit -3.5 (residual strength 0.029).
run_dataset hyrank "$HYRANK_DATA" -3.5 "" \
  "base 0 gap off 0" \
  "drse 0 guided off 0" \
  "drse_rftm 1 guided off 0" \
  "full 1 guided off 1"

echo
echo "================ FINAL SUMMARY ================"
python - "$SEED" <<'PY'
import glob
import json
import sys

seed = sys.argv[1]
for ds in ("houston", "pavia", "hyrank"):
    print(f"\n### {ds}")
    for p in sorted(glob.glob(f"results_ablation_clean/{ds}/*_seed{seed}/target_final_once.json")):
        m = json.load(open(p))["metrics"]
        print(f"{p.split('/')[-2]:34s} OA={m['OA']:6.2f} AA={m['AA']:6.2f} Kappa={m['Kappa']:.4f}")
PY
