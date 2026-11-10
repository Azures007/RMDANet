#!/usr/bin/env bash
# Pavia baseline comparison, second batch: MTLDA / CACL / UIPCL / PCDM / VCMoE.
# Same protocol as the first batch, median (amplitude-preserving) normalization
# so that the baseline numbers are comparable to their published settings.
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=${1:-1237}
OUT=results_benchmark_median

for M in mtlda cacl uipcl pcdm vcmoe; do
  echo "===== $M ====="
  python -u "baselines/${M}.py" \
    --dataset pavia --seed "$SEED" \
    --epochs 50 --steps-per-epoch 40 --batch-size 128 \
    --source-per-class 180 --lr 1e-3 --normalize-mode median --out-root "$OUT" \
    2>&1 | tee "/root/autodl-tmp/bench_${M}_median_seed${SEED}.log"
done

python -u baselines/aggregate.py --dataset pavia --seed "$SEED" --out-root "$OUT"
