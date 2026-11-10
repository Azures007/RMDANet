#!/usr/bin/env bash
# Pavia UDA baseline comparison under the RMDANet protocol.
#   source: PaviaU (180 labelled samples per class)
#   target: PaviaC (30000 unlabelled patches for training)
#   50 epochs x 40 steps = 2000 updates, batch 128, patch 7, lr 1e-3
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=/root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets/Pavia
SEED=${1:-1237}
OUT=results_benchmark

for M in dan dann jan; do
  echo "===== $M ====="
  python -u "baselines/${M}.py" \
    --dataset pavia --seed "$SEED" \
    --epochs 50 --steps-per-epoch 40 --batch-size 128 \
    --source-per-class 180 --lr 1e-3 --out-root "$OUT" \
    2>&1 | tee "/root/autodl-tmp/bench_${M}_seed${SEED}.log"
done

python -u baselines/aggregate.py --dataset pavia --seed "$SEED" --out-root "$OUT"
