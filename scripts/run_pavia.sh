#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SEED="${1:-1236}"
SAVE_ROOT="$PKG_ROOT/results_pavia_seed${SEED}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$SAVE_ROOT"
cd "$PKG_ROOT/datasets/pavia"

python -u train.py \
    --gpu 0 \
    --data_path /root/autodl-tmp/Test_1/PTMDA-Net_ReliabilityCore_Final_Standalone/Datasets/Pavia \
    --save_path "$SAVE_ROOT" \
    --seed "$SEED" \
    2>&1 | tee "$SAVE_ROOT/run.log"
