#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# < 2 ));then
 echo "Usage: bash scripts/run_three_v71.sh C0|C1|C2|C3|C4 SEED [--epochs N --batch-size N --patch-size N --source-train-ratio R ...]" >&2
 exit 2
fi
MODE="$1"; SEED="$2";shift 2
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
cd "$ROOT"
for DS in pavia houston hyrank;do
 echo "========== $DS / $MODE / seed $SEED =========="
 python -u scripts/run_v71.py "$DS" "$MODE" "$SEED" "$@"
done
