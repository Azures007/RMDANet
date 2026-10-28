#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# < 2 ));then
 echo 'Usage: bash scripts/eval_three_v7.sh MODE SEED [EPOCHS|default] [RESULTS_ROOT] [DATA_ROOT]' >&2
 exit 2
fi
MODE="$1"; SEED="$2"; EPOCHS="${3:-default}"; OUTROOT="${4:-$ROOT/results_v7}"; DATAROOT="${5:-/root/autodl-tmp/Datasets}"
for DS in pavia houston hyrank;do
 case "$DS" in
   pavia) D=Pavia; E=110;;
   houston) D=Houston; E=280;;
   hyrank) D=HyRANK; E=120;;
 esac
 if [[ "$EPOCHS" != 'default' ]];then E="$EPOCHS";fi
 RUN="$OUTROOT/$DS/${MODE}_epochs${E}_seed${SEED}"
 if [[ -f "$RUN/target_final_once.json" ]];then
   echo "SKIP: already evaluated $RUN";continue
 fi
 python -u "$ROOT/scripts/final_eval_locked.py" \
  --checkpoint "$RUN/selected_model.pt" --data-path "$DATAROOT/$D" \
  --output "$RUN/target_final_once.json" --confirm-final-eval
done
