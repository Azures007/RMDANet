#!/usr/bin/env bash
# Audit same-protocol fairness for baseline logs (five conditions from the paper protocol).
# Usage: bash scripts/audit_protocol.sh <log_dir_or_file...>
set -uo pipefail

check_file() {
  local file="$1"
  [ -f "$file" ] || return 0
  local fail=0
  echo "===== $file"

  # 1) label budget: 180/class -> Pavia/Houston 1260, HyRANK ~2130, YRD 1155
  if grep -qiE "(processed all data|trainx|train samples|nTrain|样本数)" "$file"; then
    grep -iE "processed all data|train samples|nTrain" "$file" | head -2 | sed 's/^/  [budget] /'
  else
    echo "  [budget] WARN: cannot find the source sample count in the log"
  fi

  # 2) target GT must not be used during training
  if grep -qiE "target[ _-]?gt.*(loss|train)|use_gt_target_mask.*True" "$file"; then
    echo "  [target-GT] FAIL: target GT appears to be used in training"
    fail=1
  else
    echo "  [target-GT] OK: no target-GT training signature"
  fi

  # 3) epochs / update steps recorded
  grep -ioE "epoch[^0-9]{0,4}[0-9]+|step[^0-9]{0,4}[0-9]+|iter[^0-9]{0,4}[0-9]+" "$file" | tail -2 | sed 's/^/  [schedule] /'

  # 4) TTA / ensemble must be off (report if present)
  if grep -qiE "\bTTA\b|test.?time.?aug|flip.*eval|ensemble.*aug" "$file"; then
    echo "  [TTA] WARN: TTA/ensemble keywords found - verify manually"
  else
    echo "  [TTA] OK: no TTA keyword"
  fi

  # 5) final OA present
  if grep -qiE "\bOA\b\s*[:=]" "$file"; then
    grep -iE "\bOA\b\s*[:=]" "$file" | tail -1 | sed 's/^/  [OA] /'
  else
    echo "  [OA] WARN: no OA line found"
  fi

  [ "$fail" -eq 0 ] || echo "  RESULT: FAIL"
}

if [ "$#" -eq 0 ]; then
  set -- results_comparison/*.log
fi
for target in "$@"; do
  if [ -d "$target" ]; then
    find "$target" -maxdepth 2 -name "*.log" -print0 | while IFS= read -r -d '' log; do check_file "$log"; done
  else
    check_file "$target"
  fi
done
