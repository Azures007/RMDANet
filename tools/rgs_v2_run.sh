#!/usr/bin/env bash
# RGS v2 (independent selector) - 3 datasets x 3 seeds, all innovations on.
# Usage: bash rgs_v2_run.sh [selector]   (default: independent_joint)
set +e
cd ~/autodl-tmp/tapda_repro

SEL="${1:-independent_joint}"
STAMP=$(date +%Y%m%d-%H%M%S)
D=/root/autodl-tmp/experiments/${STAMP}_rgs_${SEL}
mkdir -p "$D/logs"

git fetch https://gh-proxy.com/https://github.com/Azures007/RMDANet.git rgs2
git reset --hard FETCH_HEAD
git log --oneline -1 | tee "$D/git_head.txt"
git fetch https://gh-proxy.com/https://github.com/Azures007/RMDANet.git tapda-base
git show FETCH_HEAD:tools/make_tapda_hyrank.py > tools/make_tapda_hyrank.py
python tools/make_tapda_hyrank.py --tapda-dir . --source up2pc.py --target dioni2loukia.py >/dev/null 2>&1

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CFG="--ssti --ratc --reliab_ce_weight 0.05 --rgs_selector ${SEL}"

run () {
  local tag="$1"; shift
  echo "== $tag start $(date +%H:%M:%S)"
  python -u "$@" 2>&1 | tee "$D/logs/${tag}.log" \
    | grep -E "\[rgs_|\[ratc\]|Eval\(ensemble\)|Traceback"
  echo "== $tag end $(date +%H:%M:%S)"
}

run Houston houston13218.py --gpu 0 --num_seeds 3 $CFG
run HyRANK  dioni2loukia.py --gpu 0 --num_seeds 3 $CFG
run Pavia   up2pc.py       --gpu 0 --num_seeds 3 $CFG

B=/root/autodl-tmp/runs_auto/logs
python /root/autodl-tmp/summarize_runs.py $B/Houston_TAPDA_3.log "$D/logs/Houston.log" --out "$D/R_Houston.md"
python /root/autodl-tmp/summarize_runs.py $B/HyRANK_TAPDA_3.log  "$D/logs/HyRANK.log"  --out "$D/R_HyRANK.md"
python /root/autodl-tmp/summarize_runs.py $B/Pavia_TAPDA_3.log   "$D/logs/Pavia.log"   --out "$D/R_Pavia.md"
cat "$D"/R_*.md

# 追加到实验记录
cat >> "$D/RESULTS.md" <<EOF
# RGS v2 selector=${SEL}  $(date)

$(cat "$D"/R_*.md)
EOF

echo "DONE -> $D"
