#!/bin/bash
# True-parity campaign driver (C3, C4). Signed itchyshin/GLLVModels.jl#684 item 4. Runs ON Totoro, from the lane
# directory, after env.sh in that directory has been written (it exports R_LIBS, GLLVMTMB_P1_LIB,
# GLLVMTMB_P1_SRC, JULIA_DEPOT_PATH, JLPROJ, GLLVM_JL_SHA, TPC, CAMPAIGN_DATA, CAMPAIGN_OUT).
#
#   cd ~/hsq_work/true-parity-campaign-20261002 && nohup bash launch_totoro.sh > logs/driver.log 2>&1 &
#
# What it does, in order:
#   1. waits while the 1-minute load average is above LOAD_LIMIT (150; the lab's hard ceiling is 150 cores);
#   2. generates every data file once (gen_data.R), recording its sha256;
#   3. runs one R job and one Julia job per cell, at most MAXJOBS (8) at a time, each under `timeout` at twice
#      its written estimate. A job that hits its timeout is STOPPED, not retried, and the row is reported
#      unbound by write_receipts.py ("Julia run not finished").
# Thread caps: OPENBLAS_NUM_THREADS=1, JULIA_NUM_THREADS=2, R single-threaded. At most 8 jobs x 2 = 16 cores.
set -u
LANE="${1:-$PWD}"; cd "$LANE"
source ./env.sh
mkdir -p logs data out
MAXJOBS=8; LOAD_LIMIT=150
while true; do
  L=$(awk '{print int($1)}' /proc/loadavg)
  [ "$L" -le "$LOAD_LIMIT" ] && break
  echo "$(date -u +%FT%TZ) load $L > $LOAD_LIMIT, waiting"; sleep 30
done
{ date -u +%FT%TZ; uptime; } > logs/start.txt

for c in gaussian poisson nb2 binomial ordinal temporal isdm crabs spider beetle fungi; do
  Rscript "$TPC/gen_data.R" "$c" . > "logs/gen_$c.log" 2>&1 || echo "GEN FAILED $c"
done

# cell : cap in seconds for the Julia job (twice the written estimate, except fungi: 300 s against a 2 min estimate); R jobs are capped at 600 s
CAPS_J="nb2:1440 beetle:600 fungi:300 ordinal:600 poisson:360 binomial:360 isdm:240 gaussian:180 temporal:180 crabs:180 spider:180"
run_one() {  # engine:cell:cap
  IFS=: read -r eng cell cap <<< "$1"
  t0=$(date +%s)
  if [ "$eng" = J ]; then
    timeout "$cap" julia --project="$JLPROJ" "$TPC/run_J.jl" "$cell" > "logs/J_$cell.log" 2>&1; rc=$?
  else
    timeout "$cap" Rscript "$TPC/run_R.R" "$cell" > "logs/R_$cell.log" 2>&1; rc=$?
  fi
  echo "$eng $cell rc=$rc wall=$(( $(date +%s) - t0 ))s cap=${cap}s $(date -u +%FT%TZ)" >> logs/jobs.log
}
export -f run_one
export LANE
{
  for kv in $CAPS_J; do echo "J:${kv%%:*}:${kv##*:}"; done
  for cell in nb2 ordinal beetle fungi poisson binomial isdm gaussian temporal crabs spider; do echo "R:$cell:600"; done
} | xargs -P "$MAXJOBS" -I{} bash -c 'run_one {}'
{ date -u +%FT%TZ; uptime; } > logs/end.txt
echo "ALL JOBS EXITED"
