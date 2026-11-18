#!/bin/bash
# usage: run_pair.sh dsA coreJA coreRA dsB coreJB coreRB
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
cd ~/hsq_work/claude-n1b-20261103
J=/home/snakagaw/.juliaup/bin/julia; P=$HOME/hsq_work/claude-n1-20261103/src
run() { ds=$1; cj=$2; cr=$3
  taskset -c $cj nice $J --project=$P scripts/orig_j.jl $ds $PWD out/p$ds full > logs/J_$ds.log 2> logs/J_$ds.err &
  while ! grep -q "Y written" logs/J_$ds.log 2>/dev/null; do sleep 5; done
  taskset -c $cr nice Rscript scripts/paired_r.R $ds $PWD out/p$ds > logs/R_$ds.log 2>&1
}
run $1 $2 $3 &
run $4 $5 $6 &
wait
