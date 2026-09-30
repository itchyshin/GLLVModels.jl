#!/bin/bash
# Precompile, then fit the cheapest row (nb, n 30, p 10, K 1, rep 200) end to end before the array starts.
#SBATCH --account=def-snakagaw_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs-nbrerun/smoke.out
module load julia/1.10.10
export JULIA_DEPOT_PATH=$HOME/projects/def-snakagaw/snakagaw/julia_depot
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
cd $HOME/projects/def-snakagaw/snakagaw/auto-d-nbrerun
L=LOOP/lanes/auto-d-20260926/pilot
julia --project=. -e 'using Pkg; Pkg.precompile()' || exit 1
julia --project=. $L/pilot.jl out-nbrerun/smoke.csv 1 listrange 4800 4800 $L/nb_rerun_list.csv || exit 1
test $(wc -l < out-nbrerun/smoke.csv) -ge 2
