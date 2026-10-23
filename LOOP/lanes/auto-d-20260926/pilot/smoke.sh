#!/bin/bash
#SBATCH --account=def-snakagaw_cpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/smoke.out
module load julia/1.10.10
export JULIA_DEPOT_PATH=$HOME/projects/def-snakagaw/snakagaw/julia_depot
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4
cd $HOME/projects/def-snakagaw/snakagaw/auto-d-pilot
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. LOOP/lanes/auto-d-20260926/pilot/pilot.jl out/smoke.csv 1 pre
