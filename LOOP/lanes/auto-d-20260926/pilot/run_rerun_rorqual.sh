#!/bin/bash
#SBATCH --account=def-snakagaw_cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-543%300
#SBATCH --output=logs/rerun-%a.out
module load julia/1.10.10
export JULIA_DEPOT_PATH=$HOME/projects/def-snakagaw/snakagaw/julia_depot
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
cd $HOME/projects/def-snakagaw/snakagaw/auto-d-pilot
julia --project=. LOOP/lanes/auto-d-20260926/pilot/pilot.jl out-rerun/task-$SLURM_ARRAY_TASK_ID.csv 1 list $SLURM_ARRAY_TASK_ID LOOP/lanes/auto-d-20260926/pilot/rerun_list_rorqual.csv
