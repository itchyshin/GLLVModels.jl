#!/bin/bash
# NB grid re-run on the corrected NB2 kernel (#521), nibi. Approved by Shinichi 2026-09-28 ("1-3 as you
# recommended": DRAC, about 5,500 core-h; plan and pre-run in ../nb-rerun-plan.md). Code: #518 head
# be55464e0 (main with #521 merged, plus the select_lv lane code), recorded in DEPLOYED_SHA.
# Submit after smoke_nb_rerun_nibi.sh, with --dependency=afterok:<smoke job id>.
#SBATCH --account=def-snakagaw_cpu
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-943%300
#SBATCH --output=logs-nbrerun/task-%a.out
module load julia/1.10.10
export JULIA_DEPOT_PATH=$HOME/projects/def-snakagaw/snakagaw/julia_depot
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
cd $HOME/projects/def-snakagaw/snakagaw/auto-d-nbrerun
L=LOOP/lanes/auto-d-20260926/pilot
row=$(awk -F, -v t=$SLURM_ARRAY_TASK_ID 'NR == t + 1 {print $2, $3}' $L/nb_rerun_tasks.csv)
set -- $row
julia --project=. $L/pilot.jl out-nbrerun/task-$SLURM_ARRAY_TASK_ID.csv 1 listrange $1 $2 $L/nb_rerun_list.csv
