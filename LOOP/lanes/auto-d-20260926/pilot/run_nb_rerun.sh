#!/bin/bash
# NB grid re-run on the corrected NB2 kernel (#521). DO NOT SUBMIT until (1) #521 is merged and
# the DRAC checkout is on that main, and (2) Shinichi has approved the estimate in
# ../nb-rerun-plan.md (D-287). Regenerate the map first with the measured ratio:
#   python3 nb_rerun_tasks.py <ratio> 3
# then set --time to about 1.5x the largest est_secs in nb_rerun_tasks.csv and --array to 1-<ntasks>.
#SBATCH --account=def-snakagaw_cpu
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-943%300
#SBATCH --output=logs-nbrerun/task-%a.out
module load julia/1.10.10
export JULIA_DEPOT_PATH=$HOME/projects/def-snakagaw/snakagaw/julia_depot
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
cd $HOME/projects/def-snakagaw/snakagaw/auto-d-pilot
L=LOOP/lanes/auto-d-20260926/pilot
row=$(awk -F, -v t=$SLURM_ARRAY_TASK_ID 'NR == t + 1 {print $2, $3}' $L/nb_rerun_tasks.csv)
set -- $row
julia --project=. $L/pilot.jl out-nbrerun/task-$SLURM_ARRAY_TASK_ID.csv 1 listrange $1 $2 $L/nb_rerun_list.csv
