#!/bin/bash
#SBATCH --account=def-snakagaw_cpu
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-7%4
#SBATCH --job-name=gllvm-fw
#SBATCH --output=logs/gllvm-fw-%A_%a.out
#SBATCH --error=logs/gllvm-fw-%A_%a.err
#
# Phase B first-wave GLLVM (7 cells). Skips Latte + Totoro-banked phylo/spatial smokes.
set -euo pipefail
module load StdEnv/2023 julia/1.11.3 || module load julia/1.12.5 || module load julia/1.10.10
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/julia_depot_fir}"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs bench/results docs/dev-log/evidence/2026-09-23-speed-firstwave-drac
OUT="docs/dev-log/evidence/2026-09-23-speed-firstwave-drac/gllvm_task${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.tsv"
export SPEED_ARRAY_INDEX="$SLURM_ARRAY_TASK_ID"
export GLLVM_FIRSTWAVE_OUT="$OUT"
export GLLVM_FIRSTWAVE_REPS=3
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. bench/speed_board_firstwave.jl
