#!/usr/bin/env bash
# Totoro certify: ≤16 cores; Julia 4 threads / OpenBLAS 1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.juliaup/bin:${PATH}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export LATTE_WALL_REPS="${LATTE_WALL_REPS:-5}"
export LATTE_WALL_OUT="${LATTE_WALL_OUT:-$ROOT/docs/dev-log/evidence/2026-09-23-latte-off-on-wall}"
mkdir -p "$LATTE_WALL_OUT"
SHA=$(git rev-parse --short HEAD)
META="$LATTE_WALL_OUT/meta_${SHA}.txt"
{
  echo "sha=$SHA"
  echo "full_sha=$(git rev-parse HEAD)"
  echo "host=$(hostname -s)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nproc=$(nproc)"
  echo "load=$(cut -d' ' -f1-3 /proc/loadavg)"
  echo "julia=$(julia --version)"
  echo "JULIA_NUM_THREADS=$JULIA_NUM_THREADS"
  echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
  echo "REPS=$LATTE_WALL_REPS"
  echo "default_flip=NO"
} | tee "$META"
# Cap OS threads via taskset when available (16 cores)
LOG="$LATTE_WALL_OUT/wall_${SHA}.log"
CMD=(julia --project=. bench/latte_gap_retime/off_on_wall.jl)
if command -v taskset >/dev/null 2>&1; then
  CMD=(taskset -c 0-15 "${CMD[@]}")
fi
echo "CMD=${CMD[*]}" | tee -a "$META"
"${CMD[@]}" 2>&1 | tee "$LOG"
echo "done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$META"
