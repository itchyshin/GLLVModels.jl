#!/usr/bin/env bash
# OWED 2 re-run on Totoro: NATIVE-06-NB2 and NATIVE-10-STUDENT, Julia 1.10.12, PR #475 head.
# Shinichi: "run the Totoro re-run" (2026-09-24). Modelled on trackA.sh (Track A, same day).
# Reuses Track A's verified oracle build and R deps read-only; never writes into that folder.
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=2 MAKEFLAGS=-j1
T0=$(date +%s)
RUN=~/gllvmodels-owed24-20260924
TA=~/gllvmodels-track-a-20260924/GLLVModels.jl
BRANCH=claude/owed-2-4-decisions-20260924
EXPECT_HEAD="$1"
export R_LIBS_USER=~/gllvmodels-track-a-20260924/r-deps
JL="$HOME/.juliaup/bin/julia +1.10.12"
step(){ echo "=== STEP $1 $(date -Is) elapsed_min=$(( ($(date +%s)-T0)/60 ))"; }
mkdir -p "$RUN" && cd "$RUN" || exit 2
step 1-clone
git clone -q --depth 1 -b "$BRANCH" https://github.com/itchyshin/GLLVModels.jl.git GLLVModels.jl || { echo "CLONE_FAILED"; exit 3; }
cd GLLVModels.jl
HEAD=$(git rev-parse HEAD); echo "GLLVM_HEAD $HEAD"
[ "$HEAD" = "$EXPECT_HEAD" ] || { echo "HEAD_MISMATCH expected $EXPECT_HEAD"; exit 4; }
step 2-oracle-receipts
mkdir -p .unlazy/core070-aghq/oracle-receipts .unlazy/core070-aghq/oracle-source
cp "$TA/.unlazy/r-build/build.json" .unlazy/core070-aghq/oracle-receipts/build.json
cp "$TA/.unlazy/r-archive/source.json" .unlazy/core070-aghq/oracle-source/source.json
python3 tools/core070_build_oracle.py verify --destination "$TA/.unlazy/r-build" 2>&1 | tail -1; echo "VERIFY_EXIT=${PIPESTATUS[0]}"
step 3-julia-env
$JL --project=test/parity -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate(); Pkg.build("RCall"); Pkg.precompile()' > ../step3-julia.log 2>&1; echo "JULIAENV_EXIT=$?"; tail -3 ../step3-julia.log
step 4-env
export LD_PRELOAD="$($JL -e 'print(abspath(joinpath(Sys.BINDIR, "..", "lib", "julia", "libunwind.so.8")))')"
export GLLVM_PARITY_TESTS=1 CORE070_PARITY_REQUIRED=1 CORE070_PARITY_CASE_IDS=NATIVE-06-NB2,NATIVE-10-STUDENT
export GLLVM_PARITY_RECEIPT_DIR="$RUN/receipts-$(date +%Y%m%d-%H%M%S)"
export R_LIBS="$TA/.unlazy/r-build/library" GLLVM_PARITY_R_LIBS="$TA/.unlazy/r-build/library"
export GLLVM_PARITY_R_SOURCE_PIN="$R_LIBS/gllvmTMB/CORE070_SOURCE_PIN.toml" R_HOME="$(R RHOME)"
export LD_LIBRARY_PATH="$(R RHOME)/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$GLLVM_PARITY_RECEIPT_DIR"; echo "RECEIPT_DIR $GLLVM_PARITY_RECEIPT_DIR"
step 5-runparity
$JL --project=test/parity test/parity/runparity.jl > "$GLLVM_PARITY_RECEIPT_DIR/runparity.log" 2>&1; echo "RUNPARITY_EXIT=$?"; tail -8 "$GLLVM_PARITY_RECEIPT_DIR/runparity.log"
step done; echo "WALL_MIN $(( ($(date +%s)-T0)/60 ))"; echo "OWED24_DONE $(date -Is)"
