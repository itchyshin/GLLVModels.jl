#!/usr/bin/env bash
# #323 Track A full run (Claude lane true-parity-20260924). Shinichi: "go Track A" (2026-09-24).
# Follows docs/dev-log/after-task/2026-09-14-issue-323-totoro-launch-pack.md Runner steps 1-5a.
# Deviation (recorded): R_LIBS_USER isolated under the run dir so install_deps never writes the shared user R library.
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 MAKEFLAGS=-j1
T0=$(date +%s)
export GLLVM_ROOT=~/gllvmodels-track-a-20260924/GLLVModels.jl TRACK=A RECEIPT_STAMP=$(date +%Y%m%d-%H%M%S)
export R_LIBS_USER=~/gllvmodels-track-a-20260924/r-deps; mkdir -p "$R_LIBS_USER"
JL="$HOME/.juliaup/bin/julia +1.10.12"
step(){ echo "=== STEP $1 $(date -Is) elapsed_min=$(( ($(date +%s)-T0)/60 ))"; }
cd "$GLLVM_ROOT" && git fetch -q origin 94a7b56f9d2ae9934e5f2019f5e398eb82f686d5 && git checkout -q --detach 94a7b56f9d2ae9934e5f2019f5e398eb82f686d5 && echo "GLLVM_HEAD $(git rev-parse HEAD)"
step 1; ( cd .unlazy/r-source && echo "R_SOURCE $(git rev-parse HEAD)" )
step 2a-deps; Rscript -e 'options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/2026-08-31")); install.packages("remotes"); remotes::install_deps(".unlazy/r-source", dependencies=c("Depends", "Imports", "LinkingTo"), upgrade="never")' > ../step2a-deps.log 2>&1; echo "DEPS_EXIT=$?"; tail -3 ../step2a-deps.log
step 2b-prepare; python3 tools/core070_build_oracle.py prepare --repo .unlazy/r-source --destination .unlazy/r-archive 2>&1 | tail -2
step 2c-build; python3 tools/core070_build_oracle.py build --archive .unlazy/r-archive/gllvmTMB-core070.tar --source-receipt .unlazy/r-archive/source.json --destination .unlazy/r-build --r-binary "$(command -v R)" --timeout 1200 > ../step2c-build.log 2>&1; echo "BUILD_EXIT=$?"; tail -3 ../step2c-build.log
step 2d-verify; python3 tools/core070_build_oracle.py verify --destination .unlazy/r-build 2>&1 | tail -3; echo "VERIFY_EXIT=${PIPESTATUS[0]}"
step 3-julia-env; $JL --project=test/parity -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate(); Pkg.build("RCall"); Pkg.precompile()' > ../step3-julia.log 2>&1; echo "JULIAENV_EXIT=$?"; tail -3 ../step3-julia.log
mkdir -p .unlazy/core070-aghq/oracle-receipts .unlazy/core070-aghq/oracle-source
cp .unlazy/r-build/build.json .unlazy/core070-aghq/oracle-receipts/build.json; cp .unlazy/r-archive/source.json .unlazy/core070-aghq/oracle-source/source.json
step 4-env
export LD_PRELOAD="$($JL -e 'print(abspath(joinpath(Sys.BINDIR, "..", "lib", "julia", "libunwind.so.8")))')"
export GLLVM_PARITY_TESTS=1 CORE070_PARITY_REQUIRED=1
export GLLVM_PARITY_RECEIPT_DIR="$GLLVM_ROOT/.unlazy/totoro-parity-receipts-${RECEIPT_STAMP}"
export R_LIBS="$GLLVM_ROOT/.unlazy/r-build/library" GLLVM_PARITY_R_LIBS="$GLLVM_ROOT/.unlazy/r-build/library"
export GLLVM_PARITY_R_SOURCE_PIN="$R_LIBS/gllvmTMB/CORE070_SOURCE_PIN.toml" R_HOME="$(R RHOME)"
export LD_LIBRARY_PATH="$(R RHOME)/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$GLLVM_PARITY_RECEIPT_DIR"; echo "RECEIPT_DIR $GLLVM_PARITY_RECEIPT_DIR"
step 5a-runparity; $JL --project=test/parity test/parity/runparity.jl > "$GLLVM_PARITY_RECEIPT_DIR/runparity.log" 2>&1; echo "RUNPARITY_EXIT=$?"; tail -15 "$GLLVM_PARITY_RECEIPT_DIR/runparity.log"
step done; echo "WALL_MIN $(( ($(date +%s)-T0)/60 ))"; echo "TRACKA_DONE $(date -Is)"
