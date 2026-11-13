#!/bin/bash
# Launcher for the namespace-3 (D5) batch: one R fit, then one Julia run, sequential, single threads.
#
# usage: GLLVMTMB_ORACLE_LIB=<frozen P1 library> RUN_COMMIT=<40-hex commit of the exported tree> \
#        bash tools/core070_namespace_3_d5_run.sh <tree> <julia-env> <outdir>
#   <tree>       the GLLVModels tree the Julia half runs on (a `git archive` export of RUN_COMMIT is fine)
#   <julia-env>  a Julia project that has <tree> developed (Pkg.develop(path = <tree>))
#   <outdir>     created here; must not already hold results (one run, no reseeding)
# Writes run-commit.json: {"glvmodels_commit": RUN_COMMIT, "dirty": [], ...} next to the results.
# Applies no bar; tools/core070_namespace_3_p1_receipts.py reads the directory.
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: see header" >&2; exit 2; }
TREE=$1; ENVDIR=$2; OUT=$3
[[ "${RUN_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || { echo "RUN_COMMIT must be a 40-hex commit" >&2; exit 2; }
[ -n "${GLLVMTMB_ORACLE_LIB:-}" ] || { echo "GLLVMTMB_ORACLE_LIB must be set" >&2; exit 2; }
[ ! -e "$OUT/julia-results.json" ] || { echo "$OUT already holds results; one run only" >&2; exit 2; }
mkdir -p "$OUT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1
Rscript "$TREE/tools/core070_namespace_3_d5_batch.R" "$OUT"
julia --project="$ENVDIR" "$TREE/tools/core070_namespace_3_d5_batch.jl" "$OUT"
printf '{"glvmodels_commit": "%s", "dirty": [], "tree_export": "git archive of the commit, no .git directory", "threads": {"julia": 1, "openblas": 1, "omp": 1}}\n' \
  "$RUN_COMMIT" > "$OUT/run-commit.json"
echo RUN-COMPLETE
