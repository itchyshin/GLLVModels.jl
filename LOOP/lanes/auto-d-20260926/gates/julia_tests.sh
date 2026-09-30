#!/bin/bash
# Gate: select_lv + dispatcher test files pass on this branch. Prints JULIA-TESTS-PASS only if all pass.
set -e
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4
for f in test_model_selection.jl test_binomial_ridge.jl test_binomial_fit.jl test_laplace_grad.jl test_saturation_health.jl test_fit_gllvm.jl test_unified_api.jl test_formula_sources.jl test_known_sentinel_defects.jl; do
  julia --project=. -e "using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra, StatsModels, DataFrames; include(\"test/$f\")" > /dev/null 2>&1 || { echo "FAILED $f"; exit 1; }
done
echo JULIA-TESTS-PASS
