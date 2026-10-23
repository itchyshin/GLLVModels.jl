#!/bin/bash
# Gate: explicit-K fits give bit-identical logLik on this branch and on the pre-lane source (commit 4b2cd7c97 = origin/main src).
set -e
BASE=${BASE:-/private/tmp/claude-503/-Users-z3437171-Dropbox-Github-Local-glmmTMB/4f055e97-b708-44a2-bba7-707dda5c4eff/scratchpad/export-4b2cd7c97}
[ -d "$BASE/src" ] || { mkdir -p "$BASE"; git archive 4b2cd7c97 | tar -x -C "$BASE"; }
cp Manifest.toml "$BASE/Manifest.toml"   # identical dependency versions on both sides
CODE='using GLLVModels, Distributions, Random
Random.seed!(5); p, n = 6, 80
Yp = [rand(Poisson(exp(1.0 + 0.5*randn()))) for t in 1:p, s in 1:n]
Yb = [rand(Bernoulli(0.4)) for t in 1:p, s in 1:n]
Yg = randn(p, n)
for (Y, F) in ((Yp, Poisson()), (Yb, Binomial()), (Yg, Normal())), k in (1, 2)
    println(repr(GLLVModels._loglik(fit_gllvm(Y; family = F, K = k))))
end'
export OPENBLAS_NUM_THREADS=1
a=$(julia --project=. -e "$CODE" 2>/dev/null) || { echo "BRANCH FAILED TO RUN"; exit 1; }
b=$(cd "$BASE" && julia --project=. -e "$CODE" 2>/dev/null) || { echo "BASELINE FAILED TO RUN"; exit 1; }
[ -n "$a" ] && [ "$a" = "$b" ] && echo "EXPLICIT-K-IDENTICAL ($(echo "$a" | wc -l | tr -d ' ') fits)" || { echo "DIFF"; paste <(echo "$a") <(echo "$b"); exit 1; }
