# Ordinal warm-start warning test proposal, 2026-11-04

Base f34d5c175773729cb5a8f228184b56f877234f1a. Separate local branch codex/p1-ordinal-warning-test-20261104; no engine, API, tolerance, receipt or scoreboard change.

Removed the test's unsupported requirement that an AGHQ final fit must remain above loading 8 after its Laplace warm start emitted a warning. The positive AGHQ warning assertion and both Laplace/covariate loading checks remain unchanged. The source permits a successful AGHQ step to move below the threshold. Independent pre-edit and actual-diff reviews found no blocker to this limited contract correction.

Recorded CI failure: run 41743394305, job 126290256683, Julia 1.13.3, test_convergence_sweep.jl:108 observed maximum loading 5.208452513947484. That environment's full BLAS configuration was not recorded.

Focused Totoro Julia 1.13.3, four threads, one BLAS thread, same retained source/environment: original file 21/21 passed, exit 0; proposed file 20/20 passed, exit 0. Therefore the failing hosted result was not reproduced, and this is not an observed red-to-green runtime comparison. One invalid assertion was intentionally removed; its numerical threshold was not widened elsewhere.

Runtime download used an owned JULIAUP_DEPOT_PATH and a writable dependency overlay with read-through shared caches. No shared runtime configuration was changed. The Juliaup path behavior follows its official README: https://github.com/JuliaLang/juliaup#path-used-by-juliaup .

Remote evidence directory: /home/snakagaw/hsq_work/codex-p1-ordinal-warning-20261104, files baseline-1133.log, patched-1133.log, comparison-1133.json and associated exits. A mutation that suppressed warnings only around G_aghq failed exactly the retained AGHQ log assertion: 19 passed, 1 failed, 20 total; exit 1. This verifies the positive warning check remains effective. No full-suite, hosted-CI or parity pass is claimed.
