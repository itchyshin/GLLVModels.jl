# After-task: OrderedBeta / COMPoisson post-fit (#1473)

**Goal:** Wire `fitted`, `residuals`, and `simulate` for `OrderedBetaFit`; fix `fitted` for `COMPoissonFit` (`:response` routing).

**Claim:** `predict(..., type = :response)` aliases `:mean` (ordered-beta) or `:rate` (CMP); `OrderedBetaFit` joins `AnyGllvmFit`; residuals and simulate follow existing mixture / bootstrap patterns.

**Files:** `src/families/ordered_beta.jl`, `src/families/com_poisson.jl`, `src/postfit.jl`, `src/simulate_fit.jl`, `test/test_ordered_beta_postfit.jl`, `test/runtests.jl`.

**Tests:** `julia --project=. test/test_ordered_beta_postfit.jl` — 7 passed, 0 failed, ~11 s (Julia 1.10.14, cloud VM).

**R-parity:** Not exercised (native Julia post-fit surface only).

**Rose verdict:** PASS WITH NOTES — Julia-only post-fit wiring; no user-facing API rename; docstrings updated on `predict`/`residuals`/`simulate`.