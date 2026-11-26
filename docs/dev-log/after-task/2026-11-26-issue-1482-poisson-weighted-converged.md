# After-task: #1482 weighted Poisson converged vs continuation BFGS

**Branch:** `cursor/1126fx-1482-1e2c`

## Change

- `_bfgs_continuation` accepts optional `conv_test` (default `Optim.converged`).
- `fit_poisson_gllvm` passes `conv_test = r -> Optim.converged(r) && _nb1_grouped_g_met(r, g_tol)` for default L-BFGS + continuation and explicit optimizers.

## Verification

- `test/test_issue_1482_poisson_weighted_converged.jl` (issue repro, seed 777).
- `test/test_convergence_553_554.jl` 23/23.
- `test/test_optimizer_selection.jl` 62/62.

## Rose

Julia-only `converged` flag change on Laplace Poisson; no API or threshold edits; estimates unchanged on passing repro (non-convergence only).