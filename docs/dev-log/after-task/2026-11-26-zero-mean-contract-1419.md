# After-task: zero-mean Gaussian contract (#1419, #1434, #1415)

**Claim:** Document the zero-mean `fit_gaussian_gllvm` contract, warn when trait
`|mean|/sd > 3` (once per session), point users to `fit_gllvm(Normal())`; no
silent centring, no `dof` change. Docs-only: masked Gaussian Pearson residuals
are `NaN` at masked cells while `fitted`/`predict` stay finite.

**Approved:** Shinichi, 26 Nov 2026 9:41 AM MT (relayed Brian).

**Files:** `src/fit.jl`, `src/families/aghq_gaussian_fit.jl` (docstring),
`docs/src/working-with-a-fit.md`, `test/test_zero_mean_contract.jl`,
`test/runtests.jl`, `docs/dev-log/check-log.md`.

**Tests:** `julia --project=. --startup-file=no -e 'include("test/test_zero_mean_contract.jl")'`
— 9 passed, 0 failed, 0 errored (~43s on Julia 1.13.4 VM).

**Skipped:** PR #1427 (silent centring) left untouched per instruction.

Rose verdict: PASS — docs match behaviour; warning is additive; no estimand or df change.