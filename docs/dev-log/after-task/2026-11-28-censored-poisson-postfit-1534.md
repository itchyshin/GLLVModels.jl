# After-task — CensoredPoissonFit post-fit (#1534)

- **Branch:** `cursor/1128b-1534`
- **Cause:** `CensoredPoissonFit` had `_nparams` only; no `predict`/`getLV`/`residuals`/`simulate` dispatches.
- **Fix:** Laplace post-fit mirroring `PoissonFit` for uncensored-at-`Y` modes; uncensored Poisson structural `simulate`.
- **Tests:** `test/test_issue_1534_censored_poisson_postfit.jl` — 8/8 on Julia 1.13.4.
- **Rose:** Julia post-fit claim only; not twin Δ.

Tag: `cursor-cloud-1128`