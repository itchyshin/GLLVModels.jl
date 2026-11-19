# After-task: Beta grouped probit / cloglog default curvature (#1156)

**Date:** 2026-11-19. **Branch:** `cursor/beta-grouped-default-hessian-1156-8da0`. **Issue:** #1156 (refs only).

## Goal

`fit_gllvm(Y; family = Beta(), link = ProbitLink() | CLogLogLink())` must return a fit, not `loglik = -Inf`, `iterations = 0`.

## Change

- `src/families/grouped_dispersion.jl`: the `hessian` keyword of `fit_beta_gllvm_grouped`, `fit_beta_gllvm_grouped_cov` and `beta_grouped_marginal_loglik_laplace` defaults to `_default_hessian(Beta(1.0, 1.0), link)` (the shared-precision fitter's default) instead of the constant `:observed`. Logit keeps `:observed`. Docstrings updated.
- `test/test_grouped_dispersion_beta_gamma.jl`: new testset. Both links finite, `iterations > 0`, converged, stored `hessian === :fisher`, loglik equal to the explicit `hessian = :fisher` fit, the covariate fitter and the marginal likewise; the logit default is still `:observed`; explicit `:observed` with probit still throws.

## Evidence

See `docs/dev-log/check-log.md` entry of 2026-11-19. New testset: 8 fail and 1 error on the old source, 12 of 12 pass on the fix.

## Not done / limits

- The choice of `:fisher` as default for probit and cloglog is the shared fitter's existing rule and the issue's first suggestion. It is not an R-parity statement: CRAN `gllvm` uses the observed curvature (#1229, design question, untouched).
- The bare `catch` in the fitter objective that hid the error (#774, #1049) is untouched.
- Gamma, NB1 and NB2 grouped fitters have the same constant default for non-log links; not changed.
- Full `Pkg.test()`, JET and Aqua were not run. No tolerance or expectation was changed. Rose verdict: not run.
