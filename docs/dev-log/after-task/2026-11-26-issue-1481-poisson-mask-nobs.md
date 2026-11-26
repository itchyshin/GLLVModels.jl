# After-task: #1481 Poisson mask nobs/bic

## Goal

`nobs(fit, Y)` and `bic(fit, Y)` after a Poisson fit with `mask=` must count only
cells that entered the likelihood, without re-passing the mask.

## Change

- `PoissonFit.obs_mask` retains the resolved training mask from `fit_poisson_gllvm`.
- `StatsAPI.nobs(fit, Y)` consults `_fit_stored_obs_mask` when `mask` is omitted
  (Poisson `obs_mask`; AGHQ integration `data.mask` when not all-true).

## Verification

- `test/test_issue_1481_poisson_mask_nobs.jl`: 7/7
- `test/test_statsapi.jl`: 74/74 (unchanged expectations)

## Rose

Julia-only postfit accounting fix; no parity or capability-matrix promotion.