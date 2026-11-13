# After Task: derive the Gaussian observation mask from missing (#1107)

## Goal

Stop `fit_gaussian_gllvm`, `fit_gllvm(Normal())`, and
`fit_gaussian_pervar_gllvm` from throwing `MethodError` when `Y` contains
`missing` and no mask is passed. Derive the mask, and say so once.

## Implemented

Shinichi's decision of 10 Nov 2026: derive the mask rather than refuse, and
emit one informational message. `_derive_obs_mask` (in
`src/families/gaussian_pervar.jl`) returns an explicit mask unchanged after
`_resolve_obs_mask`, returns `nothing` for a `Float64` matrix or a union
matrix with no actual `missing`, and otherwise logs

`"$caller: derived the observation mask from missing entries in Y"`

with `@info ... maxlog=1` and returns `observed_mask(Y)`.
`fit_gaussian_gllvm` derives the mask before the exact-path gate, so the
existing masked marginal is used. `fit_gllvm(Normal())` reaches that fitter
through the trait-intercept design. `fit_gaussian_pervar_gllvm` stays on the
complete-data EM path unless the derived or explicit mask has a `false`;
incomplete data uses L-BFGS on the observed-subvector marginal, with
intercepts or `X` coefficients profiled by GLS. `src/families/laplace.jl`
and `src/families/variational.jl` are not edited.

## Mathematical Contract

Complete-data Gaussian likelihoods are unchanged (Tipping & Bishop 1999
PPCA marginal; the shared-σ closed form in `src/fit.jl`).

With a mask, the per-species marginal at site `s` with observed index set
`obs` is the normal density of `y_s[obs]` with mean `(X_s β)[obs]` and
covariance `Λ[obs,:] Λ[obs,:]' + diag(φ²[obs])`. `β` is the GLS estimate on
the observed cells. The shared-residual masked route is the existing exact
marginal, not a new parameterisation.

## Files Changed

- `src/families/gaussian_pervar.jl` — `_derive_obs_mask`, observed-subvector
  log-likelihood, masked fitter, `mask` on the public fitter and log-likelihood
- `src/families/aghq_gaussian_fit.jl` — derive the mask before the exact path
- `src/families/fit_gllvm.jl` — `Normal()` docstring sentence
- `test/test_fix_1107.jl`, `test/runtests.jl`
- `docs/src/api.md`, `docs/src/response-families.md`, `docs/src/changelog.md`
- `README.md`, `CHANGELOG.md`, `AGENTS.md` phase snapshot
- `docs/dev-log/check-log.md`, this report

## Tests Added

`test/test_fix_1107.jl` (27 assertions on the fixed tree).

Tests of the tests:

- The file fails on current main `3f6d74ecb`: 3 passed, 2 failed, 4 errored.
  The three issue calls throw `MethodError` (`ppca_init` `symmetric_type`,
  trait-intercept `convert` of `Missing`, per-species `copyto!`).
- Independent calculation: an all-true mask matches the complete-data
  per-species factorisation (`atol = 1e-8`); a masked cell's value does not
  enter the log-likelihood; the reported per-species `loglik` matches
  `gaussian_pervar_marginal_loglik` at the returned `Λ`, `φ²`, and `β`.
- Failure path: a mask that marks a `missing` cell observed throws
  `ArgumentError` on both fitters. Complete `Float64` data logs nothing.
  Two derived fits under one `SimpleLogger` log the message once.

## Benchmark Numbers

N/A — no hot-path change. A `Float64` matrix returns before any missing
scan. The new path runs only when `Y` actually contains `missing`.

## R-Parity Verdict

Parity: N/A — mask derivation and the existing masked Gaussian marginal.
The complete-data likelihood, packing, and Λ orientation are unchanged.
No gllvmTMB cell was re-run.

## JET / Allocs / Aqua Verdicts

- JET: not run (focused likelihood tests only; full suite not run)
- Allocs: not run
- Aqua: not run (`Project.toml` and exports unchanged)

## Checks Run

Julia 1.10.13, `JULIA_NUM_THREADS=1`.

```sh
julia --project=. -e 'include("test/test_fix_1107.jl"); include("test/test_ledermann_pervar_k.jl"); include("test/test_mask_missing_763.jl"); include("test/test_gaussian_pervar.jl")'
```

- `test_fix_1107.jl`: 27 passed, 0 failed, 0 errored, 35.4s
- `test_ledermann_pervar_k.jl`: 6 passed, 1.7s
- `test_mask_missing_763.jl`: 47 passed, 54.1s
- `test_gaussian_pervar.jl`: 14 passed, 1.9s

Same regression file on main `3f6d74ecb`: 3 passed, 2 failed, 4 errored, 33.5s.

Full `Pkg.test()` / `test/runtests.jl` was not run.

## Consistency Audit

```sh
rg "missing" README.md CLAUDE.md docs/src/api.md docs/src/response-families.md
```

README previously listed response-missing masks only for GLM Laplace rows.
That bullet now names the ordinary and per-species Gaussian fitters.
`docs/src/response-families.md` and `docs/src/api.md` state the derivation
and the one `@info`. The 2026-09-27 check-log sentence that missing cells
`MethodError` on the Gaussian path is left as history. `Project.toml` stays
`0.3.0`. No speed or machine-precision claim was edited.

## GitHub Issue Maintenance

No issue comment. #1107 and #1108 have no comments. The draft PR links
#1107. #1108 is out of scope (variational dispatch) and is not edited.

## What Did Not Go Smoothly

`maxlog` is per call site and per logger. Both Gaussian callers share one
`@info` line, so one `SimpleLogger` records the sentence only from the first
fitter that derives a mask. A new logger records it again. `Test.TestLogger`
does not honour `maxlog`; the regression test uses `SimpleLogger`.

On main, `fit_gaussian_gllvm` with an explicit mask that marks a missing
cell observed did not throw (the AGHQ evaluator skipped the cell). The
public fitter now throws via `_resolve_obs_mask`. No existing test expected
the silent skip.

## Team Learning

A helper that only adds the notice, and leaves `_resolve_obs_mask` alone,
keeps the Poisson derivation silent and leaves the parallel #1108 lane a
smaller overlap.

## Remaining Risks

- Full suite, JET, Allocs, and Aqua were not run.
- The derived mask is not stored. `predict_missing` still needs
  `mask = observed_mask(Y)`.
- `lambda_constraint`, phylogenetic and diagonal random-effect blocks, and
  `X_lv` still raise the existing `ArgumentError` once a mask is present.
- A wrong-size explicit mask is now a `DimensionMismatch` from
  `_derive_obs_mask` before the older AGHQ size message. No public-fitter
  test matched that older string.
- Per-species incomplete data always uses L-BFGS, including `method=:em`.

## Known Limitations

Variational fitters (#1108) still do not accept `Matrix{Union{Missing,T}}`.
Structured Gaussian routes that already refuse `mask` still refuse it after
derivation. The per-species masked likelihood is the observed-subvector
normal, not an imputation of the missing cells.

## Next Command

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Rose Verdict

Rose verdict: PASS WITH NOTES — focused tests pass and fail on main as
required; full suite, JET, Aqua, and R parity were not run, and structured
Gaussian routes still refuse a mask.
