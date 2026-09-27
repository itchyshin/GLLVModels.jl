# After-task: temporal source alone at gllvmTMB P1, slice 1 (2026-09-27)

Branch `claude/temporal-slice1` from `origin/main` @ `824d22a4b`; draft PR #543.
Design spec: `docs/design/temporal-port-spec.md` (draft PR #535, head `5da58438f`)
and its independent review.

## 1. Goal

Port slice 1 of gllvmTMB P1's temporal source (`9539352f6`, 0.7.1): the
temporal source by itself through a separate door, with all three trait modes, AR1 and
OU, the exact Gaussian marginal, R-ordered parameters, the extractor and the four
helpers. Twin R's P1 test blocks from literal, hash-guarded receipts.

## 2. What I did this session

- New `src/temporal.jl` (constructors, pre-pass, `TemporalContractError`),
  `src/temporal_likelihood.jl` (exact NLL, dense per-series Cholesky, AR1 integer
  powers), `src/temporal_fit.jl` (`fit_temporal_gllvm`, `TemporalGaussianFit`),
  `src/temporal_methods.jl` (`extract_temporal`, `forecast_temporal`,
  `profile_temporal` with the `TMB::tmbprofile` port, `bootstrap_temporal`,
  `compare_temporal`, `simulate`, in-sample `predict`, `getLV`, refusals).
- Installed gllvmTMB P1 into a temporary library from a detached worktree and
  wrote `test/fixtures/temporal_p1/generate_temporal_p1.R`, which records R's
  `opt$par` names and values, objective, gradient, report, `extract_temporal`,
  forecasts, profiles with full traces, AIC tables and a 200-draw bootstrap into
  TOML receipts. A `cross` stage evaluates R's objective at Julia's optima.
- Five test files, all tagged `# gllvm-parity-tag: P1`, registered in
  `test/runtests.jl`. Reference page `docs/src/temporal.md`; API and
  low-level manual entries; three stale pages reconciled (spec 3.7).

## 3a. Decisions and Rejected Alternatives

- **Parameter order follows R, not the spec.** R's `names(opt$par)` at P1 is
  `b_fix, log_sigma_eps, theta_temporal_time, theta_temporal_rr,
  theta_temporal_diag`; spec 3.3 put `log_sigma_eps` last. Recorded in
  `docs/dev-log/decisions/2026-09-27-temporal-slice1-likelihood.md`; tests
  assert R's recorded name vector.
- **Between-optima parameters are compared at a Newton-polished R point.** R's
  `nlminb` stops with gradients up to 1.8e-3 even at `rel.tol = 1e-14`; one Newton
  step from R's recorded gradient (checked equal to Julia's to 1e-8) brings the
  agreement to 1e-10. The raw gaps to R's stopped point are also asserted at 1e-4
  (measured max 1.9e-5 relative).
- **Bootstrap seeds use `Random.Xoshiro`, not `StableRNG`.** StableRNGs is a
  test-only dependency and `Project.toml` was out of bounds; the reproducibility
  promise is within a Julia version.
- **Line-search fallback.** Hager-Zhang asserts when a trial step hits a singular
  covariance (boundary fits); the fit reruns with cubic backtracking.
- Rejected: reusing `fit_gaussian_sources` (fixed `C`, cannot carry a parametric
  kernel, spec 3.1).

## 4. Files Touched

`src/temporal.jl`, `src/temporal_likelihood.jl`, `src/temporal_fit.jl`,
`src/temporal_methods.jl`, `src/GLLVModels.jl` (includes, exports);
`test/test_temporal_{api,oracles,fit_receipts,helpers}.jl`, `test/runtests.jl`,
`test/fixtures/temporal_p1/*`; `docs/src/{temporal,api,low-level-reference,gllvmtmb-parity}.md`,
`docs/make.jl`, `r/README_bridge.md`, `ROADMAP.md`, `CHANGELOG.md`, the decision note,
this report and `docs/dev-log/check-log.md`. Not touched: `_laplace_mode`,
`src/families/mixed.jl`, `grouped_dispersion.jl`, `model_selection.jl`, `cv.jl`,
`src/formula.jl`, `Project.toml`.

## 5. Checks Run

- Per file, `julia +1.10` and `+1.13 --project=. -e 'using Test, GLLVModels; include(...)'`:
  api 81/81, oracles 258/258, fit receipts 738/738 (after review: raw-gap and
  getLV receipt assertions), helpers 129/129 plus latent scores 8/8, on both versions.
  getLV scores vs R's `report$z_temporal_state` at R's coordinates: 1.3e-15. The full suite was not run (instruction).
- `Test.detect_ambiguities(GLLVModels)`: 0.
- Local Documenter build (`docs/make.jl --local`): exit 0;
  `tools/check_reader_surface.py` source and rendered: pass.

## 6. Tests of the Tests

- The dense oracle is a second implementation (own unpack, float powers,
  `Distributions.MvNormal`) and agrees with the structured NLL to 2.3e-13.
- R's native objective and gradient at 32 fixed points, including the extreme
  `theta = ±20` (AR1) and `±45` (OU), agree to 2.3e-13 and 1.8e-12.
- Receipt files and every response column are sha256-guarded.
- The `iid-Psi` alternative covariance is rejected by more than 1e-4 (oracles.R:375).

## 7. Issue Ledger

No GitHub issue numbers. Case-map rows (`temporal/*`, PR #526) are not edited here.

## 8. Consistency Audit

Docstrings, API page, low-level page, reference page, CHANGELOG and the three
reconciled pages describe the same scope: temporal source alone; no `unit` /
`unit_obs`, cross-source cells, `traits()`, offsets, new-data prediction or bridge.

## 9. What Did Not Go Smoothly

- My first simulated panels left `sigma_eps` or `psi` on the boundary; I enlarged
  them (20 series, `sigma = 0.6`) and added an indep-generated panel.
- `bootstrap_temporal` in R failed for every refit when the fit came from a helper
  function, because `update()` re-evaluates the saved call; the generator now uses a
  literal top-level call and asserts the refits ran.
- `g_tol = 1e-9` sits below the round-off floor on one cell; receipts fit at 1e-8.

## 10. Known Residuals

- On the R fixture `profile_ar1` with latent-unique AR1 the likelihood is
  multimodal. R converged to a stationary local optimum (tight-run gradient 8.7e-7,
  objective 47.5671); Julia reached a different, higher one on the `psi = 0` boundary
  (theta_diag near -179, -27.5, -107 and one loading near 0; objective 47.2888, confirmed
  by R's `fn`). This is not an R early stop. Recorded as a finding in the test.
  (Revised after independent review; the first version called it an early stop.)
- Not twinned: ar1-methods.R:59 `select_lv` and VA refusals (Julia `select_lv`
  takes no temporal formula; the door has no integration keyword), the
  `update()` S3 row (refit is internal), `extract_ordination` for the latent cell,
  and the wide-form assertions of engine.R:119.
- Slice 2 and later: `unit` / `unit_obs` composition and its twins, the `sigma_eps`
  suppression rule, the `formula.jl` hook, the cross-source cells (Q1), the bridge.
- Case-map `evidence` fields for the eight `temporal/*` rows are still empty.

## 11. Team Learning

Record R's gradient at its optimum in every receipt: it separates an optimiser
stopping point from a likelihood mismatch without widening a tolerance.

## 12. Cross-Product Coverage

Modes indep, dep, latent, latent-unique × AR1, OU; unreplicated (all) and
replicated (indep, dep, latent-unique, AR1). Helpers on unreplicated indep
(forecast, profile, bootstrap) and all modes and workflows (compare).

Rose audit verdict: not yet run; this report lists the open items above.
