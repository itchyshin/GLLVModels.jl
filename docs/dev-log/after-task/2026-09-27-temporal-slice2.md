# After-task: temporal source beside ordinary unit / unit_obs terms, slice 2 (2026-09-27)

Branch `claude/temporal-slice2`, stacked on `claude/temporal-slice1` @ `dafe3b9d2`
(draft PR #543). Design spec: `docs/design/temporal-port-spec.md` (draft PR #535, head
`af130f704`), its review, and the signed scope (vault D-300).

## 1. Goal

Port slice 2 of gllvmTMB P1's temporal source (`9539352f6`, 0.7.1): the temporal source
composed with ordinary `unit` / `unit_obs` terms (`indep`, `dep`, `latent`, and the
`(1 | g)` intercept) through `fit_temporal_gllvm`'s own `structure` argument, with R's
admission rules, the sigma_eps suppression rule, R's composed-model refusals for the
helpers, and R's measured parameter order. Twin the spec's slice-2 blocks red-first from
hash-guarded P1 receipts. No edit to `src/formula.jl`.

## 2. What I did this session

- `src/temporal.jl`: `TemporalOrdinaryTier`, `TemporalComposition`, the
  `composition` field on `TemporalSpec`, and `_temporal_composition` (nesting,
  partition, grouping, sigma_eps rule). The slice 1 "not implemented" refusal for
  ordinary terms is gone.
- `src/temporal_likelihood.jl`: `TemporalLayout` gains the `_B`, `_W` and `re_int`
  blocks in R's measured order and a fixed-sigma mode; the NLL and `_temporal_covariance`
  add the unit, unit_obs and random-intercept blocks and factor over independent row
  blocks (union-find over series, unit, unit_obs, intercept group).
- `src/temporal_fit.jl`: `unit` / `unit_obs` keywords; `Sigma_B`, `Sigma_W`,
  `sigma_re_int` fields; R's start scales for the ordinary tiers; the optimiser runs both
  LBFGS line searches and keeps the lower, then Newton-polishes to `g_tol`.
- `src/temporal_methods.jl`: `_temporal_other_tiers` returns R's tier names (so the four
  helpers refuse composed fits with R's classes); `simulate` redraws the ordinary tiers;
  `update(fit; ...)` (exported) replays the call with named overrides;
  `extract_ordination(fit; level)`.
- Temporary P1 install (3 minutes) from a detached worktree; new generator
  `test/fixtures/temporal_p1/generate_temporal_p1_slice2.R` writes `composed.toml`
  (25 fits over 7 datasets, the oracles.R:318 point) and, after
  `julia_optima_composed.jl`, `composed_cross.toml`.
- Tests: `test/test_temporal_composed.jl` (the R-block twins) and
  `test/test_temporal_composed_receipts.jl` (numeric parity), both tagged P1 and
  registered; one slice 1 assertion updated (ordinary terms are now admitted).
- Docs: `docs/src/temporal.md` (new section), `api.md` (`update`),
  `low-level-reference.md`, `gllvmtmb-parity.md`, `r/README_bridge.md`, `ROADMAP.md`,
  `CHANGELOG.md`, decision note
  `docs/dev-log/decisions/2026-09-27-temporal-slice2-composition.md`, check-log.

## 3a. Decisions and Rejected Alternatives

- **Parameter order measured from R.** `b_fix, log_sigma_eps, theta_rr_B,
  theta_temporal_time, theta_temporal_rr, theta_temporal_diag, theta_diag_B, theta_rr_W,
  theta_diag_W, log_sigma_re_int`. The spec gives no composed order; every receipt test
  asserts R's recorded name vector.
- **Optimiser.** Both line searches from the same start, keep the lower, then Newton
  polish. Measured reason: on `sim_u__Tdep_Bindep` Hager-Zhang reached the
  `sigma_eps -> 0` limit (objective 687.0329) where backtracking reached R's optimum
  (687.02046); five cells stopped with gradients 2e-8 to 8e-7 above `g_tol = 1e-8`.
  Rejected: multi-start (not R semantics) and changing the start away from R's.
- **Suppressed-sigma gradient tolerance.** R's TMB gradient at its optimum on
  `sim_rw__Tindep_Wrowlatent` is 1.9e-8 from a 256-bit central difference; Julia's is
  8e-14 from it. Those two cells check Julia against the 256-bit reference at 1e-10 and
  against R at 5e-8; all other cells stay at 1e-8. Measured, not a silent widening.
- **Julia-side limits.** One ordinary term per level, one `(1 | g)`, no `common = true`;
  `unit = nothing` means the series column. Stated in the docstring and reference page.
- Rejected: editing `src/formula.jl` (the `gllvm()` hook needs the grammar lane).

## 4. Files Touched

`src/temporal.jl`, `src/temporal_likelihood.jl`, `src/temporal_fit.jl`,
`src/temporal_methods.jl`, `src/GLLVModels.jl` (export `update`, include comments);
`test/test_temporal_composed.jl`, `test/test_temporal_composed_receipts.jl`,
`test/test_temporal_api.jl` (one assertion), `test/runtests.jl`,
`test/fixtures/temporal_p1/{generate_temporal_p1_slice2.R, julia_optima_composed.jl,
composed.toml, composed_cross.toml, julia_optima_composed.csv, fixture_helpers.jl}`;
`docs/src/{temporal,api,low-level-reference,gllvmtmb-parity}.md`, `r/README_bridge.md`,
`ROADMAP.md`, `CHANGELOG.md`, the decision note, this report, `docs/dev-log/check-log.md`.
Not touched: `_laplace_mode`, `src/families/mixed.jl`, `src/grouped_dispersion.jl`,
`src/model_selection.jl`, `src/cv.jl`, `src/formula.jl`, `Project.toml`.

## 5. Checks Run

Per file, `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia +<v> --project=. -e 'using
Test, GLLVModels; include(...)'` (project environment; no test-only package needed):

| File | 1.10 | 1.13 |
| --- | --- | --- |
| test_temporal_composed_receipts.jl | 643/643 | 643/643 |
| test_temporal_composed.jl | 55/55 | 55/55 |
| test_temporal_api.jl | 81/81 | 81/81 |
| test_temporal_oracles.jl | 258/258 | 258/258 |
| test_temporal_fit_receipts.jl | 738/738 | 738/738 |
| test_temporal_helpers.jl | 129/129 + 8/8 | 129/129 + 8/8 |

Red first: `test_temporal_composed.jl` against slice 1's source failed (2 failures, 8
errors of 10 before the top-level composed fit aborted). The full suite was not run
(instruction).

## 6. Tests of the Tests

- R's TMB `fn` / `gr` at deterministic coordinates away from the optimum (every block
  moved) on all 25 cells: 3.0e-9 / 3.4e-9.
- The dense oracle is a second implementation (own unpack, `Distributions.MvNormal`,
  keyed on R's parameter names): 9.1e-13 at R's optimum and at the fixed coordinates.
- R's `report$eta` (conditional predictor, all tiers) at R's coordinates: 1.8e-15; R's
  `extract_ordination(level = "unit")` scores: 8.9e-16.
- Receipts and every response column are sha256-guarded.

## 7. Issue Ledger

No GitHub issue numbers. Case-map rows (`temporal/*`, PR #526) are not edited here.

## 8. Consistency Audit

Docstrings, reference page, API page, CHANGELOG, ROADMAP, parity page and bridge README
describe the same scope: temporal source alone or with ordinary unit / unit_obs terms
through `fit_temporal_gllvm`; no `gllvm()` hook, cross-source cells, `traits()`,
offsets, new-data prediction or bridge.

## 9. What Did Not Go Smoothly

- R's engine test fixture is deterministic (`value = trait + occasion / 10`), so its
  fits are degenerate; interior-optimum panels were simulated for between-optima.
- `Optim.converged` is true on a function-change stop, so the first Newton-polish
  version skipped the polish; it now always checks the gradient.

## 10. Known Residuals

- The `gllvm()` long-data hook (`src/formula.jl`) needs the grammar lane's consent.
- The spec says 14 slice-2 blocks (32 expectations); its table marks 13 rows "slice 2"
  (20 expectations). All 13 are twinned; the fourteenth is not identifiable from the
  table.
- Two of the 13 are partial twins by design (program-bootstrap.R:40,
  program-selection.R:22: a `unit` source in place of R's kernel fixture).
- Not receipted: two ordinary terms at one level, `common = true`, several `(1 | g)`.
- Case-map `evidence` fields remain empty.

## 11. Team Learning

Record R's gradient at deterministic off-optimum coordinates as well as at its optimum:
it separates a likelihood mismatch from TMB's inner-solve noise without widening a
tolerance.

## 12. Cross-Product Coverage

Temporal indep / dep / latent / latent-unique × AR1 / OU; unreplicated and replicated;
unit tier indep / dep / latent / latent without unique; unit_obs tier indep / dep /
latent (per-row and series-occasion levels); `(1 | series)`; unit + unit_obs together;
sigma suppression on replicated per-row indep and latent.

Rose audit verdict: not yet run.
