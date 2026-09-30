# R port report: warm-start guard + runaway detector in gllvmTMB's select_lv()

Worktree: `/Users/z3437171/local-scratch/lanes/gllvmTMB-auto-d-20260926`
Branch: `claude/lane-auto-d-r-20260926`, based on `origin/main 9539352f6`
Commit: `978f4bba2` — "feat(select_lv): port GLLVM.jl's warm-start-retry guard and runaway detector"

## Oracle

`/Users/z3437171/local-scratch/lanes/GLLVM.jl-auto-d-20260926/src/model_selection.jl`
(`select_lv()`, `_lv_warm_start()`, `_lv_converged()`, `_lv_runaway()`), and its tests
in `test/test_model_selection.jl` (testsets "warm-start safeguard", "runaway detector").

## What changed

`R/select-lv.R`:

- File banner (line 19–25): new note explaining the one divergence this port makes
  from the oracle (warm-start mechanism), forced by the file restriction on this lane.
- `.select_lv_runaway()` (new, ~line 27): SCALE check (`max row norm > max_latent_sd`,
  skipped for identity-link fits) and RATIO check (`Binomial` only: largest per-trait
  max `|loading|` / median of those maxima `>= ratio_max`). Reads the loading matrix via
  `getLoadings(fit, level = "unit", rotate = "none")`.
- `.select_lv_warm_control()` (new, ~line 78): builds a `control()` copy with
  `start_from = <accepted fit>` for the retry.
- `select_lv()` signature (~line 276): added `warm_start = TRUE, tol = 1e-3,
  max_latent_sd = 10, ratio_max = 25, .fitter = gllvmTMB`. `.fitter` is an internal
  test hook mirroring the oracle's `_fitter` keyword (documented `@keywords`-style as
  "Internal test hook... Not intended for ordinary use").
- Main sweep loop (~line 346–470): replaced the old error/convergence-only guard with
  `try_fit()`, a closure that classifies every attempted `d` into `status` ∈ {"ok",
  "warm_start", "nonmonotone", "unconverged", "runaway", "failed"} and carries
  `accepted_ll`/`accepted_fit` (the last ACCEPTED `d`, not the last attempted) across
  iterations for the monotonicity check and the warm-start retry. A rejected
  non-`"failed"` fit is retried once (if `warm_start` and an accepted fit exists) via
  `control(start_from = accepted_fit)`, kept as `"warm_start"` only if the retry is
  itself `"ok"`.
- Eligibility (~line 476): simplified to `table$status %in% c("ok", "warm_start")`
  (previously three separate flag columns folded together ad hoc).
- `print.gllvmTMB_select_lv()` (~line 547): "Failed fits" section renamed "Excluded
  fits" and now lists every non-selectable row with its `status` and reason (previously
  only rows with a hard error were listed).
- Removed `.select_lv_isTRUE_vec()` (orphaned — its only caller was replaced).
- `table` gained two columns: `status` and `message` (existing columns `d`, `npar`,
  `logLik`, `aic`, `bic`, `aicc`, `converged`, `pd_hessian`, `seconds`, `error` are
  unchanged, so the two existing test files did not need updating).
- Roxygen: new `@param`s for `warm_start`/`tol`/`max_latent_sd`/`ratio_max`/`.fitter`,
  a new "Guard against a non-nesting or runaway fit" `@details` section, the
  "selected d is itself an estimate, conditional intervals" note, and `@return`
  updated for the two new table columns.

`man/select_lv.Rd`: regenerated via `devtools::document()`. (Note: `document()` in this
shared worktree also regenerated 18 unrelated `man/*.Rd` files as collateral from other
lanes' unsynced roxygen comments elsewhere in `R/`; those were reverted via
`git show HEAD:<path> > <path>` each time, twice, to keep this commit scoped to the
four files the brief permits.)

`NEWS.md`: one bullet added at the top of "Development (unreleased)".

`tests/testthat/test-select-lv-guard.R` (new): TDD, written and run failing before
implementation. Mirrors the oracle's two testsets. Uses a stubbed `.fitter` (returns
fake fit objects of a `"gllvmTMB_select_lv_test_fake"` S3 class with a `logLik` method
registered via `registerS3method()`) and `testthat::local_mocked_bindings(getLoadings =
..., .package = "gllvmTMB")` to stub the loading extractor, so every failure mode is
exact and fast — no real TMB optimisation.

One test design note: the oracle's `@test_throws InterruptException` test cannot be
ported literally — signalling a condition of class `"interrupt"` (or wrapping the call
in `expect_error()`) trips testthat's own interrupt handling and aborts the whole test
run rather than reporting pass/fail. The port instead signals a custom non-`"error"`
condition class and catches it directly with `tryCatch()` (not `expect_error()`),
proving the same thing the oracle's test proves: `select_lv()`'s `tryCatch(error = )`
does not swallow a non-error condition.

## How convergence / start values / loadings are accessed in gllvmTMB

- **Convergence**: `fit$opt$convergence == 0L` (optimizer-level), combined with
  `isFALSE(fit$sd_report$pdHess)` for a CONFIRMED non-positive-definite Hessian.
  `pdHess` is `NA` (not `FALSE`) when `control(se = FALSE)` skips `sdreport()`
  entirely — that is "not determined," not "known bad," and does not disqualify a fit
  (this was already the convention in the pre-existing code; the guard preserves it,
  folding both reasons into a single `"unconverged"` status).
- **Start values**: `gllvmTMBcontrol(start_from = <fitted gllvmTMB object>)`. Internally
  (`R/init-warmstart.R::.gllvmTMB_apply_start_from()`, not touched by this lane) it
  extracts the source fit's TMB parameter list via
  `start_from$tmb_obj$env$parList(start_from$opt$par, par_full)` and copies **only
  same-shaped** entries into the new fit's starting parameters — same length, same
  `dim()`. There is no argument that accepts a caller-supplied starting parameter list
  or a specific starting loading matrix (no TMB `parameters=` passthrough at the
  `gllvmTMB()` level, no `Lambda_init`/`beta_init`-style keyword). This is the one
  documented divergence from the oracle (see below).
- **Loadings**: `getLoadings(fit, level = "unit", rotate = "none")` (`R/output-methods.R`,
  a thin wrapper around `extract_ordination()`), returning the native (unrotated)
  `n_traits × d` lower-triangular Λ for the between-unit tier — the tier `select_lv()`
  sweeps (`latent(0 + trait | unit, d = ...)`).

## Test counts

Commands run from the worktree root with `devtools::load_all()`:

```r
testthat::test_file("tests/testthat/test-select-lv-guard.R", reporter = "summary")
testthat::test_file("tests/testthat/test-select-lv-anova.R", reporter = "summary")
testthat::test_file("tests/testthat/test-example-model-selection-rank.R", reporter = "summary")
```

- `test-select-lv-guard.R` (new): **15 passed, 0 failed, 0 skipped** (6 of the 15
  produce an expected `cli_warn` about excluded fits, correctly asserted on).
- `test-select-lv-anova.R` (pre-existing): **34 passed, 0 failed, 3 skipped**
  (heavy/real-fit tests, gated behind `GLLVMTMB_HEAVY_TESTS=1` per house convention).
- `test-example-model-selection-rank.R` (pre-existing): **36 passed, 0 failed, 0
  skipped** (does not call `select_lv()`; calls `gllvmTMB()` directly, unaffected).

Real end-to-end sweep, requested "if time permits": re-ran `test-select-lv-anova.R`
with `GLLVMTMB_HEAVY_TESTS=1 GLLVMTMB_O5_NSIM=5`, which exercises `select_lv()` through
**real** `gllvmTMB()` fits (not mocked) across `d = 1..4`. It passed: `select_lv()`
recovered the true rank `d = 2` under both BIC and AIC on the rank-2 DGP fixture, with
the new guard active and zero fits rejected on that fixture (a healthy sweep, so no
retries were triggered — confirms the guard is a no-op on the pre-existing passing
case, per its own design intent).

`devtools::document()` ran clean (no roxygen warnings, including for the internal
`.fitter` argument, which now has a `@param` entry marked as an internal test hook).

## Divergences from the oracle, and why

1. **Warm-start mechanism (the significant one).** The oracle passes an explicit
   `Λ_init`/`β_init` keyword pair to its fitter, with `Λ_init` built as the accepted
   `(d-1)` loading matrix plus one new column that is lower-triangular (zeros above the
   diagonal, `0.1` below). gllvmTMB's fitter (`gllvmTMB()`) has no equivalent
   keyword — its only start-value route is `control(start_from = <fitted gllvmTMB
   object>)`, and that route copies only same-shaped parameter blocks
   (`R/init-warmstart.R`, outside this lane's file allowance). Since the loading matrix
   changes shape between `d` and `d+1`, `start_from`'s copy loop skips it entirely; the
   new column in a warm-started refit starts at gllvmTMB's ordinary default init, not
   the oracle's specific lower-triangular values. Fixed effects and dispersion
   parameters DO carry over (same shape at every `d`), so the retry is a genuine,
   useful warm start — just not the fine-grained one the oracle builds. This is
   documented in the file banner, the `warm_start` roxygen `@param`, and the `@details`
   section of `?select_lv`.
2. **Table shape.** The brief allowed either dropping rejected rows or adding a
   `status` column, "whichever is least disruptive to the existing tests." Added a
   `status`/`message` column pair and kept all rows (existing behaviour) rather than
   dropping rejected `d` from `table`, since `test-select-lv-anova.R`'s heavy recovery
   test reads `sel$table$d`/`aic`/`bic`/`converged` positionally across `1:d_max` and
   dropping rows would have broken that indexing.
3. **Interrupt test**, discussed above under "Test counts"/test file — mechanical
   necessity, not a semantic difference in the guard's own `tryCatch(error =)` behaviour.

No other behavioural divergence: `tol` (`1e-3`), `max_latent_sd` (`10`), and
`ratio_max` (`25`) defaults and check semantics (SCALE check on any non-identity-link
family, RATIO check `Binomial`-only) match the oracle exactly.
