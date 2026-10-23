# After-task: row-random, multinomial and covariate-GLLVM bootstrap refits report their own verdict (part of #504, 2026-09-28)

Lane: Claude, `LANE_ID=claude:rest-boot-verdict-504:6081`, branch `claude/rest-boot-verdict-504` from
`origin/main` @ `5b9af3763`, worktree `.worktrees/rest-boot-verdict-504`. The last slice of the #504
per-family migration.

## 1. Goal

Move the three remaining bare-vector bootstrap refit closures (`RowRandomFit`, `MultinomialFit`,
`GllvmCovFit`) onto the #508/#516 contract, so a refit that ends on the fitter's failure verdict is
excluded from the bootstrap.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `RowRandomFit`, `MultinomialFit`
  and `GllvmCovFit` return `(θ = ..., converged = fb.converged, loglik = fb.loglik)`. For
  `RowRandomFit` and `GllvmCovFit` the dispersion ternary now assigns `θb` and both branches share one
  return; the branch expressions are unchanged. Every `return nothing` stays (the throw guard in all
  three, and the two shape guards in the multinomial closure).
- `test/test_confint_bootstrap_verdict_rest.jl` (new, registered after the Poisson file).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **Both ternary branches tested.** Row-random and covariate GLLVM are each run with Poisson (no
  dispersion) and NegativeBinomial (dispersion), so five routes in all.
- **Real failing draws on the four count routes.** One response of -1: the log-pmf is -Inf at every θ,
  every evaluation returns the 1e12 sentinel, and each fitter reports `converged = false`,
  `loglik = -Inf`, finite θ, no throw. Same verdict on Julia 1.10.12 with and without
  `--check-bounds=yes` and on 1.13.0.
- **Rejected count candidates.** 0.5 and 1e300 also fail softly on all four routes; -1 was kept for
  being a plain invalid count. NaN and Inf throw an `ArgumentError` (the refit returns `nothing`, which
  main already rejects). -Inf fails softly. 1e18 fails softly on three routes but throws a
  `DomainError` on row-random NB.
- **A labelled stub on the multinomial route.** Every invalid category tried throws before fitting: 0,
  -1 and 4 (with 3 categories) raise an `ArgumentError`, and 0.5, NaN and Inf an `InexactError`. Draws
  that are valid but degenerate fit with `converged = true`: one category never observed, every
  observation in category 1, and all but two in category 1. So its rejection and end-to-end testsets
  use a stub; adapter parity pins the migrated closure.
- **No `upper_boundary` flag,** per the slice scope.
- **Endpoint equality on the multinomial route,** the fastest (fixed effects only, under 0.01 s per
  fit), with `n_boot = 10`.

## 4. Files Touched

- `src/confint_family.jl` (modified, three closures)
- `test/test_confint_bootstrap_verdict_rest.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-28-rest-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, per-file, full suite not run.

- Probe (Julia 1.10.12, p = 4, n = 120, K = 1 for the count routes, n = 200 with one covariate and 3
  categories for multinomial; the test's data). Healthy fits, all `converged = true`: row-random
  Poisson 0.29 s (19 iterations), row-random NB 0.80 s (26), covariate Poisson 0.28 s (24), covariate
  NB 0.69 s (24), multinomial under 0.01 s (9). The chosen bad draw (-1) takes 0.02 to 0.03 s per
  route. Same verdicts for every candidate under `--check-bounds=yes` (identical log-likelihoods) and on
  1.13.0 (different seeded data, same verdicts; row-random Poisson with 1e18 took 2 iterations instead
  of 1).
- RED, new file on `origin/main` 5b9af3763, Julia 1.10.12: 14 pass, 19 fail, 28 error of 61.
- GREEN: 61/61 on Julia 1.10.12 (30.0 s wall, 20.8 s in the testset), 61/61 on 1.10.12 with
  `--check-bounds=yes` (25.8 s), 61/61 on Julia 1.13.0 (23.4 s).
- Neighbours on 1.10.12, each file alone: `test_bridge_x.jl` 200/200 (128.5 s),
  `test_confint_family.jl` 341/341 (458.9 s). No failures and no broken markers.
- Closure census: `src/confint_family.jl` has 48 refit closures. On `origin/main`, 45 return a bare
  vector (the two AGHQ closures screen `fb.converged` themselves). Apart from this branch's three, the
  other 42 are each migrated on one of the open `claude/*-boot-verdict-504` branches (beta, bhob,
  binom, gamma-main, hurdle-delta, misc, nb, nb1, ordinal, trunc, tweedie, zi), checked by scanning
  each branch's copy of the file. Once all land, no bare-vector refit remains.

## 6. Tests of the Tests

- RED shows the file fails on main in every testset that reads the adapter's output. The 14 passes
  there are the five healthy-fit checks, the four direct-fit failure checks, the three multinomial
  stub-rejection checks, the multinomial stub end-to-end check and one assertion of the endpoint
  testset. The stub testsets pass on main by construction: they exercise `_bootstrap_refit_ok`, not the
  multinomial closure, which adapter parity pins instead.
- The real-draw testset asserts the direct fit is a genuine failure, that the whole returned θ is
  finite, that the old bare-vector contract accepts it, and that the new contract rejects it.
- The end-to-end testset mixes healthy and failing draws and asserts `0 < n_converged < 6` on all five
  routes.
- The endpoint-equality testset asserts all 10 replicates converge and that the bounds are finite
  before comparing, so it cannot pass vacuously.
- `!haskey(raw, :upper_boundary)` pins the deliberate absence of the flag.

## 7a. Issue Ledger

- #504 stays open until the sibling branches and this one land (part of).

## 8. Consistency Audit

- No other #504 branch touches these three closures (branch scan above).
- The sibling slices all add a CHANGELOG bullet at the same place and an include next to the Poisson
  one in `test/runtests.jl`, so textual merge conflicts are expected there; both are additive.
- The covariate refit passes `family`, `X`, `K`, `link`, `N` (Binomial only) and `γ_fixed`; row-random
  passes `family`, `K`, `link` and `N` (Binomial only); multinomial passes `X`, `n_categories` and
  `link`. Unchanged here. The test's direct refit passes the same keywords for the families used.

## 9. What Did Not Go Smoothly

- Nothing blocking. The multinomial route has no soft failure, which the probe showed before any test
  was written, so it follows the stub pattern of the lognormal and truncated slices.

## 10. Known Residuals

- The multinomial fitter reports `converged = true` under complete separation: every observation in
  category 1 gives `loglik = -1.74e-5` on both Julia versions. The supremum is 0 with coefficients
  diverging, so LBFGS stops on a vanishing gradient. Such a bootstrap replicate is kept with large
  coefficients. The size of those coefficients was not measured. Pre-existing; not changed.
- Binomial row-random and covariate fits, and `γ_fixed` masks, are not exercised by the new file.
- Full suite not run; CI will run it.

## 11. Team Learning

A fitter whose response is validated up front (multinomial categories) cannot be driven to a soft
failure by bad data, so the stub pattern is the right one there. For count families, one negative count
is the cheapest soft failure and behaved the same on every route and run mode.

## 12. Cross-Product Coverage

Covered: all three fit types, row-random and covariate GLLVM with and without a dispersion parameter
(Poisson, NB), multinomial with one covariate, Julia 1.10.12 and 1.13.0, aarch64 macOS, with and without
bounds checks on 1.10.12. Not covered: Linux and Windows (CI), K > 1, Binomial, Beta, Gamma and NB1
families on these routes, non-default links, `γ_fixed`, multinomial without covariates.
