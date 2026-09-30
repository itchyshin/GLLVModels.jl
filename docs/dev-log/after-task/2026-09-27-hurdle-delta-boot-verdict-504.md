# After-task: hurdle and delta bootstrap refits report their own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:hurdle-delta-boot-verdict-504:6081`, branch
`claude/hurdle-delta-boot-verdict-504` from `origin/main` @ `1214e948e`, worktree
`.worktrees/hurdle-delta-boot-verdict-504`. One slice of the #504 per-family migration, following the
Gamma, Ordinal and zero-inflated slices.

## 1. Goal

Move the hurdle and delta two-part bootstrap refit closures onto the #508/#516 contract, so a refit
that ends on the fitter's failure sentinel is excluded from the bootstrap.

## 2. Implemented

- `src/confint_family.jl`: six `refit` closures now return
  `(θ = ..., converged = fb.converged, loglik = fb.loglik)`. They are the single closures in
  `_family_ci` for `HurdlePoissonFit` and `HurdleNBFit`, and the `predictor = :shared` and default
  two-predictor closures in `_family_ci` for `DeltaLogNormalFit` and `DeltaGammaFit`. The θ expression
  in each is byte-identical to before, and the delta closures keep their early `return nothing` when
  the refit's dispersion has the wrong shape.
- `test/test_confint_bootstrap_verdict_hurdle_delta.jl` (new, registered after the Poisson file).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **One real failing draw on every route, no stub.** A single cell set to `1e300` makes all six
  fitters end on their sentinel: `converged = false`, `loglik = -Inf`, finite θ, no throw, with and
  without `--check-bounds=yes`. On the hurdle routes the count log-density calls `Int(y)` and the
  `InexactError` is caught by the objective, as with the ZI half count. On the delta routes the fitted
  minimum is the sentinel, so the objective already returned it at the warm start and L-BFGS never left
  it; which term overflows was not traced (inference, not checked).
- **Rejected bad draws, from the probe.** A negative value and NaN are scored as zeros (`y > 0` is
  false) and every route converges. A half count fails on the hurdle routes but is a valid positive
  value on the delta routes, which converge. `Inf` makes every fitter throw an `ArgumentError`, so the
  refit returns `nothing`, which main already rejects; it tests nothing new. `1e-300` fails on the
  hurdle routes and on the default delta-lognormal route, but the shared-predictor delta-lognormal and
  both delta-gamma routes converge on it, so it is not usable across all six.
- **No `upper_boundary` flag,** per the slice scope. Whether the hurdle-NB size `r` needs one is not
  decided here.
- **Endpoint equality on the Hurdle-Poisson route** (0.3 s per healthy fit, among the two fastest),
  with `n_boot = 10`.

## 4. Files Touched

- `src/confint_family.jl` (modified, six closures)
- `test/test_confint_bootstrap_verdict_hurdle_delta.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-hurdle-delta-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, machine load about 55, per-file, full
suite not run.

- Probe (Julia 1.10.12, p = 4, n = 120, K = 1, the test's data). Healthy fits, all
  `converged = true`: Hurdle-Poisson 0.3 s (20 iterations), Hurdle-NB 0.6 s, delta-lognormal 0.6 s,
  delta-lognormal shared 0.3 s, delta-gamma 1.0 to 2.1 s, delta-gamma shared 1.2 s. The `1e300` cell:
  `converged = false`, `loglik = -Inf`, finite θ on all six, 0.01 to 0.16 s. Same verdicts under
  `--check-bounds=yes`. Other candidates as listed in 3a.
- RED, new file on `origin/main` 1214e948e, Julia 1.10.12: 13 pass, 26 fail, 30 error of 69 (49 s).
- GREEN: 69/69 on Julia 1.10.12 (52 s wall), 69/69 on 1.10.12 with `--check-bounds=yes` (57 s),
  69/69 on Julia 1.13.0 (57 s).
- Neighbour on 1.10.12, alone: `test_confint_family.jl` 341/341 (466 s). It is the only other test
  file that mentions both bootstrap and a hurdle or delta fitter.

## 6. Tests of the Tests

- RED shows the file fails on main in every testset. The 13 passes there are the six healthy-fit
  checks, the six direct-fit failure checks and one assertion of the endpoint testset; every assertion
  on the adapter's output fails or errors.
- The failing-draw testset asserts the direct fit is a genuine failure, that the old bare-vector
  contract accepts its θ, and that the new contract rejects it.
- The end-to-end testset mixes healthy and failing draws and asserts `0 < n_converged < 6` on all six
  routes, so each family and each delta predictor mode is covered.
- The endpoint-equality testset asserts all 10 replicates converge and that the bounds are finite
  before comparing, so it cannot pass vacuously.
- `!haskey(raw, :upper_boundary)` pins the deliberate absence of the flag.

## 7a. Issue Ledger

- #504 stays open (part of).

## 8. Consistency Audit

- No other #504 or verdict branch on `origin` touches these six closures (checked with
  `git diff origin/main...<branch> -- src/confint_family.jl` over the 14 such branches).
- The sibling slices all add a CHANGELOG bullet at the same place and an include next to the Poisson
  one in `test/runtests.jl`, so textual merge conflicts are expected there; both are additive.
- The refits call the fitters with `K` (plus `predictor = :shared` and `disp_group` for delta) and
  defaults for every other keyword. Unchanged here.

## 9. What Did Not Go Smoothly

- Nothing blocking. The ZI half-count draw does not carry over to the continuous delta families, and
  `1e-300` splits the routes; probing six candidates on all six routes before writing the test found
  one draw that fails on every route.

## 10. Known Residuals

- Negative values and NaN are silently treated as zeros by the hurdle and delta likelihoods, and a fit
  on such data reports `converged = true`. This is input validation, outside the bootstrap; not
  changed.
- The refits pass no `hessian`, `offset` or `objective` keyword, so a fit made with non-default values
  (for example a delta-gamma VA fit) is refit with the defaults (pre-existing; not probed).
- Full suite not run; CI will run it.

## 11. Team Learning

Probe several corruptions per route, not one per family: the same value (`1e-300`) failed on some
delta routes and converged on others, and only a sweep made that visible.

## 12. Cross-Product Coverage

Covered: `HurdlePoissonFit`, `HurdleNBFit`, `DeltaLogNormalFit` and `DeltaGammaFit`, both delta
predictor modes, `disp_group = :species`, Julia 1.10.12 and 1.13.0, aarch64 macOS, with and without
bounds checks on 1.10.12. Not covered: Linux and Windows (CI), K > 1, `disp_group = :shared`, offsets,
the delta-gamma VA objective.
