# After-task: Beta-hurdle and ordered-beta bootstrap refits report their own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:bhob-boot-verdict-504:6081`, branch `claude/bhob-boot-verdict-504` from
`origin/main` @ `5b9af3763`, worktree `.worktrees/bhob-boot-verdict-504`. One slice of the #504
per-family migration, following the misc, zero-inflated and zero-truncated slices.

## 1. Goal

Move the beta-hurdle and ordered-beta bootstrap refit closures onto the #508/#516 contract, so a
refit that ends on the fitter's failure verdict is excluded from the bootstrap.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `BetaHurdleFit` and
  `OrderedBetaFit` return `(θ = ..., converged = fb.converged, loglik = fb.loglik)`. The θ expression
  in each is byte-identical to before, and each keeps its `return nothing` on a throw. Both fit
  structs already carry `converged` and `loglik` (set by `_fit_verdict`).
- `test/test_confint_bootstrap_verdict_bhob.jl` (new, registered after the Poisson file).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **A real failing draw on the ordered-beta route.** One response of -1 (outside [0, 1]) makes the
  fitter report `converged = false`, `loglik = -Inf`, a finite θ and no throw, after 0 iterations, on
  Julia 1.10.12 with and without `--check-bounds=yes` and on 1.13.0.
- **A labelled stub on the beta-hurdle route.** The fitter scores `y > 0` as the positive part and
  clamps it to (1e-6, 1 - 1e-6); anything else (0, negatives, NaN, -Inf) is an absence. One cell of
  0, 1, -1, 1.5, 0.5, 1e300, NaN, Inf, -Inf or 1e-300 all fit with `converged = true` and a finite
  log-likelihood in all three run modes. Its rejection and end-to-end testsets use a stub; adapter
  parity pins the migrated closure.
- **Rejected ordered-beta candidates.** 0, 1 and 0.5 are valid and converge. 1.5, 1e300, Inf, -Inf and
  1e-300 also fail softly; -1 was chosen as the plainest out-of-support value. NaN throws an
  `ArgumentError` (the refit returns `nothing`, which main already rejects).
- **The ordered-beta adapter's simulator is a stub that errors** (bootstrap is not offered for that
  family on main). The parity draw is a separately simulated data set (seed 504), and the end-to-end
  testset supplies its own simulator. The migrated closure is still what `_family_bootstrap` calls.
  Wiring up an ordered-beta simulator is out of scope.
- **No `upper_boundary` flag,** per the slice scope. Whether the Beta precision φ needs one is not
  decided here.
- **Endpoint equality on the beta-hurdle route,** the fastest route with a working simulator (about
  0.5 s per fit), with `n_boot = 10`.

## 4. Files Touched

- `src/confint_family.jl` (modified, two closures)
- `test/test_confint_bootstrap_verdict_bhob.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-bhob-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, machine load about 57, per-file, full
suite not run.

- Probe (p = 4, n = 120, K = 1, the test's data). Healthy fits, all `converged = true`: beta-hurdle
  0.51 s (19 iterations), ordered-beta 1.25 s (31 iterations) on 1.10.12; 0.45 s and 0.95 s on 1.13.0.
  Chosen ordered-beta bad draw: 0.03 s on 1.10.12, 0.16 s on 1.13.0. Same verdicts for every candidate
  under `--check-bounds=yes` and on 1.13.0.
- RED, new file on `origin/main` 5b9af3763, Julia 1.10.12: 8 pass, 7 fail, 10 error of 25.
- GREEN: 25/25 on Julia 1.10.12 (40.5 s wall), 25/25 on 1.10.12 with `--check-bounds=yes` (41.9 s),
  25/25 on Julia 1.13.0 (28.7 s).
- Neighbours on 1.10.12, each file alone: `test_confint_family.jl` 341/341 (449 s),
  `test_beta_hurdle.jl` 62/62 (16 s), `test_ordered_beta.jl` 49/49 (19 s). No other test file uses
  the bootstrap with either family.

## 6. Tests of the Tests

- RED shows the file fails on main in every testset that reads the adapter's output on a real draw.
  The 8 passes there are the two healthy-fit checks, the direct-fit failure check, the three
  beta-hurdle stub-rejection checks, the beta-hurdle stub end-to-end check and one assertion of the
  endpoint testset. The stub testsets pass on main by construction: they exercise
  `_bootstrap_refit_ok`, not the beta-hurdle closure, which adapter parity pins instead.
- The real-draw testset asserts the direct fit is a genuine failure, that the whole θ is finite, that
  the old bare-vector contract accepts it, and that the new contract rejects it.
- The end-to-end testset mixes healthy and failing draws and asserts `0 < n_converged < 6` on both
  routes.
- The endpoint-equality testset asserts all 10 replicates converge and that the bounds are finite
  before comparing, so it cannot pass vacuously.
- `!haskey(raw, :upper_boundary)` pins the deliberate absence of the flag.

## 7a. Issue Ledger

- #504 stays open (part of).

## 8. Consistency Audit

- No other #504 branch touches these two closures. `claude/hurdle-delta-boot-verdict-504` and
  `claude/zi-boot-verdict-504` edit other two-part closures with the same `fb.βz, fb.βc` prefix, not
  the beta-hurdle one (`log(fb.φ)`).
- The sibling slices all add a CHANGELOG bullet at the same place and an include next to the Poisson
  one in `test/runtests.jl`, so textual merge conflicts are expected there; both are additive.
- Both refits pass only `K`; unchanged here.

## 9. What Did Not Go Smoothly

- Nothing blocking. The beta-hurdle route has no soft failure, which the probe showed before any test
  was written, so it follows the misc slice's stub pattern.

## 10. Known Residuals

- Degenerate data give `converged = true` at a positive log-likelihood. Beta-hurdle: all cells 1
  gives +12738.8 and all cells 0.5 gives +2.6e141; ordered-beta: all cells 0.5 gives +6341.0 (same on
  1.10.12 and 1.13.0). These are continuous densities, so a positive value is not impossible, but the
  sizes point to φ running to its limit on constant data. Outside the bootstrap; not changed.
- The beta-hurdle fitter silently treats negative, NaN and -Inf responses as absences and clamps
  values of 1 or more into (0, 1), so out-of-support data neither throw nor fail. Pre-existing; not
  changed.
- The ordered-beta bootstrap stays unavailable through `confint` (simulator stub).
- The refits pass no keyword beyond `K`, so a fit made with non-default keywords (for example
  `hessian`, `mask`, cutpoint starts) is refit with the defaults. Pre-existing; not probed.
- Full suite not run; CI will run it.

## 11. Team Learning

Check the adapter's simulator before planning an end-to-end test: the ordered-beta one errors, so the
migrated refit is only reachable through a custom `_FamilyCI`.

## 12. Cross-Product Coverage

Covered: both fit types, Julia 1.10.12 and 1.13.0, aarch64 macOS, with and without bounds checks on
1.10.12. Not covered: Linux and Windows (CI), K > 1, masks and missing data, non-default `hessian`.
