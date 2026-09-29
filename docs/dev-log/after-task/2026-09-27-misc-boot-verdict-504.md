# After-task: Exponential, lognormal, Student-t and GP-1 bootstrap refits report their own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:misc-boot-verdict-504:6081`, branch `claude/misc-boot-verdict-504` from
`origin/main` @ `5b9af3763`, worktree `.worktrees/misc-boot-verdict-504`. One slice of the #504
per-family migration, following the zero-inflated and zero-truncated slices.

## 1. Goal

Move the four single-type bootstrap refit closures (Exponential, lognormal, Student-t, GP-1) onto the
#508/#516 contract, so a refit that ends on the fitter's failure verdict is excluded from the
bootstrap.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `ExponentialFit`, `LognormalFit`,
  `StudentTFit` and `GP1Fit` return `(θ = ..., converged = fb.converged, loglik = fb.loglik)`. The θ
  expression in each is byte-identical to before, and each keeps its `return nothing` on a throw. The
  Student-t closure serves both shared and per-species σ.
- `test/test_confint_bootstrap_verdict_misc.jl` (new, registered after the Poisson file).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **Real failing draws on four of five routes.** Exponential: one negative response. Student-t,
  shared and per-species σ: one response of 1e300. GP-1: one count of -1. Each fitter then reports
  `converged = false`, `loglik = -Inf`, finite θ, no throw, on Julia 1.10.12 with and without
  `--check-bounds=yes` and on 1.13.0.
- **A labelled stub on the lognormal route.** The fitter is closed form (it reuses the Gaussian profile
  fit on log y), so no probed value fails softly: 0, -1, NaN, Inf and -Inf throw an `ArgumentError`
  (the refit returns `nothing`, which main already rejects), and 0.5, 1e-300, 1e18, 1e150 and 1e300 all
  fit with `converged = true`. Its rejection and end-to-end testsets use a stub; adapter parity pins
  the migrated closure.
- **Rejected candidates.** Exponential: 0, 0.5, 1e-300 and 1e18 converge (all valid positive-support
  values); 1e300 and 1e150 also fail softly but take longer. Student-t: 0, -1, 0.5 and 1e-300 converge;
  NaN and ±Inf throw. GP-1: 0.5, NaN, ±Inf, 1e150, 1e300 and 1e-300 throw an `InexactError` in
  `Integer.(...)`; 0 and 10^18 converge.
- **No `upper_boundary` flag,** per the slice scope. Whether the GP-1 α (capped by `α_bound`) or the
  Student-t σ needs one is not decided here.
- **Endpoint equality on the lognormal route,** the fastest one (closed form, under 0.01 s per fit),
  with `n_boot = 10`.

## 4. Files Touched

- `src/confint_family.jl` (modified, four closures)
- `test/test_confint_bootstrap_verdict_misc.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-misc-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, machine load about 65, per-file, full
suite not run.

- Probe (Julia 1.10.12, p = 4, n = 120, K = 1, the test's data). Healthy fits, all
  `converged = true`: Exponential 0.3 s (13 iterations), lognormal under 0.01 s, Student-t shared σ
  0.2 s, per-species σ 0.8 s, GP-1 2.3 s. Chosen bad draws: Exponential 0.02 s, Student-t shared under
  0.01 s, per-species 4.5 s, GP-1 0.2 s. Same verdicts under `--check-bounds=yes` and on 1.13.0.
- RED, new file on `origin/main` 5b9af3763, Julia 1.10.12: 14 pass, 19 fail, 24 error of 57.
- GREEN: 57/57 on Julia 1.10.12 (67.1 s wall), 57/57 on 1.10.12 with `--check-bounds=yes` (66.6 s),
  57/57 on Julia 1.13.0 (51.0 s).
- Neighbours on 1.10.12, each file alone: `test_confint_family.jl` 341/341 (466 s),
  `test_bridge_capabilities.jl` 242/242, `test_confint_hessian_consistency.jl` 12/12,
  `test_known_sentinel_defects.jl` 25 pass, 1 broken. The broken one is a static `@test_broken` on the
  phylo σ_phy sign in a file this branch does not touch; not re-run on main.

## 6. Tests of the Tests

- RED shows the file fails on main in every testset that reads the adapter's output on a real draw.
  The 14 passes there are the five healthy-fit checks, the four direct-fit failure checks, the three
  lognormal stub-rejection checks, the lognormal stub end-to-end check and one assertion of the
  endpoint testset. The stub testsets pass on main by construction: they exercise
  `_bootstrap_refit_ok`, not the lognormal closure, which adapter parity pins instead.
- The real-draw testset asserts the direct fit is a genuine failure, that the old bare-vector contract
  accepts its θ, and that the new contract rejects it.
- The end-to-end testset mixes healthy and failing draws and asserts `0 < n_converged < 6` on all five
  routes.
- The endpoint-equality testset asserts all 10 replicates converge and that the bounds are finite
  before comparing, so it cannot pass vacuously.
- `!haskey(raw, :upper_boundary)` pins the deliberate absence of the flag.

## 7a. Issue Ledger

- #504 stays open (part of).

## 8. Consistency Audit

- No other #504 branch touches these four closures. The only hit from
  `git diff origin/main...<branch>` over the #504 branches is `claude/hurdle-delta-boot-verdict-504`,
  whose `logσb` lines are in the delta-lognormal closures, not the Student-t one.
- The sibling slices all add a CHANGELOG bullet at the same place and an include next to the Poisson
  one in `test/runtests.jl`, so textual merge conflicts are expected there; both are additive.
- The Exponential and GP-1 refits pass the fit's `link` and `hessian`, Student-t passes `nu`, `link`,
  `hessian` and `disp_group`, and lognormal passes only `K`. Unchanged here.

## 9. What Did Not Go Smoothly

- Nothing blocking. The lognormal route has no soft failure, which the probe showed before any test
  was written, so it follows the truncated slice's stub pattern.

## 10. Known Residuals

- GP-1 with one count of 10^18 reports `converged = true` with a positive log-likelihood (+5189.6 on
  1.10.12, +4417.6 on 1.13.0). A marginal of a probability mass function cannot exceed 0, so this is a
  numerical defect in the GP-1 marginal for very large counts. Outside the bootstrap; not changed.
- The lognormal refit passes no keyword beyond `K`, so a fit made with non-default Gaussian keywords is
  refit with the defaults (pre-existing; not probed).
- Full suite not run; CI will run it.

## 11. Team Learning

Sweep the candidate bad values per route: the same value (a negative response) fails softly for
Exponential and GP-1, is absorbed by Student-t, and throws for lognormal.

## 12. Cross-Product Coverage

Covered: all four fit types (Student-t with shared and per-species σ), Julia 1.10.12 and 1.13.0,
aarch64 macOS, with and without bounds checks on 1.10.12. Not covered: Linux and Windows (CI), K > 1,
non-default links, masks and offsets, per-trait ν vectors.
