# After-task: zero-truncated bootstrap refits report their own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:trunc-boot-verdict-504:6081`, branch `claude/trunc-boot-verdict-504`
from `origin/main` @ `4357e4652`, worktree `.worktrees/trunc-boot-verdict-504`. One slice of the #504
per-family migration, following the Gamma, Ordinal, zero-inflated and hurdle/delta slices.

## 1. Goal

Move the zero-truncated count bootstrap refit closures onto the #508/#516 contract, so a refit that
ends on the fitter's failure verdict with a finite θ is excluded from the bootstrap.

## 2. Implemented

- `src/confint_family.jl`: three `refit` closures now return
  `(θ = ..., converged = fb.converged, loglik = fb.loglik)`, one each in `_family_ci` for
  `TruncatedPoissonFit`, `TruncatedNegBin2Fit` and `TruncatedNegBin2PerTraitFit`. The θ expression in
  each is byte-identical to before, and each keeps its `return nothing` when the fitter throws. All
  three fit structs carry `converged` and `loglik`.
- `test/test_confint_bootstrap_verdict_truncated.jl` (new, registered after the Poisson file).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **Real failing draw on the truncated-Poisson route.** One cell set to the integer count `10^18`
  makes the fitter stop after one iteration with a finite θ, `converged = false` and `loglik = -Inf`,
  no throw, on Julia 1.10.12 with and without `--check-bounds=yes` and on 1.13.0 (the GREEN run). A
  direct evaluation of the marginal log-likelihood with that cell, at a hand-picked θ rather than the
  fitter's warm start, returned a finite value of about `-2.6e19`; so the failure most likely comes
  from `_fit_verdict`'s 1e11 threshold rather than a caught throw (inference, warm start not traced).
- **Labelled stub on the two truncated-NB2 routes,** as in the Ordinal slice. The probe found no draw
  there that fails with a finite θ on both Julia versions. Counts of `10^14` to `10^18` in one cell do
  end on the failure verdict, but the size estimate underflows to `r = 0`, so θ holds
  `log(0) = -Inf` and the old contract already rejects it (the first GREEN run failed on exactly this
  assertion for both NB2 routes). Placing the large count in two cells kept θ finite on 1.10.12 for
  some placements (per-trait with two `10^14` cells, shared-r with two `10^16` cells) but not on
  1.13.0, so those draws were rejected as version-fragile. The adapter-parity testset is what pins the
  migrated NB2 closures.
- **Other rejected bad draws.** 0, a negative value, 0.5, NaN, Inf and `1e300` make every fitter
  throw (`ArgumentError` for values below 1, `InexactError` from `Integer.(Y)` otherwise), so the
  refit returns `nothing`, which main already rejects. Counts of `10^6`, `10^9`, `10^12` and `10^13`
  converge.
- **No `upper_boundary` flag,** per the slice scope. Whether the truncated-NB2 size `r` needs one is
  not decided here.
- **Endpoint equality on the truncated-Poisson route** (0.2 s per healthy fit, the fastest), with
  `n_boot = 10`.

## 4. Files Touched

- `src/confint_family.jl` (modified, three closures)
- `test/test_confint_bootstrap_verdict_truncated.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-trunc-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, machine load about 100, per-file, full
suite not run.

- Probe (Julia 1.10.12, p = 4, n = 120, K = 1, the test's data). Healthy fits, all
  `converged = true`: truncated Poisson 0.2 s (19 iterations), truncated NB2 0.7 s (19 iterations),
  truncated NB2 per-trait 4.5 s (133 iterations). Bad-draw verdicts identical with and without
  `--check-bounds=yes`; details in 3a.
- RED, new file on `origin/main` 4357e4652, Julia 1.10.12 with `--check-bounds=yes`: 13 pass, 8 fail,
  13 error of 34 (51 s).
- GREEN: 34/34 on Julia 1.10.12 (46 s wall), 34/34 on 1.10.12 with `--check-bounds=yes` (48 s),
  34/34 on Julia 1.13.0 (42 s).
- Neighbours on 1.10.12, each alone: `test_confint_family.jl` 341/341 (455 s),
  `test_bridge_capabilities.jl` 242/242 (2 s). The latter is the only other test file that mentions
  both bootstrap and a truncated fitter.

## 6. Tests of the Tests

- RED fails on main in the parity testsets (all three routes), the truncated-Poisson real-draw and
  end-to-end testsets, and the endpoint testset. The 13 passes there are the three healthy-fit checks,
  the direct-fit failure check on the real draw, the six stub assertions, the two NB2 end-to-end
  stub mixes, and one endpoint assertion. The stub testsets pass on main by construction: they test
  `_bootstrap_refit_ok`, not the closures, which is why the parity testset carries the NB2 routes.
- The real-draw testset asserts the direct fit is a genuine failure, that the old bare-vector
  contract accepts its θ, and that the new contract rejects it.
- The endpoint-equality testset asserts all 10 replicates converge and that the bounds are finite
  before comparing, so it cannot pass vacuously.
- `!haskey(raw, :upper_boundary)` pins the deliberate absence of the flag.

## 7a. Issue Ledger

- #504 stays open (part of).

## 8. Consistency Audit

- No other #504 or verdict branch on `origin` touches these three closures (checked with
  `git diff origin/main...<branch> -- src/confint_family.jl` over the 15 such branches).
- The sibling slices all add a CHANGELOG bullet at the same place and an include next to the Poisson
  one in `test/runtests.jl`, so textual merge conflicts are expected there; both are additive.
- The refits call the fitters with `K`, `link`, `mask` (and `hessian` for NB2) and defaults for every
  other keyword. Unchanged here.

## 9. What Did Not Go Smoothly

- The first test draft used the `10^18` draw on all three routes and failed on both NB2 routes: the
  probe had checked only β and Λ for finiteness, not `log(r)`. A second probe that printed `r` showed
  the underflow, and a third, on both Julia versions, showed the two-cell placements were fragile.

## 10. Known Residuals

- On valid data with one very large count, the truncated-NB2 fitters report `converged = true` with a
  degenerate size estimate: one `10^13` cell gave shared `r = 1.5e-109`, and per-trait
  `r = [3e-72, 6e14, 3.4, 2e78]`. This is outside the bootstrap and not changed.
- The refits pass no `offset`, so a fit made with an offset is refit without it (pre-existing; not
  probed).
- Full suite not run; CI will run it.

## 11. Team Learning

When a probe checks whether a failing refit keeps a finite θ, check the whole packed θ, including
log-scale dispersion, and repeat it on each Julia version the test runs on.

## 12. Cross-Product Coverage

Covered: `TruncatedPoissonFit`, `TruncatedNegBin2Fit`, `TruncatedNegBin2PerTraitFit`, log link,
no mask, K = 1, Julia 1.10.12 and 1.13.0, aarch64 macOS, with and without bounds checks on 1.10.12.
Not covered: Linux and Windows (CI), K > 1, a mask, offsets, `hessian = :fisher`, and a real failing
draw on the two NB2 routes.
