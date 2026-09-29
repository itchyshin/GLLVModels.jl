# After-task: Tweedie bootstrap refits report their own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:tweedie-boot-verdict-504:6081`, branch `claude/tweedie-boot-verdict-504`
from `origin/main` @ `cb5688f7e`, worktree `.worktrees/tweedie-boot-verdict-504`. One slice of the
#504 per-family migration, following the Poisson (#516) and Gamma (draft #565) pattern.

## 1. Goal

Move the three Tweedie bootstrap refit closures onto the #508/#516 contract, so a non-converged refit
is excluded from the bootstrap.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `TweedieFit`, `TweedieGroupedFit`
  and `TweediePerTraitPowerFit` return `(θ, converged = fb.converged, loglik = fb.loglik)`. The θ
  vector is unchanged.
- `test/test_confint_bootstrap_verdict_tweedie.jl` (new, registered after the Poisson file).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **No `upper_boundary` flag.** The CI layer holds the power fixed at its fitted value and profiles
  only φ on the log scale, and `_tweedie_verdict` already reports a power at the closed end of (1, 2)
  as `converged = false` (`:power_at_boundary`), so a separate flag would duplicate it.
- **Bad draw is one cell at 1e300, not a negative value.** A negative cell throws a `DomainError`
  (log of a negative number in the warm start) and a `NaN` or `Inf` cell throws a LAPACK
  `ArgumentError` on all three routes; a throw is already mapped to `nothing` and does not exercise the
  sentinel path. A 1e300 cell gives `converged = false`, `loglik = -Inf`, finite θ, and no throw.
- **Endpoint-equality test on the shared route**, the fastest (per-trait power fits take about 2.5
  times as long).
- **Seeded simulation, no fixture**, as #516: every assertion is a relation.

## 4. Files Touched

- `src/confint_family.jl` (modified, three closures)
- `test/test_confint_bootstrap_verdict_tweedie.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-tweedie-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, load average about 95, per-file, full
suite not run.

- Probe (Julia 1.13.0, `_tweedie_sample`, φ = 1, power = 1.5, seed 71). p = 5, n = 100: shared,
  one-group shared-power and one-group per-trait-power fits all `converged = true` (84 s, 84 s, 159 s
  per fit). p = 4, n = 50 (the test size): all three `converged = true` (31 s, 29 s, 77 s), and refits
  on three parametric-bootstrap draws each also `converged = true` (about 26 s, 26 s, 70 s per refit).
  Bad cells: -1, `NaN` and `Inf` throw on every route; 1e300 gives the soft failure on every route in
  under 0.1 s with the "could not be evaluated at any accepted point" warning.
- RED, new file on `origin/main` cb5688f7e, Julia 1.10.12: 7 pass, 14 fail, 18 error of 39.
- GREEN: 39/39 on Julia 1.10.12 (1398 s) and 39/39 on Julia 1.13.0 (1482 s).
- Neighbours on 1.10.12, each file alone: `test_confint_family.jl` 341/341,
  `test_grouped_hessian_consistency.jl` 23/23, `test_known_sentinel_defects.jl` 25 pass + 1 broken
  (the pre-existing `@test_broken`, as recorded for the Gamma slice), `test_second_order_tweedie_grouped_ci.jl`
  18 pass + 1 broken (the `@test_skip` for the live R cell, which needs `GLLVM_PARITY_TESTS=1`).

## 6. Tests of the Tests

- The failing-draw testset asserts the old bare-vector contract accepts the sentinel refit and the new
  one rejects it, and that the sentinel θ is finite (so the old contract really would keep it).
- The endpoint-equality testset uses `n_boot = 10` and asserts finite bounds before comparing.
- `!haskey(raw, :upper_boundary)` pins the deliberate absence of the flag.
- The RED run shows the file fails on main in every testset.

## 7a. Issue Ledger

- #504 stays open (part of).

## 8. Consistency Audit

- `test_confint_family.jl`'s Tweedie bootstrap asserts `n_converged ≥ 4` of 8; it still passes with the
  stricter count.
- The grouped closure's `power_fixed` branch is unchanged; the test covers only the estimated shared
  power route for `TweedieGroupedFit`.

## 9. What Did Not Go Smoothly

- Running `Pkg.instantiate()` on Julia 1.10 and 1.13 at the same time in one worktree left a
  1.13-resolved `Manifest.toml` that 1.10 could not precompile (`StaticData` not defined). Worked
  around with a 1.10-resolved `Manifest-v1.10.toml` copied from the scratch main worktree (untracked,
  removed afterwards).
- The machine was heavily loaded, so each run of the new file took about 23 to 25 minutes.

## 10. Known Residuals

- The new file is slow for a unit test (about 23 minutes here; mostly the ten-replicate endpoint test
  and the per-trait-power route). CI runners are unloaded and should be several times faster, but it is
  worth watching the shard time.
- Full suite not run; CI will run it.

## 11. Team Learning

When a failure sentinel must be reached without a throw, probe several bad values: for Tweedie, a
negative, `NaN` or `Inf` cell throws, but an extreme finite value reaches the sentinel.

## 12. Cross-Product Coverage

Covered: all three Tweedie fit types, Julia 1.10.12 and 1.13.0, aarch64 macOS. Not covered: Linux and
Windows (CI), non-log links, masked data, fixed-power grouped refits, groups larger than one.
