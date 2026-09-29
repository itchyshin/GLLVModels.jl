# After-task: Gamma bootstrap refits report their own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:gamma-boot-verdict-main-504:6081`, branch `claude/gamma-boot-verdict-main-504`
from `origin/main` @ `cb5688f7e`, worktree `.worktrees/gamma-boot-verdict-main-504`. Done while Shinichi
was away ("keep going autonomously"), continuing "start on migrating the next family for #504".

## 1. Goal

Move the three Gamma bootstrap refit closures onto the #508/#516 contract, so a non-converged refit is
excluded from the bootstrap.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `GammaFit`, `GammaGroupedFit` and
  `GammaGroupedCovFit` return `(θ, converged = fb.converged, loglik = fb.loglik)`.
- `test/test_confint_bootstrap_verdict_gamma.jl` (new, registered after the Poisson file).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **No `upper_boundary` flag for α, so based on main, not stacked on #550.** I first set the branch up on
  top of the NB branch to reuse the #542 flag. A probe then showed that for Gamma a large shape is not a
  flat-likelihood limit: on data drawn with true α = 1e8 (Julia 1.10.12, 3 datasets), the shared,
  one-group and covariate routes estimate α at 8.4e7 to 9.2e7. Flagging α > 1e6 would exclude every
  replicate on such data and report a `NaN`/`Inf` interval, worse than main. So the migration is
  contract-only, like Binomial (#564).
- **Seeded simulation, no fixture**, as #516: every assertion is a relation.

## 4. Files Touched

- `src/confint_family.jl` (modified, three closures)
- `test/test_confint_bootstrap_verdict_gamma.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-gamma-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, per-file, full suite not run.

- Probe (Julia 1.10.12, p = 6, n = 120, K = 2): α = 2 data, seeds 1-3: all four routes converge
  (per-species max α̂ up to 2.1e5 on seed 3, still `converged = true`). α = 1e8 data, seeds 1-3: shared
  fit `converged = true` at α̂ 8.4e7 to 9.2e7; one-group, per-species and covariate grouped fits
  `converged = false` (the grouped `_dispersion_group_boundary` rule, α > 1e6). A zero cell gives
  `converged = false`, `loglik = -Inf` at iteration 0 on all three routes.
- RED, new file on `origin/main` cb5688f7e, Julia 1.10.12: 7 pass, 14 fail, 15 error of 36.
- GREEN: 36/36 on Julia 1.10.12 and 36/36 on Julia 1.13.0.
- Neighbours on 1.10.12, each file alone: `test_bridge_grouped_dispersion.jl` 129/129, `test_bridge_x.jl`
  200/200, `test_confint_hessian_consistency.jl` 12/12, `test_grouped_hessian_consistency.jl` 23/23,
  `test_known_sentinel_defects.jl` 25 pass + 1 broken (the same pre-existing `@test_broken` on main),
  `test_confint_family.jl` 341/341, `test_lv_ci.jl` 196/196.

## 6. Tests of the Tests

- The failing-draw testset asserts the old bare-vector contract accepts the sentinel refit and the new
  one rejects it.
- The endpoint-equality testset uses `n_boot = 10` and asserts finite bounds before comparing.
- `!haskey(raw, :upper_boundary)` pins the deliberate absence of the flag.

## 7a. Issue Ledger

- #504 stays open (part of).

## 8. Consistency Audit

- Open draft PR #363 (2026-09-24) adds new `_family_ci` methods directly after `GammaFit`'s; it does
  not change the lines edited here, but the two hunks are adjacent, so a textual merge conflict is
  possible when both land.

## 9. What Did Not Go Smoothly

- The first Gamma branch was stacked on the NB branch; after the probe it was rebuilt from main under a
  new name, `claude/gamma-boot-verdict-main-504`. The old local branch `claude/gamma-boot-verdict-504`
  is identical to `claude/nb-boot-verdict-504` (no unique commits, never pushed); the destructive-command
  guard blocked `git branch -D`, so it is left for Shinichi to delete.

## 10. Known Residuals

- The grouped Gamma point-fit verdict treats α > 1e6 as a boundary (`_dispersion_group_boundary`) and
  reports `converged = false` although α is well identified there. Near-deterministic positive data is
  rare in practice, but the rule is questionable for Gamma; not changed here.
- Full suite not run; CI will run it.

## 11. Team Learning

Before reusing a boundary flag across families, check that the boundary is a flat-likelihood limit in
that family: it is for NB2 `r → ∞` and beta-binomial `φ → ∞`, but not for the Gamma shape.

## 12. Cross-Product Coverage

Covered: all three Gamma fit types, Julia 1.10.12 and 1.13.0, aarch64 macOS. Not covered: Linux and
Windows (CI), non-log links, masked data, per-species grouped refits in the test (probed only).
