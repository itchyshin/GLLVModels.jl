# After-task: Beta bootstrap refits report their own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:beta-boot-verdict-504:6081`, branch `claude/beta-boot-verdict-504` from
`origin/main` @ `cb5688f7e`, worktree `.worktrees/beta-boot-verdict-504`. Done while Shinichi was away
("keep going autonomously"), continuing "start on migrating the next family for #504".

## 1. Goal

Move the three Beta bootstrap refit closures onto the #508/#516 contract, so a non-converged refit is
excluded from the bootstrap.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `BetaFit`, `BetaGroupedFit` and
  `BetaGroupedCovFit` return `(θ, converged = fb.converged, loglik = fb.loglik)`.
- `test/test_confint_bootstrap_verdict_beta.jl` (new, registered after the Poisson file), built from the
  Gamma test (#565).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **Contract-only, based on main.** A large Beta φ is the near-deterministic end (variance → 0), which
  the data identify, like the Gamma shape (#565), not a flat-likelihood limit like NB2 `r`. So no #542
  `upper_boundary` flag, and no dependence on #550.
- **Bad draw = a value of 1.5.** Probed on Julia 1.10.12: 0, 1 and 1.5 all give `converged = false`,
  `loglik = -Inf` with finite θ at iteration 0 on all three routes; `NaN` throws, which the closure
  already turns into `nothing`.

## 4. Files Touched

- `src/confint_family.jl` (modified, three closures)
- `test/test_confint_bootstrap_verdict_beta.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-beta-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, per-file, full suite not run.

- RED, new file on `origin/main` cb5688f7e, Julia 1.10.12: 7 pass, 14 fail, 15 error of 36.
- GREEN: 36/36 on Julia 1.10.12 and 36/36 on Julia 1.13.0.
- Neighbours on 1.10.12, each file alone: `test_bridge_grouped_dispersion.jl` 129/129, `test_bridge_x.jl`
  200/200, `test_grouped_hessian_consistency.jl` 23/23, `test_lv_ci.jl` 196/196, `test_confint_family.jl`
  341/341, `test_beta_grouped_convergence.jl` 19/19 (908 total).

## 6. Tests of the Tests

- The failing-draw testset asserts the old bare-vector contract accepts the sentinel refit and the new
  one rejects it; the endpoint-equality testset uses `n_boot = 10` with finite bounds asserted first;
  `!haskey(raw, :upper_boundary)` pins the deliberate absence of the flag.

## 7a. Issue Ledger

- #504 stays open (part of).

## 8. Consistency Audit

- No open PR touches the Beta closures.

## 9. What Did Not Go Smoothly

- Nothing new; the Gamma template carried over with renames.

## 10. Known Residuals

- The grouped Beta point fit also applies `_dispersion_group_boundary` (φ > 1e6 reported as not
  converged), the same questionable rule noted for Gamma in #565; not changed here.
- Full suite not run; CI will run it.

## 11. Team Learning

Contract-only migrations are now mechanical: copy the Gamma test, swap the simulator and the bad draw,
check that the bad draw gives a soft failure (not a throw) before relying on it.

## 12. Cross-Product Coverage

Covered: all three Beta fit types, Julia 1.10.12 and 1.13.0, aarch64 macOS. Not covered: Linux and
Windows (CI), non-logit links, masked data.
