# After-task: NB1 bootstrap refits report their own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:nb1-boot-verdict-504:6081`, branch `claude/nb1-boot-verdict-504` from
`origin/main` @ `cb5688f7e`, worktree `.worktrees/nb1-boot-verdict-504`. Done while Shinichi was away
("keep going autonomously"), continuing "start on migrating the next family for #504".

## 1. Goal

Move the three NB1 bootstrap refit closures onto the #508/#516 contract, so a non-converged refit is
excluded from the bootstrap.

## 2. Implemented

- `src/confint_family.jl`: the `refit` closures in `_family_ci` for `NB1Fit`, `NB1GroupedFit` and
  `NB1GroupedCovFit` return `(θ, converged = fb.converged, loglik = fb.loglik)`.
- `test/test_confint_bootstrap_verdict_nb1.jl` (new, registered after the Poisson file), built from the
  Gamma test (#565).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **Contract-only, based on main.** `grouped_dispersion.jl` notes that NB1's `φ → ∞` end is flat
  ("Fisher information in log φ vanishes"), so an upper flag would be valid, but the boundary seen in
  practice is the Poisson limit `φ → 0`, a lower boundary the #542 upper-only flag cannot express. A
  lower-boundary flag is a design question for Shinichi.
- **Endpoint test on the shared route.** A one-group grouped NB1 fit took 58 s under this machine's
  load; 20 refits would add about 20 minutes, so the endpoint-equality testset uses `NB1Fit`.
- **Bad draw = a negative count** (probed: `converged = false`, `loglik = -Inf`, finite θ at iteration 0
  on all three routes).

## 4. Files Touched

- `src/confint_family.jl` (modified, three closures)
- `test/test_confint_bootstrap_verdict_nb1.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-nb1-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, per-file, full suite not run.

- Probe (Julia 1.10.12, p = 5, n = 120, K = 1, φ = 2): all three routes converge (φ̂ 2.22 to 2.23).
- RED, new file on `origin/main` cb5688f7e, Julia 1.10.12: 7 pass, 14 fail, 15 error of 36.
- GREEN: 36/36 on Julia 1.10.12 (716 s) and 36/36 on Julia 1.13.0 (832 s). The file is slow on this
  loaded machine (load average above 100): the one-group grouped NB1 fits dominate.
- Neighbours on 1.10.12, each file alone: `test_bridge_grouped_dispersion.jl` 129/129,
  `test_bridge_missing_mask.jl` 92/92, `test_grouped_hessian_consistency.jl` 23/23,
  `test_confint_family.jl` 341/341 (585 total).

## 6. Tests of the Tests

- The failing-draw testset asserts the old bare-vector contract accepts the sentinel refit and the new
  one rejects it; the endpoint-equality testset uses `n_boot = 10` with finite bounds asserted first;
  `!haskey(raw, :upper_boundary)` pins the deliberate absence of the flag.

## 7a. Issue Ledger

- #504 stays open (part of).

## 8. Consistency Audit

- No open PR touches the NB1 closures.

## 9. What Did Not Go Smoothly

- Nothing new.

## 10. Known Residuals

- Lower-boundary (`φ → 0`) replicates are excluded, not signalled; see 3a.
- Full suite not run; CI will run it.

## 11. Team Learning

A dispersion parameter can have its practically relevant boundary at the lower end (NB1 Poisson limit);
the #542 flag only covers upper ends.

## 12. Cross-Product Coverage

Covered: all three NB1 fit types, Julia 1.10.12 and 1.13.0, aarch64 macOS. Not covered: Linux and
Windows (CI), non-log links, masked data, the grouped route in the endpoint test.
