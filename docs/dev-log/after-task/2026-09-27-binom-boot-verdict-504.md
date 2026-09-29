# After-task: Binomial bootstrap refit reports its own verdict (part of #504, 2026-09-27)

Lane: Claude, `LANE_ID=claude:binom-boot-verdict-504:6081`, branch `claude/binom-boot-verdict-504` from
`origin/main` @ `cb5688f7e`, worktree `.worktrees/binom-boot-verdict-504`. Done while Shinichi was away
("keep going autonomously"), continuing his "start on migrating the next family for #504".

## 1. Goal

Move the Laplace-route Binomial bootstrap refit closure onto the #508/#516 contract, so a refit that did
not converge is excluded from the bootstrap.

## 2. Implemented

- `src/confint_family.jl`: the Laplace `refit` closure in `_family_ci(fit::BinomialFit, ...)` returns
  `(θ = vcat(fb.β, pack_lambda(fb.Λ)), converged = fb.converged, loglik = fb.loglik)`. The AGHQ route
  already returned `nothing` for a non-converged refit and is unchanged.
- `test/test_confint_bootstrap_verdict_binomial.jl` (new, registered after the Poisson file).
- `CHANGELOG.md`: entry after the Poisson #516 one.

## 3a. Decisions and Rejected Alternatives

- **Based on main, not stacked on #550.** Binomial has no dispersion parameter, so there is nothing to
  flag with `upper_boundary`; the plain #508 contract is enough, and this PR is independent of #550.
- **Seeded simulation, no fixture**, as #516 did: every assertion is a relation, so the seeded draw
  differing across Julia versions does not matter.

## 4. Files Touched

- `src/confint_family.jl` (modified, one line)
- `test/test_confint_bootstrap_verdict_binomial.jl` (new)
- `test/runtests.jl` (modified, one include)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-binom-boot-verdict-504.md` (this file)

## 5. Checks Run

`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, per-file, full suite not run.

- RED, new file on `origin/main` cb5688f7e, Julia 1.10.12: 7 pass, 8 fail, 14 error of 29.
- GREEN: 29/29 on Julia 1.10.12 and 29/29 on Julia 1.13.0.
- Neighbours on 1.10.12, each file alone: `test_aghq_public_binomial.jl` 108/108, `test_bridge_ci.jl`
  65/65, `test_bridge_x.jl` 200/200, `test_se_machinery.jl` 1096/1096, `test_confint_family.jl` 341/341,
  `test_lv_ci.jl` 196/196 (2006 total).

## 6. Tests of the Tests

- The failing-draw testset asserts both halves: the old bare-vector contract accepts the sentinel refit
  (`_bootstrap_refit_ok(raw_bad.θ, m)` is true, since θ is the finite warm start) and the new one rejects it.
- The endpoint-equality testset asserts `n_converged >= 10` and finite bounds before comparing, because
  `_family_bootstrap` returns `NaN` below 10 usable replicates.

## 7a. Issue Ledger

- #504 stays open (part of).

## 8. Consistency Audit

- No open PR touches the Binomial closure.
- `BinomialFit` has one Laplace `_family_ci` method; no grouped or covariate Binomial CI adapter exists.

## 9. What Did Not Go Smoothly

- A zsh loop over an unquoted file list ran nothing (zsh does not word-split `$L`); rerun with an
  explicit list.
- `git worktree add` on Dropbox outlasted the shell timeout; a status taken mid-checkout looked like a
  mass deletion. It was not: the checkout finished and only the intended file changed.

## 10. Known Residuals

- A Bernoulli refit that separates (loadings running off) and still reports `converged = true` is not
  caught; that is a point-fit question, not the bootstrap contract.
- Full suite not run; CI will run it.

## 11. Team Learning

Families without a dispersion boundary need only the one-line contract change; the test pattern from
#516 carries over unchanged.

## 12. Cross-Product Coverage

Covered: Laplace-route `BinomialFit`, Julia 1.10.12 and 1.13.0, aarch64 macOS. Not covered: Linux and
Windows (CI), `objective = :va` refits (the refit always uses the Laplace fitter), masked data.
