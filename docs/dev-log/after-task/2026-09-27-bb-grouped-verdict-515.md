# After-task: grouped beta-binomial verdict (part of #515, 2026-09-27)

Lane: Claude, `LANE_ID=claude:GLLVM.jl:6081`, branch `claude/bb-grouped-verdict-515`
from `origin/main` @ `52ed4281b` (the #522 merge; rebased onto `824d22a4b`), worktree
`.worktrees/bb-grouped-verdict-515`.

## 1. Goal

PR #522 gave `fit_beta_binomial_gllvm` a per-family verdict (`_beta_binomial_verdict`)
and a stable log-pmf at `φ >= 1e6`. The two grouped routes,
`fit_beta_binomial_gllvm_grouped` and `fit_beta_binomial_gllvm_grouped_cov`, got the
stable log-pmf but kept the shared `_fit_verdict`. Apply the same verdict to both
without editing `_fit_verdict` or `_laplace_mode`.

## 2. Implemented

- `src/families/beta_binomial.jl`: new `_beta_binomial_grouped_verdict(nll,
  optim_converged, iterations, φg)`. It runs the shared `_fit_verdict` screen first,
  then `_beta_binomial_verdict` at `maximum(φg)`, and returns `(loglik, converged,
  iterations)` in `_fit_verdict`'s order. Both grouped fitters call it in place of
  `_fit_verdict(res)`. Nothing else changes: the optimisation path is identical, so
  only the reported flag (and, for an impossible objective, the loglik) can differ.
- `test/test_beta_binomial_grouped_verdict_515.jl` (new, registered in
  `test/runtests.jl` after the #522 file): helper on recorded absurd states, three
  live boundary fits, twelve healthy fits. It reuses #522's hash-verified fixture;
  no new data.
- `docs/src/low-level-reference.md`: the new internal docstring is listed
  (Documenter runs with `warnonly = false`).
- `CHANGELOG.md`: new entry; the #522 entry's "do not yet have the verdict gate"
  sentence now points to it.

## 3a. Decisions and Rejected Alternatives

- **Compose, not replace.** `_beta_binomial_verdict` alone treats `nll` in
  `[1e11, 1e12)` as a real value, while `_fit_verdict` already rejects it. Running
  `_fit_verdict` first keeps that screen, so the grouped gate is strictly stronger
  than main's.
- **`maximum(φg)`, not a per-group rule.** One group at the boundary is enough:
  that group's precision is not identifiable, whatever the others do. This matches
  `_dispersion_group_boundary`'s "any group" rule for NB1/NB2.
- **Rejected: a boundary restart** (as #477 did for NB2). It would change fitted
  values, which is out of scope for a verdict fix.
- **Rejected: a new fixture.** The #522 fixture already contains datasets that hit
  the boundary on the grouped routes on both Julia versions.

## 4. Files Touched

- `src/families/beta_binomial.jl` (modified)
- `test/test_beta_binomial_grouped_verdict_515.jl` (new)
- `test/runtests.jl` (modified, one include)
- `docs/src/low-level-reference.md` (modified, one line)
- `CHANGELOG.md` (modified)
- `docs/dev-log/check-log.md` (modified)
- `docs/dev-log/after-task/2026-09-27-bb-grouped-verdict-515.md` (this file)

## 5. Checks Run

All with `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, aarch64 macOS, per-file
includes (not the full suite).

- RED, new test file against `origin/main` 52ed4281b source, Julia 1.10.12: 73 pass,
  3 fail, 5 error. The 3 failures are `!fit.converged` on the three live boundary
  fits; the 5 errors are the helper not existing. All 12 healthy fits pass on main.
- GREEN, the five beta-binomial test files (`test_beta_binomial.jl`,
  `test_beta_binomial_mode_search.jl`, `test_beta_binomial_verdict_515.jl`,
  `test_beta_binomial_grouped_verdict_515.jl`, `test_betabinomial_x_identity.jl`):
  906/906 on Julia 1.10.12 and 906/906 on Julia 1.13.0.
- Main vs branch, 52 grouped fits per version (13 fixture datasets x per-species and
  one-group x both routes), same script on main and branch. Julia 1.10.12: all 49 fits
  with every φ below 1e6 on main are bitwise identical on the branch (converged,
  loglik, φ, iterations); the 3 boundary fits keep loglik and φ exactly and change
  `converged` from true to false. Julia 1.13.0: 47 identical, 5 boundary fits (the
  same 3 plus 9002 per-species on both routes) change only the flag.
- Documenter build (`julia +1.10 --project=docs docs/make.jl`, `warnonly = false`): exit 0, no errors; `_beta_binomial_grouped_verdict` renders on the low-level reference page.
- After rebasing onto `origin/main` @ `824d22a4b` (brings #530, a docstring-only
  reword in this file, plus #523/#524 parity tooling): Documenter rebuilt (exit 0,
  92 s); `tools/check_reader_surface.py --rendered docs/build/1` and
  `--landing-contract` both pass. Tests were not re-run after the rebase: the only
  upstream change to `src/` since 52ed4281b is that docstring.

## 6. Tests of the Tests

The live testset fails on main for the stated reason (`!fit.converged`, 3 of 3) and
passes on the branch; it also asserts `maximum(fit.φ) >= 1e6`, so it cannot pass
vacuously if a future change stops the fit from reaching the boundary. The healthy
testset passes on both main and branch, which is its job. The helper testset covers
the shared-screen composition (`nll = 5e11` and `NaN` still give `-Inf`), so dropping
the `_fit_verdict` step would fail it.

## 7a. Issue Ledger

- #515: grouped routes now gated. This PR says "part of #515" and does not close it.
- No new issue opened.

## 8. Consistency Audit

- No other beta-binomial fitter uses the bare `_fit_verdict` (grep of `src/`).
- `confint` bootstrap refits for both grouped routes (`src/confint_family.jl`,
  `_family_ci` for `BetaBinomialGroupedFit` and `BetaBinomialGroupedCovFit`) keep a
  refit's estimates without looking at its `converged` flag. Unchanged here; the
  Poisson analogue was #504.
- The bridge (`_bridge_fit_onepart`, `_bridge_fit_onepart_cov`) and
  `fit_gllvm(...; family = BetaBinom())` route to `fit_beta_binomial_gllvm_grouped`
  with `group = 1:p`, so per-species boundary fits there now report
  `converged = false` too.
- No other open PR touches `src/families/beta_binomial.jl`.

## 9. What Did Not Go Smoothly

- My first shell wait loop matched its own command line under `pgrep -f` and never
  exited; I switched to waiting on the PID.
- The grouped routes did not reproduce #515's positive-loglik state (the stable
  log-pmf removed it), so the live failing case is the φ-boundary branch only. The
  impossible-loglik branch is covered by applying the helper to recorded states.

## 10. Known Residuals

- Per-species fits often stop with a group φ between 1e5 and 1e6 (for example
  seeds 9105, 9106, 9108, 9112, and 9002 on Julia 1.10). They are as unidentified as
  the ones past 1e6 but still report `converged = true`. The 1e6 gate is a boundary
  tripwire, not an identifiability test.
- 9002 per-species crosses 1e6 on Julia 1.13 only, so its flag differs between
  versions after this change (on main it was `true` on both). Its per-species fits are left out of the
  test sets for that reason.
- Full suite not run; CI will run it.

## 11. Team Learning

A per-species dispersion route can drift one group's precision to the numerical
boundary on perfectly ordinary data. Probe the grouped routes as well as the
shared-dispersion route whenever a boundary verdict ships.

## 12. Cross-Product Coverage

Covered: both grouped routes, both groupings (per-species, one group), Julia 1.10.12
and 1.13.0, aarch64 macOS. Not covered: Linux and Windows (CI only), non-logit links,
masked or missing data, `confint` on a boundary fit.
