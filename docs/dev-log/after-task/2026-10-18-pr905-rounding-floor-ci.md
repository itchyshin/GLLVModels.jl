# After Task: PR #905 reviewer unblock (refs #822)

## Goal

Address the BLOCKED review on PR #905: test the issue's own DEP cell, never
report `converged = true` when `gradient_norm > g_tol`, leave the 1e-3
acceptance untouched, and rebase onto current `main`.

## Implemented

1. `_phylo_latent_accept_rounding_floor` still uses the existing floor check
   (PD Hessian, Newton decrement, `gmax <= 100 * max(g_tol, 1e-5)`). It now
   returns the fit unchanged when `fit.gradient_norm > g_tol`, so
   `converged = true` is never paired with a stored gradient above the gate.
2. The 1e-3 line at `src/phylo_latent.jl` (`_PHYLO_LATENT_FLOOR_GRAD_FACTOR *
   max(g_tol, 1e-5)`) is unchanged.
3. `test/test_phylo_latent_rounding_floor.jl` runs the issue's own default-start
   DEP fit (`d = 4`, 120 tips, `test/fixtures/cov_phylo_twins_p1.toml`) and
   asserts `converged`, the `gradient_norm > g_tol` invariant, and stationarity.
4. Branch rebased onto current `origin/main`.

Not touched: `src/em_phylo.jl`, `src/em_squarem.jl` (#868). No new PR. Not
merged.

## Mathematical Contract

The floor *eligibility* rule is unchanged (Kristensen-style Newton decrement vs
raw `g_tol`). The public flag obeys `converged ⇒ gradient_norm <= g_tol`.

## Files Changed

- `src/phylo_latent.jl`
- `test/test_phylo_latent_rounding_floor.jl`
- `test/test_cov_phylo_twins_p1.jl`
- `test/runtests.jl` (include from the original #905 commit)
- `CHANGELOG.md`
- `docs/dev-log/check-log.md`
- `docs/dev-log/after-task/2026-10-18-pr905-rounding-floor-ci.md`

## Tests Added

- Issue #822 DEP `d = 4`, 120-tip default-start fit (would fail on a
  `converged = false` / `gradient_norm > g_tol` report).
- Invariant: `!(converged && gradient_norm > g_tol)` on the forcing
  `g_tol = 1e-9` path and on the three COV-PHYLO twins.

## Benchmark Numbers

`N/A — no hot-path change`. The extra gate is one comparison after the existing
floor check.

## R-Parity Verdict

`Parity: N/A — change does not touch the parity surface`. The DEP cell still
matches R logLik to `1e-6`.

## JET / Allocs / Aqua Verdicts

- JET / Aqua: via `Pkg.test()`; see Checks Run.
- Allocs: not run.

## Checks Run

- `test/test_phylo_latent_rounding_floor.jl`: **14 + 10 passed / 0 failed**
  (Julia 1.10.10). DEP probe: `converged=true`,
  `stopping_reason=:converged`, `gradient_norm=6.2049e-6` (`<= 1e-5`).
- `test/test_cov_phylo_twins_p1.jl`: **40 passed / 0 failed**.
- Full `Pkg.test()`: tally filled after the local canonical run.

Because the issue's own repro passed locally, the PR body keeps `Closes #822`.

## Consistency Audit

- `rg rounding.floor|#822` in `src/phylo_latent.jl`, the two test files, CHANGELOG.
- Line `gmax <= _PHYLO_LATENT_FLOOR_GRAD_FACTOR * max(g_tol, 1e-5)` unchanged.
- Historical COV-PHYLO CHANGELOG note that the `d = 4` fit reported
  `converged = false` left as written (#810).

## GitHub Issue Maintenance

No issue action. Body keeps `Closes #822` because the DEP repro passed.
Wrong-close guard still fails until Shinichi adds `intended-closes` or
rewrites the keyword. Did not merge.

## What Did Not Go Smoothly

The original #905 override set `converged = true` while leaving
`gradient_norm` above `g_tol`. That is now refused. On this Linux 1.10.10
box the DEP cell already lands inside `g_tol` after Newton polish
(`6.2e-6`); the platform-dependent stall in the issue may still appear on
other BLAS. If CI's DEP cell reports `converged = false`, the body must
become `Refs #822`.

## Team Learning

A rounding-floor *eligibility* rule and the public `converged` flag have to
be specified separately when the stored gradient is the user's diagnostic.

## Remaining Risks

- Wrong-close guard red while the body says `Closes #822`.
- The 1e-3 eligibility bound is still Shinichi's.
- DEP `converged = true` here may not reproduce on every CI BLAS.

## Known Limitations

The override never flips a fit whose stored `gradient_norm` is still above
`g_tol`, so a user on a BLAS that stalls at `4e-5` still sees
`converged = false`. That is the reviewer's (2).

## Next Command

Wait for CI on the rebased head. Do not merge. Keep the PR draft.

## Rose Verdict

Rose verdict: PASS WITH NOTES — (2) is implemented; the DEP repro passed on
this box so `Closes #822` stays; the 1e-3 bound and the Wrong-close guard
remain Shinichi's.
