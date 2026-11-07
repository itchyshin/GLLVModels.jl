# After Task: Reviewer 2 follow-up for #1005 (Refs #948)

## Goal

Unblock PR #1005: track the NB sparse-vs-dense loglik gap, merge main
and resolve conflicts, and run the cheap profile ArgumentError guards
in CI.

## Implemented

1. The NB sparse-augmented vs dense-leaf identity is `@test_broken` at
   `atol = 1e-6`. The observed gap on Julia 1.10.12 is 1.3e-2
   (`ll_sparse = -112.424`, `ll_dense = -112.412`) at the same
   parameters. The contract is not widened. Draft issue text is in
   `docs/dev-log/owed/2026-10-30-nb-xlv-sparse-dense-gap.md`. The NB
   evaluator is not changed.
2. `origin/main` was merged (63 commits). The only conflict was
   `docs/dev-log/check-log.md`; both 2026-10-25 entries were kept.
   `test/runtests.jl` auto-merged and kept the live `_shard_include`
   lines.
3. Profile ArgumentError checks now live in their own testsets, using
   a dummy `(Lambda = zeros(p, K),)` NamedTuple. They do not call a
   fit. The gated `GLLVM_XLV_CANARY` canaries still hold the
   optimiser-path profile-CI cells.

PR body: single `Refs #948`. No `Closes`. No duplicated close/ref
markers.

## Mathematical Contract

Unchanged. Tests and docs only. The 1e-6 sparse-vs-dense identity
remains the contract; it is marked broken, not loosened.

## Files Changed

- `test/test_phylo_nb_xlv.jl` - `@test_broken` identity; CI profile
  ArgumentError testset
- `test/test_phylo_binomial_xlv.jl` - CI profile ArgumentError testset
- `test/runtests.jl` - comment update after merge
- `docs/dev-log/owed/2026-10-30-nb-xlv-sparse-dense-gap.md` - draft
  issue
- `docs/dev-log/check-log.md` - merge resolution plus this entry
- `docs/dev-log/after-task/2026-10-31-wire-xlv-nb-binomial-948-review.md`

## Tests Added

The profile ArgumentError testsets (7 NB + 6 binomial) would have been
skipped by CI when they sat inside the canary gate. They now throw on
empty, duplicate, out-of-range, length-mismatch, and keyword-guard
inputs without a fit. Clause: failure path.

The NB identity is the same assertion as before, now `@test_broken` so
CI records it.

## Benchmark Numbers

N/A. No hot-path change.

## R-Parity Verdict

Parity: N/A. Change does not touch the parity surface.

## JET / Allocs / Aqua Verdicts

- JET: quality testset 11 passed (Aqua + JET via `Pkg.test`)
- Allocs: not run
- Aqua: quality testset 11 passed

## Checks Run

Focused:

```
julia --project=. -e 'include("test/test_xlv_nb_binomial_wired.jl");
  include("test/test_phylo_binomial_xlv.jl");
  include("test/test_phylo_nb_xlv.jl")'
```

42 passed, 1 broken, 0 failed, 0 errored.

Full:

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

37793 passed, 2 failed, 0 errored, 108 broken. 225m26.4s on this
1.10.12 x86_64 snapshot (first-run compile).

This PR's xlv rows in that tally:

- Phylo x Binomial S1: 14 passed
- Phylo x Binomial profile ArgumentError: 6 passed
- Phylo x NB2 S1: 11 passed, 1 broken
- Phylo x NB2 profile ArgumentError: 7 passed
- wiring guard: 4 passed

The two failures are `test/test_masked_gaussian_converged_735.jl`
(`Optim.g_residual` 1.16e-6 vs `g_tol` 1e-6 on both the tight and
loose starts). That file is from main. This PR does not touch the
Gaussian fitter.

## Consistency Audit

`rg "test_phylo_binomial_xlv|test_phylo_nb_xlv" test/runtests.jl` shows
live `_shard_include` calls. Ordinal xlv and `test_sparse_phy_grad.jl`
remain commented. `rg "Closes #948"` is not used in the PR body.

`rg "Gaussian only|not yet implemented|planned next|TODO|FIXME"` on
README.md / CLAUDE.md / docs is historical; no wording in those
user-facing files was changed by this slice.

## GitHub Issue Maintenance

Refs #948. The wiring is in the suite; #948 is not closed here because
the NB sparse-vs-dense gap still has no filed issue. Draft text is in
`docs/dev-log/owed/2026-10-30-nb-xlv-sparse-dense-gap.md`. The issue
was not opened from this lane.

## What Did Not Go Smoothly

The branch was 63 commits behind main. Only `check-log.md` conflicted.
The full suite on a cold depot took about 225 minutes. Two #735
masked-Gaussian assertions failed on this snapshot; they are not part
of this PR's diff.

## Team Learning

A skipped identity behind an env gate that CI never sets hides a
deterministic bug. `@test_broken` keeps the contract and flags a later
fix.

## Remaining Risks

- The 1.3e-2 NB sparse-vs-dense gap is unexplained. The NB path is
  not fixed here.
- Profile-CI canaries still stay behind `GLLVM_XLV_CANARY=1`.
- `test_phylo_ordinal_xlv.jl` and `test_sparse_phy_grad.jl` stay
  unwired.
- The #735 masked-Gaussian gradient residual (1.16e-6 vs 1e-6) failed
  on this 1.10.12 x86_64 run of main's tests.

## Known Limitations

Does not fix the NB evaluator. Does not wire ordinal xlv or
`test_sparse_phy_grad.jl`. Does not change package defaults.

## Next Command

File the draft in `docs/dev-log/owed/2026-10-30-nb-xlv-sparse-dense-gap.md`
as a new issue, then a numerics slice on `_phylo_nb_xlv_marginal_loglik`
vs `_dense_leaf_phylo_nb_xlv_loglik`.

## Rose Verdict

Rose verdict: PASS WITH NOTES. Reviewer-2 items done; NB gap tracked,
not fixed; #735 failures are main's, not this wiring.
