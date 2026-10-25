# After Task: Phylo Beta and Ordinal xLV maxiter flag (refs #747)

## Goal

Close the remaining #747 items: flag a maxiter exit in the phylo Beta and
Ordinal xLV mode searches, and stop the Poisson xLV constrained-refit
objective from swallowing InterruptException.

## Implemented

`_phylo_beta_xlv_mode` and `_phylo_ordinal_xlv_mode` now match Poisson,
binomial, Gamma, and NB2: `mode_found` is set only when the damped step
meets `tol`. Exhausting `maxiter` returns `(nothing, nothing, nothing)`,
so the marginal is `-Inf`. `_phylo_poisson_xlv_constrained_refit` rethrows
InterruptException from the objective, the optimiser call, and the final
unpacks. Ordinary errors still become the 1e12 barrier or NaN.

## Mathematical Contract

Unchanged Laplace marginal. The mode used for the log-det term must be a
point that met the Newton tolerance. A search that ran out of iterations
is treated as a failed search, like a non-SPD Hessian.

## Files Changed

- `src/phylo_beta_xlv.jl` - maxiter flag
- `src/phylo_ordinal_xlv.jl` - maxiter flag
- `src/phylo_poisson_xlv.jl` - InterruptException rethrow in constrained refit
- `test/test_phylo_beta_xlv_mode.jl`, `test/test_phylo_ordinal_xlv_mode.jl`
- `test/test_interrupt_rethrow.jl` - constrained-refit cases
- `test/runtests.jl` - wire the two new mode files

## Tests Added

2 files plus 2 assertions in `test_interrupt_rethrow.jl`. Each forces
`maxiter = 1` (would have returned a non-mode before) or an injected
InterruptException (would have become 1e12). Clause: boundary / failure path.

## Benchmark Numbers

N/A - no hot-path change. The extra boolean is not on the inner Newton
iteration cost.

## R-Parity Verdict

Parity: N/A - change does not touch the parity surface.

## JET / Allocs / Aqua Verdicts

- JET: not run (focused mode-search and catch change)
- Allocs: not run
- Aqua: not run

## Checks Run

```
julia --project=. -e 'include("test/test_phylo_beta_xlv_mode.jl");
  include("test/test_phylo_ordinal_xlv_mode.jl");
  include("test/test_interrupt_rethrow.jl");
  include("test/test_phylo_binomial_xlv_mode.jl")'
```

22 passed, 0 failed, 0 errored (6 + 6 + 6 + 4).

## Consistency Audit

`rg "mode_found|maxiter exhausted" src/phylo_*_xlv.jl` - Beta and Ordinal
now match the other four families. No README or capability claim changed.

## GitHub Issue Maintenance

Closes #747 if these were the last named leftovers. The 24 Oct comment
listed only Beta, Ordinal, and the Poisson constrained-refit catch.

## What Did Not Go Smoothly

Julia was missing from the cloud snapshot; 1.10.12 was installed via
juliaup for this run. Full `Pkg.test()` was not run.

## Team Learning

Sibling family mode loops should share the maxiter flag in the first PR,
not in a follow-up wave.

## Remaining Risks

- Other bare `catch` sites in phylo xLV files (Cholesky) still swallow
  InterruptException. Those are Hessian failures, not the named leftover.
- Full suite not run.

## Known Limitations

The constrained-refit objective still turns ordinary errors into 1e12.
That is the existing infeasible-parameter contract, not a silent
argument error.

## Next Command

`julia --project=. test/runtests.jl` on the reviewer's machine if a
full-core pass is wanted before merge.

## Rose Verdict

Rose verdict: PASS WITH NOTES - focused tests green; full suite and JET
not run on this snapshot.
