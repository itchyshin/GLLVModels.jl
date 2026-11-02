# After Task: Family CI bootstrap leftover (#774)

## Goal

Stop `_family_bootstrap` and `_family_profile_refit` from swallowing
InterruptException, and raise ArgumentError before the bootstrap loop
when simulate is the "not supported" stub.

## Implemented

`_family_bootstrap` probes `simulate` once. A "not supported" error
becomes `ArgumentError` before any replicate. InterruptException is
rethrows from the probe, each replicate, and `_family_profile_refit`.
Ordinary replicate failures still become a failed draw. Adapter-level
`catch; return 1e12` sites are unchanged, so this is a leftover slice,
not a full close.

## Mathematical Contract

Unchanged intervals. Only error handling.

## Files Changed

- `src/confint_family.jl` - `_family_catch_or`, bootstrap probe, profile
  refit rethrow
- `test/test_interrupt_rethrow.jl` - five new assertions

## Tests Added

Injected InterruptException and a "not supported" stub on `_FamilyCI`.
Would have returned all-NaN / n_converged = 0 before. Clause: failure
path.

## Benchmark Numbers

N/A - one extra simulate probe per bootstrap call.

## R-Parity Verdict

Parity: N/A - change does not touch the parity surface.

## JET / Allocs / Aqua Verdicts

- JET: not run
- Allocs: not run
- Aqua: not run

## Checks Run

```
julia --project=. -e 'include("test/test_interrupt_rethrow.jl")'
```

9 passed, 0 failed, 0 errored.

## Consistency Audit

`rg "InterruptException && rethrow" src/confint_family.jl` now hits the
Wald Hessian (already present), profile refit, and bootstrap. Many
adapter `catch; return 1e12` sites remain.

## GitHub Issue Maintenance

Refs #774. Mixed and ordinal sites were fixed in #956. This slice
covers the named leftover drivers. Adapter nll catches stay open.

## What Did Not Go Smoothly

A full pass over every adapter catch would be its own PR.

## Team Learning

A "not supported" stub inside a catch-all loop hides the message.
Raise it before the loop.

## Remaining Risks

- About 100 adapter `catch` sites still swallow InterruptException.
- Probe `occursin("not supported", msg)` is string-based.

## Known Limitations

Does not add a stored first-error field on the CI table.

## Next Command

A follow-up that wraps adapter nll/refit catches with `_family_catch_or`.

## Rose Verdict

Rose verdict: PASS WITH NOTES - named leftover drivers fixed; adapter
catches remain, so Refs not Closes.
