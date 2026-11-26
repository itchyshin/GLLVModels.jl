# After-task — #1493 simulate for ZIP / ZINB / Hurdle-Poisson fits

## Goal

Add public `simulate(fit, n; rng=...)` methods for `ZIPFit`, `ZINBFit`, and
`HurdlePoissonFit`, matching the bootstrap parametric samplers already in
`confint_family.jl` and the `PoissonFit` / `NBFit` calling convention.

## Changes

| File | Change |
| --- | --- |
| `src/simulate_fit.jl` | Three `simulate` methods + `_twopart_sim_offset` guard for stored p×n count offsets |
| `test/test_simulate_twopart.jl` | Issue repro (shape, `Int` ≥ 0, RNG reproducibility) |
| `test/runtests.jl` | Register new test file (both shard lists) |

## Tests (cloud VM, Julia 1.13.4)

- `julia --project=. test/test_simulate_twopart.jl` — **9 / 9 pass**, 31.3 s
- `julia --project=. test/test_simulate.jl` — **5 / 5 pass**, 27.0 s (unchanged)

Full `Pkg.test()` not run (time budget on 16 GB cloud VM).

## R-parity / twin

Julia-only post-fit convenience; no gllvmTMB twin route. No RCall Δ.

## Rose verdict

**PASS WITH NOTES** — docstrings added on the three new methods; capability
matrix / tutorial simulate columns were already honest via prior audits; full
suite and Documenter not re-run in this session.