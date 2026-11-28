# After-task: simulate(COMPoissonFit, n) — refs #1533

## Goal

Fix `MethodError` on `simulate(fit::COMPoissonFit, n)` while `predict` already worked.

## Change

- `src/families/com_poisson.jl`: unexported `_compoisson_sample(rng, η, ν)` — inversion on the truncated normalised pmf (`compoisson_logpdf` / `compoisson_logz`, same `_CMP_LOGZ_TOL` / `_CMP_LOGZ_CAP` as the likelihood). `ν = 1` → `Poisson(λ)`.
- `src/simulate_fit.jl`: `simulate(fit::COMPoissonFit, n; rng)` — fresh `z_s ~ N(0,I_K)`, sample counts at `exp(η)`.
- `test/test_simulate_compoisson.jl`: pmf vs empirical (80k draws, `MersenneTwister(1533)`) for `ν ∈ {0.6, 1.0, 1.8}`; fit-level simulate reproducibility and loose rate sanity vs `predict(..., type=:rate)`.

## Verification

```text
julia --project=. --startup-file=no -e 'include("test/test_simulate_compoisson.jl")'
```

## Overlap

No open PR with `1533` in body. PR #1474 touches COMPoisson post-fit elsewhere (fitted/residuals), not simulate.

## Rose

Julia-only simulate claim; no R twin Δ; no new public API surface.