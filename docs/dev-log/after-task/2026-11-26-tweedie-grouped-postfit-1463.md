# After-task — TweedieGroupedFit post-fit methods (#1463)

## Goal

Wire `predict`, `fitted` (via `StatsAPI.fitted`), `residuals`, and `getLV` for
`TweedieGroupedFit` after `fit_tweedie_gllvm_grouped`, matching the grouped
dispersion pattern (Gamma/NB2) and the scalar `TweedieFit` residual definitions.

## Changes

- `src/families/grouped_dispersion.jl`: route `_grouped_site_mode` for
  `Vector{TweedieED}` through `_tweedie_grouped_mode`; add `getLV` and a
  `_grouped_prediction_offset` overload (no stored training offset on the fit yet).
- `src/families/nb_grouped_postfit.jl`: `predict(::TweedieGroupedFit, …)`.
- `src/postfit.jl`: `residuals(::TweedieGroupedFit, …)` with per-species `φ`.
- `test/test_tweedie_grouped_postfit.jl`; listed in `test/runtests.jl`.

## Verification

```text
julia --project=. test/test_tweedie_grouped_postfit.jl
# Test Summary: #1463 TweedieGroupedFit post-fit … | Pass: 18  Total: 18
```

## Rose

Julia in-sample post-fit wiring only; no API/estimand/default change; no twin Δ claim.