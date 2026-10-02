# NB2 call text: bare `NegativeBinomial()` versus `disp_group = :species` (2026-10-01)

Until itchyshin/GLLVModels.jl#665 merges, a bare
`fit_gllvm(Y; family = NegativeBinomial(...))` fits one dispersion `r` per trait
(`NBGroupedFit`). After #665 a bare call fits one shared `r` (`NBFit`), and the
per-trait fit is requested with `disp_group = :species`. R's `gllvmTMB` nbinom2
twin has one dispersion per trait, so the per-trait call is the right equivalent.

## What this PR changed (executing code)

The parity code that runs a bare NB fit and expects per-trait `r` now passes
`disp_group = :species` explicitly. This does not change behaviour today and keeps
the code correct after #665.

- `tools/core070_data_surface_probe.jl` (call and label string)
- `tools/core070_bridge_models.jl` (NB2 branch only; reads `fit.r_group`)
- `test/test_namespace_numeric_p1_twin_b.jl` (the `nb2` testset; itchyshin/GLLVModels.jl#663)
- `tools/true_parity_julia_receipts.jl` (the `nb2` namespace section and its cited call text)

## What was left as recorded (hash-guarded or frozen)

These files record the bare call as text (`julia_call` strings, labels). Each is
frozen, or its sha256 is checked by a verifier or by receipts, so they were not
edited. In each, the bare NB call recorded there meant per-species `r` when it was
written. After #665 the equivalent call is the same call plus `disp_group = :species`.

- `docs/dev-log/core070/frozen-r070-contract.toml`
- `docs/dev-log/core070/true-parity-latest/frozen-r070-contract-p1.toml`
- `docs/dev-log/core070/registered-models-contract.json`
- `docs/dev-log/core070/family-model-catalogue.json`
- `docs/dev-log/core070/family-required-case-plan.json`
- `docs/dev-log/core070/family-route-contract.json` and `family-route-contract.md`
- `docs/dev-log/core070/family-link-boundary-contract.json`

## Receipts

The `CORE070-DATA-*` receipts and `receipts/data/data-p1/data-batch-julia-surface-probe.json`
were produced by the probe with the per-species meaning (label
`fit_gllvm(Y; family=NegativeBinomial(1.0, 0.5))`). They are executed-call records,
need R to rebuild, and are not rewritten by hand. The probe's label now carries
`disp_group=:species`; the old receipts keep the old label and mean the same fit.
