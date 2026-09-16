# D1 remeasure runbook — Delta Option A (post-paste only)

**Status:** **PASTE-GATED** — run only after maintainer paste `accept delta dispersion A` and #399 post-paste closeout (public default coerce + ACCEPTED block live).

**Contract:** [`second-order-parity-contract.md`](../core070/second-order-parity-contract.md) §4 — **do not widen rtol**.

**Prerequisite:** DRAFT [#399](https://github.com/itchyshin/GLLVM.jl/pull/399) merged with `core070_second_order` Delta cells at `disp_group=:species` and Julia-side Wald packing green.

## Commands (local Mac; R live Δ needs `GLLVM_PARITY_TESTS=1`)

```bash
cd "/Users/z3437171/Dropbox/Github Local/GLLVM.jl"
export GLLVM_PARITY_TESTS=1   # optional; required for R paired Δ

julia --project=. test/test_second_order_delta_followup.jl

julia --project=. tools/core070_second_order/smoke_delta_lognormal_eoo.jl \
  docs/dev-log/core070/delta-lognormal-2so-d1-remeasure-receipt.json

julia --project=. tools/core070_second_order/smoke_delta_gamma_eoo.jl \
  docs/dev-log/core070/delta-gamma-2so-d1-remeasure-receipt.json
```

## Pass criteria (D1 tier — not EOO smoke alone)

| Cell | Gate | Prior FAIL class (2026-09-15) |
|------|------|-------------------------------|
| `delta_lognormal` seed 61 | §4 SE / vcov / CI endpoint Δ vs R at own optimum | logLik Δ −1.923; SE rel 0.145 |
| `delta_gamma` seed 62 | same | SE rel 0.212 |

Record outcomes in `docs/dev-log/after-task/YYYY-MM-DD-delta-dispersion-a-d1-remeasure.md`. **Do not** claim D1 pass in decision doc or README until both cells meet §4 at remeasured values.

## Rose fence

- EOO smoke `eoo_smoke_pass` ≠ D1 programme gate.
- Remeasure failure → fix engine/identity; **never** widen §4 tolerances to green-wash.
