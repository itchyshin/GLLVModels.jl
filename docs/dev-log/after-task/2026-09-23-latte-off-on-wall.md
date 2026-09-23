# After-task: Latte OFF vs ON walls (2026-09-23)

Active lenses: Shannon, Ada, Rose (perspectives). Spawned subagents: none.
Lane: `cursor/latte-retime-20260923` (measure-only; G0 still open).

## Scope

Paired wall `diag_precision_kernel=false` vs `true` on tip after #449/#451.
No default flip. No merge of a default-ON PR.

## Outcome

1. Tip measured: `4e976e259`.
2. Totoro walls (≥5 warm; threads 4 / BLAS 1; ≤16 cores via taskset):

| cell | OFF (s) | ON (s) | × |
|---|---:|---:|---:|
| `glmm_200x5` | 0.587 | 0.203 | 2.90 |
| `glmm_5000x3_g500` | 10.879 | 9.245 | 1.18 |

3. Identity: `|Δll|=0` on both cells; cite existing S4 **26/26**.
4. Receipts under `docs/dev-log/evidence/2026-09-23-latte-off-on-wall/`.
5. Harness only: `bench/latte_gap_retime/` (no `src/`).

## Recommendation

**Keep opt-in (default OFF).** Gate ~≥1.5× clears on the small cell only.
Large cell 1.18× is below the dual-fixture bar.

## Rose

No README/NEWS speed claim. Default remains `false`. Evidence PR only.
