# After-task: Latte OFF vs ON walls (2026-09-23)

Active lenses: Shannon, Ada, Rose (perspectives). Spawned subagents: none.
Lane: `cursor/latte-retime-20260923`.

## Scope

Paired wall `diag_precision_kernel=false` vs `true` on tip after #449/#451.
Gate (locked sequence): ≥1.5× on **at least one** certify cell, plus identity.

## Outcome

1. Tip measured: `4e976e259`.
2. Totoro walls (≥5 warm; threads 4 / BLAS 1; ≤16 cores via taskset):

| cell | OFF (s) | ON (s) | × |
|---|---:|---:|---:|
| `glmm_200x5` | 0.587 | 0.203 | 2.90 |
| `glmm_5000x3_g500` | 10.879 | 9.245 | 1.18 |

3. Identity: `|Δll|=0` on both cells; cite existing S4 **26/26**.
4. Receipts under `docs/dev-log/evidence/2026-09-23-latte-off-on-wall/`.
5. Harness only in the evidence PR: `bench/latte_gap_retime/` (no `src/`).

## Verdict

**FLIP (default ON).** Gate clears: `glmm_200x5` at **2.90×** (≥1.5×) with `|Δll|=0`.
Large cell stays at 1.18× (honest; no public speed claim from it). Owner standing
approval for this goal: merge evidence, then merge default-ON noting the gate.

## Rose

No README/NEWS speed claim. Default-ON is a behaviour flip only; quote walls from
dated TSV, not as a package headline.
