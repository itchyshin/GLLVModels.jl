# After-task: Latte diag_precision_kernel default ON (2026-09-23)

Active lenses: Shannon, Ada, Rose (perspectives). Spawned subagents: none.
Lane: `cursor/latte-default-on-20260923`.

## Scope

Flip `diag_precision_kernel` default from `false` to `true` after A1 walls
cleared the locked gate (≥1.5× on ≥1 certify cell + identity).

## Evidence cited

- Walls tip `4e976e259`: `glmm_200x5` 2.90×, `glmm_5000x3_g500` 1.18×; `|Δll|=0`.
- S4 identity 26/26 (#449).
- Packet: `docs/dev-log/evidence/2026-09-23-latte-off-on-wall/`.

## Change

`src/grouped_nongaussian_fit.jl`: all `diag_precision_kernel::Bool` defaults
`true`; docstring updated. Opt-out remains `diag_precision_kernel=false`.

## Rose

No README/NEWS speed claim. Large-cell 1.18× is not advertised. Behaviour flip
only.

## Owner note

DRAFT PR only — awaits Shinichi G0 before any merge. Do not merge from this lane.
