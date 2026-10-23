# 2026-09-23 — Latte-kernel S4 Totoro certify

| Field | Value |
|---|---|
| #448 | MERGED `35790b91d` (2026-09-23T11:31:57Z) |
| Branch | `cursor/latte-kernel-20260923` @ `c151fb348` |
| PR | draft #449 |
| Host | Totoro (`totoro.biology.ualberta.ca`; ControlMaster) |
| Julia | 1.12.6; `JULIA_NUM_THREADS=4` `OPENBLAS_NUM_THREADS=1` |
| Command | `julia --project=. test/test_latte_kernel_identity.jl` |
| Result | **26 Pass / 0 Fail**; `GATE S4 latte-kernel identity complete` |
| rtol | 1e-8 (Poisson, NB2 direct+fit, phylo-correlated W) |
| Receipt | `docs/dev-log/simulation-artifacts/2026-09-23-latte-kernel-retime/s4-identity-totoro.log` |
| TSV | `…/S4-identity.tsv` (laptop + totoro rows) |
| Defaults | `diag_precision_kernel=false`; `nelder_mead=true` (NM-alone not used) |
| Public claim | none (no README/NEWS speed) |

Laptop twin: same file → 26/26 in `s4-identity.log`.
