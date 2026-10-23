# 2026-09-23: Latte OFF vs ON wall (measure-only)

| Field | Value |
|---|---|
| Tip SHA | `4e976e259` (origin/main after #449/#451) |
| Branch | `cursor/latte-retime-20260923` |
| Host | Totoro; load ~229 at launch; `taskset -c 0-15` |
| Julia | 1.12.6; `JULIA_NUM_THREADS=4` `OPENBLAS_NUM_THREADS=1` |
| Script | `bench/latte_gap_retime/off_on_wall.jl` |
| REPS | 5 warm timed + 1 untimed warm-up per arm |
| Cells | `glmm_200x5`, `glmm_5000x3_g500` |
| Result | OFF/ON **2.90×** / **1.18×**; `|Δll|=0` both |
| Identity cite | S4 26/26 (#449) + this-run Δll=0 |
| Dual-fixture ~1.5× | small PASS; large **FAIL** (1.18×) |
| Verdict | **KEEP** opt-in (default OFF); Shinichi G0 2026-09-23 |
| Default flip | **NO** (no `src/` edit; no default-ON PR) |
| Receipt | `docs/dev-log/evidence/2026-09-23-latte-off-on-wall/` |
| Public claim | none |
