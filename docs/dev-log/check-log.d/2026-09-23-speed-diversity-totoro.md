# 2026-09-23 — GLLVM speed-board diversity Totoro (8 new kinds)

| Field | Value |
|---|---|
| Tip SHA | `4e976e259` (post-#451) |
| Branch | `cursor/speed-diversity-20260923` |
| Host | Totoro; load ~224–236 at launch |
| Julia | 1.12.6; `JULIA_NUM_THREADS=4` `OPENBLAS_NUM_THREADS=1`; `taskset -c 0-3` |
| Script | `bench/speed_board_diversity.jl` |
| Cells | 8 new kinds (ordinal, ZIP, ZINB, phylo Poisson GLM, SPDE, concurrent, missing mask, profile CI medium) |
| R H2H | none (no cheap paired script) |
| TSV | `docs/dev-log/evidence/2026-09-23-speed-diversity-totoro/board_gllvm_diversity_4e976e2.tsv` |
| Board | GLLVM **18/18** `has_receipt` (toward 20) |
| Public claim | none; Latte default stays OFF |

| 2026-09-23 | diversity | Totoro 8/8 ok @ `4e976e2` | walls 0.0187–3.8452 s; missing-mask retry after PosDef fix |
