# 2026-09-23 — GLLVM speed-board Totoro cells (post-#449)

| Field | Value |
|---|---|
| Tip SHA | `8d58a0c94` (post-#448/#449/#450) |
| Branch | `cursor/speed-board-20260923` |
| Host | Totoro (`totoro.biology.ualberta.ca`; load ~144 at launch) |
| Julia | 1.12.6; `JULIA_NUM_THREADS=4` `OPENBLAS_NUM_THREADS=1` |
| Script | `bench/speed_bench.jl` |
| Cell 1 | `GLLVM_SPEED_BENCH_GRID=8,40,1` `REPS=3` `ITERS=120` `PROFILE_CI=0` |
| Cell 2 | `GLLVM_SPEED_BENCH_GRID=30,100,2` (same knobs) |
| Skipped | Latte identity (already PASS on #449); `gllvm-profile-ci-small` |
| Sibling skip | Latte lane held `warm_identity` / grouped lease; not these cell_ids |
| TSV | `bench/results/board_gllvm_20260923_8d58a0c94.tsv` |
| Logs | `docs/dev-log/evidence/2026-09-23-speed-board-totoro/` |
| Public claim | none (no README/NEWS; no Latte default-ON) |

Headline walls (median of 3 warm reps):

| cell_id | family / variant | wall_s |
|---|---|---:|
| `gllvm-gauss-unstruct-small` | Gaussian closed 8×40×1 | 0.0001 |
| `gllvm-gauss-unstruct-large` | Gaussian closed 30×100×2 | 0.0016 |
| `gllvm-nb2-or-binom-unstruct` | NB :analytic 8×40×1 | 0.0222 |
| `gllvm-nb2-or-binom-unstruct` | Binomial :analytic 8×40×1 | 0.0130 |
