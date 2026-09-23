# Check-log: GLLVM Phase B first-wave Totoro (2026-09-23)

| Field | Value |
|---|---|
| Lane | `cursor/speed-diversity-20260923` |
| Tip | `4e976e259` |
| Host | Totoro (load≈229 at launch; staggered) |
| Threads | `JULIA_NUM_THREADS=4` `OPENBLAS_NUM_THREADS=1` |
| Script | `bench/speed_board_firstwave.jl` |
| Cells | 10/10 plan §4.1 `ok` |
| Evidence | `docs/dev-log/evidence/2026-09-23-speed-firstwave-totoro/board_gllvm_firstwave_4e976e2.tsv` |
| Not run | Local heavy Julia; Latte default flip; R paired H2H |
| Commands | `ssh totoro` + `SPEED_CELL_ID=… julia --project=. bench/speed_board_firstwave.jl` ×10 |
