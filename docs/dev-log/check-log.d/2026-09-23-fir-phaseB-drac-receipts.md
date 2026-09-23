# check-log · Fir Phase B DRAC receipts

| date | slice | command | result |
|---|---|---|---|
| 2026-09-23 | Fir GLLVM `61148081` | `sbatch tools/speed_firstwave_gllvm.sbatch` after scratch depot instantiate | 7/7 COMPLETED; all `status=ok`; walls in `evidence/2026-09-23-speed-firstwave-drac/` |
| 2026-09-23 | Fir DRM `61148082` | `sbatch tools/speed_firstwave_drm.sbatch` | 10/10 COMPLETED batch; all `status=error` (MethodError); Totoro CSV remains board authority |
| 2026-09-23 | predecessor | `61144758` / `61144759` | FAILED concurrent `Pkg.instantiate` into `~/.julia`; cancelled; depot fix + resubmit |

