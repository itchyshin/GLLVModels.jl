# After-task: Fir Phase B DRAC receipts

**Lane:** Shannon (perspectives: Ada, Rose). No subagents.
**Jobs:** GLLVM `61148081` (successor of failed `61144758`); DRM `61148082` (successor of `61144759`).

## Outcome

1. Fixed Fir depot race (concurrent `Pkg.instantiate` into `~/.julia`) by pre-instantiating
   `/scratch/snakagaw/julia_depot_speed_phaseB` and pointing both sbatch files at it.
2. GLLVM: **7/7 ok** Fir abs walls landed under `docs/dev-log/evidence/2026-09-23-speed-firstwave-drac/`.
3. DRM: **10/10 batch complete, 0 usable walls** (harness MethodError). Totoro CSV `e9d50a110` banks
   Phase B cells 11–20; Fir TSVs kept as negative receipts.
4. Three-package report counts moved to **20 / 20 / 20 = 60**. GLLVM #454 already merged
   (`36479c28d`); this follow-up lands Fir receipts + DRM §4.2 fill.

## Rose

- No Fir×Totoro ×.
- DRM Fir errors are not silent skips.
- Public README/NEWS still withheld.
