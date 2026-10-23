# DRAC first-wave GLLVM receipts (Fir)

**Job:** `61148081` (`gllvm-fw`, array 1–7%4) on Fir `cpubackfi` / `cpubase_by` Priority.
Predecessor `61144758` failed on concurrent `Pkg.instantiate` into `~/.julia`; resubmitted after a shared scratch depot
`/scratch/snakagaw/julia_depot_speed_phaseB` and dropping per-task instantiate.

**Outcome:** 7/7 COMPLETED with `status=ok`. Combined table: `gllvm_fir_61148081_combined.tsv`.

Skipped on DRAC (by harness design): `gllvm-latte-gap-200x5`, plus phylo/spatial smokes already banked on Totoro
(`…/speed-firstwave-totoro/`). Totoro tip-abs remains the board authority for cells 11–20; Fir walls are
same-cell cross-host abs only (no cross-host ×).

| task | cell_id | wall_s (median of 3) | host |
|---:|---|---:|---|
| 1 | `gllvm-binom-glmm-200x5` | 0.158222 | fc20317 |
| 2 | `gllvm-gauss-lv-t4-p20n500` | 0.006743 | fc20355 |
| 3 | `gllvm-pois-lv-t4-p20n500` | 1.208765 | fc20355 |
| 4 | `gllvm-nb2-lv-t4-p20n500` | 4.256158 | fc20420 |
| 5 | `gllvm-gauss-unstruct-p50n2k` | 0.100006 | fc20317 |
| 6 | `gllvm-gauss-phylo-fit-p200` | 0.020452 | fc20355 |
| 7 | `gllvm-profile-ci-glmm` | 1.191450 | fc20355 |

Threads: `JULIA_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`. Julia 1.11.3 (StdEnv/2023).
