# NB grid re-run on the corrected NB2 kernel (#521): plan and pre-run test

Status: COMPLETE 2026-09-29, all 4,800 datasets (0.934, see Progress); was SUBMITTED 2026-09-28 16:40Z on nibi (Shinichi approved, "1-3 as you recommended"). Smoke job 22840888 (precompile + cheapest row), array 22840890 (943 tasks, `--dependency=afterok`). Code: #518 head be55464e0 in `~/projects/def-snakagaw/snakagaw/auto-d-nbrerun` (`DEPLOYED_SHA`, `SUBMITTED`); scripts `pilot/*_nibi.sh`. Stop and re-report if it overruns about 5,500 core-h (D-287).
(D-287: a run over 3 h needs a plan, a pre-run test with results, and approval).

## Why

Every NB number in the auto-d recovery grid (mean exact recovery 0.895 with `:bic_sites`) was
measured on the NB2 grouped kernel before #521, which 2-cycled at sites with y much larger than
μ and let L-BFGS stop at poor optima reported as converged. Those numbers are withdrawn in the
docs (#518, #1324). The 24 NB cells need re-running on the fixed kernel.

## What

- Cells: `nb` × n ∈ {30, 60, 120, 300} × p ∈ {10, 20} × K_true ∈ {1, 2, 3}; 200 reps each;
  4,800 datasets. Same DGP and seeds as `pilot/pilot.jl` (`MersenneTwister(hash((fam, n, p,
  K, r)))`), fits K = 1 .. K_true + 2.
- Code: GLLVModels.jl `main` after #521 merges (and after #518, if it has merged by then).
- Harvest: `pilot/analyze.py`, as for the original grid; report NB alongside the old numbers.

## Estimate (before the pre-run test)

Old-kernel mean seconds per dataset, from the harvest (DRAC, complete datasets only): 196 s
(n 30, p 10, K 1) to 3,073 s (n 300, p 20, K 3). Summed over 24 cells × 200 datasets: about
1,730 core-h on the old kernel, an underestimate because the 1,513 missing datasets were the
slowest. #521 reported NB fits about 70% slower: about 2,900 core-h, plausible range 2,000 to
5,000. Too large for Totoro's share of a day; DRAC arrays as before (per-task `--time` from the
pre-run ratio, not the old 6 h, which timed out on the heaviest cells).

## Pre-run test (local Mac, 1 thread per process, 4 at a time)

Six cells, rep 1, fitted with the old kernel (#518 head) and the new one (#518 head plus #521's
`grouped_dispersion.jl`), same data, 2026-09-27. Old-kernel seconds are for K = 1 .. K_true + 2.

| cell (n, p, K_true) | old s | new s | new/old | K chosen (`:bic_sites`, guarded) old / new |
|---|---|---|---|---|
| 30, 10, 1 | 72 | 91 | 1.26 | 1 / 1 |
| 30, 10, 3 | 443 | 861 | 1.94 | 3 / 3 |
| 60, 20, 2 | 934 | 1,427 | 1.53 | 2 / 2 |
| 120, 10, 2 | 444 | 500 | 1.13 | 2 / 2 |
| 300, 10, 1 | 110 | 134 | 1.21 | 1 / 1 |
| 300, 20, 3 | 1,467 | 4,626 | 3.15 | **5 / 3** |
| total | 3,471 | 7,638 | 2.20 | |

Totals are sums of unrounded seconds, so they differ by 1 s from the sum of the rounded rows.

- Cost: 2.2x overall, not the 1.7x #521 measured on its panel, and 3.15x on the heaviest cell,
  which dominates the grid's cost.
- Accuracy preview (one rep): on 300 × 20, K = 3 the old kernel's logliks fell at K = 3 and 4
  (-19113, -20240 below K = 2's -18874), so the guard skipped them and chose K = 5; the new kernel
  is monotone (-18400, -17618, -16743, -16728, -16720) and chooses the true K = 3. Other cells agree.
- DRAC nodes ran these fits 1.5 to 3.9x slower than the Mac; the estimate below uses DRAC's own
  harvest times, so it needs no hardware correction.

## Revised estimate and recommendation

At ratio 2.2: about 3,800 core-h. At ratio 3.2 (the heavy-cell ratio applied everywhere): about
5,500 core-h. **Recommend sizing at 3.2**: 943 tasks with 5 h of estimated work each (under the
1,000-job limit), largest task 7.1 h, `--time=11:00:00`; the committed `nb_rerun_tasks.csv` and
`run_nb_rerun.sh` are generated at that setting. Expected wall time with 300 running at once:
about a day, queue permitting. A run that overruns this estimate stops and re-reports (D-287).

## Submission kit (prepared, not submitted)

- `pilot/pilot.jl` gained a `listrange` mode (rows `a:b` of a list in one Julia process); smoke-
  tested against `list` mode on the same rows (identical output).
- `pilot/nb_rerun_tasks.py <ratio> 3` writes `nb_rerun_list.csv` (4,800 rows, heaviest cells first)
  and `nb_rerun_tasks.csv` (cost-balanced, about 3 h of estimated work per task). At ratio 1.7:
  850 tasks, about 2,940 core-h, largest task 4.35 h, largest single dataset 1.45 h.
- `pilot/run_nb_rerun.sh`: sbatch array over the task map; `--time` and `--array` are placeholders
  until the pre-run ratio is in. Outputs to `out-nbrerun/`, never mixed with `out-rerun/` (old
  kernel, cancelled run).

## Progress

- 2026-09-28 ~22:30Z: 235 of 943 tasks completed, 95 running, 0 failed; 786 of 4,800 datasets done (3,639 fits, all `ok`), heaviest cells first. Heaviest tasks took 3.7 to 4.0 h against the 5 h sizing, so the run is under its estimate. (The nibi socket was down 19:10 to 22:25Z; reopened with a Duo push.)
- 2026-09-29 03:27Z: 321 completed, 87 running, 2 FAILED (tasks 388, 389: node c63, `julia: command not found` after `module load`, died in 20 s, no fits). Node fault, not code; no other task ran on c63. Rerun: `sbatch --array=388,389 --exclude=c63 LOOP/lanes/auto-d-20260926/pilot/run_nb_rerun_nibi.sh` (needs Shinichi; about 11 core-h).
- 2026-09-29 07:29Z: 659 completed, 282 running, 0 pending, still only the 2 c63 failures. Interim harvest (2,983 of 4,800 datasets, `pilot/harvest-report-nbrerun-interim.md`): default rule (lenient guard, bic_sites = `len_bic_n`) mean exact recovery 0.919 vs the withdrawn old-kernel 0.895; strict guard 0.64 to 0.67, because the corrected kernel reports converged = false for boundary dispersion at 63% of K = 3 and 89% of K = 4 fits. Light cells are incomplete, so the mean will move.
- 2026-09-29 10:29Z: array DONE, 941 completed, 2 failed (c63). Final harvest `pilot/harvest-report-nbrerun.md`: 4,794 datasets, default rule 0.934 (old kernel 0.895), bic 0.840, aic 0.793, strict 0.747. 1,888 core-h (accounting `pilot/nbrerun_sacct.txt`). Outputs stay on nibi at `auto-d-nbrerun/out-nbrerun/`.
- 2026-09-29 19:3xZ: tasks 388/389 (rows 1062-1067, 6 datasets, cell n 120 p 10 K 3) rerun on Totoro (Shinichi: "rerun on Totoro") instead of nibi: ~/hsq_work/auto-d-nbrerun-388-389, same code be55464e0, Julia 1.10.10, 2 processes x 1 thread. Expected 2 to 4 h.
- 2026-09-29 20:28Z: tasks 388/389 finished on Totoro (all 30 fits ok, about 45 min each). Full harvest on all 4,800 datasets (`pilot/harvest-report-nbrerun-full.md`): default rule 0.934, bic 0.840, aic 0.793, strict 0.746; cell n 120 p 10 K 3 = 0.91 on 200. Headline unchanged from the 4,794-dataset figure already in the docstring and design/74. RUN COMPLETE.
