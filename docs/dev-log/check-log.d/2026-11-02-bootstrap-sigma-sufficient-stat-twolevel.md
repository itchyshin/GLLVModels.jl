# TwoLevelFit sufficient-statistic loglik on the fixture-B bootstrap batches (diagnostic only)

Nothing tracked was edited, no PR, no push, nothing deleted. (One `git fetch origin` attempt at the start printed a harmless "cannot lock ref refs/remotes/origin/gh-pages" error; no repo state was changed.)
Patched code = scratch `~/prof_scratch/pkg3/src/twolevel.jl` on Totoro (the `_twolevel_suffstats` / `_twolevel_loglik_ss` variant from the profile note); stock = `~/prof_scratch/pkg` (origin/main 79857ff87, `twolevel.jl` byte-identical to the one used in the optimum note). Both read-only here.

Setup: fixture B (p=5, 60 individuals x 4 observations = 240 columns, balanced, K_B=2, K_W=1), the two saved batches of the optimum note (`jsim_Y_2001`, `rsim_Y_2001`, 999 replicates each), refit with the driver's own call `fit_twolevel_gaussian(Y, ind; K_B=2, K_W=1, center=true)` (g_tol 1e-8, 1000 iterations, PPCA start). One Julia thread per process, `OPENBLAS_NUM_THREADS=1`, 3 cores (taskset 330-332), Totoro load ~290. Estimate stated: under 15 min wall, actual about 25 min including diagnostics. All my processes have exited.

Validity of the baseline: my stock rerun reproduces the saved stock rows of the optimum note bit-for-bit (max |logLik diff| 0.0, iteration counts equal in all 999, both batches).

Reference optimum per replicate = max(R logLik if R converged, best of the 12 saved Julia variants per replicate [stock, warm, polish, iter3000, s2 starts, 4 random starts, ...], stock, SS). "Deficit" = reference minus Julia logLik.

Files (scratch, local): `/private/tmp/claude-503/-Users-z3437171-Dropbox-Github-Local-GLLVM-jl/eb56923c-569d-427b-bdc5-d1f448428f52/scratchpad/ss/twolevel/` (`out/*.csv` per-replicate tables, `analysis.txt`, `diag_out.csv`, scripts `tl_ss_fit.jl`, `analyse.py`, `analyse2.py`, `diag.jl`, `flags.jl`). On Totoro: `~/claude-ss-twolevel/`.

## 1. Convergence flags and iteration counts

| | jsim stock | jsim SS | rsim stock | rsim SS |
|---|---|---|---|---|
| converged | 999/999 | **998/999** | 999/999 | 999/999 |
| flags differing | | 1 (replicate 217) | | 0 |
| iterations mean / median / max | 182.7 / 153 / 950 | 185.6 / 156 / 1000 | 180.4 / 152 / 810 | 184.7 / 155 / 745 |
| total iterations | 182,473 | 185,422 (+1.6%) | 180,252 | 184,547 (+2.4%) |
| replicates with iterations changed | 940/999 (94%) | | 928/999 (93%) | |
| change, min / max | -490 / +276 | | -402 / +333 | |
| abs change > 10 / > 50 | 346 / 77 | | 334 / 59 | |

The 20-replicate profile (16/20 changed, +-5) understated the path divergence: on this fixture nearly every replicate stops at a different iteration, sometimes hundreds apart. Final logLik is nonetheless the same optimum in 997/999 replicates per batch (section 3).

How "converged" is decided (measured, 5 replicates, plus saved `g_residual` for all 999): Optim reports `f_converged = true`, `g_converged = false`, `x_converged = false` (f_abstol = f_reltol = 0, so it fires when the objective is bit-identical between iterations). Final gradient sup-norm is 6e-7 to 2.3e-5 (median 2.3e-6), and 100% of the 999 stock fits sit above g_tol = 1e-8. So the stopping iteration is set by when floating-point noise freezes f, which is why any rounding-level change in the objective shifts iteration counts. This also corrects a wording in the optimum note ("gradient below 1e-8 (g_tol)" for the worse optima: measured 8.9e-7 to 8.1e-6).

The one new non-converged replicate (jsim 217): stock converged at 841 iterations, SS hit the 1000 cap (|g| = 7e-5). With a 5000 cap SS converges at 1026 iterations to logLik -1688.5625411275, identical to stock's -1688.5625411275 (diff 1.8e-9). It is the same optimum; the cap bound by 26 iterations. Stock is also near the cap in other replicates (jsim 795: 950 iterations, 217: 841; rsim 954: 810), so cap exposure is generic to this fitter; SS merely moved which replicate tipped over (795 now takes 460). Rate: 1/1998 SS vs 0/1998 stock, not distinguishable from path luck.

## 2. Same replicates in worse optima? (deficit > 1e-3, "R > Julia by > 1e-3, both converged" as in the optimum note)

| batch | stock | SS | in both | stock only | SS only |
|---|---|---|---|---|---|
| jsim (n=978 stock-and-R converged; 977 with SS) | 6 (0.61%) | 5 (0.51%) | 5 | 1 (b=624) | 0 |
| rsim (n=992) | 9 (0.91%) | 8 (0.81%) | 8 | 1 (b=628) | 0 |

Stock counts reproduce the optimum note exactly (6/978, 9/992; max deficits 0.868 and 0.340). Replicates:

- jsim persistent in both: 121 (0.0355), 196 (0.868), 719 (0.0031), 820 (0.0205), 891 (0.0093). |logLik SS - stock| <= 6.8e-13 in all five, i.e. the same local optimum to 12 digits. Stock only: 624 (deficit 2.36e-3, 171 iterations; SS 447 iterations, ends 1.7e-6 below best).
- rsim persistent in both: 252 (0.340), 346, 363, 364, 501, 535, 548, 589 (deficits 0.0079 to 0.028). |logLik SS - stock| <= 2.6e-11. Stock only: 628 (3.13e-3, 174 iterations; SS 384 iterations, 4.3e-8 below best).
- Deficit sets at other thresholds (jsim / rsim): > 0.01: stock 3 / 7, SS 3 / 7 (identical sets); > 0.1: 1 / 1, identical; > 1e-6: stock 11 / 11, SS 13 / 9 (SS-only in jsim: 624 at 1.7e-6 and 265 at 1.2e-6).
- No SS-only deficit above 1e-3 in either batch. No replicate where Julia beats R by more than 1e-3 (0 / 0, both variants).
- The 2 rescued replicates (624, 628) are not true local optima of stock: min Hessian eigenvalue at the stock end point is -2.2e-6 and -1.8e-6 (negative), gradient 8e-6 and 5.6e-6, and restarting stock from its end point does not move (polish dll 0.0 in 2 iterations). SS ends at positive eigenvalue (1.7e-6, 4.3e-8). So these are optimiser stalls on the ridge that a different rounding path happened to avoid; I would not count it as a systematic gain.
- The persistent worse optima are flat or boundary points (min Hessian eigenvalue |lambda| < 1e-6 in 121, 196, 719, 820, 891, and in rsim 252, 346, 363, 364, 501, 535, 548, 589). Among 12 randomly drawn ordinary replicates (diag.jl), 5 have a singular Hessian (|lambda| < 1e-6), so boundary/Heywood solutions are common on this fixture (inference: some sigma^2 tends to 0; I did not inspect the parameters).
- The stock objective and the SS objective agree: gradient sup-norms of the two objectives at the same theta are identical to the printed 3 digits (for example 4.05e-06 and 4.05e-06), and the Hessian-eigenvalue and logLik values coincide.

## 3. logLik, draws, bounds and SE versus stock

logLik: |SS - stock| > 1e-9 in 13 (jsim) and 15 (rsim) replicates, > 1e-6 in 2 and 2, > 1e-3 in 1 and 1 (the rescues 624: +2.36e-3, 628: +3.13e-3). In jsim 265 SS is 1.2e-6 lower. Max |dlogLik| 2.4e-3 / 3.1e-3. In the other 997 replicates per batch it is below 1e-6.

Draws (Sigma_B and Sigma_W, 50 entries), replicates with the same optimum (|dlogLik| < 1e-6, n=997 each): max per-replicate abs difference 2.46e-6 (jsim) and 1.01e-6 (rsim), median 6.0e-8, 99th percentile 2.9e-7 / 3.1e-7. Larger than the 20-replicate profile value (5e-8 / 1.3e-7) because 999 replicates include flat directions; still 1e-6 scale. For the two rescue replicates per batch the difference is 2.2e-3 / 2.1e-3.

Percentile bounds (2.5/97.5) and SE (= SD of draws) over all 115 summary entries the driver reports (Sigma, R, communality, ICC for B and W):

| comparison | max abs bound diff (lo / hi) | max SE diff |
|---|---|---|
| jsim, stock vs SS, all 999 draws kept | 1.24e-7 / 1.13e-7 | 0.013% rel (2.5e-5 abs) |
| rsim, stock vs SS, 999 draws | 1.54e-7 / 1.06e-7 | 0.009% rel (1.6e-5 abs) |
| jsim, stock (999) vs SS as the driver would run it (217 dropped, 998) | **1.93e-3 / 2.23e-3** | **0.549% rel** (4.2e-4 abs) |
| jsim, stock vs SS on the same 998 replicates | 1.21e-7 / 1.11e-7 | 0.013% |

Versus the best-of-12-variants draws (50 Sigma entries, all 999 replicates): stock bound error and SS bound error are the same to the printed digits. jsim: lo 8.6e-8 (stock) / 5.3e-8 (SS), hi 1.57e-3 / 1.57e-3, SE 0.187% / 0.188%. rsim: lo 9.1e-7 / 9.1e-7, hi 2.46e-7 / 2.47e-7, SE 0.231% / 0.231%. So the worse-optimum distortion (<= 1.6e-3 in bounds, <= 0.23% in SE) is unchanged by SS.

Note the jsim 217 case: one dropped replicate out of 999 moves bounds by up to 2.2e-3 and SE by 0.55%, which is as large as the whole worse-optimum effect (1.6e-3 / 0.23%). Consistent with the optimum note's point that dropping replicates hurts more than keeping them.

## 4. Wall time

| measurement | stock | SS | ratio |
|---|---|---|---|
| full batch jsim (999 fits, 3 concurrent processes, load ~290) | 240.1 s (0.240 s/rep) | 13.7 s (0.0137 s/rep) | 17.6x |
| full batch rsim | 234.6 s | 11.1 s | 21.2x |
| per iteration, jsim / rsim | 1.32e-3 / 1.30e-3 s | 7.4e-5 / 6.0e-5 s | 17.9x / 21.7x |
| paired rerun, first 100 replicates x 3, run 1 (stock/SS on one core, back to back) | 33.5 s (0.112 s/rep) | 2.42 s (0.0081 s/rep) | 13.8x |
| paired rerun run 2 | 40.7 s (0.136 s/rep) | 1.54 s (0.0051 s/rep) | 26.6x |

So 14x to 27x on this fixture (profile note: 9-16x on p=10, 80x5, K_B=K_W=1). CPU clock equals wall clock in every run (single thread). Timings are inflated 2-4x by node load and varied by about 2x between identical repeats; trust the order of magnitude, not the digits. Balanced groups only: one group size means one pair of Dual p x p Cholesky solves per objective call, the best case for SS. Unbalanced designs untested.

## 5. Does it change any conclusion?

Unchanged:
1. The worse-local-optimum deficit is a property of the likelihood and start, not of the arithmetic. SS lands in the same worse optimum to 1e-12 in 5/6 (jsim) and 8/9 (rsim) of the replicates; no new deficit above 1e-3 appears; bounds/SE distortion versus the best-of-12 reference is identical (1.6e-3, 0.23%). The optimum note's TwoLevel recommendation (no fit change; state the 0.6-0.9% rate in the docstring; filtering is worse than keeping) stands.
2. The multi-start remedy is untouched in effect (SS reaches the same optima as stock, so the 1-2 residual replicates after 10 starts would remain). What SS would change is its cost: 3 starts at SS speed would be about 0.1-0.2x of one stock fit (extrapolated from the timings, not run), but it still needs a start keyword in `fit_twolevel_gaussian`.

Changed or newly relevant:
3. Convergence-flag tail: 1 new `converged = false` (jsim 217) from a cap of 1000, same optimum as stock. If SS were adopted for the TwoLevel refit, the cap would need raising for the bootstrap (5000 is cheap at SS speed; 217 needs 1026), otherwise about 1 in 2000 replicates is dropped and moves bounds by up to 2e-3 / SE 0.55%. Stock has the same exposure (950 and 841 iterations in the same batch).
4. Draw reproducibility: per-replicate Sigma entries change by up to 2.5e-6 (median 6e-8), and iteration counts change in 93-94% of replicates, not the 16/20 of the profile. Pinned values and seed-level reproducibility in tests would need re-pinning at about 1e-5 tolerance; the bound/SE summaries move by only 1.5e-7 and 0.013%.
5. Two stock stall points (624, 628) are rescued by SS; I treat this as luck of path, not a gain.

## 6. Caveats and what was not covered

- One DGP per simulator (fixture B, p=5, 60 x 4, K_B=2, K_W=1, a single saved batch each). Balanced groups only. Not run on the GllvmFit fixture c (the other task).
- 12 diagnostic replicates per batch beyond the named ones were drawn at random (rng 5) for the Hessian check; counts of singular Hessians (5/12) are rough. Parameter values at the boundary were not inspected.
- The multi-start cost under SS is extrapolated, not measured. The 3000-iteration SS run was done only for replicate 217 (5000 cap).
- Timing under load ~290 and 2x repeat spread; ratios are order of magnitude.
- No src change proposed as final; the maintainer decides.
