# GllvmFit sufficient-statistic objective on the bootstrap fixtures a, b, c (diagnostic, 2026-11-02)

Diagnostic only. No tracked file edited, nothing pushed, no PR. Source: `git archive origin/main` (7b32680ee; `fit.jl` and `profile.jl` are identical to the a2fc5ce03 used by the optimum note) patched in a scratch copy. Runs on Totoro in `~/claude-bsig-ss/gllvmfit/` only. Three single-core processes pinned with `taskset` (cores 201, 203, 208), `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, load average 290. Estimate stated before the run: 10-30 min wall; actual 4 min per fixture-c run, 1.5 min for b, 10 s for a (about 6 min of wall for the whole grid). Nothing deleted; no process left running.

Local copies: `/private/tmp/claude-503/-Users-z3437171-Dropbox-Github-Local-GLLVM-jl/eb56923c-569d-427b-bdc5-d1f448428f52/scratchpad/ss/gllvmfit/` (`pkg/` patched source, `profile_ss.patch`, `fit_ss.patch` as `diff patched orig`, `scripts/`, `out/ss_*.csv` per-replicate tables, `out/analysis_*.txt`, `out/supp_*.txt`).

## 1. What was implemented (re-derived)

Where the earlier patch could not reach: `fit_gaussian_gllvm` keeps beta in the parameter vector (`profile_beta = false` is hard-coded in `fit.jl`), and the bootstrap driver passes `X` to the refit whenever the original fit has beta (q > 0). All three fixtures use a six-intercept X (q = 6). So the fixtures run the "beta supplied" pass-through branch of `gaussian_profile_nll`, with beta Dual. The earlier note's patch covered q == 0 only ("fits with X keep the old path"), so it would not have touched these refits at all. I therefore derived the form for q >= 0:

- Q = sum_s r_s' At^-1 r_s = tr(At^-1 R), r_s = y_s - X_s beta.
- Q = tr(Ainv Syy) - 2 beta' v + beta' M beta, with Syy = y y', C_j = sum_s x_sj y_s', Sxx_jk = sum_s x_sj x_sk', v_j = tr(Ainv C_j), M_jk = tr(Ainv Sxx_jk).
- Ainv = At^-1 = D^-1 - D^-1 L A_K^-1 L' D^-1 is formed once per call (p x p, Dual). The Float64 cross-products are computed once per fit, outside AD.
- Everything else (logdet, sigma_eps profile, final `profile_recover`) is unchanged. `profile_recover` still evaluates the stock form once on `y`, so the reported logLik is the stock formula at the SS optimum: logLik differences below are differences in optimum location, not formula noise.
- A keyword `ss = true` on `fit_gaussian_gllvm` switches it on; `ss = false` is code-identical to stock.

Checks of the objective itself (`check_obj.jl`, fixtures a/b/c specs, 20 random draws each of: random params, beta = column means, Heywood-like tau = e^+6 and e^-8): max relative difference in NLL 6e-16, in the ForwardDiff gradient 7e-15. The two formulas are the same function to rounding; any change in outcome comes from the optimiser trajectory.

Validity of the harness: regenerating the driver batches (MersenneTwister(seed), n_boot = 999) under this source reproduces the saved `Y_*.csv` that R refit **bit-identically** (max abs diff 0.0). My stock reruns reproduce the saved `opt_*.csv` rows ("stock" and "tight12") exactly: logLik, iteration counts, converged flags and all 42 draw entries, for a, b, c101, c102. R tables are `R_refit_*.csv` (R `gllvmTMB` on the same data, 0 non-converged everywhere).

Configs, all 999 replicates: `s500` stock default (cap 500, f_tol 0); `s2000` stock with `iterations = 2000, f_tol = 1e-12`; `q500`, `q2000` the same with the SS objective. Configs were interleaved per replicate in rotating order so load drift hits all four equally.

## 2. Results

### 2.1 Failures, iterations, logLik versus R (non-converged / at cap; replicates with R logLik minus Julia > 1e-4; max deficit)

| fixture, seed | cfg | non-converged | at cap | iters med / mean / max | R-ll > 1e-3 | R-ll > 1e-4 | max deficit | Julia > R by 1e-4 |
|---|---|---|---|---|---|---|---|---|
| a, 101 | all four | 0 | 0 | 0 / 0 / 0 | 0 | 0 | -5.8e-11 | 0 |
| b, 101 | s500 | 0 | 0 | 41 / 41.6 / 67 | 0 | 0 | -6e-11 | 0 |
| b, 101 | s2000 | 0 | 0 | 30 / 30.2 / 45 | 0 | 0 | 4.8e-9 | 0 |
| b, 101 | q500 | 0 | 0 | 40 / 40.4 / 60 | 0 | 0 | -6e-11 | 0 |
| b, 101 | q2000 | 0 | 0 | 29 / 29.2 / 44 | 0 | 0 | 4.1e-9 | 0 |
| c, 101 | s500 | **5** | 5 | 70 / 77.2 / 500 | 2 | 5 | 1.75e-3 | 0 |
| c, 101 | s2000 | **0** | 0 | 53 / 62.3 / 1755 | 0 | 0 | 7.6e-5 | 0 |
| c, 101 | q500 | **7** | 7 | 69 / 77.0 / 500 | 0 | 5 | 8.6e-4 | 0 |
| c, 101 | q2000 | **0** | 0 | 52 / 61.5 / 1767 | 0 | **1** | 1.5e-4 | 0 |
| c, 102 | s500 | **3** | 3 | 71 / 77.3 / 500 | 2 | 3 | 1.53e-3 | 0 |
| c, 102 | s2000 | **0** | 0 | 53 / 60.3 / 1738 | 0 | 0 | 5.5e-5 | 0 |
| c, 102 | q500 | **3** | 3 | 70 / 76.6 / 500 | 1 | 3 | 1.24e-3 | 0 |
| c, 102 | q2000 | **0** | 0 | 52 / 59.1 / 1269 | 0 | 0 | 7.0e-5 | 0 |

Fixture a is a null comparison: the PPCA start is already within g_tol of the optimum, `n_iter = 0` in all 999 replicates, the optimiser stops at the start point, and all outputs are bit-identical with and without SS (about 1.5 ms per replicate either way). It says nothing about the SS objective's behaviour.

### 2.2 SS versus stock, same options

| fixture, seed | pair | converged flag changed | iteration count changed | iteration change min / max | max abs logLik diff | max abs draw diff (42 entries) | replicates with draw diff > 1e-5 |
|---|---|---|---|---|---|---|---|
| b, 101 | q500 vs s500 | 0 | 912 / 999 | -18 / +17 | 2.1e-12 | 8.4e-8 | 0 |
| b, 101 | q2000 vs s2000 | 0 | 859 / 999 | -11 / +10 | 6.0e-9 | 8.0e-6 | 0 |
| c, 101 | q500 vs s500 | **2** (0 rescued, 2 lost: b = 600, 714) | 955 | -255 / +115 | 1.19e-3 | 2.3e-3 | 5 |
| c, 101 | q2000 vs s2000 | 0 | 906 | -345 / +280 | 1.08e-4 | 3.7e-4 | 15 |
| c, 102 | q500 vs s500 | 0 | 947 | -71 / +71 | 7.8e-4 | 2.7e-3 | 3 |
| c, 102 | q2000 vs s2000 | 0 | 882 | -469 / +23 | 2.8e-5 | 4.4e-4 | 9 |

The trajectories differ for about 90% of replicates (the same function to 1e-15 but not bit-identical, so LBFGS wanders differently), which the earlier TwoLevel note also saw; the 0/20 iteration changes the profile note reported for GllvmFit does not hold for the has_diag fixtures with X. Typical per-replicate draw difference under 2000/1e-12 is 2.4e-6 (median) and 4.7e-6 (90th percentile); the 99th percentile is 9e-6 to 1.6e-5.

### 2.3 Bounds and SE (2.5/97.5 type-7 percentiles and SD of draws, converged replicates, 42 entries = 36 Sigma_B + 6 communality_B)

Max over entries of |difference|; SE as percent of the comparator.

| fixture, seed | pair | max abs lower | max abs upper | max SE diff % |
|---|---|---|---|---|
| b, 101 | q500 vs s500 | 2.0e-8 | 2.3e-8 | 1.1e-6 |
| b, 101 | q2000 vs s2000 | 1.8e-6 | 3.5e-6 | 6.1e-5 |
| b, 101 | s2000 vs R / q2000 vs R | 4.2e-6 / 3.8e-6 | 6.6e-6 / 5.6e-6 | 4.5e-4 / 4.9e-4 |
| c, 101 | q500 vs s500 | **3.7e-4** | **1.9e-3** | **0.53** |
| c, 101 | q2000 vs s2000 | 8.1e-6 | 3.5e-6 | 5.5e-4 |
| c, 101 | s2000 vs R / q2000 vs R | 6.5e-6 / 7.5e-6 | 7.6e-6 / 9.7e-6 | 4.6e-3 / 4.7e-3 |
| c, 101 | s500 vs R / q500 vs R | 3.2e-3 / 3.1e-3 | 2.2e-3 / 2.6e-3 | 1.32 / 1.84 |
| c, 102 | q500 vs s500 | 3.4e-8 | 3.3e-8 | 1.4e-6 |
| c, 102 | q2000 vs s2000 | 3.1e-6 | 5.5e-6 | 8.2e-4 |
| c, 102 | s2000 vs R / q2000 vs R | 5.3e-6 / 5.8e-6 | **1.13e-5 / 1.35e-5** | 4.7e-3 / 4.6e-3 |
| c, 102 | s500 vs R / q500 vs R | 1.2e-3 / 1.2e-3 | 1.4e-3 / 1.4e-3 | 0.86 / 0.86 |

Number of survivors: c101 s500 994, q500 992, s2000 = q2000 = R = 999; c102 s500 = q500 = 996, s2000 = q2000 = R = 999; a, b all 999.

The c101 q500 vs s500 gap (up to 1.9e-3, SE 0.53%) is entirely the change in which two borderline replicates fail at the cap (see 2.5), not the objective. When all finite draws are used regardless of the flag, q500 vs s500 on c101 differs by 3.4e-6 / 5.5e-8 and SE 0.0045%. For scale, 999-draw bounds have adjacent order statistics 5.8e-4 to 1.3e-3 apart at the 2.5/97.5 tails, so a 1e-5 shift is about 1% of the Monte Carlo resolution of the bound.

### 2.4 Heywood and flat-ridge replicates (defined from the stock 2000/1e-12 fit: min of sigma_eps^2, sigma2_B[t], sigma2_W[t] < 1e-4)

Fixture c, 999 replicates: 211 (seed 101) and 224 (seed 102) are Heywood by this definition; 55 / 64 have a variance below 1e-6 and 21 / 23 below 1e-8 (exact boundary, values to 1e-323 and exact 0). The 5 / 3 replicates that stock fails at the 500 cap are all exact-boundary replicates with 1000-1750 iterations needed (the flat ridge).

| subgroup (c101 / c102) | n | converged s2000 / q2000 | iteration count changed (2000 cfg) | max abs logLik diff | max abs draw diff |
|---|---|---|---|---|---|
| Heywood, not ridge | 206 / 221 | 206 / 206 ; 221 / 221 | 190 / 197 (mean abs change 5.3 / 4.5) | 1.4e-7 / 1.3e-8 | 3.2e-5 / 2.6e-5 |
| exact boundary (< 1e-8), not ridge | 16 / 20 | all converge both | 16 / 18 (mean 15 / 9.6) | 1.4e-7 / 1.3e-8 | 3.2e-5 / 2.6e-5 |
| ridge (cap failures under stock) | 5 / 3 | 5 / 5 ; 3 / 3 | 5 / 3 (mean abs change 205 / 232; range -469 to +280) | 1.1e-4 / 2.8e-5 | 3.7e-4 / 4.4e-4 |
| not Heywood, not ridge | 788 / 775 | all | 711 / 682 | 1.2e-8 / 2.2e-8 | 9.0e-6 / 1.1e-5 |

Do the Heywood replicates behave the same? In the sense that matters for the failure count: yes. Every Heywood and ridge replicate converges under 2000/1e-12 with both objectives, and none is rescued or lost. In the sense of reproducing the same optimiser path: no. They stop at different points: a boundary replicate's draw moves up to 3e-5 (not ridge), and the 8 ridge replicates move 1.7e-4 to 4.4e-4 in some entry, because the fit slides along a flat ridge and stops on the relative-f test at a slightly different point. Their logLik stays within 1.1e-4 of stock and within 1.6e-4 of R. One replicate under SS (c101 b = 602) ends 1.52e-4 below R's logLik (stock: 4.4e-5, still converged = true); the stock max over the 1998 refits is 7.6e-5. So the count "deficit > 1e-4 vs R" goes 0 to 1 on c101 and stays 0 on c102.

### 2.5 The 500-cap replicates one by one (c101; stock s500 / SS q500 iterations, converged flags s500 q500 s2000 q2000, R logLik minus Julia logLik)

| b | iters s500 | q500 | s2000 | q2000 | conv flags | R-ll s500 / q500 / s2000 / q2000 |
|---|---|---|---|---|---|---|
| 30 | 500 | 500 | 1196 | 1086 | 0 0 1 1 | 2.2e-4 / 3.0e-4 / 2.3e-5 / 3.4e-5 |
| 65 | 500 | 500 | 1755 | 1537 | 0 0 1 1 | 1.75e-3 / 5.6e-4 / 2.8e-5 / 3.1e-6 |
| 140 | 500 | 500 | 1011 | 1085 | 0 0 1 1 | 5.5e-4 / 8.6e-4 / 7.6e-5 / 6.6e-5 |
| 600 | 420 | **500** | 317 | 343 | 1 0 1 1 | -8e-9 / -8e-9 / 1e-8 / 3e-8 |
| 602 | 500 | 500 | 1144 | 799 | 0 0 1 1 | 5.5e-4 / 6.0e-4 / 4.4e-5 / **1.5e-4** |
| 714 | 489 | **500** | 345 | 376 | 1 0 1 1 | -2e-8 / -2e-8 / 9e-8 / 3e-9 |
| 993 | 500 | 500 | 1487 | 1767 | 0 0 1 1 | 1.6e-3 / 6.7e-4 / 4.1e-5 / 3.7e-6 |

b = 600 and 714 are Heywood replicates that stock finishes just under the cap (420, 489 iterations) and that SS, on a slightly different path, does not finish by 500. So at the 500 cap the failing set is trajectory-dependent at the margin (5 vs 7 on c101; 3 vs 3 on c102; the same replicate classes). c102: b = 35, 392, 746 fail under both. Under 2000/1e-12 the maximum iteration count is 1755 (stock) and 1767 (SS), i.e. 12% of the 2000 cap in both: the cap margin is the same.

### 2.6 Wall time (single pinned core, Totoro load 290, interleaved configs; absolute seconds inflated 2-4x, ratios are the figure to use)

| fixture, seed | cfg | total wall s, 999 refits | total CPU s | median ms / rep | GC share | allocation per refit |
|---|---|---|---|---|---|---|
| a, 101 | s500 / q500 | 1.76 / 1.73 | 1.76 / 1.73 | 1.46 / 1.44 | 10% / 10% | 1.9 MB / 1.7 MB |
| b, 101 | s500 | 46.9 | 39.5 | 35.6 | 35% | 0.157 GB |
| b, 101 | s2000 | 33.5 | 27.8 | 25.5 | 32% | 0.114 GB |
| b, 101 | q500 | 4.63 | 3.91 | 3.42 | 14% | 0.0076 GB |
| b, 101 | q2000 | 3.97 | 3.45 | 2.66 | 20% | 0.0059 GB |
| c, 101 | s500 | 113.9 | 113.8 | 98.4 | 30% | 0.397 GB |
| c, 101 | s2000 | 96.0 | 95.9 | 75.5 | 31% | 0.317 GB |
| c, 101 | q500 | 8.54 | 8.53 | 7.22 | 16% | 0.0183 GB |
| c, 101 | q2000 | 7.45 | 7.45 | 5.61 | 20% | 0.0148 GB |
| c, 102 | s500 | 114.5 | 114.4 | 96.4 | 29% | 0.398 GB |
| c, 102 | s2000 | 87.2 | 87.1 | 70.1 | 29% | 0.308 GB |
| c, 102 | q500 | 9.10 | 9.11 | 7.09 | 21% | 0.0182 GB |
| c, 102 | q2000 | 8.17 | 8.20 | 5.61 | 30% | 0.0143 GB |

Speed-up (total wall, SS vs stock): c101 13.3x (500) and 12.9x (2000); c102 12.6x and 10.7x; b 10.1x and 8.4x; a 1.0x. Median of per-replicate ratios 9.5x to 13.6x (5th-95th percentile 3.9x to 20x). Heywood replicates speed up as much as the rest (12.7x to 14.3x on c101, 10.9x to 13.6x on c102); the 8 ridge replicates 15x to 19x. Allocation falls about 22x. This is 10-13x, not the 23-32x of the profile note: that was a different case (p = 30, n = 200, K = 2, has_diag, `y` passed, ~119 parameters; q = 0 inferred from "q == 0 only", not stated); here p = 6 and q = 6 betas sit in the AD vector. The earlier figure should not be carried over to the fixtures.

## 3. Does the faster objective change any conclusion?

- **Failure counts.** Under `iterations = 2000, f_tol = 1e-12`: 0 non-converged out of 999 on every fixture, with both objectives (stock c101 0, c102 0; SS 0, 0). No change. At the stock 500 cap: c101 5 vs 7, c102 3 vs 3, a and b 0 vs 0. The two extra c101 failures are replicates stock finished in 420 and 489 iterations, so the 500-cap failure count is a trajectory-dependent margin (0.3-0.7%), not something the objective rescues or fixes.
- **Bounds beyond 1e-5.** SS vs stock under 2000/1e-12: no bound differs by 1e-5 (max 8.1e-6 on c101, 5.5e-6 on c102, 3.5e-6 on b, 0 on a), SE at most 8.2e-4 %. The margin on c101 is thin (8.1e-6 against a 1e-5 line). Against R: SS 2000 bounds are 7.5e-6 / 9.7e-6 (c101) and 5.8e-6 / 1.35e-5 (c102) away, stock 2000 6.5e-6 / 7.6e-6 and 5.3e-6 / 1.13e-5; so c102's upper bound already differed from R by more than 1e-5 under stock and moves by 2.2e-6 further under SS. At the 500 cap the SS-vs-stock bound gap on c101 (1.9e-3, SE 0.53%) is entirely the two-replicate failure-set change. The original 500-cap-versus-R gaps (3.2e-3 and 1.2e-3 on c, 1.3% and 0.86% SE) are reproduced and the recommended `2000 / 1e-12` setting removes them with either objective.
- **logLik versus R.** SS 2000 is within 1.6e-4 of R in every replicate (stock 7.6e-5). One replicate on c101 is above 1e-4 (1.52e-4) under SS and none under stock. Nobody is better than R by more than 1e-4 in any configuration.
- **Conclusion on the 2000/1e-12 recommendation.** Unchanged: failures 0, bounds within the optimum note's 6e-6 to 1.1e-5 of R, cap margin 12%. Adopting SS would not undermine it, but it would move individual draws (not bit-identical, iteration counts change in about 90% of replicates) and every pinned bootstrap value and seed-reproducibility claim would need rechecking, as the profile note already says.

## 4. Caveats (what this does not cover)

- One DGP per fixture, p = 6, n = 200, Gaussian; GllvmFit seeds 101 and 102 for c, seed 101 only for a and b (as requested). The 8 ridge replicates (5 + 3) are the only cap failures; their statistics are small-number.
- Fixture a never iterates (0 iterations), so it tests nothing about SS.
- Only the optimisation objective is replaced. `profile_recover` and the Hessian-based post-fit code were left on the stock form. K_W > 0 (the other flat-ridge route) was not run: no fixture uses it, though it shares the code path through `L_U = hcat(L_B, L_W)`. The phylo path is untouched and not covered. Fits without X (q = 0) are covered by the same code via `suff.q = 0`, but only the objective equality check was run for them, not a refit grid.
- The Heywood grouping is my own definition (min variance < 1e-4); the sets are given by threshold counts so another cut can be made from the saved tables.
- Timings are under load 290 with 3 of my processes; ratios are interleaved within each replicate, absolute seconds are not reliable. Allocation figures are `@timed` bytes.
- I peeked at `~/prof_scratch/profile_ss.jl` (the earlier q == 0-only version) after deriving mine, only to confirm the algebra; mine differs (p x p Ainv contraction, includes beta terms). The earlier patch was not used.

## 5. Files

Totoro `~/claude-bsig-ss/gllvmfit/`: `pkg/` (patched copy), `scripts/{env.sh,common.jl,check_obj.jl,run_ss.jl,analyse.jl,supp.jl}`, `out/ss_{a_s101,b_s101,c_s101,c_s102}.csv`, `out/analysis_*.txt`, `out/supp_*.txt`, `logs/`. Reads (never writes) `~/claude-bsig-opt/{data,out}`. Same files locally under the scratchpad path at the top.
