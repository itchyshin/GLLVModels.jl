# 74: Estimating the number of latent dimensions from data (auto-d)

Status: SIGNED OFF (G1, 2026-09-27, vault D-293: "go with your recommendations, push and open draft PRs"). Default criterion `:bic_sites` applied in both packages. Lane `auto-d-20260926` (Julia) with the R twin on
gllvmTMB branch `claude/lane-auto-d-r-20260926` (vault D-292). Nothing here is merged.

## Destination

A user fits a GLLVM without supplying the number of latent dimensions. The package chooses it by
one documented rule, returns the chosen value with the per-candidate criterion values and every
rejected candidate with its reason, the rule's recovery rate is measured on simulated data across
n, p and family, and every place a user reads an interval says it is conditional on the choice.

## What the literature says (vault `dr-auto-d-latent-dimension-selection`, NotebookLM `c2c564e3`, 34 sources)

- **Feasible, and the gap is real.** Gaussian factor-number rules are well validated (Bai & Ng
  2002; Ahn & Horenstein 2013; Dobriban & Owen 2019: near-100% recovery once n, p are in the
  hundreds, failing for weak factors). **No published benchmark checks any rule on count or
  binary GLLVM data at ecological n, p.** Ecology packages (gllvm, boral, glmmTMB `rr()`, HMSC)
  leave d to the user and suggest "compare AIC/BIC/CV"; on one real dataset (gllvm 2.0 beetles,
  n = 88, m = 68) AIC picks 3 and BIC picks 2.
- **Chen & Li (2022, Biometrika; arXiv:2010.02326, read directly 2026-09-27)** give the only
  criterion with a consistency proof for binary and count factor models:
  JIC(K) = −2 l̂_K + K (N ∨ J) log{n / (N ∨ J)}, where l̂_K is the **joint** likelihood (factor
  scores estimated as parameters, constrained joint MLE), N persons, J items, n observed cells.
  Their simulation is binary only, J = 100–400 with N = J or 5J, K* = 3; the paper itself reports
  over-selection when N = J and J is small, and under-selection of a weak factor when N = 5J.
  Our fits maximise the **marginal** (Laplace) likelihood and ecological p is 10–20, so JIC does
  not transfer directly; the paper notes the marginal likelihood approaches the joint one only
  when both N and J are large. Not a v1 candidate.
- **Ordered factor LASSO** (Hui, Tanaka & Warton 2018, Biometrics) is the only GLLVM-native
  single-fit rule. No software implementation was found and no precedent for a non-smooth group
  penalty inside a Laplace/TMB fit. Parked as a follow-up lane (Shinichi, 2026-09-26).
- **The d vs d+1 likelihood-ratio test is non-standard** (Hayashi, Bentler & Yuan 2007; Drton
  2009). Only a parametric bootstrap is a valid reference. `src/boundary_inference.jl` no longer
  lists K-selection as a chi-bar-squared use case.
- **Nothing addresses inference after choosing d in GLLVMs.**

## What we measured (this lane)

- Existing NB fits can sit in poor optima while reporting `converged = true` (NB2, n = 300,
  p = 20, true K = 3; `LOOP/lanes/auto-d-20260926/pilot/runaway_probe.txt`):

  | K | logLik | converged | max latent SD | median | ratio |
  |---|---|---|---|---|---|
  | 2 | −18874 | true | 26.9 | 2.69 | 9.1 |
  | 3 | −19113 | true | 8.7 | 1.40 | 6.1 |
  | 4 | −20240 | true | (not probed) | | |
  | 5 | −16720 | true | 2.45 | 1.41 | 1.75 |

  K = 5 (20 min) reached a region ~2400 logLik units above K = 2–4 with healthy loadings, so the
  K = 2–4 fits are poor optima, not the maxima. The old `select_lv` picks K = 2, itself a runaway.
  The guarded `select_lv` (measured, `pilot/j3_nb_guarded_prefix.txt`, 21.6 min) rejects K = 1
  (latent SD 12.6), K = 2 (26.9) and K = 4 (10.4) as runaways and chooses K = 5 by BIC: it stops
  the runaway being chosen but cannot recover K = 3, because the K = 3 fit is a bad optimum.
  K = 4's 10.4 sits just above the provisional cutoff of 10. **Recovery here needs better NB optimisation (multi-start or warm start in the
  per-species NB route), which the guard cannot supply.** An earlier note in this lane called
  K = 5 a likely runaway; the probe shows it is not.
- Binomial, same size: K = 4 unconverged, latent SD 156, ratio 106 (a clear separation runaway,
  caught by both checks); AIC on the old `select_lv` would have chosen it. K = 3 and K = 5 healthy.
- Recovery grid (4 families × n {30,60,120,300} × p {10,20} × true K {1,2,3} × 200 reps, K fitted
  1..K+2, pre-lane fitting code; loadings 0.8·N(0,1)): DRAC nibi 22744942 (tasks 1–480) and narval
  4064148 (481–960). Interim at 01:00Z, 251 of 960 tasks
  (`pilot/harvest-report-interim-0100Z.md`; `old` = argmin over every returned fit, `new` = guarded):
  - **Gaussian** (all fits healthy): BIC log(p·n) under-selects at small n (n = 30, p = 10, K = 3:
    0.21 exact); BIC log(n sites) 0.53 there and 0.93–1.00 once n ≥ 60 with p = 20; AIC overshoots
    by 10–15% everywhere but is best in the hardest cells. Guard changes nothing (as expected).
  - **Binomial (Bernoulli)**: fits are the problem, not the criterion. Share of attempted fits
    unconverged at K = 2–5: 70–87%; **every unconverged fit is a runaway** (median latent SD 141–295
    on the logit scale), and 17–64% of converged fits exceed latent SD 10. Old AIC over-selects
    wildly (picks broken fits); guarded rules fall back to K = 1 and under-select K = 2, 3
    (≤ 0.07 exact at n ≤ 60). Both BIC variants find K = 1 correctly (≥ 0.96).
  - Update 02:00Z, 818 of 960 tasks, 13 506 datasets (`pilot/harvest-report-interim-0200Z.md`).
    Mean exact recovery across cells (unweighted), by rule (`len` = guard that keeps an
    unconverged fit if it is not runaway and not non-monotone):

    | family | old AIC | old BIC pn | old BIC n | new AIC | new BIC pn | new BIC n | len AIC | len BIC pn | len BIC n |
    |---|---|---|---|---|---|---|---|---|---|
    | binomial | 0.333 | 0.374 | 0.461 | 0.564 | 0.391 | 0.440 | 0.564 | 0.391 | 0.440 |
    | gaussian | 0.858 | 0.864 | **0.948** | 0.858 | 0.864 | **0.948** | 0.858 | 0.864 | **0.948** |
    | nb (partial) | 0.806 | 0.862 | 0.893 | 0.818 | 0.851 | 0.866 | 0.815 | 0.873 | **0.904** |
    | poisson | 0.982 | 0.998 | **0.999** | 0.974 | 0.989 | 0.991 | 0.982 | 0.998 | **0.999** |

    **Final (07:33Z, 960 task files, 17 687 of 19 200 datasets; `pilot/harvest-report-final.md`
    has every cell with MCSE and too-few/too-many splits).** Mean exact recovery:
    Gaussian len BIC-n 0.948; Poisson len BIC-n 0.999; NB len BIC-n 0.893 (old AIC 0.775,
    new BIC-n 0.827); binomial best 0.56 (new/len AIC). The 1 513 missing datasets are all NB
    (17 cells, tasks that hit the 6 h limit); those are the slowest fits, so NB recovery there
    may read optimistic.
    **NB numbers withdrawn (2026-09-27).** Every NB figure on this page was measured with the
    NB2 grouped kernel before #521, whose per-site mode search 2-cycled where y ≫ μ and let
    L-BFGS stop at poor optima reporting convergence. They are kept here as history only; the
    24 NB cells are to be re-run after #521 merges.
    **Gaussian re-run with trait intercepts (2026-09-28, pre-merge).** #518 + #519 in a throwaway
    tree, uncentred data (trait means 3 + N(0, 1), same seeds for loadings, scores and noise), all
    4,800 datasets: exact recovery 0.950 with `:bic_sites` (grid: 0.948) and 0.864 with `:bic`
    (grid: 0.865), 0 failures. A 50-dataset pre-run chose the same K as the old route on centred data
    in 50/50. The Gaussian conclusion holds with intercepts. Evidence:
    `LOOP/lanes/auto-d-20260926/pilot/gaussian_prerun/`.
    Poisson: every BIC rule ≥ 0.95 in every cell. NB: 27–47% of K = 2–4 fits unconverged; small
    cells under-select. Rejecting on the convergence flag alone cost recovery (Poisson, NB), so
    the guard now keeps an unconverged fit unless it is runaway or non-monotone
    (`require_converged = false` default, flagged in `attempts`).

## Built so far (lane branch, not merged)

1. `select_lv` guard: every attempted K recorded with a status; failed, unconverged, runaway and
   non-monotone K are never chosen; interrupts no longer swallowed (caddc8653, bf8940ad2).
2. Warm-start retry from the (K−1) solution plus a lower-triangular new column, where the
   fitter accepts `β_init`/`Λ_init` (same commit).
3. Runaway detector: scale check (max loading row norm = latent SD on the link scale > 10, any
   non-identity link) and binomial ratio check (≥ 25), from the gllvmTMB runaway study
   (commit 9653b1778). Thresholds provisional; recalibrate on the grid's healthy fits.
4. R twin: the same guard and detector in gllvmTMB `select_lv()` (branch claude/lane-auto-d-r-20260926, commit 978f4bba2; 15 new + 70 existing tests pass). Warm start uses `control(start_from = <accepted fit>)`: matching blocks carry over, the new column starts at the default. The table keeps rejected rows with `status` and `message`. Review fixes 21319c652; lenient convergence rule 3fea6b68a (136 expectations pass). Remaining difference: R still rejects a fit whose Hessian is not positive definite (`pd_hessian = FALSE`); Julia has no such flag.

Tests: 75/75 in `test/test_model_selection.jl` (16 existing + 59 new). Most new tests drive the
guard with a stand-in fitter (exact, fast); real-fit coverage is the Poisson omitted-K test, the
explicit-K identity gate (6 real fits bit-identical to the pre-lane source), and the NB and
binomial probes. The interim Gaussian grid shows old = new because Gaussian fits are healthy and
skip the runaway check and warm start; it is not evidence for the guard.

Independent review (D-43 panel, 2026-09-27: Opus statistical lens, Sonnet code lens) returned
PASS WITH REQUIRED FIXES; fixed in bf8940ad2: tolerance is relative (max(1e-3, 1e-6·|ℓ|)); the
"best converged, non-runaway fit at any smaller K" bar the review asked for turned out to equal
the last accepted fit (a third reviewer showed the extra update was dead code), so the practical
change is the tolerance; warm start pads from the last accepted K; the runaway check
skips the Normal family; `mask` reaches the criteria; an ArgumentError at K = 1 is re-raised;
`fit_gllvm` without K refuses row_eff/pervar. Known limits kept: no warm start for Gaussian
(`GllvmFit` keeps Λ in `pars`) or for the NB/Beta per-species route (no `Λ_init`); the returned
fit carries no "K was estimated" flag, so `confint`/`summary` do not yet print the caveat.

## Decisions needed at G1 (each with a recommendation)

| ticket | question | recommendation | why |
|---|---|---|---|
| T2 route | fit-and-compare, shrinkage, spectral, or hybrid? | **Fit-and-compare with the two safeguards (v1). Spectral guess to narrow the K window later (v1.1). OFAL in a separate lane.** | Reuses `select_lv`; no new estimator; the failure we found is in fits, not in criteria. |
| T2 criterion | default criterion | **BIC with log(number of sites) (`:bic_sites`)**: best or joint best for Gaussian, Poisson and NB on the grid (final: 0.948 / 0.999 / 0.893 with the lenient guard); log(p·n), the current convention, under-selects at small n. Chen–Li JIC read and set aside (joint likelihood, J ≥ 100). | No published discrete-data evidence exists; our grid is the evidence. |
| T4 API, Julia | what does omitting K do? | **Omitting K runs `select_lv` and returns the chosen fit**, printing the candidate table once (`@info`); `select_lv` stays the way to get the full record. Today omitting K throws, so nothing that works now changes. Default `Kmax = min(5, p − 1)`. | Shinichi: "OK what if we do not supply d - yes". Return type stays a fit, so downstream code is unchanged. |
| T4 API, R | what does omitting `d` in `latent()` do? | **Flag: today omitting `d` silently means d = 1**, so switching it to auto changes existing users' fits. Recommend `d = "auto"` now, and switching the default in a later minor release with a NEWS warning. | Julia's switch is error → behaviour; R's would be behaviour → different behaviour. |
| T8 runaway thresholds | recalibrate `max_latent_sd = 10`, `ratio_max = 25` on the grid? | **Keep 10 / 25** (2026-09-28, `pilot/runaway_calibration/`, harvest data, NB excluded as old-kernel). Over max_latent_sd ∈ {6..20} × ratio_max ∈ {10..50}, guarded `bic_sites` recovery is identical for Gaussian (0.948) and Poisson (0.989), and moves at most 0.2 points for binomial (0.436 to 0.438), inside Monte Carlo noise; binomial loading statistics are bimodal (almost nothing between 5 and 40), so any threshold in the grid splits them the same way. Open: at ratio 25 about 46% of binomial fits with K ≤ K_true are flagged here (mostly quasi-separation at n = 30 or 60), which does not match the "0/551 false positives" quoted from gllvmTMB; that figure came from a different grid. | Unverified proxy: "broken" = row norm > 50 or non-monotone logLik; no ground truth for runaway. |
| T7 binary data | how to estimate K for Bernoulli data when most K ≥ 2 fits run away? | **Sweep with a loading ridge (gllvmTMB `aghq_ridge = 2`, Laplace + ridge) and compare BIC on the unpenalised logLik at the ridge optimum (what gllvmTMB's `logLik` already returns, with a warning); add the same ridge to Julia's binomial fitter (a family-kernel change, after the overnight lane).** Below a size where even the ridge cannot recover K (p = 10, n ≤ 120 here), return the table and say the data cannot resolve d. | R experiment (`ridge/`, 10 reps per cell, BIC): loadings 1.5·N(0,1), n = 120, p = 20: K = 2 correct 8/10 with ridge vs 4/10 without; K = 3 4/10 vs 1/10; p = 10, K = 2: 1/10 vs 0/10; without ridge ~half of sweeps have no admissible d. Loadings 0.8: ridge recovers K = 1 10/10 (vs 6/10) but not K = 2. The ridge still under-selects because its optimum shrinks the loadings, trimming each added dimension's logLik gain. Sweeps with ridge are ~7× faster. **Julia (2026-09-27, `ridge/ridge_binary_julia_L1.5.csv`, 10 reps per cell, loadings 1.5·N(0,1), `:bic_sites`, `Kmax = K + 2`, `binary_ridge = 2` vs `Inf`):** n = 60, p = 10, K = 2: 5/10 vs 0/10 (4 with no admissible K); n = 120, p = 10, K = 2: 9/10 vs 1/10 (5); n = 120, p = 20, K = 2: 10/10 vs 5/10; n = 120, p = 20, K = 3: 9/10 vs 1/10 (3). Ridge sweeps 1 to 11 s vs 22 to 148 s. **Not like-for-like with R:** R used `criterion = "bic"` (log(p·n)) and `d_max = 4`; its misses at p = 10 are nearly all d = 1 picks, the direction the heavier penalty predicts. **Matched run** (`ridge/ridge_binary_julia_L1.5_bic.csv`, Julia `:bic`, `Kmax = 4`, same Julia datasets): ridge / none = 1/10 / 0/10, 8/10 / 0/10, 8/10 / 4/10, 4/10 / 0/10 for the four cells; R = 1/10 / 0/10, 1/10 / 0/10, 8/10 / 4/10, 4/10 / 1/10. Julia and R agree in three cells; at n = 120, p = 10 with the ridge they differ (8/10 vs 1/10) on different random datasets, an open question. `:bic_sites` beats `:bic` under the ridge in every cell. **Same datasets in both engines (2026-09-28, `ridge/matched/`: gllvmTMB `select_lv` run on the 40 exported Julia datasets):** with the ridge, `bic`: Julia 1, 8, 8, 4 of 10 vs R 1, 7, 8, 3; `bic_sites`: Julia 5, 9, 10, 9 vs R 3, 8, 10, 4. So the n = 120, p = 10 gap (8 vs 1) came from the two languages' different random datasets, not the engines. The remaining gap (n = 120, p = 20, K = 3, `bic_sites`: 9 vs 4) is a guard difference: R marks ridge fits at d ≥ 2 as "unconverged" when their Hessian is confirmed non-positive-definite (or, rarely, their logLik cannot be read) and excludes them (6 of 10 datasets fall to d = 2 or 1), while the Julia guard does not check the Hessian and keeps them. Which rule is right is open (Shinichi). Lane recommendation (2026-09-28): keep Julia's rule and relax R's, keeping `pd_hessian = FALSE` as a message in the table rather than a rejection under the ridge, because on these same datasets the check discards the correct K = 3 fits (9/10 recovered without it, 4/10 with it). Cause, confirmed 2026-09-28 (`ridge/matched/pdhess_check.R`, `.log`): gllvmTMB adds the loading ridge in R, outside the TMB template, and `TMB::sdreport(obj, par.fixed = opt$par)` then tests the Hessian of the unpenalised objective at the penalised optimum. At all six flagged fits (reps 3, 5, 6; d = 3 and 4) the unpenalised Hessian has a negative eigenvalue (-0.005 to -0.16) while the penalised Hessian, with 1/τ² added on `theta_rr_B`, is positive definite (smallest eigenvalue 0.088 to 0.245) and the penalised gradient is 1e-4 to 3e-4. This is the same defect gllvmTMB #1092 fixed for the gradient. The R fix is therefore to test the penalised Hessian under the ridge, not to drop the check, so both engines accept the same fits. **The grid's own binomial cells with the default ridge (2026-09-28, `ridge/grid_binomial/`: loadings 0.8·N(0,1), β = 0, 24 cells × 50 reps):** exact recovery 0.412 with `bic_sites` (grid without ridge, 200 reps: 0.437) and 0.364 with `bic`; K = 1 400/400; K = 2 and 3 found almost only at n = 300, p = 20 (35/50, 24/50); 706 under-selections and 0 over-selections. At this weak signal the ridge does not raise recovery (within Monte Carlo noise of 50 reps) but removes over-selection; its gain is at strong loadings (1.5·N(0,1): 9 to 10 of 10 vs about 4 of 10). The default `binary_ridge = 2` therefore buys safety (never too many dimensions) rather than accuracy at weak signal; docs should say so. It also removes a failure mode: without the ridge, 27 to 29% of the grid's binary datasets get no accepted K at all (`select_lv` errors); with it, 0 of 1,200 do. |
| T5 caveat | wording only, or propagate K uncertainty? | **Wording now** (below). A bootstrap that re-selects K per replicate is a later option. | Honest now; the bootstrap costs Kmax fits per replicate. |
| T6 scope | which routes get auto in v1? | **Default family route, including NB/Beta via their per-species dispersion route.** Those routes need `β_init`/`Λ_init` in `grouped_dispersion.jl` for the warm-start retry; without it the guard still rejects bad K. That file belongs to the overnight lane until 11:00Z; raise after. | NB is the most-used count family. |

## Caveat text (for the docstring and docs page)

> The number of latent dimensions is chosen from the same data the model is then fitted to.
> Intervals, p-values and tests reported for that fit are conditional on the chosen number and do
> not include uncertainty about it; when the top two candidates are close, treat the choice as
> uncertain. Species correlations, variance partitions and ordination axes all depend on the
> chosen number, and individual axes can change meaning when it changes. Under model
> misspecification the chosen number tends to grow with sample size, so read it as "dimensions
> the data support at this sample size", not as the number of true gradients. Recovery rates in
> the documentation hold for the simulated settings reported there.

## Out of scope

OFAL / shrinkage build (follow-up lane), grouped/phylo/row-effect/per-variance routes, selective
inference theory, HSquared reuse (after this lands).
