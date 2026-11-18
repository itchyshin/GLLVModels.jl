# N1 follow-up diagnosis: why was the phy-tier R-vs-Julia bootstrap comparison inconclusive?

Evidence note only. No row status, scoreboard, receipt or tolerance is changed. It follows the N1 result note (`docs/dev-log/check-log.d/2026-11-03-bootstrap-sigma-phy-n1-result.md`, draft PR #1090) and the N1 draft registration (`docs/dev-log/plans/2026-11-02-bootstrap-sigma-phy-n1-prereg-DRAFT.md`, draft PR #1078). The text below is the diagnostic's own report, carried over with its numbers unchanged. The N1b proposal at the end is a proposal for Shinichi's approval, written up as an unsigned draft in `docs/dev-log/plans/2026-11-03-bootstrap-sigma-phy-n1b-prereg-DRAFT.md`.

Date 2026-11-03. Read-only diagnostic. No tracked file edited, no PR, no push; nothing deleted. PR #990, `src/confint_derived_wald.jl` and Codex branches not touched.
Everything below is evidence from re-running the same Julia code (origin/main 85e47a2a9 tree copied on Totoro) and the same R oracle library. Proposals at the end are for Shinichi's approval only.

Compute: Totoro, single-thread processes, `OPENBLAS_NUM_THREADS=1`, `taskset` on at most 4 cores. Estimated 1 to 2 core-hours before starting (under the 3 h line); actual about 1.5 core-hours, about 30 min wall. Scripts, per-replicate outputs and logs: `~/hsq_work/claude-n1diag-20261103/` on Totoro (`out_<ds>/fail_*.csv`, `Rrefit_*.csv`, `pairedR_*.csv`, `jcmp_*.txt`, `q2_out.txt`, `q1r_out.txt`, `q4_out.txt`, `null_out.txt`, `nb_out.txt`, `paired_out.txt`). Scripts were also kept in the session scratchpad, which is purged; the Totoro folder is the durable copy.

Definitions used (they reproduce the N1 note exactly: within-engine floors 0.770, 0.448, 0.499, 0.440 for R and 0.457, 0.245, 0.588, 0.725 for Julia): 20 cells (15 lower-triangle Sigma_phy entries + 5 communalities), 2 endpoints = 40; type-7 percentiles of finite draws; z = |difference of endpoint| / SE_pool, SE_pool = sqrt((SD_a^2 + SD_b^2)/2) of the finite draws; cells with SE_pool < 1e-3 skipped.

Reproduction check: regenerating the batch with `_bsig_pmv_simulate_batch(MersenneTwister(seed), fit, 999)` and refitting with the driver's own call reproduced exactly the failed index sets (5/0, 7/6, 12/11, 38/52) from the NaN rows of the stored draws.

---------------------------------------------------------------------------------------------------

## Q1. What are the 131 Julia-failed replicates?

Total failed = 5+0+7+6+12+11+38+52 = 131 (seed 2001 + 2002).

**Type.** None was thrown (0 of 131). The driver rule (`converged = Optim.converged && finite objective && FD gradient max-norm <= g_tol = 1e-5`, default `iterations = 400`) failed them in two ways:

| Dataset | failed | stopped early, FD gradient above 1e-5 ("gradient_not_converged") | hit the 400-iteration limit |
|---|---|---|---|
| 1101 | 5 | 5 | 0 |
| 1102 | 13 | 12 | 1 |
| 1103 | 23 | 14 | 9 |
| 1104 | 90 | 14 | 76 |
| total | 131 | 45 | 86 |

- Early stops (45): final max-norm gradient finite in 42, median 1.97e-05, range 1.0e-05 to 2.6e-04; in 3 the finite-difference gradient was Inf (an FD step leaves the valid objective region: 1102/2001/b160, 1103/2001/b531, 1104/2001/b340). These are line-search stalls at the finite-difference noise floor: the objective is flat to 1e-12 and LBFGS cannot move, but the FD gradient is just above 1e-5. Iterations to stop: 54 to 269.
- Iteration-limit (86): final gradient median 3.1e-04 (range 1.1e-05 to 0.173); 59 of 86 are at or below 1e-3. 76 of the 86 are on dataset 1104, whose original fit already has a boundary unique variance (1.6e-10) and a non-PD Hessian.

**Gap to the best logLik** (best over: default, 3000-iteration cold start, warm start from the original fit, restart from the default's final estimate; "gap" = best minus default final):

- gap <= 1e-6: 100 of 131; gap <= 1e-3: 123 of 131; gap > 1e-3: 8 of 131, all iteration-limit replicates on 1104 (0.0020, 0.0138, 0.0140, 0.0221, 0.0222, 0.336, 0.615, 1.059).
- Early stops (45): maximum gap 5.5e-05 (the rest below 1e-8). They are at the optimum.
- Re-running: with `iterations = 3000` and the same cold start, 84 of the 86 iteration-limit failures converge; 47 of the 131 still fail (the 45 early stops plus 2 iteration-limit cases). Remaining per seed (2001/2002): 1101 5/0, 1102 6/6, 1103 7/7, 1104 8/8. Under `g_tol = 1e-3` plus `iterations = 3000`, only 4 of 131 would still fail (3 Inf gradients and one gradient > 1e-3 on 1104/2001).
- Caveat: a warm start from the original fit can land in a worse local optimum (e.g. 1102/2001/b116 by 1.35, b160 by 1.15 logLik units), so the warm start is not a safe default.

**Would R's rule have called them converged?** I refit every one of the 131 failed datasets with R (gllvmTMB at the pinned oracle library, exact formula, `gllvmTMBcontrol(se = FALSE)` as `bootstrap_Sigma` uses, cold start). R returned `convergence == 0` ("relative convergence (4)") for 131 of 131. R's logLik equals the best Julia logLik within 1e-4 in 121 of 131 (1101 4/5, 1102 11/13, 1103 23/23, 1104 83/90); in the other 10 the two engines stopped at different points: Julia best higher in 9 (by 0.014 to 0.62; 1101 1, 1102 1, 1104 7) and R higher in 1 (1102/2002/b350, by 0.61).

So: R's rule would have kept all 131; the gap in failure rate (3.8% vs 0% on 1104) is a rule difference (Julia's extra FD-gradient check plus a 400-iteration cap), not evidence that these replicates are bad data. R's own refit also took tens of seconds on these Ys against about 0.1 s on typical ones, so they are genuinely harder than average, but R did not drop them.

---------------------------------------------------------------------------------------------------

## Q2. How much do the Julia intervals move if failed replicates are kept?

Scenario S1 keeps each failed replicate at the optimiser's final estimate; S2 keeps the best-logLik refit; S3 keeps the default final estimate only if its gap < 1e-3. Max over the 40 endpoints of |kept - dropped| / SD of the survivors' draws:

| Dataset | seed | failed | S1 keep final | S2 keep best refit | S3 near-optimal only |
|---|---|---|---|---|---|
| 1101 | 2001 | 5 | 0.022 | 0.022 | 0.022 |
| 1102 | 2001 | 7 | 0.045 | 0.045 | 0.045 |
| 1102 | 2002 | 6 | 0.073 | 0.073 | 0.073 |
| 1103 | 2001 | 12 | 0.037 | 0.037 | 0.037 |
| 1103 | 2002 | 11 | 0.027 | 0.027 | 0.027 |
| 1104 | 2001 | 38 | 0.097 | 0.097 | 0.061 |
| 1104 | 2002 | 52 | 0.144 | 0.144 | 0.121 |

The largest shift is 0.144 SE (1104, seed 2002, Sigma11 upper). Keeping the failures also barely changes the cross-engine comparison: max z against R on the same seed is 0.909/0.327/0.370/0.642 (dropped) against 0.931/0.332/0.369/0.659 (S2) for 1101 to 1104 on seed 2001.

---------------------------------------------------------------------------------------------------

## Q3. Is 0.35 a plausible noise-floor limit for the max over 40 endpoints at n_boot = 999?

No. It is below the median of the maximum.

- Normal-draw theory: asymptotic SE of one 2.5% or 97.5% endpoint = sqrt(p(1-p)/(n f^2)) = 0.0845 draw-SD at n = 999; difference of two independent runs has SD 0.1195 draw-SD. Monte Carlo (200,000 reps, 999 normal draws): single endpoint |d|/SE_pool mean 0.094, 95th percentile 0.232. Maximum over K independent endpoints: K = 20: mean 0.259, q95 0.360; K = 30: 0.278 / 0.375; K = 40: 0.290 / q95 0.385, q99 0.438. So even for ideal normal draws, 0.35 is roughly the q90 to q95, a 5 to 10% false-alarm rate, under independence.
- Real draws are skewed or bounded, so the max is larger. Monte Carlo from the actual stored draws (R and Julia pooled, both seeds, as the population; two independent resamples of n_boot each; 2000 reps for 999, 400 for the others; at n_boot above about 2000 the finite 4000-draw population makes these approximate): max |dEnd|/SE_pool, median / q90 / q95 / q99:

| Dataset | n_boot 999 | 1999 | 4999 | 9999 |
|---|---|---|---|---|
| 1101 (boundary communalities) | 0.50 / 1.09 / 1.28 / 1.70 | 0.36 / 0.74 / 1.00 / 1.21 | 0.22 / 0.48 / 0.56 / 0.76 | 0.16 / 0.33 / 0.40 / 0.52 |
| 1102 | 0.39 / 0.54 / 0.60 / 0.70 | 0.28 / 0.39 / 0.44 / 0.53 | 0.18 / 0.25 / 0.29 / 0.34 | 0.12 / 0.17 / 0.19 / 0.23 |
| 1103 | 0.39 / 0.58 / 0.66 / 0.75 | 0.28 / 0.38 / 0.44 / 0.55 | 0.19 / 0.27 / 0.31 / 0.37 | 0.14 / 0.20 / 0.23 / 0.30 |
| 1104 (boundary, non-PD Hessian) | 0.45 / 0.71 / 0.79 / 1.00 | 0.33 / 0.54 / 0.60 / 0.71 | 0.22 / 0.36 / 0.40 / 0.55 | 0.15 / 0.25 / 0.29 / 0.37 |

Every observed within-engine floor in the N1 note (0.25 to 0.77) lies at or below the q95 for its dataset (R 1101: 0.77 against q95 1.28). They are ordinary Monte Carlo noise. A limit of 0.35 has about a 40 to 60% chance of being exceeded by two same-engine runs at n_boot = 999 (median 0.39 to 0.50).

- The observed cross-engine maxima are consistent with that null. Null = two independent n=999 samples from the pooled R+J draws. Observed R-vs-J max z and p(null >= observed): 1101 0.909 (p 0.17) and 0.399 (p 0.70); 1102 0.327 (p 0.79) and 0.407 (p 0.43); 1103 0.370 (p 0.58) and 0.421 (p 0.40); 1104 0.642 (p 0.16) and 0.518 (p 0.35). Nothing separates R from Julia at this resolution. The pooled population contains any real engine difference, so this is a consistency check, not a proof of agreement.

Scaling: noise falls as 1/sqrt(n_boot). To get the median max to 0.35 you need n_boot of about 2000 (interior datasets); to get q95 under 0.35 you need about 5000 (interior) or 10,000 (boundary datasets 1101, 1104).

---------------------------------------------------------------------------------------------------

## Q4. Is selection from Julia's extra losses a plausible cause of cross-engine differences?

Not a material one.

- The failed replicates are not extreme. Using the best refit of each failed replicate: the fraction of failed replicates falling outside the survivors' 2.5/97.5 percentiles is 4.0 to 10% on average over the 20 cells (null 5%), 4.7% (1104/2001, 38 failures) and 5.7% (1104/2002, 52 failures); mean 2*|percent rank - 0.5| is 0.48 to 0.57 (null 0.50). They look like a random subset of the draws.
- Actual effect of keeping them: at most 0.144 SE (Q2).
- Worst case (all k lost replicates were the k most extreme draws in the same direction, which the data show did not happen): endpoint shift 0.24 to 0.48 SE for the datasets with 5 to 12 failures and 1.14 SE (k = 38) to 1.51 SE (k = 52) on 1104. So the 3.8% rate on 1104 could in principle matter, but the data show it did not.
- Rule 1b (rate gap 3.8 pp on 1104) therefore flags a real difference in failure rules, but not a bias in the intervals.

Separate finding from the paired runs below: the engines do not always settle at the same optimum.

---------------------------------------------------------------------------------------------------

## Extra: paired design pilot (same simulated Y for both engines)

I took the Julia batch for seed 2001 (999 Ys per dataset, simulated by `_bsig_pmv_simulate_batch`), refit every Y with R (default `gllvmTMBcontrol(se = FALSE)`, R's convergence rule) and extracted with R's own `.extract_summaries` (phy, Sigma/R/communality), then compared replicate by replicate with the Julia draws.

| Dataset | Julia failed | R failed (code != 0) | median over reps of max paired |R-J|/SD | replicates with a paired difference > 0.05 SD | interval max z, each engine's own rule | max z, common survivors |
|---|---|---|---|---|---|---|
| 1101 | 5 | 0 | 2.3e-06 | 40 of 994 | 0.323 | 0.324 |
| 1102 | 7 | 0 | 2.5e-06 | 59 of 992 | 0.215 | 0.215 |
| 1103 | 12 | 1 | 2.2e-06 | 8 of 986 | 0.037 | 0.032 |
| 1104 | 38 | 1 | 2.3e-06 | 48 of 960 | 0.098 | 0.134 |

- For most replicates the engines agree to about 1e-6 SD, so pairing removes nearly all simulation noise. The interval max z is 0.04 to 0.32 (against unpaired null medians 0.39 to 0.50) and the SE ratio R/J lies in 0.92 to 1.06.
- 0.8% to 6% of replicates (8, 40, 48, 59) have a paired disagreement above 0.05 SD, up to 5.7 SD. Refitting those in Julia and comparing logLik with R's shows both engines report convergence at different local optima, with logLik differences of 0.01 to 4 units: R higher in 6 of 8 (1103), 34 of 40 (1101), 41 of 59 (1102) and 27 of 48 (1104); Julia higher in the rest (2, 6, 18, 21). Zero replicates agree at the same optimum within 1e-4 in this list, by construction. This is a real engine behaviour (cold-start path dependence on a multimodal likelihood), not noise, and it needs a descriptive line in any N1b report. It does not move the interval summaries much.
- Limitation: the paired design checks refit plus extraction, not the simulator. R's `simulate()` and Julia's `_bsig_pmv_simulate_batch` are not compared by this pilot.

---------------------------------------------------------------------------------------------------

## Proposal for N1b (for Shinichi's approval; nothing below is adopted)

1. **Make pairing the primary design.** Julia (or R) simulates once, both engines refit the same Ys. In the pilot this brings the max z to 0.04 to 0.32 at n_boot = 999, where the unpaired null median is about 0.4. Keep a separate cheap check that R's `simulate()` and Julia's batch simulator agree (for example sample covariance and phylogenetic-covariance moments of a few thousand simulated Ys against the theoretical Sigma), because pairing bypasses it.
2. **Threshold.** The 0.35 limit is not defensible for unpaired runs at n_boot = 999. For the paired design the observed values in the pilot are at most 0.32, so I propose Agree if paired max z <= 0.50 at n_boot = 999 (a margin over the pilot maximum, with a pre-registered report of the count of replicates whose two logLik differ by more than 1e-4). For an unpaired fallback use the n_boot-specific null: q95 of the max of about 0.6 to 0.8 at 999 for interior and mildly boundary datasets (1.3 for 1101 type boundary cases) or n_boot = 5000 with limit 0.50 (q99 0.34 to 0.55 on the three well-behaved datasets, 0.76 on 1101). Report the observed value together with its null quantile rather than a bare cut-off.
3. **n_boot.** 999 is enough if paired. If unpaired is kept, 4999 for interior datasets and 9999 for boundary datasets; Julia 1104 took 865 s per 999 refits, so 4999 is about 1.2 core-hours per run, still under the 3 h line per run.
4. **Failure-rule parity.** Either (a) state both rules and compare R on its own rule against Julia on a wrapped refit that uses `iterations = 3000` and a gradient check at `g_tol = 1e-3` (in the diagnostic this leaves only 4 of 131 failures), or (b) report Julia with the driver rule and R after applying the same rule, and drop rule 1b as a verdict rule (keep it descriptive). A warm start from the original fit is not recommended (worse local optimum in some replicates). The failure rate gap should not be a verdict rule while the rules differ.
5. **Boundary datasets.** 1101 and 1104 have boundary unique variances (R 1.2e-08 / 6.5e-09, Julia 1.4e-10 / 1.6e-10) and 1104 a non-PD Hessian. Keep them but pre-register that they are expected to be noisier, and add two interior-only datasets (1102 and 1103 are cleaner: unpaired q95 of max about 0.6 to 0.66 at n_boot = 999).
6. **Control parity.** The original N1 R fit used BFGS with `reltol = 1e-14`, but `bootstrap_Sigma` refits with the default nlminb control (`se = FALSE`). State this explicitly in N1b; it does not matter for the comparison but it differs from what the draft described.
