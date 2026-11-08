# N1b pre-registration, DRAFT: paired R-versus-Julia bootstrap comparison, phy tier

Status: APPROVED BY DELEGATION on 2026-11-03 (Section 10). The full run (Section 7) may start. Nothing here changes a row status, a scoreboard, a receipt or a tolerance. No run was made for this draft. The thresholds below are proposals taken from the null analysis cited in Section 1; they were not simulated beyond that analysis.

Date drafted: 2026-11-03. Drafted by a Claude session; not yet reviewed by anyone.

## 1. Why N1b, and what it changes

N1 (result note `docs/dev-log/check-log.d/2026-11-03-bootstrap-sigma-phy-n1-result.md`, draft PR #1090; registration `docs/dev-log/plans/2026-11-02-bootstrap-sigma-phy-n1-prereg-DRAFT.md`, draft PR #1078) was Inconclusive on all four datasets. The reason was the noise-floor rule: the within-engine seed-to-seed largest standardised endpoint difference was 0.25 to 0.77, above the 0.35 limit. The diagnosis (`docs/dev-log/check-log.d/2026-11-03-bootstrap-sigma-phy-n1-diagnosis.md`) found:

- 0.35 is below the null median of the maximum over 40 endpoints at n_boot = 999. For normal draws the mean is 0.29 and the 95th percentile 0.385. For the real, skewed draws the median is 0.39 to 0.50 and the 95th percentile 0.60 to 1.28. Every observed within-engine floor lies at or below its dataset's 95th percentile.
- The 131 Julia failed replicates were 45 early gradient stalls at the finite-difference noise floor and 86 iteration-cap hits, 76 of them on boundary dataset 1104. 100 of the 131 were within 1e-6 of the best logLik. R returned convergence 0 on all 131. Keeping the failed replicates shifts endpoints by at most 0.144 SE.
- A paired pilot (the same 999 Julia-simulated Ys, seed 2001, refit by R) gave a largest endpoint z of 0.323, 0.215, 0.037 and 0.098 for datasets 1101 to 1104. In that pilot 40, 59, 8 and 48 replicates (about 1% to 6%) ended at different local optima in the two engines.

N1b therefore makes three changes to N1: the primary design is paired (both engines refit the same bootstrap datasets), the thresholds come from the null analysis, and the failure rules are stated for both sides with the failure-rate gap descriptive only. It is still not a coverage or recovery study, and truth is not used in the agree rule.

## 2. Pinned versions

Same as N1 Section 2, with the same pins:

- R: gllvmTMB 0.7.1 at commit `9539352f66f2db2cc26b1c393e67212a359b60c9`, library `/home/snakagaw/hsq_work/gllvm-w2-3/oracle-build/library` on Totoro, checked on the day with `verify_noversion.py` (the stock verify fails on the missing `gllvmTMB_version` key, PR-F1). Output stored with the results.
- Julia: GLLVModels.jl at the exact commit recorded at launch (Section 10), Julia 1.12.6, and the frozen Manifest.toml (sha256 `ba1f3285b0191304b17a003cb8baf07024b566c27c55caac6bb8913a3fea4c40` in N1; record the hash actually used).

## 3. Data-generating process and datasets

Same DGP as N1 Section 3 (`sim.jl`, 30 tips, d = 5, 2 replicates per species, 300 rows, rank 1 phylogenetic factor plus phylogenetic unique plus shared residual, same truth values).

Datasets, K = 4, fixed before any N1b run:

| Dataset | `sim.jl` seed | Role | Source |
|---|---|---|---|
| 1101 | 1101 | boundary (N1: unique variance trait 4, R 1.2e-08, Julia 1.4e-10) | kept from N1 |
| 1104 | 1104 | boundary, non-PD Hessian in Julia (N1: trait 1, R 6.5e-09, Julia 1.6e-10) | kept from N1 |
| 1201 | 1201 | interior target | new seed |
| 1202 | 1202 | interior target | new seed |

The new seeds 1201 and 1202 are consecutive integers, not screened and not run before this draft. N1 datasets 1102 and 1103 were interior but are not reused, so the interior arm is not a re-analysis of data already seen.

Interior definition, applied at the eligibility step before the comparison: a dataset counts as interior if the smallest unique variance in both engines' original fit exceeds 1e-3 (N1 interior values were 0.024 and 0.033; boundary values were 1e-8 to 1e-10). If 1201 or 1202 turns out to be a boundary fit, it is replaced by the next seed in that series (1203, 1204, and so on) and the replacement is reported. Boundary datasets 1101 and 1104 are pre-registered as expected to be noisier and are not replaced for being boundary.

Eligibility, as in N1: both engines converge on the original fit (R `convergence == 0`, Julia `converged == true`) and the two Sigma_phy point estimates agree within 1e-5 elementwise. A dataset that fails is replaced by the next seed in its series and the replacement is reported.

## 4. Design and settings

Primary design, paired. For each dataset:

1. Fit the original model in both engines as in N1 (R: `value ~ 0 + trait + phylo_latent(species, d = 1, tree = tree) + phylo_unique(species, tree = tree)`, Gaussian, `REML = FALSE`, `engine = "tmb"`; Julia: `fit_precision_multivariate(rank = 1, mode = :explicitunique, residual_mode = :shared)`).
2. Simulate one batch of 999 bootstrap datasets from the Julia fit, using the Julia batch simulator with `MersenneTwister(2001)`. The diagnosis reproduced this exact step (`_bsig_pmv_simulate_batch(MersenneTwister(seed), fit, 999)`).
3. Refit every one of the 999 Ys in Julia and in R, cold start in both (no warm start: it can land in a worse local optimum, shown on 1102).
4. Extract Sigma_phy, R_phy and communality_phy from each refit in each engine (R with its own `.extract_summaries`), and compare replicate by replicate and by interval.

R control: the refit uses the default `nlminb` control with `gllvmTMBcontrol(se = FALSE)`, as `bootstrap_Sigma` does. N1 Section 4 described a BFGS control with `reltol = 1e-14`; that applies to the original fit only, not to the bootstrap refits. This is stated here so it is not read from the N1 draft.

Julia refit: the driver's refit function wrapped (through the `_refit` argument) to use `iterations = 3000` and a finite-difference gradient check at `g_tol = 1e-3`. In the diagnosis this leaves 4 of 131 previously failed replicates still failing. n_boot = 999 on every run.

Summaries compared and quantile rule: as N1 Section 5 (15 Sigma_phy entries and 5 communalities, 2 endpoints each, 40 endpoint comparisons; type-7 percentiles of finite draws recomputed outside both packages; the packages' own bounds checked against the recomputation; R_phy descriptive only; SE is the SD of the finite draws; cells with SE_pool below 1e-3 skipped in the z rule and handled by the absolute rule in Section 6).

## 5. Simulator-moment check (separate, small)

Pairing bypasses the simulator, because both engines refit the Julia-simulated Ys. The diagnostic pilot did not compare R's simulation with Julia's. This check does so cheaply, with no refitting:

- For each dataset, simulate a few thousand Ys with the Julia batch simulator and with the simulation step used by R `bootstrap_Sigma` (to be confirmed at signing which R function that is and how it is called), both from the same fitted parameters.
- Compare the sample covariance of the traits and a phylogenetic-covariance moment (species-level covariance across traits) of each batch with the theoretical values implied by the fit.
- Proposed rule, not simulated: every entry's absolute difference from theory is at most 4 Monte Carlo standard errors for each simulator. Failures are reported descriptively with the entries named. This check does not enter the Section 6 verdict; it limits what an Agree verdict can claim (an Agree is about refit and extraction given the same Ys, and about simulation only if this check passes).

## 6. Decision rules

Notation. SE_pool = sqrt((SD_R^2 + SD_J^2) / 2) of each engine's finite draws. dEnd = endpoint_J minus endpoint_R. z = |dEnd| / SE_pool. The paired max z is the largest z over the 40 endpoint comparisons, each engine using its own survivors (its own failure rule; Section 4). The same value on common survivors (replicates finite in both engines) is reported beside it. Both were reported in the diagnosis pilot and differed by at most 0.036.

Eligibility first (Section 3). Then, per dataset:

1. Inconclusive if either engine has fewer than 950 finite draws in any cell. (This replaces N1 rules 1a to 1c: the failure-rate gap is no longer a verdict rule and there is no within-engine noise-floor rule, because pairing removes the Monte Carlo noise those rules guarded against.)
2. Agree if the paired max z is at most 0.50, and the SE ratio SD_J / SD_R lies in [0.80, 1.25] for all 20 cells (the pilot range was 0.92 to 1.06), and the absolute rule below holds. The 0.50 limit is a proposal: it sits above the pilot maximum of 0.32. For a cell with SE_pool below 1e-3 the rule |dEnd| <= 0.01 is used instead of z. For communality_phy the absolute difference |dEnd| <= 0.05 must also hold at every endpoint.
3. Disagree if the paired max z exceeds 1.00 or an SE ratio falls outside [0.67, 1.50]. The 1.00 limit and this ratio band are placeholders for Shinichi to set; nothing in the null analysis fixes them, because it concerns unpaired runs.
4. Otherwise Inconclusive.

Overall verdict across the K = 4 datasets: Agree if all four Agree; Disagree if at least one Disagree (the dataset and cell are named); Inconclusive otherwise. An Agree verdict is evidence about four datasets, one tree size, rank 1, d = 5, the explicit-unique shared-residual configuration, and cold-start refits under the stated failure rules. It is not a coverage, recovery or general parity claim and does not by itself move any row status.

Pre-registered descriptive reports, not used in the verdict:

- The count of replicates, per dataset, whose two logLik values differ by more than 1e-4, split by which engine has the higher logLik. The pilot found 40, 59, 8 and 48 replicates ending at different local optima (about 1% to 6%) and attributed it to cold-start path dependence on a multimodal likelihood; N1b reports the same counts, and does not fit or interpret them further.
- Per replicate, the largest paired |R - J| / SD, with the number above 0.05 SD.
- Failed replicates per engine and their rates. The gap in failure rate is descriptive only, because the two rules differ (Julia: wrapper with `iterations = 3000` and `g_tol = 1e-3`; R: `nlminb` convergence code not equal to 0).
- Truth containment of each engine's 95% intervals (four datasets cannot estimate coverage).
- R_phy, as in N1.

Unpaired fallback. Used only if the paired design cannot be run (for example R cannot refit Julia-simulated Ys). It is the N1 design with a threshold taken from the n_boot-specific null rather than a bare cut-off. For each dataset report the observed within-run cross-engine max z together with the null quantile for that dataset and n_boot, computed as in the diagnosis (two independent resamples of n_boot from the pooled draws). Proposed limit: Agree if the observed max z is at most the null 95th percentile; at n_boot = 999 the diagnosis gives about 0.6 to 0.8 for interior and mildly boundary datasets and 1.28 for a 1101-type boundary dataset. Or use n_boot = 4999 with limit 0.50 (null 99th percentile 0.34 to 0.55 on the three well-behaved datasets, 0.76 on 1101); for boundary datasets the diagnosis suggests 9999. The unpaired fallback does not claim equivalence; a result at or below the null quantile means the data cannot tell the engines apart at that resolution.

## 7. Full run (NOT launched; requires approval)

4 datasets, one paired batch each: 4 Julia runs and 4 R runs of 999 refits (about 8,000 refits), plus 8 original fits and the Section 5 simulator check.

Compute estimate, grounded in the N1 run: 16 runs of 999 refits took about 1.6 core-hours (about 27 minutes wall at 8 processes, load average 10 to 18), roughly 0.1 core-hour per run on average. Eight runs are therefore about 0.8 core-hours. Dataset 1104 on the Julia side took 865 s (0.24 core-hours) per 999 refits in the diagnosis; if both boundary datasets and the wrapper's larger iteration cap push Julia runs to that cost, the total is about 8 x 0.24 = 1.9 core-hours as a ceiling. The wrapper (3000 iterations) was not timed for a full batch; in the diagnosis it converged 84 of the 86 iteration-limit failures, so the extra cost falls on a few percent of replicates. Estimate for the plan: 0.8 to 1.9 core-hours, under the 3 h line. A run that overruns its estimate stops and re-reports.

Resource plan: Totoro, one thread per process (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`), `taskset` on at most 4 cores, `nice`, well under the 150-core cap. No GitHub Actions. Raw outputs, per-replicate refit results and logs go to `~/hsq_work/claude-n1b-<date>/` on Totoro (not `/scratch`). Julia warnings go to a file, not the terminal.

Unpaired fallback at n_boot = 4999 on interior datasets would cost about 1.2 core-hours per Julia run on a 1104-like dataset (865 s per 999 refits, scaled); n_boot = 9999 on boundary datasets would be about 2.4 core-hours per such run, close to the 3 h line, and would need its own plan and pre-run test before approval.

## 8. Pilot status

No N1b pilot was run for this draft. The evidence behind it is the diagnosis note: a paired pilot at seed 2001 on datasets 1101 to 1104 (R refits of Julia-simulated Ys), the reproduction of the Julia failed index sets, and the null analysis. The 1201 and 1202 datasets have not been simulated or fitted. The wrapper settings in Section 4 were evaluated on the 131 failed replicates only, not on a full batch.

## 9. Known limits of the design

- Four datasets, one tree size, rank 1, d = 5, shared residual, no missing cells, no REML, one Julia and one R version.
- Pairing checks refit plus extraction, not the simulator; Section 5 is a small separate check and its rule is proposed, not simulated.
- The engines can end at different local optima on 1% to 6% of replicates. This is reported, not removed, and it limits how tightly any paired threshold could be set.
- The Disagree limits (1.00, ratio band [0.67, 1.50]) and the SE ratio band for Agree are not supported by the null analysis, which concerns unpaired runs. The Agree limit 0.50 rests on a pilot maximum of 0.32 from four datasets.
- Cold-start refits on boundary datasets can stall or hit the iteration cap in Julia; the wrapper reduces but does not remove this (4 of 131 still failed in the diagnosis).
- Interior datasets 1201 and 1202 are unscreened and might not be interior; the replacement rule in Section 3 covers that.
- Thresholds are proposals for Shinichi to change.

## 10. Approval (by delegation)

Shinichi delegated this approval in chat on 2026-11-03 ("you can do 1 and 2 if you like", in reply to a request to sign this pre-registration). It is recorded here as approval by delegation, not as Shinichi's own signature; Shinichi can still change any threshold below before the results are read, and the thresholds are fixed from the moment the N1b run is launched.

- Approved by: Claude (Sonnet 5.5) under Shinichi's chat delegation of 2026-11-03
- Date: 2026-11-03
- Thresholds adopted exactly as proposed: Agree limit paired max z 0.50 with SE ratio in [0.80, 1.25]; Disagree limit paired max z 1.00 or SE ratio outside [0.67, 1.50]; simulator-moment rule within 4 Monte Carlo SE; interior dataset means smallest unique variance above 1e-3.
- Section 5 (R simulation step): not confirmed. If the R simulation function cannot be called as stated, the simulator-moment check is reported as NOT RUN and an Agree verdict then makes no claim about simulation. This does not block the paired primary design.
- Compute: 0.8 to 1.9 core-hours (Section 7), under the 3 h line; a run that overruns stops and re-reports.
- Changes required before launch: none
- Julia commit and Manifest hash at launch: to be recorded by the launch script
