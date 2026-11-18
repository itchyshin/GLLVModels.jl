# N1b result: paired R-versus-Julia bootstrap comparison, phy tier

Evidence note only. No row status, scoreboard, receipt or tolerance is changed. Registration: `docs/dev-log/plans/2026-11-03-bootstrap-sigma-phy-n1b-prereg-DRAFT.md` (approved by delegation, its Section 10). Thresholds were applied exactly as written there and were not changed after the data were seen. Background: `docs/dev-log/check-log.d/2026-11-03-bootstrap-sigma-phy-n1-diagnosis.md`.

Date 2026-11-03. Run by a Claude session on Totoro. PR #990, `src/confint_derived_wald.jl`, Codex branches and other lanes were not touched. R gllvmTMB was used read-only.

## Verdict

Per dataset (Section 6 rules):

| Dataset | Role | Verdict | Reason |
|---|---|---|---|
| 1101 | boundary | Inconclusive | z and ratio rules pass; the absolute communality rule fails (max abs endpoint difference 0.088 against the 0.05 limit) |
| 1104 | boundary | Agree | all rules pass |
| 1205 | interior (replaces 1201) | Agree | all rules pass |
| 1206 | interior (replaces 1202) | Agree | all rules pass |

Overall verdict: **Inconclusive** (the rule is Agree only if all four Agree; no dataset Disagreed).

The Inconclusive on 1101 comes from one pre-registered absolute rule, not from the interval-width z or the SE ratio. The paired max z on 1101 (0.339) is far below the Disagree limit of 1.00 and below the Agree limit of 0.50. I report that plainly and do not re-read the rule. Details in "Dataset 1101" below.

## What the verdict does not claim

- It is not a coverage, recovery or general parity claim, and it does not move any row status.
- Three Agree verdicts cover three datasets (one boundary, two interior), one tree size (30 tips), rank 1, d = 5, explicit-unique shared residual, no REML, cold-start refits, one R and one Julia version.
- The comparison is of the refit and the summary extraction given the same simulated Ys. The simulator is a separate check (Section 5 below); it passed, but see its limits.
- The two engines end at different local optima in 1% to 6% of replicates (below). The intervals still agree, but the replicate-level results do not.
- Truth is not part of the agree rule. Truth containment is descriptive, and four datasets cannot estimate coverage.
- Nothing here says the Julia and R point estimates or bootstrap draws are interchangeable replicate by replicate.

## Datasets, eligibility and the replacement of 1201 and 1202

The data-generating process is `sim.jl` of N1 (30 tips, d = 5, two replicates per species, 300 rows). Regenerating seed 1101 with the copied `sim.jl` reproduced the N1 `Y.csv` and `tree.nwk` byte for byte.

The original fit was made in both engines for every seed screened. Eligibility (both converged, Sigma_phy point estimates within 1e-5 elementwise) held for all ten seeds screened; the largest elementwise Sigma_phy difference was 8.8e-07. Interior means the smallest unique variance exceeds 1e-3 in both engines.

| Seed | Smallest unique variance, Julia / R | Status |
|---|---|---|
| 1101 | 1.4e-10 / 1.2e-08 | boundary, kept (pre-registered) |
| 1104 | 1.6e-10 / 6.5e-09 | boundary, kept (pre-registered) |
| 1201 | 7.5e-11 / 6.0e-09 | boundary, replaced |
| 1202 | 4.5e-10 / 8.4e-08 | boundary, replaced |
| 1203 | 6.3e-10 / 7.7e-09 | boundary, skipped |
| 1204 | 1.4e-10 / 2.4e-08 | boundary, skipped |
| 1205 | 6.75e-02 / 6.75e-02 | interior, used for the 1201 slot |
| 1206 | 1.02e-01 / 1.02e-01 | interior, used for the 1202 slot |
| 1207 | 1.75e-01 / 1.75e-01 | interior, screened only, not run |
| 1208 | 2.01e-02 / 2.01e-02 | interior, screened only, not run |

Both new target seeds (1201, 1202) were boundary fits, so the stated replacement rule applied. I screened 1203 to 1208 in one batch, then assigned the first interior seed in the series (1205) to the 1201 slot and the next unused interior seed (1206) to the 1202 slot. Seeds 1207 and 1208 were screened for the original fit only and were not run in the comparison, so no choice among interior seeds was made after seeing comparison results. Original-fit screening used no bootstrap data.

## Pinned versions and provenance

- R: gllvmTMB 0.7.1 from the Totoro oracle library `/home/snakagaw/hsq_work/gllvm-w2-3/oracle-build/library`, built from reference commit `9539352f66f2db2cc26b1c393e67212a359b60c9` (`build.json`), R 4.6.1. `verify_noversion.py` could not be found on Totoro or in this repo, so it was not re-run today. I checked the installed version string and `build.json` instead; the N1 log `CORE070_ORACLE_VERIFY_PASS` is from the N1 date, not today.
- Julia: GLLVModels.jl at commit `85e47a2a9fa3f7882cb0ae420c13f810a79eaa23` (origin/main when the Totoro tree was copied for the diagnosis; `git diff 85e47a2a9 HEAD -- src` on the n1b branch is empty, so `src/` is identical). Julia 1.12.6. `Manifest.toml` is not tracked at that commit; the frozen Totoro manifest has sha256 `ba1f3285b0191304b17a003cb8baf07024b566c27c55caac6bb8913a3fea4c40`, the same hash as N1. `src/bootstrap_sigma_r.jl` sha256 `27b2d6128f156ea3be98087839aaa58de93337fe070a83d3accd3f59336e5668` on Totoro and on the branch.
- The Totoro Julia tree used is `~/hsq_work/claude-n1-20261103/src`.

## Design as run

For each dataset: original fit in both engines (R with the N1 BFGS control, `reltol = 1e-14`; Julia default); one batch of 999 Ys simulated from the Julia fit with `_bsig_pmv_simulate_batch(MersenneTwister(2001), fit, 999)`; every Y refit cold in both engines.

- R refit: `gllvmTMBcontrol(se = FALSE)`, default nlminb, same formula as the original fit. A replicate failed if `convergence != 0` or extraction failed.
- Julia refit: `fit_precision_multivariate(...; iterations = 3000)` with the Optim `g_tol` left at the default 1e-5. A replicate counted as successful if the final objective was finite and the finite-difference gradient max-norm was at most 1e-3 (the diagnosis's "g_tol = 1e-3 check", applied after the fit). This is the reading of the registration's wrapper that I used; the registration does not say whether 1e-3 goes to the optimiser. I did not call the driver `bootstrap_Sigma_r`; I looped over the same simulated Ys and used the same extraction (`_bsig_entries`).
- Summaries: 15 Sigma_phy entries and 5 communalities, two endpoints each (40 comparisons). Endpoints are type-7 percentiles (numpy `linear`) of each engine's finite draws, computed outside both packages; SD is the SD of finite draws. No cell had SE_pool below 1e-3, so the absolute small-cell rule never replaced a z comparison.
- The registration also asked for the packages' own bounds to be checked against the recomputation. NOT RUN: the paired design does not use either package's interval function, and the endpoints come only from the recomputation.

## Results per dataset

| Dataset | Paired max z (own survivors) | At | Max z, common survivors (n) | SE ratio SD_J/SD_R, 20 cells | Finite draws, min over cells (R / Julia) |
|---|---|---|---|---|---|
| 1101 | 0.339 | comm5 upper | 0.339 (999) | 0.954 to 1.082 | 999 / 999 |
| 1104 | 0.029 | Sigma54 upper | 0.031 (996) | 0.989 to 1.004 | 998 / 997 |
| 1205 | 0.107 | comm2 lower | 0.107 (999) | 0.999 to 1.024 | 999 / 999 |
| 1206 | 0.232 | comm5 upper | 0.234 (995) | 0.995 to 1.031 | 996 / 998 |

Absolute rules (communality `|dEnd| <= 0.05` at every endpoint; small-cell rule not triggered):

| Dataset | Max communality abs difference | Rule |
|---|---|---|
| 1101 | 0.0881 (comm4 lower: R 0.138, Julia 0.050); 0.0603 (comm5 upper: R 0.706, Julia 0.646) | FAIL |
| 1104 | 0.0020 | pass |
| 1205 | 0.0232 | pass |
| 1206 | 0.0439 | pass |

Rule 1 (fewer than 950 finite draws in a cell) did not trigger anywhere. No dataset exceeded the Disagree limits (max z above 1.00, or SE ratio outside 0.67 to 1.50).

### Dataset 1101

Both endpoint failures are communality endpoints, where the draws pile up against 1 (upper endpoints equal 1.0 in four of five traits) and the SD of a communality is 0.17 to 0.29, so 0.088 is 0.33 of an SD (the z rule gives 0.339 there, which passes). The failing differences sit at comm4 lower and comm5 upper. The same dataset was the noisiest in the diagnosis pilot (paired max z 0.323). The rule as written is an absolute 0.05 bound, and it is exceeded, so the dataset is Inconclusive. I did not examine whether a different bound would be better; the thresholds stay fixed.

## Pre-registered descriptive reports (not used in the verdict)

Replicates whose two logLik values differ by more than 1e-4 (all 999 replicates have both logLik values; Julia logLik is that of the refit even if it failed the wrapper rule):

| Dataset | Count differing | Julia higher | R higher | Largest absolute logLik difference |
|---|---|---|---|---|
| 1101 | 41 | 7 | 34 | 13.77 |
| 1104 | 60 | 25 | 35 | 1.26 |
| 1205 | 10 | 2 | 8 | 4.75 |
| 1206 | 25 | 14 | 11 | 2.40 |

That is 1.0% to 6.0% of replicates, in line with the diagnosis (1% to 6%). It is not fitted or interpreted further. The 13.77 logLik gap on 1101 is much larger than the 4 units seen in the pilot; I did not look into which replicate or why.

Per replicate, the largest paired `|R - J| / SD` (over common survivors): the median over replicates is 2.2e-06 to 3.0e-06 in all four datasets. Replicates above 0.05 SD, and the largest value:

| Dataset | Above 0.05 SD | Largest paired difference (SD units) |
|---|---|---|
| 1101 | 41 | 5.58 |
| 1104 | 52 | 4.62 |
| 1205 | 9 | 4.54 |
| 1206 | 23 | 5.25 |

Failed replicates (rules differ, so the gap is descriptive only):

| Dataset | R failed (code != 0) | Julia failed (wrapper rule) | Julia would fail under the driver tolerance 1e-5 at 3000 iterations |
|---|---|---|---|
| 1101 | 0 (0.0%) | 0 (0.0%) | 5 |
| 1104 | 1 (0.1%) | 2 (0.2%) | 8 |
| 1205 | 0 (0.0%) | 0 (0.0%) | 6 |
| 1206 | 3 (0.3%) | 1 (0.1%) | 14 |

No Julia refit threw. The last column is the count with `converged == false` in the run, that is Optim convergence plus gradient at most 1e-5, with iterations capped at 3000 (not the default 400), so it is not the N1 driver count.

Truth containment of each engine's 95% interval (20 cells; truth Sigma_phy = lambda lambda' + diag(s), communality lambda^2 / (lambda^2 + s) from `sim.jl`):

| Dataset | R | Julia | Cells missed (identical for both engines unless noted) |
|---|---|---|---|
| 1101 | 16/20 | 16/20 | Sigma51, Sigma52, Sigma53, Sigma54 |
| 1104 | 14/20 | 14/20 | Sigma31, Sigma32, Sigma33, Sigma43, Sigma53, comm1 |
| 1205 | 19/20 | 19/20 | Sigma44 |
| 1206 | 19/20 | 18/20 | Sigma33 (both); comm2 (Julia only) |

The two engines miss the same cells nearly everywhere, which is consistent with the finding that the intervals agree. The low containment on 1101 and 1104 is a feature of one realisation of a boundary dataset with n = 30 tips and says nothing about coverage.

R_phy (10 off-diagonal entries, same paired rule, descriptive): max z 0.373 (1101), 0.020 (1104), 0.043 (1205), 0.053 (1206); SE ratio ranges 0.982 to 1.050, 0.998 to 1.005, 0.999 to 1.025, 0.992 to 1.027.

## Section 5: simulator-moment check

Run. The R simulation step in `bootstrap_Sigma` is `simulate(fit, nsim = n_boot)` (line 69 of the pinned function body), unconditional, so the same call was made on the original R fit, 3000 draws, seed 424242. The Julia batch simulator produced 3000 draws from the Julia fit (`MersenneTwister(31337)`). For each draw and each simulator, centred on the fitted mean, I computed 15 within-observation trait covariances and 15 same-species cross-replicate trait covariances (the species-level covariance across traits). The theory is the Julia fit's Sigma_phy plus residual variance (within) and Sigma_phy (cross); the R and Julia original fits agree to 7e-07 in Sigma_phy, so one theory serves both. Monte Carlo SE is the SD of the per-draw statistic divided by the square root of 3000.

Rule (proposed in the registration, 4 Monte Carlo SE for every entry): **passed for both simulators on all four datasets.** Largest absolute standardised deviation: 1101 Julia 1.91, R 2.09; 1104 Julia 1.85, R 1.61; 1205 Julia 2.15, R 1.22; 1206 Julia 1.05, R 2.15. No entry exceeded 4.

Limits: the check covers trait covariances within a row and within a species, not the covariance between different species implied by the tree (the A matrix), and it uses only 30 entries per simulator. It also does not enter the verdict. It supports saying the Julia and R simulators agree with the fitted covariances in these first and second moments, and nothing more. The Section 5 rule was proposed, not simulated, so its power is unknown.

## Compute used

Estimated before the run: 0.8 to 1.9 core-hours (registration Section 7). Actual about 0.9 core-hours, about 15 minutes wall. Julia refit batches 326.5 s (1101), 739.4 s (1104), 345.7 s (1205) and 522.3 s (1206), 0.54 core-hours; R refit batches about 85 to 110 s each (about 0.1 core-hours); original-fit screening of ten seeds in both engines about 10 core-minutes; Section 5 about 0.1 core-hours. All single-thread (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1`), `taskset` on four named cores, `nice`, at most four of my processes at once. Load on Totoro was shared; uptime was checked first (load average 7.6 to 51). Far under the 3 h line, so no D-287 stop applied. The R refits ran much faster than the estimate assumed.

Raw outputs, per-replicate refit results and logs: `~/hsq_work/claude-n1b-20261103/` on Totoro (`out/p<ds>/` for allY, `jdraws`, `jinfo`, `pairedR`, `pairedR_info`; `out/o<ds>/` original fits; `out/s5/` simulator moments; `logs/`). Small scripts and the machine-readable analysis are in `docs/dev-log/check-log.d/2026-11-03-bootstrap-sigma-phy-n1b-scripts/` (`analyze.py` produces `analysis.json`; `s5_and_comm_output.txt` holds the Section 5 and communality printouts).

## Things worth knowing

- Neither new target seed (1201, 1202) was interior; both needed the replacement rule.
- On 1101 the one failing rule is the absolute communality bound. The z and ratio rules were comfortable on every dataset.
- The 1104 and 1206 batches show a few R failures (1 and 3) that Julia did not have, and Julia failures (2 and 1) that R did not, which reflects the different rules and different local optima rather than a pattern.
- The wrapper reading (gradient check applied after the fit, Optim `g_tol` unchanged) is my interpretation of an ambiguous sentence in the registration. With 0 to 2 Julia failures per dataset it is unlikely to matter, but it is not the only possible reading.
- The cost was lower than estimated on both sides, so a larger paired design or a repeat on additional seeds would be cheap if wanted.
