# N1 pre-registration, DRAFT: R-versus-Julia bootstrap comparison, phy tier

Status: UNSIGNED DRAFT. This document has not been approved. The full run (Section 7) must not start until Shinichi has read it, changed what he wants, and signed Section 10. Nothing here changes a row status, a scoreboard, a receipt or a tolerance. Only the pilot in Section 8 has been run.

Date drafted: 2026-11-02. Drafted by a Claude session; not yet reviewed by anyone.

## 1. Question

Does `bootstrap_Sigma_r` on a `PrecisionMultivariateFit` (Julia, `level = :phy`) give the same bootstrap distribution as `bootstrap_Sigma` in R gllvmTMB 0.7.1 for the same data and the same model?

The object compared is the bootstrap distribution of three phy-tier summaries. It is not a coverage study and not a recovery study. Truth is known (Section 2) but is not used in the agree/disagree rule.

Starting evidence (point estimates only, `docs/dev-log/check-log.d/2026-11-02-bootstrap-sigma-phy-fuzz-and-r-point.md`): on two datasets the Julia and R point estimates of Sigma_phy agree within 1.8e-6, communality_phy within 2.5e-6. That says the two engines fit the same model. It says nothing about the spread of refits, which depends on how each side simulates and refits.

## 2. Pinned versions

- R: gllvmTMB 0.7.1 at commit `9539352f66f2db2cc26b1c393e67212a359b60c9`, library `/home/snakagaw/hsq_work/gllvm-w2-3/oracle-build/library` on Totoro.
- Oracle verification: the stock `core070_build_oracle.py verify` fails on this build because the receipt has no `gllvmTMB_version` key (PR-F1 in the check-log). Use the same workaround as that note: `verify_noversion.py` (the verify with the two version checks removed), kept at `~/hsq_work/claude-phy-20261102/fz/r-point/tools/` on Totoro. The run is valid only if that check passes on the day, and its output is stored with the results.
- Julia: GLLVModels.jl at the exact commit recorded at launch (to be filled in at signing, Section 10), and the Julia version and Manifest recorded with the results. The earlier note resolved dependencies fresh, so versions must be frozen this time.

## 3. Data-generating process

Same DGP as the point-estimate check (`sim.jl` in `~/hsq_work/claude-phy-20261102/fz/r-point/`), so the starting evidence and this study describe the same model.

- Tree: random ultrametric, height 1, 30 tips (`sp01` to `sp30`), a new tree per dataset.
- Traits d = 5, 2 replicates per species, 300 rows per dataset. No missing cells.
- Model: y[t, o] = beta[t] + lam[t] g[sp] + u[t, sp] + e, with g ~ N(0, A) (rank 1 phylogenetic factor), u[t, .] ~ N(0, s[t] A) (phylogenetic unique), e ~ N(0, 0.3^2) shared residual, A the tip covariance of the tree.
- Truth: lam = (0.9, 0.6, -0.5, 0.7, 0.4); s = (0.50, 0.40, 0.60, 0.30, 0.50); beta = (0.4, -0.2, 0.3, 0.0, 0.1); residual SD 0.3.
- Datasets: K = 4, generated with `sim.jl` seeds 1101, 1102, 1103, 1104 (arguments ntip 30, reps 2). Both engines fit the same files (`long.csv`, `Y.csv`, `tree.nwk`). Tip-to-index mapping uses Newick leaf order, as in the earlier check (PR-F3).
- Dataset eligibility, fixed before looking at results: a dataset enters the primary analysis only if both engines converge on the original fit (R `opt$convergence == 0`; Julia `converged == true`) and the two point estimates of Sigma_phy agree within 1e-5 elementwise. A dataset that fails this is replaced by the next seed (1105, 1106, ...) and the replacement is reported. Boundary fits (a unique variance near 0) are not excluded, but are flagged, because the earlier note showed non-PD Hessians there (PR-F4).

Seeds are listed so the run can be reproduced; they were chosen as consecutive integers and not screened.

## 4. Model fitted and bootstrap settings

R: `value ~ 0 + trait + phylo_latent(species, d = 1, tree = tree) + phylo_unique(species, tree = tree)`, Gaussian, `REML = FALSE`, `engine = "tmb"`, BFGS control as in `fit_r.R` (`maxit` 2000, `reltol` 1e-14, `n_init` 1). Then `bootstrap_Sigma(fit, n_boot = 999, level = "phy", what = c("Sigma", "R", "communality"), conf = 0.95, seed = S, n_cores = 1, progress = FALSE, keep_draws = TRUE)`.

Julia: `fit_precision_multivariate(Y, pp; rank = 1, mode = :explicitunique, residual_mode = :shared, species_id)`. Then `bootstrap_Sigma_r(fit; n_boot = 999, level = :phy, what = [:Sigma, :R, :communality], conf = 0.95, seed = S, progress = false, keep_draws = true)`.

Reminders about what differs by construction (not defects to be hidden): Julia's refit convergence also requires a finite-difference gradient norm below `g_tol`; R's rule is the optimiser convergence code only. The Julia `explicitunique` refit uses 2000 iterations and `f_tol` 1e-12. The two sides cannot share a random stream, so draws are not paired.

Bootstrap seeds: S = 2001 (primary) and S = 2002 (repeat), on each side and each dataset. The same integer on both sides is a label only.

n_boot = 999 on every run. conf = 0.95, percentile method, as both functions implement.

## 5. Summaries compared

Per dataset, per engine, from the stored draws:

- Sigma_phy: all 15 unique entries (5 diagonal, 10 off-diagonal).
- communality_phy: 5 entries.
- R_phy: stored and reported descriptively, not part of the agree rule (it is a function of Sigma_phy, so it adds correlated cells, not information).

For each of the 20 cells the following are computed on each side: lower endpoint (2.5th percentile), upper endpoint (97.5th percentile), SE of the draws (SD of the finite draws), number of finite draws. Both sides use the same quantile type (type 7) applied to the stored draws, recomputed outside both packages, so a difference in quantile code cannot enter. The packages' own reported bounds are also checked against the recomputation and any mismatch is reported.

## 6. Decision rules

Notation. For a cell, SE_pool = sqrt((SE_R^2 + SE_J^2) / 2). dEnd = endpoint_J minus endpoint_R. All rules apply to the primary seed (2001). The repeat seed (2002) gives the noise floor.

Failure handling, applied to each side with that side's own rule. A replicate is failed if it is in the package's `n_failed`, or if its draws are not finite. Failed replicates are excluded from that side's percentiles (this is what each package does), and the failure count and rate are reported per side. They are not hidden, matched or imputed. Draws that are finite but come from a non-PD Hessian are kept on both sides (both packages keep them); the count is reported when available.

Noise floor. Compare seed 2001 against seed 2002 within each engine. Let m_R and m_J be the largest |dEnd| / SE_pool across all cells and endpoints for each engine. This is Monte Carlo noise at n_boot = 999 only.

Per dataset verdict, in this order:

1. Inconclusive for the dataset if any of: (a) either side has more than 5% failed replicates on the primary seed; (b) the failure rates differ by more than 2.5 percentage points; (c) the within-engine noise floor m_R or m_J exceeds 0.35 (the study is too noisy to separate engines at this n_boot); (d) fewer than 950 finite draws in any cell on either side.
2. Otherwise Agree if all of the following hold across the 20 cells:
   - Endpoints: |dEnd| / SE_pool <= 0.25 for at least 95% of the 40 endpoint comparisons, and <= 0.50 for all 40. For a cell with SE_pool < 1e-3 (a boundary-pinned cell), the absolute rule |dEnd| <= 0.01 is used instead.
   - Spread: SE_J / SE_R in [0.90, 1.10] for at least 95% of cells, and in [0.80, 1.25] for all 20.
   - For communality_phy, which piles up at 0 and 1, the absolute difference |dEnd| <= 0.05 must also hold for every endpoint.
3. Otherwise Disagree for the dataset if any cell has |dEnd| / SE_pool > 0.50 (outside the SE_pool < 1e-3 rule) or SE ratio outside [0.80, 1.25] (not caused by a noise floor failure, which is rule 1c). Cells between the Agree and Disagree bands, with the 95% conditions failing, give Inconclusive.

Rationale for the numbers (an assumption, open to change at signing). At n_boot = 999 the Monte Carlo SE of a 2.5% percentile is about 0.08 draw-SD under normality, so the difference of two independent endpoints has SD about 0.12 draw-SD. A 0.25 bar is about 2 of those SDs; 0.50 is about 4, a reasonable maximum over 40 correlated comparisons. The SE ratio bands follow from the SE of an SD estimate from 999 draws (about 2.2%, so about 3% for a ratio). Heavy tails and boundary pile-up make these only approximate, which is why the noise-floor check (1c) is required.

Overall verdict across the K = 4 datasets:

- Agree: all four datasets Agree.
- Disagree: at least one dataset Disagree (no overruling by the others; the cell and dataset are named).
- Inconclusive: otherwise (including any dataset Inconclusive with none Disagree).

An Agree verdict would be recorded as evidence about four datasets, one tree size, one rank, d = 5, and the explicit-unique shared-residual configuration only. It would not be a coverage, recovery or general parity claim, and it would not by itself move any row status; that is a separate decision.

Secondary, descriptive only: truth recovery of the bootstrap intervals (whether the 95% interval contains the true Sigma_phy and communality), tabulated per engine and not used in the verdict; four datasets cannot estimate coverage.

Optional paired follow-up, not part of this registration: generate the bootstrap datasets once in Julia and refit each in both engines, which removes Monte Carlo noise from the endpoint comparison. It would need its own approval.

## 7. Full run (NOT launched; requires approval)

4 datasets x 2 engines x 2 seeds = 16 bootstrap runs of 999 refits each (about 16,000 refits), plus 8 original fits.

Resource plan: Totoro, one thread per process (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`), at most 2 cores at a time, under the standing Totoro cap. No GitHub Actions. Raw outputs go to `~/hsq_work/claude-n1-20261102/` on Totoro (not `/scratch`).

## 8. Pilot (done) and compute estimate

Pilot, R side only. Dataset: the earlier interior-solution dataset B6 (30 tips, d = 5, 2 replicates; its generating seed was not recorded in the check-log, so it is not one of the K = 4 datasets). Call: `bootstrap_Sigma(fit, n_boot = 20, level = "phy", what = c("Sigma", "R", "communality"), conf = 0.95, seed = 101, n_cores = 1, progress = FALSE, keep_draws = TRUE)`, same formula and BFGS control as Section 4. Run under `taskset -c 0,1`, `nice`, one thread, Totoro load average about 286 at launch. Script and log: `~/hsq_work/claude-n1-20261102/pilot.R`, `pilot.log`, `pilot_bs.rds`.

Measured:

- Original R fit: 1.77 s.
- 20 replicates: 6.17 s in total, 0.308 s per refit. 0 failed (`n_failed = 0`).
- One R warning only: n_boot = 20 cannot reach 95% coverage (ceiling 0.905), as expected.
- The function runs on the oracle library 0.7.1 and returns the object with fields `point_est`, `ci_lower`, `ci_upper`, `n_effective`, `boot_median` (empty), `ci_method` ("percentile"), `link_residual` ("auto"), `conf`, `n_boot`, `coverage_ceiling`, `n_failed`, `level`, `what`, `draws`.
- Level and summary names for phy: `level = "phy"`; entries `Sigma_phy` (5 x 5 matrix, dimnames t1 to t5), `R_phy` (5 x 5), `communality_phy` (named vector t1 to t5). `n_effective` has only `communality_phy` (a vector of 20, one per trait). `keep_draws = TRUE` gives a list of n_boot draws, each a list of the same three entries. These match the names the Julia function documents (`Sigma_phy`, `R_phy`, `communality_phy`).
- With 20 draws, communality_phy intervals already run to the boundaries (for example trait t3: 8e-6 to 0.998, trait t1 upper bound 1.0), which is why Section 6 adds an absolute rule for communality.

Estimate for the R side of the full run: 0.308 s x 999 = about 308 s per run, so 8 R runs take about 41 min (at the same load; the earlier note saw R fit time vary about 14-fold with load, so a range of 0.7 to 3 h cannot be excluded).

Estimate for the Julia side: NOT grounded. No Julia bootstrap timing was measured in this pilot, and the check-log reports completed 199-replicate runs without timings. The Julia total therefore cannot be stated. Before signing, a Julia pilot of the same shape (n_boot = 20, B6 or a new dataset) is needed to turn this into a number; for scale, 8 Julia runs x 999 refits x t seconds is 2.2 h at t = 1 s and 6.7 h at t = 3 s, so the 3 h line is crossed at about t = 1.3 s. Under the repository's estimate-before-run rule, if the Julia pilot puts the total above 3 h, this plan goes to Shinichi with that number and a pre-run result before anything is committed. A run that overruns its estimate stops and reports.

## 9. Known limits of the design

- Four datasets, one tree size, rank 1, d = 5, shared residual, no missing cells, no REML. A different configuration is a different study.
- n_boot = 999 gives Monte Carlo noise comparable in size to the disagreement the thresholds are meant to catch; the repeat-seed floor is the safeguard.
- Julia's refit rule is stricter than R's; failure rates are not like for like (Section 4). The comparison of intervals uses each side's survivors.
- Thresholds in Section 6 are proposed by the drafter without prior simulation of their operating characteristics. They are a proposal for Shinichi to change.
- The Julia fitter emitted thousands of per-likelihood warnings in earlier tests (S1-F1); the run must capture them to a file, not the terminal.
- R `nlminb` can return code 1 on interior optima that match (PR-F5); this design uses BFGS on the R side to avoid that, which is a choice, not a neutral default.

## 10. Approval (unsigned)

This registration is a DRAFT and is UNSIGNED. The full run needs Shinichi's approval after he has read Sections 6 to 8. No signature, date or approval is recorded here.

- Approved by: ______ (not filled)
- Date: ______ (not filled)
- Changes required before launch: ______ (not filled)
- Julia commit and Manifest hash at launch: ______ (not filled)
