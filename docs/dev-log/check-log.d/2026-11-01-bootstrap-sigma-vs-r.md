## 2026-11-01: bootstrap_Sigma_r vs pinned R bootstrap_Sigma, first output comparison (evidence, NOT a parity claim)

Docs-only note. It records two read-only comparison runs of `bootstrap_Sigma_r` (merged in PR #1054, origin/main a266924a1) against the pinned R `bootstrap_Sigma`. No source, test, row status, tolerance, receipt or scoreboard change. The `bootstrap_Sigma` row stays required-core, REGISTRATION-ONLY, unbound. The legacy `bootstrap_Sigma` and `confint_derived_wald.jl` are untouched. `bootstrap_Sigma_r` is still not wired into `src/GLLVModels.jl`; both runs loaded it by hand with `Base.include(GLLVModels, "src/bootstrap_sigma_r.jl")`.

Scope of this note: this is evidence from five fixtures, not a bar met. The draws cannot match bit for bit (different RNG streams, no common random numbers), so only seed-averaged summaries were compared. The 1e-6 point-estimate bar was NOT met (details below).

### Correction, 2026-11-01 (later same day): three statements below are superseded or refined

A follow-up read-only investigation (Totoro, same fixtures; its notes are not committed here) changed three statements in this note. The original text is kept and marked; nothing else here is altered. This is still evidence, not parity: the `bootstrap_Sigma` row stays unbound, and no row status, tolerance or receipt changes. The findings below come with fixes proposed in a separate PR; nothing in the code changes here.

1. **`boot_median` is not a difference.** R's `boot_median` is also empty (`list()`) for these fits. The pinned `R/bootstrap-sigma.R` fills it only for `multiple_r_*` entries (needs a multinomial trait), and the ledger spec already says "empty in v1". Julia's empty `NamedTuple()` matches R. The earlier "populated" came from a comparison script that computed its own median from the draws instead of reading R's field. Checked on R's saved result for fixture (a): `str(r$boot_median)` gave `list()`.
2. **The TwoLevel fixture B `Sigma_B[5,4]` SE gap (z = -6) is not an engine effect.** It came from a standard deviation estimated from only 5 seeds (4 df). On identical simulated data the engines agree: SE of `Sigma_B[5,4]` 0.1120 (R refit) vs 0.1122 (Julia refit) on an R-simulated batch, and 0.1190 vs 0.1183 on a Julia-simulated batch (999 draws each). Over 16 seeds the two simulators agree: 0.1214 (R-simulated) vs 0.1195 (Julia-simulated), z = -2.0, with 0 of 15 unique Sigma_B entries beyond |z| = 3 (max |z| 2.0). A true effect below about 1.5% is not excluded; 16 seeds cannot separate 1% from zero. Limits: R refits were not run on the 16 batches (Julia refit only), and only fixture B was examined.
3. **Silent Julia refit failures in GllvmFit (c) (has_diag, K=2) have a cause.** Failures of 3 to 9 of 999 are the LBFGS 500-iteration cap (inspected for seeds 101-103 only; the 7 and 7 failures of seeds 104/105 were not inspected, UNVERIFIED) on a flat Heywood ridge. Seeds 101/102/103 reproduce n_failed 5/3/9; all 17 failed refits hit `iteration_limit_reached` at 500 iterations with no exception, residual gradient 2e-3 to 3e-2 in 16 of 17 (one was 7.7e-6). A 20000-iteration reference run on the 14 failed batches of seeds 101 and 103: 12 needed 7,443 to 16,237 iterations, one (an interior optimum, b=381) needed 623, and one (b=105) stopped unconverged at 7,326; 13 of the 14 reference optima sit on a boundary (`min_t(sigma2_B + sigma2_W) = 0`, sigma_eps about 5e-5 to 6e-4). Warm start, BFGS and `g_tol = 1e-4` each failed 5 of 5 at seed 101, so it is not the start values. A tested remedy: `iterations = 2000` with `f_tol = 1e-10` gave `n_failed = 0` (was 5) in a full 999-refit bootstrap at seed 101, with bounds moving by at most 3.2e-3 (`Sigma_B` lower) and 2.2e-3 (`communality_B`); this was run on seed 101 only. Because the failures are mostly Heywood boundary draws, dropping them probably biases `communality_B` bounds slightly inward (direction inferred from R's source comment on survivor-only bounds, not measured; only the magnitude, at most 2.2e-3 when failures are kept, was measured). R has none because its `nlminb` stops on relative objective change, so the convergence definitions differ (best explanation, not proven). Side finding: in the TwoLevel fixture, Julia lands in a worse local optimum than R in about 1% of replicates (9 of 992 on the R-simulated batch, 6 of 978 on the Julia-simulated one, R logLik higher by more than 1e-3; one by 0.34 and one by 0.87), and flags them as converged. The effect on the SEs above is within noise. These are findings; fixes (refit settings, a warning when `n_failed > 0`, a `boot_median` docstring line) are proposed in a separate PR.

### Oracle and environment

Both runs on Totoro, `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, at most 4 concurrent processes, all finished. R 4.6.1, gllvmTMB 0.7.1, pin commit 9539352f66f2db2cc26b1c393e67212a359b60c9. Julia 1.12 (GllvmFit run) and 1.12.6 (TwoLevel run). Nothing was rebuilt in either R library. Runtime estimates were stated before the runs (D-287); none exceeded 3 h (TwoLevel bootstrap about 45 min wall).

| Run | R library (Totoro) | How the pin was checked |
|---|---|---|
| GllvmFit | `~/hsq_work/true-parity-campaign-20261002/Rlib` (TMB 1.9.25, built 2026-10-01) | One source hash and five function bodies were checked and all matched (`extract_Sigma` and the rest of the namespace were not compared). `R/bootstrap-sigma.R` sha256 c30fbad6...4238 matches the ledger; `bootstrap_Sigma`, `.extract_summaries`, `.na_summaries`, `.summarise_draws`, `.reconstruct_multi_formula` deparse identically to the installed namespace. |
| TwoLevel | `~/hsq_work/true-parity-prerun-20261001/Rlib` (built from `gllvmTMB-p1`) | Weaker. Only `DESCRIPTION` (ce3ba8fd...2283) and `R/extract-sigma.R` (dbd33782...0a90) hashes equal the pin file `tools/true_parity/campaign/p1_source_sha256.json`. `R/bootstrap-sigma.R` (c30fbad6...4238) is not in that file. No `.git`, so the commit is checked through two hashes only. The installed binary was NOT byte-compared with the source. |

### Alignment tables (written before any run)

GllvmFit fixtures. Both engines: tier unit (R "unit" is legacy "B"), `what` = Sigma, R, communality, ICC, cross_corr; conf 0.95; n_boot 999; draws kept so SE could be computed (neither function returns an SE).

| Quantity | R | Julia | Same estimand? |
|---|---|---|---|
| Sigma_B, has_diag FALSE | `extract_Sigma(fit,"unit","total")` = LL' | `_r_tier_total(:unit)` = LL' | Yes (point diff 1.9e-5) |
| Sigma_B, has_diag TRUE, K_W=0 | LL' + diag(sd_B^2); sigma_eps about 0 here | `sigma_y_site(fit)` = LL' + diag(sigma2_B + sigma_eps^2) | Total yes (diff 3.8e-6 / 4.2e-6); the split differs |
| R_B | `$R` of the same call | cov2cor(Sigma_B) | Yes |
| communality_B | shared / total, unit tier | diag(LL') ./ diag(Sigma_B) | Yes (fixture a: both exactly 1) |
| ICC_site | only if B and W both kept | dropped with `@info` | Both drop; message differs |
| cross_corr | NULL (needs multinomial trait) | skipped | Both omit silently |
| ci_lower / ci_upper | type-7 quantile, na.rm | type 7 on finite draws | Yes by design |
| boot_median | ~~populated~~ empty `list()` (corrected: filled only for `multiple_r_*`) | empty `NamedTuple()` | Yes (earlier "NO" superseded, see Correction) |
| Data simulation | `simulate(fit, nsim)`, RE redrawn | marginal draw N(Xbeta, sigma_y_site) | Same law, independent RNG streams |

TwoLevel fixtures (derived from `src/twolevel.jl` and the R sources). R model: `value ~ 0 + trait + latent(0+trait | ind, d=K_B) + latent(0+trait | site_species, d=K_W)`, `unit="ind"`, `unit_obs="site_species"`, Gaussian, REML=FALSE.

| Quantity | Julia `TwoLevelFit` | R | Same? |
|---|---|---|---|
| Sigma_B | `L_B L_B' + diag(sigma2_B)` | `extract_Sigma(level="unit", part="total")` | Same object |
| Sigma_W | `L_W L_W' + diag(sigma2_W)` (the only within-row variance) | `extract_Sigma(level="unit_obs", part="total")` | Same object, with the caveat below |
| Residual | none | R auto-fixes `sigma_eps` at 1e-3*sd(y) (about 1.5e-3, sigma_eps^2 about 2.4e-6) and `extract_Sigma` does NOT add it back | NOT identical. R's likelihood carries about 2.4e-6 on the W diagonal that R's reported Sigma_W omits (about 5e-6 relative) |
| communality, ICC_site | diag ratios | `extract_communality`, `extract_ICC_site` | Same (Gaussian) |
| Intervals | type-7 percentile of finite draws | type 7, na.rm | Same |

### Fixtures

- GllvmFit (p=6, n=200, one row per site-trait, trait intercepts fitted, one DGP seed each): (a) K=1 rr-only, sigma_eps 0.5; (b) K=1 plus unique (u sd 0.3-0.6, eps sd 0.3); (c) K=2 plus unique.
- TwoLevel (p=5, 60 individuals x 4 repeats = 240 rows, one data file read by both engines): A (seed 20261101) K_B=1, K_W=1; B (seed 20261102) K_B=2, K_W=1.

### Point-fit agreement

| Fixture | |dlogLik| | Max rel diff of estimates | R max abs gradient | Julia max abs gradient | Julia PD Hessian | R pdHess |
|---|---|---|---|---|---|---|
| GllvmFit (a) | 4.5e-8 | Sigma_B 3.8e-5 | 5.6e-3 | 5.5e-13 | TRUE | TRUE |
| GllvmFit (b) | 1.5e-9 | Sigma_B 3.4e-6 | 8.7e-4 | 1.5e-6 | **FALSE** | TRUE |
| GllvmFit (c) | 1.5e-8 | Sigma_B 6.2e-5 | 1.4e-3 | 2.1e-6 | **FALSE** | TRUE |
| TwoLevel A | 3.3e-8 | Sigma_B 2.8e-5, Sigma_W 4.3e-5 | 1.3e-3 | 1.6e-6 | TRUE (min eig 3.50) | TRUE |
| TwoLevel B | 6.8e-9 | Sigma_B 1.6e-5, Sigma_W 5.1e-6 | 5.2e-4 | 2.1e-6 | TRUE (min eig 0.90) | TRUE |

Reading: the logLik agrees to 4.5e-8 or better, so these are the same optimum. The Sigma entries differ by 3.4e-6 to 7e-5 relative; some derived quantities are below 1e-6 (GllvmFit (b) R_B 6.2e-7, communality_B 9.9e-7). The 1e-6 criterion is NOT met for the Sigma entries in any fixture. R stops at gradients of 5e-4 to 6e-3 (looser than Julia's). In TwoLevel a tight `nlminb` restart from R's own optimum did not move it, and adding R's fixed sigma_eps^2 back to R's Sigma_W removes only part of the gap. In the GllvmFit run R's convergence was not tightened. So looser R convergence is an inference, not proven, and the two causes were not separated. The Julia Hessian is not positive definite in the has_diag fixtures (b) and (c) (unidentified sigma_eps / sigma2_B split); Julia warns once per session at fit.

### Bootstrap agreement (n_boot 999, conf 0.95)

Method: per engine, mean over seeds of SE, 2.5% and 97.5% bound per entry; MC SE = seed sd / sqrt(seeds); z = (Julia mean - R mean) / sqrt(MCSE_J^2 + MCSE_R^2); flagged if |z| > 3. Flags are not independent (symmetric entries, shared simulated datasets).

GllvmFit (seeds per engine: a 15, b 35, c 5):
- (a) 0 of 234 comparisons flagged. Median relative difference 0.2-0.5%, max 1.2%. R_B and communality_B are degenerate (exactly +/-1 and 1, zero-width intervals) and equal in both engines.
- (b) Median relative difference 0.1-0.9% (max 1.5% for SE, 0.9% for bounds). One symmetric pair remains flagged: Sigma_B upper bound (4,2)=(2,4), z = -4.26, R mean 0.6265, Julia mean 0.6211 (0.9% relative). With 35 seeds only this pair is flagged. (The earlier 15-seed flag count quoted in the source report, 12, is unverified: that run was not retained as a table.) I cannot tell MC noise from a small real difference without more seeds.
- (c) Only 5 seeds per engine, so low power. 0 flagged, max |z| 2.95. Large max relative differences (0.2, 0.58) are on near-zero entries with |z| < 3.

TwoLevel (5 seeds per engine, 1001-1005; the SD from 5 seeds has 4 df and 65 unique correlated elements are tested per statistic, so z is a rough guide):
- Fixture A: SE and 2.5/97.5% bounds of Sigma_B, Sigma_W, communality_B, communality_W and ICC_site within about 1-2% (communality_W lower bound 7%). Beyond |z| > 3: R_B[2,1] SE z=-3.2, R_B lower z=3.4, Sigma_B upper z=4.2. ICC_site agrees for every trait (all |z| < 2.1).
- Fixture B: ICC_site, communality_W, Sigma_W and the communality_B interior agree. Two Sigma_B entries differ: Sigma_B[5,4] SE Julia .1183 vs R .1225 (about 3.5% smaller, z = -6.0), Sigma_B[5,5] SE Julia .2341 vs R .2255 (3.8% larger, z = +3.4), and derived R_B[5,4] SE z = -3.7. **Superseded (see Correction, item 2): this was Monte Carlo noise from a 5-seed SD, not an engine difference; the engines agree on identical data and over 16 seeds.** Original text follows. Cause not isolated. Candidates: R refits that fail (2-12 of 999) being dropped and possibly the extreme draws; K_B=2 sitting near a boundary (communality_B upper bound exactly 1) where two optimisers can reach different local optima; the 1e-5 point-estimate gap. Treat as an unresolved difference of a few percent in the SE of two between-individual covariance entries, not as an error in either engine. [Superseded, see above.] Also beyond 3 MC SE in B: Sigma_B lower bound (z 3.6), Sigma_B upper bound (z 3.1) and R_B lower bound (z 3.4).
- B communality_B upper bound: both engines return exactly 1.0 for four of five traits; the reported z of 10-16 is 0/0 noise, not a finding.

### Answer to the has_diag question

Question (the ledger's UNVERIFIED flag): for `has_diag` with `K_W == 0`, is Julia's tier-B total `sigma_y_site(fit)` the same quantity as R's unit-tier Sigma? **YES, for the data shape tested.** Julia `sigma_y_site` equals R `Sigma_B` to max abs 3.83e-6 (b) and 4.16e-6 (c); the diagonals agree to 4 d.p. (b: R and Julia both [1.2243, 0.9926, 0.9053, 0.5301, 0.5684, 0.3978]).

Reason found, and the caveat that goes with it: with one row per (site, trait), R drives sigma_eps to about 1e-3 (sigma_eps^2 = 7.7e-7 in b, 1.1e-6 in c) and puts the whole diagonal into the unit unique part sd_B^2. So R's Sigma_B = LL' + diag(sd_B^2) equals the identified total. Julia's split between sigma2_B and sigma_eps^2 is different and unidentified (PD Hessian FALSE); R's sd_B^2 is NOT Julia's sigma2_B (b: R sd_B^2 = [0.378, 0.237, 0.403, 0.199, 0.257, 0.231]; Julia sigma2_B alone = [0.107, 0.036, 0.119, 0.017, 0.046, 0.033]). Only the total matches. This was observed for one-row-per-cell data only; R's behaviour with replicated rows per site was not tested. The ledger flag is resolved YES for these two fixtures only; I did not edit the ledger.

### Behaviour differences found

1. Failed refits, GllvmFit (c): Julia 5, 3, 9, 7, 7 of 999 (0.3-0.9%) across 5 seeds; R 0 in all five. Julia fixtures (a) and (b): 0 failures in both engines. The Julia failures are non-converged refits on the has_diag sigma_eps/sigma2_B ridge (cause since identified: LBFGS 500-iteration cap, see Correction, item 3). Nothing warned (the attrition warning fires only below 80% retained); bounds use survivors; the effect on bounds was within MC error at 5 seeds. TwoLevel is the reverse: Julia 0 failures in all 10 runs (5 seeds x fixtures A and B), R failed 2-15 per 999 (0.2-1.5%), also silently.
2. ~~`boot_median`: R returns it; Julia returns an empty `NamedTuple()`.~~ **Withdrawn (see Correction, item 1): R's `boot_median` is also empty for these fits (filled only for `multiple_r_*`); Julia matches.**
3. Messages: R repeats the "Sigma_unit is latent-only" note once per replicate for the unique=FALSE fit (20 copies at n_boot=20, so 999 at 999); Julia emits one `@info` about communality being exactly 1. R's default-level call emits two deprecation warnings ("B", "W"); Julia emits an `@info` that W and phy are dropped, R drops them silently. Both drop ICC with a message and both warn that n_boot < 39 cannot give 95% coverage (coverage_ceiling equal: 0.9048 at B=20, 0.998 at B=999).
4. R fails every refit when the within-unit column has a custom name. `bootstrap_Sigma` does not forward `unit_obs` to its refit, so with `unit_obs="obs"` all refits error inside `tryCatch` and return NA (n_failed 9/9, only the attrition warning "kept 0%"; error "Unsupported grouping obs"). Workaround for all R runs: name the column `site_species`. Julia has no counterpart but needs `individual=` passed again (the fit stores only `nindiv`).
5. Counterpart caveat (TwoLevel): R fixes sigma_eps at about 1.5e-3 and omits it from the reported Sigma_W, so R's likelihood carries about 2.4e-6 that its Sigma_W does not.
6. Julia needs `X=` (or y/n_sites) for a model with fixed effects; without it the fit is zero-mean. R needs nothing.
7. Output structure: R `level` is a character vector and `what` a character vector; Julia `level = [:B]`, `what = Symbol[]`, and Julia returns a `BootstrapSigmaR` struct of NamedTuples. Field names otherwise match. `n_effective` covers vector entries only in both.
8. Speed (not a claim, just observed): Julia about 2x (c) to 25x (a) faster per 999 refits in GllvmFit; 5-8x (40-93 s vs 325-465 s) in TwoLevel.

### What this does NOT establish

- Not a parity claim. The bootstrap_Sigma row stays unbound; the 1e-6 point-estimate bar was missed for the Sigma entries in all five fixtures.
- Only Gaussian data; GllvmFit: p=6, n=200, one row per site-trait, K <= 2, one DGP seed per fixture; TwoLevel: p=5, 60 x 4 balanced, two fixtures. Seeds: 5-35 per engine, so z is a rough guide and the (c) fixture is low power.
- Not covered: replicated rows per site in GllvmFit, unbalanced TwoLevel designs, phy, the PrecisionMultivariateFit and JointPhyloGroupedGaussianFit routes, non-Gaussian families, W-tier and ICC for GllvmFit (not reportable), tightened R convergence in GllvmFit.
- The TwoLevel oracle is pinned by two file hashes only (see the table above).
- Raw scripts, data and logs live on Totoro (`~/claude-bsig-vs-r/gllvmfit`, `~/claude-bsig-vs-r/twolevel`) and in the session scratchpad; they are not committed here.

### Open follow-ups

1. ~~Isolate the TwoLevel fixture B Sigma_B[5,4] / [5,5] SE difference.~~ Withdrawn: explained as 5-seed noise (Correction, item 2). Process lesson: compare bootstrap SEs on common simulated batches, or with at least 15 seeds per side using the SD of the per-seed SEs.
2. Resolve the GllvmFit (b) Sigma_B upper-bound pair (4,2) with more seeds (z = -4.26 at 35 seeds).
3. Raise GllvmFit (c) beyond 5 seeds; the cause of the 3-9 of 999 silent Julia refit failures is now known (Correction, item 3); what remains is the decision on the fix (refit iterations and `f_tol`, a warning whenever `n_failed > 0`, or counting only throws/non-finite as failures, which changes the ledger failure definition), and a look at the roughly 1% worse-local-optimum replicates.
4. Tighten R convergence (and add back R's fixed sigma_eps^2) to separate the sources of the 3.4e-6 to 7e-5 Sigma-entry point-estimate gap.
5. Test the has_diag total equality with replicated rows per site, where R may not push sigma_eps to about 0.
6. (`boot_median` question withdrawn: Julia matches R, see Correction, item 1; at most a docstring line saying it is empty by design.) Decide whether R's per-replicate note and the `unit_obs` forwarding failure are to be recorded as R-side defects or matched.
7. Byte-verify the TwoLevel R library against the pin (or rebuild from a verified source) so its oracle matches the strength of the GllvmFit one.
8. A signed estimand and bound row would still be needed before any of this counts toward the row; this note changes nothing there.
