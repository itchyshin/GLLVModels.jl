# Pre-run results: true-parity campaign plan, section 4.3

Status: RESULTS of the pre-run only. The full campaign is not approved and was not started. No case-map row, scoreboard, checker or test file was touched. Plan: `campaign-plan-c3-c5-2026-10-01.md` on this branch (PR #650).

Run on Totoro on 2026-09-30 (Mac clock), 20:36 to 21:08 local. Julia side: GLLVModels.jl at `origin/main` 9ad7b0036405c5ddfda8a9e9bfce35da39bd0ccd, Julia 1.12.6 (the Mac pre-run used 1.10). R side: gllvmTMB 0.7.1 built from a clean `git archive` of 9539352f66f2db2cc26b1c393e67212a359b60c9 into a lane-local library, R 4.6.1, TMB 1.9.21, gllvm 2.0.13 (for the two real datasets).

## 1. Verdict

NO-GO for the full campaign as planned. The plan's rule (section 4.3, item 3) needs every cell to finish inside its cap and 3 times its point estimate. Three of the six cells did not finish on the Julia side inside 15 minutes: C3 temporal, C4 beetle, C4 fungi subsample. The other three (NB2, ordinal, iSDM) and the Gaussian smoke test passed, all inside |ΔlogLik| 1e-6.

By the plan's rule, the three cells that hit the cap are reported as findings and the rows are listed as "not feasible at this size", with no retry at a larger cap. I did run a set of smaller diagnostic fits (section 4) to say why, and they are labelled as diagnostics, not cells. Section 6 gives a revised estimate and options for you to choose from.

## 2. Per-cell results (the pre-run proper)

Size for every C3 cell: n = 500, p = 20, K = 2 unless the row says otherwise. "Wall" is one process, one thread (`OPENBLAS_NUM_THREADS=1`, `JULIA_NUM_THREADS=2`), with a 900 s cap per process. Julia fit times include first-call compile time (the harness times the first fit; the Mac pre-run in the plan excluded a warm-up).

| Cell | What ran | R fit wall (s) | Julia fit wall (s) | R conv / pdHess | Julia conv / pd_hessian | R logLik | Julia logLik | abs diff | Result |
|---|---|---|---|---|---|---|---|---|---|
| Smoke: Gaussian, 500 x 20, K = 2, seed 1002 | `core070_realistic_size_cell` R and jl, as in P0 | 3.3 | 15.0 (+ confint 9.9) | TRUE / sd report present | true / true | -11500.6916332996 | -11500.6916331756 | 1.2e-7 | PASS |
| C3 NB2, seed 1018 | same harness, per-trait dispersion | 25.1 | 211.7 (+ confint 148.9; job 507 incl. cond(H)) | TRUE / sd report present | true / true | -23350.0547483898 | -23350.0547477228 | 6.7e-7 | PASS (margin small) |
| C3 ordinal logit, 4 categories | new generator; R `ordinal_logit()` vs Julia `Ordinal()` | 5.8 (+ sdreport 5.2) | 55.1 (+ confint 82.7; job 141) | 0 / TRUE | true / true | -11506.9285615101 | -11506.9285612997 | 2.1e-7 | PASS |
| C3 iSDM, 500 cells x 20 species x 2 sources, d = 2 | new generator; `isdm_sources(gbif = poisson, survey = cloglog)` | 14.1 (+ sdreport 6.4) | 44.4 (job 47) | 0 / TRUE | true (cells_converged true) / not computed | -23699.3261706562 | -23699.3261702471 | 4.1e-7 | PASS |
| C3 temporal, AR(1) latent, 25 series x 20 times x 20 traits, d = 1 | new generator | 2.1 (+ sdreport 0.3) | NOT FINISHED at 900 s | 0 / TRUE | none | -9721.0184631054 | none | none | CAP HIT (finding) |
| C4 beetle, 87 sites x 68 species, NB2, 3 covariates | `gllvm::beetle`, shared slopes, d = 2 | 9.2 (+ sdreport 10.8) | NOT FINISHED at 900 s | 0 / FALSE | none | -8933.7812030706 | none | none | CAP HIT (finding) |
| C4 fungi subsample, 300 sites x 59 species, binomial logit, 3 covariates | `gllvm::fungi`, shared slopes, d = 2 | 7.4 (+ sdreport 9.7) | NOT FINISHED at 900 s | 0 / TRUE | none | -5792.1862524503 | none | none | CAP HIT (finding) |

R's `max_abs_gradient` at the optimum was 4.4e-2 (temporal), 3.0e-3 (beetle), 3.9e-3 (fungi), 4.3e-4 (ordinal), 2.1e-2 (iSDM). R's `cond(H)` values: Gaussian 8.6e3, ordinal 1.3e3, iSDM 7.2e2, temporal 2.3e3, fungi 2.7e4, beetle 3.4e8 (pdHess FALSE). Julia's `cond(H)` was computed only by the Gaussian and NB2 harness (2.6e3 and 1.9e2); it is on a different parameter basis from R's (Gaussian 8.6e3 against 2.6e3, NB2 808 against 192), so I do not compare them and the plan should not treat them as comparable.

Both sides' loadings, fixed effects and standard errors were not compared in this pre-run. The pre-run's job was timing, convergence and logLik agreement; coefficient comparison belongs to the campaign.

## 3. Failures split by mechanism

Environment (1):
- The `rlang` binary in the shared personal library `~/R/lib` does not load under R 4.6.1 (`undefined symbol: SETLENGTH`). Every package that imports rlang failed (lifecycle, tidyselect, fmesher, so gllvmTMB would not install). I did not touch the shared library. I installed a current rlang (1.3.0) into the lane-local library and put it first on the library path. The shared library is stale for R 4.6.1 and other lanes that use it with rlang will hit the same error.

Convergence (3 cells, Julia side only):
- Temporal, beetle and fungi did not finish in 900 s. The cause is cost, not a stopped optimiser (details in section 4). Nothing was cut off at a wrong answer; there is no Julia answer to compare for these three.

Real disagreement: none established. Among the cells that finished, the largest gap is 6.7e-7 (NB2), inside the 1e-6 tolerance. Among the reduced diagnostics, three fits differ from R by more than 1e-6 (section 4). Their cause is not resolved, so they are not classed as (a), (b) or (c). I did not run the cross-objective check (R's objective at Julia's estimate), which is what would split them.

## 4. Diagnostics at reduced size (not pre-run cells; 300 s cap each)

Run only to explain the three cap hits and to give the campaign estimate something measured. Same scripts, same data, first rows or species only.

| Diagnostic | R fit (s) | Julia fit (s) | R logLik | Julia logLik | Julia minus R | Julia flag |
|---|---|---|---|---|---|---|
| temporal, 3 series x 10 times | not run | 66.3 | none | -569.8276503 | none | converged |
| temporal, 5 series x 10 times | 0.6 | 96.2 | -939.8855465722 | -939.8855465407 | +3.2e-8 | converged, 37 iterations |
| temporal, 10 series x 10 times | not run | 180.8 | none | -1879.1852176 | none | converged, 37 iterations |
| temporal, 3 series x 20 times | 0.7 | NOT FINISHED at 300 s | -1134.6832021 | none | none | none |
| beetle, 8 species, 87 sites | 0.9 | 213.0 | -800.3549716 | -797.7679443 | +2.59 | converged = false |
| beetle, 12 species, 87 sites | 1.1 | 287.7 | -1507.0582721 | -1507.0196840 | +0.039 | converged = true |
| beetle, 20 species, 87 sites | 1.6 | NOT FINISHED at 300 s | -2432.6406796 | none | none | none |
| fungi, 10 species, 100 sites | 1.3 | 48.3 | -530.4840709 | -531.7190141 | -1.23 | converged = false |
| fungi, 20 species, 100 sites | 1.8 | 115.1 | -870.5427247 | -871.2091607 | -0.67 | converged = false |

What these show:
- Temporal. The Julia fit agrees with R at small size (3.2e-8). Julia time grows about 14 s per series at 10 time points plus a fixed 40 s or so of compile. Doubling the series length from 10 to 20 times changed 3 series from about 30 s of fit to more than 300 s, so the cost rises steeply with the number of time points (consistent with a dense covariance block per series of size 20 x T). At 25 series x 20 times the fit is much longer than 15 minutes; I cannot give an upper bound. R takes 2 seconds.
- Beetle and fungi. Both Julia routes are the covariate fitters (`fit_nb_gllvm_grouped_cov`, `fit_gllvm_cov`) reached through `gllvm(@formula ...)`. Julia time rises from 213 s at 8 species to 288 s at 12 and passes 300 s at 20, against 1 to 2 s for R at every size. The source code states these covariate fitters use a finite-difference gradient, which would explain the cost, but I have not profiled it. Full size (68 species, or 59 species by 300 sites) is far beyond the cap.
- The logLik differences in the beetle and fungi diagnostics run in both directions. Julia is higher than R for beetle (R stopped at relative convergence with pdHess FALSE at 8 and 20 species) and lower than R for fungi (Julia flagged unconverged). That pattern suggests optimiser stopping on both sides, but this is an inference, not a measurement. It should be settled with the cross-objective check before any C4 beetle or fungi row is written.

## 5. Where the pre-run differs from the plan (please read)

1. Temporal rank. The plan asks for K = 2 for the temporal cell. Both gllvmTMB at P1 and `temporal_latent()` in Julia admit only `d = 1`. The cell ran with d = 1. A rank-2 temporal row cannot be built at P1 on either side.
2. Slope structure for beetle and fungi. The plan's formula `traits(spp) ~ 1 + env` expands in gllvmTMB to per-species slopes (`(0 + trait):env`). Julia's formula routes with per-species dispersion give shared slopes across species, and the per-species-slope route (`fit_gllvm_speciescov`) has a single shared NB2 dispersion. There is no single Julia route for per-species slopes plus per-species NB2 dispersion. I ran shared slopes on both sides (`value ~ 0 + trait + pH + Moist + Org + latent(...)`) so the two models match. The plan's `data-RD-BEETLE-NB2` and `data-RD-FUNGI-BINOMIAL` rows describe a model that, as written, Julia does not fit. The plan should say which model each row means.
3. Bridge route not exercised. All cells compare direct engines (R TMB fit against a direct Julia fit), not R's `engine = "julia"` bridge. C4 rows are defined through the bridge, so bridge overhead (P0 urbanisation: 364 s through the bridge against 59 s in TMB) is not in these numbers.
4. Fungi subsample rule. The rule "the 60 most prevalent species at prevalence 5 to 60 percent" selects only 59 species in `gllvm::fungi` (prevalence taken over all 1666 sites). I used all 59 and a seeded draw of 300 sites (seed 20261004); none was absent from the draw. The rule needs one line of amendment.
5. Julia compile time is inside the Julia fit times above. The Mac figures in the plan section 4.1 excluded it. For the quick cells this is a visible part of the number (Gaussian 15 s fit of which most is compile); it does not change any verdict.
6. Julia 1.12.6 here, 1.10 on the Mac. Julia dependency versions were resolved fresh (no `Manifest.toml` is tracked); the resolved manifest sha256 is below.

## 6. Time measured and the updated campaign estimate

Measured on Totoro (load average 25 to 90 from other users; 8 jobs at most, 2 Julia threads each):
- Smoke test: 35 s for both engines together.
- Pre-run proper (6 cells, 12 processes, run 8 at a time): 15.0 minutes wall, set by the three capped Julia processes.
- Reduced diagnostics: about 11 minutes wall.
- Setup (shipping source, building R library with the rlang repair, Julia precompile): about 5 minutes.
- Total Totoro wall time from first command to last result: about 32 minutes. Mac-side reading, scripting and this write-up are not counted. Well inside the 1 h target and the 2.5 h hard stop.

Updated serial estimate for the full campaign. "M" is measured here, "D" is from the diagnostics above, "P" is carried from the plan unchanged.

| Row | Plan point / upper (min) | Now (min) | Source |
|---|---|---|---|
| C3 Gaussian | 1 / 3 | 1 | M |
| C3 Poisson | 2 / 5 | 2 | P (not run) |
| C3 NB2 | 6 / 15 | 9 | M |
| C3 Binomial | 2 / 5 | 2 | P (not run) |
| C3 ordinal | 8 / 20 | 3 | M |
| C3 phylo latent | 0 / 0 | 0 | P |
| C3 temporal | 10 / 30 | more than 15, no upper bound | M lower bound, D |
| C3 iSDM | 10 / 30 | 1.5 | M |
| C3 total without temporal | 40 / 110 | about 18 | |
| C4 urbanisation | 8 / 15 | 8 | P |
| C4 crabs | 1 / 3 | 1 to 3 | P (Gaussian route, not tested here) |
| C4 eSpider (100 x 12 NB2) | 3 / 10 | about 5 | D (beetle 12 species took 288 s) |
| C4 beetle (68 species) | 20 / 60 | more than 15, no upper bound | M lower bound, D |
| C4 fungi (300 x 59) | 8 / 20 | more than 15, no upper bound | M lower bound, D |
| C4 total without beetle and fungi | | about 15 to 20 | |
| C5 | 5 / 10 | 5 | P |

Serial total: about 40 to 45 minutes for everything except temporal, beetle and fungi at full size, which are open-ended (each more than 15 minutes, probably hours). The campaign cannot be shown to stay under the 3 hour line with those three at the sizes in the plan. With 8 parallel jobs, wall time would then be set by the slowest of those three.

Options for you (these change the plan, so they are yours to choose):
1. Drop temporal, beetle and fungi from this campaign as "not feasible at this size" (the plan's own rule) and run the remaining 14 rows. Estimated serial about 45 minutes, so under the line.
2. Shrink them to sizes that finish. Measured to finish: temporal 10 series x 10 times (181 s); beetle 12 species by 87 sites (288 s); fungi 20 species by 100 sites (115 s, but Julia flagged unconverged). These are smaller than the stated size and would need to be written as limits in the row notes.
3. Treat the slow covariate fitters as an engineering finding for a separate lane, and resolve the beetle and fungi disagreements by the cross-objective check first.
4. Reapprove the three open-ended cells with a plan and a proper time estimate. I cannot give that estimate from this pre-run.

My recommendation is option 1 now, with option 3 recorded as the reason the two real-data rows are deferred. That is a judgment call about what C4 should claim, and it is yours.

## 7. Provenance

Lane directory on Totoro: `~/hsq_work/true-parity-prerun-20261001` (not shared, not `/scratch`). Totoro has no `/project`; `~/hsq_work` is the convention in `tools/totoro-setup.md`.

- gllvmTMB: clean `git archive` of 9539352f66f2db2cc26b1c393e67212a359b60c9, installed into `Rlib` in that directory; `packageVersion("gllvmTMB")` reported 0.7.1. No shared R library was written. The only other R packages used were already installed in `~/R/lib`, read only, except rlang, which was replaced lane-locally.
- GLLVModels.jl: `git archive` of origin/main 9ad7b0036405c5ddfda8a9e9bfce35da39bd0ccd into `jl`, Julia depot `depot` first on the depot path. Resolved `Manifest.toml` sha256 1912be8b782a0f419aaff9c8acde546cddf0686d7d03185542c2c5ece90e3a32.
- Scripts (committed with this file): `tools/true_parity/prerun/prerun_gen.R` (data generation and the GPL data loaders), `prerun_R.R`, `prerun_J.jl`. Gaussian and NB2 used the existing `tools/core070_realistic_size_cell.R` and `.jl`.
- Data files are not committed. Hashes (sha256) of what each engine read:

| File | sha256 |
|---|---|
| gaussian_p20_n500_K2.csv (seed 1002) | 5dedd05b0fa245cbecabb9425ad9f2b82a2763dee1cfe3802ec28210ca72acca |
| nb2_p20_n500_K2.csv (seed 1018) | a91ea44e095f3058b72b786c43a4d2b50b0c7039027da180377bae6df406e8fd |
| ordinal_p20_n500_K2.csv (seed 20261001) | df6a45d4f51cabe50a394aad203fae7368fd2abb88729d067e7683d2d193c0c1 |
| temporal_s25_t20_p20_d1.csv (seed 20261002) | f0ac0008345314a095dadd0f892b1709c772962ca9979ed15848cf568dcf2681 |
| isdm_c500_sp20_s2_K2.csv (seed 20261003) | 48fb0c500f961ee1dd8808d64d72cd5d4fd5f5354ca13b416be24b6818c0dc10 |
| beetle_wide.csv (`gllvm::beetle`, GPL-2; pH, Moist, Org scaled) | 1d7603ddceac61ea7179ed9519cb87bc5b7557ec1dd30f9267bd6224cd952c9b |
| fungi_wide.csv (`gllvm::fungi`, GPL-2; 59 species, 300 sites, seed 20261004; TEMPR, PRECIP, log.AREA scaled) | 498b84b93d2759785272cf33e7620f556033cb28c8a017596e146d6b5318b1bb |

The two real-data CSVs are derived from GPL-2 package data and stay on Totoro. They are regenerated by `prerun_gen.R beetle` and `prerun_gen.R fungi` from the installed `gllvm` 2.0.13.

## 8. Cleanup

Every process started for this pre-run has exited (checked by process listing after the last job; the 3 capped Julia jobs were ended by `timeout`, the three capped diagnostics likewise). Nothing is running from this lane. Files remain under the lane directory on Totoro for the maintainer to inspect or delete.
