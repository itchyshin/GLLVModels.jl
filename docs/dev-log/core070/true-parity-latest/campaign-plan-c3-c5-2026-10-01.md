# Campaign plan: true-parity clauses C3, C4, C5 (realistic size, real data, grouping levels)

Status: DRAFT PLAN for the maintainer. Nothing here is run, signed or classified. No case-map, scoreboard, checker or test file is touched by this document.
Pin P1 = gllvmTMB main `9539352f6` (0.7.1). Boundary: temporal and phylo latent inside; column grammar and spatial outside.
Written 2026-09-30 against `origin/main` `9ad7b0036`.

Terms. A *row* is one scoreboard line (`| id | requires | status | receipt | notes |`). A *receipt* is a tracked file recording an R-versus-Julia comparison. A *twin* is a Julia test that reproduces an R result on the same data. The *bridge* is R's `engine = "julia"` route. *D-287* is the rule that any run over 3 hours needs a plan, a pre-run test and your approval.

## 0. Summary

| Clause | Rows proposed | New compute | Where |
|---|---|---|---|
| C3 realistic size | 8 EVIDENCED rows | about 45 min point estimate, about 1.7 h upper bound, serial | Totoro |
| C4 real data | 5 EVIDENCED rows plus 3 signed-disposition rows (8) | about 55 min point, about 2.7 h upper bound, serial | Totoro (urbanisation and bridge checks may stay on the Mac) |
| C5 grouping levels | 4 EVIDENCED rows | none new (receipts exist from PR #593); about 5 min to re-verify | Mac |
| Total | 20 rows (17 EVIDENCED, 3 DISPOSITION-SIGNED) | about 1.7 h point, about 4.7 h upper bound, serial | wall time about 1 h with at most 8 parallel jobs |

Because the serial upper bound exceeds 3 hours, this plan treats the campaign as over the D-287 line. It needs your approval, and section 4 gives the pre-run test that must come first. The point estimate alone would be under the line; I am not relying on it, because 5 of the 8 C3 cells and 2 of the 5 C4 fits have never been timed.

Measured today: only two Julia families (Poisson, Binomial) at reduced sizes on the Mac. Everything else is extrapolated from the P0 Totoro grid or is a stated guess. Section 4 marks which is which.

## 1. Proposed rows

### 1.1 Id convention

The scoreboard id is the case-map `source_id` with `/` replaced by `-` (for example `family/FAMILY-00-IDENTITY` becomes `family-FAMILY-00-IDENTITY`). The checker splits clauses by id shape: suffix `-RSZ` is C3, prefix `RD-` is C4, prefix `GRP-` is C5. The open question in decisions item 5 (match after the family prefix, or rename) is being fixed by a separate PR; this plan assumes it accepts `data-RD-01`-style ids. The ids below follow that shape, so they work under either answer except "rename rows", which would only drop the `family-` style prefix.

Tolerance column legend: **[twin]** means the value is the one an existing twin or receipt already uses; **[PROPOSED]** means I chose it and you should confirm it. A row passes only if every listed quantity is inside its tolerance on both engines' converged fits. "Converged" means R `convergence == 0` with a positive-definite Hessian and Julia `converged == true`; a non-converged fit is recorded and the row stays unbound.

### 1.2 C3 realistic-size rows (8)

All cells: K = 2, seed fixed per row, same literal data to both engines (CSV or JSON written once by the R generator, sha256 pinned). Each receipt records both logLik values, fixed-effect estimates and SEs, `cond(H)` on both sides, wall times, seed, host, and `measured_against: P1`.

| Row id (case-map source_id) | Capability | Model | Size (n, p, K) | Quantities and tolerances | Pass rule |
|---|---|---|---|---|---|
| `family-GAUSSIAN-IDENTITY-RSZ` (`family/GAUSSIAN-IDENTITY-RSZ`) | Gaussian latent ordination, per-trait intercepts | `value ~ 0 + trait + latent(0 + trait \| site, d = 2)` | 500, 20, 2 | logLik abs diff 1e-6 [twin; P0 worst was 2.3e-7]; beta abs 1e-4 [PROPOSED]; beta SE rel 1e-3 [PROPOSED]; Lambda Lambda' elementwise abs 1e-4 [PROPOSED] | all within tolerance, both converged |
| `family-POISSON-LOG-RSZ` | Poisson, log link | same, `family = poisson()` | 500, 20, 2 | logLik 1e-6 [twin]; other quantities as above | same |
| `family-NB2-LOG-RSZ` | NB2 with dispersion | same, `family = nbinom2()` | 500, 20, 2 | as above plus dispersion (per trait) rel 1e-3 [PROPOSED] | same |
| `family-BINOMIAL-LOGIT-RSZ` | Bernoulli logit, the bridge-backed family with no size receipt | same, `family = binomial()` | 500, 20, 2 | as above | same |
| `family-ORDINAL-LOGIT-RSZ` | ordinal logit (4 categories) | `ordinal_logit` latent, shared thresholds | 500, 20, 2 | logLik 1e-6 [twin: ordinal twin], thresholds abs 1e-4 [PROPOSED], loadings as above | same |
| `covariance-COV-PHYLO-LATENT-RSZ` | phylo latent, first order (A14/A15 shape) | `phylo_latent` on a 100-species tree, 5 replicates per species | 500 observations, 20 traits, rank 2 | reuse the existing P1 receipts (PR #547): objective, Sigma_phy, `cond(H)` 8.17e4; logLik tolerance as the A15 receipt | same, with the A15 caveat: Julia ended `converged = false` within 2e-10 of the optimum (maintainer packet row 25 settles whether the flag counts) |
| `covariance-COV-TEMPORAL-RSZ` | temporal latent (inside P1 boundary), Gaussian | `fit_temporal_gllvm` with AR(1) latent, 25 units x 20 time points | 500 observations, 20 traits, rank 2 | logLik 1e-6 [twin: temporal fit receipts]; AR parameter abs 1e-4 [PROPOSED]; loadings 1e-4 [PROPOSED] | same |
| `isdm-ISDM-HEADLINE-RSZ` | headline integrated SDM: non-spatial, Laplace, first order, plus `predict` | `value ~ 0 + trait + trait:env + trait:src_<source> + offset(log_support) + latent(0 + trait \| cell_id, d = 2, unique = FALSE)`, two sources (cloglog presence/absence, count with support offset) | 500 cells, 20 species, 2 sources, K = 2 | logLik 1e-6 [twin: iSDM P1 twins]; b_fix abs 1e-4 [PROPOSED]; predict link and response on fitted cells and on newdata with offset 0, abs 1e-6 [PROPOSED, the isdm fixtures already store these vectors] | same |

Not proposed (kept out to stay small): `family/ZI-1FO-RSZ` (2 of 20 NB2 draws had no usable Laplace optimum in either engine; fenced by decision 27), Beta, Gamma, NB1, Tweedie, two-part and truncated families (they have P1 twins but no size cell). Add them later as rows of the same shape if you want C3 to cover every twinned family.

### 1.3 C4 real-data rows (8: 5 EVIDENCED, 3 signed dispositions)

A C4 row is a real-data workflow end to end through `engine = "julia"`, passing the eight acceptance classes of `docs/dev-log/core070/real-workflow-acceptance-lessons.md` (Rscript gate, data shape, labels, name translation, control symmetry, gradient exposure, scale feasibility, claimed limits). Evidence is an ACC receipt in the shape of `acc-bridge-urbanisation-receipt-2026-09-05.json` made at P1: TMB logLik, bridge logLik, wall times, gradient health, per-class verdicts, dataset sha256.

Comparison quantities for every EVIDENCED C4 row: logLik abs diff 1e-6 [twin; the P0 urbanisation gap was 1.6e-7], fixed-effect estimates abs 1e-4 [PROPOSED], Lambda Lambda' elementwise abs 1e-4 [PROPOSED], `predict` link on the training data abs 1e-6 [PROPOSED]. Pass rule: all within tolerance, eight classes all PASS, no class waived.

| Row id | Capability | Model | Data source and licence | Notes |
|---|---|---|---|---|
| `data-RD-URBANISATION-BINOMIAL` | binary probit ordination | `traits(items) ~ 1 + latent(1 \| review, d = 2)`, `binomial(probit)` | the maintainer's own `urbanisation_map` matrix (191 x about 44 binary items), local file used in the P0 ACC receipt; licence: the maintainer's own repo, confirm it may be redistributed before any copy is tracked | the only cell with a P0 receipt (P0: TMB 59 s, bridge 364 s). Track the sha256 and a loader, not the data, unless you confirm |
| `data-RD-CRABS-GAUSSIAN` | Gaussian ordination, classic morphometrics | `traits(FL, RW, CL, CW, BD) ~ 1 + sex:sp + latent(1 \| specimen, d = 1)`, log-transformed traits | `MASS::crabs`, ships with R (recommended package), 200 specimens x 5 traits, GPL-2 or GPL-3 | tiny; checks the Gaussian bridge on real, strongly collinear data (the d = 1 axis is body size) |
| `data-RD-SPIDER-NB2` | count ordination with covariates | `traits(spp) ~ 1 + env + latent(1 \| site, d = 2)`, `nbinom2` | `gllvm::eSpider` (hunting spiders, 100 sites x 12 species, counts, 26 covariates, use 3 to 4), GPL-2 | the smallest real count community; also the first real NB2 bridge run |
| `data-RD-BEETLE-NB2` | larger count community, the scale-feasibility case | `traits(spp) ~ 1 + env + latent(1 \| site, d = 2)`, `nbinom2` | `gllvm::beetle` (ground beetles, 87 sites x 68 species, counts, land-use covariates), GPL-2 | p = 68 is the heaviest real case and the main timing risk (section 4) |
| `data-RD-FUNGI-BINOMIAL` | binary occupancy, logit | `traits(spp) ~ 1 + env + latent(1 \| site, d = 2)`, `binomial(logit)` | `gllvm::fungi` presence/absence, 1666 sites x 215 species, GPL-2. **Use a fixed subsample of 300 sites x 60 species** (rule written down before the run: the 60 most prevalent species at prevalence 5 to 60 percent, sites by a seeded draw) | the full table is outside a bridge run's feasible scale; the subsample is a stated limit, recorded under the "claimed limits" class |
| `data-RD-PHYLO-DISPOSITION` | phylo-structured real data | `gllvm::fungi` ships a phylogeny (`fungi$tree`), so a real phylo workflow exists | DISPOSITION-SIGNED: blocked at the bridge gate `GJL-GATE-STRUCTURED-TERMS` until gllvmTMB #1236 (A4a) lands; revisit at P2 | needs your `signed_by` and `signed_on` |
| `data-RD-TEMPORAL-DISPOSITION` | temporal real data | none of `gllvm`, `vegan`, `MASS`, `ape` ships a multivariate ecological time series I would call a real workflow | DISPOSITION-SIGNED: no bridge route for temporal at P1 (D-296) | signed by you |
| `data-RD-ISDM-DISPOSITION` | integrated SDM real data | no real multi-source dataset in the packages checked | DISPOSITION-SIGNED: no bridge route for iSDM at P1 (D-296, D-300 row 1c-5) | signed by you |

Checked locally (installed R packages, `data()` listing and dimensions only; no fits): `gllvm` 2.0.13 (GPL-2): beetle, eSpider, fungi, kelpforest, microbialdata, Skabbholmen. `vegan` 2.7-5 (GPL-2): mite 70 x 35, dune 20 x 30, varespec 24 x 44, BCI 50 x 225. `MASS` 7.3-65: crabs. `ape` 5.8-1: carnivora and others. gllvmTMB's `inst/extdata/examples/*.rds` (joint-sdm-example and similar) look like simulated package examples; I did not confirm provenance, so they are not used as "real data". Alternatives if a row proves infeasible: `vegan::mite` (Poisson or NB2, 70 x 35) and `gllvm::kelpforest`.

Licence handling. GLLVModels.jl is MIT. The GPL-2 datasets above load by name from the installed packages. The plan tracks a loader script, the dataset sha256, the subsample rule and the numeric receipts, and does not copy the tables into this repo. The Julia side reads a CSV that the R generator writes into scratch at run time. This makes the replay depend on R plus the named package, which is the price of not redistributing GPL data under an MIT repo. If you prefer tracked fixtures (the convention for twins), that is a licence call for you; I have not assumed it.

### 1.4 C5 grouping-level rows (4)

PR #593 (merged) already holds P1 receipts for `unit`, `unit_obs`, `cluster`, `cluster2` (`receipts/grouping/{unit,unit_obs,cluster,cluster2}.json`, fixtures under `receipts/grouping/fixtures/`). Each records name parity measured by calling each engine with the keyword and a misspelt negative control, plus a paired Gaussian fit on the same literal data. These receipts say `row_status: receipt only`. The work here is to turn them into rows, not to re-run anything.

| Row id (case-map source_id) | Capability | Quantities and tolerances | Pass rule |
|---|---|---|---|
| `fit-input-GRP-UNIT` (`fit-input/GRP-UNIT`) | `unit` accepted on `gllvmTMB()` and `fit_gllvm(...; unit)`, Gaussian pair agrees | name parity PASS with negative control rejected on both sides; paired logLik abs 1e-6 [twin: #576 reports 1e-8, #563 reports 9e-8, so 1e-6 has margin] | name parity PASS and paired logLik inside 1e-6 |
| `fit-input-GRP-UNIT-OBS` | `unit_obs` | same | same |
| `fit-input-GRP-CLUSTER` | `cluster` (third grouping) | same; Gaussian only, non-Gaussian numerical pairing fenced in the row notes | same |
| `fit-input-GRP-CLUSTER2` | `cluster2` | same | same |

Placement. There is no grouping case map. I propose adding these four to `case-map-fit-input.json` (grouping arguments are fit inputs), which needs no new assembler input. The alternative is a new `case-map-grouping.json` and an assembler change; either is a classification-adjacent choice for you.

## 2. Realistic size for C3 (n, p, K and why)

One size for all non-structured rows: n = 500 sites, p = 20 species, K = 2.

- It is the smallest cell of the P0 24-cell grid (`core070_realistic_size_cells.tsv`, p in {20, 50}, n in {500, 2000}), so each new row has a P0 sibling and the P1 numbers can be compared with P0 history.
- Community-ecology sets in this very plan range from 12 species (eSpider) through 35 (mite), 68 (beetle) to 215 (fungi); site counts range from 24 to 1666. p = 20 and n = 500 sit inside that range, in the lower middle, and cross the "toy" line (the twins use p of 2 to 6 and n of 30 to 150).
- K = 2 is the usual choice in the gllvm literature and in all the real-data rows here (d = 2).
- It keeps cost bounded: the P0 grid showed the cost is in the largest cells (NB2 p = 50, n = 2000 took 6578 s on Totoro and 9380 s of Julia on DRAC). This plan deliberately does not re-run p = 50, n = 2000.
- Honest limit: p = 20 is on the small side for community data, and one cell per structure is a spot check, not a scaling study. The P0 grid remains the historical evidence for p = 50 and n = 2000.
- Structured rows: phylo latent is 100 species x 5 replicates x 20 traits (the existing A15 shape); temporal is 25 units x 20 time points x 20 traits; iSDM is 500 cells x 20 species x 2 sources. The temporal and iSDM shapes are my proposal and are untimed.

## 3. Where it runs

Totoro (`snakagaw@totoro.biology.ualberta.ca`: 384 cores, no GPU, no queue), reached only through the existing ControlMaster socket `~/.ssh/cm-snakagaw@totoro.biology.ualberta.ca:22`. No fresh login, no Duo. I checked on 2026-09-30 that the socket is live (`ssh -O check`: master running). If the socket is gone when the run starts, flag once and stop; do not open a new login.

- Cap: at most 8 concurrent jobs, each `OPENBLAS_NUM_THREADS=1`, `JULIA_NUM_THREADS=2`, R single-threaded; about 24 cores at most, well under the 150-core ceiling (D-143).
- Totoro is shared. On 2026-09-30 20:22 its load average was 105 (about 1-minute 106, 15-minute 20), from other users. Check `uptime` before launching and wait for load under about 150 on the 384 cores.
- R and Julia: R (`/usr/bin/R`) and `~/.juliaup/bin/julia` exist there (the auto-d recovery lane uses them under `~/autod-recovery`). The P1 gllvmTMB library is lane-local on the Mac (`~/local-scratch/gllvmTMB-p1-scratch/Rlib`); it must be rebuilt or copied to Totoro, from a detached worktree at `9539352f6`, into a private library. Confirm `gllvmTMB` loads there and reports P1. This is not done yet.
- No `/scratch`: put the R library, the Julia depot and the keepers under `/project`.
- Mac: C5 re-verification (minutes) and the urbanisation ACC run (about 8 min) can stay on the Mac with `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4`. Nothing else needs the Mac.
- Not used: DRAC (cells too small for a queue), kohaku (no GPU need).

## 4. Time estimates

### 4.1 Measured pre-run (Mac, Julia only, no R)

Setup: Mac Studio, `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4`, Julia 1.10, worktree at `origin/main` `9ad7b0036`, K = 2, seed 42, a warm-up fit first so compile time is excluded. Script: `fit_poisson_gllvm(Y; K, hessian = :observed)` or `fit_binomial_gllvm(Y; K)`, then `confint(fit, Y)`. The cond(H) step the P0 harness adds was not run. All fits converged.

| Family | p | n | fit (s) | confint (s) |
|---|---|---|---|---|
| Poisson | 10 | 250 | 8.8 | 3.1 |
| Poisson | 10 | 500 | 9.0 | 6.7 |
| Poisson | 20 | 250 | 7.6 | 19.1 |
| Binomial | 10 | 250 | 3.2 | 3.5 |
| Binomial | 20 | 250 | 3.6 | 17.3 |
| Poisson (target size, one check) | 20 | 500 | 8.3 | 38.6 |

Scaling read from these: fit time is nearly flat in n and p at this scale (optimiser-iteration bound, 8 to 9 s Poisson, 3 to 4 s Binomial). `confint` grows about linearly in n (3.1 to 6.7 s from n = 250 to 500) and steeply in p (3.1 to 19.1 s from p = 10 to 20 at n = 250, exponent about 2.6, consistent with a Hessian of dimension about p x (K + 1)).

Extrapolation check. From the reduced cells alone, the Poisson target cell (p = 20, n = 500) was predicted at confint 19.1 x 2 = 38 s; one direct run measured 38.6 s. That is a single check on one family; it supports the scaling rule for Poisson and Binomial only. The Julia side of those two rows is therefore about 50 s (Poisson) and about 40 s (Binomial) at the target size on the Mac. Totoro single-core speed relative to the Mac was not measured.

### 4.2 Per-row estimates

"Source" says how each number was obtained: **M** = measured above, **P0** = from the P0 Totoro grid (`2026-09-05-totoro-t4-p6-grid.md`, serial R fit + Julia fit + Julia confint per cell), **X** = extrapolated with the rule above, **G** = guess, never timed. The upper bound is what I would put in the pre-run's go/no-go test.

| Row | Point (min) | Upper (min) | Source |
|---|---|---|---|
| C3 Gaussian | 1 | 3 | P0 smallest Gaussian cell 19 s |
| C3 Poisson | 2 | 5 | M (Julia about 50 s) plus R, P0 smallest cell 54 s |
| C3 NB2 | 6 | 15 | P0: smallest NB2 cell 217 s; n = 500, p = 20 should be near it |
| C3 Binomial | 2 | 5 | M (Julia about 40 s) plus R, X |
| C3 Ordinal | 8 | 20 | G |
| C3 Phylo latent | 0 | 0 | existing P1 receipts (Julia fit 18 s on the Mac); only re-wrapping |
| C3 Temporal | 10 | 30 | G |
| C3 iSDM | 10 | 30 | G |
| **C3 total, serial** | **about 40** | **about 110** | |
| C4 urbanisation | 8 | 15 | P0 (59 s + 364 s) |
| C4 crabs | 1 | 3 | G (200 x 5, d = 1) |
| C4 eSpider | 3 | 10 | X (100 x 12 is below the C3 cells) |
| C4 beetle | 20 | 60 | X: scaling the P0 NB2 p = 50, n = 2000 confint (3898 s) by (68/50)^2.6 x (87/2000) gives about 6 min of confint, then add the fit, covariates and the R side |
| C4 fungi subsample | 8 | 20 | X: binomial confint about 17 s at p = 20, n = 250, scaled to p = 60, n = 300 about 6 min, plus fit and R |
| C4 eight-class checks | +30 percent | +50 percent | G (the ACC checks add overhead on each run) |
| **C4 total, serial** | **about 55** | **about 160** | |
| C5 four rows | 5 | 10 | receipts exist; re-verify hashes and replay only |
| **Campaign total, serial** | **about 1.7 h** | **about 4.7 h** | |

With 8 parallel jobs the wall time is set by the longest single job (beetle, 20 to 60 min) plus setup: about 1 h point, about 1.5 h upper. The serial sum is what D-287 is guarding against, and the upper bound is over 3 h, so approval is required.

### 4.3 Pre-run test (run this before the full campaign)

Purpose: replace the guessed cells (G) and the extrapolated beetle and fungi cells (X) with measurements, and confirm the P1 library works on Totoro. A wrong answer is as important as a slow one, so each pre-run also checks that the two engines agree.

1. Environment smoke (10 min cap): on Totoro, load the private P1 library, print `packageVersion("gllvmTMB")` and the commit it was built from, fit the Gaussian p = 20, n = 500 cell in R and Julia, confirm |ΔlogLik| inside 1e-6.
2. One cell each, run in parallel, 15 min cap per cell, 1 thread per job: C3 NB2, C3 ordinal, C3 temporal, C3 iSDM, C4 beetle, C4 fungi subsample. Record Julia and R wall times separately, `converged` and `pd_hessian` on both sides, logLik difference, and `cond(H)`.
3. Go rule: proceed to the full campaign only if every pre-run cell finished inside 3 times its point estimate and the revised serial sum stays at or under 3 h. If the revised sum exceeds 3 h, the full run waits for a second approval with the measured numbers. A cell that does not finish in its cap is reported as a finding; the row is dropped from this campaign and listed as "not feasible at this size", not retried with a larger cap.
4. Wrong-answer partition. For any cell with |ΔlogLik| above 1e-6, first split by mechanism: (a) optimiser stopped early (compare each engine's objective at the other's optimum, the `xobj` method the iSDM fixtures use), (b) different local optimum, (c) a genuine model difference. Only (c) is a parity failure; (a) and (b) are recorded with the evidence and decided by you.

Pre-run total: about 1 h wall, at most 6 concurrent jobs on Totoro.

## 5. How each row becomes a receipt

Per row, in order. Agents never edit R source; the P1 library is private and read-only.

1. **R generator** (new files under `tools/true_parity/`, one per clause): installs nothing, uses `GLLVMTMB_P1_LIB=~/local-scratch/gllvmTMB-p1-scratch/Rlib` (Totoro copy for the Totoro run) and `GLLVMTMB_P1_SRC` checked to be at `9539352f6` (the check `export_p1_fixtures.R` already does). It writes the literal data (CSV/JSON), fits in R, and records logLik, estimates, SEs, convergence, `cond(H)`, wall time and the sha256 of its own file. C3 starts from `tools/core070_realistic_size_cell.R` and `.jl`, which already write most of these fields; the P0 pin in them is replaced by P1.
2. **Fixture**: the data file plus an R-values TOML, tracked under `test/fixtures/` (C3, C5) or, for GPL data, a loader plus sha256 only (C4, see section 1.3).
3. **Julia twin test**: one file per clause (for example `test/test_true_parity_rsz_p1.jl`) reading the fixture, checking the sha256, refitting in Julia and asserting each tolerance in section 1. It follows the `test_phylo_latent_paired_p1.jl` pattern. Enrolling it needs one added line in `test/runtests.jl`; that line is the only edit outside new files and I flag it.
4. **Receipt**: new cases in `tools/true_parity_julia_receipts.jl`, which already copies R values from the tracked fixture, recomputes the Julia value, reads each tolerance from the test assertion (file and line recorded, run aborts on exceedance) and writes under `docs/dev-log/core070/true-parity-latest/receipts/julia-twins/`. C4 receipts use the ACC receipt shape instead (bridge route).
5. **Case-map row**: one row per id in section 1 with `classification`, `executable_case_ids`, `evidence_tier`, `measured_against: 9539352f6...` and the receipt paths. Then `python3 tools/true_parity_assemble.py` regenerates `scoreboard.md`; `node tools/true_parity_check.mjs C3`, `C4`, `C5` report the counts.

Case-map rows that would be added (all need your sign-off, because adding rows is classification-adjacent and sets what counts toward a clause):

| Map | New `source_id` rows |
|---|---|
| `case-map-family.json` | `family/GAUSSIAN-IDENTITY-RSZ`, `family/POISSON-LOG-RSZ`, `family/NB2-LOG-RSZ`, `family/BINOMIAL-LOGIT-RSZ`, `family/ORDINAL-LOGIT-RSZ` |
| `case-map-covariance.json` | `covariance/COV-PHYLO-LATENT-RSZ`, `covariance/COV-TEMPORAL-RSZ` |
| `case-map-isdm.json` | `isdm/ISDM-HEADLINE-RSZ` |
| `case-map-data.json` | `data/RD-URBANISATION-BINOMIAL`, `data/RD-CRABS-GAUSSIAN`, `data/RD-SPIDER-NB2`, `data/RD-BEETLE-NB2`, `data/RD-FUNGI-BINOMIAL`, `data/RD-PHYLO-DISPOSITION`, `data/RD-TEMPORAL-DISPOSITION`, `data/RD-ISDM-DISPOSITION` |
| `case-map-fit-input.json` (or a new grouping map) | `fit-input/GRP-UNIT`, `fit-input/GRP-UNIT-OBS`, `fit-input/GRP-CLUSTER`, `fit-input/GRP-CLUSTER2` |

Signatures only you can give: the three C4 disposition rows (`signed_by`, `signed_on`), the tolerance choices marked PROPOSED, the row placement above, and the licence handling for GPL data.

## 6. Risks and what this plan does not cover

Risks.
- Convergence at size. The A15 cell ended `converged = false` within 2e-10 of the optimum on an absolute gradient tolerance (packet row 25); the same can happen on any C3 cell at objective values near 1e4. Mitigation: record both flags and the cross-objective check; do not widen a tolerance to hide it.
- NB2 and ordinal at size. P0 NB2 cells took up to 6578 s and, in the DRAC grid, timed out once at 2 h; two of 20 zero-inflated NB2 draws had no usable optimum in either engine. Beetle (p = 68, NB2) is the likeliest real-data cell to fail the scale-feasibility class.
- R runtime. R time was not measured in this pre-run (instruction: no R). The P0 Totoro numbers include R, but P1's R library is newer and the bridge adds overhead (P0 urbanisation: 364 s via the bridge against 59 s in TMB).
- Totoro contention. Load was 105 on a shared host at the time of checking; timings can inflate. Pre-run timings must be taken at the same load class as the full run.
- Data licences. Four datasets are GPL-2 or GPL-3; the repo is MIT. The plan avoids copying them. The urbanisation matrix is the maintainer's own, not public; its redistribution status is unconfirmed.
- Synthetic C3 data. The C3 cells are simulated (as in P0), so they test size, not realism; real-data realism is C4's job.
- Tolerances marked PROPOSED are untested at size. A tolerance that proves unreachable for an engine-agnostic reason (optimiser stopping) is reported, not relaxed silently.
- Library reconstruction. The P1 R library on Totoro does not exist yet; building it is a step in the pre-run and could surface compiler or dependency differences from the Mac.

Not covered.
- p = 50 and n = 2000 cells (kept as P0 history), and any scaling study beyond one cell per structure.
- Zero-inflated, Beta, Gamma, NB1, Tweedie, two-part and truncated families at size.
- Real data for phylo, temporal and iSDM (signed dispositions until the bridge routes exist).
- Non-Gaussian numerical pairing for grouping levels (C5 pairing is Gaussian; non-Gaussian is fenced in the row notes).
- Spatial and column-grammar structures (outside the P1 boundary).
- Checker changes (a separate PR), the Packet 2 rulings on other groups, and any push, merge or signature.
- Running anything: this document is a plan, and the only runs were the timed Julia pre-run in section 4.1.
