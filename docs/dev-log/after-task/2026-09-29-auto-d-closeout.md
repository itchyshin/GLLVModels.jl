# After-task: auto-d lane close-out (2026-09-27 to 2026-09-29)

## 1. Goal

Finish the auto-d lane: re-measure negative-binomial (NB) recovery for `select_lv` on the corrected NB2
kernel (#521), make the advisory frozen-R smoke cells pass for the right reasons, align gllvmTMB's ridge
guard with Julia's, and land the owed PRs.

## 2. Implemented

- NB recovery grid re-run (nibi array 22840890, plus tasks 388/389 on Totoro after a node fault): all 4,800
  datasets. Default rule (`:bic_sites`, lenient guard) 0.934 mean exact recovery (withdrawn old-kernel value
  0.895); `:bic` 0.840; AIC 0.793; strict guard 0.746. Figure restored in the `select_lv` docstring and
  design/74 (#518).
- #608: NB2 smoke cell on a new interior-dispersion fixture; R's gradient recorded, not gated.
- #610: Student-t. Real defect fixed: an estimated nu could run to the Gaussian limit past a higher interior
  optimum (0.064 logLik below gllvmTMB); `fit_studentt_gllvm` now restarts boundary traits from nu = 20 and 50.
  Student cells fit stored draws (seeded streams differ between Julia versions); the near-Gaussian diagnostic
  accepts a nu boundary on either engine.
- #529 test premise corrected (the 2-cycle contrast vanished because of #601, not the platform).
- #551, #540, #519, #529 refreshed and merged through the grouped-getLV lane's train (#606).
- gllvmTMB #1324 (closed for the 0.7.1 release, branch kept at 215f9544f): `select_lv` tests the penalised
  Hessian under the loading ridge; NB figures restored in `?select_lv` and NEWS.

## 3a. Decisions and Rejected Alternatives

- Shinichi: NB2 and truncated-NB2 smoke cells record R's gradient instead of gating (option a). Rejected:
  a one-step Newton polish of R's optimum (option b), more machinery for the same result.
- Shinichi: relax R's Hessian rejection under the ridge. Implemented as "test the penalised Hessian", not
  "skip the check"; rejected adding R's unpenalised check to Julia (would have made both engines 4/10).
- Student-t: restart from nu = 20 and 50 only when a nu reaches the boundary. Rejected a single restart value
  (probes from 3, 10 and 100 all missed the interior peak) and always-restart (cost on healthy fits).
- Stored-draw fixtures per seed, from the Julia version whose draw keeps both engines off a degenerate
  boundary (seed 71 from 1.10.12, seed 73 from 1.13.1), following the 2026-09-24 NB2 precedent.
- NB rerun on DRAC (not Totoro) so Totoro stayed free for other lanes; the 2 lost tasks later ran on Totoro.

## 4. Files Touched

GLLVModels.jl (merged): `src/model_selection.jl`, `docs/design/74-auto-latent-dimension.md` (#518);
`src/families/studentt.jl`, `test/test_studentt_boundary_honesty.jl`, `test/parity/test_studentt_parity.jl`,
`test/parity/fixtures/{generate_studentt_smoke_data.jl, studentt_cell9_data.toml, studentt_neargauss_data.toml}`,
`CHANGELOG.md` (#610); `test/parity/{test_negbin_parity.jl, nb2_health.jl}`,
`test/parity/fixtures/{nb2_smoke_data.toml, generate_nb2_smoke_data.jl}` (#608);
`test/test_grouped_getlv_mode.jl` (#529); merge resolutions in `CHANGELOG.md`, `docs/dev-log/check-log.md`,
`test/runtests.jl` on #518, #529, #540, #551, #610.
GLLVModels.jl (#629, open): `LOOP/lanes/auto-d-20260926/{checkpoint.md, nb-rerun-plan.md,
pilot/harvest-report-nbrerun-full.md, ridge/matched/r-pdhess-fix.md, ridge/matched/R_pdhessfix_n120_p20_K3.csv}`,
this report. Earlier lane-kit files (`pilot/run_nb_rerun_nibi.sh`, `pilot/smoke_nb_rerun_nibi.sh`,
`ridge/matched/pdhess_check.R`, `.log`, harvest reports) landed with #518.
gllvmTMB (branch `claude/lane-auto-d-r-20260926`): `R/select-lv.R`, `man/select_lv.Rd`,
`tests/testthat/test-select-lv-ridge.R`, `NEWS.md`, `docs/dev-log/check-log.md` (merge).
Brain vault: `memory/AGENT_LOG.md` (4403be7e).
Remote: nibi `~/projects/def-snakagaw/snakagaw/auto-d-nbrerun/`; Totoro `~/hsq_work/auto-d-nbrerun-388-389/`.

## 5. Checks Run

- NB harvest `pilot/analyze.py` on 943 task files: 4,800 complete datasets, 0 incomplete.
- #518 lane tests (binomial_ridge, model_selection, model_comparison, boundary_inference): 286/286.
- #551 getLV, Beta, NB2, NB1 tests: 1014/1014. #529: 307/307, then 384/384 after the main merge.
- #610 Student-t unit tests + #588 bootstrap verdict: 1953/1954 (1 pre-existing broken); NATIVE-10 with R
  (local oracle): 33/33 on Julia 1.13.1 and 1.10.12, then 32/32 after the R-boundary change.
- #608 NATIVE-06 NB2 cell with R: 17/17.
- gllvmTMB `test-select-lv-ridge.R` 10 tests (47 expectations), `-guard` 21, `-anova` 30: 0 failures; CI all green.
- CI evidence: #608's smoke run NB2 green; #610's smoke run all Student cells green.

## 6. Tests of the Tests

- Student restart regression test uses the stored 1.13 draw where the pre-fix fitter ended at logLik
  -1430.162 and nu1 = 5e9; the test asserts logLik > -1430.10 and 10 < nu1 < 30, which the old code fails.
- The penalised-Hessian unit test uses an objective with an indefinite ridge block (eigenvalue -0.1): it
  asserts FALSE without the ridge and with a ridge too weak to fix it (1/tau^2 = 0.04), TRUE with 0.25.
- R evidence after the fix on the 10 matched datasets: bic_sites 4/10 before, 9/10 after (Julia 9/10).

## 7a. Issue Ledger

- Fixed: NB figures withdrawn (restored, 0.934); Student-t nu boundary optimum (#610); NB2 and truncated-NB2
  gradient gates (#608, #613); gllvmTMB ridge Hessian (branch kept); #529 test misattribution.
- Deferred: NB sentence in gllvmTMB `R/gllvmTMB.R` still says "not yet measured" (release lane owns the file);
  gllvmTMB #1324 closed until after 0.7.1; `test_nb2_formula_parity.jl:44` gradient gate taken by the
  true-parity lane.
- Flagged to owners: GP-1 verdict pinned logLiks in train #606 (platform); truncated-NB2 smoke cell (#613).

## 8. Consistency Audit

- Swept every `test/parity/*.jl` on main for R-gradient gates: three found (NB2, truncated NB2, NB2 formula);
  all now handled or owned.
- Swept my own "old code was wrong" premise tests (NB2, Beta, Student, getLV) for platform or branch
  dependence: only #529's needed a change.
- The penalised-Hessian defect is the same class as gllvmTMB #1092 (gradient); both call sites that set
  `pdh` were changed.

## 9. What Did Not Go Smoothly

- Twice misattributed a cause before measuring: the NB2 smoke failure (boundary explains Julia, not R's
  gradient) and #529's premise (said platform; it was #601). Both corrected on evidence.
- DRAC submission was blocked by the permission classifier; Shinichi ran `sbatch` and a duplicate smoke job
  resulted. The nibi socket dropped mid-run; reopened with an `expect` script and a Duo push.
- A backgrounded `&&` chain on Totoro launched one task wrong; a `ln -s` into an existing directory hid a
  receipt; a closeout tool wrote its skeleton into the vault. Each was cleaned up file by file.
- GitHub Actions queue saturation made CI waits run to hours.

## 10. Known Residuals

- gllvmTMB fix and NB figures are on a closed branch until after the 0.7.1 release.
- The Student restart values (20, 50) come from one dataset's probe.
- The docstring and design/74 cite "4,794 datasets"; the full 4,800 gives the same 0.934.
- #613 and #629 await Shinichi's merge.

## 11. Team Learning

- A check on R's own optimiser gradient tests the machine, not the model (same data 5.6e-5 on Totoro,
  2.4e-3 on CI, 4.9e-3 on a Mac). Record it; gate on R's convergence code and logLik agreement.
- A penalty applied outside TMB must also be applied to every TMB-derived diagnostic (gradient, Hessian).
- Before attributing a vanished bug to the platform, run the old check against each branch alone.
- Seeded parity data differs between Julia versions; store the draw.
Memory receipt: routed guards for lanes (lane_preflight, lane_lease), D-287 estimate-before-run, D-64 Duo,
D-276 agent mentions, merge-only-on-word; all applied. Logged in `memory/AGENT_LOG.md` 2026-09-29.
Golden Set: not in scope (no golden-set surfaces touched).

## 12. Cross-Product Coverage

Covers: GLLVModels.jl `select_lv` NB recovery (NB2 grouped default route); NB2, truncated-NB2 and Student-t
frozen-R smoke cells; Student-t estimated-nu fitting; gllvmTMB `select_lv` guard under the binary ridge.
Does NOT cover: NB1, Tweedie or other families' recovery on corrected kernels; Student-t fits with shared
dispersion or covariates beyond the parity cells; gllvmTMB release 0.7.1 itself; Julia's guard under AGHQ;
the frozen-R cells owned by other lanes beyond the gradient-gate sweep.
