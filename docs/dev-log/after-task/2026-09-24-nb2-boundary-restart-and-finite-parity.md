# After-task: NB2 boundary-stall fix (#477) and finite-dispersion parity check (#476, B-lite) (2026-09-24)

Lane: `claude/nb2-finite-dispersion-parity-20260924`, built on #475's branch
(`claude/owed-2-4-decisions-20260924` @ `d653ef416`).

## 1. Goal

Shinichi chose option B for #476 ("new NB2 parity data where every trait keeps finite
overdispersion"). B touches the frozen contract, so the lighter form (B-lite) was taken when he
said "keep going": a developer parity check outside the frozen contract, NATIVE-06 left as a
documented boundary case. Screening data for it exposed a Julia defect (#477), which this lane
also fixes.

## 2. Implemented

- **Fix (#477).** `_nb_boundary_restart` in `src/families/grouped_dispersion.jl`, called by
  `fit_nb_gllvm_grouped` (the default NB2 route). When a group's `r` ends outside
  `[1e-6, 1e6]`, restart once from the returned point with those groups at `log r = 0` and keep
  the restart only if the negative log-likelihood drops by more than 1e-6. Docstring and
  CHANGELOG updated.
- **Stored data** `test/fixtures/nb2_finite_dispersion_data.toml` (seed 46, true `r = 1`, NATIVE-06
  design, drawn on Julia 1.10.12, SHA-256 `fcc7f4be…`), written by the new
  `tools/nb2_parity_data_draw.jl`, which reads the NATIVE-06 samplers from
  `test/parity/test_negbin_parity.jl` so they stay identical.
- **Core regression test** `test/test_nb_boundary_restart.jl` (in `runtests.jl`): on that data the
  fit must have no boundary group, every `r` in `(0.5, 5)`, and logLik −835.369277 ± 1e-5.
- **Developer parity check** `test/parity/test_nb2_finite_dispersion_parity.jl`, added to
  `runparity.jl`'s developer cohort only. (i) Finite data: both engines converge, no boundary on
  either, |Δ logLik| ≤ 1e-6 relative, per-trait `r` within 1e-3. (ii) NATIVE-06 data: Julia flags
  traits 1 and 3, R puts them above 1e5, the identified traits match within 1e-3, and the
  log-likelihoods agree. R's `r_gradient_max` is printed, not gated.

## 3a. Decisions and Rejected Alternatives

- **B-lite over B-full.** B-full means editing `frozen-r070-contract.toml`, which changes the
  contract hash recorded by every receipt and pinned in two evidence files, plus about 8 JSON
  registries and the formula and public-bridge NB2 cases. Left for the next contract revision.
- **Restart (b) over a new default start (a).** Screening showed both recover every Julia stall
  (3 of 10 datasets). (a) changes every fit; (b) changes only fits that reach the boundary.
- **Rejected: also restart from a fresh start at `r = 1`.** On seeds 51 and 52 the fresh start
  finds gllvmTMB's finite point, but that point has a *lower* log-likelihood (by 0.30 and 0.046)
  than Julia's boundary fit, so keeping the better fit already makes it a no-op there.
- **R's gradient is recorded, not gated, in the developer check.** Across 32 screened datasets
  with the fix, gllvmTMB stopped with `r_gradient_max` between 1e-5 and 2e-3 even where both
  engines agree to 1e-11 in logLik; the value measures R's stopping tolerance, not parity.
- **Rejected for now: the NB2-with-covariates fitter and other grouped families.** Same start
  pattern, not screened; listed in #477.

## 4. Files Touched

- `src/families/grouped_dispersion.jl` (helper, call site, docstring)
- `CHANGELOG.md`
- `tools/nb2_parity_data_draw.jl` (new)
- `test/fixtures/nb2_finite_dispersion_data.toml` (new)
- `test/test_nb_boundary_restart.jl` (new), `test/runtests.jl`
- `test/parity/test_nb2_finite_dispersion_parity.jl` (new), `test/parity/runparity.jl`
- `docs/dev-log/after-task/2026-09-24-nb2-boundary-restart-and-finite-parity.md` (this file)
- `docs/dev-log/check-log.md`, `AGENTS.md` (snapshot line)

## 5. Checks Run

Mac Studio, Julia 1.10.12, single-threaded; R side is the locally built frozen oracle
`b4d5fee64`.

Estimates written before each run: screening 2 to 5 min, variant tests about 2 min, the
regression file sweep 15 to 25 min. Actuals: 1 min, 6 min (32 datasets), 1 to 2 min per variant
run, sweep below.

| Check | Result |
|---|---|
| Screen, current code, 9 datasets + NATIVE-06 design | Julia below the optimum in 3 of 10 (Δ −0.65, −0.86, −1.56); Julia's own objective at gllvmTMB's answer equals gllvmTMB's logLik |
| Variants (a) start at `r = 1`, (b) boundary restart | both recover all 3; neither worse than current on any dataset |
| `test/test_nb_boundary_restart.jl` before the fix | 2 pass, **3 fail** (logLik −836.924, boundary groups) |
| same, after the fix | 5 of 5 pass; converged, 63 iterations, logLik −835.3692767, `r` = 1.089, 1.115, 2.736, 1.617, 0.808 |
| Existing files that exercise the fitter (Julia 1.10.12, test deps env), 15.5 min | all pass: `test_grouped_dispersion` 20, `test_nb_fit` 8, `test_fit_gllvm` 11, `test_aicbic_newfits` 18, `test_unified_api` 24, `test_known_sentinel_defects` 25 + 1 known broken, `test_grouped_hessian_consistency` 23, `test_bridge_grouped_dispersion` 129, `test_bridge_x` 192, `test_confint_family` 326 |
| Developer parity check on the Mac | finite data: 6 of 6 plus hash; NATIVE-06 boundary agreement: 4 of 4 |
| NATIVE-06 Julia fit before vs after the fix | identical, logLik −820.5485342071867 |
| Totoro, Julia 1.10.12, R 4.5.3, Track A's oracle, code = #475 `d653ef416` + this branch's diff (tree `02c03beb…`, equal to the local commit), 4.75 min | regression test 5 of 5; developer check passes: finite data Δ logLik 2.0987e-8 and the same five `r` as the Mac; NATIVE-06 boundary agreement passes (R trait 3 `r` 9.2e5, above the 1e5 bar) |

## 6. Tests of the Tests

- The regression test failed on the unfixed fitter with the defect's exact numbers and passes
  after the fix.
- The developer check's boundary-agreement testset passes only because both engines reach the
  boundary on traits 1 and 3; on the finite data the same assertions would fail.

## 7a. Issue Ledger

- #477 (Julia stall): fixed for `fit_nb_gllvm_grouped`; other grouped fitters open.
- #476 (NATIVE-06 cannot pass): B-lite carried out; B-full deferred to the next contract revision.
- gllvmTMB also stops below the optimum on this design in 4 of 10 datasets (up to 1.2 log-likelihood
  units); recorded in #477 for gllvmTMB's owners, not changed here (read-only).

## 8. Consistency Audit

- Callers of `fit_nb_gllvm_grouped`: `fit_gllvm` (default NB route), `src/bridge.jl:1249`,
  and the bootstrap in `src/confint_family.jl:615`. All get the fix; the bridge and confint test
  files were in the sweep.
- The boundary-flag tests (`test_grouped_dispersion.jl`, `test_confint_family.jl`,
  `test_bridge_x.jl`) use `fit_nb_gllvm_grouped_cov`, which this change does not touch.
- The frozen required cell NATIVE-06 still asserts `jl_fit.converged`, which a genuine boundary
  makes false; unchanged by design under B-lite.

## 9. What Did Not Go Smoothly

- The first reading of the post-fix screen called seeds 51 and 52 Julia stalls. A signed
  comparison showed Julia's boundary fit is the better one there and gllvmTMB is the one stuck.
- My first "keep the old data as a separate check" plan assumed B was a local fixture change;
  the frozen-contract pins only surfaced when grepping for the data hash.

## 10. Known Residuals

- On Totoro, R's trait-3 `r` on the NATIVE-06 data is 9.2e5, about 9 times the 1e5 bar in the boundary-agreement check; other machines sit higher (3.1e7 on the Mac).
- `fit_nb_gllvm_grouped_cov` and the Beta, Gamma, NB1 and other grouped fitters share the start
  pattern and were not screened.
- A fit that reaches the boundary now costs one extra optimisation.
- The full `Pkg.test()` suite was not run locally; CI shards cover it.

## 11. Team Learning

When one engine sits at a parameter boundary and the other does not, compare signed
log-likelihoods, and evaluate each engine's objective at the other's answer, before calling
either one stuck. Here that test separated a Julia stall (Julia lower) from an R stall
(Julia higher) on the same design.

## 12. Cross-Product Coverage

The restart is cross-cutting for every caller of the NB2 grouped fitter.

- Covers ✓: `fit_gllvm` NB default route, the R bridge's NB route, the confint bootstrap refits.
- This change does NOT cover: `fit_nb_gllvm_grouped_cov`, other grouped-dispersion families, the
  frozen NATIVE-06 cell, or gllvmTMB's own stalls.

Memory receipt: `route.py` has no LOAD-FIRST manifest for this repo. Applied: this repo's
`AGENTS.md` (tests with the change, docstring and CHANGELOG for the behaviour change, no widened
tolerance, no push without instruction), a lane lease on the touched paths (D-88), and a time
estimate before each run (D-139).

Golden Set: not in scope. No memory or retrieval class was touched.
