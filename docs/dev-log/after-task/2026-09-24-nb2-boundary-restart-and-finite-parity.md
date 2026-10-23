# After-task: NB2 boundary restart (#477) and B-lite parity check (#476) (2026-09-24)

Lane: `claude/nb2-finite-dispersion-parity-20260924`, PR #478 (opened stacked on #475; reopened
and retargeted to `main` after #475 merged and its branch deletion closed it).

## 1. Goal

Shinichi chose option B for #476 ("new NB2 parity data where every trait keeps finite
overdispersion"). B rewrites the frozen contract, so the lighter form (B-lite) was taken when he
said "keep going": a developer parity check outside the frozen contract, NATIVE-06 left as a
documented boundary case. Screening data for it exposed a Julia fitting defect (#477), which this
lane also addresses.

## 2. Implemented

- **Restart (#477).** `_nb_boundary_restart` in `src/families/grouped_dispersion.jl`, called by
  `fit_nb_gllvm_grouped` (the default no-covariate NB2 route). When groups end outside
  `[1e-6, 1e6]`, restart from the returned point with the boundary groups at `log r = 0`: all of
  them together and, when there are several, each on its own. Keep the best result only if the
  negative log-likelihood drops by more than 1e-6. Docstring (including that `iterations` counts
  the kept run only) and CHANGELOG updated. The same restart is applied in
  `fit_nb_gllvm_grouped_cov` (NB2 with site covariates), which the sibling screen showed has the
  same stall.
- **Core tests** `test/test_nb_boundary_restart.jl` (in `runtests.jl`), 10 checks: seed 46 must
  reach at least gllvmTMB's point (−835.369277); seed 52 must reach the point that keeps trait 2
  at the boundary (−833.181377) and stay flagged; and a deterministic check of the helper on two
  toy objectives, one where the restart must be discarded (a genuine boundary) and one where it
  must be kept (a stall). Data `test/fixtures/nb2_restart_seed46.toml` and
  `nb2_restart_seed52.toml`, drawn by the new `tools/nb2_parity_data_draw.jl` on Julia 1.10.12,
  and a covariate case `nb2cov_restart_d7.toml` (12 checks in total).
- **Developer parity check** `test/parity/test_nb2_finite_dispersion_parity.jl`, in
  `runparity.jl`'s developer cohort only. (i) `test/fixtures/nb2_interior_n200_seed46.toml`
  (n = 200, true r = 1): both engines converge with no boundary, logLik within 1e-6 relative,
  every r within 1e-3. This dataset was chosen because its interior maximum survives pushing any
  single trait or any pair to the boundary. (ii) NATIVE-06 data: both engines put traits 1 and 3
  at the boundary and agree on the rest. R's `r_gradient_max` is printed, not gated.
- **Receipts** `docs/dev-log/core070/nb2-boundary-screen-20260924/`: every screen script with its
  printed output, and a README saying which code each ran against.

## 3a. Decisions and Rejected Alternatives

- **B-lite over B-full.** B-full means editing `frozen-r070-contract.toml`, whose hash every
  receipt records and two evidence files pin, plus about 8 JSON registries and the formula and
  public-bridge NB2 cases. Left for the next contract revision.
- **Restart from the returned point over a new default start.** Both lift the stalls, but a new
  default start changes every fit. The restart only runs when a group reaches the boundary.
- **Reset each boundary group on its own as well as together.** An independent review found that
  resetting all at once loses a genuine boundary group (seed 52: −833.61 together, −833.18 with
  group 5 alone). One extra optimisation per boundary group, only on boundary fits.
- **Rejected: search for boundary maxima of traits that are not at the boundary.** The Laplace
  objective often has a higher point with one more trait at the Poisson limit (7 of 16 datasets,
  0.10 to 0.69). A near-exact AGHQ check (section 5) shows the Laplace gaps can be badly
  distorted in either direction, so a wider Laplace search would not reliably find a better
  answer. This is a question for Shinichi (section 10).
- **R's gradient is recorded, not gated.** gllvmTMB stops with `r_gradient_max` up to 2e-3 even
  where both engines agree to 1e-11 in logLik, and the value depends on the machine.
- **Covariate route included; other families not.** The sibling screen found the same stall in
  `fit_nb_gllvm_grouped_cov` (3 of 10), so it gets the restart here. Gamma (#479) and Beta (#480)
  have different bugs that need design choices; NB1 was clean; Tweedie was too slow to screen.

## 4. Files Touched

- `src/families/grouped_dispersion.jl` (helper, call site, docstring)
- `CHANGELOG.md`
- `tools/nb2_parity_data_draw.jl` (new)
- `test/fixtures/nb2_restart_seed46.toml`, `nb2_restart_seed52.toml`, `nb2cov_restart_d7.toml`, `nb2_interior_n200_seed46.toml` (new)
- `test/test_nb_boundary_restart.jl` (new), `test/runtests.jl`
- `test/parity/test_nb2_finite_dispersion_parity.jl` (new), `test/parity/runparity.jl`
- `docs/dev-log/core070/nb2-boundary-screen-20260924/` (new: scripts, outputs, README, `aghq/`, `siblings/`)
- `docs/dev-log/after-task/2026-09-24-nb2-boundary-restart-and-finite-parity.md` (this file)
- `docs/dev-log/check-log.md`, `AGENTS.md` (snapshot line)

## 5. Checks Run

Mac Studio, Julia 1.10.12, single-threaded; R side is gllvmTMB `b4d5fee64` built locally with
`tools/core070_build_oracle.py`. Time estimates were written before each run; every run finished
inside its estimate.

| Check | Result |
|---|---|
| Final-code screen, 16 NATIVE-06-design datasets (`nb2_final_screen.out`) | restart lifted 7 fits by 0.46 to 2.59 and lowered none; new fit never below gllvmTMB (worst −5e-10), above it on 6 (0.29 to 1.15); a higher single-trait boundary point under Laplace on 7 (0.10 to 0.69) |
| `test/test_nb_boundary_restart.jl`, first version, before the fix | 3 of 5 fail (logLik −836.924) |
| `test/test_nb_boundary_restart.jl`, final | 12 of 12 (covariate case: −853.0785 before, −851.7809 after) |
| Sibling screen, 5 fitters (`siblings/`) | NB2-cov same class (3 of 10); Gamma and Beta different bugs (#479, #480); NB1 clean; Tweedie not screened |
| `test/parity/test_x_covariate_parity.jl` against the frozen oracle, with the covariate restart | 65 of 65 |
| 12 test files that use either fitter, final code (with the covariate restart) | all pass: `test_nb_boundary_restart` 12, `test_nb_beta_x_identity` 14, `test_grouped_dispersion` 20, `test_nb_fit` 8, `test_fit_gllvm` 11, `test_aicbic_newfits` 18, `test_unified_api` 24, `test_known_sentinel_defects` 25 + 1 known broken, `test_grouped_hessian_consistency` 23, `test_bridge_grouped_dispersion` 129, `test_bridge_x` 192, `test_confint_family` 326 |
| n = 200 interior screen (`nb2_interior_screen.out`) | 3 of 8 datasets have an interior maximum that survives single-trait pushes |
| Chosen dataset, all 15 single and pair pushes (`nb2_pairs_check.out`) | every push 1.27 to 6.51 lower |
| Developer parity check, Mac | interior data: Δ logLik 4.95e-8 (2.4e-11 relative), every r within 1e-5, estimates 0.89 to 1.15 (true 1); NATIVE-06 boundary agreement passes |
| NATIVE-06 Julia fit before vs after | identical, logLik −820.5485342071867 |
| Totoro, Julia 1.10.12, R 4.5.3, Track A's oracle, fresh clone of PR head `025df3fca`, 4.2 min | core test 10 of 10; developer check passes with the Mac's numbers (interior data Δ logLik 4.9537e-8) |

**Laplace against near-exact integration (AGHQ), verified by an independent skeptic.** An
evaluator built from the package's own NB2 pieces reproduces the package Laplace value with one
node (|Δ| 4.5e-13) and converges by 15 nodes; a brute-force grid agrees to 3e-8, and the
skeptic reproduced the numbers with a separate evaluator.

- Seed 46: under Laplace the point with trait 3 at the Poisson limit leads by 0.317. Under AGHQ it
  leads by 0.0285 at the same points and by 0.0128 once both are re-maximised, a practical tie:
  trait 3's dispersion is effectively not identified from about r = 4.5 upward.
- Seed 52: the boundary point for trait 2 leads by 0.427 under Laplace and 0.419 under AGHQ, and
  AGHQ has no interior maximum. The interior Laplace point (−833.6085) is an artefact, and the
  new restart's boundary answer is the right one.
- The Laplace error is not one-signed: it overstates the seed-46 boundary point by 0.238 but
  understates the seed-52 boundary point by 0.236. Two datasets only; no general direction is
  established.

Scripts and logs: `docs/dev-log/core070/nb2-boundary-screen-20260924/aghq/`.

## 6. Tests of the Tests

- The seed-46 core test fails on the unfixed fitter (−836.924). The seed-52 test fails on the
  first version of the restart (−833.6085). The toy "genuine boundary" test fails if the restart
  is always kept; the toy "stall" test fails if it is never run.
- The developer check's boundary-agreement testset passes only because both engines reach the
  boundary on NATIVE-06 traits 1 and 3.

## 7a. Issue Ledger

- #477: fixed for `fit_nb_gllvm_grouped` and `fit_nb_gllvm_grouped_cov`; sibling results posted
  on the issue.
- #479 (Gamma inner mode search diverges; converged reported 43 units low) and #480 (Beta
  converged reported at a non-stationary point): filed, not fixed here.
- #476: B-lite carried out; B-full deferred.
- gllvmTMB stops below Julia's new fit on 6 of 16 datasets (0.29 to 1.15); recorded in #477 for
  its owners, not changed here.

## 8. Consistency Audit

- Callers of `fit_nb_gllvm_grouped`: `fit_gllvm`, `src/bridge.jl:1249`, and the bootstrap in
  `src/confint_family.jl:615`. All get the restart; the bridge and confint files are in the sweep.
  A bootstrap replicate that reaches the boundary now runs one to three more optimisations.
- The boundary-flag tests use `fit_nb_gllvm_grouped_cov`, which is unchanged.
- The frozen NATIVE-06 cell still asserts `jl_fit.converged`, which its genuine boundary makes
  false; unchanged by design under B-lite.

## 9. What Did Not Go Smoothly

- My first version claimed the fix reaches "the optimum" and that the seed-46 data has finite
  dispersion at its maximum. An independent review (three reviewers, each finding challenged by a
  skeptic) showed both were false: the likelihood has several maxima. The tests, fixture names,
  CHANGELOG and this report were rewritten, and the screens were committed as receipts.
- The first reading of a post-fix screen called seeds 51 and 52 Julia stalls; a signed comparison
  showed gllvmTMB was the one stuck.
- The AGHQ agent first said Laplace "systematically favours the boundary"; its skeptic showed the
  error changes sign between the two datasets.
- Stacking #478 on #475 and merging #475 with `--delete-branch` closed #478. It was reopened by
  restoring the base branch at its merged commit, retargeting to `main`, and deleting it again.

## 10. Known Residuals

- **Decision for Shinichi:** should the NB2 fit go further than the restart? The AGHQ check says
  the Laplace ranking of these maxima can be off by about 0.3 in either direction, so a wider
  Laplace search is not the honest route; an NB2 AGHQ fit would be. The evaluator is in the
  receipts folder.
- A fit that reaches the boundary now costs one to three more optimisations, including inside
  the bootstrap.
- Gamma (#479) and Beta (#480) are real correctness bugs in default or bridge routes and need a
  design decision. Tweedie is unscreened and never flags a dispersion boundary.
- The full `Pkg.test()` suite was not run locally; CI shards cover it.

## 11. Team Learning

Before calling either engine stuck, compare signed log-likelihoods and evaluate each engine's
objective at the other's answer. Before calling a point "the optimum", push each parameter that
can run to a boundary and re-optimise: small-data GLLVM likelihoods are often multimodal. And a
Laplace objective can rank those maxima differently from the exact likelihood, in either
direction.

## 12. Cross-Product Coverage

The restart is cross-cutting for every caller of the NB2 grouped fitter.

- Covers ✓: `fit_gllvm` NB default route, the R bridge's NB route, the confint bootstrap refits.
- Covers ✓ also: `fit_nb_gllvm_grouped_cov` (formula route with NB2 and covariates, bridge NB2
  with X, the covariate confint bootstrap).
- This change does NOT cover: the Gamma, Beta, NB1 and Tweedie grouped fitters, the frozen
  NATIVE-06 cell, boundary maxima of traits not already at the boundary, or gllvmTMB's own
  stalls.

Memory receipt: `route.py` has no LOAD-FIRST manifest for this repo. Applied: this repo's
`AGENTS.md` (tests with the change, docstring and CHANGELOG, no widened tolerance), a lane lease on
the touched paths (D-88), and a time estimate before each run (D-139).

Golden Set: not in scope. No memory or retrieval class was touched.
