# After-task: Beta grouped fits report converged only at a stationary point (#480, 2026-09-24)

Lane: `claude/beta-convergence-480`, from `origin/main` @ `ef2287317`. Local commits only:
not pushed, no PR. Issue: #480.

## 1. Goal

`fit_beta_gllvm_grouped` is the default route for `fit_gllvm(...; family = Beta())`. It could
report `converged = true` at a point that is not stationary. On the sibling-screen dataset d05
(true φ = 2), Optim stopped after a zero-length line-search step. Optim counts that as "the
objective did not change" (`f_converged`) and so reports success, but the gradient residual was
9.06 against `g_tol = 1e-5`. The public fit returned logLik 265.976379 with `converged = true`
and φ[5] = 36.17. A fresh start with every log φ = 0 reaches 272.609413, where the largest
gradient component is 6e-6.

Required outcome: (1) never report `converged = true` unless the optimizer's gradient
criterion is met; (2) when a run stops without it, restart once from the warm start with every
log φ = 0 and once from the returned point, and keep the best run only if it lowers the
negative log-likelihood by more than 1e-6. The public d05 fit must reach at least
272.609413 - 1e-5. Fits whose first run already meets the gradient criterion must be unchanged.
The shared `_fit_verdict` must not change.

## 2. Implemented

- `src/families/grouped_dispersion.jl`, Beta section only:
  - New helper `_beta_grouped_gradient_restart(negll, res, θ_warm, ls, opts, first_log_phi)`.
    If `Optim.g_converged(res)` it returns `res` itself. Otherwise it runs two more optimizations
    with the same line search and options: one from the warm start with the log φ block set to
    0, and one from the returned point. It keeps the lowest run, but only if that run lowers the
    minimum by more than 1e-6.
  - `fit_beta_gllvm_grouped` and `fit_beta_gllvm_grouped_cov` call the helper after the first
    run, and then set `conv = conv && Optim.g_converged(res)` after `_fit_verdict`. The boundary
    flag still forces `converged = false` as before. For the cov route, the warm start used in
    the restart is `γ = 0` (as in the original start) with every log φ = 0.
  - Docstrings of both fitters and both result structs now say what `converged` means, and
    that `iterations` counts the kept run only.
- `test/test_beta_grouped_convergence.jl` (new, 19 assertions), wired into `test/runtests.jl`
  after `test_grouped_dispersion_beta_gamma.jl`.
- `test/fixtures/beta_grouped_screen_d05.toml` and `..._d01.toml`: the screen datasets stored
  as data plus a SHA-256 of the Float64 bytes, with the Julia and Distributions versions.
- `tools/beta_grouped_screen_data_draw.jl`: redraws a screen dataset and writes the fixture;
  it checks that the TOML text gives back the same bits.

Result on d05, public route: logLik 265.976379 with `converged = true` became 272.609413 with
`converged = true`, and φ became [2.576, 2.888, 1.783, 2.079, 1.723]. The largest
central-difference gradient went from 5.985 to 6.3e-6. The kept run is the log φ = 0 restart,
at 50 iterations. The cov route with an all-zero covariate went from 266.008692 with
`converged = true` to the same 272.609413.

## 3a. Decisions and Rejected Alternatives

- **Gradient test: `Optim.g_converged`, as the brief specified.** In Optim 1.13.3 this is
  `r.stopped_by.g_converged`, so it tests `g_residual <= g_tol` in absolute terms.
  **This choice has a cost that needs a decision; see section 10, item 1.** The alternative I
  measured, and did not commit, is the scale-aware rule already used by `_tweedie_verdict` in
  this file: `g_residual <= max(g_tol, g_tol * |nll|)`. That keeps the CI-required parity cell
  NATIVE-08-BETA unchanged and still catches #480 by three orders of magnitude (d05 threshold
  2.7e-3 against a residual of 9.06). The patch, relative to this branch's commit, is at
  `/tmp/claude-503/fix-beta-480/option-b-scale-aware-gradient.patch`. It changes the helper's
  first line and the two `conv = ...` lines, and adds a one-line helper
  `_beta_grouped_g_met(res, g_tol)`. With it, all 19 new assertions pass, and NATIVE-08-BETA
  and the Beta + X parity cell give `converged = true` on their own data.
- **The restart runs on any stop without the gradient criterion, iteration-limit stops
  included.** That follows the brief's wording. A caller who sets `iterations = 40` can now get
  up to three runs of 40. PR #478's NB2 restart gives each restart the same budget; I followed
  that.
- **`iterations` counts the kept run only**, as in PR #478.
- **`_fit_verdict` is untouched.** The extra condition sits in the two Beta fitters only.
- **The inner mode search, which is the likely root cause, is not changed.** See section 8. A
  fix there would change the objective value for every Beta grouped fit, so it needs its own
  issue and its own evidence.
- **The helper sits in the Beta section, away from PR #478's NB2 hunks.** `git merge-tree`
  between this commit and `origin/claude/nb2-finite-dispersion-parity-20260924` is clean,
  including the `runtests.jl` line.
- **The cov fitter is changed too, because it is the same code path**: the same optimizer, the
  same Beta marginal, and the same verdict. Its test uses `X = zeros(p, n, 1)`. Then the offset
  Xγ is exactly zero, so the objective is the no-X objective plus a coordinate with zero
  gradient, and the d05 stall shows up there as well (266.008692, `converged = true`, before the
  fix).
- **The fixture stores the draw, not the seed.** Beta draws depend on the Julia and
  Distributions versions. The draw tool is committed so the fixture can be traced. The harness
  that found the bug lives in `/tmp` and will not last.
- **Rejected: a quadratic as the helper's unit-test objective.** It could not catch an "always
  restart" mutant, because on a quadratic every restart finds the same minimum (section 6).

## 4. Files Touched

- `src/families/grouped_dispersion.jl` (Beta section: new helper, two fitters, four docstrings)
- `test/test_beta_grouped_convergence.jl` (new)
- `test/runtests.jl` (one `_shard_include` line)
- `test/fixtures/beta_grouped_screen_d05.toml` (new)
- `test/fixtures/beta_grouped_screen_d01.toml` (new)
- `tools/beta_grouped_screen_data_draw.jl` (new)
- `docs/dev-log/after-task/2026-09-24-beta-grouped-convergence.md` (this file)

Not edited, as instructed: `CHANGELOG.md`, `docs/dev-log/check-log.md`, `AGENTS.md`.

## 5. Checks Run

Mac Studio, Julia 1.10.12, `JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, test environment built
from `test/Project.toml` (Optim 1.13.3, Distributions 0.25.131), at most four runs at once.

The regression test on unfixed code (`origin/main` source) fails 5 assertions and throws 1
error: d05 logLik 265.976379 against the bound 272.609403; d05 gradient 5.985; cov logLik
266.008692; cov gradient 3.1e5; d01 `g_tol = 1e-9` reports converged; the helper is not defined.
After the fix it passes 19 of 19 (25 s).

Existing test files that exercise the Beta grouped fitters (grep for `fit_beta_gllvm_grouped`
and `BetaGroupedFit`), plus `test_bridge_x.jl` and `test_fit_gllvm.jl`, all on the fixed code:

| File | Result | Time |
|---|---|---|
| test_beta_fit.jl | 8 / 8 | 15 s |
| test_grouped_dispersion_beta_gamma.jl | 24 / 24 | 18 s |
| test_grouped_hessian_consistency.jl | 23 / 23 | 95 s |
| test_nb_beta_x_identity.jl | 14 / 14 | 10 s |
| test_aicbic_newfits.jl | 18 / 18 | 29 s |
| test_bridge_grouped_dispersion.jl | 129 / 129 | 35 s |
| test_bridge_x.jl | 200 / 200 (8 + 192) | 105 s |
| test_fit_gllvm.jl | 11 / 11 | 32 s |

Wider set that reaches the Beta route through the bridge, the formula route, `cv_gllvm` or the
latent-score path, also on the fixed code: test_bridge_capabilities.jl 242 / 242,
test_bridge_missing_mask.jl 92 / 92, test_cv.jl 62 / 62, test_lv_ci.jl 196 / 196,
test_formula_input.jl 28 / 28, test_core070_link_boundaries.jl 21 / 21,
test_bridge_lv_predictor.jl 207 / 207. `test_confint_family.jl` was not run: it fits no Beta
grouped model (checked by grep), and it takes about 8 minutes.

Not run: the R-driven parity files `test/parity/test_beta_parity.jl` and
`test/parity/test_x_covariate_parity.jl` (they need RCall and the pinned gllvmTMB twin library,
which is not staged on this machine). Their Julia side was checked by hand: the data-generating
code was pulled from the fixture files and fitted with the fixed and unfixed source. The results
are in section 10.

Probes (scripts and outputs in `/tmp/claude-503/fix-beta-480/probe/`):

- `before.out` / `after.out`: all ten screen datasets. Every dataset whose first run met the
  gradient criterion (d01 to d04, d07, d08) has the same logLik and the same iteration count
  before and after. d05 moves from 265.976379 to 272.609413. The boundary datasets (d06, d09,
  d10) keep the same logLik (the restarts were not kept) and still report `converged = false`,
  but take about 9 s instead of about 4 s.
- `gating.out`: d01 with `g_tol` 1e-9, 1e-11, 1e-13. Optim stops on "objective did not change"
  with residual 4.4e-7 each time. The restarts do not do better, and the fit reports
  `converged = false`.
- `native08.out` and `parity_native*.out`: see section 10.

## 6. Tests of the Tests

Each mutant was run against the new test file, then the committed source was restored:

| Mutant | Result |
|---|---|
| M0: unfixed source | 5 failed, 1 error |
| M1: gating line removed, restart kept | 1 failed (d01 `g_tol = 1e-9` reports converged) |
| M2: restart removed, gating kept | 6 failed (d05 and cov logLik, converged, gradient) |
| M3: pass-through removed (always restart) | 1 failed (helper returns a new run on the two-well objective) |

The first version of the helper test used a quadratic, and M3 passed it. It was replaced with a
two-well objective: the first run converges in the upper well, and a restart with log φ = 0
would reach the lower well. The test also checks that the lower well really is lower by more than
1, so a pass cannot be an accident of the objective.

The independent gradient check in the test uses the public
`beta_grouped_marginal_loglik_laplace` with central differences, not Optim's own bookkeeping.
It reads 5.985 on the unfixed d05 fit and 6.3e-6 on the fixed one.

## 7a. Issue Ledger

- #480: fixed on this branch (not pushed, no PR).
- New, needs a decision before a PR: the CI-required parity cell NATIVE-08-BETA would turn red
  (section 10, item 1).
- New, for a separate issue: `fit_gamma_gllvm_grouped` has the same weakness, measured by the
  sibling screen (section 8).
- New, for a separate issue: the inner site mode search can run out of iterations while
  oscillating, and says nothing (section 8).
- Known, not reopened: `_fit_verdict` accepts Optim's `converged`, which counts `f_converged`
  and `x_converged` (93 call sites). The class is wider than Beta; see section 8.

## 8. Consistency Audit

Sibling code paths checked for the same weakness (converged reported without the gradient
criterion):

- `fit_beta_gllvm_grouped_cov`: the same optimizer path. It showed the stall on d05 with X = 0
  (266.008692, `converged = true`, central-difference gradient 3.1e5 because the objective is
  jagged there). **Fixed in this change.**
- `fit_gamma_gllvm_grouped`: **same weakness, not fixed** (a different fitter, not the same code
  path). Sibling screen `/tmp/claude-503/sibling-screen/gamma*.out`, dataset 2 (α = 2): the
  public fit reports `converged = true` after 2 iterations, with gradient residual 53.4 and a
  zero-length step, at logLik -610.2247. A fresh start with log α = 0 reaches -567.2326 (gain
  43.0), where the residual is 6e-6. The same restart-and-gate would very likely fix it, but it
  needs its own test.
- `fit_gamma_gllvm_grouped_cov`, `fit_nb1_gllvm_grouped(_cov)`, `fit_nb_gllvm_grouped(_cov)` and
  the beta-binomial grouped fitters all end in `_fit_verdict(res)`, so they carry the same
  acceptance of `f_converged`. The NB1 screen found two interior local optima (gains 0.064 and
  1.087 from a different start) with both runs reporting converged. That is a separate
  "several maxima" class, and the screen did not check the gradient criterion. NB2 is PR #478's
  lane, and I stayed out of it.
- Tweedie grouped already gates on the gradient through `_tweedie_verdict` (the scale-aware
  rule above). Its screen did not finish within its compute cap.
- The shared-precision route (`fit_beta_gllvm`, used by the bridge's X_lv path): **clean on
  this screen.** On all ten datasets it reports converged with a central-difference gradient
  below 1e-5 (`probe/shared_beta.out`). It uses an analytic gradient by default. It still ends in
  `_fit_verdict`, so it is not protected against the class. `fit_gllvm_cov` (shared φ plus X) and
  the joint `GroupedNonGaussianFit` Beta route were not measured.
- The inner site mode searches (`_beta_grouped_loglik_site`, and the NB, Gamma and NB1 siblings
  with the same loop) use a Fisher-scored Newton step with no step control. When they reach
  `maxiter`, nothing is reported. At the stalled d05 point, site 12 hits 100 iterations with
  last step 0.105, and the objective value there is 266.0087 against 265.9763 from a tight inner
  solve (`probe/cov_diag.out`). The Gamma skeptic found the same at its stalled point (site 7).
  This is the likely reason the line search could only take a zero-length step. Not fixed.
- Bootstrap refits in `src/confint_family.jl` (about lines 705 and 880) call both Beta grouped
  fitters, so replicates now get the restart as well. `test_grouped_hessian_consistency.jl`
  passes.
- Docs: `src/families/fit_gllvm.jl` and `src/formula.jl` only name the fitters and make no
  convergence claim, so they need no change. `CHANGELOG.md` is left to the orchestrator.

## 9. What Did Not Go Smoothly

- I found the NATIVE-08-BETA flip only after the commit, and only because I rebuilt the opt-in
  parity cells' data by hand. The core suite cannot see it, because it asks for the default
  `g_tol = 1e-5`.
- I expected the cov route with X = 0 to follow the no-X route exactly. It did not (266.0087
  against 265.9764, 40 iterations against 26), although it lands in the same region. The test
  does not rely on the two routes being identical. I did not chase the reason.
- The first helper test could not catch the M3 mutant (section 6).
- `timeout` is not on macOS, so the long runs were bounded by the tool's own time limit instead.

## 10. Known Residuals

1. **The CI-required parity check NATIVE-08-BETA will very likely fail with this commit.**
   `test/parity/poisson_beta_health.jl` checks `native_converged` for
   `fit_gllvm(Y; family = Beta(), K = 1, g_tol = 1e-7, iterations = 800)`, and CI runs it on
   pull requests (`test-parity` job, `CORE070_PARITY_REQUIRED=1`). Julia side measured on this
   machine, on Julia 1.10.12 and on 1.13.0 (both draws give logLik 173.070526375): unfixed,
   `converged = true`; fixed, `converged = false`, at the same point and logLik. The reason is
   that the first run stops on "no x change" with gradient residual 3.0e-7, which is above the
   requested 1e-7. That residual is at the finite-difference noise floor: the central-difference
   gradient is also 3.0e-7, and both restarts end at the same value with residuals 4.0e-7 and
   3.0e-7. So the point is stationary, and `converged = false` here is a false alarm caused by a
   `g_tol` below what finite differences can deliver. I did not edit the parity cell or its
   pinned contract. Options: (a) adopt the scale-aware rule (patch in section 3a); (b) keep the
   strict rule and change the pinned cell's `g_tol`, which is a contract change for the
   maintainer; (c) keep the strict rule and accept a red required check, which is not
   recommended. The Beta + X parity cell (default `g_tol`) stays `converged = true` either way.
2. Cost: fits that stop without the gradient criterion now run two more optimizations. On the
   boundary datasets this about doubles the time (about 4 s to 9 s), with no change in the
   result. Iteration-capped fits can use up to three times the cap.
3. The restart improves the local search. It does not promise the global maximum. These
   likelihoods have several stationary points: on d03 the default start converges (gradient
   criterion met) at 245.624 with φ[1] = 604.5, while the log φ = 0 start converges at 244.243.
   A fit that meets the gradient criterion is never restarted, so a converged but lower mode
   would be kept.
4. The inner mode search, the Gamma grouped sibling, and the `_fit_verdict` class are unfixed
   (section 8).
5. The parity files were not run against R.

## 11. Team Learning

- `Optim.converged(res)` is true after a zero-length step, because `f_converged` fires when f
  does not change at all. For any fitter that claims a stationary point, read
  `Optim.g_converged(res)` or `Optim.g_residual(res)`.
- An absolute `g_tol` below the finite-difference noise floor (about 3e-7 on these Beta
  objectives) cannot be met, so a strict gradient gate turns stationary fits into "not
  converged". Scale the tolerance, as `_tweedie_verdict` does, or check the callers' `g_tol`
  before tightening a verdict.
- Test a "leave it alone if it converged" branch with an objective where a restart would change
  the answer. Otherwise an "always restart" mutant passes.
- Before claiming the core suite is clean, check the opt-in parity cells that CI requires. Their
  kwargs can differ from every core test.

## 12. Cross-Product Coverage

- Covers: `fit_beta_gllvm_grouped` and `fit_beta_gllvm_grouped_cov`, and so every route into
  them: `fit_gllvm(...; family = Beta())`, the bridge's "beta" rows without and with fixed-effect
  X (masked no-X included), the formula route, `cv_gllvm`, and the bootstrap refits in
  `confint_family.jl`. Both `hessian` settings go through the same new lines, but only
  `:observed` was exercised.
- This change does NOT cover: the Gamma, NB1, NB2, beta-binomial and Tweedie grouped fitters;
  the shared-precision `fit_beta_gllvm`; `fit_gllvm_cov`; the joint `GroupedNonGaussianFit` Beta
  route; the inner mode search; non-logit links; datasets with an offset or a mask in the new
  test; or any run of the R parity suite.

Memory receipt: `route.py` has no LOAD-FIRST manifest for this worktree path, so none was
loaded. Applied: this repo's `AGENTS.md` and `CLAUDE.md` rules as relayed in the brief (own
worktree, stage by name, no push, no PR, test first, no widened tolerance), and
`lane_preflight.sh`. It flagged PR #478's branch on this file; I read its diff and checked the
merge with `git merge-tree` (clean). I made a time estimate before each run and kept every run
under 15 minutes (D-139).

Golden Set: not in scope. No memory or retrieval class was touched.
