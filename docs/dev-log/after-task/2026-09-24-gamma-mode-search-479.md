# After-task: Gamma grouped mode search is damped and fails loudly (#479, 2026-09-24)

Lane: Claude, branch `claude/gamma-mode-search-479`, worktree `~/local-scratch/gllvm-gamma-479`. It was cut from `origin/main` @ `ef2287317` and rebased onto `d9bc77412` (the #478 merge) before finishing. Local commits only: nothing pushed, no PR.

## 1. Goal

Fix #479. `fit_gamma_gllvm_grouped` reported `converged = true` after 2 iterations on the sibling screen's seed-2 dataset (true alpha = 2, one group per trait). Its log-likelihood was -610.224727, while a fresh start with log alpha = 0 reaches -567.232613.

The cause: the per-site mode search in `_gamma_grouped_loglik_site` took full Fisher-scoring steps with no damping. At the fitter's own warm start it diverged on site 7 (z about 1e11). It still returned a FINITE site log-likelihood of about -1.4e23, so the objective's 1e12 sentinel never fired.

Required outcomes:
- The search must never silently return a value from a diverged or unconverged search.
- The seed-2 fit must reach -567.232613 - 1e-5, or must not claim convergence.
- The converged value must stay the same (to 1e-8) wherever the old loop converged.

## 2. Implemented

- New helper `_gamma_grouped_mode` in `src/families/grouped_dispersion.jl` (Gamma section only). It returns the mode and a converged flag.
  - A step that lowers the per-site log-posterior is halved (up to 30 times). This is the same rule the generic core `_laplace_mode` uses, so small steps and accepted full steps are bit-identical to the old loop.
  - "Converged" means the full proposed step is below `tol`, the old loop's own stopping test. A heavily halved step does not count.
- `_gamma_grouped_loglik_site` now works in three stages:
  1. The Fisher-scored search runs first, as before.
  2. If it did not converge, and only under `LogLink`, a damped Newton search runs from z = 0 on the observed curvature alpha*y/mu. That weight is never negative for valid Gamma data, so `Lambda'W Lambda + I` stays positive definite.
  3. If neither converged, the site returns `-Inf`. The fitters' objective then returns its 1e12 sentinel, and `_fit_verdict` reports `loglik = -Inf`, `converged = false` if the fit ends there.
- One docstring sentence on `gamma_grouped_marginal_loglik_laplace` states the new `-Inf` behaviour.
- Regression test `test/test_gamma_grouped_mode_search.jl` (18 assertions) with fixture `test/fixtures/gamma_grouped_mode_search_seed2.toml`. The fixture holds the data, its sha256 and the draw recipe. The test is wired into `test/runtests.jl` next to `test_gamma_x_identity.jl`.

Result on the seed-2 dataset:

| Fit | Before | After |
|---|---|---|
| `fit_gamma_gllvm_grouped` (group = 1:p) | -610.224727, 2 iterations, converged = true | -567.232613, 40 iterations, converged = true |
| `group = fill(1, p)` (the bridge's no-X Gamma route) | -610.109101, 2 iterations | -569.700833, 24 iterations |
| `fit_gamma_gllvm_grouped_cov` with X = 0 | -610.224727, 2 iterations | reaches -567.232613 (test passes) |
| `hessian = :fisher` | -609.062376, 2 iterations | -568.247447, 44 iterations |

The group = fill(1, p) value matches the shared-shape `fit_gamma_gllvm`, which uses a different kernel: -569.700833 on the same data.

## 3a. Decisions and Rejected Alternatives

- **Chosen: Fisher first (with halving), then Newton as a fallback, then -Inf.** Measured before choosing, on 30 probe points (10 screen datasets, 3 parameter points each):
  - Fisher + halving alone left sites unconverged at 100 iterations at 5 of the 30 points.
    - Two of those were healthy datasets (4 and 5) at the warm start. A bare -Inf rule would have turned those working fits into failures.
    - The #479 site did not converge even after 2000 halved Fisher iterations.
  - Damped Newton on the observed curvature converged at every site of all 30 points, in at most 6 iterations.
- **Rejected: Newton only.** It changes the path on every site. That leaves residuals around 1e-10 against identity tests that compare the grouped kernel with the shared Gamma kernel at atol 1e-10. With Fisher first, every site that converged before follows the old path exactly.
- **Rejected: the generic core's convergence test** (step taken times |step| below tol). It would count a heavily halved step as converged.
- **Newton fallback only for LogLink.** Only there is the observed weight provably never negative. Other links go straight to -Inf after a failed Fisher search.
- **Siblings not fixed** (details in section 8). Each has its own copy of the loop, so none shares this code path:
  - The NB2 kernel belongs to open PR #478.
  - The Beta section belongs to the #480 lane (`claude/beta-convergence-480`).
  - NB1, Tweedie, `_laplace_mode_off`, `_grouped_laplace_mode` and the generic `_laplace_mode` are separate functions.
  - The routes that do share the Gamma kernel are fixed by this change: `fit_gamma_gllvm_grouped_cov`, the bridge's G = 1 route, and confint's grouped Gamma objective.
- **No #480-style gradient-criterion guard for Gamma.** #479 is fixed at its root: the objective is no longer garbage. The guard is a separate fitter-level policy.
- **Include placement.** First placed after `test_grouped_dispersion_beta_gamma.jl`, where the #480 branch inserts its own line; a trial merge showed a conflict. Moved to after `test_gamma_x_identity.jl`, also a grouped-Gamma test.

## 4. Files Touched

- `src/families/grouped_dispersion.jl`, Gamma section only:
  - new `_gamma_grouped_mode`;
  - `_gamma_grouped_loglik_site` loop replaced;
  - one docstring sentence.
- `test/test_gamma_grouped_mode_search.jl` (new)
- `test/fixtures/gamma_grouped_mode_search_seed2.toml` (new)
- `test/runtests.jl` (one `_shard_include` line)
- this report

Not touched: `CHANGELOG.md`, `docs/dev-log/check-log.md`, `AGENTS.md`, any NB2 code.

Outside the repo:
- a read-only baseline worktree `~/local-scratch/gllvm-gamma-479-base` (detached `origin/main`), used only for before/after runs;
- scratch scripts and outputs in `/tmp/claude-503/fix-gamma-479/`.

## 5. Checks Run

All runs used Julia 1.10.12, single-threaded, with the test environment built from `test/Project.toml`.

- New test: 18 of 18 pass on the final commit. It ran twice, once before and once after the docstring-only edit and the include move.
- Existing tests that reach the Gamma grouped code, all passing on the fix:

| Test file | Assertions |
|---|---|
| `test_grouped_dispersion_beta_gamma.jl` | 24/24 |
| `test_gamma_x_identity.jl` | 7/7 |
| `test_gamma_curvature_cross_kernel.jl` | 14/14 |
| `test_grouped_hessian_consistency.jl` | 23/23 |
| `test_aicbic_newfits.jl` | 18/18 |
| `test_bridge_grouped_dispersion.jl` | 129/129 |
| `test_exponential.jl` | 20/20 |
| `test_bridge_missing_mask.jl` | 92/92 |
| `test_bridge_x.jl` | 192/192 |
| `test_bridge_capabilities.jl` | 242/242 |
| `test_bridge_lv_predictor.jl` | 207/207 |

  No existing test changed outcome and no tolerance was touched.
- Sibling screen `gamma.jl` (10 datasets) re-run on the fix:
  - dataset 2 is fixed: -567.232613, 40 iterations;
  - datasets 1, 3 to 7, 9 and 10 give the same log-likelihood to the printed 6 decimals, and 0 stalls;
  - iteration counts moved on dataset 7 (350 to 423) and dataset 10 (368 to 351);
  - dataset 8 changed. It is explained in section 10.
- Julia side of the R parity cells, before and after (baseline worktree against the fix): Gamma no-X seed 54 (G = p and G = 1) and Gamma + X seed 46. The log-likelihood and iteration count are identical to 12 decimals. The R parity suite itself was not run (section 10).
- Trial merges (`git merge-tree`): clean against #478, against the #480 branch, and against both together.
- #478 merged into `main` during the task. After rebasing onto it, these were re-run on the combined code, all passing: the new test (18/18), `test_grouped_dispersion_beta_gamma.jl` (24/24), `test_gamma_x_identity.jl` (7/7), `test_gamma_curvature_cross_kernel.jl` (14/14), `test_grouped_hessian_consistency.jl` (23/23), and #478's own `test_nb_boundary_restart.jl` (12/12). The other results in this section are from the pre-rebase code, which differs only in the NB2 section, and that section is not on any Gamma path.

## 6. Tests of the Tests

- **The test fails on the unfixed code.** 11 of 18 assertions fail, measured by the independent
  reviewer against the verbatim `origin/main` kernel. (My first run showed 12; the twelfth came
  from a mis-set guard in an earlier draft of the test, since corrected.) The failures:
  - site 7 is off by 7.2e22 from the independent evaluator;
  - `maxiter = 2` returns finite values instead of -Inf;
  - the `newton_maxiter = 2` fit reported -567.572 as if converged;
  - the three public fits sit at -610.2247, -610.1091 and -610.2247.

  Output: `/tmp/claude-503/fix-gamma-479/prefix_test_run.out` (scratch).
- **The old-loop copy in the test really is the old loop.** On the unfixed package it matched `_gamma_grouped_loglik_site` bit-for-bit at 420 of 420 site evaluations, converged or not. So the "unchanged where the old loop converged" check compares against the true pre-fix values. The old loop converged at 418 of those 420, and the new kernel agrees at every one of them to under 1e-8.
- **The independent site evaluator shares no code with the kernel.** It uses `Distributions.logpdf`, the analytic Gamma/log score and observed curvature, and its own damped Newton to 1e-13. It agrees with the fixed kernel at all 80 warm-start sites to under 1e-8.
- **One assertion I wrote was wrong and was corrected before use.** The guard against an empty comparison said "of 480 site evaluations"; there are 420 (3 x 80 + 3 x 60). It was set to `>= 400` with the measured 418 in the comment. It does not depend on the fix.
- **The dataset 8 change was checked rather than accepted:**
  - At both the old and the new optimum, the old and new kernels agree exactly, and the old loop converged at every site.
  - The profile in alpha_1 keeps rising: 62.038551 at 1e5, 62.040280 at 6.28e5, 62.040541 at 3.2e6, 62.040605 at 1e8.

## 7a. Issue Ledger

- #479: addressed on this local branch. Not closed and not commented on (no push, no PR).
- Found in passing, for the orchestrator to file or route. None is fixed here.
  1. `fit_gllvm_cov(Y; family = Gamma(...), X, K)` throws `DomainError` (Gamma alpha > 0) on the #479 data, with X = 0 and with a random covariate, before and after this fix.
     - Its objective at its own warm start is -7.2e22: the same undamped-loop divergence, in `_laplace_mode_off` (`src/families/covariates.jl`).
     - The line search then jumps to an extreme log alpha, and `_cov_family` builds `Gamma(0.0, 1.0)` outside the objective's `try`.
  2. The Beta grouped kernel silently returns unconverged values. That is 7 of 3600 site evaluations at stress points (loadings x3 or x5, precision x0.25 or x4), off by up to 0.112 log-likelihood units per site. NB2 had 1 (1.15e-8); NB1 and Tweedie had 0.
  3. The `gamma_grouped_marginal_loglik_laplace` docstring still says `hessian=:fisher` is "the default"; the code default is `:observed`. This predates the change and was left alone.

## 8. Consistency Audit

The same weakness (a loop with no convergence flag that returns whatever `z` it holds) was checked in every sibling path:

| Path | Damping | Convergence reported | Measured | Action |
|---|---|---|---|---|
| `_nb_grouped_loglik_site` (NB2) | none | no | 1/3600 slow, 1.15e-8 | left to PR #478 |
| `_beta_grouped_loglik_site` | none | no | 7/3600, up to 0.112 | reported (item 2) |
| `_nb1_grouped_loglik_site` | none | no | 0/3600 | reported |
| `_tweedie_grouped_loglik_site` | none | no | 0/3600 | reported |
| `_laplace_mode_off` (covariates.jl: `fit_gllvm_cov`, constrained ordination, fourth-corner, covariate getLV) | none | no | diverges at the #479 warm start (-7.2e22) | reported (item 1) |
| `_grouped_laplace_mode` (getLV for every grouped family) | halving | no | #479 warm-start site 7 stops 1.8e-4 short of the mode; exact (0.0) at the fixed optimum | reported |
| generic `_laplace_mode` (shared `fit_gamma_gllvm`, the bridge's X_lv Gamma route) | halving | no | shared fit reaches -569.700833 on the #479 data | reported |
| Routes that call the fixed Gamma kernel (`fit_gamma_gllvm_grouped_cov`, bridge G = 1, `confint_family.jl` grouped Gamma objectives) | fixed here | yes | tests pass | fixed by this change |

Stale comments now partly out of date:
- `src/families/exponential.jl` says "The undamped per-site loops remaining in `grouped_dispersion.jl` are recorded engine debt (Arc 2)". That is still true for NB2, Beta, NB1 and Tweedie; Gamma is no longer undamped.
- `test/test_exponential.jl` (around line 38) describes `_gamma_grouped_loglik_site` as "NOT FIXED". Exponential no longer routes through it, so the comment is history, not a live claim.

Neither comment was edited.

## 9. What Did Not Go Smoothly

- My first plan (Fisher + halving, then -Inf) failed its own measurement. It would have broken healthy fits at their warm start, which is what led to the Newton fallback.
- One batch of existing tests "ran" in 0 s: macOS has no `timeout` command, so nothing executed. I caught it from the wall-clock times and re-ran without it.
- The first include position conflicted with the #480 branch. A trial merge caught it and the line was moved.
- A scratch script that `eval`s the test helpers resolved `@__DIR__` to the wrong folder; fixed by substituting the path.

## 10. Known Residuals

- **Behaviour change near alpha = infinity.** On sibling dataset 8 (true alpha = 200), the fit moved from 62.040282 (converged = true, alpha_1 = 6.28e5) to 62.040541 (converged = false, `dispersion_boundary = [1]`, alpha_1 = 3.2e6).
  - The likelihood still rises as alpha_1 grows, so the new flag is the documented boundary rule working, and the old stop was arbitrary.
  - Still, a fit that used to report converged can now report a boundary. The same data, restarted from the old optimum, stays near 6.28e5 under both old and new code. The path, not the surface, decided it.
- **A `tol` below rounding level now fails loudly.** If a caller sets `newton_tol` below what floating point can certify, every site returns -Inf and the fit reports `converged = false`. Before, it returned a truncated value. The default 1e-9 is far above that floor.
- **Other links have no fallback.** Under a non-log link, a Fisher search that fails returns -Inf directly.
- **R parity suite not run.** The twin library path in the parity README (`/tmp/R-gllvmtmb-x-parity-20260802`) no longer exists. Instead, the Julia fits of the Gamma parity cells were shown identical before and after, to 12 decimals.
- **Sibling kernels keep the weakness** (section 8), including the throwing `fit_gllvm_cov` Gamma route.
- **Iteration counts changed** on some fits whose optimum did not (dataset 7: 350 to 423).
- **Runtime.** An independent reviewer measured healthy fits about 20 to 25 percent slower with
  the same log-likelihood and iteration count (for example 2.02 s to 2.53 s), from the extra
  log-posterior evaluation on large steps. Measured while another Julia job ran, so approximate.
- **One platform.** The public-fit assertions were run on macOS aarch64 with Julia 1.10.12; CI's
  Linux shards are the cross-platform check.

Review: an independent adversarial reviewer returned **ok-with-notes**. It reproduced the fix
on 21,200 stress-site evaluations (converged values unchanged to 6.6e-9, no new -Inf) and
raised the count, runtime and docstring notes now reflected here.

## 11. Team Learning

- A sentinel keyed on `isfinite` does not catch a finite garbage value. The inner solver must report whether it converged, and the caller must act on that report.
- Measure a safeguard before adopting it. Step-halving alone looked sufficient on paper; it left 5 of 30 probe points unconverged, including healthy datasets.
- Keeping the old path first, and adding a fallback only where it failed, gave bit-identical results wherever the old code was right. That protected every identity test without touching a tolerance.
- Trial-merge against every open branch that touches the same file before finishing, not only the one named in the brief. A second lane (#480) appeared mid-task.

## 12. Cross-Product Coverage

This work does NOT cover:
- the NB2, Beta, NB1 or Tweedie grouped kernels;
- `_laplace_mode_off` or the `fit_gllvm_cov` Gamma `DomainError`;
- `_grouped_laplace_mode` (getLV) or the generic `_laplace_mode`;
- non-log links, beyond returning -Inf when Fisher fails;
- a masked-data case where the fallback fires (masks pass through the same code, but no test forces the fallback under a mask);
- an offset case with a non-zero covariate effect (the covariate route was tested with X = 0 only);
- Wald, profile or bootstrap intervals for grouped Gamma under a failing site (the hessian-consistency test passed, but nothing forces a -Inf inside a finite-difference Hessian);
- Julia versions other than 1.10.12;
- the R parity suite.

It also does not claim the global maximum on any dataset, only that the seed-2 fit reaches the known better point.

Memory receipt: no shinichi-brain lookup was run in this slice. The inputs were the orchestrator's brief, the sibling-screen evidence in `/tmp/claude-503/sibling-screen/`, and repo sources: `exponential.jl` and `test_exponential.jl` already record this kernel's divergence as Arc 2 engine debt.

Golden Set: no memory-regression run. The applicable repeated-failure checks were run: failure recorded before the fix, old-loop copy verified bit-for-bit against the pre-fix code, trial merges against the open branches editing the same file, and no tolerance widened.
