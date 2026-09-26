# PR #502 review: is the convergence rule correct?

PR: itchyshin/GLLVModels.jl #502, branch `claude/fit-verdict-gradient-485`, head `802f3965e`.
Reviewer lane: `claude:GLLVM.jl:review502-correctness`. Read-only on GitHub. Throwaway worktree
`~/local-scratch/review-502-correctness` (detached at the PR head, removed after the review).
Toolchain: Julia 1.10, Optim 1.13.3, `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, Mac Studio.
Scripts and logs: `reviews/pr-502-correctness-evidence/`.

## Verdict: DO NOT MERGE (NOT READY)

The rule has the right form. It is character-for-character the rule the package already
ships in `_tweedie_verdict` and `_beta_grouped_g_met`, and it can only turn `converged`
from true to false. The problem is what it is fed. The gradient size it tests (`gres`) does
not reliably measure distance from the optimum. For several fitters it measures
finite-difference noise, a jump in the objective, or a finite-difference step that landed
on the 1e12 failure value. Measured on this branch:

1. **Existing tests that pass on main fail here.** Local runs: `test_twolevel.jl:107`,
   `test_phylo_poisson_xlv.jl:158`, `test_phylo_beta_xlv.jl:183`, plus
   `test_phylo_binomial_xlv.jl:182` and `test_phylo_gamma_xlv.jl:172`, which are in the
   repo but not listed in `runtests.jl`. Each passes when the same run uses main's
   verdict. CI shard 4/4 is red on both Julia 1.10.12 and 1.13.0. Shards 1 to 3 were
   still running when I wrote this.
2. **The PR's own new regression test fails on CI.** On Linux, `test_fit_verdict_gradient.jl:50`
   gets loglik −1038.6318 against the pinned −1038.5504 (both Julia versions). On 1.13.0,
   `:51` (`!fit.converged`) also fails: the NB1 fit converges there. So the test does not
   reliably reproduce the defect.
3. **Fits that really are at their optimum get flagged "not converged" (Gamma, shared shape).**
   `fit_gamma_gllvm` flips 5 of 20 simulated datasets on the finite-difference path. That
   path runs whenever there is an `offset` or `X_lv`, a non-default `hessian`, or
   `gradient=:finite`. The default analytic path flips 1 of 20. I checked that these points
   are genuine optima (section 3).

## 1. Is `gres <= max(g_tol, g_tol*|nll|)` the right scale rule?

**Same as the shipped rules?** Yes, exactly the same.
- `_tweedie_verdict` (`src/families/tweedie.jl:218`, `:233`):
  `isfinite(gres) && gres <= max(g_tol, g_tol * abs(nll))`.
- `_beta_grouped_g_met` (`src/families/grouped_dispersion.jl:762`) uses the same expression.
- Those two take `g_tol` from the caller. The new helper reads `Optim.g_tol(res)` instead.
  That equals the caller's value, because `Optim.Options(g_tol=x)` sets `g_abstol = x`
  (Optim `types.jl:115-117`) and `g_tol(r) = r.g_abstol` (`api.jl:159`).
- Both old rules are also ANDed with the optimiser's own convergence flag, as here.

**What Optim measures.** `r.g_residual = maximum(abs, gradient(d))`
(`utilities/assess_convergence.jl:13,15`).
- It is the largest absolute gradient component (the infinity norm) at the final state.
- It is taken in the optimiser's own coordinates: log α, log φ, log σ, packed Λ.
- It uses whatever gradient the optimiser had. That is a FiniteDiff central difference for
  `autodiff=:finite`, ForwardDiff for `:forward`, or the user-supplied gradient for
  `only_fg!`.
- `g_tol` comes from the result's own options.
- Nobody calls Fminbox. If someone did, `g_tol(res)` would return the *inner* `g_abstol`,
  not `outer_g_abstol`. That mismatch is harmless today.

**The rule never overrides a gradient stop.** `g_converged` means `gres <= g_abstol <= threshold`,
so every flip comes from a run that stopped on an x or f criterion. In practice, every fit
that stops because the gradient is small still passes. This is why the fits most at risk
are the ones that stall at the noise floor.

**At large |nll|.** With nll ≈ 3000 and g_tol = 1e-5, the threshold is 0.03.
- The rule only ANDs, so it cannot accept any fit that main rejected. It adds no new false
  "converged" results.
- It can still pass an x/f stall along a weakly curved direction. The loglik gap is about
  g²/(2h), where h is the curvature. With g = 0.03 and h = 1e-3, the gap is about 0.45
  loglik units.
- The deeper issue is units. |nll| grows with the total number of observations (N·p),
  while the curvature of a per-species or per-group parameter grows only with the
  observations that inform it. For wide data this makes the threshold roughly p times too
  loose for those parameters.
- A curvature-scaled test would be principled: the Newton decrement g'H⁻¹g, or lme4's
  `max|H⁻¹g|`. That is a follow-up, not a blocker.

**At tiny |nll|.** The threshold falls back to plain `g_tol`, and the rule depends on
arbitrary additive constants in the nll: dropped normalising constants, the data's units,
Jacobians.
- Test: I rescaled Gamma data so that the fitted nll was about 0 (`scale.jl`). This leaves
  the fit's geometry unchanged, because the log link just shifts β. The threshold fell from
  1.6e-2 to 1e-5, and gres/threshold rose from about 3e-4 to 0.30–0.99.
- Those runs survived only because they had stopped on the gradient.
- Healthy Poisson fits with p=15, n=300 stop on f with gres = 1.1e-5 to 4.8e-5, which is
  1 to 5 times `g_tol`. Any such fit with |nll| below about 5 would flip.

## 2. NelderMead callers

There are **8** NelderMead call sites, not 6 as the after-task report says:
- the 6 `phylo_*_xlv.jl` profile refits,
- `confint_family.jl:3352` (GLM profile),
- `grouped_nongaussian_fit.jl:841`.

**None of them feeds `_fit_verdict`.**
- The phylo refits compute `Optim.converged(last_res)` locally.
- `confint_family.jl` and `grouped_nongaussian_fit.jl` contain no `_fit_verdict` call at
  all. `grouped_nongaussian_fit.jl` has its own gradient test.
- If a NelderMead result ever did reach the rule, `g_residual` would be the spread of the
  simplex's function values (`state.nm_x`), not a gradient. NelderMead stops only through
  `g_converged` (`nelder_mead.jl:318`), so such a result would always pass. The check would
  be harmless but empty.

All 102 `_fit_verdict(res)` occurrences in `src/` use `Optim.LBFGS`.

## 3. Finite-difference gradients near a converged point (measured)

**Healthy fits** (`probe.jl`): 40 fits covering Poisson (FD and analytic), Binomial FD,
Gamma-grouped FD and Gaussian (ForwardDiff). Sizes were p=8, n=100 and p=15, n=300, with
4 seeds each.
- All 39 fits that main calls converged still pass. The largest gres/threshold ratio is 1.4e-3.
- One Binomial fit hit its 500-iteration limit (gres 0.17). It is unconverged under both
  versions.
- 10 of the 39 stopped on x or f with gres 1–5 times the raw `g_tol`. They pass only
  because of the |nll| scaling.

| family / path | nll | gres | threshold | stop (x/f/g) |
|---|---|---|---|---|
| Poisson FD, p8 n100 s4 | 1895 | 1.94e-5 | 1.90e-2 | x,f |
| Poisson FD, p15 n300 s1 | 9122 | 3.80e-5 | 9.12e-2 | f |
| Binomial FD, p15 n300 s1 | 3016 | 7.10e-6 | 3.02e-2 | g |
| Gamma-grouped FD, p15 n300 s1 | 9057 | 4.78e-5 | 9.06e-2 | f |
| Gaussian ForwardDiff, p15 n300 | ~5700 | ~5e-13 | ~5.7e-3 | g (0 iterations) |

**Gamma with a shared shape (`fit_gamma_gllvm`): good fits flagged as not converged.**
Flip counts (`gamma_rate.jl`): 5 of 20 seeds on the FD path (`offset = zeros`: seeds
1, 3, 7, 8, 14), and 1 of 20 on the default analytic path (seed 7). The flipped seeds
really are at their optimum (`gamma_check.jl`, `gamma_fd.jl`):
- The analytic and FD fits reach the same loglik to within 2e-4 on every flipped seed.
- Starting the FD fit from the analytic optimum stalls again with the same gres: 12.9 on
  seed 3, 1.6 on seed 7.
- A step along −g *increases* f at t = 1e-4, 1e-3 and 1e-2 on both seeds 3 and 7.
- On seed 7, the central-difference gradient grows as 1/h: 0.033, 0.12, 1.0, 9.8 and 98
  for h = 1e-3 down to 1e-7. The objective jumps by +1.97e-5 just to the right of log α.
  A step change in the objective, not a genuine slope, is producing the large "gradient".
  This matches the undamped inner mode search the after-task report lists in its ledger
  (item 1).

**Fits whose FD step hit the failure value.** On this branch, the phylo_*_xlv flips report
gres = 1.2e14 to 8e16. That is roughly 1e12/(2h): the FD stencil straddles the 1e12
failure value.
- `phylo_gamma_xlv` stopped at iteration 2 and `phylo_beta_xlv` at iteration 8, each
  warm-started at the truth. There the optimiser never really moved, so calling them
  "not converged" is arguably honest.
- The Poisson (124 iterations) and Binomial (91 iterations) fits did move before stalling,
  so their flips are ambiguous.
- `delta_gamma` (gres up to 1e32), `delta_disp_group` and `ordered_beta` (1e7) flip the same
  way without failing any test.
- Either way the PR is silent on this case. A gres above about 1e11/h means "the stencil
  hit the failure value", not "the gradient is large". It deserves its own reason code.

## 4. Test assertions near the threshold

I ran an instrumented sweep of 42 at-risk test files (`sweep.jl`), logging every
`_fit_verdict` call (291 verdicts). There were 19 flips.

I then re-ran the files that failed with main's verdict swapped in (`sweep_main.jl`). The
failures listed first below go away under main's verdict, so the PR causes them. The
second group fails either way.

**Caused by the PR (fail on the branch, pass under main's verdict):**
- `test/test_twolevel.jl:107` `@test fit.converged` (`fit_twolevel_gaussian`, ForwardDiff,
  default `g_tol = 1e-8` at `src/twolevel.jl:200`).
  - The whole replicate loop sits at the threshold: 25 of 26 fits stop on f with ratio
    0.1–0.52, and one reaches **1.13** (gres 1.57e-4 at nll 13830).
  - The gradient is exact here, and a relative gradient of 1e-8 means the fit is converged
    for any statistical purpose.
  - This is a wrong "not converged" caused by the caller's very tight `g_tol` together with
    an f stop the caller chose on purpose.
- `test/test_phylo_poisson_xlv.jl:158`, `test/test_phylo_beta_xlv.jl:183` (also red in CI on
  1.13.0), `test/test_phylo_binomial_xlv.jl:182`, `test/test_phylo_gamma_xlv.jl:172`.
  All are `@test fit.converged` on fits whose FD stencil hit the failure value (section 3).

**Fail under both verdicts (not the PR):** `phylo_binomial_xlv:194-203`,
`phylo_gamma_xlv:123, 185-195`, `phylo_nb_xlv:123`.

**Still passing, but within 10× of the threshold** (fragile to platform or BLAS changes):
- `test_hurdle_nb.jl:64` (ratio 0.12),
- `test_aghq_public_gaussian.jl:17` (0.17),
- one of `test_formula.jl:61/68/75/83` (0.18),
- `test_delta_gamma.jl` fits (0.13–0.19).

**Platform fragility.** `phylo_beta_xlv:183` fails on macOS 1.10 and Linux 1.13 but passes
on Linux 1.10. The NB1 regression test changes its verdict between Julia versions. Flips
this close to the line are decided by floating-point details.

## 5. Other correctness notes

- **A code comment in the PR is wrong.** `src/fit_verdict.jl` says "this package's callers
  all leave `x_abstol = x_reltol = f_abstol = f_reltol = 0.0`". In fact
  `fit_gaussian_gllvm` (`src/fit.jl:372-373`, `:476-477`) and `aghq_gaussian_fit.jl:109`
  set `x_abstol = 1e-8` and `f_reltol = 1e-10`. For those callers an x or f stop is a
  criterion they chose, and the new rule silently overrides it. I saw no flips there, since
  the Gaussian fits stop on the gradient at iteration 0, but the comment and the design
  premise are wrong.
- `Optim.g_residual` is taken at `state.x`. When a run ends on `f_increased`,
  `Optim.minimum(res)` may come from `x_previous`. The two can then refer to different
  points. This is a rare edge case.

## Blocking

1. The PR turns five existing `@test fit.converged` assertions red (twolevel:107,
   phylo_poisson_xlv:158, phylo_beta_xlv:183, and the unlisted phylo_binomial_xlv:182 and
   phylo_gamma_xlv:172), and CI shard 4/4 is red on 1.10.12 and 1.13.0.
2. The new regression test is platform-fragile. The loglik pin fails on Linux (both
   versions), and `!fit.converged` fails on 1.13.0.
3. Good fits are flagged as not converged: Gamma shared shape, 5/20 on the FD path and 1/20
   default; twolevel ratio 1.13 at an exact gradient. The after-task report calls every
   flip an "honesty correction", and that claim is falsified.

## Should fix

- Before downgrading an x/f stop, confirm that it is a real stall. For example, require that
  a trial step along −g does not decrease f by more than some tolerance. Or recompute the
  gradient with a larger step (about 1e-4) before judging. Or at least exempt ForwardDiff
  and analytic gradients whose ratio is within about 10× of the threshold.
- Treat a gres above about 1e11/h as a separate reason ("the FD step hit the failure
  value"), not as "the gradient is large".
- Correct the false x/f-tolerance comment. Decide what callers that set their own `x_tol`
  and `f_tol` should get (`fit.jl`, `aghq_gaussian_fit.jl`).
- Make the regression test platform-robust: drop the loglik pin at `atol=1e-6`, and assert
  the verdict relation (gres > threshold implies `!converged`) rather than a fixed outcome.
- Correct the after-task report: 8 NelderMead sites, not 6; the "all flips are honesty
  corrections" claim; and add the twolevel, phylo_xlv and Gamma evidence.
- Follow-up issue: replace the |nll| scaling with a curvature-scaled gradient test
  (Newton decrement or `max|H⁻¹g|`). Fix the jumps in the Gamma Laplace objective.
