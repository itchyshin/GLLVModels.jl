DRAFT ISSUE, NOT FILED. For maintainer review before filing.

---

Title: Track remaining families with an undamped per-site Laplace mode search (post #479/#480/#486/#484 pattern)

## Background

#479 (Gamma grouped) and #480 (Beta grouped) fixed the same defect: a per-site inner Laplace
mode search takes undamped Fisher-scoring steps, can diverge at the fitter's own warm start, and
still returns a finite site log-likelihood, so a fit reports `converged = true` far from the
optimum. #486 found and #494 fixed the same defect in the shared covariate kernel
(`_laplace_mode_off`, `src/families/covariates.jl`), which also happened to cover the Poisson
and Binomial rows of the original class-wide audit even though it was filed and reviewed as a
Gamma fix. PR #500 (open) targets the twopart-family kernel (ZIP, ZINB, ZIB, HurdlePoisson,
HurdleNB, DeltaGamma, Delta-lognormal, Beta-hurdle) for the same reason (#484).

Each family has its own copy of the per-site loop, so each fix has so far been scoped to one
family at a time (rationale recorded in #479's after-task report: sharing a fix path risks
changing results on sites that used to converge correctly, and the fixes so far measure that
they do not). This issue is the tracking list for what is left, so the pattern is fixed
family-by-family rather than lost between individual PRs.

Source: class-wide audit at
`origin/claude/lane-true-parity-finish-20260925:docs/dev-log/core070/class-audit-20260924/`
(2026-09-24), re-measured against `origin/main` @ `b90641c97` on 2026-09-26 with reduced trial
counts (see the triage report this issue is drawn from for exact counts and the probe scripts).
"Non-mode rate" = the fraction of stress draws where the kernel's own z has
`|grad log-posterior| > 1e-4` while a from-scratch restart converges cleanly at the same site.

## Still-live families (Class A: mode search itself, not just the convergence flag)

| Family / kernel | Audit rate | Re-measured rate (2026-09-26, reduced n) | Default-route reachable | Coverage |
|---|---|---|---|---|
| NB1 grouped (`_nb1_grouped_loglik_site`) | 108/3000 (3.6%) | 20/600 (3.3%) | yes, confirmed (`fit_gllvm(Y; family=NB1(), K=2)`) | none |
| Ordered beta (`_ordered_beta_mode`) | 212/1500 (14.1%) | 72/500 (14.4%) | yes | issue #501 open (fit-level symptom, same underlying kernel is the suspected mechanism per #501's own hypothesis) |
| StudentT, shared kernel | 246/1500 (16.4%) | 78/500 (15.6%) | yes | none |
| StudentT, grouped kernel (default route) | 273/1500 (18.2%) | 101/500 (20.2%) | yes | none |
| GP1 (generalised Poisson), generic kernel | 120/1000 (12.0%) | 44/333 (13.2%) | likely, not independently confirmed | none |
| BetaBinomial (`_beta_binomial_mode`) | 103/1500 (6.9%) | 27/500 (5.4%) | likely, not independently confirmed | none |
| COMPoisson (`_compoisson_mode`) | 19/800 (2.4%) | 7/266 (2.6%) | likely, not independently confirmed | none |
| Mixed-family bridge (`_mixed_laplace_mode`) | 42/1200 (3.5%) | 16/400 (4.0%) | yes, this is the multi-family bridge path | none |
| Tweedie, grouped kernel (disp_group route) | 13/300 (4.3%) | 11/150 (7.3%, small n) | narrower route per audit notes | none |
| NB2, grouped kernel (excluded fitters) | 147/1500 (9.8%) | not re-measured | audit notes "excluded fitters" | none |

Priority suggestion, by user-facing risk (default-route reachability plus rate):

1. NB1 grouped and StudentT (both routes): common families, confirmed or likely default-route
   reachable, no open PR.
2. Ordered beta: already has issue #501 open for the fit-level symptom; this row is the
   supporting kernel-level evidence that the mode search itself, not only the verdict flag, is
   implicated (consistent with #501's own hypothesis about `_ordered_beta_mode` switching
   between local modes).
3. GP1, BetaBinomial, mixed-family bridge, COMPoisson: same shape, lower or unconfirmed
   default-route reachability.
4. Tweedie grouped, NB2 grouped: narrower routes ("excluded fitters" per the audit); confirm
   reachability before prioritizing further.

## Not in scope for this issue

- Twopart families (ZIP, ZINB, ZIB, HurdlePoisson, HurdleNB, DeltaGamma, Delta-lognormal,
  Beta-hurdle): tracked by PR #500 (open), not by this issue.
- The Class B `_fit_verdict` gradient-criterion gap (roughly 85 fitters reporting
  `converged=true` on `x`/`f` convergence alone, without checking the gradient): tracked by PR
  #502 (draft, Fixes #485). That PR fixes the flag, not the mode search; a family can appear in
  both this issue and #502's scope, and fixing one does not fix the other.
- The `confint_family.jl` bootstrap and profile-refit helpers accepting any finite value without
  reading `.converged`/`.loglik`: filed as a separate issue (see the sibling draft), since it is
  a distinct code path (confidence intervals, not point fits) that affects every family
  regardless of which per-family mode-search fix has landed.

## Suggested approach

Same as #479/#480/#486/#494: one family (or one shared kernel) per PR, damped step (halve any
step that raises the per-site objective) with a documented convergence test that does not count
a heavily-halved step, a damped-Newton fallback only where the observed curvature is provably
non-negative, and `-Inf` (not a finite garbage value) when neither converges. Each PR should
carry its own regression fixture reproducing a diverging site at the fitter's own warm start,
and a behaviour-change table for any fit that flips from `converged=true` to `converged=false`
or `-Inf`, per the precedent in #479/#480/#486's after-task reports.
