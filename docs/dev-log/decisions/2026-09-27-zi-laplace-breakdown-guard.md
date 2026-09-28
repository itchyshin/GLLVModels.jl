# Laplace breakdown guard on the zi_* route (PR #557)

Status: implemented on branch `claude/twin-zi`; Julia-side guard, not an R-semantics
change. Worth reporting to the gllvmTMB side (its objective shares the surface).

## Finding

The zero-inflated mixture's observed count curvature is negative at y = 0 for
moderate means: a zero can come from either process, so `log f(0 | eta)` is locally
convex in `eta`. The per-site Laplace precision `A = I + Λ' diag(W_obs) Λ` can then
approach singularity while the site mode search still converges, and
`-1/2 logdet(A)` inflates the Laplace value.

Reviewer case (`test/fixtures/zi_nb2_reviewer_seed12.csv`, p = 4, n = 400, K = 1):

| point | Laplace (Julia) | TMB objective (R, P1) | exact marginal (quadrature) | min site eig(A) |
|---|---|---|---|---|
| Julia pre-fix optimum | -3012.805 | -3012.815 | -3376.08 | 1.8e-4 |
| R's optimum | -3311.455 | -3311.455 | -3310.37 | 0.93 |
| truth | -3316.93 | | -3315.87 | 0.91 |
| best point with eig(A) >= 0.5 | -3305.21 | | -3332.31 | 0.50 |

Quadrature: 6001-point trapezoid on z in [-12, 12] (K = 1, exact to far below the
differences shown). The pre-fix optimum is a Laplace artefact 363 units above the
exact marginal there; R's objective returns the same inflated value, so R reaches the
sensible optimum only through its start. On 20 NB2 draws at the reviewer's setting
(`MersenneTwister(1:20)`), R at P1 stopped in this region on 2 (seeds 6 and 10,
`convergence = 1`, smallest site eigenvalue about 0).

## Decision

1. `zi_marginal_loglik_laplace` returns `-Inf` (the fitter's 1e12 sentinel) when any
   site's `A` has an eigenvalue below `ZI_LAPLACE_EIGMIN_FLOOR = 0.1`. Floor reason:
   half the smallest site eigenvalue measured at any sensible R optimum (0.20 at the
   P1 `zi_binomial` fixture; >= 0.50 on the 18 sensible NB2 draws and the other two
   fixtures), so it never touches those, while it cuts off the near-singular end.
   The three P1 twin receipts are unchanged (logLik to 1e-8 as before).
2. The floor does not remove Laplace error above it (walling at 0.5 still left a
   point with 27 units of error on the reviewer case). So an optimum at the guard
   (smallest site eigenvalue < 1.1 x floor) is reported with `converged = false` and
   a warning, and `ZiFit.min_site_eigen` records the value.
3. Start hardening: NB2 `phi` from a moment estimate of the positive counts
   (clamped [0.2, 20]) instead of 1; SVD loadings at half scale. With it the
   reviewer case reaches R's optimum (-3311.4547, logLik to 1e-6).

## Measured effect, 20 NB2 draws (p = 4, n = 400, K = 1, reviewer parameters)

- Before: 3 of 20 fits reported `converged = true` at a near-singular Laplace
  maximum (seeds 6, 10, 16; smallest site eigenvalue 3e-4 to 3.5e-3).
- After: 0 of 20 silent breakdowns. Seed 16 reaches R's optimum (-3371.265). Seeds
  6 and 10 end at the guard and are reported `converged = false`; a fit started at
  the true parameters also runs to the guard on both, and R fails on the same two.
  The other 17 fits are unchanged, except seed 18, which moved from a different
  (non-breakdown) local maximum with one `phi` at its boundary (-3381.617) to R's
  optimum (-3386.720).

## Not decided here

Whether gllvmTMB should guard or flag the same region (its `convergence = 1` on the
two failing draws is the only signal it gives), and whether an adaptive-quadrature
route should replace Laplace for these families. Both are for the maintainer.
