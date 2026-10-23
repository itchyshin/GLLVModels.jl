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
   (A single shrunk-start retry at the guard was added after the second review; see
   below.)

## Measured effect, 20 NB2 draws (p = 4, n = 400, K = 1, reviewer parameters)

- Before: 3 of 20 fits reported `converged = true` at a near-singular Laplace
  maximum (seeds 6, 10, 16; smallest site eigenvalue 3e-4 to 3.5e-3).
- After: 0 of 20 silent breakdowns. Seed 16 reaches R's optimum (-3371.265). Seeds
  6 and 10 end at the guard and are reported `converged = false`; a fit started at
  the true parameters also runs to the guard on both, and R fails on the same two.
  The other 17 fits are unchanged, except seed 18, which moved from a different
  (non-breakdown) local maximum with one `phi` at its boundary (-3381.617) to R's
  optimum (-3386.720).

## Second review (PR #557, head d0a57e05d) applied

### Load-bearing flag rule

Optim reports `converged = true` for a fit stalled at the wall: when the line search
runs into the 1e12 sentinel, L-BFGS with `BackTracking` takes a zero step and declares
convergence (true on every wall-stalled fit the review measured). So the rule "an
optimum within 10% of the floor is reported `converged = false`" is load-bearing; the
floor is safe for the optimiser only together with it. Stated in the code comment above
`ZI_LAPLACE_EIGMIN_FLOOR` and at the rule in `fit_zi_gllvm`. Do not relax it as
redundant.

### One shrunk-start retry at the guard

The review found a Julia-only stall: at the recovery test's NB2 DGP
(`MersenneTwister(2)`, beta = [2.0, 1.6, 2.2, 1.8], lambda = [0.6, -0.5, 0.4, 0.5],
phi = [1.5, 2, 1, 2], n = 350) the default start stalled at the guard (Laplace
-3749.88, flagged) while gllvmTMB at P1 and a Julia fit from the truth both reach
-3820.2665 (min site eigenvalue 0.727). A fit that ends at the guard is now retried once
from the warm start with the loadings scaled by 0.1 (beta, logit_zi and the moment NB2
phi as in the warm start); the retry is kept only if it ends off the guard, otherwise
the first fit is reported flagged. Scale 0.25 and 0.5 (with or without phi = 1) and
a doubled start all stalled at the guard on that draw; 0.1 reached -3820.2665062895.

Measured (Julia 1.10, same generators as the review scripts):

| set | draws | converged off the guard | flagged | flagged draws where gllvmTMB also fails |
|---|---|---|---|---|
| builder's sweep (reviewer phi, n = 400, seeds 1 to 20) | 20 | 18 | 2 (seeds 6, 10) | 2 of 2 (R `convergence = 1`) |
| recovery DGP, seeds 11 to 22 | 12 | 10 | 2 (seeds 11, 21) | 2 of 2 (R error; R `convergence = 1`, phi 431) |
| recovery DGP, probe seeds 1 to 3 | 3 | 2 (seed 2 red to green) | 1 (seed 1) | 1 of 1 (R `convergence = 1`) |

Seed 2: before the retry `converged = false`, -3749.8774, min eig 0.10000; after,
`converged = true`, -3820.2665063, min eig 0.727 (R -3820.266506). All other fits are
unchanged to the printed digits (the retry runs only at the guard). So the Julia-only
stall rate went from 1 of 35 to 0 of 35 measured NB2 draws; every remaining flag is on
a draw where gllvmTMB also fails. This is a measured rate on these draws, not a
guarantee that the start reaches the sensible basin on every dataset.

### Strong loadings

Recovery is shown only for `|lambda| <= 0.6`. At `|lambda|` 0.8 to 1.8 (Poisson, three
draws) both engines' default starts land far below the truth's Laplace value: on
`zi_poisson_s3.0_seed2` Julia and R converge to the same point (-3900.5913, min eig
1.49) 711 units below the truth, both reporting convergence. Twin-consistent and not a
guard artefact; logged as a start-quality gap (multi-start would address both sides).

### Floor margin: zi_poisson and zi_binomial sweep

15 draws each at the recovery-test DGP (p = 4, n = 350, K = 1, `MersenneTwister(1:15)`;
Poisson beta = [1.2, 0.8, 1.4, 1.0]; binomial beta = [0.2, -0.3, 0.5, 0.1], trials 3 to 8),
guarded fit and unguarded fit (`eigmin_floor = -Inf`):

| family | converged | flagged | guarded = unguarded optimum | min site eig at optima (min / median) | at truth (min) |
|---|---|---|---|---|---|
| zi_poisson | 15 of 15 | 0 | 15 of 15 | 0.318 / 0.458 | 0.436 |
| zi_binomial | 15 of 15 | 0 | 15 of 15 | 0.549 / 0.781 | 0.560 |

No legitimate optimum sat below 0.15; the smallest measured at any sensible optimum
remains the P1 `zi_binomial` fixture (0.202, 1.8x the 0.11 flag threshold), then the
Poisson draw above (0.318) and the NB2 draws (>= 0.39). The floor stays at 0.1. Its
margin is least measured for `zi_binomial` with stronger loadings or smaller trials.

## Not decided here

Whether gllvmTMB should guard or flag the same region (its `convergence = 1` on the
two failing draws is the only signal it gives), and whether an adaptive-quadrature
route should replace Laplace for these families. Both are for the maintainer.
