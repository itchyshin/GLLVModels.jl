# Laplace breakdown guard on `fit_truncated_nbinom2_gllvm`

Status: implemented on branch `claude/fix-truncnb2-laplace-breakdown` (draft PR).
Julia-side guard, not an R-semantics change. Same design as the zi_* guard of PR #557
(`2026-09-27-zi-laplace-breakdown-guard.md`), adapted in `truncated_nbinom2.jl` only;
`laplace.jl` and `zi_twin.jl` are untouched.

## Finding

The shared-r truncated NB2 fitter defaults to the observed curvature in the Laplace
log-determinant. That curvature, `_truncnb2_observed_weight`, is negative for small r
(-0.033 at r = 0.2, y = 1, mu = 7.4), and more negative as r -> 0. The per-site
precision `A = I + Λ' diag(W) Λ` can then approach singularity while the site mode
search still converges, and `-1/2 logdet(A)` inflates the Laplace value. A PD check
cannot catch this: at a certified mode `A` is PSD anyway; the failure is PD but
near-singular.

Impossible-loglik audit (2026-09-27, finding F1), generator `genTNB` (p = 4, n = 150,
K = 1, r = 0.3, `MersenneTwister(100 + s)`), exact marginal by a 4001-point trapezoid
on z in [-12, 12]:

| draw | reported loglik | exact there | min site eig(A) | healthy optimum (exact) |
|---|---|---|---|---|
| 104 | -1663.100 (converged) | -1741.215 | 7.9e-6 | -1686.333 (-1686.37) |
| 106 | -2221.222 (converged) | -2359.845 | 1.3e-4 | -1888.350 (-1889.90) |
| 110 | -1787.275 (converged) | -1827.147 | 1.3e-5 | -1637.240 (-1638.93) |

On 104 the breakdown point is the global maximum of the Laplace objective (23 units
above the healthy optimum, 55 units below it in exact terms), so a better optimiser
finds the artefact more often, not less.

## Decision

1. The site kernel `_truncnb2_pertrait_loglik_site` takes `eigmin_floor` (default
   `-Inf`, which skips the check, so the public marginals and the per-trait fitter are
   unchanged bit for bit). `fit_truncated_nbinom2_gllvm` passes
   `TRUNCNB2_LAPLACE_EIGMIN_FLOOR = 0.1`: a site with an eigenvalue of `A` below it
   makes the objective `-Inf`, which the fitter's existing 1e12 sentinel handles.
   Floor value as on the zi_* route; healthy truncated-NB2 optima measured 0.79 to 2.04
   in the sweeps below (0.79 to 9.6 in the audit).
2. Flag rule, load-bearing as on the zi_* route: an optimum whose smallest site
   eigenvalue is within 10% of the floor is reported `converged = false` with a
   warning. Optim reports `converged = true` for a fit stalled at the wall.
   `TruncatedNegBin2Fit.min_site_eigen` records the value.
3. One retry when the first fit ends at the guard, from the same loadings with r from a
   moment estimate (`_truncnb2_moment_logr`: geometric mean over traits of
   `m^2 / (v - m)`, each clamped to [0.2, 20]) instead of the default 10. Kept only if it
   ends off the guard; otherwise the first fit is reported flagged (the #557 rule).
   One measured departure from #557, whose retry scales the loadings by 0.1:
   - The r reset is what matters. From r = 10, a loadings-x-0.1 start fell into poor
     basins on all three audit draws (-2162.8, -2313.7, -1737.6 with r -> 5e-7). With
     the moment r (0.37, 0.37, 0.56 on the three draws), loadings x 0.1 reached the
     healthy optimum on Julia 1.10.12 but, on Julia 1.13.0, stopped at a genuine local
     maximum on draw 104 (-1705.63, 19 units below; a restart from there does not
     move). Loadings x 0.5 and x 1.0 reached -1686.33 on both versions. On the 12
     breakdown draws among 40 draws per version (8 on 1.10, 4 on 1.13; the draws differ
     by version because the sampler stream does), the three scales ended at the same
     point every time. The unshrunk loadings are used.

## Measured effect, 20 draws (`genTNB`, `MersenneTwister(101:120)`)

"Before" is `eigmin_floor = -Inf`, which is main's code path (the floor check and the
retry are both skipped; the draw-104 value reproduces main's -1663.1001 exactly). The
draws differ between Julia versions (sampler stream), so the two columns are
different data.

| | Julia 1.10.12 | Julia 1.13.0 |
|---|---|---|
| silent breakdowns before (converged, min eig < 0.1) | 4 (104, 106, 110, 119; min eig 8e-6 to 1.3e-4; Laplace 40 to 139 above exact) | 2 (107, 117; min eig 1.0e-4 and 1.5e-4; Laplace 24 above and 35 below exact) |
| silent breakdowns after | 0 | 0 |
| flagged `converged = false` after | 0 | 0 |
| breakdown draws at the healthy optimum after | 4 of 4 (min eig 0.95 to 1.11, abs(Laplace - exact) at most 1.7) | 2 of 2 (min eig 1.04, 1.09; abs gap at most 1.5) |
| healthy draws | 14 of 16 bit-identical; 108 within 2.3e-12; 102 moved from -1218.082 to a higher healthy optimum -1215.971 | 15 of 18 bit-identical; 101, 111 and 103 end 1.5e-6, 1.1e-6 and 5.0e-4 lower at the same optimum |

Where a healthy fit changed, its first L-BFGS run evaluated a walled-off point during a
line search, which changes the path; the retry did not run (the first fit ended off the
guard). The healthy optima are therefore unchanged to 1e-10 on 30 of 34 draws, not on
all: draw 102 (1.10) reaches a better point and draw 103 (1.13) stops 5e-4 short of
the same one, within the optimiser's g_tol.

Smallest site eigenvalue at the healthy optima: 0.79 to 2.04 (1.10) and 0.91 to 1.64
(1.13), against the 0.1 floor.

Compute: breakdown draws take 1.2x to 3x longer (the retry); healthy draws are
unchanged within timing noise. The sweeps took about 12 minutes on four processes.

## What this does not cover

- The floor does not remove ordinary Laplace error above it (5 to 8.5 units on draws
  101, 103 and 107, with min eig 0.79 to 2.0; the sign varies).
- `fit_truncated_nbinom2_gllvm_pertrait` shares the site kernel but is not guarded. It
  did not break down on the three audit draws (min eig 0.86 to 2.2; some r_t at the
  boundaries), which does not show immunity.
- The shared-r CI adapter (`_family_ci(::TruncatedNegBin2Fit, ...)`) profiles the
  unguarded marginal; a profile could walk into the breakdown region. Not probed.
- One DGP, 20 draws. The rate (4 of 20) is a point estimate.
