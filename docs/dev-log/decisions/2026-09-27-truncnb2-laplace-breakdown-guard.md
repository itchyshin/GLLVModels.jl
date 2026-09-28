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
| healthy draws | 14 of 16 bit-identical; 108 within 2.3e-12; 102 moved from -1218.082 to a higher healthy optimum -1215.971 | 15 of 18 bit-identical; on 101, 111 and 103 main's fit stopped early (see below) and the guarded fit ends at the gradient-converged point 1.5e-6, 1.1e-6 and 5.0e-4 lower |

Where a healthy fit changed, its first L-BFGS run evaluated a walled-off point during a
line search, which changes the path; the retry did not run (the first fit ended off the
guard). The healthy optima are therefore unchanged to 1e-10 on 30 of 34 draws, not on
all: draw 102 (1.10) reaches a better point, and on draws 101, 111 and 103 (1.13) the
guarded fit ends slightly lower.

Those three lower values are not early stops of the guard. The independent review of
the draft PR inspected Optim's flags and the finite-difference gradient at both end
points on Julia 1.13.0:

| draw (1.13) | main path: loglik, flags, gradient norm | guarded: loglik, flags, gradient norm |
|---|---|---|
| 103 | -1352.048581, `f_converged` only, 45.3 | -1352.049080, `g_converged`, 8.4e-6 |
| 101 | -1597.341775, `f_converged` only, 0.16 | -1597.341777, `g_converged`, 4.7e-6 |
| 111 | -1578.782500, `f_converged` only, 0.056 | -1578.782501, `g_converged`, 4.1e-6 |

Main's fits are the ones that stopped early: on `f_converged` alone, at a point where the
Laplace objective is discontinuous. On draw 103, a step of 1e-5 in log r raises the
negative log-likelihood by 5.49e-4 in one direction and by 1.5e-8 in the other, so main's
fit sits on the upper lip of a pre-existing cliff of about 5.5e-4 in the objective. The
guarded fits are gradient-converged points in a smooth region about 2e-3 away; polishing
them with the floor off takes 0 iterations. The 5e-4 difference is the height of that
cliff, not a cost of the wall; the same effect goes the other way on a 1.13 draw in the
review's own sweep (seed 121: guarded 6.1e-4 higher). The discontinuity exists on main
and is out of scope here (follow-up note below).

Smallest site eigenvalue at the healthy optima: 0.79 to 2.04 (1.10) and 0.91 to 1.64
(1.13), against the 0.1 floor. The review's wider sweeps (45 draws, r from 0.05 to 0.3)
add values down to 0.716; every breakdown was below 2e-4, and nothing fell between 0.11
and 0.7.

Compute: breakdown draws take 1.2x to 3x longer (the retry); healthy draws are
unchanged within timing noise. The sweeps took about 12 minutes on four processes.

## What this does not cover

- The floor does not remove ordinary Laplace error above it (5 to 8.5 units on draws
  101, 103 and 107, with min eig 0.79 to 2.0; the sign varies).
- `fit_truncated_nbinom2_gllvm_pertrait` shares the site kernel but is not guarded. It
  did not break down on the three audit draws (min eig 0.86 to 2.2; some r_t at the
  boundaries), which does not show immunity.
- The shared-r CI adapter (`_family_ci(::TruncatedNegBin2Fit, ...)`) profiles the
  unguarded marginal; a profile could walk into the breakdown region. The review probed
  one draw (104) by a hand profile of r and found no breakdown over the range a 95%
  interval would cover (smallest eigenvalue at least 0.89). One draw is not a proof.
  The same probe found that the adapter's own intervals for r are not trustworthy on
  this family (a Wald interval far narrower than the hand profile, and a profile run
  that returned `status = :failed`); that is pre-existing and separate from this guard.
- One DGP, 20 draws. The rate (4 of 20) is a point estimate.
- The discontinuity in the Laplace objective described above. Follow-up note, not a fix:
  its likely source is the inner mode solve. On draw 103 (Julia 1.13, main's end point),
  one site (y = [121, 1, 1, 2]) accounts for the whole 5.5e-4 jump. There the
  Fisher-scoring iterates of `_grouped_laplace_mode` (`grouped_dispersion.jl`) alternate
  around the mode (-1.62047, -1.61958, -1.62048, -1.61957, ...) instead of converging,
  and the solve stops at a point that jumps from 7e-6 to 1.3e-3 away from the true mode
  (-1.62003) when log r moves by 1e-5; the site's Laplace value is then evaluated off the
  mode. A Newton step on the observed curvature, or a stopping rule on the inner
  gradient, would be the place to look. Measured on one site of one draw; the same
  mechanism is a plausible cause of the unreliable Wald interval for r noted above.
