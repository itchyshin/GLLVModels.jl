# Issue #501 (ordered-beta false-converged flag): macOS check on current main (2026-09-27)

**Verdict: RESOLVED on macOS main.** All three asks in #501 hold for all 10 seeds
checked: (a) zero non-stationary sites at every returned fit (independent
ForwardDiff gradient, not the package's own score/mode-search code), (b) no
jump along the line from the fit to its best restart, (c) every previously
"flagged" seed's fit already sits at the restart's optimum (gap ≈ machine
precision, `1e-4` threshold not remotely approached). This is on top of, and
consistent with, #511 (`d55a8e3af`, "damped per-site mode search", merged
2026-09-26), which is an ancestor of the commit checked here.

## Setup

Own detached worktree `~/local-scratch/check-501/repo`, `git worktree add
--detach` from `origin/main` (main clone: `/Users/z3437171/Dropbox/Github
Local/GLLVM.jl`). Checked out at **`46461f883`** (current `origin/main` tip,
one commit past `d55a8e3af` = #511; verified `git merge-base --is-ancestor
d55a8e3af HEAD` = true). `julia +1.10` = 1.10.12, `--project=.` (root
`Project.toml`, not `test/parity` — avoids an unneeded RCall precompile).
`JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, one Julia process at a time.

**DGP**: `sibling_screen.jl`'s `route_ordered_beta` recipe exactly — `p=5,
n=40, K=2, sd=0.5, c0true=-1.0, c1true=1.0, φtrue=8.0, βlo=-0.3, βhi=0.3`,
`MersenneTwister(seed)`, `lowtri`/`rand_orderedbeta` reproduced verbatim.
Fit: `fit_ordered_beta_gllvm(Y; K=2)`, all other args left at their defaults.

**Method** (own script, `~/local-scratch/check-501/check501.jl`, not run
from the sibling-screen or verify scripts):
- **Stationarity**: for every site, took the fitted per-site Laplace mode
  `ẑ` via the public `getLV(fit, Y; rotate=false)`, then built a fresh
  per-site log-posterior `q(z) = -½z'z + Σ_t ordered_beta_logp(y_t, β_t+Λ_t·z,
  c0, c1, φ)` (reuses only the density *formula*, not `_ob_score_weight` or
  `_ordered_beta_mode`) and took `ForwardDiff.gradient(q, ẑ)` — a full
  multivariate AD gradient, independent of the package's own per-trait
  scalar-derivative score/weight code. Flagged non-stationary at scale-aware
  `|g|·(1+|z|) > 1e-3` (the sibling screen's own bar).
- **Restarts**: reconstructed `negll(θ)` from the public
  `ordered_beta_marginal_loglik_laplace` wrapper (same packing, same
  `maxiter=100/tol=1e-9` inner defaults as the fitter), then ran the SAME
  optimizer config the fitter uses (`LBFGS` + `BackTracking(order=3)`,
  `g_tol=1e-5`, `iterations=500`, `autodiff=:finite`) from **two** starts:
  a perturbed θ̂ (`θ̂ .+ (0.3|θ̂|+0.1)·N(0,1)`, seeded `seed+9000`) and the
  true generating θ. Gap = best-of-two minus `fit.loglik`; "better" flagged
  at gap `> 1e-4`.
- **Line trace** (macOS-flagged seeds only): 21 equally spaced points on the
  straight line in packed-θ space from `θ̂` to the best restart's minimizer,
  reporting the max single-step |Δll| against the median step (a ratio ≫1
  would signal a discontinuity/jump; smooth curvature gives a ratio of a
  few).

## Per-seed results

| seed | converged | loglik | non-stat. sites | max scaled `\|g\|` | best-restart gap | better? (>1e-4) | macOS-flagged |
|---|---|---|---|---|---|---|---|
| 2001 | true | -169.0038 | 0/40 | 9.4e-15 | -2.5e-12 | no | no |
| 2002 | true | -194.6364 | 0/40 | 3.7e-15 | -6.4e-12 | no | **yes** |
| 2003 | true | -175.1267 | 0/40 | 7.5e-15 | -7.9e-12 | no | **yes** |
| 2004 | true | -196.7429 | 0/40 | 5.3e-15 | +1.7e-11 | no | **yes** |
| 2005 | true | -182.8099 | 0/40 | 6.4e-15 | +2.4e-11 | no | no (not macOS-flagged) |
| 2006 | true | -198.9214 | 0/40 | 5.3e-15 | +2.9e-11 | no | **yes** |
| 2007 | true | -184.9645 | 0/40 | 3.2e-15 | -2.6e-12 | no | **yes** |
| 2008 | true | -192.2081 | 0/40 | 1.9e-15 | -1.9e-12 | no | **yes** |
| 2009 | true | -183.9922 | 0/40 | 4.9e-15 | +2.1e-12 | no | no |
| 2010 | true | -172.6168 | 0/40 | 8.9e-15 | +4.3e-12 | no | **yes** |

All 10 fits: `converged = true`, 0/40 non-stationary sites, restart gap at the
floor of floating-point noise (`~1e-11`, six orders of magnitude below the
`1e-4` "better" threshold). Loglik values match the issue's own reproduction
data closely: seed 2005 (-182.8099) and seed 2003 (-175.1267) reproduce
exactly what the issue's comment records as the *good* macOS optimum for
those seeds — obtained here directly by the default fit, no restart needed.

## Line trace (7 macOS-flagged seeds)

| seed | max single-step \|Δll\| | median step | ratio | smooth or jump? |
|---|---|---|---|---|
| 2002 | 0.864 | 0.555 | 1.56 | smooth |
| 2003 | 0.000 | 0.000 | 2.38 | flat (fit ≡ restart to float precision) |
| 2004 | 0.000 | 0.000 | 1.84 | flat |
| 2006 | 0.000 | 0.000 | 2.26 | flat |
| 2007 | 0.000 | 0.000 | 2.18 | flat |
| 2008 | 4.280 | 1.895 | 2.26 | smooth |
| 2010 | 0.000 | 0.000 | 2.80 | flat |

For 5 of 7 seeds the fit and best restart are numerically the same point (gap
below `1e-11`), so the "line" is a point and there is nothing to jump across.
For seeds 2002 and 2008 the fit and restart minimizer differ by enough to
trace a real line; both traces are smooth, unimodal dips of a few to ~22
log-lik units across the 21-point path (jump-ratio 1.6–2.3, i.e. the largest
single step is only 1.6–2.3× the typical step) — nothing resembling the
issue's reported 1/h-scaling gradient blow-up or a discontinuous cliff. None
of the 7 traces show a jump.

## Verdict on the three asks

- **(a) no finite non-stationary sites at the returned fit** — **HOLDS.**
  0/40 sites flagged at every one of the 10 seeds; the independent
  ForwardDiff gradient is at the ~1e-14 level (machine precision), not merely
  "small."
- **(b) the objective no longer jumps along the line from the old flagged
  estimate to its restart** — **HOLDS**, with a caveat: on current main the
  fit itself already lands at (or numerically at) the restart's optimum, so
  for 5/7 macOS-flagged seeds there is no longer a meaningfully separated
  "old flagged estimate" to draw a line from — the two endpoints coincide.
  Where a real line exists (2002, 2008), it is smooth, not jumping.
- **(c) the previously flagged fits reach the better optimum, matching a
  restart** — **HOLDS.** Every one of the 7 macOS-flagged seeds (2002, 2003,
  2004, 2006, 2007, 2008, 2010) now converges directly to the loglik a
  restart finds; gaps are all at the floating-point noise floor.

**Caveat**: this check ran macOS only (Julia 1.10.12, aarch64), as scoped.
**Seed 2005 was Linux-flagged, not macOS-flagged, per the issue's own
correction comment, and the Linux leg was not run here** — nothing in this
check speaks to current Linux behavior.

## Compute record

Total wall time **≈ 49 s**: `Pkg.instantiate()` 17 s (86/87 deps already
precompiled in the shared depot; only `GLLVModels` itself needed a fresh
precompile for this worktree, 13 s of that), then the full 10-seed run
(fit + stationarity check over 40 sites + 2 restarts each, plus line-trace
for the 7 flagged seeds) in **31.8 s**. One Julia process at a time, none left
running afterward (`ps aux | grep julia` after the run showed only other
agents' unrelated sessions, none under `~/local-scratch/check-501`). Well
inside the 40-minute cap; no restarts were cut.

## Files

- `~/local-scratch/check-501/check501.jl` — the check script (own DGP
  reproduction + stationarity + restart + line-trace code, not a copy of the
  sibling-screen or verify-501 scripts).
- `~/local-scratch/check-501/run_output.log` — full run output.
- Worktree `~/local-scratch/check-501/repo` (detached at `46461f883`) removed
  after this check; nothing else under `~/local-scratch/check-501` touches
  any other lane's files.
