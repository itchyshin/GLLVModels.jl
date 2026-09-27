# Whole-fit check: #510 (beta-binomial) and #512 (COM-Poisson) damped per-site mode search

Question: on datasets where `main`'s OLD undamped per-site Newton search already reports
a healthy, fully-stationary fit, does damping it (as #510/#512 do) change the WHOLE FIT's
final answer, the way it did for the sibling PR #511 (ordered beta, +52 / +9 loglik at 5x
loading scale)? Independent reviews of #510 and #512 checked only that individual
non-mode sites get repaired and healthy sites are numerically unchanged (max ~1e-11); this
check instead re-fits whole datasets end-to-end on `main` and on each PR head and compares
the outer optimiser's final answer, at the family's own test loading scale and at 5x that
scale.

## Setup

- `main`: detached worktree at `origin/main` `2847b5dbf451ca5ba84c6fd6047c0aedf11b2655`
  (`~/local-scratch/wholefit-main`, created and removed by this check).
- `#510` (beta-binomial): existing worktree `~/local-scratch/GLLVM.jl-503-betabinomial-ovA-bb`,
  head `d8a696571e8c584d4588998b43842a667a125689` — read/run only, not modified, not removed
  (pre-existing, owned by another lane).
- `#512` (COM-Poisson): detached worktree at `3c5c1d3da435e645c39817807b8ecb97f6036c22`
  (`~/local-scratch/wholefit-512`, created and removed by this check).
- Julia 1.10.12, `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`, each fit a separate process
  with `--project` pointing at the relevant worktree.
- Sizes: `p=6, n=120, K=2` for every dataset (within the requested p 5-8/n 100-150 band).
- Loading scale ("the scale its tests use"): `test_beta_binomial.jl`'s "fit smoke" testset
  uses `Λ_true = reshape(0.9 .* randn(p), p, K)` → BB base `sd=0.9`, 5x `sd=4.5`.
  `test_com_poisson.jl`'s "underdispersion smoke fit" testset uses `Λ_true = reshape(0.7 .*
  randn(p), p, K)` → CMP base `sd=0.7`, 5x `sd=3.5`. Λ is generated lower-triangular with a
  positive diagonal (the package's own `pack_lambda`/`unpack_lambda` identifiability
  convention; same generator this backlog's `sibling_screen.jl` and the #511 review used).
- Generative model: BB draws real Beta-Binomial data (`φ_true=12.0`, `N∼Uniform{8..15}`,
  `β∼N(0,0.5²)`, matching `test_beta_binomial.jl`'s own true values). CMP draws real
  Conway-Maxwell-Poisson data via a hand-written inverse-CDF sampler (`test_com_poisson.jl`'s
  own smoke test never draws real CMP data — it fits Poisson-generated data as a proxy — so
  `ν_true=1.5`, mild underdispersion, and `β∼N(0,0.3²)` with `η` clamped to `[-6,6]` were
  chosen to keep counts numerically tame; without the clamp, 5x loadings compounded through
  the exp() link produce counts in the thousands).
- Per dataset: fit with the public fitter (`fit_beta_binomial_gllvm`/`fit_compoisson_gllvm`,
  all defaults) on `main` and on the PR head. Stationarity check: pull the per-site latent
  scores via `getLV(fit, Y; rotate=false)` (the actual z the fit's own internal per-site
  search settled at, unrotated), then an INDEPENDENT, from-scratch ForwardDiff gradient of
  the per-site log-posterior at that z (own reimplementation of the beta-binomial/CMP
  log-density — own `loggamma` formula / own log-sum-exp series — not a call into the
  package's `betabinomial_logp`/`compoisson_logpdf`/mode-search internals). A site is
  non-stationary if the scale-aware gradient `max|g|·(1+|z|) > 1e-4`, or if `z`/`g` is
  non-finite (a diverged z can poison a naive comparison into a false "0 non-stationary" —
  see the #510 table below).

## Scope cut (time budget)

CMP fits at 5x loading scale are far more expensive than every other cell: `main`'s single
per-site Newton search took **340s** for one 5x dataset (`cmp_8101_5x`) against **158s** on
the `#512` branch for the identical data — a real slowdown, not noise, and consistent with
the hypothesis under test (an undamped inner search can make the OUTER optimiser's landscape
noisier and slower to traverse). Running all 6 planned 5x CMP datasets on `main` at that rate
would have overrun the 40-minute budget by itself. **Cut CMP 5x from 6 datasets to 2**
(seeds 8101-8102); base-scale CMP (6 datasets) and both BB scales (6+6 datasets) ran at their
full planned count. This is the only cut made.

## Results — #510 (beta-binomial), p=6, n=120, K=2

| seed | scale (Λ sd) | main converged | main all-sites stationary | main loglik | branch loglik | diff (branch−main) | class |
|---|---|---|---|---|---|---|---|
| 7001 | 0.90 (base) | true | yes | −1591.292780 | −1591.292780 | 0.000000 | HEALTHY_UNCHANGED |
| 7002 | 0.90 (base) | true | yes | −1602.503333 | −1602.503332 | +0.000001 | HEALTHY_UNCHANGED |
| 7003 | 0.90 (base) | true | yes | −1612.139606 | −1612.139606 | 0.000000 | HEALTHY_UNCHANGED |
| 7004 | 0.90 (base) | true | yes | −1593.939208 | −1593.939208 | 0.000000 | HEALTHY_UNCHANGED |
| 7005 | 0.90 (base) | true | trivially yes (z≡0) | **7.18e54** (garbage) | −1636.142978 | n/a | MAIN_UNHEALTHY* |
| 7006 | 0.90 (base) | true | yes | −1607.810666 | −1607.810666 | 0.000000 | HEALTHY_UNCHANGED |
| 7101 | 4.50 (5x) | true | yes (maxg 1.5e-9) | −1038.622283 | −1031.269776 | **+7.352507** | HEALTHY_CHANGED_BETTER |
| 7102 | 4.50 (5x) | **false** | — | garbage | −1048.550289 | n/a | MAIN_UNHEALTHY (not converged) |
| 7103 | 4.50 (5x) | true | yes (maxg 1.9e-7) | −937.401731 | −931.320796 | **+6.080935** | HEALTHY_CHANGED_BETTER |
| 7104 | 4.50 (5x) | true | **no** (1 site, maxg 23.85) | −1119.952160 | −1087.182776 | n/a | MAIN_UNHEALTHY (site non-stationary) |
| 7105 | 4.50 (5x) | true | yes | −1293.872169 | −1293.872169 | 0.000000 | HEALTHY_UNCHANGED |
| 7106 | 4.50 (5x) | true | trivially yes (z≡0) | **3.57e30** (garbage) | −1148.176102 | n/a | MAIN_UNHEALTHY* |

\* **7005 and 7106 are not the class-A/B mode-search defect at all.** Inspecting the fit
directly: `main`'s OUTER LBFGS optimiser diverged to `φ≈3.3e65`, `β` in the hundreds,
`Λ` entries in the hundreds — a pre-existing complete/quasi-separation-type fragility in the
beta-binomial mean-precision parameterisation, unrelated to the per-site inner search this PR
touches. At that degenerate point every per-site latent mode trivially sits at `z=0` (the
fixed effects alone already saturate the fit), so the per-site stationarity check reports
"0 non-stationary" even though the fit is obvious garbage. Applying the task's literal rule
("main not converged or some site non-stationary") would NOT flag these; I am flagging them
anyway on a plain sanity gate (`|loglik|` astronomically large) because a "healthy" baseline
this test relies on for a fair comparison cannot itself be nonsensical. Notably, the branch
did NOT diverge on the identical data for either seed — plausibly because the damped inner
search changed the outer optimiser's path enough to dodge the same ridge, a beneficial side
effect, not evidence against the fix.

**Counts (12 fits):** HEALTHY_UNCHANGED = 6 (7001-7004, 7006, 7105); HEALTHY_CHANGED_BETTER
= 2 (7101, 7103); HEALTHY_CHANGED_WORSE = 0; MAIN_UNHEALTHY = 4 (7005, 7106: outer-optimiser
divergence unrelated to the fix; 7102: main did not converge; 7104: main genuinely left one
site non-stationary — the textbook instance of the exact defect #510 targets).

**Largest HEALTHY_CHANGED difference:** seed 7101 (5x scale, `sd=4.5`), **+7.352507** loglik
units (branch better) — main converged with every site nominally stationary
(max scaled gradient 1.5e-9), yet the branch's damped search still landed the outer optimiser
on a measurably better optimum. Seed 7103 shows the same thing at +6.08. Both are far above
the 1e-6 noise floor set by the other, genuinely-unchanged seeds (≤1e-6) and by the
independent #510 review's own healthy-site bound (max 2.59e-13 at the per-site level).

## Results — #512 (COM-Poisson), p=6, n=120, K=2

| seed | scale (Λ sd) | main converged | main all-sites stationary | main loglik | branch loglik | diff (branch−main) | class |
|---|---|---|---|---|---|---|---|
| 8001 | 0.70 (base) | true | yes | −1021.948931 | −1021.948931 | 0.000000 | HEALTHY_UNCHANGED |
| 8002 | 0.70 (base) | true | yes | −976.885556 | −976.885556 | 0.000000 | HEALTHY_UNCHANGED |
| 8003 | 0.70 (base) | true | yes | −1053.724216 | −1053.724216 | 0.000000 | HEALTHY_UNCHANGED |
| 8004 | 0.70 (base) | true | yes | −907.927550 | −907.927550 | 0.000000 | HEALTHY_UNCHANGED |
| 8005 | 0.70 (base) | true | yes | −970.945225 | −970.945225 | 0.000000 | HEALTHY_UNCHANGED |
| 8006 | 0.70 (base) | true | yes | −845.646031 | −845.646031 | 0.000000 | HEALTHY_UNCHANGED |
| 8101 | 3.50 (5x) | true | yes (maxg 1.2e-9) | −1606.348078 | −1606.288705 | **+0.059373** | HEALTHY_CHANGED_BETTER |
| 8102 | 3.50 (5x, cut short — see below) | | | | −1487.301895 | | |

(8102's `main` fit was still running when the 40-minute budget forced a stop; see Scope cut.
Its branch value is included for completeness but has no main-side comparison.)

**Counts (7 completed fits with a main comparator):** HEALTHY_UNCHANGED = 6 (8001-8006);
HEALTHY_CHANGED_BETTER = 1 (8101); HEALTHY_CHANGED_WORSE = 0; MAIN_UNHEALTHY = 0.

**Largest HEALTHY_CHANGED difference:** seed 8101 (5x scale, `sd=3.5`), **+0.059373** loglik
units (branch better) — main converged, every site nominally stationary (max scaled gradient
1.2e-9), branch still lands on a measurably (if modest) better optimum. Smaller than #510's
or #511's magnitudes, but the same sign and the same qualitative phenomenon, and well above
the ~1e-9 to 1e-11 noise floor every base-scale seed and the independent #512 review's own
200-site probe (max 8.0e-13 at the per-site level) establish.

## Conclusion

**Looked-healthy whole fits DO change under both #510 and #512, not just #511.** At the
family's own test loading scale, both fixes are exactly as advertised: 0 unhealthy-repair
cases fire (base-scale healthy fits are unchanged to the display precision used, ≤1e-6) and
none of the base-scale datasets exercised the damping path meaningfully. At 5x that scale,
where main sometimes still reports full convergence and full per-site stationarity, the
outer fit's final answer measurably moves on the branch in 3 of the 3 genuinely-healthy 5x
cases observed (BB seeds 7101 +7.35, 7103 +6.08; CMP seed 8101 +0.059) — always toward a
*better* (higher) log-likelihood in this sample, never worse, and always at a site-stationary
point on both sides (confirmed by the same independent ForwardDiff check, not just the
package's own `converged` flag). The magnitudes are smaller than ordered-beta's +52/+9 (BB is
roughly an order of magnitude smaller, CMP nearly three), but the mechanism the task
describes — damping altering intermediate outer-optimiser iterates enough to change which
final optimum is found, even when the final iterate on both sides is genuinely stationary —
reproduces in both #510 and #512. Separately, #510's simulation also surfaced a real but
apparently unrelated `main`-side fragility (outer-optimiser divergence to `φ→1e65`-scale
degenerate fits at both loading scales, 2 of 12 seeds) that is not the per-site defect #510
targets and should not be read as evidence against the fix; it is flagged here because it
complicates "main reports converged" as a health signal on its own and may be worth a
separate look.

## Time

Setup + BB sweep (12+12 fits, ~30-60s/fit) ran well within budget. CMP base scale (6+6 fits,
~10-24s/fit) likewise. CMP 5x scale was the budget-driver (158-340s/fit) and was cut from 6
to 2 datasets as described above to keep the whole check under the 40-minute limit.

## Cleanup

`~/local-scratch/wholefit-main` and `~/local-scratch/wholefit-512` (both created by this
check) removed after the run. `~/local-scratch/GLLVM.jl-503-betabinomial-ovA-bb` (#510,
pre-existing, owned by another lane) was read/run only and left untouched. No pushes,
comments, PR edits, or merges were made; GitHub access was read-only throughout.
