# Sibling screen: silent inner-search failure / false-converged-flag (2026-09-26)

Screens the families NOT covered by `docs/dev-log/core070/class-audit-20260924/` (that
audit's `README.md`/`notes.md`/`fitters.txt` were read first, on
`origin/claude/lane-true-parity-finish-20260925`, and its method — reconstruct each
fitter's negative log-likelihood via its own public `*_marginal_loglik_laplace` wrapper,
FD-gradient at theta-hat, and restart from a perturbed/true start — is reused here) for
the same two defect classes found in #477/#479/#480:

- **Class A** — a per-site inner mode search stops silently without reaching its own
  optimum (proxy used here: does bumping the inner Newton `maxiter`/`tol` far past the
  fitter's own defaults, AT THE SAME theta-hat, find a lower negative log-likelihood?).
- **Class B** — `converged = true` is reported at a point that is not stationary (checked
  two ways: a scale-aware FD gradient norm at theta-hat, and whether a restart from a
  perturbed theta-hat or from the true generating parameters finds materially higher
  log-likelihood).

Families screened: mixed-family, beta-binomial, NB1 (per-trait, non-grouped), GP-1,
COM-Poisson, ordered beta, Student-t, ordinal (per-trait). All eight are implemented in
`itchyshin/GLLVModels.jl` at commit `d4da31544` (2026-09-26, `main`) — none skipped as
unimplemented. Per the task's out-of-scope list, the two-part/hurdle/delta families were
NOT touched (a builder is fixing #484 there) and grouped-dispersion NB1/Gamma/Beta/Tweedie
routes already covered by the prior audit were not re-run.

## Headline result

Two families reproduce the exact #480/#485 signature (`converged = true` reported at a
point that demonstrably is not a stationary point, recoverable by a plain restart):
**ordered beta** (severe — 7/10 fits, log-lik gaps up to +2859) and **ordinal per-trait**
(confirmed but modest — 2/10 fits, gaps up to +0.18). **Mixed-family** shows the same
symptom at a smaller, more marginal magnitude (2/10 fits clear the >0.1 restart bar, up to
+1.31; a further 4/10 show an elevated scale-aware gradient that restart did NOT
corroborate). **COM-Poisson** shows a separate, unconfirmed anomaly (2/10 fits with an
enormous FD-gradient at theta-hat) that restart does NOT corroborate as a missed optimum —
flagged for follow-up, not asserted as a bug. NB1, beta-binomial, GP-1 and Student-t came
back clean.

## Method

Julia 1.10.12 (`juliaup`), `Optim` 1.13.3, `GLLVModels.jl` @ `d4da31544` (2026-09-26,
`main`), single-threaded per fit (`JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`, ≤4 cores
on Totoro). For each family: 10 datasets (seeds 2001–2010), p=5 traits (p=4 for
mixed-family: Poisson/Binomial/Gamma/Beta), n=40 sites, K=2 latent dimensions, simulated
from the family's own generative model at parameters chosen to sit well inside its
feasible region (no boundary stress-testing — this screen targets the *default*
simulate-fit-restart path a user would actually hit). Per fit:

1. Fit with the family's public named fitter at its defaults (e.g. `fit_nb1_gllvm(Y; K=2)`).
2. Reconstruct its negative log-likelihood by calling the SAME public
   `*_marginal_loglik_laplace` wrapper the fitter itself calls internally, with the same
   packing, `link`, `hessian` (read off the fit object where applicable) and Newton
   `maxiter`/`tol`. Validity check: `|negll(theta_hat) - (-loglik)|` — 0.0 to machine
   precision for all 100 fits across all 8 families (confirms the reconstruction is
   faithful, not an independent-but-wrong objective).
3. Class B (local): two-sided FD gradient of the reconstructed negll at theta-hat;
   `gscaled = max_i |g_i| * (1 + |theta_hat_i|)` (scale-aware, so a large-magnitude
   dispersion parameter doesn't mask a real residual and a tiny one doesn't manufacture a
   false positive); flag if `converged && gscaled > 1e-3` (100x the fitter's own
   `g_tol=1e-5`).
4. Class A (inner-search): re-evaluate the SAME negll at the SAME theta-hat with the inner
   Newton `maxiter` raised 10x and `tol` tightened 1000x (100→1000 iters, 1e-9→1e-12);
   `dinner = negll_default(theta_hat) - negll_tight(theta_hat)`; flag if `dinner > 0.05`
   nats (should be ≈0 if the per-site mode search already reached its own optimum at
   default settings).
5. Class B (global / restart): re-optimize the SAME negll with the SAME optimizer
   (`Optim.LBFGS` + `BackTracking(order=3)`, `autodiff=:finite`, `g_tol=1e-5,
   iterations=500` — or, for mixed-family, the fitter's own `ForwardDiff`+`only_fg!` setup)
   from (a) a perturbed theta-hat (`theta_hat .+ (0.3|theta_hat|+0.1).*randn(...)`) and
   (b) the true generating parameters (packed the same way); `drestart = max(restart
   log-lik) - fit log-lik`; flag if `fit.converged && drestart > 0.1`.

A family is only flagged CLEAN-vs-not on a criterion when **≥2 of 10** fits show it, per
the task's instruction not to call a bug from one dataset.

Reconstruction details that matter for reading the raw log: GP-1's fitter profiles a grid
over the shape parameter α + a Brent refinement, then a final L-BFGS over (β,Λ) at that
α — the reconstructed negll/restart here holds α fixed at the FITTED value and only
perturbs (β,Λ), i.e. it audits that final inner optimization, not the whole grid+Brent
chain. COM-Poisson and ordered beta simulate via hand-written inverse-CDF/compound
samplers copied from the family file's own documented pmf/pdf (no built-in `rand` exists
for either in `Distributions.jl`); GP-1 reuses the package's own `_rand_gp1` sampler.

## Results table

| family | fits | claimed converged | scale-aware grad flag (>1e-3) | restart found better (>0.1), max gap | inner-search flag (Δ>0.05 nats) | verdict |
|---|---|---|---|---|---|---|
| negbin1 (per-trait) | 10 | 10/10 | 0/10 | 0/10, max +2.0e-11 | 0/10 | **CLEAN** |
| gp1 | 10 | 10/10 | 0/10 (max gscaled 2.1e-4) | 0/10, max +1.2e-3 | 1/10 (max 0.077 nats) | **CLEAN** (soft trend, below 2/10 bar) |
| compoisson | 10 | 10/10 | 2/10 (gscaled 9e8, 2e14) | 0/10, max +0.059 (one restart was *worse*, -4.94) | 0/10 | **CLEAN vs restart bar; anomaly flagged** (see below) |
| beta_binomial | 10 | 10/10 | 0/10 | 0/10, max +5.3e-12 | 0/10 | **CLEAN** |
| ordered_beta | 10 | 10/10 | **7/10** (gscaled up to 1.8e4) | **7/10**, max **+2859** | 0/10 | **SAME CLASS AS #480/#485** |
| studentt | 10 | 6/10 (4/10 honestly non-converged: ν-boundary self-guard) | 0/10 | 0/10, max +1.4e-10 | 0/10 | **CLEAN** (boundary self-correction working as designed) |
| ordinal (per-trait) | 10 | 10/10 | 3/10 (gscaled up to 2.4e6) | **2/10**, max **+0.18** | 0/10 | **SAME CLASS AS #480/#485** (modest) |
| mixed | 10 | 10/10 | 6/10 (gscaled up to 1.4e8) | **2/10**, max **+1.31** | 0/10 | **SAME CLASS AS #480/#485** (marginal — 4/10 gradient flags unconfirmed by restart) |

Per-fit rows (all 100) are in `sibling-screen-2026-09-26-raw/screen_full.log`.

## Per-family notes

**negbin1** (`fit_nb1_gllvm`, per-trait dispersion, non-grouped): the family covered by
the prior audit's SUSPECT list was the *grouped* route
(`fit_nb1_gllvm_grouped`/`grouped_dispersion.jl:1382`); this per-trait route
(`negbin1.jl`) is clean across all three checks.

**gp1**: one dataset (seed 2002) shows `dinner=0.077` nats and three more show smaller
positive `dinner` (0.02–0.05) — a consistent-direction but small residual suggesting the
per-site Newton search is occasionally left slightly short of its own optimum at the
default `newton_maxiter=100`. It never reaches the 2/10 flagging bar and never shows up as
a wrong `converged` flag or a restart-recoverable gap (max +1.2e-3), so it is reported as
a soft trend worth a cheap fix (bump default `newton_maxiter`) rather than a confirmed
Class A failure.

**compoisson**: seeds 2003 and 2005 show FD gradients of 9e8 and 2e14 respectively at a
fit claiming convergence — nominally a severe Class B signature. But restarting from a
perturbed theta-hat and from the true parameters does NOT find a better optimum (seed
2003's restart is actually *worse*, -4.94; seed 2005's is a marginal +0.059, below the
0.1 bar), and the reconstructed negll matches the fitter's own log-lik to machine
precision (`valid|Δ|=0`). The likely explanation is not a missed optimum but a kink in the
objective surface itself: `compoisson_logz` (the CMP log-normalizer) switches from a
truncated series sum to a Shmueli asymptotic formula once the series mode
`j* = λ^(1/ν)` exceeds 80% of a hard-coded term cap (`com_poisson.jl`, 2026-08-28 fix) —
a two-sided finite-difference step straddling that branch boundary can produce an
enormous but spurious apparent gradient without the true objective actually having a
large gradient there. This is flagged as a **candidate new anomaly for follow-up**, not
asserted as a Class A/B bug — the restart evidence (the task's own bar for calling a bug)
does not support it, and I have not verified whether theta-hat for these two seeds is
actually near the 80%-of-cap crossover (that check would confirm or rule out the
mechanism and takes ~5 minutes with `GM.compoisson_logz` called directly).

**ordered_beta** — the headline finding. 7 of 10 fits report `converged = true` while the
reconstructed FD gradient at theta-hat is 3 to 8 orders of magnitude above `g_tol`, and a
plain restart (perturbed start or true-parameter start, SAME optimizer settings the
fitter itself used) finds a substantially better optimum every time this happens — gaps
of +21.7, +75.2, +32.3, +67.3, +199.7, +295.2, and, worst, **+2859 log-lik units** (seed
2005, where the reported loglik is -3041 and the true optimum is near -182). This is not
multimodality: a genuinely converged alternative mode would have a small gradient at its
own point; here the gradient itself is enormous (up to 1.8e4), meaning the reported point
is nowhere near stationary. `fit_ordered_beta_gllvm` calls the shared `_fit_verdict`
helper exactly like ~85 other fitters; this looks like the same false-converged-flag class
as #480 (Beta grouped fit) and/or #485 (`_fit_verdict` converged on a zero-length step,
currently being fixed) landing in a family neither of those fixes reached. I did not
instrument Optim's own `stopped_by.{x,f,g}` flags (unlike the prior audit's
`fit_verdict_classB_probe.jl`), so I can describe the symptom precisely but not which of
Optim's three stopping criteria fired — that instrumentation would take ~15 minutes to
add and should be done before a fix is written.

**studentt**: 4 of 10 fits report `converged = false` because the estimated ν ran to the
flat Gaussian-limit boundary (ν > 1e6) and the family's own boundary guard
(`_studentt_nu_boundary`) forces `converged = false` regardless of what Optim itself
reported — this is the SAME kind of honesty check #480 introduced for Beta, already
present and working correctly here. Among the 6 fits that did converge normally, all three
checks are clean.

**ordinal (per-trait)**: 2 of 10 fits (seeds 2003, 2009) show a restart-confirmed gap
(+0.18, +0.13) alongside million-scale gradients at `converged = true`; a third (seed
2006) has an equally large gradient (2.16e6) but its restart found a very slightly *worse*
point (-0.024), so it does not clear the restart bar even though the gradient evidence
alone is just as damning. Smaller magnitude than ordered_beta, same signature, same shared
`_fit_verdict` caller.

**mixed**: 2 of 10 fits (seeds 2003, 2005) clear both bars (gradient + restart-confirmed
gap of +0.47, +1.31). A further 4/10 (seeds 2004, 2006, 2007, 2010) show an elevated
scale-aware gradient (0.01–0.05, far smaller than ordered_beta/ordinal's — these are three
orders of magnitude smaller) that restart does NOT confirm as a missed optimum (deltas
near zero or slightly negative). Given `fit_mixed_gllvm` uses a `ForwardDiff` analytic
gradient (not finite-difference like every other family here) through a dense joint
Laplace over mixed link scales, some of that gradient inflation may be ordinary
cross-family scale imbalance rather than the same defect — reported as a marginal,
partially-confirmed member of the same class rather than with ordered_beta's confidence.

## Distinguishing defect from genuine multimodality

None of the flagged families show the "genuine multimodality" signature (small gradient
at BOTH the reported point and a materially different restart point, i.e. two real local
optima). In every restart-confirmed case (ordered_beta, ordinal, mixed), the reported
point has a large gradient — it is not a stationary point at all, so it cannot be a real
(if suboptimal) mode. This is squarely a fitter defect, not something a correctly
implemented alternative engine would also report as "converged."

## Reproduce

```bash
ssh -S ~/.ssh/cm-snakagaw@totoro.biology.ualberta.ca:22 snakagaw@totoro.biology.ualberta.ca
cd ~/hsq_work/gllvm-sibling-screen-20260926
# one-time environment setup (already done; env/ has GLLVModels dev'd + Optim/Distributions/ForwardDiff/SpecialFunctions):
#   ~/.juliaup/bin/julia +1.10.12 --startup-file=no setup_screen.jl
export JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
# smoke test (1 dataset/family):
SCREEN_N=1 ~/.juliaup/bin/julia +1.10.12 --startup-file=no --project=env sibling_screen.jl
# full screen (10 datasets/family, ~29 min wall time observed):
SCREEN_N=10 ~/.juliaup/bin/julia +1.10.12 --startup-file=no --project=env sibling_screen.jl
# single family, e.g. ordered_beta only:
SCREEN_N=10 SCREEN_ONLY=ordered_beta ~/.juliaup/bin/julia +1.10.12 --startup-file=no --project=env sibling_screen.jl
```

`sibling_screen.jl` and `setup_screen.jl` are copied verbatim into
`sibling-screen-2026-09-26-raw/`. The Totoro worktree (`~/hsq_work/gllvm-sibling-screen-20260926/repo`,
commit `d4da31544`) and its `env/` were left in place (not cleaned up) in case a follow-up
session wants to re-run a single family or add the `stopped_by` instrumentation noted
above for ordered_beta; no process was left running.

## Wall time

Totoro compute: environment setup ~2 min, 1-dataset smoke test (all 8 families)
~3.2 min (one bug found and fixed in the mixed-family route — see below), full 10-dataset
screen 29.2 min (`total elapsed 1750.4 s` in the log) — roughly 35 min of Totoro compute
against the 90-minute budget, every per-family run well under the 30-minute-per-run cap
(compoisson was the slowest family at ~9 min for its 10 datasets, driven by the CMP
log-normalizer's series summation).

One bug in the screening harness itself, not the package: the first mixed-family smoke
test showed `valid|Δ|=270` (my reconstructed negative-log-likelihood was missing its sign)
— fixed before the full run; all 100 fits across all 8 families now show `valid|Δ|=0.0` to
machine precision, confirming every reconstruction matches the fitter's own reported
log-likelihood exactly.
