# Ordered-beta restart gap: bounded-likelihood verification (2026-09-26)

**Lens:** is the ordered-beta likelihood unbounded/degenerate along the restart
direction, making the `+2859` log-lik gap a spike rather than a genuine optimum?

**Verdict: REFUTED.** The restart is a genuine better stationary point, not a
degenerate or unbounded artefact. The `converged = true` verdict at the FIRST
fit is still wrong (huge gradient there), independent of this lens.

## 1. Log-likelihood contributions (from `src/families/ordered_beta.jl`)

Per-trait conditional log-density, `μ = σ(η)` (identity-on-η link, per-trait
Laplace, cutpoints `c0 < c1`, precision `φ`):

```
y = 0        : log σ(c0 − η)
y = 1        : log σ(η − c1)
0 < y < 1    : log(σ(η−c0) − σ(η−c1)) + logpdf(Beta(μφ, (1−μ)φ), y)
```

(`ordered_beta.jl:78-103`; the interior branch was rewritten 2026-08-26 into a
`logσ`/`log1mexp` form to fix an underflow-to-`-Inf`, not to change bounds.)
The fit driver (`ordered_beta.jl:297-356`) packs `θ = [β; pack_lambda(Λ); c0;
Δ = log(c1−c0); logφ]` and runs `Optim.LBFGS` + `BackTracking` on the negative
Laplace marginal, same as every other family here.

## 2. Analytic: can it diverge to +∞?

- **Point masses** (`y=0`, `y=1`): `log σ(·) ≤ 0` always — bounded above by 0,
  unbounded only *below* (as the argument → −∞). No positive-divergence risk.
- **Interior logmass** `log(σ(η−c0) − σ(η−c1))`: a log-probability, `≤ 0`
  always. No positive-divergence risk from this piece either.
- **Interior Beta term** `logpdf(Beta(μφ,(1−μ)φ), y)`: this is the only piece
  that can grow without bound. A standard Laplace/Stirling expansion in `φ`
  gives, for fixed `y, μ ∈ (0,1)`,

  ```
  log f(y; μ, φ) ≈ −φ·KL(μ‖y) + ½ log φ + O(1),   φ → ∞
  ```

  (`KL(μ‖y) = μ log(μ/y) + (1−μ) log((1−μ)/(1−y)) ≥ 0`, i.e. the Beta term is
  a peaked density collapsing onto its mean, exactly like a Gaussian density
  evaluated at its own mean as its variance → 0.) So:
  - if `μ ≠ y` (imperfect fit), the `−φ·KL` term dominates and the term → −∞
    linearly in `φ` — cranking precision up while missing the data point is
    catastrophic, not beneficial.
  - if `μ → y` (η tracks the observation), the `+½ log φ` term survives and
    the term diverges to `+∞`, but only **logarithmically** in `φ`.

  This positive direction is real in principle — it is the ordinary "spike"
  pathology any Beta/Gaussian-type continuous likelihood has when its
  precision is unconstrained and its location can track a data point exactly.
  But the Laplace correction (`−½ z'z − ½ log det(Λ'WΛ + I)`) taxes exactly
  this direction: the per-trait weight `W_t = −∂²logp/∂η²` for the Beta piece
  also scales with `φ`, so `log det(Λ'WΛ+I)` grows roughly `K·log φ` per site
  whenever several traits are simultaneously well-fit — netting against the
  `~½·(#interior traits)·log φ` gain. With `p=5` traits and `K=2` latent
  dimensions (this screen's design), exactly matching `μ=y` for more than `K`
  traits at one site is a codimension-deficient event (`Λz = logit(y)−β` is 5
  equations in 2 unknowns), so this direction is not generically free —
  **conclusion from source alone: a genuine unbounded direction exists in the
  model class as written, but nothing in the density or packing rules it out
  a priori; it has to be checked numerically.**

## 3. Numerical: trace along the line fit → restart, and beyond (seeds 2005, 2002)

Reconstructed the SAME negll (`ordered_beta_marginal_loglik_laplace`, same
`maxiter=100/tol=1e-9` the fitter uses), re-ran the restart from the true
generating parameters and from a perturbed θ-hat (identical to
`sibling_screen.jl`), kept the actual **minimizer** (not just its log-lik),
and evaluated negll at 20 points on `[θhat, θrestart]` plus 5 points extending
to `1.5×` past the restart.

**Seed 2005 (the worst case, reported gap +2859, first fit `ll=-3041.3`,
restart `ll=-182.8`):**

```
t=0.000  ll=-3041.35  φ=5.09     <- first fit (claimed converged=true)
t=0.421  ll= -579.81  φ=6.55
t=0.632  ll= -246.26  φ=7.43
t=0.895  ll= -192.44  φ=8.70
t=1.000  ll= -182.81  φ=9.26     <- restart optimum
t=1.100  ll= -196.61  φ=9.83     <- past the restart: already falling
t=1.300  ll= -258.03  φ=11.08
t=1.500  ll= -557.99  φ=12.50    <- falling steeply, not diverging
```

**Seed 2002 (gap +295, first fit `ll=-489.8`, restart `ll=-194.6`):**

```
t=0.000  ll=-489.84  φ=2.06      <- first fit
t=1.000  ll=-194.64  φ=7.48      <- restart optimum
t=1.100  ll=-198.96  φ=8.51      <- falling immediately past t=1
t=1.500  ll=-265.61  φ=14.27     <- falling steeply
```

Both traces rise **monotonically** from the first fit to the restart point,
then **fall** immediately past `t=1.0` in both directions tested. Neither
keeps climbing toward `1.5×` — the opposite of an unbounded direction. `φ`
does increase along the path (as the analytic section says it must, since the
better fit needs somewhat higher precision) but only by a **finite, modest
factor** (1.8× for seed 2005, 3.6× for seed 2002), not diverging toward
infinity, and it turns around and falls again immediately past the restart
point rather than continuing to climb.

**Gradient check at the restart point** (same scale-aware FD gradient the
screen used at the first fit): the restart is a genuine near-zero-gradient
stationary point, and `Optim.converged(...)` on the restart run itself is
`true`:

```
seed=2005  gscaled@fit1=1.82e+04   gscaled@restart=8.19e-06   Optim.converged(restart)=true
seed=2002  gscaled@fit1=2.34e+03   gscaled@restart=7.18e-06   Optim.converged(restart)=true
```

`gscaled` drops from 3-4 orders of magnitude above `g_tol=1e-5` at the first
fit to below `g_tol` at the restart — the restart lands on an actual
stationary point, not another false-convergence spot.

## 4. Parameters that moved most (seed 2005, the worst case)

```
Δβ (5 traits)      = [0.88, 0.12, 1.09, 1.01, 3.07]   <- one trait's intercept moved a lot (β5)
max |ΔΛ|           = 5.79                              <- loadings moved substantially
Δc0 = -0.35, Δc1 = +0.31   (cutpoint GAP: 1.677 → 2.332 — widened, did not collapse)
φ: 5.09 → 9.26 (×1.82)                                 <- moved, but nowhere near a boundary
```

Seed 2002: `Δβ` up to 1.47, `max|ΔΛ|=4.59`, cutpoint gap `1.079 → 1.782`
(widened), `φ: 2.06 → 7.48` (×3.6). In both cases the cutpoint gap **widens**
rather than collapsing onto a data point, and no parameter runs to a boundary
(`c0`/`c1` stay well inside a plausible range, `φ` stays under 10, nowhere
near the huge values a genuine "precision → ∞" spike would need to move
`0.5 log φ` by thousands of nats).

## 5. Conclusion

- **Not unbounded / not a spike.** A genuine `+∞` direction exists in the
  model class in principle (Beta density concentrating on a data point as
  `φ → ∞`), but it requires `φ` to diverge while the low-rank fit gets
  arbitrarily good — neither happens here. `φ` moves by a small finite factor
  and the log-lik peaks exactly at the restart point in both directions
  tested (short of it and 50% past it).
- **Not a cutpoint collapsing onto a 0/1 point mass.** `c1 − c0` widens at the
  restart in both seeds checked, the opposite of collapse.
- **Not multimodality either** (per the screen's own framing) — the first fit
  has an enormous gradient (`gscaled` up to `1.8e4`), so it was never a
  stationary point to begin with; only the restart is.
- **The restart is a genuine, better local optimum**, confirmed by
  `Optim.converged = true` at the restart and a scale-aware FD gradient at
  the restart 3-4 orders of magnitude below `g_tol`.
- **`converged = true` at the first fit is still wrong**, and for the same
  reason the screen originally flagged: the reported point is nowhere near
  stationary (`gscaled` up to `1.8e4` against `g_tol=1e-5`). This lens does
  not change that verdict — it only rules out "the restart itself is an
  artefact" as the explanation, leaving the false-converged-flag itself (the
  same `_fit_verdict`/#480/#485-class bug the screen named) as the standing,
  now doubly-confirmed defect.

## Compute record

Totoro (`ssh -S ~/.ssh/cm-snakagaw@totoro.biology.ualberta.ca:22`), existing
`ControlMaster` socket, no fresh login. Own directory
`~/hsq_work/obeta-verify-boundedness-20260926/` (copied the screen's `env/`
only — `Manifest.toml` already points at the screen's `repo/` by absolute
path, read-only, never modified; the screen's directory itself was not
touched). `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`. Estimate before
running: a few minutes (2 datasets × [fit + 2 restarts + 25-point trace +
1 gradient check], each fit/restart ~1s per the screen's own log). Actual:
two runs, `13.2s` and `14.7s` wall time — both foreground, completed, nothing
left running (verified via `ps aux` after). Well under the 30-min/run and
60-min/job budgets.

Scripts: `obeta_verify.jl` (trace + parameter diffs), `obeta_verify2.jl`
(restart-convergence + gradient-at-restart check), both in
`~/hsq_work/obeta-verify-boundedness-20260926/` on Totoro.
