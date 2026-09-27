# Auto-d: morning report for Shinichi (overnight 2026-09-26/27)

Estimating d automatically works for Gaussian, Poisson and negative binomial
data with BIC on log(number of sites) and a guard against broken fits; it does not yet work for
binary data, where most fits at d ≥ 2 run away. Nothing is merged or pushed; four decisions need you.

## Your decisions (G1), each with a recommendation

1. **Default criterion.** Recommend BIC with log(number of sites) (`:bic_sites`). Recovery grid,
   mean exact recovery: Gaussian 0.95, Poisson 0.999, NB 0.90. The current convention, BIC with
   log(sites × species), picks too few dimensions at small n (Gaussian n = 30: 0.21 vs 0.53).
2. **Julia API.** Recommend: omitting K runs the guarded sweep and returns the chosen fit, with a
   one-line message. Today omitting K throws, so no working call changes. Built on the branch.
3. **R API.** Today `latent()` without `d` silently means d = 1, so making it automatic would
   change existing users' fits. Recommend `d = "auto"` first, and switching the default later with
   a NEWS warning.
4. **Binary data.** Recommend: for Bernoulli data, return the candidate table and do not auto-pick
   until a ridge-stabilised sweep with an unpenalised criterion exists (a follow-up). Evidence: 70–87%
   of binomial fits at d ≥ 2 do not converge and all of those are runaways; a loading ridge makes
   every fit healthy but its penalised logLik then pushes BIC to d = 1 (R experiment, 10 reps × 4
   cells, loadings 0.8 and 1.5).

## What was built (branches only, not pushed)

- **Julia** `claude/lane-auto-d-20260926` (GLLVModels.jl): `select_lv` records every candidate d
  with a status; never chooses a failed, runaway or non-monotone fit; retries a rejected d from the
  last accepted solution where the fitter accepts start values; runaway detector (latent SD on the
  link scale > 10; binomial loading ratio ≥ 25); `:bic_sites`; `fit_gllvm` without K estimates it;
  chi-bar-squared docstring corrected. 81 tests; explicit-K fits bit-identical to before.
- **R** `claude/lane-auto-d-r-20260926` (gllvmTMB): the same guard and detector in `select_lv()`,
  136 expectations pass.
- **Independent review** (Opus statistics, Sonnet code, Sonnet tests): all "pass with required
  fixes"; every fix applied or stated as a limit.
- **Decision memo** `docs/design/74-auto-latent-dimension.md` on the Julia branch.
- **Literature**: NotebookLM `c2c564e3` (34 sources), vault note
  `dr-auto-d-latent-dimension-selection`. Nobody has benchmarked d-selection on count or binary
  GLLVMs; our grid is the first such evidence.

## Problems found on the way (flagged as separate tasks, not fixed here)

- **NB fitter lands in poor optima** at d = 2–4 while reporting converged (one case: d = 3 fit
  2400 logLik units below d = 5). Task chip "Fix NB per-species route stuck in poor optima".
- **Julia's default Gaussian fit has no species intercepts**: shifting data by 5 drops logLik by
  335. Task chip "Add species intercepts to default Gaussian fit_gllvm". Changes existing fits, so
  needs your sign-off.
- **gllvmTMB `latent()` adds a per-species residual (Ψ) by default**; GLLVModels does not. They
  agree like-for-like only with `unique = FALSE`.

## What it does NOT cover

HSquared; the ordered factor LASSO (parked as its own lane); grouped/phylo/row-effect routes;
confint/summary do not yet print the "conditional on the chosen d" caveat; the full grid table is
final only when the last DRAC tasks land.
