# Overnight check of data/RD-FUNGI-BINOMIAL

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`. HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`. Likelihood code was left unchanged. The scoreboard, `GATES.md`, and the decision packet were left unchanged. No commit, push, merge, Totoro job, or full suite.

Verdict: **NO_PROVEN_BUG**.

`data/RD-FUNGI-BINOMIAL` is not admitted. The tolerances stay 0.0001 for beta, 0.0001 for `Lambda Lambda'`, and 1e-06 for the training linear predictor, as written in `docs/dev-log/core070/true-parity-latest/receipts/data/campaign/RD-FUNGI-BINOMIAL.json`.

## What the receipt already shows

The miss note `docs/dev-log/2026-10-20-p1-miss-fungi.md` is the comparison. One intercept fails: `traitsp54`, absolute difference `0.000123932525228021` (R `-5.55145011088071`, Julia `-5.551574043405938`). The `Lambda Lambda'` maximum is the `sp54` diagonal, absolute difference `0.00047443752671583184` (R `7.30290967368964`, Julia `7.303384111216356`). The training linear predictor maximum is site 133, trait `sp54`, absolute difference `0.00032827901245191526`. Log-likelihood is inside `1e-06`: R `-5792.18625245028`, Julia `-5792.186252388768`, absolute difference `6.151276465971023e-08`.

The 2026-10-20 capture rebuilt the returned Julia objective at this commit. Reconstruction absolute error was 0. The maximum absolute gradient was `7.5812079072257177e-06`. The captured log-likelihood was `-5792.1862523887676`. R returned convergence 0, `pdHess` true, message `relative convergence (4)`, and maximum absolute gradient `0.00386035726088885`. R's warning names `sp54` (prevalence 0.0433, max loading 2.62) as a runaway loading. Both parameter vectors have length 179 (`RD-FUNGI-BINOMIAL.json` records `n_par` 179; Julia's binomial covariate packing is 59 intercepts, 3 shared slopes, and 117 free loadings).

## What was checked in the binomial covariate path

Fungi is `gllvm(@formula(y ~ 1 + TEMPR + PRECIP + logAREA), Y, covdata; family = Binomial(), K = 2)`, which reaches `fit_gllvm_cov` (`src/formula.jl:385`). The family has no dispersion parameter (`src/families/covariates.jl:324`). The default link is logit (`src/families/covariates.jl:318`). For binomial logit, observed curvature and Fisher curvature are the same pointwise (`src/families/binomial.jl:29-32`): the weight is `n μ (1-μ)`. R uses `stats::binomial()`, which is the same logit, and `unique = FALSE`, which drops the trait-diagonal residual. Julia's `GllvmCovFit` estimates `Lambda Lambda'` only. The three slopes already sit inside 0.0001, so the shared-slope design is the design being compared.

A local probe, seed 20261020, used p = 6, n = 60, K = 2, q = 3, with one trait at the receipt intercept `-5.551574043405938` and loading `2.62` (`Lambda Lambda'` diagonal `6.8644`). It did not load `fungi_wide.csv` and did not refit the campaign cell.

At the receipt linear predictor `-11.810706390460352`, the binomial weight and the second derivative of the coded binomial log-density agree to `1.33e-16` for y = 1 and `2.54e-21` for y = 0. The same check at the receipt intercept and at `η = -27` (near the mean clamp) stays under `3.3e-17`.

`_marginal_loglik_offset` and `marginal_loglik_laplace` with the same offset agree to absolute difference 0. The covariate mode and the generic Laplace mode agree to absolute difference 0 on all 60 sites, with every covariate mode converged.

The analytic covariate gradient matches a central finite difference (`h = 1e-6`) of that same objective. On the rare intercept the absolute gap is `2.239390095581939e-8`. On the packed loading that is `Lambda[6,1]` the absolute gap is `2.8218348546715788e-8`. On the first slope the absolute gap is `4.203544179404162e-9`. The existing binomial covariate gate is `1e-6` (`test/test_cov_analytic_grad.jl:10`).

## Why those checks do not move the three bars

The two reported log-likelihoods already agree to `6.151276465971023e-08`. Assigning that whole scalar gap to the one failing intercept gives a curvature of about 8 along that coordinate (`2 * 6.151276465971023e-08 / 0.000123932525228021^2`). R's gradient of `0.00386035726088885` on a curvature of that size is a parameter step of a few ten-thousandths, which is the size of the beta gap (`0.00012393`), the `Lambda Lambda'` gap (`0.00047444`), and the linear-predictor gap (`0.00032828`). Julia's own gradient at the returned point is `7.5812079072257177e-06`, under the fitter's `g_tol` of `1e-5` (`src/families/covariates.jl:463`). The linear-predictor tolerance is `1e-06`. An intercept gap of `0.00012393` already exceeds that tolerance on every row of `sp54`, and the miss note's leftover after removing the intercept, up to `0.00020434648722389426`, is the latent part of the same shallow direction.

A likelihood correction that kept the log-likelihood agreement would have to move the optimum by about `1e-4` while changing the objective by less than `1e-7`. The probe shows the coded objective does answer to that coordinate when the gradient is still order 1: the same intercept shift, taken where the rare-intercept gradient is `-1.7269`, changes the probe log-likelihood by `0.00021401093883355315`. The campaign pair does not have that much log-likelihood left between them.

No binomial covariate formula error was found that the existing comparison would then pass. Likelihood code was not edited.

`data/RD-FUNGI-BINOMIAL` is not admitted.

NO_PROVEN_BUG
