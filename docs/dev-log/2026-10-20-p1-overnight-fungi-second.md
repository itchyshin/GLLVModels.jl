# Second check of data/RD-FUNGI-BINOMIAL

Date: 2026-10-21. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl`. HEAD stayed `5cfc6754212588222779ccfd66ad7000375cf8d5`. No source file was edited. No tolerance, scoreboard verdict, `GATES.md`, or receipt was edited. Nothing was committed, pushed, or merged. Totoro was not used. The full suite was not run. No Julia process was started.

NO_NEW_BUG

`data/RD-FUNGI-BINOMIAL` is not admitted. The tolerances stay 0.0001 for beta, 0.0001 for `Lambda Lambda'`, and 1e-06 for the training linear predictor.

## What this pass checked

The first overnight note, `docs/dev-log/2026-10-20-p1-overnight-fungi.md`, already compared the binomial logit weight with the second derivative, the covariate Laplace with the generic Laplace, and the analytic gradient with a central difference. Those three comparisons were not run again.

This pass looked for a different defect: trial counts, covariate column order, species order, or which linear predictor the receipt compares. The inputs were the source that builds the fungi cell and the two raw engine files `fungi_R.json.gz` and `fungi_J.toml.gz`. Both raw files carry data sha256 `498b84b93d2759785272cf33e7620f556033cb28c8a017596e146d6b5318b1bb`. `fungi_wide.csv` is not in this worktree.

## Trial counts

`tools/true_parity/campaign/gen_data.R` writes fungi as presence or absence: `Yn` is `(Y[sites, sel] > 0) * 1`, then species that are absent on the 300 sites are dropped. The campaign formula is a single column, `value`, with no `weights` and no `cbind(successes, failures)`.

On that shape `R/fit-multi.R` sets `n_trials` to 1 on every row. The same file aborts when a binomial response falls outside `0 <= y <= n_trials`. R returned a log-likelihood, so every stored response was 0 or 1.

Julia's covariate fit sets the trial matrix to all ones when `N` is omitted (`src/families/covariates.jl`). The formula route does not pass `N`. With covariates in the formula it also refuses `weights` (`src/formula.jl`). `run_J.jl` does not pass `N`.

The receipt log-likelihoods are R `-5792.18625245028` and Julia `-5792.186252388768`, absolute difference `6.151276465971023e-08`. The raw Julia file stores the same Julia value. Both engines therefore used trial count 1.

## Covariate column order

R's `beta_names` are the 59 trait intercepts and then `TEMPR`, `PRECIP`, `logAREA`. Julia stores `beta[1]` through `beta[59]`, then `gamma[1]`, `gamma[2]`, `gamma[3]`. The fit has 179 parameters: 59 intercepts, 3 slopes, and 117 free loadings, so the slope block has three columns.

`_build_site_modelmatrix` drops the intercept `ConstantTerm` and keeps the remaining terms in formula order. The fungi formula is `y ~ 1 + TEMPR + PRECIP + logAREA`, so those columns are `TEMPR`, `PRECIP`, `logAREA`. `run_J.jl` also builds a design in that same `covs` order and passes it to `predict`. The stored slopes show that the fitted `gamma` is in that order too:

| Slope | R | Julia | Absolute difference |
| --- | ---: | ---: | ---: |
| TEMPR | 0.0974470390914102 | 0.0974518819226022 | 4.842831191995134e-06 |
| PRECIP | -0.0486696692011703 | -0.04866930796909813 | 3.612320721713136e-07 |
| logAREA | 0.0221938547822144 | 0.022195839472086278 | 1.9846898718783867e-06 |

The six permutations of the three slopes were compared. Only this order stays under 0.0001. The next permutation, swapping `PRECIP` with `logAREA`, has maximum absolute difference `0.07086550867325658`. The other four sit between `0.07525802714038779` and `0.1461215511237725`.

## Species order

Both raw files list `sp01` through `sp59` in that order. R's `LLt_trait_names` are that same list. Julia builds `Y` from the CSV header after dropping `site` and the three covariates, which is the header order `run_R.R` uses for `trait` levels. The fitter stores `beta` in the row order of `Y`.

The intercept gap in that order has maximum `0.000123932525228021` at `traitsp54`. All 1711 transpositions of two species were applied to Julia's intercepts. None reduced that maximum. The same 1711 transpositions were applied to Julia's `Lambda Lambda'`. None reduced its maximum, which is the `sp54` diagonal: R `7.30290967368964`, Julia `7.303384111216356`, absolute difference `0.00047443752671583184`. Both stored matrices are symmetric to 0.

## Which linear predictor

R stores `as.numeric(fit$report$eta)`. In the TMB template that vector starts as `X_fix * b_fix` plus the offset. For this cell the offset is zero, `unique = FALSE` adds no trait-diagonal residual, and AGHQ is off, so the latent block adds `Lambda_B(t, k) * z_B(k, s)` at the Laplace mode. The long data are site-major with the trait fastest, and `eta` follows that row order.

Julia stores `vec(predict(fit, Y, X; type = :link))`. That method is `beta + X * gamma + Lambda * zhat`, with `zhat` the per-site Laplace mode and no sign rotation (`src/postfit.jl`). Column-major `vec` of a traits-by-sites matrix is the same site-major, trait-fastest order.

The two stored vectors have length 17700, which is `300 * 59`. In that order their correlation is `0.9999999999042801`. The maximum absolute difference is `0.00032827901245191526`, at flat index 7900: zero-based site 133, trait `sp54` (R `-11.8103781114479`, Julia `-11.810706390460352`). The median absolute difference is `5.708180433883214e-06`. Reading Julia's vector as trait-major instead raises the maximum to `13.918357084888553`. Shifting it by one site raises the maximum to `13.524195810150587` in one direction and `13.524279451161352` in the other.

The species means of the stored layout track the intercepts (correlation `0.999055765215452`). The trait-major layout does not (correlation `0.015354793235830225`, maximum mean-versus-intercept gap `3.3978302098420774`). Within `sp54` the R linear predictor has standard deviation `2.4274497096299874`, so the stored vector includes the latent scores. A fixed-effect-only predictor would sit near the intercept `-5.55145011088071`. Subtracting each species' intercept gap from the paired linear predictors leaves a maximum absolute remainder of `0.00020434648722389426`.

## What stays outside the bars

The aligned beta maximum is still `0.000123932525228021`, above 0.0001. The aligned `Lambda Lambda'` maximum is still `0.00047443752671583184`, above 0.0001. The aligned training linear predictor maximum is still `0.00032827901245191526`, above 1e-06. No reordering of trials, covariate columns, species, or the linear predictor puts those three comparisons inside the current bars.

`data/RD-FUNGI-BINOMIAL` is not admitted.

NO_NEW_BUG
