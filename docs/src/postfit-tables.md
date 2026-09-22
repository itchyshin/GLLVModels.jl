# Post-fit tables and prediction

Start here when you want to summarise a fitted model, predict missing responses,
or examine how a result changes with the assumed cross-trait correlation.
Check the accepted fit type in each function's reference entry below: several
helpers accept only `GllvmFit`, the Gaussian model with a shared residual
variance, even when the corresponding R function accepts other families.

## Tidy tables

`tidy(fit, Y)` returns a parameter table for a Gaussian `GllvmFit`.
Its `effects` keyword selects `:fixed` (fixed effects, the default),
`:ran_pars` (random-effect standard deviations), or `:cutpoint` (ordinal
thresholds, empty for this Gaussian model).
`summary(fit, Y)` returns a [`GllvmSummary`](@ref) with fit information,
fixed-effect estimates, and covariance summaries.
Supply the same response matrix `Y` used for fitting. If you fitted covariate
effects, also supply the original design array as `X = X`; omitting it can
produce `NaN` standard errors without an error message.

`deviance(fit)` is `-2 * loglikelihood(fit)`, the standard model-comparison
statistic, defined across the full family surface.

## Loadings rotation

`rotate_loadings(fit, Y)` (varimax/promax) and its tidy wrapper
`extract_rotated_loadings_table(fit, Y; loading_scale = :raw | :standardized)`
accept `GllvmFit` at `level = :unit`. They do not accept `TwoLevelFit`.

## Prediction and imputation

Use `predict_missing(fit, Y; mask = mask, type = :response)` to predict the
cells marked `false` in a Boolean mask with the same shape as `Y`.
Rows are traits and columns are sites; the returned `row`, `col`, and `est`
vectors identify those cells and their predictions. The default `type = :link`
returns predictions on the link scale. The default `mask = nothing` treats all
cells as observed, so it selects no missing-cell predictions. Supply the mask
explicitly; it is not recovered from the fit or inferred from `Y`.

The fitted model's `predict` method must accept `mask`. This is supported for
Gaussian models fitted with adaptive Gaussian quadrature and dense-Laplace
non-Gaussian models such as Binomial. Expected-category predictions for ordinal
responses and prediction standard errors are not available through this helper.

`predict_cross_covariance(fit, K)` returns the cross-covariance implied by a
supplied kernel matrix `K`. Its output does not record `rho` or whether `K`
already includes that multiplier; retain that information with your analysis.
`imputed(fitmi, x)` returns point estimates for supported Gaussian models fitted
with full-information maximum likelihood. It does not compute conditional
standard errors; every row reports `status = :se_not_computed`.

## Sensitivity: cross-trait correlation profiling

Use `profile_cross_rho(A_H, A_P, W, refit)` to refit a model at a grid of
candidate cross-trait correlations `rho`. You supply a function `refit(K, rho)`
that fits your model at each candidate. `profile_cross_rho_ci` then interpolates
a confidence interval from the resulting `rho` and `delta_deviance` vectors.

For example, this small hypothetical profile illustrates the interval calculation
without fitting a model:

```@example postfit_profile
using GLLVModels

rho = [-0.8, -0.4, 0.0, 0.4, 0.8]
delta_deviance = [6.0, 2.0, 0.0, 2.0, 6.0]
ci = profile_cross_rho_ci(rho, delta_deviance)
(ci.lower, ci.upper, ci.lower_bounded, ci.upper_bounded)
```

If either `*_bounded` flag is `false`, that end of the interval is the edge of
the supplied grid. Extend the grid if you need to locate the threshold crossing.

## Coevolution modules

`extract_coevolution_modules(Sigma_shared)` runs an SVD-based module
decomposition (`Σ_row^(-1/2) Γ Σ_col^(-1/2)`) on any shared covariance matrix
supplied positionally with `row_traits`/`col_traits` — `scale = :shape` only;
there is no stored ρ for a `scale = :effect` variant.

## Balanced-design simulation

`simulate_unit_trait` draws from a balanced two-level Gaussian
data-generating process (units × traits, within/between variance
components). Use it to make example data or to check whether fitting recovers
known parameter values.

```@docs
tidy
Base.summary(::GllvmFit, ::AbstractMatrix)
GllvmSummary
deviance
rotate_loadings
extract_rotated_loadings_table
predict_missing
predict_cross_covariance
imputed
profile_cross_rho
profile_cross_rho_ci
extract_coevolution_modules
simulate_unit_trait
```

See also: [Working with a fit](working-with-a-fit.md) ·
[Post-fit extractors](postfit-extractors.md) ·
[Confidence intervals](confidence-intervals.md).
