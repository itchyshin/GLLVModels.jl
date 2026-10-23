# Joint named grouping models

This development route fits shared random effects jointly for Gaussian,
Poisson-log, Binomial-logit, Beta-logit and NB2-log responses. Agreement with
R and recovery of known simulated parameters have not yet been established.
Models combining grouping and phylogeny, and access through the R bridge,
remain under validation.

Responses have traits in rows and observations in columns. A grouping term
selects a covariance; its labels select observations that share an effect.

```julia
using GLLVModels, StatsModels
terms = [GroupingTerm(:unit; mode=:latent, rank=1),
         GroupingTerm(:cluster2; mode=:indep)]
# Y: traits × observations; data: one row per observation.
fit = gllvm(@formula(y ~ 1 + x), Y, data;
    grouping=terms, unit=:plot, cluster2=:batch)
ci = grouped_gaussian_intervals(Y, fit; unit=data.plot, cluster2=data.batch)
population_mean = predict(fit)
unit_covariance = extract_Sigma(fit; level=:unit, part=:total)
```

Use `fit_gllvm(Y; grouping=terms, unit=plot, cluster2=batch)` for trait-specific
intercepts without a formula. Do not supply a global `K`: each term owns its
rank. `mode=:dep` selects full trait covariance; `mode=:indep` selects diagonal
trait covariance. The latter still shares effects between observations with
the same label. `cluster2` permits only independent trait effects.

`unit_obs` labels must be globally nested within `unit` labels. `cluster` may
cross units. Identifiers alone do not add effects. Current Gaussian grouping
has a single shared residual variance; per-trait residual variance is not
silently substituted.

Intervals use the full observed marginal objective, including nuisance
parameters. Fixed-zero covariance entries are not interval targets. Inspect
the returned overall and per-target statuses: a converged point fit can have
unavailable intervals at a boundary or with insufficient replication. No
automatic profile fallback or coverage certification is implied. Replaying intervals
with changed response data or grouping incidence raises an error.

### Explicit Gaussian variance profiles

For identifiable Gaussian models whose **every** grouping term is
`mode=:indep, common=false`, request one group variance explicitly:

```julia
using GLLVModels, Random
rng = MersenneTwister(20_260_907)
unit = repeat(1:12; inner=5)
b = 0.65 .* randn(rng, 12)
Y = reshape([1.1 + b[g] + 0.35randn(rng) for g in unit], 1, :)
fit = fit_gllvm(Y; grouping=[GroupingTerm(:unit; mode=:indep, common=false)],
    unit=unit, iterations=100)
profile = grouped_gaussian_variance_profile(Y, fit; term=:unit, trait=1)
profile.status                         # check before using endpoints
profile.target                         # natural variance, not SD or log-SD
(profile.lower.endpoint, profile.upper.endpoint)
```

This example is a numerical workflow, not recovery or coverage evidence.
The profile reoptimises the mean, other eligible group variances and the
shared observation residual variance. It evaluates zero exactly and verifies
finite endpoints with fresh refits. `:unavailable` means the interval must not
be used. Read the returned reason and diagnostic details before simplifying the
model or choosing a different uncertainty summary.
An endpoint at zero is labelled `at_boundary=true`, not treated as an interior
coverage result. Aliased components, a nonstationary full fit, unsupported
covariance structures and changed data are rejected. No profile of fixed
effects, latent loadings or phylogenetic signal is supplied by this function.
The ordinary Wald route remains unchanged.

Too many covariance coordinates cause structural non-identification, even with
large samples. For example, two traits with a rank-one factor and two unique
variances have four parameters but only three covariance entries. The fitter
warns about this redundancy; `mode=:dep` removes that redundant decomposition
when the intended target is unrestricted total covariance. Passing the dimension
check alone does not establish identification: aliased groups and boundary
estimates can still prevent valid intervals.
Known coordinate redundancy returns the explicit interval status
`:nonidentifiable`, even if roundoff happens to make a fitted Hessian positive.

`predict(fit)` and `fitted(fit)` return the fixed-effect population mean with
random effects set to zero. `residuals(fit)` returns observations minus that
mean, not conditional or standardized residuals. For new observations, pass a
complete trait-major design matrix to `predict(fit, design)`; automatic formula
reconstruction for new tables is not implemented. Printed summaries distinguish
point convergence from valid marginal curvature. `extract_Sigma` returns a
selected term's trait covariance, not the covariance across all observations.

`vcov(fit)` returns the full fixed-effect covariance block, including
off-diagonal entries, from the inverse observed marginal information with all
nuisance parameters included. `stderror(fit)` uses its diagonal. Both raise an
explicit diagnostic if that calculation is unavailable; they do not substitute
a diagonal-only covariance or regularize failed curvature.
`summary(fit, Y)` returns structured fixed-effect rows with standard errors,
95% Wald intervals and inference status. The supplied response must match the
fit. One-argument `summary(fit)` remains a brief text description. These methods
also apply to the non-Gaussian grouped fits below; no nominal coverage claim is
implied.

For a small, self-contained Gaussian interface check:

```julia
using GLLVModels
Y = reshape([1.0, 1.2, 0.9, 1.1, 3.0, 2.8, 3.1, 2.9], 1, :)
unit = repeat(1:4, inner=2)
fit = fit_gllvm(Y; grouping=[GroupingTerm(:unit; mode=:indep, common=true)],
    unit=unit, iterations=80)
fixed_covariance = vcov(fit)
report = summary(fit, Y)
report.inference_status
report.fixed_effects
```

This deterministic example checks the public interface; it is not simulation
recovery or interval-coverage evidence.

## Non-Gaussian grouping

Use the same terms and labels with a supported family. For example:

```julia
using GLLVModels, Distributions
Y = [1.0 3 2 4 1 2; 2 1 3 2 4 1]
unit = repeat([:a, :b, :c], inner=2)
terms = [GroupingTerm(:unit; mode=:indep, common=true)]
fit = fit_gllvm(Y; family=Poisson(), grouping=terms, unit=unit)
ci = grouped_nongaussian_intervals(Y, fit; unit=unit)
zero_effect_mean = predict(fit)
raw_residuals = residuals(fit)
unit_covariance = extract_Sigma(fit; level=:unit).Sigma
```

This small example illustrates the interface, not an identifiable benchmark.
Inspect `fit.converged`, `fit.inner_status`, `ci.status`, and each interval's
status. All groups share one joint random-effect mode solve; crossed effects
are not duplicated into independent per-observation fits.

`Beta(phi, 1.0)` and `NegativeBinomial(r, 0.5)` are conditional-family markers.
Their positive first parameter supplies the starting dispersion, not a fixed
estimate. The default `dispersion=:trait` estimates one precision/size per
trait; `dispersion=:shared` explicitly selects one shared parameter. The interval
extractor inherits that choice. For Binomial, supply `N` with the same shape as
`Y` to both fitting and interval extraction. No grouped arbitrary dispersion
partition, alternative link, or profile fallback is supplied by this route.

Non-Gaussian `predict(fit)` and `fitted(fit)` return the response mean with random
effects set to zero. This is **not** the mean integrated over random effects:
nonlinear links make those different. `predict(fit; type=:link)` returns the
fixed linear predictor. Raw residuals subtract the zero-effect response mean.
For a new complete design, Binomial prediction requires an explicit new trials
matrix `N`; training trials are not silently reused. Overflow gives an explicit
error. Covariance extraction selects the latent-scale covariance of a fitted
group term; a Gaussian residual covariance is not defined for these families.
