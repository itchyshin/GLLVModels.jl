# Community abundance: a first GLLVM

```@raw html
<div class="gllvm-route gllvm-route--applied">
  <div>
    <span class="gllvm-route__eyebrow">Applied community route</span>
    <p>Fit a count matrix, inspect its latent ordination, and describe associations among species.</p>
  </div>
</div>
```

A generalized linear latent variable model (GLLVM) can summarize variation
among species with a few unobserved site gradients. This example fits a
two-axis Poisson model and extracts site scores, species loadings, and
model-implied correlations. These are descriptive point estimates: residual
associations alone cannot identify competition, facilitation, or other species
interactions.

!!! warning "Matrix orientation"
    Use **species in rows and sites in columns**: a `p × n` response matrix.
    If your table has sites in rows, transpose it before fitting. Retain the
    exact response matrix: the post-fit calls below also need it.

## The model

For species ``i`` at site ``j``, the model is

```math
y_{ij}\mid z_j \sim \operatorname{Poisson}(\mu_{ij}),\qquad
\log\mu_{ij}=\alpha_i+\lambda_i^\top z_j,\qquad
z_j\sim\mathcal{N}(0,I_K).
```

Each species has an intercept ``\alpha_i`` and loadings ``\lambda_i`` on
the shared site gradients. Counts are conditionally independent given those
gradients. The Poisson mean equals its variance **conditional on the latent
scores**; integrating over the scores can produce additional variation and
association among counts. This example includes no measured environmental
covariates, effort offsets, or spatial effects.

## Create a small count matrix

```julia
using GLLVModels, Random, Distributions

rng = MersenneTwister(42)
p = 6   # species
n = 30  # sites
Y = rand(rng, Poisson(2.0), p, n)
```

These independent simulated counts make a small example for learning the
interface; they do not contain a known community gradient to recover. For
your own survey, first check count coding, missing values, species and site
labels, and whether sampling effort is comparable. A Poisson model with
latent variables does not automatically account for every source of
overdispersion or imperfect detection.

## Fit and check the result

```julia
fit = fit_gllvm(Y; family = Poisson(), K = 2)
(converged = fit.converged, loglikelihood = fit.loglik)
```

Check `fit.converged` before interpreting the result. If it is `false`,
revisit the data and fitting settings before proceeding. Numerical convergence
does not establish that the Poisson family or two latent axes adequately
describe your survey. In a real analysis, assess model fit and sensitivity to
the number of axes; see [Diagnostics and model comparison](../diagnostics.md).

## Extract the ordination

```julia
site_scores = getLV(fit, Y)          # n × 2 conditional site scores
species_loadings = getLoadings(fit)  # p × 2 species loadings

size(site_scores), size(species_loadings)
```

Both calls use the same default canonical rotation, so the scores and
loadings can be interpreted together. Close sites have similar fitted latent
scores, while loading directions describe species' responses to those axes.
Axis signs and directions are not biological labels. Compare the ordination
with measured habitat variables before giving an axis an ecological name.

For an alternative presentation, `extract_ordination` rotates both matrices
to the principal axes of the site scores. Use its sites and species together;
this rotation can differ from the loading-based rotation above:

```julia
ord = extract_ordination(fit, Y)
ord.sites
ord.species
```

## Inspect model-implied association

```julia
R = correlation(fit, Y)  # p × p correlation matrix on the link scale
size(R), R[1, 2]
```

This function standardizes the covariance
``\Lambda\Lambda^\top+\operatorname{diag}(v_i)``. For the Poisson log-link
model, ``v_i=\log(1+1/\bar\mu_i)`` is a distribution-specific residual
variance approximation, using the mean fitted count for species ``i`` across
sites. Thus `R` includes the Poisson residual contribution; it is neither the
correlation of the loading matrix alone nor the correlation of raw counts.

A positive or negative entry describes association under the fitted model.
Unmeasured habitat, sampling bias, spatial dependence, and model
misspecification can all affect it. This example provides no confidence
interval or hypothesis test for an entry of `R`.

## Next steps

For measured environmental effects or a different count distribution, consult
[Response families](../response-families.md) and
[Choose R, Julia, or the bridge](../choose-r-julia-bridge.md) for the relevant
interfaces and limits. For a single continuous trait measured across a tree,
continue with the [phylogenetic Gaussian example](phylogenetic-gllvm.md).
