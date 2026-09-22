# Community abundance: a first GLLVM

```@raw html
<div class="gllvm-route gllvm-route--applied">
  <div>
    <span class="gllvm-route__eyebrow">Applied community route</span>
    <p>Start with a small count matrix, inspect a fitted latent ordination, then treat its residual associations as descriptive model output.</p>
  </div>
</div>
```

This page is a first, runnable route for a community count matrix. It fits a
two-axis Poisson generalized linear latent-variable model (GLLVM), retrieves
the fitted ordination, and inspects model-implied residual associations. These
are descriptive point estimates. The route is not a test for competition, a
network-reconstruction workflow, or an analysis of causal environmental
effects: residual associations alone cannot identify competition,
facilitation, or other species interactions.

!!! warning "Matrix orientation"
    `GLLVModels.jl` uses **species in rows and sites in columns**: a `p × n`
    response matrix. If your table has sites in rows, transpose it before
    fitting. The fitted object does not store `Y`, so retain the exact response
    matrix for the post-fit calls below.

## The question this model can answer

The model asks whether a small number of unobserved site gradients summarize
remaining co-variation among species. For species ``i`` at site ``j``, the
model is

```math
y_{ij}\mid z_j \sim \operatorname{Poisson}(\mu_{ij}),\qquad
\log\mu_{ij}=\alpha_i+\lambda_i^\top z_j,\qquad
z_j\sim\mathcal{N}(0,I_K).
```

Here ``z_j`` is a site score and ``\lambda_i`` is a species loading on the
shared site gradients; each species also has an intercept ``\alpha_i``. Counts
are conditionally independent given the gradients. The Poisson mean equals its
variance **conditional on the latent scores**; integrating over the scores can
produce additional variation and association among counts. The route is
useful for exploratory description of community structure. It includes no
measured environmental covariates, effort offsets, or spatial effects, and it
does not by itself adjust for sampling design, detection, spatial dependence,
or other mechanisms that could generate a pattern.

## A small reproducible count matrix

```julia
using GLLVModels, Random, Distributions

rng = MersenneTwister(42)

p = 6   # species
n = 30  # sites

# A p × n integer count matrix. Replace this with your own matrix after
# checking its orientation and recording how zeros and effort were handled.
Y = rand(rng, Poisson(2.0), p, n)
```

These independent simulated counts make a small example for learning the
interface; they do not contain a known community gradient to recover. For a
real analysis, begin by documenting what constitutes a site, whether sampling
effort differs among sites, and why a Poisson mean--variance relation is
plausible. Also check count coding, missing values, and species and site
labels. This minimal example deliberately does not claim to resolve
overdispersion, imperfect detection, or uneven effort.

## Fit a two-axis Poisson GLLVM

```julia
fit = fit_gllvm(Y; family = Poisson(), K = 2)

(converged = fit.converged, loglikelihood = fit.loglik)
```

Check `fit.converged` before interpreting a result. If it is `false`, stop:
do not interpret the ordination or residual associations. First recheck matrix
orientation, the treatment of zeros and effort, and whether the Poisson
mean--variance relationship is plausible for your survey. A converged
optimization is only a numerical diagnostic; it does not establish that two
axes, the Poisson family, or the assumed latent structure are scientifically
adequate. In a real analysis, assess model fit and sensitivity to the number
of axes; see [Diagnostics and model comparison](../diagnostics.md).

## Inspect the ordination

The score and loading calls need the same `Y` passed to the fit:

```julia
site_scores = getLV(fit, Y)          # n × 2 conditional site scores
species_loadings = getLoadings(fit)  # p × 2 species loadings

size(site_scores), size(species_loadings)
```

Both calls use the same default canonical rotation, so the scores and
loadings can be interpreted together. Sites close together in this fitted
latent space have similar model-based patterns after the model's species
intercepts, while loading directions describe species' responses to those
axes. Loading direction and sign are not biological labels: the package uses a
canonical rotation for presentation, but the axes are still an exploratory
low-dimensional summary. Relate an axis to habitat only after comparing it
with independently measured site variables and assessing sensitivity to the
number of axes and the observation model.

For an alternative presentation, `extract_ordination` rotates both matrices
to the principal axes of the site scores. Use its sites and species together;
this rotation can differ from the loading-based rotation above:

```julia
ord = extract_ordination(fit, Y)
ord.sites
ord.species
```

## Residual associations are not interaction evidence

```julia
R = correlation(fit, Y)  # p × p model-implied, link-scale correlation matrix
size(R), R[1, 2]
```

This function standardizes the covariance
``\Lambda\Lambda^\top+\operatorname{diag}(v_i)``. For the Poisson log-link
model, ``v_i=\log(1+1/\bar\mu_i)`` is a distribution-specific residual
variance approximation, using the mean fitted count for species ``i`` across
sites. Thus `R` includes the Poisson residual contribution; it is neither the
correlation of the loading matrix alone nor the correlation of raw counts.

These are **associations** implied by the fitted shared latent structure. A
positive or negative entry is not evidence for competition, facilitation,
exclusion, or a species-interaction network. Unmeasured habitat, shared
sampling bias, spatial structure, model misspecification, and finite-sample
variation can produce the same pattern. The call above returns a point estimate;
this vignette supplies no uncertainty interval or hypothesis test for an entry
of `R`.

## Next step

Use this route to learn the matrix convention, fit a basic count GLLVM, and
inspect its ordination cautiously. If your main question is about measured
environmental effects, a different count distribution, unequal effort,
detection, spatial dependence, or uncertainty for a residual association, stop
here: formulate that design and estimand before extending the model, then use
[Response families](../response-families.md) and [Choose R, Julia, or the
bridge](../choose-r-julia-bridge.md) to select the appropriate public route
and its limits. For the richer formula-first R workflow and its current
evidence boundary, see [gllvmTMB](https://itchyshin.github.io/gllvmTMB/) and
its [current limitations](https://itchyshin.github.io/gllvmTMB/articles/current-limits.html).

For a separate, currently supported phylogenetic Gaussian starting point (one
continuous trait measured across a tree), see the
[Phylogenetic GLLVM vignette](phylogenetic-gllvm.md).
