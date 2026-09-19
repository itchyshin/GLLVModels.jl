```@raw html
---
layout: home

hero:
  name: "GLLVModels.jl"
  text: "Which responses vary together?"
  tagline: "A matrix-first Julia companion for separating shared multivariate structure from response-specific variation."
  actions:
    - theme: brand
      text: General latent-variable route
      link: /quickstart
    - theme: alt
      text: Phylogenetic comparative route
      link: /vignettes/phylogenetic-gllvm
    - theme: alt
      text: Community and species route
      link: /vignettes/community-abundance

features:
  - title: "General latent-variable models"
    details: "Ask which responses vary together across observations. Start with the Gaussian first fit, then inspect model-implied covariance."
  - title: "Phylogenetic comparative models"
    details: "Ask how variation in one continuous trait is partitioned along a supplied tree. Start with the tree-and-trait vignette."
  - title: "Community and species-distribution models"
    details: "Ask whether latent site gradients summarize a community count matrix. Start with the Poisson community vignette."
---
```

# Choose your biology question

GLLVModels.jl has three reader routes. They use different data and answer
different questions, so choose the question before choosing a function. The
navigation keeps implementation and development reference pages separate from
these ordinary analysis routes.

## Three routes for a biology PhD

### 1. General latent-variable models

**Question:** Which continuous responses vary together across observations,
and how much variation is shared rather than response-specific?

**Start:** [General latent-variable first fit](quickstart.md) gives a complete,
simulated Gaussian response matrix in the required `p × n` orientation.

**Next action:** calculate and interpret model-implied covariance, correlation,
and shared-variance fractions; then use [Working with a fit](working-with-a-fit.md)
for the post-fit task you need.

### 2. Phylogenetic comparative models

**Question:** For one continuous trait, how is variation partitioned between a
supplied Brownian-motion phylogenetic covariance and independent residual
variation?

**Start:** [First phylogenetic Gaussian model](vignettes/phylogenetic-gllvm.md)
starts with a small tree and a trait vector in its exact tip order.

**Next action:** verify the biological tip-to-trait match before fitting, then
inspect the two variance components within that vignette's point-estimate
scope.

### 3. Community and species-distribution models

**Question:** Can a small number of unobserved site gradients summarize the
remaining co-variation in a community count matrix?

**Start:** [First community abundance model](vignettes/community-abundance.md)
defines and fits a small `p × n` Poisson count matrix.

**Next action:** inspect the fitted ordination and model-implied residual
associations as exploratory descriptions; use [Working with a fit](working-with-a-fit.md)
when you need a specific post-fit quantity.

## General latent-variable example

The compact example below belongs to the first route. It introduces the
Gaussian response-matrix model and its covariance summaries; it is not the
starting point for the phylogenetic or community routes.

!!! warning "Matrix orientation: p × n in Julia vs n × p in R"
    **GLLVModels.jl expects species/traits in rows and sites/observations in columns (p × n).**

    If you are importing data formatted for R packages such as `gllvm` or `gllvmTMB` (which use the n × p convention with sites in rows and species in columns), transpose your matrix (`Y'`) before passing it to the Gaussian fitters used here.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/itchyshin/GLLVModels.jl")
```

GLLVModels.jl is not yet in the General registry, so `Pkg.add("GLLVModels")` will not
resolve. Use Julia 1.10 or later.

## Fit your first model

Most analyses start with the same scientific question:

> Which responses vary together, and how much variation is shared rather than
> response-specific?

For continuous multivariate data, start with the Gaussian route that gives
each response its own residual variance:

```julia
using GLLVModels, Random, LinearAlgebra

Random.seed!(1)
n, p, K = 80, 5, 2                         # sites, responses, latent axes
Λ = 0.7 .* randn(p, K)
ψ = 0.15 .+ 0.10 .* rand(p)                # one residual variance per response
Y = Λ * randn(K, n) .+ sqrt.(ψ) .* randn(p, n)  # p × n response matrix

fit = fit_gaussian_pervar_gllvm(Y; K = K)

# Rotation-invariant summaries implied by the per-response fit
Σ = fit.Λ * fit.Λ' + Diagonal(fit.ψ²)
c² = diag(fit.Λ * fit.Λ') ./ diag(Σ)       # shared-variance fraction
R = Diagonal(1 ./ sqrt.(diag(Σ))) * Σ * Diagonal(1 ./ sqrt.(diag(Σ)))
```

![Model-implied cross-response correlations from a simulated two-factor GLLVM fit](assets/correlation_heatmap.png)

The heatmap is a simulated two-factor Gaussian fit. Its off-diagonal structure
is what the explicit `R` calculation reports: responses that share a latent axis correlate,
and responses with no shared axis stay near zero.

This is the matrix-first companion to the ordinary R
[`gllvmTMB`](https://itchyshin.github.io/gllvmTMB/) teaching route. Both use
`Sigma = Lambda * Lambda' + Psi`; here `Psi` has one diagonal residual
variance per response. R's wide formula is `traits(...) + latent(...)`, while
Julia's matrix has responses in rows and units in columns. The simpler
`fit_gaussian_gllvm` route has one shared residual SD, so it is a restricted
model, not an identical R comparison. GLLVModels.jl has partial parity and a
smaller applied documentation set; use gllvmTMB for the richer formula-first
workflow and its current evidence boundary.

`GaussianPerVarFit` does not yet have the `sigma_y_site()`, `correlation()`,
and `communality()` extractor methods used by the shared-residual Gaussian
fit. The explicit `Σ`, `c²`, and `R` calculation above is therefore the
current experimental per-response route; its fields and output contract may
change. It makes the model comparison explicit without promising a stable
extractor interface.

## What The Fit Gives You

For the shared-residual Gaussian fit, the usual report-ready quantities are:

- `sigma_y_site(fit)` for the among-response covariance `Σ_y`;
- `communality(fit)` for the shared-variance fraction per response;
- `correlation(fit)` for model-implied cross-response correlations;
- `phylo_signal(fit)` for the phylogenetic share of trait variation;
- `getLV(fit)` and `getLoadings(fit)` for ordination scores and loadings.

For the per-response residual fit used above, use the explicit `Σ`, `c²`, and
`R` construction until those extractors are admitted for `GaussianPerVarFit`.

## Route map and supporting guides

- Choose a runtime first: [Choose R, Julia, or the bridge](choose-r-julia-bridge.md).
- General model-interface reference: [Tutorial](tutorial.md). This is a guided
  interface tour, not a single copy-and-run analysis.
- Model equation and estimands: [Model](model.md).
- Response-family choice: [Response families](response-families.md).
- R twin comparison: [Capability parity](gllvmtmb-parity.md).

## Landing-page scope

This landing page makes a Gaussian-only promise: the shared-residual and
per-response-residual Gaussian routes shown above. It does not establish
support for non-Gaussian, mixture, variational (VA/ELBO), SPDE, or
phylogenetic-GLM workflows. Those are separate routes, and a method being
mentioned elsewhere in the repository is not evidence that it is ready for an
applied analysis. Check [Capability parity](gllvmtmb-parity.md) and the
route-specific documentation before relying on a workflow beyond this page.

## Relation To gllvmTMB

R `gllvmTMB` remains the richer formula-first model surface and applied article
set. GLLVModels.jl is the Julia companion: matrix-first today, with a partial
`engine = "julia"` bridge. The packages overlap only for the documented
workflows; check [Capability parity](gllvmtmb-parity.md) before moving a model
between them. Interval coverage has not been established for every workflow.
See [Comparison vs gllvmTMB](comparison.md) and [Benchmarks](benchmarks.md) for
the validated shared-residual Gaussian closed-form benchmark grid. Those
speed results do not generalise to non-Gaussian fits or establish speed for the
per-response-residual teaching route above.

## Citing

If you use `GLLVModels.jl` in published work, please cite:

> Nakagawa, S. (2026). GLLVModels.jl: Generalised Linear Latent Variable Models in
> Julia. <https://github.com/itchyshin/GLLVModels.jl>

When relevant, also cite the methods it builds on: Hadfield & Nakagawa
(2010, *J. Evol. Biol.*) for the sparse
phylogenetic precision; Tipping & Bishop (1999, *JRSS-B*) for the
probabilistic-PCA initialiser; and Bates et al. (2015, *J. Stat. Soft.*) for
the profile-out and sparse mixed-model machinery. The edge-incidence
phylogenetic representation follows Bolker's `phylog.rmd`.

## Getting Help

- Questions and bugs: open an issue on [GitHub](https://github.com/itchyshin/GLLVModels.jl/issues).
- Function help: in the Julia REPL, type `?` then a name, for example `?fit_gaussian_gllvm`.
- Planned work: see the [Roadmap](roadmap.md).
