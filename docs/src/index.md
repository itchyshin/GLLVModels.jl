```@raw html
---
layout: home

hero:
  name: "GLLVModels.jl"
  text: "Which responses vary together?"
  tagline: "A standalone Julia package for finding shared patterns across many species, traits, or outcomes—and separating them from variation unique to each response."
  actions:
    - theme: brand
      text: Fit your first model
      link: /quickstart
    - theme: alt
      text: Choose a scientific question
      link: /#three-starting-routes
    - theme: alt
      text: What can I fit today?
      link: /#landing-page-scope

features:
  - title: "Traits or repeated outcomes"
    details: "Which traits or outcomes vary together across individuals, and which mostly vary on their own?"
  - title: "Species across sites"
    details: "Which species tend to occur together after measured environmental conditions are accounted for?"
  - title: "Related species"
    details: "How much variation in a trait follows shared evolutionary history rather than independent differences?"
---
```

# What is a GLLVM?

**GLLVM** means **generalised linear latent-variable model**. It is a model for
several responses measured on the same sites, individuals, species, or studies.
It uses a small number of unobserved shared patterns—called latent variables—to
describe how responses vary together, while allowing each response to retain
its own variation.

In ecology, the responses may be species measured across sites. In evolution,
they may be several traits measured across species or individuals. In other
fields, they may be repeated outcomes or questionnaire items. A latent pattern
describes association in a fitted model; it does not on its own show causation
or a direct biological interaction.

!!! warning "Experimental package"
    Start with a documented example, check that the model fit is trustworthy,
    and read [tested models and current limits](gllvmtmb-parity.md) before
    reporting a result. A successful fit alone is not validation.

# Choose your scientific question

GLLVModels.jl has three reader routes. They use different data and answer
different questions, so choose the question before choosing a function.

## Three starting routes

### 1. General latent-variable models

**Question:** Which continuous responses vary together across observations,
and how much variation is shared rather than response-specific?

**Start:** [Fit your first model](quickstart.md) gives a complete, simulated
Gaussian response matrix and shows how to interpret shared variation.

**Next action:** interpret model-implied correlations and the fraction of each
response explained by shared patterns. Then use
[Working with a fit](working-with-a-fit.md) for the post-fit task you need.

### 2. Phylogenetic comparative models

**Question:** For one continuous trait, how much variation follows the
evolutionary relationships in a supplied tree, and how much remains
independent?

**Start:** [First phylogenetic Gaussian model](vignettes/phylogenetic-gllvm.md)
starts with a small tree and a trait vector in its exact tip order.

**Next action:** verify the biological tip-to-trait match before fitting, then
inspect the two variance components within that vignette's point-estimate
scope.

### 3. Community and species-distribution models

**Question:** Can a small number of unmeasured site differences summarize which
species tend to occur or be abundant together after measured conditions are
accounted for?

**Start:** [First community abundance model](vignettes/community-abundance.md)
defines and fits a small count table with species as rows and sites as columns.

**Next action:** inspect the fitted ordination and model-implied residual
associations as exploratory descriptions; use [Working with a fit](working-with-a-fit.md)
when you need a specific post-fit quantity.

## A first model for shared variation

The compact example below belongs to the first route. It is for continuous
responses, such as several body traits measured on the same individuals. Each
row is a trait and each column is an individual or site. If your data are in
the common sites-by-species layout, swap the rows and columns before fitting.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/itchyshin/GLLVModels.jl")
```

Use Julia 1.10 or later. This package is installed directly from its source
repository rather than the General registry.

## Fit your first model

Ask: **Which continuous traits vary together across individuals, and which
traits mostly vary on their own?**

For this first model, centre each response first and fit the stable
shared-residual Gaussian route. The shared-residual assumption is a useful
starting point, not a claim that every trait has identical variability.

```julia
using GLLVModels, Random, Statistics

Random.seed!(1)
n, p, K = 80, 5, 2                         # sites, responses, latent axes
Λ = 0.7 .* randn(p, K)
Y = Λ * randn(K, n) .+ 0.5 .* randn(p, n)  # traits × individuals
Y .-= mean(Y; dims = 2)                    # centre each trait

fit = fit_gaussian_gllvm(Y; K = K)

fit.converged
R = correlation(fit)          # model-implied trait correlations
shared = communality(fit)     # shared fraction for each trait, from 0 to 1
```

![Model-implied cross-response correlations from a simulated two-factor GLLVM fit](assets/correlation_heatmap.png)

The heatmap is a simulated two-pattern Gaussian fit. A positive value in `R`
means that two traits tend to vary together in this fitted model. A value near
one in `shared` means that much of a trait's modelled variation belongs to the
shared patterns. Neither result proves a causal relationship.

This first route assumes that every response has the same remaining variability
after the shared patterns are accounted for. It is the documented route with
stable result extractors. The [model guide](model.md) explains more flexible
Gaussian models after you have completed this first fit.

## What The Fit Gives You

For this shared-residual Gaussian fit, the usual report-ready quantities are:

- `sigma_y_site(fit)` for the among-response covariance `Σ_y`;
- `communality(fit)` for the shared-variance fraction per response;
- `correlation(fit)` for model-implied cross-response correlations;
- `getLV(fit, Y)` and `getLoadings(fit)` for ordination scores and loadings.

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
`engine = "julia"` bridge. The packages overlap only for the workflows listed
in [Capability parity](gllvmtmb-parity.md); do not assume a model transfers
unchanged. Interval coverage has not been established for every workflow.
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
