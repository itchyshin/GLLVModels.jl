```@raw html
---
layout: home

hero:
  name: "GLLVModels.jl"
  text: "Which responses vary together?"
  tagline: "A Julia package for generalised linear latent variable models: models for several responses that may vary together."
  actions:
    - theme: brand
      text: Fit the first Gaussian model
      link: /quickstart
    - theme: alt
      text: Interpret covariance and correlation
      link: /covariance-correlation
    - theme: alt
      text: Check capability parity
      link: /gllvmtmb-parity

features:
  - title: "Start with a Gaussian response matrix"
    details: "Responses are rows and sites are columns: p × n. This landing-page route fits and interprets Gaussian multivariate data."
  - title: "Read the covariance first"
    details: "Use model-implied Sigma, correlation, and the shared-variance fraction before attaching meaning to a rotated loading axis."
  - title: "A focused Julia companion"
    details: "Start with a response matrix in Julia. For the broader formula-first R workflow, use gllvmTMB."
---
```

# Start with the response matrix

GLLVModels.jl fits **generalised linear latent variable models (GLLVMs)**.
These models ask whether several responses, such as species abundances, traits,
or repeated measurements, vary together after allowing each response to retain
its own variation. This landing page starts with the clearest case: a Gaussian
response matrix, its model-implied covariance, and its correlations.

!!! warning "Matrix Orientation: $p \times n$ in Julia vs $n \times p$ in R"
    **GLLVModels.jl expects species/traits in rows and sites/observations in columns ($p \times n$).**

    If you are importing data formatted for R packages such as `gllvm` or `gllvmTMB` (which use the $n \times p$ convention with sites in rows and species in columns), transpose your matrix (`Y'`) before passing it to the Gaussian fitters used here.

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
model, not an identical R comparison. GLLVModels.jl currently covers a smaller
set of applied workflows. Use gllvmTMB for the richer formula-first R workflow
and its documented limits.

`GaussianPerVarFit` does not yet have the `sigma_y_site()`, `correlation()`,
and `communality()` extractor methods used by the shared-residual Gaussian
fit. The explicit `Σ`, `c²`, and `R` calculation above is therefore the
current per-response route; its fields and output contract may change. It makes
the model comparison explicit without promising a stable
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

## Start Here

- Choose a workflow: [Choose R, Julia, or the bridge](choose-r-julia-bridge.md).
- First Gaussian fit & Cheat Sheet: [Quick start](quickstart.md).
- Applied JSDM Vignette: [Community Abundance](vignettes/community-abundance.md).
- Applied Evolutionary Vignette: [Phylogenetic GLLVM](vignettes/phylogenetic-gllvm.md).
- Model equation and estimands: [Model](model.md).
- Ordination, predictions, residuals, AIC, and BIC:
  [Working with a fit](working-with-a-fit.md).
- Response-family choice: [Response families](response-families.md).
- R twin comparison: [Capability parity](gllvmtmb-parity.md).

## Landing-page scope

This landing page makes a Gaussian-only promise: the shared-residual and
per-response-residual Gaussian routes shown above. It does not establish that
non-Gaussian, mixture, spatial, or phylogenetic workflows mentioned elsewhere
in the documentation are ready for an applied analysis. Check
[Capability parity](gllvmtmb-parity.md) and the relevant guide before relying
on a workflow beyond this page.

## Relation To gllvmTMB

R `gllvmTMB` remains the richer formula-first model surface and applied article
set. GLLVModels.jl is the Julia companion: matrix-first today, with a limited
`engine = "julia"` bridge. A list of implemented functions does not show that
the two packages give interchangeable results. Julia interval studies are
diagnostic checks, not calibrated-inference certificates.
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
