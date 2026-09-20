# Quick start

```@raw html
<div class="gllvm-route gllvm-route--start">
  <div>
    <span class="gllvm-route__eyebrow">Start here</span>
    <p>Arrange a response matrix, fit a first Gaussian model, and find out which responses vary together.</p>
  </div>
</div>
```

This page fits one small continuous-data example. It shows how to estimate the
variation shared among traits or species, then how to check the result before
giving it a biological interpretation.

!!! note "Arrange the response matrix"
    GLLVModels.jl expects traits or species in rows and observations in
    columns. If your table has observations in rows, transpose it with `Y'`.

## 1. Simulate a small teaching data set

```julia
using GLLVModels, Random, Statistics

Random.seed!(20260528)

n_sites   = 80
n_traits  = 5
K         = 2                  # two shared patterns in these teaching data
sigma_true = 0.5

# True loading matrix (traits × shared patterns)
Lambda_true = randn(n_traits, K)

# Latent scores per observation
eta = randn(n_sites, K)

# Response matrix y (traits × observations)
y = Lambda_true * eta' .+ sigma_true .* randn(n_traits, n_sites)
y .-= mean(y; dims = 2)        # centre each trait for this first model
```

## 2. Fit the model

```julia
fit = fit_gaussian_gllvm(y; K = K)
```

This first model is for continuous responses and assumes that every response
has the same amount of residual variation. That keeps the example simple and
makes the output easy to inspect.

Calculate the quantities you can interpret directly:

```julia
Sigma_hat = sigma_y_site(fit)
shared = communality(fit)
R_hat = correlation(fit)
```

`Sigma_hat` describes how the responses vary together in the fitted model.
`shared` is the proportion of a response's modelled variation associated with
the shared patterns. `R_hat` is the corresponding correlation matrix. Shared
patterns describe association, not causation.

## 3. Check the fit and read the results

```julia
fit.converged
Sigma_hat
shared
R_hat
```

Check `fit.converged` before interpreting results. A positive entry of `R_hat`
means two responses tend to vary together under this model. A high value of
`shared` means that most of a response's modelled variation is shared with the
other responses. Do not attach a biological name to one individual latent axis
before considering rotation and the study design.

## Where next?

- Have a site-by-species count matrix? Try the [community abundance vignette](vignettes/community-abundance.md).
- Have one continuous trait and an evolutionary tree? Try the [phylogenetic vignette](vignettes/phylogenetic-gllvm.md).
- Moving from R, or unsure which package fits your study? Read [Choose R, Julia, or the bridge](choose-r-julia-bridge.md).
- Need a different response type or uncertainty method? Start with [What can I fit today?](what-can-i-fit-today.md).
