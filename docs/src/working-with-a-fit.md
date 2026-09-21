# Working with a fitted model

Once you have a fit from `fit_gaussian_gllvm` or `fit_gllvm(Y; family=…)`, GLLVModels.jl
gives you the standard post-fit toolkit — ordination, predictions, residual
diagnostics, and model-selection criteria — for Gaussian and non-Gaussian fits.

```julia
using GLLVModels, Random
Random.seed!(1)
n, p, K = 120, 6, 2
Λ = 0.8 .* randn(p, K)
Y = Λ * randn(K, n) .+ 0.5 .* randn(p, n)
fit = fit_gaussian_gllvm(Y; K = K)
```

## Ordination: latent scores and loadings

`getLV` returns the conditional latent-variable scores (the site ordination);
`getLoadings` the species loadings. Both come back in a canonical, reproducible
rotation by default:

```julia
Z = getLV(fit, Y)         # n×K site scores
L = getLoadings(fit)      # p×K species loadings
R = rotation(fit)         # K×K canonical rotation (Z and L share it)
```

Latent factors are identified only up to rotation, so the canonical orientation
(principal-axis SVD, signs fixed) makes the ordination reproducible — and `Λ Λᵀ`,
hence every covariance summary, is unchanged by it. Pass `rotate = false` for the
raw fitted loadings.

Plotting the site scores against the (scaled) loadings gives the model-based
ordination biplot — sites as points, species as labeled vectors:

![Model-based ordination biplot](assets/ordination_biplot.png)

*Simulated two-block data, two-factor Gaussian GLLVM. Species loading on the same
latent factor point the same way; the grey cloud is the site scores `getLV(fit, Y)`.*

For supported one-part non-Gaussian fits, `ordination_uncertainty` gives per-site
score intervals for that point cloud:

```julia
Y_count = rand(0:5, p, n)
count_fit = fit_poisson_gllvm(Y_count; K = K)
u = ordination_uncertainty(count_fit, Y_count;
                            n_boot = 200, rng = MersenneTwister(11))
u.scores     # n×K reference site scores
u.lower      # n×K lower score interval
u.upper      # n×K upper score interval
```

This is a conditional bootstrap of the scores with the fitted parameters held
fixed; it quantifies uncertainty in each site's latent score, not full
refit-level parameter uncertainty. It is currently routed for the single-`Y`
ordination fits with scalar response means: Poisson, NB2, Beta, Gamma,
Exponential, and Binomial (with `N` when trial counts are needed).

## Predictions and fitted values

```julia
η = predict(fit, Y; type = :link)        # linear predictor
μ = predict(fit, Y; type = :response)    # response scale
ŷ = fitted(fit, Y)                       # ≡ predict(…; type = :response)
```

For the Gaussian family the link is the identity, so `:link` and `:response`
coincide. For a binary fit `:response` returns fitted probabilities in `[0, 1]`.
Plain latent fits predict in-sample because the latent score is conditional on
the observed response matrix. Covariate fits also support population-level
new-site prediction from a supplied `X` design, with the latent at its prior
mean, and spatial latent fits use `predict_spatial` for new locations.

## Residual diagnostics

`residuals` gives **Dunn–Smyth** randomized quantile residuals by default — the
GLLVM standard, approximately `N(0, 1)` under a correct model and comparable
across families. For discrete families the randomization uses an `rng`; pass a
seeded one to reproduce:

```julia
r  = residuals(fit, Y)                   # Dunn–Smyth
rp = residuals(fit, Y; type = :pearson)
```

A normal quantile–quantile plot of `r` is the usual goodness-of-fit check.

## Model comparison

```julia
aic(fit)            # 2k − 2·logLik
bic(fit, n)         # k·log(n_sites) − 2·logLik   (n_sites passed explicitly)
```

`k` is the free-parameter count (loadings counted modulo the `K(K−1)/2` rotational
identifiability). Lower is better — compare fits with different `K` (or different
families) to choose a model. Displaying the fit shows family, dimensions, the
log-likelihood, AIC, and convergence:

```julia
fit        # rich summary in the REPL
```

## Binary fits

Everything above works for a binary fit — pass the integer response matrix:

```julia
fitb = fit_gllvm(Yb; family = Binomial(), K = 2)
getLV(fitb, Yb)                 # Laplace-mode scores
predict(fitb, Yb)               # fitted probabilities
residuals(fitb, Yb)             # Dunn–Smyth (set rng to reproduce)
```

See also: [Get started](quickstart.md) · [Response families](response-families.md) · [Reference](api.md).
