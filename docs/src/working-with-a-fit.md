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

For a Gaussian fit with missing cells, `fit_gaussian_gllvm` derives a mask from
`missing` in `Y`. **By design**, [`fitted`](@ref) and [`predict`](@ref) still return
finite conditional means at masked cells (the same in-sample predictor as at observed
cells), while [`residuals`](@ref) with `type = :pearson` or Dunn–Smyth returns `NaN`
there because no residual is defined without an observation. Pass the same `mask` when
you need masked post-fit behaviour on integration routes.

## Model comparison

```julia
aic(fit)            # 2k − 2·logLik
bic(fit, n)         # k·log(n_sites) − 2·logLik   (n_sites passed explicitly)
```

`k` is the free-parameter count (loadings counted modulo the `K(K−1)/2` rotational
identifiability). Lower is better — compare fits with different `K` (or different
families) to choose a model.

### Gaussian: zero-mean direct fit vs `fit_gllvm(Normal())`

[`fit_gaussian_gllvm`](@ref) with no covariate design (`X = nothing`) is a **zero-mean**
factor model: per-trait intercepts are not estimated, and `dof` / `aic` / `bic` omit
those `p` parameters. That differs from gllvmTMB's default `value ~ 0 + trait + latent(...)`
and from [`fit_gllvm`](@ref) with `family = Normal()` (which estimates one intercept per
trait and counts them in `k`). At matched log-likelihood, AIC on the zero-mean route is
exactly `2p` below the trait-intercept route. Use `fit_gllvm(Y; family = Normal(), K)`,
a trait one-hot `X`, or `@formula(y ~ 1)` when you want intercepts and twin-comparable
model comparison. The direct fitter **does not centre** `Y`; if trait means are large
relative to within-trait SD, you may see a once-per-session warning — prefer the
`Normal()` route for strongly offset data.

For non-Gaussian fits with `K ≥ 2`, make sure each candidate is the best of several
starts first; a fit stuck at a local optimum (still `converged = true`) distorts the
comparison. Displaying the fit shows family, dimensions, the log-likelihood, AIC, and
convergence:

```julia
fit        # rich summary in the REPL
```

## Binary fits

Everything above works for a binary fit — pass the integer response matrix:

The family markers come from Distributions.jl, not GLLVModels.jl, and `Yb`
here is the 0/1 matrix obtained by thresholding the `Y` simulated above:

```julia
using Distributions                # Binomial() comes from Distributions
Yb = Int.(Y .> 0)                  # p×n 0/1 response
fitb = fit_gllvm(Yb; family = Binomial(), K = 2)
getLV(fitb, Yb)                 # Laplace-mode scores
predict(fitb, Yb)               # fitted probabilities
residuals(fitb, Yb)             # Dunn–Smyth (set rng to reproduce)
```

See also: [Get started](quickstart.md) · [Response families](response-families.md) · [Reference](api.md).
