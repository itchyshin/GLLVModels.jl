# Quick start

```@raw html
<div class="gllvm-route gllvm-route--start">
  <div>
    <span class="gllvm-route__eyebrow">Start here</span>
    <p>Make the matrix orientation explicit, fit a first Gaussian model, and read the covariance before attaching meaning to individual latent axes.</p>
  </div>
</div>
```

This page walks through one end-to-end fit: simulate a Gaussian GLLVM, fit the
documented shared-residual route, and interpret model-implied covariance,
correlation, and shared variation. It concludes with optional R-to-Julia
translation material.

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
σ_true    = 0.5

# True loading matrix (traits × shared patterns)
Λ_true = randn(n_traits, K)

# Latent scores per observation
η = randn(n_sites, K)

# Response matrix y (traits × observations)
y = Λ_true * η' .+ σ_true .* randn(n_traits, n_sites)
y .-= mean(y; dims = 2)        # centre each trait for this first model
```

## 2. Fit the model

```julia
fit = fit_gaussian_gllvm(y; K = K)
```

`fit_gaussian_gllvm` fits the documented Gaussian route. It assumes that
responses share one residual standard deviation; that assumption keeps this
first example simple and gives stable result extractors.

Calculate the quantities you can interpret directly:

```julia
Σ_hat = sigma_y_site(fit)
shared = communality(fit)
R_hat = correlation(fit)
```

`Σ_hat` is the model-implied covariance; `shared` is the fraction of each
trait's variation explained by the shared patterns; and `R_hat` is the
model-implied correlation matrix. The shared patterns describe association,
not causation.

## 3. Check the fit and read the results

```julia
fit.converged
Σ_hat
shared
R_hat
```

Check `fit.converged` before interpreting results. A positive entry of `R_hat`
means two traits tend to vary together under this model. A high value of
`shared` means the trait's modelled variation is mostly shared with other
traits. Do not name an individual latent axis before considering rotation and
the biological design.

## 4. What this first model does—and does not—assume

```julia
This route assumes a common residual standard deviation. If different traits
need different residual variability, use the later model guides and check their
documented limits. The [confidence-interval guide](confidence-intervals.md)
explains available uncertainty methods for this fitted model; no interval
method is a blanket guarantee for every data set.

## 5. Compare the model with the teaching data

```julia
Σ_true = Λ_true * Λ_true' + σ_true^2 * I

maximum(abs, Σ_hat .- Σ_true)        # largest per-cell discrepancy — should be small
```

This teaching simulation is not a recovery certificate. It shows the kind of
comparison you can make when the true structure is known, not a general
guarantee for applied data.

To visualise it, with Plots.jl installed separately (`Pkg.add("Plots")` — it is
not a GLLVModels.jl dependency):

```julia
using Plots

heatmap(
    [Σ_true Σ_hat (Σ_hat .- Σ_true)],
    title  = "Σ_y true | Σ_y est | residual",
    aspect_ratio = :equal,
)
```

The residual panel should sit close to zero across the full
species-by-species surface.

---

## Cheat Sheet: R `gllvm` / `gllvmTMB` ⟷ Julia `GLLVModels.jl`

| Task / Feature | R (`gllvm` / `gllvmTMB`) | Julia (`GLLVModels.jl`) | Notes |
|:---|:---|:---|:---|
| **Matrix shape** | `Y` is n × p (sites × species) | `Y` is p × n (species × sites) | **Transpose R matrices (`Y'`) when loading into Julia** |
| **Gaussian GLLVM** | `gllvm(Y, family = "gaussian", num.lv = 2)` | `fit_gaussian_gllvm(Y; K = 2)` or `fit_gllvm(Y; family = Normal(), K = 2)` | ~340× faster closed-form profile path (single-σ² Gaussian only; see [Benchmarks](benchmarks.md)) |
| **Poisson count JSDM** | `gllvm(Y, family = "poisson", num.lv = 2)` | `fit_gllvm(Y; family = Poisson(), K = 2)` or `fit_poisson_gllvm(Y; K = 2)` | Laplace approximation with exact gradients |
| **Negative Binomial (NB2)** | `gllvm(Y, family = "negative.binomial", num.lv = 2)` | `fit_gllvm(Y; family = NegativeBinomial(), K = 2)` or `fit_nb_gllvm(Y; K = 2)` | Quadratic mean–variance relationship |
| **Negative Binomial 1 (NB1)** | `gllvm(Y, family = "NB1", num.lv = 2)` | `fit_nb1_gllvm(Y; K = 2)` | Linear mean–variance relationship |
| **Binomial / Bernoulli** | `gllvm(Y, family = "binomial", num.lv = 2)` | `fit_gllvm(Y; family = Binomial(), K = 2)` or `fit_binomial_gllvm(Y; K = 2)` | Logit or probit link |
| **Beta (continuous (0,1))** | `gllvm(Y, family = "beta", num.lv = 2)` | `fit_gllvm(Y; family = Beta(), K = 2)` or `fit_beta_gllvm(Y; K = 2)` | Precision parameter |
| **Gamma (positive continuous)** | `gllvm(Y, family = "gamma", num.lv = 2)` | `fit_gllvm(Y; family = Gamma(), K = 2)` or `fit_gamma_gllvm(Y; K = 2)` | Log link with a shape parameter |
| **Ordinal (cumulative-logit)** | `gllvm(Y, family = "ordinal", num.lv = 2)` | `fit_ordinal_gllvm(Y; K = 2)` | Shared or per-trait cutpoints |
| **Zero-inflated models** | `gllvm(Y, family = "ZIP", num.lv = 2)` | `fit_zip_gllvm(Y; K = 2)`, `fit_zinb_gllvm(Y; K = 2)` | Two-part mixture models |
| **Site latent scores** | `getLV(fit)` | Fit-specific; some Julia routes require the original `Y` | Check the worked example for the fitted model |
| **Species factor loadings** | `getLoadings(fit)` or `fit[["params"]][["theta"]]` | Fit-specific; see the fitted model's post-fit guide | The returned object and required data vary by fit type |
| **Residual correlation matrix** | `getResidualCor(fit)` | Fit-specific; some Julia routes require the original `Y` | Model-implied association, not evidence of a biological interaction |
| **Total species covariance** | `getResidualCov(fit)` | Fit-specific; see the fitted model's post-fit guide | Do not assume one covariance extractor applies to every fit |
| **Variance partitioning** | `getResidualCov(fit)[["var.part"]]` | Fit-specific; see the fitted model's post-fit guide | State the denominator before interpreting a shared-variance fraction |
| **Environmental covariates** | `gllvm(Y, X = X, formula = ~ x1 + x2, num.lv = 2)` | `fit_gllvm(Y; family = ..., X = X, K = 2)` or `@formula(Y ~ x1 + x2)` | Fixed effects for environmental predictors |
| **Species-specific slopes** | `gllvm(Y, X = X, formula = ~ (x1 \| species), ...)` | Use the fit-specific species-covariate guide | Its Julia interface uses named inputs; do not copy an R positional call |
| **Fourth-corner models** | `gllvm(Y, X = X, TR = TR, formula = Y ~ ...)` | Use the fit-specific fourth-corner guide | Its Julia interface uses named inputs; do not copy an R positional call |
| **Phylogenetic comparative model** | `gllvm(Y, tree = phy, ...)` | See the [phylogenetic vignette](vignettes/phylogenetic-gllvm.md) for the currently supported univariate route | A separate model class with its own data and uncertainty requirements |
| **Phylogenetic signal (H²)** | (derived from variance components) | See the fit-specific phylogenetic post-fit documentation | Derived summaries and their uncertainty are fit- and trait-specific |
| **Confidence intervals** | `confint(fit)` | Model-dependent; see [Confidence intervals](confidence-intervals.md) | Wald, profile, and bootstrap routes are not available for every Julia fit; `GaussianPerVarFit` currently has no public CI route |

For complete worked workflows, explore the [Community Abundance Vignette](vignettes/community-abundance.md) and the [Phylogenetic GLLVM Vignette](vignettes/phylogenetic-gllvm.md). If you have a site-by-species count matrix, take the Community Abundance route now rather than continuing with the Gaussian material below.

For ordinal fits, choose `LogitLink()` or `ProbitLink()` explicitly when translating
a model. The frozen gllvmTMB 0.7.0 ordinal route uses probit; Julia defaults to logit.
Other ordinal links raise `ArgumentError` before the native fitter reads responses.

## Poisson quadrature (local development candidate)

Ordinary log-link Poisson models can opt into adaptive Gauss–Hermite
quadrature (AGHQ), which integrates over latent scores using a grid adapted to
each site's conditional mode. Responses are rows and sites are columns:

```@example poisson_aghq
using GLLVModels
Y = [1 2 3 4 1 3 2 2; 3 2 4 5 1 3 2 4]
fit = fit_poisson_gllvm(Y; K=1, aghq=3)
fit.integration.actual                 # :aghq or :laplace
fit.integration.reason                 # stopping or fallback reason
fit.integration.result                 # retained optimization attempts
predict(fit, Y)                         # conditional rates, including offsets
intervals = confint(fit, Y; parm="beta") # fitted frozen-node objective
(actual=fit.integration.actual, nodes=fit.integration.k,
 converged=fit.converged, pd_hessian=intervals.pd_hessian)
```

`aghq=false` remains the default. `aghq=1` follows Laplace; `true` or `:auto`
selects five nodes per axis, declining at 20 response traits. This route is
unpenalized and requires one ordinary loadings-only block, a log link and
1–5 latent dimensions. Predictor-informed latent scores and ineligible direct
requests retain Laplace with a visible reason. Other families and structured
routes are not qualified by this Poisson implementation.

Convergence refers to the final **frozen-node surrogate gradient**; it does
not establish stationarity of an objective that differentiates through moving
nodes. Wald and profile intervals use that same frozen objective. Bootstrap
refits retain failed attempts; recovery and coverage validation remain pending.
Stored masks and offsets are used for the original data. For changed data with
nonzero offsets, supply the offset explicitly. Inspect `fit.converged` and
`fit.integration` before interpreting a result.

## Binomial quadrature (local development candidate)

For successes out of known trials, supply `N` with the same responses × sites
shape as `Y`. Omit `N` only for Bernoulli observations. The ordinary binomial
route accepts logit, probit and complementary-log-log links, with the same
node controls and frozen-node convergence rule as the Poisson route.

```@example binomial_aghq
using GLLVModels
Y = [0 1 2 3 1 2 0 1; 1 2 3 1 0 2 1 3]
N = fill(3, size(Y))
fit = fit_binomial_gllvm(Y; K=1, N=N, aghq=3,
    aghq_control=(n_adapt=30, multistart=false))
probabilities = predict(fit, Y) # probabilities, not expected counts
expected_counts = N .* probabilities
(actual=fit.integration.actual, nodes=fit.integration.k,
 converged=fit.converged, reason=fit.integration.reason)
```

Masks, observed trial counts and offsets are retained. Intervals require the
original observed data and use the final frozen-node objective. For changed
responses or new sites, supply trials and offsets explicitly if the training
model used nonunit trials or nonzero offsets. Finite trial/offset inputs at
masked cells still define predictions and simulation there; invalid masked
trial placeholders cannot define a simulation until valid trials are supplied.

Inspect nonconvergence before interpreting coefficients or intervals. The
original five-node binomial comparison fails convergence in both engines and
has an absolute log-likelihood difference of about 0.00894 (required ≤0.001).
Higher-node diagnostics do not replace that required case. This is a local
implementation candidate, not completed R parity or validated interval coverage.

## Gaussian adaptive quadrature candidate

The default Gaussian fitter integrates exactly and assumes zero mean without
`X`. Opt-in quadrature keeps that model. Supply `X[p,n,q]` to define fixed effects;
there is no implicit extra intercept. The residual SD is shared across traits.

```@example gaussian_aghq
using GLLVModels, Random
rng = MersenneTwister(714)
Y = reshape([0.8, 0.4, -0.3], 3, 1) * randn(rng, 1, 36) + 0.7randn(rng, 3, 36)
f = fit_gaussian_gllvm(Y; K=1, aghq=3)
(actual=f.integration.actual, nodes=f.integration.k,
 converged=f.converged, reason=f.integration.reason)
```

`aghq=1` retains exact Gaussian/Laplace fitting. Auto selection and ineligible
requests follow the rules above and record fallback reasons. For ordinary models,
missing responses, `mask`, `offset`, complete `X` and `β_fixed` are retained for
postfit and refitting. Structured masked/offset Gaussian routes are not yet wired.
Changed data with fixed effects or offsets requires explicit new inputs.

`confint(f, Y)` and `vcov(f, Y)` differentiate the fitted objective, using a
positive-definite observed Hessian. For AGHQ that objective freezes the quadrature
nodes and modes. One node can give the exact Gaussian value while failing to give
the same frozen derivatives; the numerical tests check values, gradients and
Hessians separately. `confint(...; method=:profile)` retains that objective and
`method=:bootstrap` reruns the recorded controls, keeping failed attempts.
The older `bootstrap_ci` alias keeps its working-scale residual-SD convention;
`confint` reports SD estimates and bounds on the natural scale. These functional
checks do not establish calibrated coverage or a speed advantage.
