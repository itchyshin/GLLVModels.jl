# Quick start

```@raw html
<div class="gllvm-route gllvm-route--start">
  <div>
    <span class="gllvm-route__eyebrow">Start here</span>
    <p>Make the matrix orientation explicit, fit a first Gaussian model, and read the covariance before attaching meaning to individual latent axes.</p>
  </div>
</div>
```

This page walks through one end-to-end fit: simulate a Gaussian GLLVM with one
residual variance per response, fit it with `fit_gaussian_pervar_gllvm`, inspect the recovered parameters, build
three flavours of confidence interval, and visualise the recovered
`Σ_y` against the truth. It concludes with an R `gllvmTMB` ⟷ Julia `GLLVModels.jl`
cheat sheet.

!!! warning "Matrix Orientation: $p \times n$ in Julia vs $n \times p$ in R"
    **GLLVModels.jl expects species/traits in rows and sites/observations in columns ($p \times n$).**

    If you are importing data formatted for R packages such as `gllvm` or `gllvmTMB` (which use the $n \times p$ convention with sites in rows and species in columns), you must transpose your matrix (`Y'`) before passing it to `fit_gllvm`, `fit_gaussian_gllvm`, or any other GLLVModels.jl fitter.

## 1. Simulate a small dataset

```julia
using GLLVModels, Random, LinearAlgebra

Random.seed!(20260528)

n_sites   = 80
n_species = 10
K         = 2                  # rank of the latent factor block
ψ_true    = 0.15 .+ 0.10 .* rand(n_species)  # response-specific residual variances

# True low-rank loading matrix Λ_B (n_species × K)
Λ_true = randn(n_species, K)

# Latent factor scores per site (n_sites × K)
η = randn(n_sites, K)

# Response matrix y (n_species × n_sites) — diagonal-residual Gaussian GLLVM
y = Λ_true * η' .+ sqrt.(ψ_true) .* randn(n_species, n_sites)
```

## 2. Fit the model

```julia
fit = fit_gaussian_pervar_gllvm(y; K = K)
```

`fit_gaussian_pervar_gllvm` returns a `GaussianPerVarFit` object. It fits the Gaussian
model with `Sigma_y = Lambda * Lambda' + Psi`, where `Psi` is diagonal with a
separate residual variance for each response. This is the Julia model to use
before comparing with ordinary R `gllvmTMB` `traits(...) + latent(...)`; see
the [R get-started guide](https://itchyshin.github.io/gllvmTMB/articles/gllvmTMB.html).
The shared-residual `fit_gaussian_gllvm` shortcut is a restricted model, not an
identical spelling of that R teaching fit. The packages have partial parity;
the R route remains the richer formula-first documentation path and its
[current limits](https://itchyshin.github.io/gllvmTMB/articles/current-limits.html)
apply to claims about its route.

The shared-residual `sigma_y_site()`, `communality()`, and `correlation()`
extractors do not yet accept `GaussianPerVarFit`. For this experimental
per-response route, construct the rotation-invariant quantities explicitly:

```julia
Σ_hat = fit.Λ * fit.Λ' + Diagonal(fit.ψ²)
c²_hat = diag(fit.Λ * fit.Λ') ./ diag(Σ_hat)
R_hat = Diagonal(1 ./ sqrt.(diag(Σ_hat))) * Σ_hat *
    Diagonal(1 ./ sqrt.(diag(Σ_hat)))
```

This transparent calculation is a current route, not a stable extractor
promise. Raw loading columns remain orientation-dependent; `Σ_hat`, `c²_hat`,
and `R_hat` do not.

## 3. Inspect the recovered parameters

```julia
fit.Λ, fit.ψ²             # shared loadings and response-specific residual variances
fit.loglik                # marginal log-likelihood at the optimum
```

The recovered `Λ_B` can be compared with `Λ_true` only up to an
orthogonal rotation in `K`-space — the latent factors are identified
only up to rotation in the Gaussian model.

## 4. Build confidence intervals

Three CI flavours share a common interface:

```julia
ci_wald      = confint(fit)                                # Wald via observed information
ci_profile   = profile_ci(fit, "sigma_eps")                # likelihood-profile CI
ci_bootstrap = bootstrap_ci(fit; n_boot = 200)             # parametric bootstrap
```

Wald CIs are cheapest and rely on the local quadratic approximation;
profile CIs are exact up to grid resolution; bootstrap CIs make no
distributional assumption on the sampling distribution of the estimator.

## 5. Check `Σ_y` recovery

```julia
Σ_true = Λ_true * Λ_true' + Diagonal(ψ_true)
Σ_hat  = fit.Λ * fit.Λ' + Diagonal(fit.ψ²)

maximum(abs, Σ_hat .- Σ_true)        # largest per-cell discrepancy — should be small
```

This teaching simulation is not a recovery certificate. The published
[Benchmarks](benchmarks.md) grid instead covers a matched **shared-residual**
Gaussian special case; its agreement and speed results do not establish
per-response-residual or non-Gaussian performance.

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
| **Matrix shape** | `Y` is $n \times p$ (sites $\times$ species) | `Y` is $p \times n$ (species $\times$ sites) | **Transpose R matrices (`Y'`) when loading into Julia** |
| **Gaussian GLLVM** | `gllvm(Y, family = "gaussian", num.lv = 2)` | `fit_gaussian_gllvm(Y; K = 2)` or `fit_gllvm(Y; family = Normal(), K = 2)` | ~340× faster closed-form profile path (single-σ² Gaussian only; see [Benchmarks](benchmarks.md)) |
| **Poisson count JSDM** | `gllvm(Y, family = "poisson", num.lv = 2)` | `fit_gllvm(Y; family = Poisson(), K = 2)` or `fit_poisson_gllvm(Y; K = 2)` | Laplace approximation with exact gradients |
| **Negative Binomial (NB2)** | `gllvm(Y, family = "negative.binomial", num.lv = 2)` | `fit_gllvm(Y; family = NegativeBinomial(), K = 2)` or `fit_nb_gllvm(Y; K = 2)` | Quadratic variance $V(\mu) = \mu + \phi \mu^2$ |
| **Negative Binomial 1 (NB1)** | `gllvm(Y, family = "NB1", num.lv = 2)` | `fit_nb1_gllvm(Y; K = 2)` | Linear variance $V(\mu) = (1 + \phi)\mu$ |
| **Binomial / Bernoulli** | `gllvm(Y, family = "binomial", num.lv = 2)` | `fit_gllvm(Y; family = Binomial(), K = 2)` or `fit_binomial_gllvm(Y; K = 2)` | Logit or probit link |
| **Beta (continuous (0,1))** | `gllvm(Y, family = "beta", num.lv = 2)` | `fit_gllvm(Y; family = Beta(), K = 2)` or `fit_beta_gllvm(Y; K = 2)` | Precision parameter $\phi$ |
| **Gamma (positive continuous)** | `gllvm(Y, family = "gamma", num.lv = 2)` | `fit_gllvm(Y; family = Gamma(), K = 2)` or `fit_gamma_gllvm(Y; K = 2)` | Log link with shape parameter $\alpha$ |
| **Ordinal (cumulative-logit)** | `gllvm(Y, family = "ordinal", num.lv = 2)` | `fit_ordinal_gllvm(Y; K = 2)` | Shared or per-trait cutpoints |
| **Zero-inflated models** | `gllvm(Y, family = "ZIP", num.lv = 2)` | `fit_zip_gllvm(Y; K = 2)`, `fit_zinb_gllvm(Y; K = 2)` | Two-part mixture models |
| **Site latent scores** | `getLV(fit)` | `getLV(fit)` | Returns $n \times K$ site coordinates |
| **Species factor loadings** | `getLoadings(fit)` or `fit$params$theta` | `getLoadings(fit)` | Returns $p \times K$ species loadings |
| **Residual correlation matrix** | `getResidualCor(fit)` | `correlation(fit)` | Returns $p \times p$ model-implied correlations |
| **Total species covariance** | `getResidualCov(fit)` | `sigma_y_site(fit)` | Returns $p \times p$ matrix $\Sigma_y = \Lambda\Lambda^\top + \Sigma_\varepsilon$ |
| **Variance partitioning** | `getResidualCov(fit)$var.part` | `communality(fit)` | Shared variance fraction per response |
| **Environmental covariates** | `gllvm(Y, X = X, formula = ~ x1 + x2, num.lv = 2)` | `fit_gllvm(Y; family = ..., X = X, K = 2)` or `@formula(Y ~ x1 + x2)` | Fixed effects for environmental predictors |
| **Species-specific slopes** | `gllvm(Y, X = X, formula = ~ (x1 \| species), ...)` | `fit_gllvm_speciescov(Y, X; K = 2)` | Species-specific environmental responses |
| **Fourth-corner models** | `gllvm(Y, X = X, TR = TR, formula = Y ~ ...)` | `fit_fourthcorner_gllvm(Y, X, TR; K = 2)` | Trait $\times$ environment interactions |
| **Phylogenetic GLLVM** | `gllvm(Y, tree = phy, ...)` | `fit_phylo_gaussian(Y, phy; K = 2)` or `fit_phylo_glm(Y, phy; family = Poisson(), K = 2)` | Hadfield & Nakagawa sparse precision |
| **Phylogenetic signal $H^2$** | (derived from variance components) | `phylo_signal(fit)`, `phylo_signal_wald_ci(fit)` | Transformed-Wald CIs with exact boundary bounds |
| **Confidence intervals** | `confint(fit)` | `confint(fit)`, `profile_ci(fit, "par")`, `bootstrap_ci(fit)` | Wald, profile likelihood, and parametric bootstrap |

For complete worked workflows, explore the [Community Abundance Vignette](vignettes/community-abundance.md) and the [Phylogenetic GLLVM Vignette](vignettes/phylogenetic-gllvm.md).

For ordinal fits, choose `LogitLink()` or `ProbitLink()` explicitly when translating
a model. The frozen gllvmTMB 0.7.0 ordinal route uses probit; Julia defaults to logit.
Other ordinal links raise `ArgumentError` before the native fitter reads responses.

## Experimental Poisson quadrature

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

The convergence diagnostic applies to the approximation actually fitted; it
does not prove that a version which continually retunes its quadrature nodes
would give the same answer. Wald and profile intervals use that same fitted
approximation. Recovery and interval-coverage validation remain pending. Stored
masks and offsets are used for the original data. For changed data with nonzero
offsets, supply the offset explicitly. Inspect `fit.converged` and
`fit.integration` before interpreting a result.

## Experimental binomial quadrature

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
Higher-node diagnostics do not replace that comparison. This is experimental:
it does not establish R parity or validated interval coverage.

## Experimental Gaussian quadrature

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
