# [Prior table for supported GLLVM targets](@id prior-table)

GLLVModels.jl accepts prior tables on three constrained routes. The Binomial
route uses the existing Laplace covariate fitter. Its fixed-effect targets can
be named response intercepts or formula coefficients. Normal and Student-t rows use normalized
densities on the link scale, including their normalizing constants. A whole-loading ridge row adapts the existing
`loading_ridge` penalty, with `ridge(lambda)` meaning
`lambda * sum(Λ.^2) / 2` and no normalizing constant. A separate Gaussian
intercept-only route supports a Gamma prior on the shared natural residual
standard deviation `sigma`.
An explicit Gaussian source model also accepts Normal or Student-t priors on
named mean coefficients together with Gamma priors on named source-specific
unique SDs (`Psi`).

```@example prior_table
using GLLVModels
using Distributions
using Random

rng = MersenneTwister(20261122)
response_names = ["bird", "fish", "mammal"]
p, n = length(response_names), 60
probability = [0.35, 0.5, 0.65]
Y = Matrix{Int}(undef, p, n)
for t in 1:p, s in 1:n
    Y[t, s] = rand(rng) < probability[t] ? 1 : 0
end

priors = (
    prior = ["normal(0, 2)", "student_t(0, 1.5, 4)", "ridge(0.25)"],
    class = ["fixef", "fixef", "loading"],
    coef = ["bird::(Intercept)", "mammal::(Intercept)", "all"],
    dpar = ["mu", "mu", "mu"],
    scale = ["link", "link", "loading_coef"],
)

fit = fit_gllvm(Y; family = Distributions.Binomial(), K = 1, priors, response_names)
(fit.loglik, fit.convergence.objective, fit.convergence.log_prior,
 fit.convergence.status, fit.convergence.prior_held_params)
```

For an intercept-only Gaussian model, `class = "residual_sd"` targets the
natural residual SD. The Gamma expression is parameterized by its mean and
shape; its rate is `shape / mean`. The optimizer uses log SD internally, but
the density is on SD and no log-SD Jacobian is added.

```@example prior_table
Y_gaussian = [1.1 0.9 1.3 1.0 0.8 1.2; -0.2 0.1 0.0 0.3 -0.1 0.2]
gaussian_priors = (
    prior = ["normal(0, 2)", "gamma(0.8, 2)"],
    class = ["fixef", "residual_sd"],
    coef = ["all", "(Intercept)"],
    dpar = ["mu", "sigma"],
    scale = ["link", "natural_sd"],
)
gaussian_fit = fit_gllvm(Y_gaussian; family = Distributions.Normal(), K = 0,
    priors = gaussian_priors, response_names = ["trait_a", "trait_b"])
(gaussian_fit.logLik, gaussian_fit.convergence.log_prior,
 gaussian_fit.convergence.status)
```

For an explicit Gaussian source model, `class="fixef"` resolves against the
actual `coefficient_names`. A `class="unique_variance"` row identifies its
response trait and source kernel. Its `gamma(mean, shape)` expression is
converted to shape/rate, and the normalized density is evaluated on the natural
SD without a log-SD Jacobian. This route requires the residual SD to be fixed
with `sigma_eps_fixed`. A single eligible source can be inferred when
`kernel` is empty; multi-source fits require an exact kernel name.

```julia
using GLLVModels

Y = [.8 -.25 .6 -.4; -.1 -.8 .5 -.3; .35 .6 -.2 .1]
Ko = [1. .65 .25 .12; .65 1. .4 .18; .25 .4 1. .55; .12 .18 .55 1.]
Kp = [1. -.1 .35 .05; -.1 1. .08 .3; .35 .08 1. -.12; .05 .3 -.12 1.]
sources = [SourceCovariance(K; groups=1:4, name=name, rank=1, unique=true)
           for (K,name) in ((Ko,:oak),(Kp,:pine))]
prior_table = [
    (prior="normal(0, 1)", class="fixef", coef="t1", dpar="mu", scale="link"),
    (prior="gamma(0.54, 2.7)", class="unique_variance", coef="", dpar="",
     scale="natural_sd", trait="t1", kernel="oak"),
    (prior="gamma(2, 3.6)", class="unique_variance", coef="", dpar="",
     scale="natural_sd", trait="t2", kernel="pine"),
]
fit = fit_gaussian_sources(Y; sources, coefficient_names=[:t1,:t2,:t3],
    trait_names=[:t1,:t2,:t3], sigma_eps_fixed=.2, priors=prior_table)
(fit.logLik, fit.log_prior, fit.objective, fit.convergence_status,
 fit.prior_held_params)
```

The explicit source route admits fixed-effect-only, `unique_variance`-only, and
mixed tables. Psi rows require complete Gaussian data, a fixed positive residual
SD, and named rank-one latent sources with `unique=true, common=false`; a
fixed-effect-only table may use a freely estimated residual SD. Its dense covariance has
quadratic memory and cubic factorization cost in the number of observed cells.
The legacy `psi_sd_priors` keyword remains available, but it cannot be combined
with `priors` in one fit.

`fit.logLik` and `fit.convergence.logLik` are the unpenalized marginal
log-likelihood at the penalized estimate. `fit.convergence.objective` includes
the penalties, and `fit.convergence.log_prior` records their sum on the log
scale. The fit is a
penalized estimate, not a posterior sample. Its confidence-interval helpers
refuse the fit because this curvature has not been calibrated for uncertainty.
Refit without `priors` for ordinary likelihood-based inference and comparisons.

## Current route limits

- `fit_gllvm(Y; family = Binomial(), K = ...)` with the ordinary Laplace fit is
  admitted for complete responses without `X`, `X_lv`, `offset`, `mask`, or
  `weights`. Response intercept targets use `response_name::(Intercept)`; the
  bare response name is also accepted as a selector. The `response_names` vector follows the rows of `Y` and defaults to
  `trait_1`, `trait_2`, and so on. `coef = "all"` explicitly selects all
  response intercepts for that row.
- A Binomial `gllvm(@formula(...))` route also admits named response intercept
  and fixed covariate coefficients. There, `coef = "all"` selects every
  estimated response intercept and covariate coefficient.
- `fit_gllvm(Y; family = Normal(), K = 0)` admits response intercept
  Normal/Student-t rows and one shared residual-SD Gamma row. This is a
  separate, intercept-only complete-data route; it does not support latent
  variables, covariates, missing responses, weights, or model selection.
- One whole-loading `ridge(lambda)` row can accompany fixed-effect rows. It
  cannot be combined with a finite legacy `loading_ridge` keyword.
- AGHQ, predictor-informed latent scores (`X_lv`), grouped or phylogenetic
  routes, model-selection fits, non-Binomial families beyond the stated
  Gaussian routes, covariate coefficients on the constrained Binomial and
  intercept-only Gaussian routes, named individual loadings, other residual-scale
  targets, latent-score coefficients, and correlations are refused or not
  admitted by this first adapter. The explicit Gaussian source route described
  above is separate from the `fit_gllvm` routes.
- A Gamma row for `residual_sd` is a density on natural `sigma`, not log
  variance or a log-SD density. It is available only in the Gaussian
  intercept-only `K = 0` route.

The target resolver rejects unknown and overlapping coefficient selectors,
invalid distribution arguments, and unsupported target classes before fitting.

For the complete fitting signature and keyword reference, see [the `fit_gllvm` API](@ref `GLLVModels.fit_gllvm`).
