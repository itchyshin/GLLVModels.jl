# Temporal covariance source

GLLVModels.jl fits one temporal covariance source on long data: repeated
measurements of several traits on each series (an individual, a site, a
population) at ordered occasions. It is the Julia counterpart of gllvmTMB's
`temporal_indep()`, `temporal_dep()` and `temporal_latent()` for the temporal
source by itself.

## The model

With one row per series, occasion and trait (and optionally replicate),

```math
y \sim N\bigl(X\beta,\; Z\,(K \otimes \Sigma_T)\,Z^\top + \sigma_\varepsilon^2 I\bigr).
```

``K`` is block diagonal over series. Within a series it is ``\phi^{|t-u|}``
for AR1 (`structure = :ar1`, integer-valued occasions, gaps kept) or
``\exp(-\kappa |t-u|)`` for OU (`structure = :ou`, elapsed numeric time). The
trait block ``\Sigma_T`` depends on the constructor:

| Constructor | ``\Sigma_T`` |
| --- | --- |
| `temporal_indep` | ``\operatorname{diag}(\psi_1, \dots, \psi_p)`` |
| `temporal_dep` | ``L L^\top``, lower-triangular ``p \times p`` factor |
| `temporal_latent` | ``\lambda \lambda^\top`` (rank one) |
| `temporal_latent(...; unique = true)` | ``\lambda \lambda^\top + \operatorname{diag}(\psi)`` |

With `unique = true` the diagonal part follows the same AR1 or OU process
across occasions; it is not independent occasion noise. ``\phi = (1 -
10^{-6}) \tanh\theta`` and ``\kappa = e^\theta``. The likelihood is exact (no
Laplace approximation). At least three traits and three occasions per series
are required, as in gllvmTMB.

## Fitting

```julia
using GLLVModels
term = temporal_indep(:(0 + trait | series), :occasion)          # AR1
fit = fit_temporal_gllvm(data; formula = @formula(value ~ 0 + trait),
                         temporal = term)
extract_temporal(fit)          # phi or OU rate, state table, loadings, variances
```

`data` is any Tables.jl table in long format. Replicated panels name the
replicate column: `temporal_indep(:(0 + trait | series), :occasion;
replicate = :measurement)`. Data that gllvmTMB would refuse (missing
responses, incomplete trait panels, fewer than three traits or occasions,
non-integer AR1 times) raise a [`TemporalContractError`](@ref) carrying
gllvmTMB's message and condition class.

In an unreplicated `temporal_indep` fit, the temporal variances and the
residual variance are separated only by the temporal correlation. With little
persistence the fit can put all variation in one of them.

A variance can sit on its zero boundary: a fit can return `theta_temporal_diag`
near -180 (so `psi` is numerically zero) and still report `converged = true`.
Read `fit.psi` or [`extract_temporal`](@ref), not the raw coordinates. With
`temporal_latent(...; unique = true)` the likelihood can also have more than
one local optimum.

Traits are ordered by `sort(unique(trait))`. gllvmTMB orders them by the factor
levels of the trait column, so an R analysis with custom factor levels shows the
`psi` and loading rows in a different order; the likelihood and every fitted
covariance are the same up to that permutation.

## Helper routes

| Function | What it returns | Scope |
| --- | --- | --- |
| [`forecast_temporal`](@ref) | conditional forecast and predictive SD at future occasions of fitted series | unreplicated `temporal_indep` |
| [`profile_temporal`](@ref) | likelihood profile bounds for ``\phi`` or ``\kappa`` | unreplicated `temporal_indep` |
| [`bootstrap_temporal`](@ref) | parametric bootstrap refits of ``\phi`` or ``\kappa``, failures kept | unreplicated `temporal_indep` |
| [`compare_temporal`](@ref) | AIC table, no likelihood-ratio test | every mode, structure and workflow |

None of these is a calibrated interval: the forecast SD ignores parameter
uncertainty, and the profile and bootstrap describe the fitted-parameter
likelihood only. `predict(fit)` returns the in-sample fitted predictor and
`simulate(fit; condition_on_RE)` draws new responses. `confint`,
`bootstrap_ci` and `ordination_uncertainty` refuse temporal fits, because
their algorithms assume independent latent scores.

## Not available yet

Ordinary `unit` / `unit_obs` terms beside the temporal source, the
cross-source cells (a temporal source paired with a kernel, phylogenetic,
animal or spatial source), the wide `traits()` form, offsets, new-data
prediction, latent-score extraction for the rank-one cell, and the R bridge.
Function signatures are in the [API reference](api.md#Temporal-Covariance-Source).
