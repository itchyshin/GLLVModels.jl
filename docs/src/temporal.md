# Temporal covariance source

GLLVModels.jl fits one temporal covariance source on long data: repeated
measurements of several traits on each series (an individual, a site, a
population) at ordered occasions. It is the Julia counterpart of gllvmTMB's
`temporal_indep()`, `temporal_dep()` and `temporal_latent()`, for the temporal
source by itself or beside ordinary `unit` / `unit_obs` terms.

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

## Beside ordinary unit and unit_obs terms

Ordinary covariance terms go in the `structure` argument as quoted
expressions, as in gllvmTMB:

```julia
fit = fit_temporal_gllvm(data; formula = @formula(value ~ 0 + trait),
    temporal = temporal_indep(:(0 + trait | series), :occasion),
    unit = :series, unit_obs = :unit_obs,
    structure = [:(latent(0 + trait | series, d = 1)), :(indep(0 + trait | unit_obs))])
```

| Term | Covariance added to ``V`` | Coordinates |
| --- | --- | --- |
| `indep(0 + trait \| g)` | ``J_g \circ \operatorname{diag}(\psi_g)`` | `theta_diag_B` / `theta_diag_W` |
| `dep(0 + trait \| g)` | ``J_g \circ L_g L_g^\top`` | `theta_rr_B` / `theta_rr_W` |
| `latent(0 + trait \| g, d = 1)` | ``J_g \circ (\Lambda_g \Lambda_g^\top + \operatorname{diag}(\psi_g))`` | both (`unique = false` drops the diagonal) |
| `(1 \| g)` | ``\sigma_{re}^2 J_g`` (shared by all traits) | `log_sigma_re_int` |

``J_g`` is one when two rows share a level of `g`. `g` must be the `unit`
column (the stable unit tier, reported as `Sigma_B`) or the `unit_obs` column
(the within-unit tier, `Sigma_W`); `unit` defaults to the temporal series
column. As in gllvmTMB, `unit_obs` must be nested in `unit`, and a stable unit
term needs `series` and `unit` to index the same entities (the labels may
differ). One ordinary term per level is admitted here. The parameter vector
follows gllvmTMB's measured `opt$par` order: `b_fix`, `log_sigma_eps`,
`theta_rr_B`, `theta_temporal_time`, `theta_temporal_rr`,
`theta_temporal_diag`, `theta_diag_B`, `theta_rr_W`, `theta_diag_W`,
`log_sigma_re_int`.

The residual SD follows gllvmTMB's suppression rule. When a unit or unit_obs
diagonal term is at the per-row level (each trait and level has one row) and
the workflow is replicated, ``\sigma_\varepsilon`` is fixed at
``\max(10^{-3}\,\mathrm{sd}(y), 10^{-6})``, the fit prints a note, and
`log_sigma_eps` leaves the parameter vector, so `dof` and AIC count one
coordinate fewer. An unreplicated temporal fit keeps ``\sigma_\varepsilon``
free even then; the per-row variance and the residual are then separated only
by the model structure.

[`extract_ordination`](@ref) with `level = :unit` returns the conditional unit
scores of a `latent` or `dep` unit term, one row per unit (for a rank-one
`temporal_latent` fit without one, the temporal state scores).
`simulate(fit; condition_on_RE = false)` redraws every ordinary tier as well
as the temporal states, and [`update`](@ref) refits with named overrides.

## Helper routes

| Function | What it returns | Scope |
| --- | --- | --- |
| [`forecast_temporal`](@ref) | conditional forecast and predictive SD at future occasions of fitted series | unreplicated `temporal_indep` |
| [`profile_temporal`](@ref) | likelihood profile bounds for ``\phi`` or ``\kappa`` | unreplicated `temporal_indep` |
| [`bootstrap_temporal`](@ref) | parametric bootstrap refits of ``\phi`` or ``\kappa``, failures kept | unreplicated `temporal_indep` |
| [`compare_temporal`](@ref) | AIC table, no likelihood-ratio test | every mode, structure and workflow |

The first four refuse a fit with an ordinary term, naming its tiers as
gllvmTMB does (`diag_B`, `rr_B`, `diag_W`, `rr_W`, `re_int`); gllvmTMB has no
contract for them yet. None of these is a calibrated interval: the forecast SD ignores parameter
uncertainty, and the profile and bootstrap describe the fitted-parameter
likelihood only. `predict(fit)` returns the in-sample fitted predictor and
`simulate(fit; condition_on_RE)` draws new responses. `confint`,
`bootstrap_ci` and `ordination_uncertainty` refuse temporal fits, because
their algorithms assume independent latent scores.

## Not available yet

The `gllvm()` formula route (it needs a pre-pass hook in the formula
grammar), more than one ordinary term per level and `common = true`, the
cross-source cells (a temporal source paired with a kernel, phylogenetic,
animal or spatial source), the wide `traits()` form, offsets, new-data
prediction, and the R bridge.
Function signatures are in the [API reference](api.md#Temporal-Covariance-Source).
