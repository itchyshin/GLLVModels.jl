# SE and profile machinery

Lower-level standard-error and profiling building blocks: conditional
latent-score SDs, a bootstrap driver over implied-covariance entries, a
generic profile-curve wrapper, and a thin `standard_errors` convenience
reader.

## `latent_score_sd`: conditional score uncertainty

`latent_score_sd(fit, Y)` returns `sqrt(diag(Cov(z | y, θ̂)))` for the fitted
latent factor scores `z`, conditional on the point estimate `θ̂`. The
calculation holds `θ̂` fixed, so its SDs exclude uncertainty in the fixed
effects, loadings, and dispersion. TMB's default `sdreport()` for random
effects also propagates uncertainty in the estimated fixed parameters. A
conditional TMB comparison requires `ignore.parm.uncertainty = TRUE`, matching
the conditional Laplace curvature described by Kristensen, Nielsen, Berg,
Skaug & Bell (2016, *J Stat Soft* 70(5), §2.3).

The Laplace approximation treats the random-effect block as Gaussian at the
mode. Its precision is the Hessian of the joint negative log-likelihood with
respect to the random effects, evaluated at `(ẑ, θ̂)` (equivalently, the
negative Hessian of the joint log-likelihood).

Two regimes:

- **`GllvmFit` (Gaussian)** uses the exact closed form
  `M = I_K + Λ'Ψ⁻¹Λ`.
- **Binomial, Poisson, Negative Binomial, Gamma, and Beta** use a Laplace
  approximation with the same per-site curvature assembled for the marginal
  likelihood's log-determinant term.

AGHQ-integrated fits use a different conditional covariance object, so this
function raises `ArgumentError` when `aghq > 0`. Ordinal and other unsupported
families raise `MethodError` when you call `latent_score_sd`.

This function was renamed from `getREsd`. R's `getREsd(fit, block=)` reads
auxiliary TMB random-effect blocks (`diag_unit`, `phylo`, `re_int`, ...).
GLLVModels.jl's `latent_score_sd` returns latent factor-score SDs. The old
Julia name still resolves through a deprecated forwarding call.

## `bootstrap_Sigma`: thin driver for one tier

`bootstrap_Sigma` loops the upper triangle of `sigma_y_site(fit)` and calls
the existing `bootstrap_ci_derived` per entry; it adds no new bootstrap machinery.
R's `bootstrap_Sigma()` bootstraps `Sigma`, correlation, communality, ICC, and
cross-correlation across separate unit/unit_obs/phy tiers in one call; this
driver covers only the `Sigma` entries at GLLVModels.jl's single site-level tier.
`level` is validated (only `:unit` accepted) rather than silently
accepted-and-ignored. The driver makes `O(p²)` bootstrap runs. This is
practical for a small number of responses and expensive for many responses.

## `standard_errors`: an eager wrapper

`standard_errors(fit, Y)` is a thin wrapper around `confint(fit; y=Y, ...)`,
returning only `term`/`estimate`/`se`/`pd_hessian`. R's `standard_errors()` is
a *deferred*-computation feature: `gllvmTMB(..., control =
gllvmTMBcontrol(se = FALSE))` skips TMB's `sdreport()` at fit time, and
`standard_errors(fit)` computes it later. `GllvmFit` has no `se = FALSE`
control, and its Hessian can be computed on demand. Here, `standard_errors`
is a convenience reader rather than a lazy cache.

## Profile curves

`tmbprofile_wrapper` re-walks GLLVModels.jl's existing bracket-then-bisect profile
search, but records every evaluated `(θᵢ, nll)` pair instead of discarding
them, giving the full profile trace R's `tmbprofile_wrapper()` (backed by
`TMB::tmbprofile()`) returns. `profile_curve_targets` batches this over
several named/indexed parameters into a `Dict`.

`profile_phylo_signal(fit, t)` profiles the **raw packed parameter**
`sigma_phy[t]`, the per-trait phylogenetic-unique SD scale. The
[`profile_ci_phylo_signal`](@ref) function (see
[Derived confidence intervals](derived-confidence-intervals.md)) profiles the
composite ratio `phylo_signal(fit)[t]`. These are separate estimands profiled
by different engines.

### Renamed: `profile_targets` → `profile_curve_targets`

R's `profile_targets()` is a *readiness registry*. It reports which
parameters could be profiled without running `TMB::tmbprofile()`, which uses a
live checkpoint for faster execution. GLLVModels.jl has no comparable
checkpoint step. Its `profile_curve_targets` function runs each target's
curve directly. The old name, `profile_targets`, remains as a deprecated
forwarding call and is documented here for migration.

```@docs
latent_score_sd
bootstrap_Sigma
standard_errors
tmbprofile_wrapper
profile_curve_targets
profile_phylo_signal
profile_targets
```

See also: [Confidence intervals](confidence-intervals.md) ·
[Derived confidence intervals](derived-confidence-intervals.md) ·
[Diagnostics and model comparison](diagnostics.md).
