# SE and profile machinery

Lower-level standard-error and profiling building blocks: conditional
latent-score SDs, a bootstrap driver over implied-covariance entries, a
generic profile-curve wrapper, and a thin `standard_errors` convenience
reader.

## `latent_score_sd` — conditional-on-θ̂, not a full-uncertainty SE

`latent_score_sd(fit, Y)` returns `sqrt(diag(Cov(z | y, θ̂)))` for the fitted
latent factor scores `z` — **conditional on the point estimate `θ̂`**. It does
not propagate uncertainty in `θ̂` itself (fixed effects, loadings,
dispersion). This is exactly TMB's `sdreport()` convention for random effects
(Kristensen, Nielsen, Berg, Skaug & Bell 2016, *J Stat Soft* 70(5), §2.3): the
Laplace approximation treats the random-effect block as Gaussian at the mode,
with precision equal to the negative Hessian of the joint negative
log-likelihood with respect to the random effects, evaluated at `(ẑ, θ̂)`.

Two regimes:

- **`GllvmFit` (Gaussian)** — exact closed form (`M = I_K + Λ'Ψ⁻¹Λ`).
- **Binomial, Poisson, Negative Binomial, Gamma, Beta** — Laplace-approximate,
  using the same per-site curvature the marginal likelihood's log-det term
  already assembles.

AGHQ-integrated fits (`aghq > 0`) raise `ArgumentError` — their conditional
covariance is not the single-node object this function assumes. Ordinal and
every other family are not covered; calling `latent_score_sd` there is a
`MethodError`, not a silently wrong number.

This function is renamed from `getREsd`: R's
`getREsd(fit, block=)` reads auxiliary TMB random-effect blocks
(`diag_unit`, `phylo`, `re_int`, ...), a different quantity than the latent
factor-score SD this function computes. The old name still resolves via a
deprecated forwarding call.

## `bootstrap_Sigma` — thin driver, single tier

`bootstrap_Sigma` loops the upper triangle of `sigma_y_site(fit)` and calls
the existing `bootstrap_ci_derived` per entry — no new bootstrap machinery.
R's `bootstrap_Sigma()` bootstraps `Sigma`, correlation, communality, ICC, and
cross-correlation across separate unit/unit_obs/phy tiers in one call; this
driver covers only the `Sigma` entries at GLLVModels.jl's single site-level tier.
`level` is validated (only `:unit` accepted) rather than silently
accepted-and-ignored. Cost is `O(p²)` bootstrap runs — practical for a small
number of responses, but expensive when many responses are modelled.

## `standard_errors` — an eager wrapper, not a deferred computation

`standard_errors(fit, Y)` is a thin wrapper around `confint(fit; y=Y, ...)`,
returning only `term`/`estimate`/`se`/`pd_hessian`. R's `standard_errors()` is
fundamentally a *deferred*-computation feature: `gllvmTMB(..., control =
gllvmTMBcontrol(se = FALSE))` skips TMB's `sdreport()` at fit time for speed,
and calling `standard_errors(fit)` computes it later. `GllvmFit` has no such
`se = FALSE` control — the Hessian is always computable on demand — so there
is nothing to defer here; this function is a convenience reader, not a
lazy-caching mechanism.

## Profile curves

`tmbprofile_wrapper` re-walks GLLVModels.jl's existing bracket-then-bisect profile
search, but records every evaluated `(θᵢ, nll)` pair instead of discarding
them, giving the full profile trace R's `tmbprofile_wrapper()` (backed by
`TMB::tmbprofile()`) returns. `profile_curve_targets` batches this over
several named/indexed parameters into a `Dict`.

`profile_phylo_signal(fit, t)` profiles the **raw packed parameter**
`sigma_phy[t]` (the per-trait phylogenetic-unique SD scale) — not the
composite phylogenetic-signal ratio `phylo_signal(fit)[t]` that
[`profile_ci_phylo_signal`](@ref) (see
[Derived confidence intervals](derived-confidence-intervals.md)) profiles.
These are deliberately two different estimands on two different engines, not
a naming accident.

### Renamed: `profile_targets` → `profile_curve_targets`

R's `profile_targets()` is a *readiness registry* — which parameters could be
profiled, without running anything, because `TMB::tmbprofile()` needs a live
checkpoint to run cheaply. GLLVModels.jl has no comparable checkpoint step, so
running is the cheap operation here: `profile_curve_targets` runs every
target's curve directly rather than reporting a readiness flag. The old name,
`profile_targets`, is kept only as a deprecated forwarding call to
`profile_curve_targets`; it is not documented further here.

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
