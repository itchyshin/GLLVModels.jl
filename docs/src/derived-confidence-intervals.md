# Derived confidence intervals

The [Confidence intervals](confidence-intervals.md) page covers Wald, profile,
and bootstrap intervals on a model's own working parameters. This page covers
intervals on **derived, nonlinear** quantities computed *from* a fit —
standardized loadings, two-level repeatability/ICC, random-slope SDs, and
profile bounds on total variance and phylogenetic signal. Every interval here
states its transform explicitly (`:fisher_z` for correlation-scaled
quantities, `:logit` for the ICC, `:log` for SD/variance quantities) in both
the returned `NamedTuple` and the docstring, and no interval is reported when
the point estimate sits on or outside its natural boundary — those cases come
back as `method = :failed` with `NaN` bounds rather than a fabricated
interval.

## Standardized-loading and raw-loading intervals

`standardized_loading_wald_ci` is a Fisher-z Wald interval on the
*standardized* loading `rho[t,k] = Λ[t,k] / sqrt(Σ_y_site(fit)[t,t])` — the
correlation-like quantity R's `loading_ci(loading_scale = "standardized")`
reports. `raw_loading_wald_ci` is the identity-scale analogue on `Λ[t,k]`
itself. Below the loadings' lower-triangular pin (`k > t`), both return the
exact structural zero rather than delta-methoding a fixed value.
`loading_ci` is the full per-entry table wrapping both, plus
`loading_profile_exploratory` for the profile-likelihood route on an
exploratory (unpinned) fit.

**Difference from R.** Unlike R, GLLVModels.jl has no separate
confirmatory fit mode with `lambda_constraint` pins — the lower-triangular
packing convention (`src/packing.jl`) is this package's built-in
identifiability device. `loading_ci`/`loading_profile_exploratory` therefore
run on any fit, where R's `loading_ci()`/`loading_profile()` refuse an
unpinned exploratory fit. The deprecated name `loading_profile` forwards here
but is reserved for a future confirmatory mirror of R's surface.

## Two-level repeatability and ICC

`repeatability_wald_ci` and `repeatability_bootstrap_ci` give Wald
(logit-transformed) and bootstrap intervals on `TwoLevelFit`'s per-trait
repeatability `R_t = Σ_B[t,t] / (Σ_B[t,t] + Σ_W[t,t])`; `repeatability_ci`
dispatches between them by `method`. `method = :profile` deliberately throws
[`TwoLevelRepeatabilityProfileWithdrawn`](@ref) rather than silently
substituting a different estimand — R's own profile route for this quantity
was withdrawn because it estimated a diagonal-companion ratio that omits
shared latent variance, and this package does not resurrect it under a
different method name.

## Profile bounds on total variance and phylogenetic signal

`profile_ci_total_variance` profiles the model-implied total variance
`Σ_y_site(fit)[t,t]` directly. `profile_ci_phylo_signal` profiles the
composite phylogenetic-signal ratio `phylo_signal(fit)[t]` (its point estimate
matches [`phylo_signal_wald_ci`](@ref)'s to the bit, since both build on the
same closure). On a fit with no phylogenetic block, `phylo_signal` is
all-`NaN` by contract and this function raises `ArgumentError` rather than
returning a bogus interval.

## Random-slope SD

`slope_sd_ci` is a log-transformed Wald interval on the SD of a random-slope
term, `sqrt(Σ_b[k,k])`, for `GaussianRandomSlopeFit` — the correlated,
multi-slope generalisation of R's single-slope `slope_sd_ci()`.

```@docs
standardized_loading_wald_ci
raw_loading_wald_ci
loading_ci
loading_profile_exploratory
repeatability_wald_ci
repeatability_bootstrap_ci
repeatability_ci
TwoLevelRepeatabilityProfileWithdrawn
profile_ci_total_variance
profile_ci_phylo_signal
slope_sd_ci
```

See also: [Confidence intervals](confidence-intervals.md) ·
[SE and profile machinery](se-profile-machinery.md) ·
[Post-fit extractors](postfit-extractors.md).
