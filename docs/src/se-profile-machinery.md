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

## `bootstrap_Sigma_r`: the pinned-R bootstrap behaviour

`bootstrap_Sigma_r` follows R's `bootstrap_Sigma()` (gllvmTMB commit
`9539352f6`) more closely than the legacy `bootstrap_Sigma` above. It is a
different estimand and a separate function; the legacy driver is unchanged
and no decision has been made about replacing it. One batch of `n_boot`
datasets is simulated from the fit, each is refit once, and every requested
summary (`Sigma`, `R`, `communality`, `ICC`) is read from that one refit.
Bounds are type-7 percentiles of the finite draws. The result is a
`BootstrapSigmaR`.

Supported fit classes:

- `GllvmFit` with no W tier, no pins, no phylogenetic block and no retained
  Gaussian record: tier `:B` only (`Sigma_B`, `R_B`, `communality_B`).
- `TwoLevelFit` (pass `individual`): tiers `:B` and `:W`, plus `ICC_site`.
- `PrecisionMultivariateFit`: tier `:phy` only (`Sigma_phy`, `R_phy`,
  `communality_phy`); the tier excludes the residual variance, as in R.
- `JointPhyloGroupedGaussianFit`: tier `:phy` only, refused unless the original
  fit converged and its phylogenetic, ordinary and residual covariance
  components are identified (see below).

Everything else (other fit classes, `K_W > 0` on a `GllvmFit`, a `GllvmFit`
with a phylogenetic block) throws an `ArgumentError`, as does `n_cores < 1` or `n_cores`
above `Threads.nthreads()`. `n_cores = k > 1` refits replicates on `k` Julia threads (start Julia with
`julia -t k`); the draws are bit-identical to `n_cores = 1` for the same `seed`. `cross_corr` is always
skipped. A replicate whose refit throws or does not converge is counted in
`n_failed` and left out of the percentiles. Defaults mirror R: `n_boot = 999`,
`conf = 0.95`, `progress = true`; these defaults still need maintainer
agreement.

A runnable walk-through (call, fields, bounds, warnings, refusals) is on the
[worked example page](bootstrap-sigma-r-example.md).

Caveats:

- No output has been compared with R. The behaviour target is R's source,
  not an R run.
- For a `GllvmFit` with `has_diag = true` and `K_W == 0`, tier `B` is the
  identified total `sigma_y_site(fit)`, which includes the residual
  variance. That this equals R's unit tier rests on a Julia comment in
  `extractors.jl`, and is unverified against R.
- For `has_diag = false`, `communality_B` is exactly 1 with a zero-width
  interval.
- The capability row stays unbound: no row, score or tolerance changes.

### The `:phy` tier for `JointPhyloGroupedGaussianFit`

Each replicate redraws the phylogenetic effect and, for every ordinary
grouping term, one independent draw per group through the stored incidence
matrix; loadings, fixed effects and variances are held at the fit. The
phylogenetic part of the response is the only tier reported:
`Sigma_phy = LL' + diag(unique)` contains neither the residual variance nor any
ordinary-term variance. B and W are dropped with an `@info`: R would report B/W
tiers for a comparable multi-tier fit, but how this class's ordinary terms map
to R's B/W has not been verified, so Julia does not bootstrap them.

Deviations from R, all reported on the result (`result.diagnostics`):

- A replicate counts as failed (in `n_failed`, excluded from the percentiles)
  when the refit throws, does not converge, or is not component-identified.
  R's rule is convergence only, so failure rates are not comparable with R's.
- A converged replicate with a non-positive-definite Hessian is kept, as in R.
  The number of such replicates is reported, and a warning is issued when the
  original fit's Hessian is not positive definite. In the pre-run these
  replicates were 0% to 71% of the converged ones (33.7% and 71.4% in the two
  explicit-unique fixtures), and they often have a variance on its boundary, so
  percentile ends can be truncated there.

Evidence is structural plus a Julia descriptive pre-run only: six fixtures, one
seed each, 200 replicates, in which 0% to 2.5% of replicates failed. With 200
replicates a count of 5 does not exclude a true failure rate above 5%. The
fixture refits used `g_tol` between 1e-5 and 2e-4 and at most 200 iterations.
No output has been compared with R; there is no R parity claim. Several
configurations are untested in a real refit: more than one ordinary term, rank
above 1, `:unit_obs` and `:cluster2` terms, covariates beyond trait intercepts,
larger and unbalanced trees.

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
bootstrap_Sigma_r
BootstrapSigmaR
standard_errors
tmbprofile_wrapper
profile_curve_targets
profile_phylo_signal
profile_targets
```

See also: [Confidence intervals](confidence-intervals.md) ·
[Derived confidence intervals](derived-confidence-intervals.md) ·
[Diagnostics and model comparison](diagnostics.md).
