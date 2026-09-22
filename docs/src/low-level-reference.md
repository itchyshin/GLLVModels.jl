# Low-level reference

Most users should start with the [Quick Start](quickstart.md) and the public
[API reference](api.md). This page collects numerical helpers and implementation
notes for developers and readers checking the likelihood calculations. Inclusion
here does not imply a complete fitting interface, verified R parity or calibrated
inference for every combination.

## Additional interfaces

```@docs
GLLVModels.init_theta_rr
GLLVModels.profile_nparams
GLLVModels.ordinal_loglik_site
GLLVModels.quadratic_loglik_site
GLLVModels.default_link(::GLLVModels.Distributions.Normal)
Base.summary(::GLLVModels.GllvmFit)
GLLVModels.vcov(::Union{GLLVModels.GroupedGaussianFit,GLLVModels.GroupedNonGaussianFit,GLLVModels.PrecisionMultivariateFit})
GLLVModels.stderror(::Union{GLLVModels.GroupedGaussianFit,GLLVModels.GroupedNonGaussianFit,GLLVModels.PrecisionMultivariateFit})
Base.summary(::Union{GLLVModels.GroupedGaussianFit,GLLVModels.GroupedNonGaussianFit,GLLVModels.PrecisionMultivariateFit}, ::AbstractMatrix)
Base.summary(::GLLVModels.JointPhyloGroupedGaussianFit, ::AbstractMatrix)
GLLVModels.ldiv!(::AbstractVector, ::GLLVModels.LowRankPlusDiagChol, ::AbstractVector)
```

## Internal implementation notes

```@docs
GLLVModels._grouped_indep_variance_basis_gram
GLLVModels._grouped_indep_variance_profile_objective
GLLVModels._grouped_profile_refitter
GLLVModels._profile_invert_callback
GLLVModels._grouped_gaussian_variance_profile
GLLVModels.joint_phylo_grouped_gaussian_loglik
GLLVModels.fit_joint_phylo_grouped_gaussian
GLLVModels._joint_covariance_identification
GLLVModels._validate_precision_fit_input
GLLVModels.joint_phylo_grouped_population_predict
```

Names beginning with an underscore are internal. They can change without the
stability guarantees of the public fitting API. In particular, quadrature helpers
are not the public Stage 1a AGHQ estimator. Source covariance evaluation is not a
complete source-model fitter.

```@docs
GLLVModels._mixed_family_layout
GLLVModels._phylo_beta_xlv_marginal_loglik
GLLVModels._phylo_binomial_xlv_marginal_loglik
GLLVModels._fit_phylo_poisson_xlv
GLLVModels._laplace_saturation_health
GLLVModels._default_hessian
GLLVModels._glm_obs_weight
GLLVModels._aghq_kd_bound
GLLVModels._fit_phylo_binomial_xlv
GLLVModels._phylo_gamma_xlv_marginal_loglik
GLLVModels._fit_phylo_nb_xlv
GLLVModels._phylo_poisson_xlv_marginal_loglik
GLLVModels._eta_realized_lv_effects
GLLVModels._glm_weight_matches_observed
GLLVModels._gaussian_gls
GLLVModels._phylo_nb_xlv_marginal_loglik
GLLVModels._em_map_phylo
GLLVModels._node_depths
GLLVModels._spde_latent_mode
GLLVModels._fit_phylo_ordinal_xlv
GLLVModels._fit_phylo_gamma_xlv
GLLVModels._fit_verdict
GLLVModels._aghq_gh_normal
GLLVModels._tweedie_verdict
GLLVModels._mixed_unpack
GLLVModels._phylo_ordinal_xlv_marginal_loglik
GLLVModels._gauss_hermite
GLLVModels._fit_phylo_beta_xlv
GLLVModels._gaussian_source_loglik
GLLVModels._source_fixed_sigma
```

## Internal AGHQ adaptation and optimization

These helpers expose the frozen-node surrogate used by the opt-in public
Poisson, binomial and Gaussian candidates. They are internal implementation interfaces. Passing their
checks alone does not establish public parity for other response families.

```@docs
GLLVModels.AGHQAdaptation
GLLVModels.aghq_adaptation
GLLVModels.aghq_frozen_logintegral
GLLVModels.aghq_poisson_problem
GLLVModels.aghq_binomial_problem
GLLVModels.aghq_gaussian_problem
GLLVModels.aghq_outer_optimize
GLLVModels.aghq_multistart_optimize
GLLVModels._fit_poisson_gllvm_laplace
GLLVModels._fit_binomial_gllvm_laplace
GLLVModels._fit_gaussian_gllvm_exact
```

## Structured-term grammar recognizer

Internal machinery behind the public
[`fit_gaussian_structured`](structured-term-fitting.md) wrapper: walks raw,
unevaluated `Expr` trees for the `indep`/`dep`/`scalar`/`kernel_*` term
vocabulary (StatsModels' `@formula` macro rejects the `lhs | group` bar
syntax these terms use, so this recognizer is not built on `@formula`).
Not exported; can change without notice.

```@docs
GLLVModels.SourceTermSpec
GLLVModels._recognize_source_term
GLLVModels._source_term_covariance
GLLVModels._check_source_term_exclusions
GLLVModels._read_literal_flag
GLLVModels._assert_no_augmented_lhs
GLLVModels._resolve_kernel
GLLVModels._fit_gaussian_structured_sources
```

## Other internal helpers

```@docs
GLLVModels._psd_sqrt_factor
GLLVModels.LaplaceModeWorkspace
GLLVModels._grouped_nongaussian_objective
```

## S8 analytic outer gradient internals

The grouped non-Gaussian route's analytic outer gradient, derived in
`docs/design/grouped-analytic-gradient.md`. These are internal: they are listed here because
Documenter's `checkdocs` requires every docstring in the module to appear in some `@docs`
block, and because the derivation's alignment table refers to them by name.

```@docs
GLLVModels._grouped_analytic_gradient
GLLVModels._grouped_analytic_loglik_gradient
GLLVModels._grouped_laplace_design_jacobian
GLLVModels._grouped_term_lstar_jacobian
GLLVModels._grouped_selinv_row_quadform
GLLVModels._grouped_selinv_row_crossform
GLLVModels._glm_obs_weight_deta
GLLVModels._grouped_fd_hessian_from_gradient
GLLVModels._grouped_moment_log_sd
```

## Cross-referenced internal helpers without a docstring

The following names have no `"""..."""` docstring of their own — they are
plain internal functions with an ordinary `#` code comment — but are
cross-referenced by name from other docstrings on this page and elsewhere.
Listed here only so those cross-references resolve; consult the cited source
file directly for their implementation.

### `_laplace_mode`

The inner-loop dense-Laplace mode finder for the non-Gaussian families
(`src/families/laplace.jl`, `src/families/binomial.jl`). [`GLLVModels.LaplaceModeWorkspace`](@ref)
holds its reusable buffers.

### `_profile_ci_bounded`

Boundary-aware wrapper around the generic derived-quantity profiler
(`src/confint_derived.jl`) used by [`profile_ci_total_variance`](@ref) and
[`profile_ci_phylo_signal`](@ref) (see
[Derived confidence intervals](derived-confidence-intervals.md)): clamps a
bound that overshoots the quantity's natural feasible range, and reports a
deviance plateau at the range edge as `boundary = true` rather than a bare
`NaN`/`:partial`.

### `_principal_angles`

Textbook principal-angle-between-subspaces computation (Björck & Golub 1973):
orthonormalises each column space via a thin QR, then takes the SVD of the
product of the two orthonormal bases. Used by
[`compare_loadings`](diagnostics.md) and
[`diagnose_kernel_separability`](diagnostics.md)
(`src/diagnostics.jl`) rather than the naive (and geometrically wrong)
`svd(A'B).S` on non-orthonormal bases.

## Destination B development internals

These implementation details support the developing grouping and precision
routes. Their presence here is not public R admission, frozen-reference parity,
recovery qualification, or an assurance of valid intervals at a boundary.
Use the documented unified Gaussian route in [Joint named grouping models](grouped-models.md)
for the currently exposed workflow. Unexported functions remain internal.

```@docs
GLLVModels._precision_multivariate_nll
GLLVModels.joint_grouped_laplace_loglik
GLLVModels._pmv_phylogenetic_signal
GLLVModels.fit_grouped_gaussian
GLLVModels._grouped_gaussian_nll
GLLVModels.fit_precision_multivariate
GLLVModels._grouped_gaussian_factor_nll
GLLVModels._precision_multivariate_unpack
GLLVModels.multivariate_phylo_precision_loglik
GLLVModels._marginal_target_intervals
GLLVModels.fit_grouped_nongaussian
GLLVModels.grouped_nongaussian_zero_effect_predict
GLLVModels.destination_b_population_predict
GLLVModels.grouped_trait_design
GLLVModels._grouped_laplace_design
GLLVModels._grouped_gaussian_objective
GLLVModels.JointGroupedLaplaceResult
GLLVModels._joint_grouped_state
GLLVModels._bridge_fit_precision_multivariate
```
