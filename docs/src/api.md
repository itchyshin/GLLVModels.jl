# API Reference

This page documents the public API of `GLLVModels.jl`, categorized by functional domain.

---

## Model Fitting

### Unified & General Fitters

The explicit Gaussian `phylo=PrecisionPhy` route supports precision-only
`residual_mode=:trait` (default) or `:shared`. Joint ordinary grouping retains
trait-specific residuals. See the [developer guide](precision-bridge-development.md)
for the Julia-side interface, interval diagnostics, and the fact that this model
is not currently available through the public R bridge.

```@docs
fit_gllvm
gllvm
fit_gllvm_cov
fit_dep_gllvm
fit_phylo_dep_gllvm
fit_animal_dep_gllvm
fit_animal_latent_gllvm
fit_spatial_dep_gllvm
fit_kernel_indep_gllvm
fit_kernel_dep_gllvm
fit_kernel_latent_gllvm
fit_mixed_gllvm
fit_gaussian_gllvm
fit_gaussian_pervar_gllvm
fit_gaussian_sources
GroupingTerm
GroupedGaussianFit
grouped_gaussian_intervals
grouped_gaussian_variance_profile
PrecisionMultivariateFit
precision_multivariate_intervals
JointPhyloGroupedGaussianFit
joint_phylo_grouped_intervals
GroupedNonGaussianFit
grouped_nongaussian_intervals
fit_gaussian_reml
fit_twolevel_gaussian
fit_gaussian_mi_fiml
fit_gaussian_mi_phylo
fit_gllvm_mi
fit_gllvm_mi_multi
```

### Discrete & Count Response Fitters

```@docs
fit_poisson_gllvm
fit_nb_gllvm
fit_nb1_gllvm
fit_gp1_gllvm
fit_compoisson_gllvm
fit_binomial_gllvm
fit_beta_binomial_gllvm
fit_truncated_poisson_gllvm
fit_censored_poisson_gllvm
fit_truncated_nbinom2_gllvm
fit_truncated_nbinom2_gllvm_pertrait
```

### Continuous, Proportion & Ordinal Fitters

Ordinal fitters accept only `LogitLink()` and `ProbitLink()`. Unsupported links
raise `ArgumentError` before response access; the frozen R 0.7.0 ordinal model
uses `ProbitLink()`. Julia's default logit model is a separate model choice.

```@docs
fit_beta_gllvm
fit_gamma_gllvm
fit_exponential_gllvm
fit_studentt_gllvm
fit_lognormal_gllvm
fit_tweedie_gllvm
fit_ordinal_gllvm
fit_ordinal_gllvm_pertrait
fit_ordinal_gllvm_pertrait_cov
fit_ordered_beta_gllvm
fit_multinomial_gllvm
```

### Two-Part & Zero-Inflated Fitters

```@docs
fit_delta_lognormal_gllvm
fit_delta_gamma_gllvm
fit_hurdle_poisson_gllvm
fit_hurdle_nb_gllvm
fit_beta_hurdle_gllvm
fit_zip_gllvm
fit_zip_gllvm_cov
fit_zinb_gllvm
fit_zinb_gllvm_cov
fit_zib_gllvm
fit_zib_gllvm_cov
```

### Grouped Dispersion & Covariate-Extended Fitters

```@docs
fit_nb_gllvm_grouped
fit_nb_gllvm_grouped_cov
fit_nb1_gllvm_grouped
fit_nb1_gllvm_grouped_cov
fit_beta_gllvm_grouped
fit_beta_gllvm_grouped_cov
fit_gamma_gllvm_grouped
fit_gamma_gllvm_grouped_cov
fit_tweedie_gllvm_grouped
fit_beta_binomial_gllvm_grouped
fit_beta_binomial_gllvm_grouped_cov
fit_gllvm_speciescov
fit_fourthcorner_gllvm
fit_roweffect_gllvm
fit_row_random_gllvm
fit_constrained_gllvm
fit_concurrent_gllvm
fit_rrr_gllvm
fit_quadratic_gllvm
fit_gaussian_random_slope
fit_poisson_random_slope
```

### Variational Approximation (VA) Fitters

```@docs
fit_poisson_gllvm_va
fit_binomial_gllvm_va
fit_nb_gllvm_va
fit_beta_gllvm_va
fit_gamma_gllvm_va
fit_exponential_gllvm_va
fit_delta_gamma_gllvm_va
```

### Initializers & Solvers

```@docs
ppca_init
em_fa
GLLVModels.pack_lambda
GLLVModels.unpack_lambda
GLLVModels.rr_theta_len
GLLVModels.rotate_to_lower_triangular
GLLVModels.low_rank_chol
GLLVModels.LowRankPlusDiagChol
```

---

## Post-Fit & Ordination

```@docs
getLV
getLoadings
rotation
ordination
ordiplot
ordination_uncertainty
select_lv
cv_gllvm
simulate
predict
residuals
fitted
coef_table
loglikelihood
aic
bic
dof
nobs
stderror
vcov
coeftable
coef
lognormal_response_mean
observed_mask
```

---

## Inference & Confidence Intervals

```@docs
confint
profile_ci
bootstrap_ci
transformed_wald_ci_derived
correlation_wald_ci
communality_wald_ci
icc_wald_ci
phylo_signal_wald_ci
chibar2_pvalue
variance_lrt
profile_ci_variance
confint_spde_latent
confint_speciescov
confint_fourthcorner
confint_rrr
confint_constrained
confint_lv_effects
bridge_fit
bridge_capabilities
```

---

## Covariance & Summary Extractors

```@docs
sigma_y_site
communality
correlation
phylo_signal
link_residual
repeatability
communality_B
communality_W
correlation_B
correlation_W
row_effects
extract_lv_effects
extract_Gamma
coevolution_gamma
```

---

## Structured Covariance Builders

### Spatial & SPDE Models

```@docs
spatial_cov
spde_fem
spde_precision
spde_projector
matern_correlation
spde_mesh_grid
spde_mesh_delaunay
fit_spde_gaussian
fit_spde_latent_gllvm
predict_spatial
```

### Phylogenetic & Coevolution Models

```@docs
relatedness_cov
fit_phylo_gaussian
fit_phylo_glm
fit_coevolution_gaussian
fit_coevolution_blockna
fit_coevolution_glm
make_cross_kernel
augmented_phy
PrecisionPhy
precision_logdet_check
phylo_precision_payload
admit_phylo_precision_payload
random_balanced_tree
sigma_phy_dense
node_grad
node_dσ_phy_only
node_blups
build_node_perspecies
grad_node_perspecies
FelsensteinContrasts
felsenstein_contrast_matrix
felsenstein_contrasts
contrast_transform
edge_phy
sigma_phy_dense_edge
log_det_Q
solve_Q
Q_times_x
path_membership
simulate_branch_re
branch_blups
branch_re_profile_negll
fit_branch_re
fit_branch_re_dense
clade_edges
find_clade_root
clade_detection
build_AnB_sparse
solve_AnB
blup_phylo_sparse
em_fit_phylo
em_observed_information
em_fit_phylo_squarem
edge_W_diag
Q_perbranch
simulate_relaxed_bm
estep_edge_moments
shrink_logrates
fit_relaxed_clock
spearman
shrinkage_factor
welch_t
rank_sum_z
excess_kurtosis
qq_max_dev
```

### Likelihood & Gradient Kernels

```@docs
GLLVModels.gaussian_marginal_loglik
GLLVModels.gaussian_profile_nll
GLLVModels.gaussian_nll_packed
GLLVModels.gaussian_lv_nll_packed
GLLVModels.binomial_marginal_loglik_laplace
GLLVModels.binomial_lv_nll_packed
binomial_laplace_grad
GLLVModels.poisson_marginal_loglik_laplace
GLLVModels.poisson_lv_nll_packed
poisson_laplace_grad
GLLVModels.nb_marginal_loglik_laplace
GLLVModels.nb_lv_nll_packed
nb_laplace_grad
GLLVModels.gamma_marginal_loglik_laplace
GLLVModels.gamma_lv_nll_packed
gamma_laplace_grad
GLLVModels.beta_marginal_loglik_laplace
GLLVModels.beta_lv_nll_packed
beta_laplace_grad
GLLVModels.ordinal_marginal_loglik_laplace
GLLVModels.ordinal_lv_nll_packed
GLLVModels.marginal_loglik_laplace
GLLVModels.laplace_loglik_site
GLLVModels.marginal_loglik_laplace_mi
GLLVModels.marginal_loglik_laplace_xs
GLLVModels.laplace_loglik_site_mi
GLLVModels.laplace_loglik_site_xs
gaussian_reml_loglik
gaussian_grouped_intercept_loglik
twolevel_marginal_loglik
random_slope_marginal_loglik_laplace
gaussian_pervar_marginal_loglik
compoisson_marginal_loglik_laplace
compoisson_logpdf
compoisson_logz
gaussian_marginal_loglik_sparse_phy
phylo_glm_marginal_loglik
coevolution_glm_marginal_loglik
gaussian_marginal_loglik_contrasts
gaussian_marginal_loglik_edge_phy
mixed_marginal_loglik_laplace
truncated_poisson_marginal_loglik_laplace
GLLVModels.censored_poisson_marginal_loglik_laplace
GLLVModels.censored_bounds_to_YN
truncated_nbinom2_marginal_loglik_laplace
truncated_nbinom2_pertrait_marginal_loglik_laplace
gp1_marginal_loglik_laplace
nb1_marginal_loglik_laplace
nb_grouped_marginal_loglik_laplace
beta_grouped_marginal_loglik_laplace
gamma_grouped_marginal_loglik_laplace
nb1_grouped_marginal_loglik_laplace
tweedie_grouped_marginal_loglik_laplace
studentt_marginal_loglik_laplace
lognormal_marginal_loglik
exponential_marginal_loglik_laplace
tweedie_marginal_loglik_laplace
tweedie_logpdf
tweedie_cdf
delta_lognormal_marginal_loglik_laplace
hurdle_poisson_marginal_loglik_laplace
hurdle_nb_marginal_loglik_laplace
delta_gamma_marginal_loglik_laplace
beta_hurdle_marginal_loglik_laplace
zip_marginal_loglik_laplace
zinb_marginal_loglik_laplace
zib_marginal_loglik_laplace
row_random_marginal_loglik_laplace
constrained_marginal_loglik_laplace
rrr_marginal_loglik
quadratic_marginal_loglik_laplace
ordered_beta_marginal_loglik_laplace
GLLVModels.ordered_beta_logp
betabinomial_marginal_loglik_laplace
betabinomial_grouped_marginal_loglik_laplace
GLLVModels.betabinomial_logp
GLLVModels.twopart_marginal_loglik_laplace
GLLVModels.multinomial_loglik
GLLVModels.multinomial_eta
GLLVModels.unpack_multinomial
GLLVModels.multinomial_pack_len
GLLVModels.proportions
GLLVModels.aghq_grid
GLLVModels.aghq_grid_ok
GLLVModels.aghq_stage1a_marginal_loglik
GLLVModels.aghq_stage1a_loglik_site
GLLVModels.AGHQGrid
beta_marginal_loglik_va
delta_gamma_marginal_loglik_va
poisson_marginal_loglik_va
binomial_marginal_loglik_va
nb_marginal_loglik_va
gamma_marginal_loglik_va
exponential_marginal_loglik_va
spde_gaussian_marginal_loglik
spde_latent_marginal_loglik
GLLVModels.takahashi_selinv
GLLVModels.takahashi_diag
GLLVModels.build_sparse_phy_state
GLLVModels.leaf_block_inv
make_phy
GLLVModels.profile_recover
GLLVModels.profile_ci_derived
GLLVModels.bootstrap_ci_derived
```

---

## Types & Link Functions

### Link Functions

```@docs
LogitLink
ProbitLink
CLogLogLink
IdentityLink
LogLink
GLLVModels.linkinv
GLLVModels.linkfun
GLLVModels.mu_eta
```

### Fit Result Types

```@docs
GllvmFit
GllvmModel
GllvmCovFit
GllvmSpeciesCovFit
PoissonFit
TruncatedPoissonFit
CensoredPoissonFit
NBFit
TruncatedNegBin2Fit
TruncatedNegBin2PerTraitFit
NB1Fit
NBGroupedFit
NBGroupedCovFit
NB1GroupedFit
NB1GroupedCovFit
BetaFit
BetaGroupedFit
BetaGroupedCovFit
BetaBinomialFit
BetaBinomialGroupedFit
BetaBinomialGroupedCovFit
GammaFit
GammaGroupedFit
GammaGroupedCovFit
ExponentialFit
OrdinalFit
OrdinalPerTraitFit
OrdinalPerTraitCovFit
TweedieFit
TweedieGroupedFit
TweediePerTraitPowerFit
StudentTFit
LognormalFit
MultinomialFit
DeltaLogNormalFit
DeltaGammaFit
HurdlePoissonFit
HurdleNBFit
BetaHurdleFit
OrderedBetaFit
ZIPFit
ZIPCovFit
ZINBFit
ZINBCovFit
ZIBFit
ZIBCovFit
GP1Fit
COMPoissonFit
PhyloGaussianFit
PhyloGLMFit
CoevolutionGLMFit
SPDEGaussianFit
SPDELatentFit
TwoLevelFit
GaussianREMLFit
GaussianRandomSlopeFit
PoissonRandomSlopeFit
GaussianPerVarFit
FourthCornerFit
ConstrainedOrdinationFit
ConcurrentOrdinationFit
RRRFit
QuadraticFit
RowEffectFit
RowRandomFit
MixedFamilyFit
BranchREFit
BranchRECache
RelaxedClockFit
EMPhyloFit
AnBSparseSolver
AugmentedPhy
GllvmCoefTable
LVSelection
CVResult
```

### Family & Distribution Markers

```@docs
StudentTFamily
Lognormal
Multinomial
DeltaLogNormal
DeltaGamma
HurdlePoisson
HurdleNB
BetaHurdle
OrderedBeta
Ordinal
ZIPoisson
ZINegBin
GLLVModels.ZINB
ZIB
BetaBinom
COMPoisson
TruncatedPoisson
CensoredPoisson
TruncatedNegBin2
NB1
```

## Experimental Poisson quadrature

Ordinary log-link Poisson models can opt into adaptive Gauss–Hermite
quadrature (AGHQ), which integrates over latent scores using a grid adapted to
each site's conditional mode. Responses are rows and sites are columns:

```julia
fit = fit_poisson_gllvm(Y; K=2, aghq=5)
fit.integration.actual                 # :aghq or :laplace
fit.integration.reason                 # stopping or fallback reason
fit.integration.result                 # retained optimization attempts
predict(fit, Y)                         # conditional rates, including offsets
confint(fit, Y; parm="beta")            # fitted frozen-node objective
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

## Experimental binomial quadrature

For successes out of known trials, supply `N` with the same responses × sites
shape as `Y`. Omit `N` only for Bernoulli observations. The ordinary binomial
route accepts logit, probit and complementary-log-log links, with the same
node controls and frozen-node convergence rule as the Poisson route.

```julia
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

### Gaussian integration metadata

`fit_gaussian_gllvm(...; aghq=3)` retains `GllvmFit` and its parameter layout.
The optional `integration` field distinguishes actual quadrature from exact
Gaussian/Laplace fallback and retains node counts, controls, starting vectors,
convergence, caches and observed-input identity. Default `aghq=false` leaves
legacy fitting unchanged. `X=nothing` remains zero mean, and `β_fixed` retains
fixed-zero coefficients. `mask`/missing responses and offsets are supported on
the ordinary loadings-only route. See the executed Gaussian quickstart.

`getLV`, `predict`, `residuals`, `simulate`, `confint`, `profile_ci` and
`bootstrap_ci` use recorded data and estimator identity. Gaussian `vcov` for a
recorded fit returns the full working-parameter covariance, not only its diagonal;
`confint` transforms residual-SD estimates/bounds to the natural scale while
standard errors remain on the working scale. Legacy `bootstrap_ci` outputs
working-scale bounds. Failed bootstrap attempts remain visible. This is an
experimental option, not a complete R-parity or calibrated-inference claim.
