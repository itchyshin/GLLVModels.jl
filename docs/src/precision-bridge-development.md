# Multivariate phylogenetic precision bridge (developer guide)

**Audience: developers extending the Julia interface.** This page describes the
Julia-side interface for a complete Gaussian multivariate phylogenetic model.
It is useful when you already have a canonical sparse phylogenetic-precision
payload and a traits-by-observations matrix. This model is not currently
available through the public R `phylo_rr` workflow.

The model has a rank-`d` trait loading matrix `L`, optional phylogenetic
trait-specific variances `U`, and independent observation residual variances
`psi`. Its phylogenetic trait covariance is `L * L' + Diagonal(U)`; `U` is
not observation noise. The sparse augmented precision retains unobserved
ancestors and marginalises them in the fit.

## Julia call

The example below constructs a small native payload only to make the shape of
the call reproducible. In a bridge workflow, `payload` instead comes from the
already-admitted canonical precision transport.

```julia
using GLLVModels

# Four tips; the root-dropped augmented precision also retains internal nodes.
pp = GLLVModels.PrecisionPhy(GLLVModels.random_balanced_tree(4; branch_length = 0.3))
payload = GLLVModels.phylo_precision_payload(pp)

# Rows are traits and columns are observations. Tips may repeat across columns.
y = [1.2 0.8 1.1 0.9 1.0 0.7 1.3 0.6;
     0.1 0.4 0.2 0.3 0.0 0.5 0.2 0.4;
     0.7 0.5 0.6 0.8 0.9 0.4 0.8 0.5]

fit = GLLVModels.bridge_fit(
    y = y,
    family = "gaussian",
    d = 1,                         # rank of the phylogenetic loading matrix
    phylo = payload,
    options = Dict(
        "phylo_model" => "multivariate",
        "species_id" => [1, 2, 3, 4, 1, 2, 3, 4],
        "mode" => "explicitunique", # or "barelowrank"
        "residual_mode" => "trait", # or "shared": one common residual variance
        "ci_method" => "none",      # "wald" is the only interval route here
        "iterations" => 200,
    ),
)
```

`X` may be supplied as a complete mean design accepted by the native fitter:
`size(y, 1) * size(y, 2)` rows in trait-major `vec(y)` order, or a
`n_traits x n_observations x n_coefficients` array. It is forwarded through
the bridge rather than discarded. Missing responses, `N`, latent-score
covariates, and label overrides are not supported by this route.

## Maps, scale, and retained ancestors

Two maps intentionally use different bases.

| Quantity | Where it appears | Base | Meaning |
| --- | --- | --- | --- |
| `payload.species_aug_id` | precision wire bundle | 0-based | one unique tip-to-augmented-node entry per tip |
| `options["species_id"]` | `bridge_fit` call | 1-based | one observation-to-tip entry per column of `y`; repetitions are allowed |
| `fit.species_aug_id` | returned flat result | 1-based | native tip-to-augmented-node map |
| `fit.species_id` | returned flat result | 1-based | exact observation-to-tip map used by the fitter |

The precision bundle also carries `scale` and native `log_det = log|Q|`.
`Q` already contains any correlation/unit-height scaling, so the Julia fit
does not apply `scale` a second time. The reference R normaliser's
`log_det_A_phy_rr` is a covariance log determinant and therefore has the
opposite sign; payload admission performs the required native precision
convention check. Do not construct a dense response covariance or drop
ancestor rows before calling the bridge.

The fitter warns if the requested factor-plus-unique decomposition has more
coordinates than the trait covariance has entries (for example, rank one with
two traits and two unique variances). This is structural non-identification,
not a problem that more observations can resolve. Valid marginal intervals
require an identifiable supported parameterisation as well as adequate data.
Known coordinate redundancy yields `:nonidentifiable` interval diagnostics;
a positive numerical Hessian cannot override that mathematical limitation.

## Flat result contract

`bridge_fit` returns a JuliaCall-safe flat named tuple. The key groups are:

`residual_mode` records `"trait"` or `"shared"`, and `parameter_labels`
describes the packed coordinates. Shared mode has only one residual log-SD
coordinate even though residual variances are returned for every trait.

The machine-readable `admission_status == "closed"` and
`admission_scope == "R phylo_rr"` state that this model is not available
through the legacy R route. Consumers must honour these fields; point
convergence does not change that status.

- Mean and covariance: `coefficients`, `coefficient_names`, `mean_design`,
  `loadings`, `phylo_unique_variance`, `phylo_covariance`,
  `residual_variance`, and `residual_covariance`.
- Fit diagnostics: `parameters`, `loglik`, `converged`, `gradient_max`,
  `hessian_min_eigenvalue`, `hessian_positive_definite`,
  `hessian_condition_number`, `iterations`, and `stopping_reason`.
- Precision metadata: `species_id`, `species_aug_id`, `node_labels`,
  `n_leaves`, `n_aug`, `scale`, and `log_det`.
- Signal boundary: `phylogenetic_signal_status`,
  `phylogenetic_signal_definition`, and `phylogenetic_signal_message`.
  Signal is `estimand_not_admitted`: the frozen R default uses species-level
  latent variance, excluding observation residuals. The matching species-level
  non-phylogenetic decomposition and extractor still need paired validation.
  The bridge does not substitute a share of observation variance.

With `"ci_method" => "wald"`, inspect `ci_status` and the parallel
`ci_target_names`, `ci_estimate`, `ci_lower`, `ci_upper`,
`ci_se_transformed`, `ci_transforms`, `ci_target_methods`, and `ci_statuses`
arrays. An unavailable target is a result: it can indicate nonconvergence,
nonstationarity, non-positive-definite curvature, or a boundary estimate.
With `"ci_method" => "none"`, these arrays are empty and
`ci_status == "not_requested"`.

## Native Julia fitting and joint grouping

The native interface also accepts an already constructed `PrecisionPhy`:

```julia
# Y: traits × observations; species_id: one valid tip index per observation.
fit = fit_gllvm(Y; phylo=precision, phylo_rank=1,
    phylo_mode=:barelowrank, species_id=species_id)
intervals = precision_multivariate_intervals(fit)

# Precision-only alternative: one observation residual variance for all traits.
shared_fit = fit_gllvm(Y; phylo=precision, phylo_rank=1,
    species_id=species_id, residual_mode=:shared)

# Each ordinary effect is explicit; a group label alone adds no random effect.
joint = fit_gllvm(Y; phylo=precision, phylo_rank=1, species_id=species_id,
    grouping=[GroupingTerm(:cluster; mode=:indep, common=false)],
    cluster=cluster)
joint_intervals = joint_phylo_grouped_intervals(joint)
population_mean = predict(joint)
phylogenetic_covariance = extract_Sigma(joint; level=:phylo).Sigma
group_covariance = extract_Sigma(joint; level=:cluster).Sigma
residual_covariance = extract_Sigma(joint; level=:residual).Sigma
fixed_effect_summary = summary(joint, Y)
```

The joint fit uses **one** marginal likelihood for shared phylogenetic and
ordinary effects. This initial joint route requires independent, non-common
trait variances for every ordinary term and trait-specific observation
residual variances. Covariance extraction keeps the phylogenetic, ordinary
and observation sources separate; the phylogenetic trait covariance is the
multiplier of the supplied group covariance, not a rescaled tip covariance.
`predict` returns the Gaussian population mean (also its marginal mean), not
conditional random-effect predictions or predictive intervals. `vcov` and
`stderror` refer to the full-nuisance fixed-effect covariance and fail explicitly
when that information is unavailable. Inspect interval and summary statuses;
a stopped or converged point fit alone does not validate its uncertainty.

Precision-only fitting defaults to `residual_mode=:trait`; `:shared` instead
uses one log-SD coordinate for all traits. Its per-trait reported residual
variances and interval targets then refer to the same estimated quantity,
not separate variance components. This residual layout matches the simplest
frozen-R phylogenetic model; paired-R qualification remains outstanding.
The joint grouping route still uses trait-specific residuals only.

Every fitting route validates the supplied precision matrix itself for
symmetry, positive definiteness and its determinant checksum, then retains
an owned copy of the precision and maps. A malformed raw `PrecisionPhy`
object is not admitted merely because it has the expected type.

Joint fits also check whether their actual covariance components can be
distinguished. For example, an ordinary group with one level per observation
duplicates observation residual variance; two groupings with identical group
membership identify only their sum. These cases warn and withhold component
intervals, even if a fixed-parameter likelihood is finite. The diagnostic uses
the requested loading structure, so a low-rank model is not tested as though
its trait covariance were unrestricted. A local diagnostic passing is not a
recovery or interval-coverage certificate.

For formulas, use `gllvm(@formula(y ~ 1 + x), Y, data; phylo=precision,
species_id=:tip, grouping=terms, cluster=:batch)`. The formula supplies the
complete mean design and resolves both mapping and grouping columns. Omit
`grouping` and its labels for precision-only fitting. Do not supply global `K`,
`num_lv`, `row_eff`, `disp_group`, or `pervar`; use `phylo_rank` instead.
Non-Gaussian precision fitting and unsupported ordinary covariance modes are
rejected. Native precision scaling and the bridge admission boundary are
unchanged. Neither interval function estimates phylogenetic signal.

## Bridge admission boundary

This route accepts only Gaussian/normal data, a positive rank `d`,
`"barelowrank"` or `"explicitunique"` mode, and `"none"` or `"wald"`
intervals. It rejects unsupported families, profiles, maps, and unknown
options rather than silently changing the model. It fixes the phylogenetic
scale at one to avoid confounding that scale with `L` and `U`.

Most importantly, the public R `phylo_rr` admission remains **closed**.
The ordinary R-to-Julia bridge should not be presented as supporting this
model until paired R model tests support it. This page documents the Julia-side
contract and its diagnostics; it makes no parity, profile-interval, recovery,
or coverage claim.
