# Population-level `predict(fit, design)` without responses is implemented only on a
# narrow set of fit records (latent held at prior mean or deterministic in X).
# Other covariate-capable types expose conditional `predict(fit, Y, X)` instead.

const _PopulationPredictUnsupportedCovFit = Union{
    NBGroupedCovFit, NB1GroupedCovFit, BetaGroupedCovFit, GammaGroupedCovFit,
    OrdinalPerTraitCovFit,
    ZIPCovFit, ZINBCovFit, ZIBCovFit,
    GllvmSpeciesCovFit,
    BetaBinomialGroupedCovFit,
    ConstrainedOrdinationFit,
}

function _throw_population_predict_unsupported(fit)
  T = nameof(typeof(fit))
  throw(ArgumentError(
    "population-level predict(fit, design) is not implemented for $T " *
    "(unseen sites have no estimable latent score). " *
    "For conditional in-sample predictions at the Laplace mode, use predict(fit, Y, design). " *
    "New-site prediction from the design alone is supported for GllvmCovFit " *
    "(trait-major p×n×q array), RRRFit (n×q site matrix), GroupedGaussianFit, " *
    "PrecisionMultivariateFit, GroupedNonGaussianFit, and JointPhyloGroupedGaussianFit " *
    "(predict(fit[, design]); random effects at mean zero). Spatial latent fits use predict_spatial."))
end

function predict(fit::_PopulationPredictUnsupportedCovFit,
    design::Union{AbstractArray{<:Real, 3}, AbstractMatrix{<:Real}};
    kwargs...)
    _throw_population_predict_unsupported(fit)
end

function predict(fit::GllvmFit, design::AbstractArray{<:Real, 3}; kwargs...)
    _throw_population_predict_unsupported(fit)
end