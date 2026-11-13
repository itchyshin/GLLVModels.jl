# NA input on the Gaussian fitters that have no `mask` (and so no way to use a missing cell).
#
# Before this file a `Matrix{Union{Missing,T}}` reached them as a no-method error at dispatch
# (their `Y` is typed `AbstractMatrix{<:Real}`) or deep inside the fit, and NaN went through
# `em_fa` to a NaN result without a word. A `missing` cell is now a typed ArgumentError that names
# the fitter, the count and the first cell. A `Union{Missing,T}` eltype with nothing missing is
# narrowed to `T` and fits like the plain matrix. These fitters take no `mask`, so there is no
# estimator choice here; `fit_gaussian_gllvm` / `fit_gaussian_pervar_gllvm` derive a mask from
# `missing` and are not touched.
function _refuse_na_response(caller::AbstractString, Y::AbstractMatrix; nan::Bool = false)
    if any(ismissing, Y)
        bad = findall(ismissing, Y)
        throw(ArgumentError("$caller does not accept `missing` in `Y`: $(length(bad)) missing " *
            "cell(s), first at (row, column) = ($(bad[1][1]), $(bad[1][2])). This fitter has no " *
            "`mask`; remove or impute the missing cells, or use `fit_gaussian_gllvm`, which " *
            "derives a mask from `missing`."))
    end
    if nan && eltype(Y) <: AbstractFloat && !all(isfinite, Y)
        for (kind, test) in (("NaN", isnan), ("Inf", isinf))
            bad = findall(test, Y)
            isempty(bad) || throw(ArgumentError("$caller does not accept $kind in `Y`: $(length(bad)) $kind cell(s), " *
                "first at (row, column) = ($(bad[1][1]), $(bad[1][2]))."))
        end
    end
    T = nonmissingtype(eltype(Y))
    # `Any` must be refused before the early return: its non-missing type is `Any` itself, so the
    # method that called this helper would receive the same matrix and call it again forever.
    T === Any && throw(ArgumentError("$caller needs a numeric response matrix; got element type $(eltype(Y))."))
    T === eltype(Y) && return Y
    return Matrix{T}(Y)
end

em_fa(y::AbstractMatrix{>:Missing}, K::Integer; kwargs...) =
    em_fa(_refuse_na_response("em_fa", y), K; kwargs...)
fit_gaussian_reml(y::AbstractMatrix{>:Missing}, X::AbstractArray{<:Real, 3}; kwargs...) =
    fit_gaussian_reml(_refuse_na_response("fit_gaussian_reml", y), X; kwargs...)
fit_twolevel_gaussian(y::AbstractMatrix{>:Missing}, individual::AbstractVector; kwargs...) =
    fit_twolevel_gaussian(_refuse_na_response("fit_twolevel_gaussian", y), individual; kwargs...)
fit_gaussian_random_slope(y::AbstractMatrix{>:Missing}, grouping::AbstractVector, Z::AbstractMatrix; kwargs...) =
    fit_gaussian_random_slope(_refuse_na_response("fit_gaussian_random_slope", y), grouping, Z; kwargs...)
em_fit_phylo(y::AbstractMatrix{>:Missing}, K_B::Integer, Σ_phy::AbstractMatrix; kwargs...) =
    em_fit_phylo(_refuse_na_response("em_fit_phylo", y), K_B, Σ_phy; kwargs...)
em_fit_phylo_squarem(y::AbstractMatrix{>:Missing}, K_B::Integer, Σ_phy::AbstractMatrix; kwargs...) =
    em_fit_phylo_squarem(_refuse_na_response("em_fit_phylo_squarem", y), K_B, Σ_phy; kwargs...)
fit_coevolution_gaussian(Y::AbstractMatrix{>:Missing}, K_star::AbstractMatrix; kwargs...) =
    fit_coevolution_gaussian(_refuse_na_response("fit_coevolution_gaussian", Y), K_star; kwargs...)
fit_gaussian_sources(Y::AbstractMatrix{>:Missing}; kwargs...) =
    fit_gaussian_sources(_refuse_na_response("fit_gaussian_sources", Y); kwargs...)
fit_precision_multivariate(Y::AbstractMatrix{>:Missing}, phy::PrecisionPhy; kwargs...) =
    fit_precision_multivariate(_refuse_na_response("fit_precision_multivariate", Y), phy; kwargs...)
fit_joint_phylo_grouped_gaussian(Y::AbstractMatrix{>:Missing}, phy::PrecisionPhy; kwargs...) =
    fit_joint_phylo_grouped_gaussian(_refuse_na_response("fit_joint_phylo_grouped_gaussian", Y), phy; kwargs...)
fit_grouped_gaussian(Y::AbstractMatrix{>:Missing}; kwargs...) =
    fit_grouped_gaussian(_refuse_na_response("fit_grouped_gaussian", Y); kwargs...)
fit_phylo_latent_gllvm(Y::AbstractMatrix{>:Missing}, species::AbstractVector; kwargs...) =
    fit_phylo_latent_gllvm(_refuse_na_response("fit_phylo_latent_gllvm", Y), species; kwargs...)
fit_lognormal_gllvm(Y::AbstractMatrix{>:Missing}; kwargs...) =
    fit_lognormal_gllvm(_refuse_na_response("fit_lognormal_gllvm", Y); kwargs...)
fit_delta_lognormal_gllvm(Y::AbstractMatrix{>:Missing}; kwargs...) =
    fit_delta_lognormal_gllvm(_refuse_na_response("fit_delta_lognormal_gllvm", Y); kwargs...)
