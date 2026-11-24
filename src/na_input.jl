# NA input on the Gaussian fitters that have no `mask` (and so no way to use a missing cell).
#
# These fitters already accept any `AbstractMatrix` for `Y`, so a `Matrix{Union{Missing,T}}` reaches
# the body and used to fail there with a `MethodError` from a `convert` (or, for `em_fa`, NaN went
# through to an all-NaN result without a word). `_refuse_na_response` is a check only: it throws a
# typed ArgumentError naming the fitter, the count and the first cell, and returns `nothing`.
# It adds no method and does not change the type of `Y`, so no signature changes.
# `fit_gaussian_gllvm` / `fit_gaussian_pervar_gllvm` derive a mask from `missing` and are not touched.
function _refuse_na_response(caller::AbstractString, Y::AbstractMatrix; nan::Bool = false)
    if Missing <: eltype(Y) && any(ismissing, Y)
        bad = findall(ismissing, Y)
        throw(ArgumentError("$caller does not accept `missing` in `Y`: $(length(bad)) missing " *
            "cell(s), first at (row, column) = ($(bad[1][1]), $(bad[1][2])). This fitter has no " *
            "`mask`; remove or impute the missing cells, or use `fit_gaussian_gllvm`, which " *
            "derives a mask from `missing`."))
    end
    if nan && nonmissingtype(eltype(Y)) <: AbstractFloat && !all(isfinite, Y)
        for (kind, test) in (("NaN", isnan), ("Inf", isinf))
            bad = findall(test, Y)
            isempty(bad) || throw(ArgumentError("$caller does not accept $kind in `Y`: $(length(bad)) $kind cell(s), " *
                "first at (row, column) = ($(bad[1][1]), $(bad[1][2]))."))
        end
    end
    return nothing
end

# #1147: the fitters typed `AbstractMatrix{<:Real}` have no method for a `Union{Missing,T}` eltype, so a
# `missing` cell was a dispatch MethodError. These forwarding methods apply to that eltype only: a missing
# cell is a typed ArgumentError, and a Union eltype with nothing missing is converted and fitted as before.
function _drop_missing_eltype(caller::AbstractString, Y::AbstractMatrix)
    _refuse_na_response(caller, Y)
    return Matrix{nonmissingtype(eltype(Y))}(Y)
end
fit_gaussian_sources(Y::AbstractMatrix{>:Missing}; kwargs...) =
    fit_gaussian_sources(_drop_missing_eltype("fit_gaussian_sources", Y); kwargs...)
fit_phylo_latent_gllvm(Y::AbstractMatrix{>:Missing}, species::AbstractVector; kwargs...) =
    fit_phylo_latent_gllvm(_drop_missing_eltype("fit_phylo_latent_gllvm", Y), species; kwargs...)
