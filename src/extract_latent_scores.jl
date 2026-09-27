# Julia twin of gllvmTMB's extract_latent_scores() (R/extract-latent-scores.R,
# pinned P1 9539352f66f2db2cc26b1c393e67212a359b60c9, gllvmTMB 0.7.1).
#
# R's generic dispatches on four S3 methods: .default, .gllvmTMB_multi,
# .gllvmTMB_site_trait_sim, .gllvmTMB_va. Per the P1 case map
# (GLLVModels.jl PR #526), .gllvmTMB_site_trait_sim and .gllvmTMB_va are
# EXCLUDED from this twin: Julia has neither a site-trait-simulation class
# (simulate_site_trait()'s truth$z_B/z_W generating draws) nor a
# variational-approximation fit class to extract scores from. This file
# twins only .default and .gllvmTMB_multi.

"""
    extract_latent_scores(fit, y; level=:unit, kwargs...) -> Matrix{Float64} or Nothing

Twin of gllvmTMB's `extract_latent_scores(x, level)` for a fitted model
(`R/extract-latent-scores.R`, `extract_latent_scores.gllvmTMB_multi`). R
documents the point estimate as matching `extract_ordination(x, level, component
= "innovation")`, `getLV(x, level)` at `rotate = "none"`, and
`ordination_uncertainty(x, level)\$scores`. In this package the same identity
holds against this package's own [`getLV`](@ref): `level = :unit` returns
`getLV(fit, y; component = :innovation, rotate = false, kwargs...)`, the
Gaussian posterior mean / Laplace mode of the between-unit latent scores in
native (unrotated) orientation.

`level = :unit_obs` always returns `nothing`. R's `unit_obs` (`z_W`) tier is
the within-unit block of gllvmTMB's two-tier `latent(0 + trait | site, d)`
"multi" formula grammar — a nested between-unit/within-unit reduced-rank
structure fit on a long-format trait table, entirely separate from the
single-tier wide-format `Y` (species/trait × site) GLLVM this package
implements. No fit type in this package has a `unit_obs` tier, so this is the
"no such tier" case R itself returns `NULL` for whenever `fit\$use\$rr_W` is
unset — always true here, not an approximation.

`y` and `kwargs...` (e.g. `X`, `X_lv`, `N`, `mask`, `offset`) forward to the
fit-type-specific [`getLV`](@ref) method exactly as `getLV` itself requires
them for that fit type.

# Differences from R (documented, not twinned)
- **Explicit `y`**: R's fitted object retains its TMB environment and needs
  only `(x, level)`; this package's fits do not store the data (see
  [`getLV`](@ref)), so `y` (and any of `X`/`X_lv`/`N`/`mask`/`offset` the fit
  used) must be supplied here.
- **No row/column names**: R returns unit row names and `"LV1"`, `"LV2"`, ...
  column names. This package's `getLV` already returns a plain
  `Matrix{Float64}` with no names — a pre-existing, package-wide convention,
  not introduced here.
- **`gllvmTMB_multi`'s trait-table model family**: beyond the ordinary
  single-tier abundance/trait GLLVM twinned here, R's `gllvmTMB_multi` class
  also covers the `latent(0 + trait | site, d)` two-tier trait-table grammar
  (unbalanced replication per unit, an optional `unit_obs` tier). That model
  family is a documented Julia gap, not part of this twin.

See also [`getLV`](@ref), [`extract_ordination`](@ref) is R-only (no Julia
twin exists for it; `getLV` is this package's public accessor).
"""
function extract_latent_scores(fit::AnyGllvmFit, y::AbstractMatrix;
                                level::Symbol = :unit, kwargs...)
    level === :unit || level === :unit_obs ||
        throw(ArgumentError("level must be :unit or :unit_obs; got :$level"))
    level === :unit_obs && return nothing
    return getLV(fit, y; component = :innovation, rotate = false, kwargs...)
end

"""
    extract_latent_scores(x, args...; kwargs...)

Fallback twin of `extract_latent_scores.default` (`R/extract-latent-scores.R`),
which aborts with a named error for any object without a specific method.
Without this method, calling `extract_latent_scores` on an unsupported type
would raise Julia's generic `MethodError`; this gives the same kind of
informative, named failure R raises instead.
"""
function extract_latent_scores(x, args...; kwargs...)
    throw(ArgumentError(
        "no extract_latent_scores method for $(typeof(x)); expected a fitted " *
        "GLLVModels fit object (see AnyGllvmFit in src/postfit.jl)"))
end
