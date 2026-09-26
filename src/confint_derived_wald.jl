# Transformed-scale Wald confidence intervals for *derived bounded
# quantities* of a fitted Gaussian GLLVM.
#
# Motivation
# ----------
# src/confint_derived.jl gives two CI methods for derived scalars g(θ)
# (cross-trait correlation ρ ∈ [−1, 1], communality c² ∈ [0, 1], ICC ∈
# [0, 1], phylogenetic signal H² ∈ [0, 1]): the parametric bootstrap and
# the constrained-refit profile. Both are expensive — each needs many
# refits. The per-parameter Wald machinery in src/confint.jl is cheap (one
# Hessian) and, crucially, already gets ~nominal coverage for the SD
# parameters by building the interval on the *log* scale and
# back-transforming: `exp(log σ̂ ± z·SE_log)`. The back-transform keeps the
# bound positive, and the symmetric-on-the-log-scale interval matches the
# sampling geometry far better than a raw-scale `σ̂ ± z·SE`.
#
# This file applies the same idea to the *derived* bounded quantities. For
# a derived scalar g(θ) with a natural bound we pick a link h that maps the
# bounded range to the whole real line, build the Wald interval there, and
# map back:
#
#   correlation ρ ∈ [−1, 1]:   h = Fisher-z = atanh,  back = tanh
#   c² / ICC / H² ∈ [0, 1]:    h = logit,             back = logistic
#
# Delta-method SE on the transformed scale:
#   SE_h = sqrt( ∇θ(h∘g)' · Σ · ∇θ(h∘g) ),   Σ = inv(observed information),
# with ∇θ(h∘g) obtained by ForwardDiff on the *dense marginal* packed-θ
# closure (the derived functions in confint_derived.jl are AD-friendly:
# ForwardDiff Duals flow through unpack_lambda, exp, division). The CI is
#   back( h(g(θ̂)) ± z·SE_h ),
# which is guaranteed to lie inside the natural range because `back` maps
# ℝ → (−1, 1) (resp. (0, 1)).
#
# The observed information H = ForwardDiff.hessian(nll, θ̂) and Σ = inv(H)
# are exactly the objects src/confint.jl already builds; we reuse its
# _confint_reconstruct_nll helper so the Hessian convention is identical.
#
# This file is additive: it does NOT modify confint.jl or
# confint_derived.jl. It defines packed-θ closures for the derived
# quantities that did not yet have one (correlation, phylo signal); the
# communality/ICC closure already exists as _communality_packed.

using Distributions: Normal, quantile
using LinearAlgebra: diag

# ---------------------------------------------------------------------------
# Link functions (transformed scale ↔ natural scale).
# Written generically so ForwardDiff Duals pass through h∘g unchanged.
# ---------------------------------------------------------------------------

# Fisher-z for correlations on [−1, 1].
_tw_fisher_z(ρ)     = atanh(ρ)
_tw_fisher_z_inv(z) = tanh(z)

# logit / logistic for quantities on [0, 1].
_tw_logit(x)        = log(x / (1 - x))
_tw_logistic(z)     = 1 / (1 + exp(-z))

# log / exp for strictly positive unbounded quantities (variances, SDs) — the
# repo's unconstrained-internal-scale convention (see σ_eps in packing.jl).
_tw_log(x)      = log(x)
_tw_exp(z)      = exp(z)

# identity — for raw-scale quantities with no natural bound (e.g. a raw
# loading entry Λ[t,k]), so the same generic machinery below covers them too.
_tw_identity(x) = x

# Map a transform symbol to (forward link, inverse link, natural bounds).
function _tw_link(transform::Symbol)
    if transform === :fisher_z
        return (_tw_fisher_z, _tw_fisher_z_inv, (-1.0, 1.0))
    elseif transform === :logit
        return (_tw_logit, _tw_logistic, (0.0, 1.0))
    elseif transform === :log
        return (_tw_log, _tw_exp, (0.0, Inf))
    elseif transform === :identity
        return (_tw_identity, _tw_identity, (-Inf, Inf))
    else
        throw(ArgumentError(
            "transform must be :fisher_z (correlations, [−1,1]), " *
            ":logit (communality/ICC/H², [0,1]), :log (positive SDs/" *
            "variances), or :identity (unbounded raw scale); got $(transform)"))
    end
end

# ---------------------------------------------------------------------------
# Packed-θ derived-quantity closures.
#
# _communality_packed already lives in confint_derived.jl (loaded before
# this file). We add the two that did not have a packed form: the
# cross-trait correlation and the phylogenetic signal. Both reconstruct the
# per-site covariance via _sigma_y_site_from_unpacked, exactly as the
# GllvmFit-consuming versions in confint_derived.jl, so the packed value at
# θ̂ equals the public `correlation(fit)` / `phylo_signal(fit)` entry.
# ---------------------------------------------------------------------------

# ρ[i, j] from the packed θ. AD-friendly.
function _correlation_packed(θ::AbstractVector, spec::NamedTuple,
                             i::Integer, j::Integer)
    u = _derived_unpack(θ, spec)
    Σ = _sigma_y_site_from_unpacked(u, spec)
    return Σ[i, j] / sqrt(Σ[i, i] * Σ[j, j])
end

function _make_correlation_closure(spec::NamedTuple, i::Integer, j::Integer)
    return θ -> _correlation_packed(θ, spec, i, j)
end

# H²[t] from the packed θ, mirroring phylo_signal(fit): H² = σ²_phy /
# (σ²_phy + σ²_non), the R oracle's convention (profile-derived.R:145-156).
# Σ_phy enters only through its diagonal (standardised convention → ones).
# AD-friendly.
function _phylo_signal_packed(θ::AbstractVector, spec::NamedTuple, t::Integer;
                              diag_Σphy::Union{Nothing, AbstractVector} = nothing)
    u = _derived_unpack(θ, spec)
    p = spec.p
    Λ_phy = u.Λ_phy
    σ_phy = u.σ_phy
    Λ_phy_aug = if Λ_phy !== nothing && σ_phy !== nothing
        hcat(Λ_phy, reshape(σ_phy, p, 1))
    elseif Λ_phy !== nothing
        Λ_phy
    elseif σ_phy !== nothing
        reshape(σ_phy, p, 1)
    else
        throw(ArgumentError("fit has no phylogenetic block; H² is undefined"))
    end
    Σ = _sigma_y_site_from_unpacked(u, spec)
    ΛΛt_tt = zero(eltype(Λ_phy_aug))
    @inbounds for k in 1:size(Λ_phy_aug, 2)
        ΛΛt_tt += Λ_phy_aug[t, k]^2
    end
    d = diag_Σphy === nothing ? one(eltype(Λ_phy_aug)) : diag_Σphy[t]
    σ²phy = ΛΛt_tt * d
    return σ²phy / (σ²phy + Σ[t, t])
end

function _make_phylo_signal_closure(spec::NamedTuple, t::Integer;
                                    diag_Σphy::Union{Nothing, AbstractVector} = nothing)
    return θ -> _phylo_signal_packed(θ, spec, t; diag_Σphy = diag_Σphy)
end

# ---------------------------------------------------------------------------
# Observed-information Σ = inv(H), reusing confint.jl's NLL reconstruction
# and Hessian convention. Returns (Σ, pd::Bool); Σ is `nothing` when the
# Hessian is unavailable / non-PD.
# ---------------------------------------------------------------------------
function _tw_sigma_from_hessian(fit::GllvmFit, y::AbstractMatrix,
                                X::Union{Nothing, AbstractArray{<:Real, 3}},
                                Σ_phy::Union{Nothing, AbstractMatrix})
    if _has_gaussian_record(fit)
        covariance=_gaussian_record_vcov(fit,y;X=X,Σ_phy=Σ_phy)
        return all(isfinite,covariance) ? (covariance,true) : (nothing,false)
    end
    θ̂ = fit.pars.θ_packed
    nll = _confint_reconstruct_nll(fit, y, X, Σ_phy)
    H = try
        ForwardDiff.hessian(nll, θ̂)
    catch
        return (nothing, false)
    end
    (H === nothing || !all(isfinite, H)) && return (nothing, false)
    Σ = try
        inv((H .+ H') ./ 2)
    catch
        return (nothing, false)
    end
    return (Σ, true)
end

# ---------------------------------------------------------------------------
# Public API: transformed-scale Wald CI for a scalar derived quantity.
# ---------------------------------------------------------------------------

"""
    transformed_wald_ci_derived(fit::GllvmFit, derived_fn_packed::Function;
                                transform::Symbol,
                                level = 0.95, y = nothing,
                                X = nothing, Σ_phy = nothing)
        -> NamedTuple{(:lower, :upper, :estimate, :se_transformed,
                       :transform, :pd_hessian, :method)}

Transformed-scale Wald confidence interval for a scalar-valued *derived
bounded quantity* `g(θ) = derived_fn_packed(θ_packed)`.

`transform` selects the link `h` that maps the natural range to ℝ:
  - `:fisher_z` — `h = atanh`, `back = tanh`; for correlations ρ ∈ [−1, 1].
  - `:logit`    — `h = logit`, `back = logistic`; for c²/ICC/H² ∈ [0, 1].

The interval is built on the transformed scale and mapped back:
    `CI = back( h(g(θ̂)) ± z·SE_h )`,  `SE_h = sqrt(∇θ(h∘g)' Σ ∇θ(h∘g))`,
where `Σ = inv(H)` is the asymptotic covariance from the observed
information `H = ForwardDiff.hessian(nll, θ̂)` (the same Hessian
`confint` uses), and `∇θ(h∘g)` is obtained by ForwardDiff on the dense
packed-θ marginal. Because `back` maps ℝ into the open natural range, the
returned bounds are *guaranteed* to lie inside `(−1, 1)` (resp. `(0, 1)`).

The point `estimate` is `g(θ̂)` — the raw derived quantity, identical to
the public accessor (`correlation(fit)[i,j]`, `communality(fit)[t]`, …).

`derived_fn_packed` must accept a packed-parameter vector and return a
scalar; use the closure helpers in this file / confint_derived.jl:

```julia
spec = GLLVModels._derived_spec(fit)
f_ρ  = GLLVModels._make_correlation_closure(spec, 1, 2)   # ρ[1,2]
ci   = GLLVModels.transformed_wald_ci_derived(fit, f_ρ;
                                         transform = :fisher_z, y = y)

f_c2 = GLLVModels._make_communality_closure(spec, 1)      # c²[1]
ci2  = GLLVModels.transformed_wald_ci_derived(fit, f_c2;
                                         transform = :logit, y = y)
```

Or the thin wrappers `correlation_wald_ci`, `communality_wald_ci`,
`icc_wald_ci`, `phylo_signal_wald_ci`.

Returns a NamedTuple with fields:
  - `estimate::Float64`       — `g(θ̂)` (raw derived quantity)
  - `lower::Float64`          — lower CI bound (back-transformed)
  - `upper::Float64`          — upper CI bound (back-transformed)
  - `se_transformed::Float64` — delta-method SE on the transformed scale
  - `transform::Symbol`       — `:fisher_z` or `:logit`
  - `pd_hessian::Bool`        — whether the observed information was PD
  - `method::Symbol`          — `:transformed_wald` (success) or `:failed`
                                (non-PD Hessian, non-finite SE, or the
                                point estimate on the boundary where `h`
                                is undefined → bounds `NaN`)

Boundary note: if `g(θ̂)` equals a bound exactly (ρ = ±1, c² ∈ {0, 1}),
`h(g(θ̂))` is ±∞ and the interval is undefined; this returns `NaN` bounds
with `method = :failed`. Interior estimates are the generic case.
"""
function transformed_wald_ci_derived(fit::GllvmFit, derived_fn_packed::Function;
                                     transform::Symbol,
                                     level::Real = 0.95,
                                     y::Union{Nothing, AbstractMatrix} = nothing,
                                     X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                                     Σ_phy::Union{Nothing, AbstractMatrix} = nothing)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    y === nothing && throw(ArgumentError(
        "transformed_wald_ci_derived requires the data matrix `y` " *
        "(the same matrix passed to fit_gaussian_gllvm)"))

    h, back, (lo_bound, hi_bound) = _tw_link(transform)

    θ̂ = fit.pars.θ_packed
    g_hat = Float64(derived_fn_packed(θ̂))

    failed = (; estimate = g_hat, lower = NaN, upper = NaN,
              se_transformed = NaN, transform = transform,
              pd_hessian = false, method = :failed)

    # Point estimate must be finite and strictly interior for h to be defined.
    if !isfinite(g_hat) || g_hat ≤ lo_bound || g_hat ≥ hi_bound
        return failed
    end

    Σ, pd = _tw_sigma_from_hessian(fit, y, X, Σ_phy)
    (Σ === nothing || !pd) && return merge(failed, (; pd_hessian = false))

    # ∇θ(h∘g) at θ̂ via ForwardDiff. h∘g is scalar-valued.
    hg = θ -> h(derived_fn_packed(θ))
    grad = try
        ForwardDiff.gradient(hg, θ̂)
    catch
        return merge(failed, (; pd_hessian = true))
    end
    (all(isfinite, grad)) || return merge(failed, (; pd_hessian = true))

    var_h = dot(grad, Σ * grad)
    if !isfinite(var_h) || var_h < 0
        return merge(failed, (; pd_hessian = true))
    end
    se_h = sqrt(var_h)

    z = quantile(Normal(), 0.5 + level / 2)
    h_hat = h(g_hat)
    lower = back(h_hat - z * se_h)
    upper = back(h_hat + z * se_h)

    return (; estimate = g_hat, lower = lower, upper = upper,
            se_transformed = se_h, transform = transform,
            pd_hessian = true, method = :transformed_wald)
end

function _transformed_wald_ci_with_sigma(θ̂::AbstractVector, derived_fn_packed::Function,
                                         Σ::Union{Nothing, AbstractMatrix}, pd::Bool;
                                         transform::Symbol, level::Real)
    h, back, (lo_bound, hi_bound) = _tw_link(transform)
    g_hat = Float64(derived_fn_packed(θ̂))
    failed = (; estimate = g_hat, lower = NaN, upper = NaN,
              se_transformed = NaN, transform = transform,
              pd_hessian = false, method = :failed)

    if !isfinite(g_hat) || g_hat ≤ lo_bound || g_hat ≥ hi_bound
        return failed
    end
    (Σ === nothing || !pd) && return merge(failed, (; pd_hessian = false))

    hg = θ -> h(derived_fn_packed(θ))
    grad = try
        ForwardDiff.gradient(hg, θ̂)
    catch
        return merge(failed, (; pd_hessian = true))
    end
    (all(isfinite, grad)) || return merge(failed, (; pd_hessian = true))

    var_h = dot(grad, Σ * grad)
    if !isfinite(var_h) || var_h < 0
        return merge(failed, (; pd_hessian = true))
    end
    se_h = sqrt(var_h)

    z = quantile(Normal(), 0.5 + level / 2)
    h_hat = h(g_hat)
    return (; estimate = g_hat, lower = back(h_hat - z * se_h),
            upper = back(h_hat + z * se_h), se_transformed = se_h,
            transform = transform, pd_hessian = true,
            method = :transformed_wald)
end

function _phylo_signal_wald_ci_all(fit::GllvmFit;
                                   level::Real = 0.95,
                                   y::Union{Nothing, AbstractMatrix} = nothing,
                                   X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                                   Σ_phy::Union{Nothing, AbstractMatrix} = nothing)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    y === nothing && throw(ArgumentError(
        "_phylo_signal_wald_ci_all requires the data matrix `y` " *
        "(the same matrix passed to fit_gaussian_gllvm)"))
    spec = _derived_spec(fit)
    diag_Σphy = Σ_phy === nothing ? nothing : diag(Σ_phy)
    θ̂ = fit.pars.θ_packed
    h2 = phylo_signal(fit; Σ_phy = Σ_phy)
    interior = any(h -> isfinite(h) && 0.0 < h < 1.0, h2)
    Σ, pd = interior ? _tw_sigma_from_hessian(fit, y, X, Σ_phy) : (nothing, false)
    out = Vector{NamedTuple}(undef, length(h2))
    for t in eachindex(h2)
        f = _make_phylo_signal_closure(spec, t; diag_Σphy = diag_Σphy)
        out[t] = _transformed_wald_ci_with_sigma(θ̂, f, Σ, pd;
                                                 transform = :logit, level = level)
    end
    return out
end

# ---------------------------------------------------------------------------
# Thin convenience wrappers for the four built-in bounded quantities.
# ---------------------------------------------------------------------------

"""
    correlation_wald_ci(fit, i, j; level=0.95, y, X=nothing, Σ_phy=nothing)

Fisher-z transformed-Wald CI for the cross-trait correlation `ρ[i, j]`.
Bounds are guaranteed to lie in `[−1, 1]`. See
[`transformed_wald_ci_derived`](@ref).
"""
function correlation_wald_ci(fit::GllvmFit, i::Integer, j::Integer;
                             level::Real = 0.95,
                             y::Union{Nothing, AbstractMatrix} = nothing,
                             X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                             Σ_phy::Union{Nothing, AbstractMatrix} = nothing)
    spec = _derived_spec(fit)
    f = _make_correlation_closure(spec, i, j)
    return transformed_wald_ci_derived(fit, f; transform = :fisher_z,
                                       level = level, y = y, X = X, Σ_phy = Σ_phy)
end

"""
    communality_wald_ci(fit, t; level=0.95, y, X=nothing, Σ_phy=nothing)

Logit transformed-Wald CI for the per-trait communality `c²[t]`. Bounds
are guaranteed to lie in `[0, 1]`. See [`transformed_wald_ci_derived`](@ref).
"""
function communality_wald_ci(fit::GllvmFit, t::Integer;
                             level::Real = 0.95,
                             y::Union{Nothing, AbstractMatrix} = nothing,
                             X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                             Σ_phy::Union{Nothing, AbstractMatrix} = nothing)
    spec = _derived_spec(fit)
    f = _make_communality_closure(spec, t)
    return transformed_wald_ci_derived(fit, f; transform = :logit,
                                       level = level, y = y, X = X, Σ_phy = Σ_phy)
end

"""
    icc_wald_ci(fit, derived_fn_packed; level=0.95, y, X=nothing, Σ_phy=nothing)

Logit transformed-Wald CI for an intraclass-correlation-style proportion
in `[0, 1]` supplied as a packed-θ closure (e.g. one of the
`proportions(...)` components written in packed form). Identical to
calling [`transformed_wald_ci_derived`](@ref) with `transform = :logit`;
provided for naming symmetry.
"""
function icc_wald_ci(fit::GllvmFit, derived_fn_packed::Function;
                     level::Real = 0.95,
                     y::Union{Nothing, AbstractMatrix} = nothing,
                     X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                     Σ_phy::Union{Nothing, AbstractMatrix} = nothing)
    return transformed_wald_ci_derived(fit, derived_fn_packed; transform = :logit,
                                       level = level, y = y, X = X, Σ_phy = Σ_phy)
end

"""
    phylo_signal_wald_ci(fit, t; level=0.95, y, X=nothing, Σ_phy=nothing)

Logit transformed-Wald CI for the per-trait phylogenetic signal `H²[t]`.
Bounds are guaranteed to lie in `[0, 1]`. `Σ_phy` enters only through its
diagonal (standardised convention → unit diagonal when omitted). See
[`transformed_wald_ci_derived`](@ref).
"""
function phylo_signal_wald_ci(fit::GllvmFit, t::Integer;
                              level::Real = 0.95,
                              y::Union{Nothing, AbstractMatrix} = nothing,
                              X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                              Σ_phy::Union{Nothing, AbstractMatrix} = nothing)
    spec = _derived_spec(fit)
    diag_Σphy = Σ_phy === nothing ? nothing : diag(Σ_phy)
    f = _make_phylo_signal_closure(spec, t; diag_Σphy = diag_Σphy)
    return transformed_wald_ci_derived(fit, f; transform = :logit,
                                       level = level, y = y, X = X, Σ_phy = Σ_phy)
end

# ===========================================================================
# Standardised-loading (rho) transformed-Wald CI — the `wald_asym` route of
# R's `loading_ci()` (.unlazy/core070-aghq/oracle-source/readback/R/loading-ci.R).
#
# rho[t, k] = Λ[t, k] / sqrt(Σ_y_site[t, t]) standardises the reduced-rank
# loading entry into a correlation-like quantity on (−1, 1): the share of
# trait t's per-site SD carried by latent axis k. The Fisher-z transform
# (this repo's established [−1,1] convention, src/confint_derived_wald.jl
# header) gives bounds guaranteed to stay in (−1, 1), matching R's
# `method = "wald_asym"` (asymmetric Wald via Fisher-z on the standardised
# scale — see `.lambda_ci_asym()` in the R oracle).
#
# `component` selects the loading tier: `:B` (GLLVModels.jl's "unit"/shared block,
# the R default `level = "unit"`) or `:W` (the "unit_obs"/within block).
# Standardisation always divides by the FULL per-site total variance
# `Σ_y_site[t,t]` (both blocks combined, plus σ²_eps and any diagonal
# companions) — matching R's `.standardize_loadings_by_total_variance()`,
# which standardises by model-implied total variance regardless of which
# loading tier is being reported.
# ===========================================================================

function _standardized_loading_packed(θ::AbstractVector, spec::NamedTuple,
                                      t::Integer, k::Integer; component::Symbol = :B)
    u = _derived_unpack(θ, spec)
    Λ = component === :B ? u.Λ_B :
        component === :W ? u.Λ_W :
        throw(ArgumentError("component must be :B or :W; got $(component)"))
    Λ === nothing && throw(ArgumentError(
        "fit has no Λ_$(component) block; standardized loading is undefined"))
    Σ = _sigma_y_site_from_unpacked(u, spec)
    return Λ[t, k] / sqrt(Σ[t, t])
end

function _make_standardized_loading_closure(spec::NamedTuple, t::Integer, k::Integer;
                                            component::Symbol = :B)
    return θ -> _standardized_loading_packed(θ, spec, t, k; component = component)
end

"""
    standardized_loading_wald_ci(fit, t, k; level=0.95, y, X=nothing,
                                 Σ_phy=nothing, component=:B)
        -> NamedTuple

Fisher-z transformed-Wald CI for the standardised loading
`rho[t, k] = Λ[t, k] / sqrt(Σ_y_site[t, t])` (R's `loading_ci(method =
"wald_asym", loading_scale = "standardized")`, parameter name
`rho[trait,axis]`). Bounds are guaranteed to lie in `(−1, 1)`.
`component = :B` (default) is R's `level = "unit"` (the shared/between
loading tier); `component = :W` is `level = "unit_obs"` (the within tier,
requires `fit.pars.Λ_W !== nothing`).

Delta-method SE and Hessian convention are identical to
[`transformed_wald_ci_derived`](@ref) — one ForwardDiff Hessian of the
packed NLL, reused for every entry in a `loading_ci` table.
"""
function standardized_loading_wald_ci(fit::GllvmFit, t::Integer, k::Integer;
                                      level::Real = 0.95,
                                      y::Union{Nothing, AbstractMatrix} = nothing,
                                      X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                                      Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                                      component::Symbol = :B)
    spec = _derived_spec(fit)
    f = _make_standardized_loading_closure(spec, t, k; component = component)
    return transformed_wald_ci_derived(fit, f; transform = :fisher_z,
                                       level = level, y = y, X = X, Σ_phy = Σ_phy)
end

"""
    raw_loading_wald_ci(fit, t, k; level=0.95, y, X=nothing, Σ_phy=nothing,
                        component=:B) -> NamedTuple

Symmetric (identity-link) transformed-Wald CI for the RAW loading entry
`Λ[t, k]` on its native (unbounded) scale, built with the same one-Hessian
machinery as [`standardized_loading_wald_ci`](@ref). This is the `method =
"wald"` route of R's `loading_ci()` on GLLVModels.jl's dense reduced-rank fit.
For `k > t` on the lower-triangular reduced-rank packing convention
(`src/packing.jl`), the entry is structurally pinned at `0` — `estimate ==
0`, `se == 0`, `lower == upper == 0`, mirroring `pinned = TRUE` rows in R's
output.
"""
function raw_loading_wald_ci(fit::GllvmFit, t::Integer, k::Integer;
                             level::Real = 0.95,
                             y::Union{Nothing, AbstractMatrix} = nothing,
                             X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                             Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                             component::Symbol = :B)
    if k > t
        return (; estimate = 0.0, lower = 0.0, upper = 0.0,
                se_transformed = 0.0, transform = :identity,
                pd_hessian = true, method = :pinned)
    end
    spec = _derived_spec(fit)
    f = θ -> begin
        u = _derived_unpack(θ, spec)
        Λ = component === :B ? u.Λ_B :
            component === :W ? u.Λ_W :
            throw(ArgumentError("component must be :B or :W; got $(component)"))
        Λ === nothing && throw(ArgumentError(
            "fit has no Λ_$(component) block; raw loading is undefined"))
        Λ[t, k]
    end
    return transformed_wald_ci_derived(fit, f; transform = :identity,
                                       level = level, y = y, X = X, Σ_phy = Σ_phy)
end

"""
    loading_ci(fit::GllvmFit, y; level=:unit, method=:wald, conf_level=0.95,
              loading_scale=nothing, X=nothing, Σ_phy=nothing)
        -> Vector{NamedTuple}

Per-entry confidence intervals on the reduced-rank loading matrix, one row
per `(trait, axis)` — the Julia analogue of R's `loading_ci()`.

  - `level`: `:unit` (default, R's shared/between tier) or `:unit_obs` (the
    within tier; requires the fit to carry a `Λ_W` block).
  - `method`: `:wald` (default) — symmetric Wald on `loading_scale`;
    `:wald_asym` — Fisher-z asymmetric Wald, only defined for
    `loading_scale = :standardized`; `:profile` — profile-likelihood via
    [`loading_profile_exploratory`](@ref), only defined for `loading_scale = :raw`
    (mirrors R's refusals for the same scale/method combinations).
  - `loading_scale`: `nothing` (default) resolves to `:standardized` for
    `:wald_asym` and `:raw` otherwise, matching R.

Every row carries `trait`, `axis`, `estimate`, `se`, `lower`, `upper`,
`method`, `loading_scale`, and `pinned` (`true` for the structurally-zero
upper-triangular entries of the lower-triangular reduced-rank packing
convention — GLLVModels.jl's built-in identifiability device; there is no
separate `lambda_constraint`/confirmatory-fit concept to gate on here, so
unlike R this function does not refuse exploratory fits).

**Interval calibration**: as in R, treat these as exploratory intervals —
their empirical coverage on this repo's dense reduced-rank path is not
separately certified; the point estimates (`estimate`) are the supported
claim.
"""
function loading_ci(fit::GllvmFit, y::AbstractMatrix;
                    level::Symbol = :unit,
                    method::Symbol = :wald,
                    conf_level::Real = 0.95,
                    loading_scale::Union{Nothing, Symbol} = nothing,
                    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                    Σ_phy::Union{Nothing, AbstractMatrix} = nothing)
    level ∈ (:unit, :unit_obs) || throw(ArgumentError(
        "level must be :unit or :unit_obs; got $(level)"))
    method ∈ (:wald, :wald_asym, :profile) || throw(ArgumentError(
        "method must be :wald, :wald_asym, or :profile; got $(method)"))
    scale = loading_scale === nothing ?
        (method === :wald_asym ? :standardized : :raw) : loading_scale
    scale ∈ (:raw, :standardized) || throw(ArgumentError(
        "loading_scale must be :raw or :standardized; got $(scale)"))
    method === :wald_asym && scale !== :standardized && throw(ArgumentError(
        "method = :wald_asym is defined on the standardised-loading scale; " *
        "pass loading_scale = :standardized, or use method = :wald for raw " *
        "Λ intervals"))
    method === :profile && scale !== :raw && throw(ArgumentError(
        "standardised profile intervals are not implemented; use " *
        "loading_scale = :raw with method = :profile, or a standardised " *
        "Wald method"))

    component = level === :unit ? :B : :W
    Λ = component === :B ? fit.pars.Λ : fit.pars.Λ_W
    Λ === nothing && throw(ArgumentError(
        "fit has no Λ_$(component) block at level = $(level)"))
    p, d = size(Λ)

    out = Vector{NamedTuple}(undef, p * d)
    row = 0
    for k in 1:d, t in 1:p
        row += 1
        if method === :wald
            r = scale === :standardized ?
                standardized_loading_wald_ci(fit, t, k; level = conf_level, y = y,
                                             X = X, Σ_phy = Σ_phy, component = component) :
                raw_loading_wald_ci(fit, t, k; level = conf_level, y = y,
                                    X = X, Σ_phy = Σ_phy, component = component)
        elseif method === :wald_asym
            r = standardized_loading_wald_ci(fit, t, k; level = conf_level, y = y,
                                             X = X, Σ_phy = Σ_phy, component = component)
        else # :profile
            r = loading_profile_exploratory(fit, t, k; level = conf_level, y = y, X = X,
                                            Σ_phy = Σ_phy, component = component)
        end
        out[row] = (; trait = t, axis = k, estimate = r.estimate,
                    se = get(r, :se_transformed, NaN),
                    lower = r.lower, upper = r.upper,
                    method = method, loading_scale = scale,
                    pinned = (k > t))
    end
    return out
end

# ===========================================================================
# slope_sd_ci — Wald CI on random-slope standard deviations for
# GaussianRandomSlopeFit (src/fit_random_effects.jl). Mirrors R's SLICE-1
# `slope_sd_ci()` route (log-SD Wald for the univariate case, `.unlazy/
# core070-aghq/oracle-source/readback/R/slope-sd-ci.R`), generalised via the
# log-scale delta method for the correlated q>1 case: `Var(b_k) = Σ_b[k,k]`
# is a nonlinear function of the log-Cholesky packing (`_unpack_chol_cov`),
# so `log(Σ_b[k,k])` is delta-method'd and exponentiated back — the repo's
# unconstrained-internal-scale convention for SDs (guarantees a positive
# bound, unlike a raw-scale symmetric CI).
# ===========================================================================

# Inverse of _unpack_chol_cov (src/fit_random_effects.jl): pack a q×q SPD
# matrix into the same log-Cholesky vector layout (diag = log, column-major
# lower-tri off-diag = raw).
function _pack_chol_cov(Σ_b::AbstractMatrix, q::Integer)
    L = cholesky(Symmetric(Matrix{Float64}(Σ_b))).L
    theta = Vector{Float64}(undef, _chol_cov_npar(q))
    idx = 1
    @inbounds for j in 1:q
        theta[idx] = log(L[j, j]); idx += 1
        for i in (j + 1):q
            theta[idx] = L[i, j]; idx += 1
        end
    end
    return theta
end

"""
    slope_sd_ci(fit::GaussianRandomSlopeFit, y, grouping, Z; level=0.95)
        -> Vector{NamedTuple}

Log-scale transformed-Wald CI on each random-slope standard deviation
`sd_b[k] = sqrt(Σ_b[k, k])`, `k = 1:fit.q`, of a fitted
[`GaussianRandomSlopeFit`](@ref) (`fit_gaussian_random_slope`). Column 1 of
`Z` conventionally carries the random intercept, so `sd_b[1]` is the
random-intercept SD; further columns are random-slope SDs.

Bounds are guaranteed positive via the `exp(log θ̂ ± z·SE_log)`
transformed-Wald convention this repo uses for every other SD parameter
(σ_eps in `confint.jl`, σ_B/σ_W in `confint.jl`). The observed information
is the ForwardDiff Hessian of the packed grouped-slope NLL reconstructed
from `y`, `grouping`, `Z` at `fit`'s converged
`[vec(Λ); log σ_eps; logCholesky(Σ_b)]` — the same objective
`fit_gaussian_random_slope` optimised.

`y`, `grouping`, `Z` must be the same arrays passed to
`fit_gaussian_random_slope` (this struct does not retain them).

Returns a `NamedTuple` per `k` with fields `estimate` (`sd_b[k]`),
`lower`, `upper`, `se_transformed` (SE on `log(Σ_b[k,k])`), `transform =
:log`, `pd_hessian`, `method`.
"""
function slope_sd_ci(fit::GaussianRandomSlopeFit, y::AbstractMatrix,
                     grouping::AbstractVector, Z::AbstractMatrix;
                     level::Real = 0.95)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    p, K = size(fit.Λ)
    q = fit.q
    yf = Matrix{Float64}(y); Zf = Matrix{Float64}(Z)
    codes, _ = _code_grouping(grouping)
    L = maximum(codes)
    group_idx = [findall(==(g), codes) for g in 1:L]
    rr = rr_theta_len(p, K)
    nc = _chol_cov_npar(q)

    θ̂ = vcat(pack_lambda(fit.Λ), log(fit.σ_eps), _pack_chol_cov(fit.Σ_b, q))

    nll = θ -> begin
        Λ = unpack_lambda(θ[1:rr], p, K)
        σe = exp(θ[rr + 1])
        Σ_b, _ = _unpack_chol_cov(θ[(rr + 2):(rr + 1 + nc)], q)
        -_grouped_slope_loglik(yf, group_idx, Zf, Λ, σe, Σ_b)
    end

    H = try
        ForwardDiff.hessian(nll, θ̂)
    catch
        nothing
    end
    Σcov, pd = if H !== nothing && all(isfinite, H)
        try
            (inv((H .+ H') ./ 2), true)
        catch
            (nothing, false)
        end
    else
        (nothing, false)
    end

    out = Vector{NamedTuple}(undef, q)
    for k in 1:q
        g = θ -> begin
            Σ_b, _ = _unpack_chol_cov(θ[(rr + 2):(rr + 1 + nc)], q)
            sqrt(Σ_b[k, k])
        end
        out[k] = _transformed_wald_ci_with_sigma(θ̂, g, Σcov, pd;
                                                  transform = :log, level = level)
    end
    return out
end

"""
    standard_errors(fit::GllvmFit, y; X=nothing, Σ_phy=nothing, level=0.95)
        -> NamedTuple

Julia analogue of R gllvmTMB's `standard_errors()`. R's version lazily defers
TMB's `sdreport()` to a later call when a fit was made with `control =
gllvmTMBcontrol(se = FALSE)` — a "fit fast now, get SEs later" door.
GLLVModels.jl's `GllvmFit` has no such deferred-SE control: the observed-
information Hessian is always available on demand via [`confint`](@ref).

`standard_errors` is therefore a thin, always-eager wrapper around
`confint(fit, y; level=level, X=X, Σ_phy=Σ_phy)`, returning only the
`term`/`estimate`/`se`/`pd_hessian` fields (the CI-bound-free subset R's
downstream consumers — `summary()`, `getREsd()`, `confint(method =
"wald")` — actually read off `sd_report`). Since GLLVModels.jl never skips the
computation, `fit` is returned unchanged (it is immutable) and the
`NamedTuple` result stands in for R's "populate `sd_report` and return the
fit" side effect.
"""
function standard_errors(fit::GllvmFit, y::AbstractMatrix;
                         X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                         Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                         level::Real = 0.95)
    res = confint(fit; level = level, y = y, X = X, Σ_phy = Σ_phy)
    return (term = res.term, estimate = res.estimate, se = res.se,
            pd_hessian = res.pd_hessian)
end

# ---------------------------------------------------------------------------
# APPEND (core070 E-cluster, PERF+SE-machinery): bootstrap_Sigma.
#
# Julia analogue of R's `bootstrap_Sigma()` (`.unlazy/core070-aghq/
# oracle-source/readback/R/bootstrap-sigma.R`): parametric bootstrap CIs for
# the entries of the fitted trait covariance matrix. A thin driver over the
# EXISTING derived-quantity bootstrap (`bootstrap_ci_derived`,
# src/confint_derived.jl) plus a table assembler — no new bootstrap
# machinery here, matching the "thin driver + table" scope in the plan for
# this slice.
# ---------------------------------------------------------------------------

"""
    bootstrap_Sigma(fit::GllvmFit; level=:unit, n_boot=500, conf=0.95,
                    seed=0, y=nothing, n_sites=nothing, X=nothing,
                    Σ_phy=nothing, verbose=false)
        -> NamedTuple

Parametric bootstrap percentile CIs for every entry of `sigma_y_site(fit)`
(the fitted site-level trait covariance `Σ_y = Λ Λ' + diag(d_total)`), via
`bootstrap_ci_derived` (src/confint_derived.jl) applied entrywise.

`level = :unit` is the only value accepted: GLLVModels.jl computes one
site-level `Σ_y` per fit (`sigma_y_site`), not R's separate unit / unit_obs
/ phy tiers, so `level` is kept for interface parity with R's
`bootstrap_Sigma(level = ...)` and validated rather than silently ignored.

Returns a NamedTuple table over the upper-triangle entries (`i ≤ j`, 1-based)
of `Σ_y`:
  - `i::Vector{Int}`, `j::Vector{Int}`
  - `estimate::Vector{Float64}` — `Σ_y[i,j]` at θ̂
  - `lower::Vector{Float64}`, `upper::Vector{Float64}` — percentile CI at `conf`
  - `n_converged::Vector{Int}`, `n_valid::Vector{Int}`

`y`, `n_sites`, `X`, `Σ_phy` are forwarded to `bootstrap_ci_derived` exactly
as documented there (pass the same values used for `fit_gaussian_gllvm`).

Cost: `p(p+1)/2 × n_boot` refits (one full `bootstrap_ci_derived` bootstrap
run per matrix entry) — quadratic in `p`. Use the full table for small trait
matrices. For larger `p`, lower `n_boot` or call `bootstrap_ci_derived`
directly for only the entries you need.

R's `bootstrap_Sigma()` also bootstraps `R` (correlation), `communality`, `ICC`,
and `cross_corr` in the SAME call, over multiple covariance tiers. This
driver covers only the `Sigma` entries at the single `:unit` tier GLLVModels.jl
has. The other summaries are already independently reachable via
`bootstrap_ci_derived(fit, fb -> communality(fb)[t]; ...)` /
`bootstrap_ci_derived(fit, fb -> correlation(fb)[i,j]; ...)`; a unified
multi-summary table matching R's return shape is not yet available.
"""
function bootstrap_Sigma(fit::GllvmFit;
                         level::Symbol = :unit,
                         n_boot::Integer = 500,
                         conf::Real = 0.95,
                         seed::Integer = 0,
                         y::Union{Nothing, AbstractMatrix} = nothing,
                         n_sites::Union{Nothing, Integer} = nothing,
                         X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                         Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                         verbose::Bool = false)
    level === :unit || throw(ArgumentError(
        "bootstrap_Sigma currently supports level = :unit only " *
        "(GLLVModels.jl computes one site-level Σ_y tier); got :$level"))
    p = fit.model.p

    ii = Int[]; jj = Int[]
    est = Float64[]; lo = Float64[]; hi = Float64[]
    nconv = Int[]; nvalid = Int[]

    for j in 1:p, i in 1:j
        r = bootstrap_ci_derived(fit, fb -> sigma_y_site(fb)[i, j];
                                 n_boot = n_boot, level = conf, seed = seed,
                                 y = y, n_sites = n_sites, X = X, Σ_phy = Σ_phy,
                                 verbose = verbose)
        push!(ii, i); push!(jj, j)
        push!(est, r.estimate); push!(lo, r.lower); push!(hi, r.upper)
        push!(nconv, r.n_converged); push!(nvalid, r.n_valid)
    end

    return (i = ii, j = jj, estimate = est, lower = lo, upper = hi,
            n_converged = nconv, n_valid = nvalid)
end
