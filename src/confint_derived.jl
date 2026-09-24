# Profile-likelihood and parametric-bootstrap confidence intervals for
# *derived quantities* of a fitted Gaussian GLLVM.
#
# The Wald CI machinery in src/confint.jl, the profile CI in
# src/confint_profile.jl, and the bootstrap CI in src/confint_bootstrap.jl
# all operate on individual *packed parameters*. Ecologists care about
# *derived quantities* — entries of Σ_y, communalities c² = (ΛΛ')_tt / Σ_tt,
# cross-trait correlations, ICCs, phylogenetic signal H² — which are
# *nonlinear* functions of the parameters. None of the per-parameter CIs
# transfer directly.
#
# Two complementary CI methods here:
#
# Bootstrap: replay the parametric bootstrap from src/confint_bootstrap.jl,
# compute the scalar derived quantity on each successfully refit replicate,
# and take percentiles. Single-line wrapper around the existing bootstrap
# infrastructure.
#
# Profile (constrained refit): for a candidate c we hold the derived
# quantity fixed at c via a quadratic penalty
#     NLL_pen(θ) = NLL(θ) + 0.5 · w · (derived_fn(θ) − c)²
# and re-optimise over θ. The unpenalised NLL at the constrained
# minimum θ̂(c) gives the profile log-likelihood ℓ_p(c). The deviance
# D(c) = 2(ℓ̂ − ℓ_p(c)) is ~ χ²₁ under the null derived_fn(θ) = c, so
# the 100(1−α)% profile CI is {c : D(c) ≤ qchisq(1−α, 1)}. We then
# bracket-then-bisect, mirroring src/confint_profile.jl's strategy.

using Distributions: Chisq, quantile

# Linear-interpolation percentile (matches Statistics.quantile default,
# i.e. R type 7). Self-contained so this file does not need Statistics.
function _derived_percentile(v::AbstractVector{<:Real}, p::Real)
    0 ≤ p ≤ 1 || throw(ArgumentError("p must be in [0, 1]; got $p"))
    n = length(v)
    n ≥ 1 || throw(ArgumentError("v must be non-empty"))
    s = sort(v)
    if n == 1
        return float(s[1])
    end
    h = (n - 1) * p
    lo = floor(Int, h)
    hi = ceil(Int, h)
    if lo == hi
        return float(s[lo + 1])
    end
    frac = h - lo
    return float(s[lo + 1]) * (1 - frac) + float(s[hi + 1]) * frac
end

# ---------------------------------------------------------------------------
# Spec helpers (mirror sister confint files; kept local so this slice
# does not touch any existing file).
# ---------------------------------------------------------------------------

function _derived_spec(fit::GllvmFit)
    model = fit.model
    q_full = fit.pars.β === nothing ? 0 : length(fit.pars.β)
    q = count(!,_pars_fixed_mask(fit.pars,:β_fixed,q_full))
    return (q = q, p = model.p, K_B = model.K, K_W = model.K_W,
            has_diag = model.has_diag, K_phy = model.K_phy,
            has_phy_unique = model.has_phy_unique)
end

# Unpack a packed-θ vector into a NamedTuple with the user-facing matrices.
# Mirrors the *legacy* layout returned by fit.pars.θ_packed:
#   [β; log_σ_eps;
#    log_σ_B[p]; log_σ_W[p]      (if has_diag)
#    θ_rr_B (pack_lambda Λ_B)
#    θ_rr_W (pack_lambda Λ_W)     (if K_W > 0)
#    σ_phy[p] (natural, signed)  (if has_phy_unique)
#    θ_rr_phy (pack_lambda Λ_phy)(if K_phy > 0)]
#
# Returned tuple uses raw-scale (positive) variance / SD components, like
# fit.pars.*. AD-friendly: eltype(θ) is preserved so ForwardDiff Duals
# flow through unchanged.
function _derived_unpack(θ::AbstractVector, spec::NamedTuple)
    q        = spec.q
    p        = spec.p
    K_B      = spec.K_B
    K_W      = spec.K_W
    has_diag = spec.has_diag
    K_phy          = hasproperty(spec, :K_phy)          ? spec.K_phy          : 0
    has_phy_unique = hasproperty(spec, :has_phy_unique) ? spec.has_phy_unique : false

    rr_B  = rr_theta_len(p, K_B)
    rr_W  = K_W > 0 ? rr_theta_len(p, K_W) : 0
    rr_phy = K_phy > 0 ? rr_theta_len(p, K_phy) : 0

    cursor = 0
    β = if q > 0
        b = θ[(cursor + 1):(cursor + q)]
        cursor += q
        b
    else
        nothing
    end

    σ_eps = exp(θ[cursor + 1])
    cursor += 1

    σ²_B = nothing
    σ²_W = nothing
    if has_diag
        log_σ_B = θ[(cursor + 1):(cursor + p)]
        cursor += p
        log_σ_W = θ[(cursor + 1):(cursor + p)]
        cursor += p
        σ²_B = exp.(2 .* log_σ_B)
        σ²_W = exp.(2 .* log_σ_W)
    end

    θ_rr_B = θ[(cursor + 1):(cursor + rr_B)]
    cursor += rr_B
    Λ_B = unpack_lambda(θ_rr_B, p, K_B)

    Λ_W = nothing
    if K_W > 0
        θ_rr_W = θ[(cursor + 1):(cursor + rr_W)]
        cursor += rr_W
        Λ_W = unpack_lambda(θ_rr_W, p, K_W)
    end

    σ_phy = nothing
    if has_phy_unique
        # σ_phy uses an identity (signed) link and is packed on the natural scale
        # by the Gaussian phylo fitter. It is a loading-like quantity (hcat'd with
        # Λ_phy and squared in H²), so exponentiating it here over-transforms the
        # phylo-signal numerator and destroys the sign. σ_eps/σ²_B/σ²_W above are
        # log-packed; σ_phy is the natural-scale exception in θ_packed.
        σ_phy = θ[(cursor + 1):(cursor + p)]
        cursor += p
    end

    Λ_phy = nothing
    if K_phy > 0
        θ_rr_phy = θ[(cursor + 1):(cursor + rr_phy)]
        cursor += rr_phy
        Λ_phy = unpack_lambda(θ_rr_phy, p, K_phy)
    end

    return (β = β, σ_eps = σ_eps,
            σ²_B = σ²_B, σ²_W = σ²_W,
            Λ_B = Λ_B, Λ_W = Λ_W,
            σ_phy = σ_phy, Λ_phy = Λ_phy)
end

# ---------------------------------------------------------------------------
# Σ_y_site = Λ_B Λ_B' + diag(d_total) — the per-site (within-species)
# trait covariance. The phylogenetic block (rank-1 across species) is not
# part of the per-site covariance; it contributes only to the species-level
# shared variance. Following the bootstrap-sigma R convention.
# ---------------------------------------------------------------------------
function _sigma_y_site_from_unpacked(u::NamedTuple, spec::NamedTuple)
    p   = spec.p
    K_W = spec.K_W
    has_diag = spec.has_diag
    σ² = u.σ_eps^2
    Λ_B = u.Λ_B
    A = Λ_B * Λ_B'
    @inbounds for t in 1:p
        v = σ²
        if K_W > 0 && u.Λ_W !== nothing
            for k in 1:size(u.Λ_W, 2)
                v += u.Λ_W[t, k]^2
            end
        end
        if has_diag && u.σ²_B !== nothing
            v += u.σ²_B[t]
        end
        if has_diag && u.σ²_W !== nothing
            v += u.σ²_W[t]
        end
        A[t, t] += v
    end
    return A
end

"""
    sigma_y_site(fit::GllvmFit) -> Matrix

The per-site (within-species) trait covariance
`Σ_y_site = Λ_B Λ_B' + diag(d_total)` where
`d_total[t] = (Λ_W Λ_W')[t,t] + σ²_B[t] + σ²_W[t] + σ²_eps`. For J1,
`Λ_W = nothing`, `σ²_B = σ²_W = 0`, so the diagonal collapses to `σ²_eps`.

The phylogenetic block is *not* included — for J3, the phylo
contribution is rank-1 across species and is separated out for
biological interpretation. Use `phylo_signal` to recover the phy
component, and `correlation` for the per-site cross-trait correlations.
"""
function sigma_y_site(fit::GllvmFit)
    spec = _derived_spec(fit)
    u = (β = fit.pars.β, σ_eps = fit.pars.σ_eps,
         σ²_B = fit.pars.σ²_B, σ²_W = fit.pars.σ²_W,
         Λ_B = fit.pars.Λ, Λ_W = fit.pars.Λ_W,
         σ_phy = fit.pars.σ_phy, Λ_phy = fit.pars.Λ_phy)
    A = _sigma_y_site_from_unpacked(u, spec)
    return (A + A') ./ 2
end

"""
    communality(fit::GllvmFit) -> Vector

Per-trait communality `c²[t] = (Λ_B Λ_B')[t, t] / Σ_y_site[t, t]`. This
is the fraction of the per-site trait variance explained by the shared
latent factors. Values are in [0, 1].
"""
function communality(fit::GllvmFit)
    spec = _derived_spec(fit)
    Λ_B = fit.pars.Λ
    ΛΛt = Λ_B * Λ_B'
    Σ = sigma_y_site(fit)
    return [ΛΛt[t, t] / Σ[t, t] for t in 1:spec.p]
end

"""
    proportions(fit::GllvmFit; component::Symbol = :shared) -> Vector

Per-trait variance decomposition. Each entry is in [0, 1]; the
`:shared`, `:unique_W`, `:unique_B`, and `:residual` shares sum to 1
(when has_diag and W tier are off, only `:shared` and `:residual` are
non-zero).

`component` can be:
  - `:shared`    — `(Λ_B Λ_B')[t,t] / Σ_y_site[t,t]`   (== communality)
  - `:unique_W`  — `(Λ_W Λ_W')[t,t] / Σ_y_site[t,t]`
  - `:unique_B`  — `σ²_B[t] / Σ_y_site[t,t]`            (J2-A-WD path)
  - `:unique_Wd` — `σ²_W[t] / Σ_y_site[t,t]`            (J2-A-WD path)
  - `:residual`  — `σ²_eps / Σ_y_site[t,t]`
"""
function proportions(fit::GllvmFit; component::Symbol = :shared)
    spec = _derived_spec(fit)
    p   = spec.p
    K_W = spec.K_W
    has_diag = spec.has_diag
    Σ = sigma_y_site(fit)

    if component === :shared
        Λ_B = fit.pars.Λ
        ΛΛt = Λ_B * Λ_B'
        return [ΛΛt[t, t] / Σ[t, t] for t in 1:p]
    elseif component === :unique_W
        if K_W == 0 || fit.pars.Λ_W === nothing
            return zeros(Float64, p)
        end
        Λ_W = fit.pars.Λ_W
        ΛWWt = Λ_W * Λ_W'
        return [ΛWWt[t, t] / Σ[t, t] for t in 1:p]
    elseif component === :unique_B
        if !has_diag || fit.pars.σ²_B === nothing
            return zeros(Float64, p)
        end
        σ²_B = fit.pars.σ²_B
        return [σ²_B[t] / Σ[t, t] for t in 1:p]
    elseif component === :unique_Wd
        if !has_diag || fit.pars.σ²_W === nothing
            return zeros(Float64, p)
        end
        σ²_W = fit.pars.σ²_W
        return [σ²_W[t] / Σ[t, t] for t in 1:p]
    elseif component === :residual
        σ² = fit.pars.σ_eps^2
        return [σ² / Σ[t, t] for t in 1:p]
    else
        throw(ArgumentError(
            "component must be one of :shared, :unique_W, :unique_B, " *
            ":unique_Wd, :residual; got $(component)"))
    end
end

"""
    correlation(fit::GllvmFit) -> Matrix

Cross-trait correlation derived from `Σ_y_site`:
`ρ[i, j] = Σ_y_site[i, j] / sqrt(Σ_y_site[i, i] · Σ_y_site[j, j])`.

Diagonal entries are exactly 1.0. The off-diagonals are the *site-level*
correlations driven by the shared loadings Λ_B.
"""
function correlation(fit::GllvmFit)
    Σ = sigma_y_site(fit)
    p = size(Σ, 1)
    R = similar(Σ, Float64)
    @inbounds for j in 1:p, i in 1:p
        denom = sqrt(Σ[i, i] * Σ[j, j])
        R[i, j] = Σ[i, j] / denom
    end
    return R
end

"""
    phylo_signal(fit::GllvmFit; Σ_phy = nothing) -> Vector

Per-trait phylogenetic signal
`H²[t] = σ²_phy[t] / (σ²_phy[t] + Σ_y_site[t, t])`, where
`σ²_phy[t] = (Λ_phy_aug Λ_phy_aug')[t, t] · Σ_phy[t, t]` and
`Λ_phy_aug = hcat(Λ_phy, σ_phy)` (each piece is included only when its
flag is on). This is the R oracle's convention
(`profile-derived.R:145-156`: `H² = σ²_phy / (σ²_phy + σ²_non)`) — the
denominator INCLUDES the phylogenetic variance, not just the per-site
(non-phylogenetic) covariance. Returns a vector of length `p` with
entries in `[0, 1]`; all entries are `NaN` when the fit has no
phylogenetic block (`K_phy == 0` and `has_phy_unique == false`).

`Σ_phy` defaults to the identity matrix (standardised convention, diag
== 1 per trait), so the diagonal entries reduce to
`H²[t] = (Λ_phy_aug Λ_phy_aug')[t, t] / ((Λ_phy_aug Λ_phy_aug')[t, t] +
Σ_y_site[t, t])`. Supply the fitted phylogenetic VCV explicitly when the
diagonal is not unit.
"""
function phylo_signal(fit::GllvmFit; Σ_phy::Union{Nothing, AbstractMatrix} = nothing)
    spec = _derived_spec(fit)
    p = spec.p
    has_phy = (spec.K_phy > 0) || spec.has_phy_unique
    if !has_phy
        return fill(NaN, p)
    end
    Λ_phy_aug = if fit.pars.Λ_phy !== nothing && fit.pars.σ_phy !== nothing
        hcat(fit.pars.Λ_phy, reshape(collect(Float64, fit.pars.σ_phy), p, 1))
    elseif fit.pars.Λ_phy !== nothing
        fit.pars.Λ_phy
    else
        reshape(collect(Float64, fit.pars.σ_phy), p, 1)
    end
    Σ = sigma_y_site(fit)
    diag_Σphy = Σ_phy === nothing ? ones(Float64, p) : diag(Σ_phy)
    ΛΛt = Λ_phy_aug * Λ_phy_aug'
    return [begin
        σ²phy = ΛΛt[t, t] * diag_Σphy[t]
        σ²phy / (σ²phy + Σ[t, t])
    end for t in 1:p]
end

# ---------------------------------------------------------------------------
# Packed-θ versions of the derived quantities. Used by the profile CI
# inner loop. AD-friendly: eltype propagates through unpack_lambda, exp,
# division.
# ---------------------------------------------------------------------------

function _sigma_y_site_packed(θ::AbstractVector, spec::NamedTuple)
    u = _derived_unpack(θ, spec)
    return _sigma_y_site_from_unpacked(u, spec)
end

function _communality_packed(θ::AbstractVector, spec::NamedTuple, t::Integer)
    u = _derived_unpack(θ, spec)
    Λ_B = u.Λ_B
    ΛΛt_tt = zero(eltype(Λ_B))
    @inbounds for k in 1:size(Λ_B, 2)
        ΛΛt_tt += Λ_B[t, k]^2
    end
    Σ = _sigma_y_site_from_unpacked(u, spec)
    return ΛΛt_tt / Σ[t, t]
end

# Wrapper that bundles spec for use as a generic derived_fn closure.
function _make_communality_closure(spec::NamedTuple, t::Integer)
    return θ -> _communality_packed(θ, spec, t)
end

# σ_eps² closure (for the parameter ↔ derived sanity check).
function _make_sigma_eps2_closure(spec::NamedTuple)
    return θ -> _derived_unpack(θ, spec).σ_eps^2
end

# ---------------------------------------------------------------------------
# Bootstrap simulation helpers — duplicated from src/confint_bootstrap.jl
# because that file injects definitions into Main, not into GLLVModels, so the
# names are not visible from inside this Core.eval block. The R-side
# equivalents (simulate.gllvmTMB_multi) play the same role.
# ---------------------------------------------------------------------------

function _derived_mean_from_X(fit::GllvmFit, X::AbstractArray{<:Real, 3})
    p = fit.model.p
    n = size(X, 2)
    β̂ = fit.pars.β
    if β̂ === nothing || length(β̂) == 0
        return zeros(Float64, p, n)
    end
    μ = zeros(Float64, p, n)
    q = size(X, 3)
    @inbounds for s in 1:n, t in 1:p
        v = 0.0
        for k in 1:q
            v += X[t, s, k] * β̂[k]
        end
        μ[t, s] = v
    end
    return μ
end

# Same per-site covariance reconstruction used by _sigma_y_site_from_unpacked,
# but operating directly on the fit's NamedTuple of raw-scale fitted
# components. Equivalent to the function in src/confint_bootstrap.jl.
function _derived_site_cov(fit::GllvmFit)
    p   = fit.model.p
    K_W = fit.model.K_W
    has_diag = fit.model.has_diag
    σ² = fit.pars.σ_eps^2
    Λ_B = fit.pars.Λ
    A = Λ_B * Λ_B'
    @inbounds for t in 1:p
        v = σ²
        if K_W > 0 && fit.pars.Λ_W !== nothing
            for k in 1:size(fit.pars.Λ_W, 2)
                v += fit.pars.Λ_W[t, k]^2
            end
        end
        if has_diag && fit.pars.σ²_B !== nothing
            v += fit.pars.σ²_B[t]
        end
        if has_diag && fit.pars.σ²_W !== nothing
            v += fit.pars.σ²_W[t]
        end
        A[t, t] += v
    end
    return A
end

function _derived_simulate!(rng::AbstractRNG, y_out::AbstractMatrix,
                            μ̂::AbstractMatrix,
                            L_site::LowerTriangular,
                            L_phy::Union{Nothing, LowerTriangular},
                            Λ_phy_aug::Union{Nothing, AbstractMatrix})
    p, n = size(y_out)
    Z = randn(rng, p, n)
    mul!(y_out, L_site, Z)
    if L_phy !== nothing && Λ_phy_aug !== nothing
        K_aug = size(Λ_phy_aug, 2)
        Φ = L_phy * randn(rng, p, K_aug)
        phy_contrib = vec(sum(Λ_phy_aug .* Φ, dims = 2))
        @inbounds for s in 1:n, t in 1:p
            y_out[t, s] += phy_contrib[t]
        end
    end
    @inbounds for s in 1:n, t in 1:p
        y_out[t, s] += μ̂[t, s]
    end
    return y_out
end

# ---------------------------------------------------------------------------
# Public API: parametric bootstrap CI for a derived quantity.
#
# Strategy: replay the existing parametric bootstrap from
# src/confint_bootstrap.jl, but instead of recording θ̂_b we record
# derived_fn(fit_b). Take percentiles over the converged replicates.
#
# derived_fn can either be a *GllvmFit*-consuming function (e.g.
# `fit -> communality(fit)[1]`) or a packed-θ closure (e.g.
# `θ -> _communality_packed(θ, spec, 1)`). We try the GllvmFit form
# first by passing in `fit_b`; on MethodError we fall back to passing
# `fit_b.pars.θ_packed`.
# ---------------------------------------------------------------------------

"""
    bootstrap_ci_derived(fit::GllvmFit, derived_fn::Function;
                         n_boot = 500, seed = 0, level = 0.95,
                         y = nothing, n_sites = nothing,
                         X = nothing, Σ_phy = nothing,
                         verbose = false)
        -> NamedTuple

Parametric bootstrap percentile CI for a scalar-valued *derived
quantity* of a fitted Gaussian GLLVM. Wraps the parametric bootstrap
in src/confint_bootstrap.jl: simulate y_b ~ N(μ̂, Σ̂_y) for b = 1..n_boot,
refit, evaluate `derived_fn` on each replicate, return percentile CIs.

`derived_fn` is called as `derived_fn(fit_b)` first and, if that errors
with a `MethodError`, as `derived_fn(fit_b.pars.θ_packed)`. Either form
is fine — pick the more convenient.

Returns a NamedTuple with fields:
  - `estimate::Float64`      — the derived quantity at the original MLE
  - `lower::Float64`         — percentile `100·(1-level)/2`
  - `upper::Float64`         — percentile `100·(1+level)/2`
  - `n_converged::Int`       — number of bootstrap fits that converged
  - `n_valid::Int`           — number of replicates with a finite derived value
  - `replicates::Vector{Float64}` — the n_boot derived-quantity samples
                                    (with `NaN` for failed refits)

Pass `y`, `X`, `Σ_phy` matching what was originally passed to
`fit_gaussian_gllvm` (the bootstrap needs them to simulate and refit).

`n_boot` defaults to 500 — publication-grade. Lower (e.g. 100) for quick
checks. The cost is `n_boot × per-fit time`; PERF+I already optimised
the per-fit path.
"""
function bootstrap_ci_derived(fit::GllvmFit, derived_fn::Function;
                              n_boot::Integer = 500,
                              level::Real = 0.95,
                              seed::Integer = 0,
                              y::Union{Nothing, AbstractMatrix} = nothing,
                              n_sites::Union{Nothing, Integer} = nothing,
                              X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                              Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                              verbose::Bool = false)

    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    n_boot ≥ 1   || throw(ArgumentError("n_boot must be ≥ 1; got $n_boot"))

    if _has_gaussian_record(fit)
        data=y===nothing ? fit.integration.data.responses : y
        return _gaussian_record_bootstrap_derived(fit,derived_fn;Y=data,X=X,Σ_phy=Σ_phy,
            n_sites=n_sites,n_boot=n_boot,level=level,seed=seed)
    end

    model = fit.model
    p     = model.p
    K_B   = model.K
    K_W   = model.K_W
    has_diag = model.has_diag
    K_phy    = model.K_phy
    has_phy_unique = model.has_phy_unique
    has_phy_block = (K_phy > 0) || has_phy_unique
    q = fit.pars.β === nothing ? 0 : length(fit.pars.β)

    # ----- Determine n_sites
    n = if n_sites !== nothing
        Int(n_sites)
    elseif X !== nothing
        size(X, 2)
    elseif y !== nothing
        size(y, 2)
    else
        throw(ArgumentError(
            "bootstrap_ci_derived needs n_sites. Pass one of: " *
            "`y = ...`, `X = ...`, or `n_sites = ...`."))
    end

    if q > 0 && X === nothing
        throw(ArgumentError(
            "Fitted model has q = $q fixed effects; pass X to bootstrap_ci_derived."))
    end
    if has_phy_block && Σ_phy === nothing
        throw(ArgumentError(
            "Fitted model has phylogenetic block; pass Σ_phy to bootstrap_ci_derived."))
    end

    # ----- Helper to call derived_fn with either GllvmFit or packed-θ.
    call_derived = function(fb::GllvmFit)
        try
            return Float64(derived_fn(fb))
        catch e
            if e isa MethodError
                return Float64(derived_fn(fb.pars.θ_packed))
            else
                rethrow(e)
            end
        end
    end

    # ----- Point estimate of derived quantity on the original fit.
    est = call_derived(fit)

    # ----- Build μ̂, L_site, L_phy, Λ_phy_aug — same as bootstrap_ci.
    μ̂ = X === nothing ? zeros(Float64, p, n) : _derived_mean_from_X(fit, X)
    A = _derived_site_cov(fit)
    A_sym = Symmetric((A + A') ./ 2)
    L_site = cholesky(A_sym).L

    L_phy = nothing
    Λ_phy_aug = nothing
    if has_phy_block
        Σ_phy_sym = Symmetric((Σ_phy + Σ_phy') ./ 2)
        L_phy = cholesky(Σ_phy_sym).L
        pieces = AbstractMatrix{Float64}[]
        if K_phy > 0 && fit.pars.Λ_phy !== nothing
            push!(pieces, fit.pars.Λ_phy)
        end
        if has_phy_unique && fit.pars.σ_phy !== nothing
            push!(pieces, reshape(collect(Float64, fit.pars.σ_phy), p, 1))
        end
        Λ_phy_aug = isempty(pieces) ? nothing : reduce(hcat, pieces)
    end

    rng = MersenneTwister(seed)
    replicates = fill(NaN, n_boot)
    n_converged = 0

    y_b = Matrix{Float64}(undef, p, n)

    for b in 1:n_boot
        _derived_simulate!(rng, y_b, μ̂, L_site, L_phy, Λ_phy_aug)
        try
            fit_b = fit_gaussian_gllvm(y_b;
                                       K = K_B,
                                       K_W = K_W,
                                       has_diag = has_diag,
                                       K_phy = K_phy,
                                       has_phy_unique = has_phy_unique,
                                       Σ_phy = Σ_phy,
                                       X = X)
            if fit_b.converged
                n_converged += 1
            end
            v = call_derived(fit_b)
            if isfinite(v)
                replicates[b] = v
            end
        catch e
            verbose && @info "bootstrap_ci_derived rep $b failed: $e"
        end
    end

    α = (1 - level) / 2
    valid = filter(isfinite, replicates)
    n_valid = length(valid)
    lower, upper = if n_valid ≥ 10
        (_derived_percentile(valid, α), _derived_percentile(valid, 1 - α))
    else
        (NaN, NaN)
    end

    return (estimate    = est,
            lower       = lower,
            upper       = upper,
            n_converged = n_converged,
            n_valid     = n_valid,
            replicates  = replicates)
end

# ---------------------------------------------------------------------------
# Profile CI for a scalar-valued derived quantity.
#
# Constrained refit via quadratic penalty:
#   NLL_pen(θ; c, w) = NLL(θ) + (w / 2) · (g(θ) − c)²
# where g(θ) = derived_fn(θ). Minimise over θ via LBFGS. The
# *unpenalised* NLL evaluated at θ̂(c) is the profile log-likelihood at
# c.
#
# Bracket-then-bisect mirrors src/confint_profile.jl's _profile_bisect_side.
# Initial step: heuristic based on the derived value at the MLE (we don't
# have a Wald SE for the derived quantity directly — we *could* compute
# one via the delta method, but it adds complexity and the geometric
# expansion handles the slack well enough in practice).
#
# Numerical hardening (fixes the degenerate-interval bug in phylo cells):
#
#   1. Augmented-Lagrangian-style penalty escalation. A fixed large
#      `penalty_weight = 1e6` instantly inflates the penalty term at the
#      warm-start θ̂ (where g(θ̂) ≠ c by O(0.1)), driving LBFGS into bad
#      regions of θ-space from which it cannot recover. Instead we sweep
#      w through an increasing schedule (1e2 → 1e3 → … → final), warm-
#      starting each stage from the previous minimiser. This lets the
#      optimiser move smoothly to the {g ≈ c} manifold first, then tighten
#      the constraint.
#
#   2. PosDef-safe NLL wrapper. As c moves away from g(θ̂) in phylo-active
#      cells, the constrained per-site Σ can drift to near-singular,
#      where `gaussian_nll_packed` throws `PosDefException` from its
#      internal Cholesky. We wrap the NLL to convert that exception (and
#      any non-finite NLL value) into a finite barrier — the optimiser
#      then sees a large but finite penalty and backs away from the
#      infeasible region instead of crashing.
#
#   3. BackTracking line search. The default HagerZhang line search
#      asserts `isfinite(phi_c) && isfinite(dphi_c)` at trial points and
#      crashes when steep gradients in near-singular regions produce
#      non-finite directional derivatives during cubic interpolation.
#      BackTracking only requires Armijo's sufficient-decrease condition
#      and tolerates non-finite trial values by simply halving the step
#      and trying again — which composes correctly with the safe-NLL
#      barrier.
# ---------------------------------------------------------------------------

# Safe NLL wrapper: converts PosDefException / non-finite values into a
# large finite barrier so LBFGS can back away gracefully. AD-friendly:
# the barrier value is constructed with `oftype(...)` / `T(...)` so it
# preserves ForwardDiff Dual eltype.
const _DERIVED_NLL_BARRIER = 1e10

function _derived_safe_nll(θ::AbstractVector, y::AbstractMatrix,
                           spec::NamedTuple,
                           X::Union{Nothing, AbstractArray{<:Real, 3}},
                           Σ_phy::Union{Nothing, AbstractMatrix})
    T = eltype(θ)
    v = try
        gaussian_nll_packed(θ, y; spec = spec, X = X, Σ_phy = Σ_phy)
    catch
        return T(_DERIVED_NLL_BARRIER)
    end
    return isfinite(v) ? v : T(_DERIVED_NLL_BARRIER)
end

# Constrained refit returning (ll_profile, success, θ_warm_new, g_at_min).
# Uses an increasing-w (augmented-Lagrangian-flavoured) schedule, a
# PosDef-safe NLL, and BackTracking line search — see the block comment
# above.
function _derived_refit_with_fixed(fit::GllvmFit,
                                   derived_fn_packed::Function,
                                   c::Real,
                                   y::AbstractMatrix,
                                   X::Union{Nothing, AbstractArray{<:Real, 3}},
                                   Σ_phy::Union{Nothing, AbstractMatrix};
                                   θ_warm::Union{Nothing, AbstractVector} = nothing,
                                   penalty_weight::Real = 1e6,
                                   penalty_schedule::Union{Nothing, AbstractVector{<:Real}} = nothing,
                                   x_tol::Real = 1e-6,
                                   f_tol::Real = 1e-8,
                                   g_tol::Real = 1e-4,
                                   iterations::Integer = 300)
    spec = _derived_spec(fit)
    target_nll=_has_gaussian_record(fit) ? _gaussian_record_ci(fit,y;X=X,Σ_phy=Σ_phy).nll :
        (theta->_derived_safe_nll(theta,y,spec,_profile_free_X(fit,X),Σ_phy))
    θ̂ = fit.pars.θ_packed
    θ0 = θ_warm === nothing ? collect(Float64, θ̂) : collect(Float64, θ_warm)

    c_float = float(c)
    w_final = float(penalty_weight)

    # Build the escalating-w schedule. We climb in roughly decade steps
    # from 1e2 up to w_final, capping at 6 stages so a single refit costs
    # at most ~6 LBFGS solves. If the caller passes a custom schedule we
    # honour it verbatim.
    schedule = if penalty_schedule !== nothing
        [float(w) for w in penalty_schedule]
    else
        # Geometric ramp ending at w_final. Each stage warm-starts from
        # the previous, so each LBFGS call only needs to tighten the
        # constraint by a decade — ~5-15 iters in practice.
        s = Float64[]
        w = 1e2
        while w < w_final
            push!(s, w)
            w *= 10.0
        end
        push!(s, w_final)
        s
    end

    opts = Optim.Options(
        x_abstol = x_tol,
        f_reltol = f_tol,
        g_tol    = g_tol,
        iterations = iterations,
        show_trace = false,
    )

    # BackTracking line search tolerates non-finite trial points by
    # halving the step (cf. HagerZhang's assertion-based termination).
    method = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking())

    # Sweep penalty weight w through the schedule; warm-start each stage
    # from the previous minimiser.
    for w in schedule
        nll_pen = θ -> begin
            nll = target_nll(θ)
            g = derived_fn_packed(θ)
            return nll + 0.5 * w * (g - c_float)^2
        end
        res = try
            Optim.optimize(nll_pen, θ0, method, opts; autodiff = :forward)
        catch
            return (NaN, false, θ0, NaN)
        end
        θ0 = Optim.minimizer(res)
    end

    θ_min = θ0
    nll_unpen = try
        target_nll(θ_min)
    catch e
        e isa InterruptException && rethrow()
        return (NaN,false,θ_min,NaN)
    end
    # If the final unpenalised NLL is still at the barrier, the refit
    # landed in a non-PD region of θ-space — treat as a failure.
    if !isfinite(nll_unpen) || nll_unpen ≥ _DERIVED_NLL_BARRIER / 2
        return (NaN, false, θ_min, NaN)
    end
    g_at_min = try
        derived_fn_packed(θ_min)
    catch
        return (NaN, false, θ_min, NaN)
    end
    return (-nll_unpen, true, θ_min, g_at_min)
end

# Reuse the bisection helper structure from src/confint_profile.jl but
# inlined here so this slice does not depend on internal names from a
# sibling file (Core.eval scoping makes that brittle).
#
# Degenerate-interval guard: if the expansion phase failed to cross
# `cutoff` AND the bracket's "outer" point is at the very first step
# (no real bracket established) we return NaN rather than collapsing
# to `(lo + hi)/2 ≈ x0`. The old code returned a near-x0 midpoint when
# the expansion exited on the very first refit failure, which is what
# produced the "zero-width CI" pathology in phylo-active cells.
function _derived_bisect_side(D::Function, x0::Real, step_init::Real,
                              cutoff::Real;
                              max_expand::Integer = 20,
                              max_bisect::Integer = 30,
                              tol_x::Real = 1e-4)
    sign_step = sign(step_init)
    sign_step == 0 && return NaN
    abs_step = abs(step_init)

    x_in = float(x0)
    D_in = 0.0
    x_out = x_in + sign_step * abs_step
    D_out = NaN
    found = false
    n_in_advances = 0   # how many times we successfully advanced x_in (i.e. real progress)
    for _ in 1:max_expand
        D_val = D(x_out)
        if !isfinite(D_val)
            D_out = Inf
            found = true
            break
        end
        if D_val ≥ cutoff
            D_out = D_val
            found = true
            break
        end
        x_in = x_out
        D_in = D_val
        n_in_advances += 1
        abs_step *= 2
        x_out = x_in + sign_step * abs_step
    end
    found || return NaN
    # If we found the bracket on the *very first* refit and it was a
    # non-finite refit (PosDef failure right next to x0), we have no
    # interior evidence of where the cutoff actually lies. Return NaN
    # so callers see this as a failure rather than a degenerate
    # near-x0 interval.
    if n_in_advances == 0 && !isfinite(D_out)
        return NaN
    end

    lo, hi = x_in, x_out
    D_lo, D_hi = D_in, D_out
    for _ in 1:max_bisect
        mid = (lo + hi) / 2
        D_mid = D(mid)
        if !isfinite(D_mid)
            hi = mid
            D_hi = Inf
        elseif D_mid ≥ cutoff
            hi = mid
            D_hi = D_mid
        else
            lo = mid
            D_lo = D_mid
        end
        if abs(hi - lo) < tol_x
            break
        end
    end
    return (lo + hi) / 2
end

"""
    profile_ci_derived(fit::GllvmFit, derived_fn::Function;
                       level = 0.95, y = nothing,
                       X = nothing, Σ_phy = nothing,
                       penalty_weight = 1e6,
                       initial_step = nothing,
                       max_expand = 20, max_bisect = 30)
        -> NamedTuple{(:lower, :upper, :estimate, :method)}

Profile-likelihood CI for a scalar-valued *derived quantity*
`g(θ) = derived_fn(θ_packed)`. The constraint `g(θ) = c` is enforced via
a quadratic penalty
    `NLL_pen(θ) = NLL(θ) + 0.5 · penalty_weight · (g(θ) − c)²`,
re-optimised over θ via LBFGS at each candidate c. The profile log-
likelihood at c is the unpenalised NLL evaluated at the constrained
minimum θ̂(c); the deviance D(c) = 2(ℓ̂ − ℓ_p(c)) is ~ χ²₁ under
g(θ) = c, so the CI is {c : D(c) ≤ qchisq(1−α, 1)}. Bracket-then-bisect
on each side.

`derived_fn` must accept a packed-parameter vector and return a scalar
(`Float64`). For the built-in derived quantities, use the closure
helpers:

```julia
spec = GLLVModels._derived_spec(fit)
f_c1 = θ -> GLLVModels._communality_packed(θ, spec, 1)
ci   = GLLVModels.profile_ci_derived(fit, f_c1; y = y)
```

Or, for σ²_eps (sanity check vs the parameter profile CI on σ_eps):

```julia
f_s2 = θ -> exp(2 * θ[1])    # log_σ_eps is at index 1 when q = 0
```

`penalty_weight` defaults to 1e6 — the *final* weight at the end of an
internal escalating schedule (1e2 → 1e3 → … → `penalty_weight`). The
escalation is essential: in phylogenetically active cells, jumping
straight to a large w at the warm-start θ̂ inflates the penalty term by
O(w · (g(θ̂) − c)²) and pushes LBFGS into pathological regions (the
constrained per-site covariance can drift non-PD), producing degenerate
CIs. The schedule lets the optimiser move smoothly to the {g ≈ c}
manifold first, then tightens. The internal NLL is also wrapped to
return a finite barrier on `PosDefException`.

`initial_step` (default `nothing` → `max(0.05 · |g(θ̂)|, 0.01)`) seeds
the bracket expansion. Smaller is better here — the geometric expansion
inside the bisection grows the step rapidly, while a small first step
keeps the very first constrained refit close to θ̂ (where the safe-NLL
barrier is rarely triggered).

Returns a NamedTuple with fields:
  - `estimate::Float64` — `g(θ̂)` at the original MLE
  - `lower::Float64`    — lower CI bound (NaN if bracket failed)
  - `upper::Float64`    — upper CI bound (NaN if bracket failed)
  - `method::Symbol`    — `:profile` (both bounds), `:partial`
                          (one side NaN), or `:failed` (both NaN)
"""
function profile_ci_derived(fit::GllvmFit, derived_fn::Function;
                            level::Real = 0.95,
                            y::Union{Nothing, AbstractMatrix} = nothing,
                            X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                            Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                            penalty_weight::Real = 1e6,
                            initial_step::Union{Nothing, Real} = nothing,
                            max_expand::Integer = 20,
                            max_bisect::Integer = 30)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    y === nothing && throw(ArgumentError(
        "profile_ci_derived requires the data matrix `y`"))

    θ̂ = fit.pars.θ_packed
    g_hat = Float64(derived_fn(θ̂))
    isfinite(g_hat) || throw(ArgumentError(
        "derived_fn returned a non-finite value at the MLE: $g_hat"))

    cutoff = quantile(Chisq(1), level)
    ll_full = fit.logLik

    step_init = if initial_step === nothing
        max(0.05 * abs(g_hat), 0.01)
    else
        float(initial_step)
    end

    θ_warm_lower = collect(Float64, θ̂)
    θ_warm_upper = collect(Float64, θ̂)

    deviance_lower = function(c)
        ll_c, ok, θ_new, _ = _derived_refit_with_fixed(
            fit, derived_fn, c, y, X, Σ_phy;
            θ_warm = θ_warm_lower,
            penalty_weight = penalty_weight)
        if ok
            θ_warm_lower = θ_new
            return 2.0 * (ll_full - ll_c)
        else
            return NaN
        end
    end
    deviance_upper = function(c)
        ll_c, ok, θ_new, _ = _derived_refit_with_fixed(
            fit, derived_fn, c, y, X, Σ_phy;
            θ_warm = θ_warm_upper,
            penalty_weight = penalty_weight)
        if ok
            θ_warm_upper = θ_new
            return 2.0 * (ll_full - ll_c)
        else
            return NaN
        end
    end

    lower = _derived_bisect_side(deviance_lower, g_hat, -step_init, cutoff;
                                 max_expand = max_expand,
                                 max_bisect = max_bisect)
    upper = _derived_bisect_side(deviance_upper, g_hat,  step_init, cutoff;
                                 max_expand = max_expand,
                                 max_bisect = max_bisect)

    method = if isnan(lower) && isnan(upper)
        :failed
    elseif isnan(lower) || isnan(upper)
        :partial
    else
        :profile
    end
    return (lower = lower, upper = upper,
            estimate = g_hat, method = method)
end

# ---------------------------------------------------------------------------
# Feasible-range clamp + boundary flag for a profile_ci_derived result.
#
# profile_ci_derived's bracket-then-bisect walk has no notion of the derived
# quantity's natural feasible range [lo_bound, hi_bound]: a bound can
# overshoot past it (numerical slack near a genuinely near-boundary
# optimum), or the deviance can plateau below the χ²₁ cutoff all the way out
# to the range edge without ever crossing it (max_expand exhausted → NaN /
# :partial, indistinguishable from an unrelated bracketing failure). This
# wrapper (1) clamps any bound that violates the feasible range back to the
# edge, and (2) for a NaN bound with a finite edge, evaluates the deviance
# AT the edge itself — if it is still below cutoff, the CI plateaus at the
# boundary and that is reported as the bound with `boundary = true`, rather
# than a bare NaN/:partial.
# ---------------------------------------------------------------------------
function _profile_ci_bounded(fit::GllvmFit, derived_fn::Function, r::NamedTuple;
                             level::Real, y::AbstractMatrix,
                             X::Union{Nothing, AbstractArray{<:Real, 3}},
                             Σ_phy::Union{Nothing, AbstractMatrix},
                             lo_bound::Real, hi_bound::Real)
    cutoff = quantile(Chisq(1), level)
    lower, upper, boundary = r.lower, r.upper, false

    if isfinite(lower) && lower < lo_bound
        lower, boundary = float(lo_bound), true
    elseif isnan(lower) && isfinite(lo_bound)
        ll_c, ok, _, _ = _derived_refit_with_fixed(fit, derived_fn, lo_bound, y, X, Σ_phy)
        if ok
            D = 2.0 * (fit.logLik - ll_c)
            if isfinite(D) && D ≤ cutoff
                lower, boundary = float(lo_bound), true
            end
        end
    end

    if isfinite(upper) && upper > hi_bound
        upper, boundary = float(hi_bound), true
    elseif isnan(upper) && isfinite(hi_bound)
        ll_c, ok, _, _ = _derived_refit_with_fixed(fit, derived_fn, hi_bound, y, X, Σ_phy)
        if ok
            D = 2.0 * (fit.logLik - ll_c)
            if isfinite(D) && D ≤ cutoff
                upper, boundary = float(hi_bound), true
            end
        end
    end

    method = if isnan(lower) && isnan(upper)
        :failed
    elseif isnan(lower) || isnan(upper)
        :partial
    else
        :profile
    end

    return merge(r, (; lower = lower, upper = upper, boundary = boundary, method = method))
end

# ---------------------------------------------------------------------------
# Thin profile-CI wrappers named to match the missing-surface case map
# (docs/dev-log/core070/required-source-case-map.json rows
# namespace/export/profile_ci_total_variance,
# namespace/export/profile_ci_phylo_signal). Both are one-line closures over
# the generic profile_ci_derived machinery above; no new numerical method.
# ---------------------------------------------------------------------------

# Total per-trait variance Σ_y_site[t,t] from the packed θ. AD-friendly
# (reuses _sigma_y_site_from_unpacked, already ForwardDiff-clean).
function _total_variance_packed(θ::AbstractVector, spec::NamedTuple, t::Integer)
    Σ = _sigma_y_site_packed(θ, spec)
    return Σ[t, t]
end

function _make_total_variance_closure(spec::NamedTuple, t::Integer)
    return θ -> _total_variance_packed(θ, spec, t)
end

"""
    profile_ci_total_variance(fit::GllvmFit, t::Integer; level=0.95, y=nothing,
                              X=nothing, Σ_phy=nothing, kwargs...)
        -> NamedTuple

Profile-likelihood CI for the per-trait total variance `Σ_y_site[t, t]`
(see [`sigma_y_site`](@ref) for the definition). Thin wrapper around
[`profile_ci_derived`](@ref) with `derived_fn = θ -> Σ_y_site(θ)[t, t]`; all
keyword arguments (`penalty_weight`, `initial_step`, `max_expand`,
`max_bisect`) are forwarded. `t` is a 1-based trait index. No transform is
applied — the total variance is strictly positive but unbounded above, so
the raw-scale bracket-then-bisect search (mirroring how this file already
profiles `σ²_eps`) can in principle return a `lower` bound that drifted at
or below `0`, or a plateau that never crosses the χ²₁ cutoff before the
bracket expansion gives up. [`_profile_ci_bounded`](@ref) is applied to the
result: any `lower < 0` is clamped to `0`, and a `NaN` `lower` whose
deviance-at-`0` is itself still below cutoff is reported as `lower = 0`
instead — both cases set the additional `boundary::Bool` field on the
returned NamedTuple.
"""
function profile_ci_total_variance(fit::GllvmFit, t::Integer;
                                   level::Real = 0.95,
                                   y::Union{Nothing, AbstractMatrix} = nothing,
                                   X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                                   Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                                   kwargs...)
    spec = _derived_spec(fit)
    f = _make_total_variance_closure(spec, t)
    r = profile_ci_derived(fit, f; level = level, y = y, X = X, Σ_phy = Σ_phy,
                           kwargs...)
    return _profile_ci_bounded(fit, f, r; level = level, y = y, X = X, Σ_phy = Σ_phy,
                               lo_bound = 0.0, hi_bound = Inf)
end

"""
    profile_ci_phylo_signal(fit::GllvmFit, t::Integer; level=0.95, y=nothing,
                            X=nothing, Σ_phy=nothing, kwargs...)
        -> NamedTuple

Profile-likelihood CI for the per-trait phylogenetic signal `H²[t]` (see
[`phylo_signal`](@ref)). Thin wrapper around [`profile_ci_derived`](@ref)
using the packed-θ closure `GLLVModels._make_phylo_signal_closure`
(`src/confint_derived_wald.jl`) — the same closure the transformed-Wald
route `phylo_signal_wald_ci` uses, so the point estimate matches to the
bit. `Σ_phy` enters only through its diagonal (standardised convention →
unit diagonal when omitted), consistent with `phylo_signal`/
`phylo_signal_wald_ci`. `t` is a 1-based trait index; all keyword
arguments forward to `profile_ci_derived`.

`H²[t] ∈ [0, 1]` (see [`phylo_signal`](@ref)): [`_profile_ci_bounded`](@ref)
is applied to the result, clamping any bound outside `[0, 1]` back to the
edge and reporting a boundary plateau (deviance-at-`0`-or-`1` already below
the χ²₁ cutoff) as that edge instead of `NaN`/`:partial`. The returned
NamedTuple carries the additional `boundary::Bool` field.
"""
function profile_ci_phylo_signal(fit::GllvmFit, t::Integer;
                                 level::Real = 0.95,
                                 y::Union{Nothing, AbstractMatrix} = nothing,
                                 X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                                 Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                                 kwargs...)
    spec = _derived_spec(fit)
    diag_Σphy = Σ_phy === nothing ? nothing : diag(Σ_phy)
    f = _make_phylo_signal_closure(spec, t; diag_Σphy = diag_Σphy)
    r = profile_ci_derived(fit, f; level = level, y = y, X = X, Σ_phy = Σ_phy,
                           kwargs...)
    return _profile_ci_bounded(fit, f, r; level = level, y = y, X = X, Σ_phy = Σ_phy,
                               lo_bound = 0.0, hi_bound = 1.0)
end

"""
    loading_profile_exploratory(fit::GllvmFit, t::Integer, k::Integer; level=0.95,
                                y=nothing, X=nothing, Σ_phy=nothing, component=:B,
                                kwargs...)
        -> NamedTuple

Profile-likelihood CI for the RAW reduced-rank loading entry `Λ[t, k]` on an
**exploratory (unpinned)** fit (`component = :B`, default — the shared/between
tier; `component = :W` for the within tier). GLLVModels.jl has no separate
confirmatory fit mode with `lambda_constraint` pins; the lower-triangular
packing convention (`src/packing.jl`) is this package's built-in identifiability
device, so this function runs on any fit.

**Estimand scope (Core070 D3, 2026-09-04):** this is **not** R's
`loading_profile()`, which profiles a Λ entry on a **confirmatory
(pinned-loadings)** fit gated on R's `fit\$lambda_constraint`
(`.unlazy/core070-aghq/oracle-source/readback/R/loading-profile.R`). Same
family of quantity, different estimand — the rename makes that gap visible in
the API rather than hiding it behind a matching signature.

Implementation: a thin wrapper around [`profile_ci_derived`](@ref) using the
packed-θ closure `θ -> Λ_component(θ)[t, k]`; all keyword arguments forward.

For `k > t` on the lower-triangular reduced-rank packing convention, the entry
is structurally pinned at `0`: this returns
`(lower = 0.0, upper = 0.0, estimate = 0.0, method = :pinned)` without running
the profiler, matching the `pinned = TRUE` short-circuit in R's `loading_ci()`.
"""
function loading_profile_exploratory(fit::GllvmFit, t::Integer, k::Integer;
                                     level::Real = 0.95,
                                     y::Union{Nothing, AbstractMatrix} = nothing,
                                     X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                                     Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                                     component::Symbol = :B,
                                     kwargs...)
    if k > t
        return (lower = 0.0, upper = 0.0, estimate = 0.0, method = :pinned)
    end
    spec = _derived_spec(fit)
    f = θ -> begin
        u = _derived_unpack(θ, spec)
        Λ = component === :B ? u.Λ_B :
            component === :W ? u.Λ_W :
            throw(ArgumentError("component must be :B or :W; got $(component)"))
        Λ === nothing && throw(ArgumentError(
            "fit has no Λ_$(component) block; loading_profile_exploratory is undefined"))
        Λ[t, k]
    end
    return profile_ci_derived(fit, f; level = level, y = y, X = X, Σ_phy = Σ_phy,
                              kwargs...)
end

"""
    loading_profile(fit::GllvmFit; level::Symbol = :unit,
                     entries::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                     n_grid::Integer = 11, grid_extent::Real = 6.0,
                     conf_level::Real = 0.95, y::AbstractMatrix)
        -> NamedTuple

**D3 Stage 1 confirmatory profile-likelihood grid** for the free entries of a
**pinned** `Λ` (`level = :unit`, the shared/between tier), mirroring R
gllvmTMB's `loading_profile()`. Requires `fit` to be a **confirmatory** fit —
built via `fit_gaussian_gllvm(y; K, lambda_constraint = M)` — so this refuses
plain exploratory fits (opposite of R's `loading_ci()`, which refuses
unpinned fits for the same reason: distinct estimands need distinct fit
metadata). Use [`loading_profile_exploratory`](@ref) for a penalty-based
profile CI on a single raw `Λ` entry of an ordinary (unpinned) fit.

# Keywords
- `level`: `:unit` (equivalently `:B`) is the only tier Stage 1 supports —
  the shared/between-tier `Λ`. `:unit_obs`/`:W` are reserved for a later slice
  (no within-tier block exists on the J1 fits this admits).
- `entries`: an `n × 2` integer matrix of `(trait, axis)` pairs to profile: one
  row per requested entry. `nothing` (default) profiles every free entry.
- `n_grid`: number of grid points per profiled entry (must be `≥ 3`).
- `grid_extent`: profiling half-width in Wald-SE units around the confirmatory
  MLE for each entry (falls back to `|Λ̂|/2 + 0.5` when the Hessian-based SE is
  non-finite, matching the fallback in `loading_profile_exploratory`).
- `conf_level`: confidence level recorded alongside each row (not itself used
  to trim the grid in Stage 1 — grid width is controlled by `n_grid` /
  `grid_extent`).
- `y`: the response matrix used to fit `fit` (required — same convention as
  `loading_profile_exploratory`).

# Return
A `NamedTuple` `(table, level, n_grid, grid_extent, conf_level, entries)`.
`table` is a `Vector` of row `NamedTuple`s (`Tables.jl`-compatible row table)
with fields `trait`, `axis`, `i`, `k`, `profile_value`, `objective`
(`-logLik` at that grid point), `delta_deviance` (`2 * (fit.logLik - ll)`,
`NaN` when the refit did not converge), `estimate` (the confirmatory MLE for
that entry), `conf_level`, and `converged`.

# Stage 1 scope (Rose fence)
This is a **Stage 1 receipt**, not full R grid parity and not `T5` row 8
"covered": ordinary J1 Gaussian only (`K_W = 0`, `has_diag = false`,
`K_phy = 0`, `has_phy_unique = false`), `X = nothing` fits only, one entry
pinned per refit via
[`GLLVModels._confirmatory_profile_refit_lambda_pin`](@ref) (internal), and a
grid built from a Wald-SE heuristic rather than R's exact grid-spacing rule.
See `docs/dev-log/plans/2026-09-16-d3-loading-profile-stage1-paste-gated-scaffold.md`.
"""
function loading_profile(fit::GllvmFit;
                          level::Symbol = :unit,
                          entries::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                          n_grid::Integer = 11,
                          grid_extent::Real = 6.0,
                          conf_level::Real = 0.95,
                          y::AbstractMatrix)
    level === :unit || level === :B ||
        throw(ArgumentError(
            "loading_profile Stage 1 supports level = :unit (component = :B) only; got $(level)"))
    haskey(fit.pars, :lambda_constraint) ||
        throw(ArgumentError(
            "loading_profile requires a confirmatory fit: refit with " *
            "fit_gaussian_gllvm(y; K, lambda_constraint = M) before profiling"))
    n_grid isa Integer && n_grid >= 3 ||
        throw(ArgumentError("n_grid must be an integer ≥ 3"))
    0.0 < conf_level < 1.0 || throw(ArgumentError("conf_level must lie in (0, 1)"))
    grid_extent > 0 || throw(ArgumentError("grid_extent must be positive"))

    p, K = fit.model.p, fit.model.K
    M_user = fit.pars.lambda_constraint
    free = _enumerate_free_lambda_entries(M_user, p, K; entries = entries)
    isempty(free) &&
        throw(ArgumentError("no free Λ entries to profile (all pinned/structural)"))

    Λ̂ = fit.pars.Λ
    rows = NamedTuple[]
    for (i, k) in free
        θ_idx = _lambda_b_theta_index(fit, i, k)
        se = _profile_wald_se(fit, θ_idx, y, nothing, nothing)
        c_hat = Λ̂[i, k]
        half_width = isfinite(se) && se > 0 ? grid_extent * se :
            grid_extent * (abs(c_hat) / 2 + 0.5)
        grid = range(c_hat - half_width, c_hat + half_width; length = n_grid)
        for c in grid
            ll, ok = _confirmatory_profile_refit_lambda_pin(
                fit, y, M_user, i, k, c; require_paste = false)
            obj = ok ? -ll : NaN
            delta_dev = ok ? 2 * (fit.logLik - ll) : NaN
            push!(rows, (trait = i, axis = k, i = i, k = k,
                          profile_value = c, objective = obj,
                          delta_deviance = delta_dev, estimate = c_hat,
                          conf_level = conf_level, converged = ok))
        end
    end
    return (table = rows, level = :unit, n_grid = n_grid, grid_extent = grid_extent,
            conf_level = conf_level, entries = free)
end

function loading_profile(args...; kwargs...)
    Base.depwarn(
        "loading_profile is deprecated: renamed to loading_profile_exploratory; " *
        "the name loading_profile is reserved for a future R-mirroring " *
        "confirmatory (pinned-loadings) surface",
        :loading_profile)
    return loading_profile_exploratory(args...; kwargs...)
end
