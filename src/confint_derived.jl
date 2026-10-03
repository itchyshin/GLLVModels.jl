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
# Σ_y_site = Λ_B Λ_B' + Λ_W Λ_W' + diag(d_total) — the per-site (within-species)
# trait covariance. The W tier carries its full cross-trait block (issue #135).
# The phylogenetic block (rank-1 across species) is not part of the per-site
# covariance; it contributes only to the species-level
# shared variance. Following the bootstrap-sigma R convention.
# ---------------------------------------------------------------------------
function _sigma_y_site_from_unpacked(u::NamedTuple, spec::NamedTuple)
    p   = spec.p
    K_W = spec.K_W
    has_diag = spec.has_diag
    σ² = u.σ_eps^2
    Λ_B = u.Λ_B
    A = Λ_B * Λ_B'
    if K_W > 0 && u.Λ_W !== nothing
        A = A + u.Λ_W * u.Λ_W'
    end
    @inbounds for t in 1:p
        v = σ²
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
`Σ_y_site = Λ_B Λ_B' + Λ_W Λ_W' + diag(d_total)` where
`d_total[t] = σ²_B[t] + σ²_W[t] + σ²_eps`. The W tier contributes its full
cross-trait block `Λ_W Λ_W'`, as in gllvmTMB (#135). For J1,
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
is the fraction of the per-site trait variance explained by the unit-tier
latent factors `Λ_B`. Values are in [0, 1]. The W-tier block `Λ_W Λ_W'` is
also shared across traits (#135) but is not counted here.

With `K_W > 0`, each column of `y` is one unit observed once, so the data
identify only `Λ_B Λ_B' + Λ_W Λ_W'`, not how it splits between the tiers.
This value then depends on the starting values and should not be
interpreted; see the Identifiability section of the Model page.
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
`:shared`, `:unique_W`, `:unique_B`, `:unique_Wd` and `:residual` shares
sum to 1 (when has_diag and W tier are off, only `:shared` and `:residual`
are non-zero).

Despite its name, `:unique_W` is the diagonal of the W-tier block
`Λ_W Λ_W'`, which also covaries across traits (#135). With one unit
per column of `y`, the data identify only the sum of the `:shared` and
`:unique_W` shares, and only the sum of the `:unique_B`, `:unique_Wd` and
`:residual` shares. With `K_W > 0` or `has_diag = true`, the individual
shares within each sum depend on the starting values and should not be
interpreted; see the Identifiability section of the Model page.

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
correlations driven by the loadings Λ_B and, when present, Λ_W. They depend
only on the identified total `Σ_y_site`, so they do not depend on how a
`K_W > 0` fit splits the covariance between the two tiers.

Degenerate traits: the correlation is undefined for a trait whose
`Σ_y_site[t, t] ≤ 0` (zero variance, or a round-off negative). Every entry
of row and column `t` — including the diagonal — is then `NaN`, and one
warning naming the affected traits is emitted per call. No `DomainError`
or silent `Inf` is produced.
"""
function correlation(fit::GllvmFit)
    Σ = sigma_y_site(fit)
    p = size(Σ, 1)
    R = similar(Σ, Float64)
    bad = [t for t in 1:p if !(Σ[t, t] > 0)]
    isempty(bad) || @warn "correlation: trait(s) $(bad) have non-positive variance " *
        "Σ_y_site[t, t] ≤ 0; their correlations are returned as NaN" maxlog = 1
    @inbounds for j in 1:p, i in 1:p
        if Σ[i, i] > 0 && Σ[j, j] > 0
            denom = sqrt(Σ[i, i] * Σ[j, j])
            # the product can underflow to 0 although each variance is positive
            denom > 0 || (denom = sqrt(Σ[i, i]) * sqrt(Σ[j, j]))
            R[i, j] = Σ[i, j] / denom
        else
            R[i, j] = NaN
        end
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
    if K_W > 0 && fit.pars.Λ_W !== nothing
        A = A + fit.pars.Λ_W * fit.pars.Λ_W'      # full W-tier block (issue #135)
    end
    @inbounds for t in 1:p
        v = σ²
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
                                    (with `NaN` for failed refits; values from
                                    refits that did not converge are kept for
                                    inspection but are not used for the bounds)
  - `converged::Vector{Bool}` — per refit: did it converge
  - `n_used::Int`            — replicates that entered the percentiles
                               (converged and finite)
  - `n_dropped::Int`         — `n_boot - n_used`

A refit that errors or does not converge is dropped from the percentile
computation, as in gllvmTMB's `bootstrap_Sigma`; `n_used` and `n_dropped`
report the counts, and one warning is issued per call when more than half of
the refits are dropped. The bounds are `NaN` when fewer than 10 replicates are
used.

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
                              verbose::Bool = false,
                              # Internal test seam, not API (see `bootstrap_ci`, #140).
                              _refit::Function = fit_gaussian_gllvm)

    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    n_boot ≥ 1   || throw(ArgumentError("n_boot must be ≥ 1; got $n_boot"))

    if _has_gaussian_record(fit)
        data=y===nothing ? fit.integration.data.responses : y
        r=_gaussian_record_bootstrap_derived(fit,derived_fn;Y=data,X=X,Σ_phy=Σ_phy,
            n_sites=n_sites,n_boot=n_boot,level=level,seed=seed)
        # The record refit already keeps only converged refits, and its percentiles use
        # the finite derived values among them (`n_valid`), so those are the ones used (#140).
        n_dropped=n_boot-r.n_valid
        _bootstrap_warn_dropped("bootstrap_ci_derived",n_dropped,n_boot)
        return merge(r,(n_used=r.n_valid,n_dropped=n_dropped))
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

    X = _mean_X(fit, X, n)
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
    converged = fill(false, n_boot)

    y_b = Matrix{Float64}(undef, p, n)

    for b in 1:n_boot
        _derived_simulate!(rng, y_b, μ̂, L_site, L_phy, Λ_phy_aug)
        try
            fit_b = _refit(y_b;
                           K = K_B,
                           K_W = K_W,
                           has_diag = has_diag,
                           K_phy = K_phy,
                           has_phy_unique = has_phy_unique,
                           Σ_phy = Σ_phy,
                           X = X)
            converged[b] = fit_b.converged
            v = call_derived(fit_b)
            if isfinite(v)
                replicates[b] = v
            end
        catch e
            verbose && @info "bootstrap_ci_derived rep $b failed: $e"
        end
    end

    α = (1 - level) / 2
    n_valid = count(isfinite, replicates)
    n_converged = count(converged)
    # Percentiles over converged refits only (#140): a finite derived value from a
    # refit that stopped far from the optimum is not a draw from the estimator.
    usable = [converged[b] && isfinite(replicates[b]) for b in 1:n_boot]
    valid = replicates[usable]
    n_used = length(valid)
    n_dropped = n_boot - n_used
    _bootstrap_warn_dropped("bootstrap_ci_derived", n_dropped, n_boot)
    lower, upper = if n_used ≥ 10
        (_derived_percentile(valid, α), _derived_percentile(valid, 1 - α))
    else
        (NaN, NaN)
    end

    return (estimate    = est,
            lower       = lower,
            upper       = upper,
            n_converged = n_converged,
            n_valid     = n_valid,
            replicates  = replicates,
            converged   = converged,
            n_used      = n_used,
            n_dropped   = n_dropped)
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

# Absolute tolerance on |g(θ_refit) − c| (R: `.fix_and_refit_constraint_tol`).
const _DERIVED_CONSTRAINT_TOL = 0.05

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
    # Constraint-satisfaction gate, mirroring R's `.fix_and_refit_nll()`
    # (R/profile-derived.R `.fix_and_refit_constraint_tol` = 0.05): a refit whose
    # achieved g(θ) misses the target c by more than the tolerance did not
    # enforce the constraint, so its deviance is not a profile deviance.
    if !isfinite(g_at_min) || abs(g_at_min - c_float) > _DERIVED_CONSTRAINT_TOL
        return (NaN, false, θ_min, g_at_min)
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

# ---------------------------------------------------------------------------
# Natural-bounds handling for a derived-quantity profile CI (#142).
#
# A derived quantity often has a natural support: [0, 1] for communality,
# ICC and phylogenetic signal, [-1, 1] for a correlation, [0, Inf) for a
# variance. The bracket-then-bisect search does not know it, so a bound can
# overshoot the support (constrained refits just past the edge still meet
# the 0.05 constraint tolerance), or the search can stop with NaN when the
# profile stays below the cutoff out to the edge. Given the support
# [lo_bound, hi_bound], each side is post-processed:
#   - a finite bound outside the support is set to the edge (boundary);
#   - a NaN bound with a finite edge: the deviance is evaluated AT the
#     edge. At or below the cutoff, the profile is flat to the edge and the
#     bound is the edge (boundary). Above the cutoff, or if the refit at the
#     edge fails, the side stays NaN: the search did not locate the
#     crossing, and no bound is invented for it.
# A finite bound inside the support is returned untouched, so interior
# results do not change. Because the estimate must lie in the support, every
# finite result satisfies lower <= estimate <= upper.
# ---------------------------------------------------------------------------
function _derived_check_bounds(bounds::Tuple{Real, Real}, estimate::Real)
    lo, hi = float(bounds[1]), float(bounds[2])
    lo < hi || throw(ArgumentError(
        "bounds must satisfy lower < upper; got $(bounds)"))
    lo ≤ estimate ≤ hi || throw(ArgumentError(
        "bounds $(bounds) do not contain the estimate $(estimate)"))
    return lo, hi
end

# One side of the natural-bounds post-processing. `D(c)` is the profile
# deviance at c (NaN on a failed refit); `lower_side` says which side `bound`
# and `edge` are on. Returns (bound, at_edge::Bool).
function _derived_bound_side(D::Function, bound::Real, edge::Real,
                             cutoff::Real, lower_side::Bool)
    isfinite(edge) || return (bound, false)
    if isfinite(bound)
        outside = lower_side ? bound < edge : bound > edge
        return outside ? (float(edge), true) : (bound, false)
    end
    D_edge = D(edge)
    (isfinite(D_edge) && D_edge ≤ cutoff) && return (float(edge), true)
    return (bound, false)
end

"""
    profile_ci_derived(fit::GllvmFit, derived_fn::Function;
                       level = 0.95, y = nothing,
                       X = nothing, Σ_phy = nothing,
                       penalty_weight = 1e6,
                       initial_step = nothing,
                       max_expand = 20, max_bisect = 30,
                       bounds = nothing)
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
ci   = GLLVModels.profile_ci_derived(fit, f_c1; y = y, bounds = (0, 1))
```

For a correlation, pass its natural support `bounds = (-1, 1)`:

```julia
f_ρ12 = GLLVModels._make_correlation_closure(spec, 1, 2)
ci    = GLLVModels.profile_ci_derived(fit, f_ρ12; y = y, bounds = (-1, 1))
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

`bounds` (default `nothing`) gives the natural support `(lo, hi)` of the
quantity: `(0, 1)` for a communality, ICC or phylogenetic signal, `(-1, 1)`
for a correlation, `(0, Inf)` for a variance. The search itself does not
change; each side of its result is then checked against the support:
  - a bound outside `[lo, hi]` (a constrained refit just past the edge can
    still meet the constraint tolerance) is set to the edge;
  - a `NaN` bound is checked at the edge: if the deviance at the edge is at
    or below the χ²₁ cutoff (the profile is flat out to the edge), the bound
    is the edge; if it is above the cutoff, or the refit at the edge fails,
    the bound stays `NaN`.
A bound already inside the support is returned unchanged. `bounds` must
contain the estimate, so every finite result satisfies
`lower ≤ estimate ≤ upper`. With `bounds = nothing` no check is made, and a
bound may lie outside the support or be `NaN` on a flat profile.

Returns a NamedTuple with fields:
  - `estimate::Float64` — `g(θ̂)` at the original MLE
  - `lower::Float64`    — lower CI bound (NaN if bracket failed)
  - `upper::Float64`    — upper CI bound (NaN if bracket failed)
  - `method::Symbol`    — `:profile` (both bounds), `:partial`
                          (one side NaN), or `:failed` (both NaN)
  - `boundary::Bool`    — only when `bounds` is given: `true` when a bound
                          was set to an edge of the support
"""
function profile_ci_derived(fit::GllvmFit, derived_fn::Function;
                            level::Real = 0.95,
                            y::Union{Nothing, AbstractMatrix} = nothing,
                            X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                            Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                            penalty_weight::Real = 1e6,
                            initial_step::Union{Nothing, Real} = nothing,
                            max_expand::Integer = 20,
                            max_bisect::Integer = 30,
                            bounds::Union{Nothing, Tuple{Real, Real}} = nothing)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    y === nothing && throw(ArgumentError(
        "profile_ci_derived requires the data matrix `y`"))
    X = _mean_X(fit, X, size(y, 2))

    θ̂ = fit.pars.θ_packed
    g_hat = Float64(derived_fn(θ̂))
    isfinite(g_hat) || throw(ArgumentError(
        "derived_fn returned a non-finite value at the MLE: $g_hat"))
    lo_b, hi_b = bounds === nothing ? (-Inf, Inf) : _derived_check_bounds(bounds, g_hat)

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

    boundary = false
    if bounds !== nothing
        lower, at_lo = _derived_bound_side(deviance_lower, lower, lo_b, cutoff, true)
        upper, at_hi = _derived_bound_side(deviance_upper, upper, hi_b, cutoff, false)
        boundary = at_lo || at_hi
    end

    method = if isnan(lower) && isnan(upper)
        :failed
    elseif isnan(lower) || isnan(upper)
        :partial
    else
        :profile
    end
    r = (lower = lower, upper = upper, estimate = g_hat, method = method)
    return bounds === nothing ? r : merge(r, (; boundary = boundary))
end

# ---------------------------------------------------------------------------
# Feasible-range clamp + boundary flag for an existing profile_ci_derived
# result `r` (computed without `bounds`). Applies the same natural-bounds
# rules as `profile_ci_derived(...; bounds = (lo_bound, hi_bound))` (see
# `_derived_bound_side`), with each edge refit warm-started from θ̂ rather
# than from the search path. profile_ci_total_variance and
# profile_ci_phylo_signal now pass `bounds` directly.
# ---------------------------------------------------------------------------
function _profile_ci_bounded(fit::GllvmFit, derived_fn::Function, r::NamedTuple;
                             level::Real, y::AbstractMatrix,
                             X::Union{Nothing, AbstractArray{<:Real, 3}},
                             Σ_phy::Union{Nothing, AbstractMatrix},
                             lo_bound::Real, hi_bound::Real)
    cutoff = quantile(Chisq(1), level)
    lo_b, hi_b = _derived_check_bounds((lo_bound, hi_bound), r.estimate)
    X = _mean_X(fit, X, size(y, 2))
    D = function(c)
        ll_c, ok, _, _ = _derived_refit_with_fixed(fit, derived_fn, c, y, X, Σ_phy)
        return ok ? 2.0 * (fit.logLik - ll_c) : NaN
    end
    lower, at_lo = _derived_bound_side(D, r.lower, lo_b, cutoff, true)
    upper, at_hi = _derived_bound_side(D, r.upper, hi_b, cutoff, false)
    boundary = at_lo || at_hi

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
bracket expansion gives up. The search therefore runs with
`bounds = (0, Inf)` (see [`profile_ci_derived`](@ref)): any `lower < 0` is
clamped to `0`, and a `NaN` `lower` whose deviance-at-`0` is itself still
below cutoff is reported as `lower = 0` instead (otherwise it stays `NaN`).
Both cases set the additional `boundary::Bool` field on the returned
NamedTuple to `true`.
"""
function profile_ci_total_variance(fit::GllvmFit, t::Integer;
                                   level::Real = 0.95,
                                   y::Union{Nothing, AbstractMatrix} = nothing,
                                   X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                                   Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                                   kwargs...)
    spec = _derived_spec(fit)
    f = _make_total_variance_closure(spec, t)
    return profile_ci_derived(fit, f; level = level, y = y, X = X, Σ_phy = Σ_phy,
                              bounds = (0.0, Inf), kwargs...)
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

`H²[t] ∈ [0, 1]` (see [`phylo_signal`](@ref)): the search runs with
`bounds = (0, 1)` (see [`profile_ci_derived`](@ref)), clamping any bound
outside `[0, 1]` back to the edge and reporting a boundary plateau
(deviance-at-`0`-or-`1` already below the χ²₁ cutoff) as that edge instead
of `NaN`/`:partial`. The returned NamedTuple carries the additional
`boundary::Bool` field.
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
    return profile_ci_derived(fit, f; level = level, y = y, X = X, Σ_phy = Σ_phy,
                              bounds = (0.0, 1.0), kwargs...)
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

function loading_profile(args...; kwargs...)
    Base.depwarn(
        "loading_profile is deprecated: renamed to loading_profile_exploratory; " *
        "the name loading_profile is reserved for a future R-mirroring " *
        "confirmatory (pinned-loadings) surface",
        :loading_profile)
    return loading_profile_exploratory(args...; kwargs...)
end
