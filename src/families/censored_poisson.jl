# Right-censored Poisson family for the generic Laplace core.
#
# Identity: docs/dev-log/decisions/2026-08-15-censored-poisson-identity.md
# (Opus ceiling APPROVED). Twin gllvmTMB exports constructor-only
# `censored_poisson()` — no cpp dens / no runtime admission (FAM-16 blocked).
# Julia-forward engine; light RCall Δ FORBIDDEN.
#
# Estimand (right-censored at C ≥ 1): log P(Y ≥ C | μ) with μ = exp(η).
# Evaluation (stable): logcdf(Gamma(C, 1), μ) — NOT naive log(1 − F(C−1)).
# η-derivatives (hand-coded; AD through logcdf fails):
#   G = μ · pdf(Poisson(μ), C−1) / S,   dℓ/dη = G,   d²ℓ/dη² = G·(C − μ − G).
#
# Interval-ready encoding via (lower, upper) matrices:
#   uncensored:     lower == upper == y
#   right-censored: lower == C, upper == typemax(Int)   (U = +∞)
#   left / interval: deferred — API accepts (L,U) but v1 rejects non-right forms.
#
# Internal N-slot transport into laplace.jl hooks (weight cannot see y):
#   n == 0  ⇒ uncensored (y is the count)
#   n == C ≥ 1 ⇒ right-censored at C (y ignored for the likelihood)

"""
    CensoredPoisson()

Marker for right-censored Poisson (log link on the untruncated mean
`μ = exp(η)`). Distinct from Distributions.jl `Poisson()` and from
zero-truncated Poisson. Twin `gllvmTMB` exposes a constructor only — this
family is **Julia-forward**; do not claim twin fit parity.
"""
struct CensoredPoisson end

_clamp_mu(::CensoredPoisson, μ) = max(μ, 1e-12)

# Stable log survival: log P(Y ≥ C) = logcdf(Gamma(C, 1), μ).
@inline function _censored_poisson_logS(μ, C::Integer)
    C ≥ 1 || throw(ArgumentError("censored_poisson: C ≥ 1 required; got C=$C"))
    return logcdf(Gamma(float(C), 1.0), μ)
end

# G = dℓ/dη = μ · f_{C−1} / S  (log-link natural-parameter score).
@inline function _censored_poisson_G(μ, C::Integer)
    logS = _censored_poisson_logS(μ, C)
    # pdf(Poisson(μ), C−1) in log space then combine — avoids 0/0 when S→0.
    logf = logpdf(Poisson(μ), C - 1)
    return μ * exp(logf - logS)
end

function _glm_logpdf(::CensoredPoisson, μ, n, y)
    ni = Int(n)
    if ni == 0
        return logpdf(Poisson(μ), Int(y))
    end
    return _censored_poisson_logS(μ, ni)
end

function _glm_score(::CensoredPoisson, μ, n, me, y)
    ni = Int(n)
    if ni == 0
        # Uncensored Poisson: (y − μ)/μ · me; log link ⇒ y − μ.
        return (y - μ) / μ * me
    end
    # Hand-coded η-score G; me is unused under LogLink (G already includes μ).
    # For a general link me ≠ μ this would need a chain-rule factor — v1 is LogLink-only.
    return oftype(μ, _censored_poisson_G(μ, ni))
end

function _glm_weight(::CensoredPoisson, μ, n, me)
    ni = Int(n)
    if ni == 0
        return me^2 / μ
    end
    # Observed (−Hessian): −d²ℓ/dη² = −G·(C − μ − G) = G·(G + μ − C).
    G = _censored_poisson_G(μ, ni)
    W = G * (G + μ - ni)
    return max(W, zero(typeof(W)))
end

# This family's `_glm_weight` slot is ALREADY the observed Hessian, hand-derived
# (`−G·(C−μ−G)` on the censored branch; the uncensored branch is Poisson/log,
# where observed ≡ Fisher). It is therefore correct for the log-det as it
# stands, and takes the untouched path under either selector setting.
#
# CAVEAT, recorded rather than hidden: that slot applies `max(W, 0)`. So it is
# observed-CLAMPED, not raw observed. The clamp is pre-existing and is preserved
# here deliberately — routing this family through the generic ForwardDiff
# fallback instead would hand the log-det an unclamped, possibly negative weight
# and change shipped behaviour. Whether the clamp can actually bind for
# reachable (μ, C) is UNVERIFIED.
_glm_weight_matches_observed(::CensoredPoisson, ::LogLink) = true

_laplace_mode_should_backtrack(::CensoredPoisson) = true

"""
    censored_bounds_to_YN(lower, upper) -> (Y, N)

Translate interval-ready `(lower, upper)` bounds into the Laplace `Y`/`N` slot
encoding used by [`CensoredPoisson`](@ref) hooks:

- `lower == upper == y` → uncensored (`N = 0`, `Y = y`)
- `upper == typemax(Int)` and `lower == C ≥ 1` → right-censored at `C`
  (`N = C`, `Y = C`)

Left-censored / finite-interval rows throw (deferred Identity extension).
"""
function censored_bounds_to_YN(lower::AbstractMatrix{<:Integer},
        upper::AbstractMatrix{<:Integer})
    size(lower) == size(upper) || throw(ArgumentError(
        "censored_poisson: lower and upper must share size"))
    p, n = size(lower)
    Y = Matrix{Int}(undef, p, n)
    N = Matrix{Int}(undef, p, n)
    @inbounds for t in 1:p, s in 1:n
        L = Int(lower[t, s])
        U = Int(upper[t, s])
        if L == U
            L < 0 && throw(ArgumentError(
                "censored_poisson: uncensored y ≥ 0 required; got y=$L at ($t,$s)"))
            Y[t, s] = L
            N[t, s] = 0
        elseif U == typemax(Int)
            L < 1 && throw(ArgumentError(
                "censored_poisson: right-censored C ≥ 1 required; got C=$L at ($t,$s) " *
                "(C=0 is uninformative: P(Y≥0)=1)"))
            Y[t, s] = L
            N[t, s] = L
        else
            throw(ArgumentError(
                "censored_poisson v1 supports uncensored (L==U) and right-censored " *
                "(U==typemax(Int)) only; got (L,U)=($L,$U) at ($t,$s)"))
        end
    end
    return Y, N
end

"""
    censored_poisson_marginal_loglik_laplace(Y, N, Λ, β, link=LogLink(); kwargs...)

Laplace log-marginal for a right-censored Poisson GLLVM. `N[t,s] == 0` marks an
uncensored count `Y[t,s]`; `N[t,s] == C ≥ 1` marks a right-censored observation
at limit `C`. Prefer [`censored_bounds_to_YN`](@ref) from `(lower, upper)`.

Only `LogLink` is accepted: the censored η-derivatives in `_glm_score` /
`_glm_weight` are hand-coded against `μ = exp(η)` and silently drop the
chain-rule factor `dμ/dη` for any other link (Identity lock).
"""
function censored_poisson_marginal_loglik_laplace(Y::AbstractMatrix, N::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, link::Link = LogLink(); kwargs...)
    link isa LogLink || throw(ArgumentError(
        "censored_poisson_marginal_loglik_laplace: only LogLink is supported " *
        "(Identity lock — censored η-derivatives hardcode the log link); got " *
        string(nameof(typeof(link)))))
    return marginal_loglik_laplace(CensoredPoisson(), Y, N, Λ, β, link; kwargs...)
end

"""
    CensoredPoissonFit

Result of [`fit_censored_poisson_gllvm`](@ref). Packing is Poisson-identical:
`θ = [β; pack(Λ)]` — no dispersion; censor limits are data, not parameters.
"""
struct CensoredPoissonFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    theta_packed::Vector{Float64}
end

function Base.show(io::IO, f::CensoredPoissonFit)
    p, K = size(f.Λ)
    print(io, "CensoredPoissonFit(p=", p, ", K=", K,
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_censored_poisson_gllvm(Y; K, censored=nothing, lower=nothing, upper=nothing, …)

Fit a right-censored Poisson GLLVM by Laplace + LBFGS (finite-difference outer
gradient). **Julia-forward** — twin constructor-only; no twin Δ.

Censor encoding (interval-ready):

- Preferred: `lower` / `upper` matrices — uncensored `L==U==y`, right-censored
  `L==C`, `U==typemax(Int)`.
- Convenience: `censored::AbstractMatrix{Bool}` with `Y` holding the count or
  limit; `true` ⇒ right-censored at `Y[t,s]`.

Only `LogLink` is supported. Gradient is finite differences only (do not route
through `poisson_laplace_grad`).
"""
function fit_censored_poisson_gllvm(Y::AbstractMatrix; K::Integer,
        link::Link = LogLink(),
        censored = nothing,
        lower = nothing, upper = nothing,
        mask = nothing, offset = nothing,
        β_init = nothing, Λ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    link isa LogLink || throw(ArgumentError(
        "fit_censored_poisson_gllvm: only LogLink is supported (Identity lock)"))
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, mask = mask,
                               caller = "fit_censored_poisson_gllvm")
    rr = rr_theta_len(p, K)

    Yc, Nc = if lower !== nothing || upper !== nothing
        (lower === nothing || upper === nothing) && throw(ArgumentError(
            "censored_poisson: pass both lower and upper, or neither"))
        censored !== nothing && throw(ArgumentError(
            "censored_poisson: pass either (lower,upper) or censored, not both"))
        size(lower) == (p, n) || throw(ArgumentError("lower must be p×n"))
        size(upper) == (p, n) || throw(ArgumentError("upper must be p×n"))
        censored_bounds_to_YN(lower, upper)
    elseif censored !== nothing
        size(censored) == (p, n) || throw(ArgumentError("censored must be p×n"))
        L = Matrix{Int}(undef, p, n)
        U = Matrix{Int}(undef, p, n)
        Yraw = Integer.(_sanitize_missing(Y, 0))
        @inbounds for t in 1:p, s in 1:n
            if censored[t, s]
                C = Int(Yraw[t, s])
                C < 1 && throw(ArgumentError(
                    "censored_poisson: right-censored C ≥ 1; got C=$C at ($t,$s)"))
                L[t, s] = C
                U[t, s] = typemax(Int)
            else
                y = Int(Yraw[t, s])
                y < 0 && throw(ArgumentError(
                    "censored_poisson: uncensored y ≥ 0; got y=$y at ($t,$s)"))
                L[t, s] = y
                U[t, s] = y
            end
        end
        censored_bounds_to_YN(L, U)
    else
        # All-uncensored convenience path (must match Poisson).
        Yraw = Integer.(_sanitize_missing(Y, 0))
        Yraw, zeros(Int, p, n)
    end

    msk = mask === nothing ? (any(ismissing, Y) ? observed_mask(Y) : nothing) : mask

    # Warm start: treat censored cells as observed-at-limit for the link-scale SVD.
    Zemp = [linkfun(link, max(Float64(Yc[t, i]) + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)
    if msk !== nothing
        @inbounds for t in 1:p
            obs = view(msk, t, :)
            cnt = count(obs)
            rowmean = cnt > 0 ? sum(Zemp[t, i] for i in 1:n if msk[t, i]) / cnt : 0.0
            for i in 1:n
                msk[t, i] || (Zemp[t, i] = rowmean)
            end
        end
    end
    β0 = β_init === nothing ? vec(sum(Zemp; dims = 2)) ./ n : collect(float.(β_init))
    Zc = Zemp .- β0
    Λ0 = if Λ_init === nothing
        F = svd(Zc)
        kk = min(K, length(F.S))
        Lmat = zeros(p, K)
        @inbounds for j in 1:kk
            Lmat[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
        end
        Lmat
    else
        collect(float.(Λ_init))
    end

    θ0 = vcat(β0, pack_lambda(Λ0))
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        v = try
            -marginal_loglik_laplace(CensoredPoisson(), Yc, Nc, Λ, β, link;
                                     mask = msk, offset = offset,
                                     maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    return CensoredPoissonFit(β̂, Λ̂, link, _fit_verdict(res)...,
                              collect(Float64, θ̂))
end
