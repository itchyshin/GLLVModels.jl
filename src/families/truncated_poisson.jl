# Zero-truncated Poisson family for the generic Laplace core.
#
# Twin gllvmTMB family_id 10 (`truncated_poisson()`, log link; y ≥ 1 strictly):
#   ℓ = log Poisson(y; μ) − log(1 − e^{−μ}),   μ = exp(η), η = β + Λz.
# Score / weight (log link: me = μ) match the hurdle positive-block formulas:
#   μ_tr = μ / (1 − e^{−μ}),   s = y − μ_tr,   W = μ_tr (1 + μ − μ_tr).
# Cite: gllvmTMB `src/gllvmTMB.cpp` fid==10; `R/families.R` truncated_poisson().

"""
    TruncatedPoisson()

Marker for zero-truncated Poisson (support `{1,2,…}`; log link on the
*untruncated* mean `μ = exp(η)`). Distinct from Distributions.jl `Poisson()` and
from the hurdle occurrence×truncated two-part family.
"""
struct TruncatedPoisson end

_clamp_mu(::TruncatedPoisson, μ) = max(μ, 1e-12)

# Truncated mean / variance helpers (untruncated μ > 0).
function _truncpois_mean_var(μ)
    p0 = exp(-μ)
    denom = 1 - p0
    μtr = μ / denom
    var_tr = μtr * (1 + μ - μtr)
    return μtr, var_tr
end

# General-link score: ∂ℓ/∂η = [(y − μ_tr)/μ] · me. Log link ⇒ me = μ ⇒ y − μ_tr.
function _glm_score(::TruncatedPoisson, μ, n, me, y)
    μtr, _ = _truncpois_mean_var(μ)
    return (y - μtr) / μ * me
end

# Observed ≡ Fisher here, verified by expansion: var_tr = μtr(1+μ−μtr) gives
# exactly μ[(1−p₀)−μp₀]/(1−p₀)² = μ·dμtr/dμ, which is y-free. Zero-truncation
# does not reintroduce a y-dependence, so this family is unaffected by the
# selector.
_glm_weight_matches_observed(::TruncatedPoisson, ::LogLink) = true

function _glm_weight(::TruncatedPoisson, μ, n, me)
    _, var_tr = _truncpois_mean_var(μ)
    return (me / μ)^2 * var_tr
end

function _glm_logpdf(::TruncatedPoisson, μ, n, y)
    yi = Int(y)
    yi < 1 && return oftype(μ, -Inf)
    # log(1 − e^{−μ}); stable for μ > 0 via log1p(-exp(-μ)) when μ is not tiny.
    log_nz = μ < 1e-8 ? log(μ) : log1p(-exp(-μ))   # 1−e^{−μ} ∼ μ as μ→0
    return logpdf(Poisson(μ), yi) - log_nz
end

# Enable Fisher-scoring backtracking like Poisson.
_laplace_mode_should_backtrack(::TruncatedPoisson) = true

"""
    truncated_poisson_marginal_loglik_laplace(Y, Λ, β, link=LogLink(); kwargs...) -> Float64

Laplace log-marginal for a zero-truncated Poisson GLLVM. `Y` must be integer
counts with every observed cell `≥ 1` (zeros are invalid under this family).
"""
truncated_poisson_marginal_loglik_laplace(Y::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, link::Link = LogLink(); kwargs...) =
    marginal_loglik_laplace(TruncatedPoisson(), Y, ones(Int, size(Y)), Λ, β, link; kwargs...)

"""
    TruncatedPoissonFit

Result of [`fit_truncated_poisson_gllvm`](@ref).
"""
struct TruncatedPoissonFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    theta_packed::Vector{Float64}
end

function Base.show(io::IO, f::TruncatedPoissonFit)
    p, K = size(f.Λ)
    print(io, "TruncatedPoissonFit(p=", p, ", K=", K,
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_truncated_poisson_gllvm(Y; K, link=LogLink(), …) -> TruncatedPoissonFit

Fit a zero-truncated Poisson GLLVM by Laplace + LBFGS (finite-difference
outer gradient). Twin-aligned: log link on untruncated `μ`, support `y ≥ 1`.
Throws if any observed cell is `< 1`.
"""
function fit_truncated_poisson_gllvm(Y::AbstractMatrix; K::Integer,
        link::Link = LogLink(), mask = nothing, offset = nothing,
        β_init = nothing, Λ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    link isa LogLink || throw(ArgumentError(
        "fit_truncated_poisson_gllvm: only LogLink is supported (twin truncated_poisson)"))
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, mask = mask,
                               caller = "fit_truncated_poisson_gllvm")
    rr = rr_theta_len(p, K)
    msk = mask === nothing ? (any(ismissing, Y) ? observed_mask(Y) : nothing) : mask
    Yc = Integer.(_sanitize_missing(Y, 1))   # placeholder 1 for masked (never enters ℓ)
    # Reject zeros in observed cells.
    @inbounds for t in 1:p, s in 1:n
        (msk !== nothing && !msk[t, s]) && continue
        Yc[t, s] < 1 && throw(ArgumentError(
            "truncated_poisson requires y ≥ 1; found y=$(Yc[t, s]) at ($t,$s)"))
    end

    Zemp = [linkfun(link, max(Float64(Yc[t, i]), 1.0)) for t in 1:p, i in 1:n]
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
        L = zeros(p, K)
        @inbounds for j in 1:kk
            L[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
        end
        L
    else
        collect(float.(Λ_init))
    end

    θ0 = vcat(β0, pack_lambda(Λ0))
    N1 = ones(Int, size(Yc))
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        v = try
            -marginal_loglik_laplace(TruncatedPoisson(), Yc, N1, Λ, β, link;
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
    return TruncatedPoissonFit(β̂, Λ̂, link, _fit_verdict(res)...,
                               collect(Float64, θ̂))
end
