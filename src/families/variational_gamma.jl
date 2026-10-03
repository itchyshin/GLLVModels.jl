# Gaussian-variational (VA / ELBO) marginal for the Gamma GLLVM (log link, shape α).
# Companion to families/variational.jl. The Gamma log-density is linear in η and in
# e^{-η}, both of which have closed-form Gaussian expectations, so — like Poisson,
# and unlike Binomial/NB — the VA ELBO is CLOSED FORM (no Gauss–Hermite needed).
#
# Gamma(shape α, scale μ/α), μ = e^η (so E[y]=μ, Var=μ²/α; matches families/gamma.jl):
#   log p(y|η,α) = (α−1) log y − y α e^{−η} − α η + α log α − logΓ(α).
# Under q(z_s)=N(m_s, diag(v_s)) the predictor is η_ts ~ N(μη_t, σ²_t),
#   μη_t = β_t + (Λ m_s)_t,   σ²_t = Σ_k Λ_tk² v_sk,   E_q[e^{−η_t}] = e^{−μη_t+σ²_t/2},
# so the per-trait expectation is closed-form:
#   E_q[log p(y_t)] = (α−1) log y_t − y_t α e^{−μη_t+σ²_t/2} − α μη_t + α log α − logΓ(α).
# Per site,
#   ELBO_s = Σ_t E_q[log p(y_t)] − KL_s,   KL_s = ½ Σ_k (v_sk + m_sk² − 1 − log v_sk).
# As Λ→0 (σ²→0, μη=β, optimal q = prior m=0,v=1,KL=0) the ELBO reduces EXACTLY to the
# independent-Gamma loglik. The variational params (m_s, v_s) are profiled per site by
# minimising the negative ELBO over [m; logv] with L-BFGS (K small).

# Per-site Gamma ELBO at variational params ψ = [m (K); logv (K)].
function _va_site_gamma_elbo(ψ::AbstractVector, y::AbstractVector,
        Λ::AbstractMatrix, Λ2::AbstractMatrix, β::AbstractVector, α::Real)
    p, K = size(Λ)
    m  = @view ψ[1:K]
    lv = @view ψ[(K + 1):(2K)]
    v  = exp.(lv)
    σ2 = Λ2 * v
    μη = β .+ Λ * m
    ℓ = zero(eltype(ψ))
    @inbounds for t in 1:p
        w = exp(_clamp_eta(-μη[t] + 0.5 * σ2[t]))          # E_q[e^{−η_t}]
        ℓ += (α - 1) * log(y[t]) - y[t] * α * w - α * μη[t] + α * log(α) - loggamma(α)
    end
    kl = 0.5 * sum(v .+ m .^ 2 .- 1.0 .- lv)
    return ℓ - kl
end

# In-place gradient of the negative per-site Gamma ELBO at ψ = [m (K); logv (K)].
# Recomputes w_t = E_q[e^{−η_t}] exactly as _va_site_gamma_elbo does. The (positive)
# ELBO gradient is
#   ∂ELBO/∂m_k  = α·Σ_t Λ_tk·(y_t·w_t − 1) − m_k
#   ∂ELBO/∂lv_k = −½·α·v_k·Σ_t Λ_tk²·y_t·w_t − ½·v_k + ½
# and the objective is −ELBO, so G holds the negation of both. (_clamp_eta's
# derivative is treated as 1; the clamp is inactive for benign data.)
function _va_site_gamma_grad!(G::AbstractVector, ψ::AbstractVector, y::AbstractVector,
        Λ::AbstractMatrix, Λ2::AbstractMatrix, β::AbstractVector, α::Real)
    p, K = size(Λ)
    m  = @view ψ[1:K]
    lv = @view ψ[(K + 1):(2K)]
    v  = exp.(lv)
    σ2 = Λ2 * v
    μη = β .+ Λ * m
    @inbounds for k in 1:K
        gm = zero(eltype(ψ)); gv = zero(eltype(ψ))
        for t in 1:p
            w = exp(_clamp_eta(-μη[t] + 0.5 * σ2[t]))      # E_q[e^{−η_t}]
            gm += Λ[t, k] * (y[t] * w - 1)
            gv += Λ2[t, k] * y[t] * w
        end
        dELBO_dm  = α * gm - m[k]
        dELBO_dlv = -0.5 * α * v[k] * gv - 0.5 * v[k] + 0.5
        G[k]      = -dELBO_dm
        G[K + k]  = -dELBO_dlv
    end
    return G
end

# Profile (m_s, v_s) for one site by minimising the negative ELBO over [m; logv].
function _va_site_gamma(y::AbstractVector, Λ::AbstractMatrix, Λ2::AbstractMatrix,
        β::AbstractVector, α::Real; maxiter::Integer = 100, tol::Real = 1e-9)
    K = size(Λ, 2)
    negelbo(ψ) = -_va_site_gamma_elbo(ψ, y, Λ, Λ2, β, α)
    g!(G, ψ) = _va_site_gamma_grad!(G, ψ, y, Λ, Λ2, β, α)
    res = Optim.optimize(negelbo, g!, zeros(2K), Optim.LBFGS(),
                         Optim.Options(g_tol = tol, iterations = maxiter))
    return -Optim.minimum(res)
end

"""
    gamma_marginal_loglik_va(Y, Λ, β, α; maxiter=100, tol=1e-9) -> Float64

Gaussian-variational (VA) log-marginal lower bound (ELBO) over the `n` sites
(columns) of a Gamma GLLVM with log link and shape `α > 0` — `Y` the p×n positive
matrix, `Λ` p×K, `β` length-p. The per-site posterior `q(z_s)=N(m_s, diag(v_s))` is
profiled out; the Gamma ELBO is closed-form (no quadrature). The value is a **lower
bound** on the true log-marginal; as `Λ→0` it equals the independent-Gamma loglik
exactly. Companion to [`gamma_marginal_loglik_laplace`](@ref).
"""
function gamma_marginal_loglik_va(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, α::Real; maxiter::Integer = 100, tol::Real = 1e-9)
    size(Λ, 1) == size(Y, 1) == length(β) ||
        throw(DimensionMismatch("Λ, Y, β must share p = $(size(Y,1)) rows"))
    α > 0 || throw(ArgumentError("shape α must be > 0"))
    Λ2 = Λ .^ 2
    acc = 0.0
    @inbounds for s in axes(Y, 2)
        acc += _va_site_gamma(view(Y, :, s), Λ, Λ2, β, float(α); maxiter = maxiter, tol = tol)
    end
    return acc
end

"""
    fit_gamma_gllvm_va(Y; K, link=LogLink(), …) -> GammaFit

Fit a Gamma GLLVM by maximising the **variational** lower bound
([`gamma_marginal_loglik_va`](@ref)) over `[β; vec(Λ); log α]` with L-BFGS, jointly
estimating the shape `α` — the VA counterpart of [`fit_gamma_gllvm`](@ref) (which
maximises the Laplace marginal). Same warm start (log row-mean intercepts + SVD
loadings + `α₀=2`) and finite-difference gradient. The returned `GammaFit`'s
`loglik` field holds the maximised ELBO (a lower bound on the true log-marginal).
The ELBO is derived for the log link only: any other `link` throws `ArgumentError`.
"""
function fit_gamma_gllvm_va(Y::AbstractMatrix{<:Real}; K::Integer,
        link::Link = LogLink(),
        g_tol::Real = 1e-5, iterations::Integer = 500,
        maxiter::Integer = 100, tol::Real = 1e-9)
    _va_require_canonical_link("fit_gamma_gllvm_va", link, LogLink())
    p, n = size(Y)
    rr = rr_theta_len(p, K)

    Zemp = log.(max.(Y, 1e-6))
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc)
    kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    logα0 = log(2.0)

    θ0 = vcat(β0, pack_lambda(Λ0), logα0)
    # Shape bounds: α·logα − logΓ(α) suffers catastrophic cancellation at very large
    # α, which lets the optimiser run α away to nonsense (e.g. 1e125). Confine the
    # shape to a generous, numerically safe range; the true shape is always inside it.
    logαlo, logαhi = log(1e-3), log(1e3)
    function negelbo(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        α = exp(clamp(θ[p + rr + 1], logαlo, logαhi))
        v = try
            -gamma_marginal_loglik_va(Y, Λ, β, α; maxiter = maxiter, tol = tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negelbo, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    α̂ = exp(clamp(θ̂[p + rr + 1], logαlo, logαhi))
    return GammaFit(β̂, Λ̂, α̂, link, _fit_verdict(res)...)
end
