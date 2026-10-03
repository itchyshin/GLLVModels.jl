# Gaussian-variational (VA / ELBO) marginal for the Beta GLLVM with logit link.
# The non-conjugate companion to families/beta.jl's Laplace marginal. The
# per-trait expectation E_q[log p(y_t | η_t, φ)] is computed by 1-D Gauss–Hermite
# (GH) quadrature, exactly mirroring the Negative-Binomial VA path.
#
# Model. q(z_s) = N(m_s, diag(v_s)), prior N(0, I_K). Under q the linear predictor
# is Gaussian: η_ts ~ N(μη_t, σ²_t) with
#   μη_t = β_t + (Λ m_s)_t,   σ²_t = Σ_k Λ_tk² v_sk.
# Per site,
#   ELBO_s = Σ_t E_q[log p(y_t | η_t, φ)] − KL_s,
#   KL_s   = ½ Σ_k (v_sk + m_sk² − 1 − log v_sk).
# Beta (same parameterisation as families/beta.jl), μ = logistic(η):
#   log p(y|μ,φ) = loggamma(φ) − loggamma(μφ) − loggamma((1−μ)φ)
#                  + (μφ−1)·log y + ((1−μ)φ−1)·log(1−y).
# The expectation has no closed form, so with g(η) = log p(y_t | μ=logistic(η), φ):
#   E_q[g(η)] ≈ Σ_{j=1}^G (w_j/√π)·g( _clamp_eta(μη_t + √(2σ²_t)·x_j) ),
# with (x_j, w_j) the G-point Gauss–Hermite nodes/weights (Σ w = √π). Inside g, μ
# is clamped into (1e-12, 1−1e-12) before the logs/loggammas.
#
# As Λ→0 (⇒ σ²=0) the GH rule collapses to g(μη_t)=g(β_t) and the optimal q is the
# prior (m=0, v=1, KL=0), so the ELBO reduces EXACTLY to the independent-Beta loglik.

# `_gauss_hermite(G)` is shared with the other VA families (defined in
# families/variational.jl, included first).

# Beta conditional log-density at η (μ = logistic(η)), shared form with beta.jl.
@inline function _beta_logpdf_eta(η, y, φ)
    μ = clamp(1.0 / (1.0 + exp(-η)), 1e-12, 1 - 1e-12)
    return loggamma(φ) - loggamma(μ * φ) - loggamma((1 - μ) * φ) +
           (μ * φ - 1) * log(y) + ((1 - μ) * φ - 1) * log1p(-y)
end

# Per-site Beta ELBO at variational params packed as ψ = [m (K); logv (K)].
# Returns ELBO_s = Σ_t E_q[log p(y_t|η_t,φ)] − KL_s, with E_q by GH quadrature.
function _va_site_beta_elbo(ψ::AbstractVector, y::AbstractVector,
        Λ::AbstractMatrix, Λ2::AbstractMatrix, β::AbstractVector,
        φ::Real, x::AbstractVector, w::AbstractVector)
    p, K = size(Λ)
    m  = @view ψ[1:K]
    lv = @view ψ[(K + 1):(2K)]
    v  = exp.(lv)
    σ2 = Λ2 * v
    μη = β .+ Λ * m
    G  = length(x)
    invsqrtpi = 1.0 / sqrt(pi)
    ℓ = zero(eltype(ψ))
    @inbounds for t in 1:p
        sd = sqrt(2.0 * σ2[t])
        et = zero(eltype(ψ))
        for j in 1:G
            η = _clamp_eta(μη[t] + sd * x[j])
            et += w[j] * _beta_logpdf_eta(η, y[t], φ)
        end
        ℓ += invsqrtpi * et
    end
    kl = 0.5 * sum(v .+ m .^ 2 .- 1.0 .- lv)
    return ℓ - kl
end

# In-place gradient of the negative per-site Beta ELBO at ψ = [m (K); logv (K)].
# Recomputes the GH nodes η_{tj} EXACTLY as _va_site_beta_elbo does (same (x,w),
# same _clamp_eta, same μ-clamp into (1e-12,1−1e-12)). With g(η)=log p(y|μ,φ),
# μ=logistic(η), and dμ/dη = μ(1−μ), the η-derivative is
#   ℓ'(y,η) = μ(1−μ)·φ·[digamma((1−μ)φ) − digamma(μφ) + log y − log(1−y)],
# and the generic GH chain rule gives
#   a_t = Σ_j (w_j/√π)·ℓ'(y_t,η_{tj}),  b_t = Σ_j (w_j/√π)·ℓ'(y_t,η_{tj})·x_j,
#   ∂E_q_t/∂m_k  = Λ_tk·a_t,
#   ∂E_q_t/∂lv_k = (Λ_tk²·v_k / √(2σ²_t))·b_t   (0 when σ²_t==0).
# With KL = ½ Σ_k (v_k + m_k² − 1 − lv_k),
#   ∂ELBO/∂m_k  = Σ_t Λ_tk·a_t − m_k,
#   ∂ELBO/∂lv_k = Σ_t (Λ_tk²·v_k/√(2σ²_t))·b_t − ½(v_k − 1).
# The objective is −ELBO, so G holds the negation. (_clamp_eta's derivative ≈ 1.)
function _va_site_beta_grad!(G::AbstractVector, ψ::AbstractVector, y::AbstractVector,
        Λ::AbstractMatrix, Λ2::AbstractMatrix, β::AbstractVector,
        φ::Real, x::AbstractVector, w::AbstractVector)
    p, K = size(Λ)
    m  = @view ψ[1:K]
    lv = @view ψ[(K + 1):(2K)]
    v  = exp.(lv)
    σ2 = Λ2 * v
    μη = β .+ Λ * m
    invsqrtpi = 1.0 / sqrt(pi)
    a = zeros(eltype(ψ), p)
    b = zeros(eltype(ψ), p)
    @inbounds for t in 1:p
        sd = sqrt(2.0 * max(σ2[t], 0.0))
        at = zero(eltype(ψ)); bt = zero(eltype(ψ))
        ly  = log(y[t]); l1y = log1p(-y[t])
        for j in eachindex(x)
            η = _clamp_eta(μη[t] + sd * x[j])
            μ = clamp(1.0 / (1.0 + exp(-η)), 1e-12, 1 - 1e-12)
            ℓp = μ * (1 - μ) * φ *
                 (digamma((1 - μ) * φ) - digamma(μ * φ) + ly - l1y)
            at += w[j] * ℓp
            bt += w[j] * ℓp * x[j]
        end
        a[t] = invsqrtpi * at
        b[t] = invsqrtpi * bt
    end
    @inbounds for k in 1:K
        gm = zero(eltype(ψ)); gv = zero(eltype(ψ))
        for t in 1:p
            gm += Λ[t, k] * a[t]
            if σ2[t] > 0
                gv += (Λ2[t, k] * v[k] / sqrt(2.0 * σ2[t])) * b[t]
            end
        end
        dELBO_dm  = gm - m[k]
        dELBO_dlv = gv - 0.5 * (v[k] - 1)
        G[k]      = -dELBO_dm
        G[K + k]  = -dELBO_dlv
    end
    return G
end

# Profile (m_s, v_s) for one site by jointly minimising the negative ELBO over
# [m (K); logv (K)] with L-BFGS (analytic gradient), from m=0, logv=0.
function _va_site_beta(y::AbstractVector, Λ::AbstractMatrix, Λ2::AbstractMatrix,
        β::AbstractVector, φ::Real, x::AbstractVector, w::AbstractVector;
        maxiter::Integer = 100, tol::Real = 1e-9)
    K = size(Λ, 2)
    negelbo(ψ) = -_va_site_beta_elbo(ψ, y, Λ, Λ2, β, φ, x, w)
    g!(G, ψ) = _va_site_beta_grad!(G, ψ, y, Λ, Λ2, β, φ, x, w)
    ψ0 = zeros(2K)
    res = Optim.optimize(negelbo, g!, ψ0, Optim.LBFGS(),
                         Optim.Options(g_tol = tol, iterations = maxiter))
    return -Optim.minimum(res)
end

# Sibling of `_va_site_beta` that ALSO returns the converged variational params.
# Returns (ELBO_s, m*, v*) where m*, v* are length-K vectors (v on the natural,
# not log, scale). Used by the envelope-theorem outer gradient: at the inner
# optimum ∂ELBO/∂(m,v)=0, so dELBO/dθ = ∂ELBO/∂θ holding (m*,v*) fixed.
function _va_site_beta_mv(y::AbstractVector, Λ::AbstractMatrix, Λ2::AbstractMatrix,
        β::AbstractVector, φ::Real, x::AbstractVector, w::AbstractVector;
        maxiter::Integer = 100, tol::Real = 1e-9)
    K = size(Λ, 2)
    negelbo(ψ) = -_va_site_beta_elbo(ψ, y, Λ, Λ2, β, φ, x, w)
    g!(G, ψ) = _va_site_beta_grad!(G, ψ, y, Λ, Λ2, β, φ, x, w)
    ψ0 = zeros(2K)
    res = Optim.optimize(negelbo, g!, ψ0, Optim.LBFGS(),
                         Optim.Options(g_tol = tol, iterations = maxiter))
    ψ̂ = Optim.minimizer(res)
    m = ψ̂[1:K]
    v = exp.(ψ̂[(K + 1):(2K)])
    return (-Optim.minimum(res), m, v)
end

# One full inner-solve pass over all `n` sites for given (Λ, β, φ): returns the
# total ELBO together with the converged variational means/variances stacked as
# K×n matrices M, V (columns = sites). This is the single pass the envelope-theorem
# outer gradient consumes — both objective and gradient come out of one solve.
function _va_beta_solve_all(Y::AbstractMatrix, Λ::AbstractMatrix, Λ2::AbstractMatrix,
        β::AbstractVector, φ::Real, x::AbstractVector, w::AbstractVector;
        maxiter::Integer = 100, tol::Real = 1e-9)
    K = size(Λ, 2)
    n = size(Y, 2)
    M = Matrix{Float64}(undef, K, n)
    V = Matrix{Float64}(undef, K, n)
    acc = 0.0
    @inbounds for s in 1:n
        elbo_s, m_s, v_s = _va_site_beta_mv(view(Y, :, s), Λ, Λ2, β, float(φ),
                                            x, w; maxiter = maxiter, tol = tol)
        acc += elbo_s
        M[:, s] = m_s
        V[:, s] = v_s
    end
    return acc, M, V
end

# Envelope-theorem outer gradient of the TOTAL ELBO over θ = [β; pack_lambda(Λ); log φ],
# given the converged (M, V) from `_va_beta_solve_all`. Returns the gradient of
# the OBJECTIVE (−ELBO) in θ-packed layout, written into `G`.
#
# At the inner optimum the (m,v) partials vanish, so the total θ-gradient equals the
# partial ∂ELBO/∂θ at fixed (m*,v*). The KL term is θ-independent and drops out.
# With μ = logistic(η) (clamped into (1e-12,1−1e-12)) and per-(t,s) GH-weighted sums:
#   ℓ'_η = μ(1−μ)·φ·[digamma((1−μ)φ) − digamma(μφ) + log y − log(1−y)]
#   ℓ'_φ = digamma(φ) − μ·digamma(μφ) − (1−μ)·digamma((1−μ)φ) + μ·log y + (1−μ)·log(1−y)
#   a_ts = Σ_j (w_j/√π)·ℓ'_η,   b_ts = Σ_j (w_j/√π)·ℓ'_η·x_j,   c_ts = Σ_j (w_j/√π)·ℓ'_φ
#   ∂ELBO/∂β_t  = Σ_s a_ts
#   ∂ELBO/∂Λ_tk = Σ_s [ a_ts·M[k,s] + b_ts·(2·Λ_tk·V[k,s]/√(2σ²_ts)) ]   (σ²=0 ⇒ 0)
#   ∂ELBO/∂φ    = Σ_s Σ_t c_ts,   ∂ELBO/∂(log φ) = φ·∂ELBO/∂φ
function _va_beta_outer_grad!(G::AbstractVector, Y::AbstractMatrix,
        Λ::AbstractMatrix, Λ2::AbstractMatrix, β::AbstractVector, φ::Real,
        M::AbstractMatrix, V::AbstractMatrix, x::AbstractVector, w::AbstractVector)
    p, K = size(Λ)
    n = size(Y, 2)
    invsqrtpi = 1.0 / sqrt(pi)
    gβ = zeros(Float64, p)
    gΛ = zeros(Float64, p, K)
    gφ = 0.0
    @inbounds for s in 1:n
        m_s = view(M, :, s)
        v_s = view(V, :, s)
        σ2 = Λ2 * v_s
        μη = β .+ Λ * m_s
        for t in 1:p
            σ2t = σ2[t]
            sd = sqrt(2.0 * max(σ2t, 0.0))
            ly  = log(Y[t, s]); l1y = log1p(-Y[t, s])
            at = 0.0; bt = 0.0; ct = 0.0
            for j in eachindex(x)
                η = _clamp_eta(μη[t] + sd * x[j])
                μ = clamp(1.0 / (1.0 + exp(-η)), 1e-12, 1 - 1e-12)
                dgμ  = digamma(μ * φ)
                dg1μ = digamma((1 - μ) * φ)
                ℓη = μ * (1 - μ) * φ * (dg1μ - dgμ + ly - l1y)
                ℓφ = digamma(φ) - μ * dgμ - (1 - μ) * dg1μ + μ * ly + (1 - μ) * l1y
                wj = w[j]
                at += wj * ℓη
                bt += wj * ℓη * x[j]
                ct += wj * ℓφ
            end
            a_ts = invsqrtpi * at
            b_ts = invsqrtpi * bt
            c_ts = invsqrtpi * ct
            gβ[t] += a_ts
            gφ += c_ts
            for k in 1:K
                gΛ[t, k] += a_ts * m_s[k]
                if σ2t > 0
                    gΛ[t, k] += b_ts * (2.0 * Λ[t, k] * v_s[k] / sd)
                end
            end
        end
    end
    # Objective is −ELBO ⇒ negate. log-φ chain rule: ∂/∂(log φ) = φ·∂/∂φ.
    rr = rr_theta_len(p, K)
    @inbounds for t in 1:p
        G[t] = -gβ[t]
    end
    gΛpacked = pack_lambda(gΛ)
    @inbounds for i in 1:rr
        G[p + i] = -gΛpacked[i]
    end
    G[p + rr + 1] = -(φ * gφ)
    return G
end

"""
    beta_marginal_loglik_va(Y, Λ, β, φ; maxiter=100, tol=1e-9, gh=20) -> Float64

Gaussian-variational (VA) log-marginal lower bound (ELBO) over the `n` sites
(columns) of a Beta GLLVM with logit link — `Y` the p×n matrix of proportions in
(0,1), `Λ` p×K, `β` length-p, precision `φ > 0` (mean `μ = logistic(η)`,
per-observation `Beta(μφ, (1−μ)φ)`, `Var = μ(1−μ)/(1+φ)`). The per-site variational
posterior `q(z_s)=N(m_s, diag(v_s))` is profiled out by jointly minimising the
negative ELBO over `[m; logv]`; the per-trait expectation is evaluated by `gh`-point
Gauss–Hermite quadrature. The returned value is a **lower bound** on the true
log-marginal (≤ it for any `q`); as `Λ→0` it equals the independent-Beta loglik
exactly. Companion to [`beta_marginal_loglik_laplace`](@ref).
"""
function beta_marginal_loglik_va(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, φ::Real; maxiter::Integer = 100, tol::Real = 1e-9,
        gh::Integer = 20)
    size(Λ, 1) == size(Y, 1) == length(β) ||
        throw(DimensionMismatch("Λ, Y, β must share p = $(size(Y,1)) rows"))
    φ > 0 || throw(ArgumentError("precision φ must be > 0"))
    Λ2 = Λ .^ 2
    x, w = _gauss_hermite(gh)
    acc = 0.0
    @inbounds for s in axes(Y, 2)
        acc += _va_site_beta(view(Y, :, s), Λ, Λ2, β, float(φ), x, w;
                             maxiter = maxiter, tol = tol)
    end
    return acc
end

"""
    fit_beta_gllvm_va(Y; K, link=LogitLink(), …) -> BetaFit

Fit a Beta GLLVM by maximising the **variational** lower bound
([`beta_marginal_loglik_va`](@ref)) over `[β; vec(Λ); log φ]` with L-BFGS, jointly
estimating the precision `φ` — the VA counterpart of [`fit_beta_gllvm`](@ref)
(which maximises the Laplace marginal). `Y` is a p×n matrix of proportions in
(0,1); `K` the latent dimension. Same warm start as the Laplace driver (empirical
logit-mean intercepts + SVD loadings + a moderate `φ₀`). The outer gradient is
exact via the envelope theorem (∂ELBO/∂(m,v)=0 at the inner optimum), so each
L-BFGS step costs ONE inner-solve pass rather than a finite-difference sweep. The
returned `BetaFit`'s `loglik` field holds the maximised ELBO (a lower bound on the
true log-marginal), so it sits slightly below the Laplace `loglik` for the same data.
The ELBO is derived for the logit link only: any other `link` throws `ArgumentError`.
"""
function fit_beta_gllvm_va(Y::AbstractMatrix{<:Real}; K::Integer,
        link::Link = LogitLink(),
        β_init = nothing, Λ_init = nothing, φ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        maxiter::Integer = 100, tol::Real = 1e-9)
    _va_require_canonical_link("fit_beta_gllvm_va", link, LogitLink())
    p, n = size(Y)
    rr = rr_theta_len(p, K)

    Zemp = [linkfun(link, clamp(float(Y[t, i]), 1e-6, 1 - 1e-6)) for t in 1:p, i in 1:n]
    β0 = β_init === nothing ? vec(sum(Zemp; dims = 2)) ./ n : collect(float.(β_init))
    Λ0 = if Λ_init === nothing
        Zc = Zemp .- β0
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
    logφ0 = φ_init === nothing ? log(10.0) : log(float(φ_init))

    θ0 = vcat(β0, pack_lambda(Λ0), logφ0)
    x, w = _gauss_hermite(20)
    # Combined objective/gradient: ONE inner-solve pass per evaluation. The outer
    # gradient is exact via the envelope theorem (∂ELBO/∂(m,v)=0 at the inner
    # optimum), eliminating the finite-difference factor of ~2·length(θ).
    function fg!(F, G, θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        Λ2 = Λ .^ 2
        φ = exp(θ[p + rr + 1])
        elbo, M, V = _va_beta_solve_all(Y, Λ, Λ2, β, φ, x, w;
                                        maxiter = maxiter, tol = tol)
        if G !== nothing
            _va_beta_outer_grad!(G, Y, Λ, Λ2, β, φ, M, V, x, w)
        end
        if F !== nothing
            return isfinite(elbo) ? -elbo : 1e12
        end
        return nothing
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(Optim.only_fg!(fg!), θ0, ls,
                         Optim.Options(g_tol = g_tol, iterations = iterations))
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    φ̂ = exp(θ̂[p + rr + 1])
    return BetaFit(β̂, Λ̂, φ̂, link, _fit_verdict(res)...)
end
