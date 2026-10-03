# Gaussian-variational (VA / ELBO) marginal for the Negative-Binomial (NB2) GLLVM
# with log link — the non-conjugate companion to the Poisson VA in
# families/variational.jl. Where Poisson admits a closed-form E_q[e^η], the NB2
# log-pmf has no Gaussian-conjugate expectation, so the per-trait expectation
# E_q[log p(y_t | η_t, r)] is computed by 1-D Gauss–Hermite (GH) quadrature.
#
# Model. q(z_s) = N(m_s, diag(v_s)), prior N(0, I_K). Under q the linear predictor
# is Gaussian: η_ts ~ N(μη_t, σ²_t) with
#   μη_t = β_t + (Λ m_s)_t,   σ²_t = Σ_k Λ_tk² v_sk.
# Per site,
#   ELBO_s = Σ_t E_q[log p(y_t | η_t, r)] − KL_s,
#   KL_s   = ½ Σ_k (v_sk + m_sk² − 1 − log v_sk).
# NB2 (same parameterisation as families/negbin.jl), μ = e^η:
#   log p(y|μ,r) = loggamma(y+r) − loggamma(r) − loggamma(y+1)
#                  + r·log(r/(r+μ)) + y·log(μ/(r+μ)).
# The expectation has no closed form, so with g(η) = log p(y_t | μ=e^η, r):
#   E_q[g(η)] ≈ Σ_{j=1}^G (w_j/√π)·g( clamp(μη_t + √(2σ²_t)·x_j) ),
# with (x_j, w_j) the G-point Gauss–Hermite nodes/weights (Σ w = √π).
#
# As Λ→0 (⇒ σ²=0) the GH rule collapses to g(μη_t)=g(β_t) and the optimal q is the
# prior (m=0, v=1, KL=0), so the ELBO reduces EXACTLY to the independent-NB loglik.
# As r→∞ the NB2 log-pmf → Poisson, so this marginal → the Poisson VA marginal.

# `_gauss_hermite(G)` is shared with the other VA families (defined in
# families/variational.jl, included first).

# NB2 conditional log-pmf at η (μ = e^η), shared with families/negbin.jl's form.
@inline function _nb_logpmf_eta(η, y, r)
    μ = exp(η)
    return loggamma(y + r) - loggamma(r) - loggamma(y + 1) +
           r * log(r / (r + μ)) + y * log(μ / (r + μ))
end

# Per-site NB2 ELBO at variational params packed as ψ = [m (K); logv (K)].
# Returns ELBO_s = Σ_t E_q[log p(y_t|η_t,r)] − KL_s, with E_q by GH quadrature.
function _va_site_negbin_elbo(ψ::AbstractVector, y::AbstractVector,
        Λ::AbstractMatrix, Λ2::AbstractMatrix, β::AbstractVector,
        r::Real, x::AbstractVector, w::AbstractVector)
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
            et += w[j] * _nb_logpmf_eta(η, y[t], r)
        end
        ℓ += invsqrtpi * et
    end
    kl = 0.5 * sum(v .+ m .^ 2 .- 1.0 .- lv)
    return ℓ - kl
end

# In-place gradient of the negative per-site NB2 ELBO at ψ = [m (K); logv (K)].
# Recomputes the GH nodes η_{tj} EXACTLY as _va_site_negbin_elbo does (same (x,w),
# same _clamp_eta). With g(η) = log p(y|μ=e^η,r) the η-derivative is
#   ℓ'(y,η) = y − μ·(r+y)/(r+μ),  μ = e^η,
# and the generic GH chain rule gives
#   a_t = Σ_j (w_j/√π)·ℓ'(y_t,η_{tj}),  b_t = Σ_j (w_j/√π)·ℓ'(y_t,η_{tj})·x_j,
#   ∂E_q_t/∂m_k  = Λ_tk·a_t,
#   ∂E_q_t/∂lv_k = (Λ_tk²·v_k / √(2σ²_t))·b_t   (0 when σ²_t==0).
# With KL = ½ Σ_k (v_k + m_k² − 1 − lv_k),
#   ∂ELBO/∂m_k  = Σ_t Λ_tk·a_t − m_k,
#   ∂ELBO/∂lv_k = Σ_t (Λ_tk²·v_k/√(2σ²_t))·b_t − ½(v_k − 1).
# The objective is −ELBO, so G holds the negation. (_clamp_eta's derivative ≈ 1.)
function _va_site_negbin_grad!(G::AbstractVector, ψ::AbstractVector, y::AbstractVector,
        Λ::AbstractMatrix, Λ2::AbstractMatrix, β::AbstractVector,
        r::Real, x::AbstractVector, w::AbstractVector)
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
        for j in eachindex(x)
            η = _clamp_eta(μη[t] + sd * x[j])
            μ = exp(η)
            ℓp = y[t] - μ * (r + y[t]) / (r + μ)    # ℓ'(y,η) = y − μ(r+y)/(r+μ)
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
function _va_site_negbin(y::AbstractVector, Λ::AbstractMatrix, Λ2::AbstractMatrix,
        β::AbstractVector, r::Real, x::AbstractVector, w::AbstractVector;
        maxiter::Integer = 100, tol::Real = 1e-9)
    K = size(Λ, 2)
    negelbo(ψ) = -_va_site_negbin_elbo(ψ, y, Λ, Λ2, β, r, x, w)
    g!(G, ψ) = _va_site_negbin_grad!(G, ψ, y, Λ, Λ2, β, r, x, w)
    ψ0 = zeros(2K)
    res = Optim.optimize(negelbo, g!, ψ0, Optim.LBFGS(),
                         Optim.Options(g_tol = tol, iterations = maxiter))
    return -Optim.minimum(res)
end

# Sibling of `_va_site_negbin` that ALSO returns the converged variational params.
# Returns (ELBO_s, m*, v*) where m*, v* are length-K vectors (v on the natural,
# not log, scale). Used by the envelope-theorem outer gradient: at the inner
# optimum ∂ELBO/∂(m,v)=0, so dELBO/dθ = ∂ELBO/∂θ holding (m*,v*) fixed.
function _va_site_negbin_mv(y::AbstractVector, Λ::AbstractMatrix, Λ2::AbstractMatrix,
        β::AbstractVector, r::Real, x::AbstractVector, w::AbstractVector;
        maxiter::Integer = 100, tol::Real = 1e-9)
    K = size(Λ, 2)
    negelbo(ψ) = -_va_site_negbin_elbo(ψ, y, Λ, Λ2, β, r, x, w)
    g!(G, ψ) = _va_site_negbin_grad!(G, ψ, y, Λ, Λ2, β, r, x, w)
    ψ0 = zeros(2K)
    res = Optim.optimize(negelbo, g!, ψ0, Optim.LBFGS(),
                         Optim.Options(g_tol = tol, iterations = maxiter))
    ψ̂ = Optim.minimizer(res)
    m = ψ̂[1:K]
    v = exp.(ψ̂[(K + 1):(2K)])
    return (-Optim.minimum(res), m, v)
end

# One full inner-solve pass over all `n` sites for given (Λ, β, r): returns the
# total ELBO together with the converged variational means/variances stacked as
# K×n matrices M, V (columns = sites). This is the single pass the envelope-theorem
# outer gradient consumes — both objective and gradient come out of one solve.
function _va_negbin_solve_all(Y::AbstractMatrix, Λ::AbstractMatrix, Λ2::AbstractMatrix,
        β::AbstractVector, r::Real, x::AbstractVector, w::AbstractVector;
        maxiter::Integer = 100, tol::Real = 1e-9)
    K = size(Λ, 2)
    n = size(Y, 2)
    M = Matrix{Float64}(undef, K, n)
    V = Matrix{Float64}(undef, K, n)
    acc = 0.0
    @inbounds for s in 1:n
        elbo_s, m_s, v_s = _va_site_negbin_mv(view(Y, :, s), Λ, Λ2, β, float(r),
                                              x, w; maxiter = maxiter, tol = tol)
        acc += elbo_s
        M[:, s] = m_s
        V[:, s] = v_s
    end
    return acc, M, V
end

# Envelope-theorem outer gradient of the TOTAL ELBO over θ = [β; pack_lambda(Λ); log r],
# given the converged (M, V) from `_va_negbin_solve_all`. Returns the gradient of
# the OBJECTIVE (−ELBO) in θ-packed layout, written into `G`.
#
# At the inner optimum the (m,v) partials vanish, so the total θ-gradient equals the
# partial ∂ELBO/∂θ at fixed (m*,v*). The KL term is θ-independent and drops out.
# With per-(t,s) GH-weighted sums (μ=e^η):
#   a_ts = Σ_j (w_j/√π)·ℓ'_η,   ℓ'_η = y − μ(r+y)/(r+μ)
#   b_ts = Σ_j (w_j/√π)·ℓ'_η·x_j
#   c_ts = Σ_j (w_j/√π)·ℓ'_r,   ℓ'_r = ψ(y+r) − ψ(r) + log(r/(r+μ)) + 1 − (r+y)/(r+μ)
#   ∂ELBO/∂β_t  = Σ_s a_ts
#   ∂ELBO/∂Λ_tk = Σ_s [ a_ts·M[k,s] + b_ts·(2·Λ_tk·V[k,s]/√(2σ²_ts)) ]   (σ²=0 ⇒ 0)
#   ∂ELBO/∂r    = Σ_s Σ_t c_ts,   ∂ELBO/∂(log r) = r·∂ELBO/∂r
function _va_negbin_outer_grad!(G::AbstractVector, Y::AbstractMatrix,
        Λ::AbstractMatrix, Λ2::AbstractMatrix, β::AbstractVector, r::Real,
        M::AbstractMatrix, V::AbstractMatrix, x::AbstractVector, w::AbstractVector)
    p, K = size(Λ)
    n = size(Y, 2)
    invsqrtpi = 1.0 / sqrt(pi)
    gβ = zeros(Float64, p)
    gΛ = zeros(Float64, p, K)
    gr = 0.0
    @inbounds for s in 1:n
        m_s = view(M, :, s)
        v_s = view(V, :, s)
        σ2 = Λ2 * v_s
        μη = β .+ Λ * m_s
        for t in 1:p
            σ2t = σ2[t]
            sd = sqrt(2.0 * max(σ2t, 0.0))
            at = 0.0; bt = 0.0; ct = 0.0
            for j in eachindex(x)
                η = _clamp_eta(μη[t] + sd * x[j])
                μ = exp(η)
                rpμ = r + μ
                ℓη = Y[t, s] - μ * (r + Y[t, s]) / rpμ
                ℓr = digamma(Y[t, s] + r) - digamma(r) + log(r / rpμ) +
                     1.0 - (r + Y[t, s]) / rpμ
                wj = w[j]
                at += wj * ℓη
                bt += wj * ℓη * x[j]
                ct += wj * ℓr
            end
            a_ts = invsqrtpi * at
            b_ts = invsqrtpi * bt
            c_ts = invsqrtpi * ct
            gβ[t] += a_ts
            gr += c_ts
            for k in 1:K
                gΛ[t, k] += a_ts * m_s[k]
                if σ2t > 0
                    gΛ[t, k] += b_ts * (2.0 * Λ[t, k] * v_s[k] / sd)
                end
            end
        end
    end
    # Objective is −ELBO ⇒ negate. log-r chain rule: ∂/∂(log r) = r·∂/∂r.
    rr = rr_theta_len(p, K)
    @inbounds for t in 1:p
        G[t] = -gβ[t]
    end
    gΛpacked = pack_lambda(gΛ)
    @inbounds for i in 1:rr
        G[p + i] = -gΛpacked[i]
    end
    G[p + rr + 1] = -(r * gr)
    return G
end

"""
    nb_marginal_loglik_va(Y, Λ, β, r; maxiter=100, tol=1e-9, gh=20) -> Float64

Gaussian-variational (VA) log-marginal lower bound (ELBO) over the `n` sites
(columns) of a negative-binomial (NB2) GLLVM with log link — `Y` the p×n integer
count matrix, `Λ` p×K, `β` length-p, dispersion `r > 0` (`Var = μ + μ²/r`). The
per-site variational posterior `q(z_s)=N(m_s, diag(v_s))` is profiled out by jointly
minimising the negative ELBO over `[m; logv]`; the per-trait expectation is
evaluated by `gh`-point Gauss–Hermite quadrature. The returned value is a **lower
bound** on the true log-marginal (≤ it for any `q`). As `Λ→0` it equals the
independent-NB loglik exactly; as `r→∞` it tends to the Poisson VA marginal
([`poisson_marginal_loglik_va`](@ref)). Companion to
[`nb_marginal_loglik_laplace`](@ref).
"""
function nb_marginal_loglik_va(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, r::Real; maxiter::Integer = 100, tol::Real = 1e-9,
        gh::Integer = 20)
    size(Λ, 1) == size(Y, 1) == length(β) ||
        throw(DimensionMismatch("Λ, Y, β must share p = $(size(Y,1)) rows"))
    r > 0 || throw(ArgumentError("dispersion r must be > 0"))
    Λ2 = Λ .^ 2
    x, w = _gauss_hermite(gh)
    acc = 0.0
    @inbounds for s in axes(Y, 2)
        acc += _va_site_negbin(view(Y, :, s), Λ, Λ2, β, float(r), x, w;
                               maxiter = maxiter, tol = tol)
    end
    return acc
end

"""
    fit_nb_gllvm_va(Y; K, link=LogLink(), …) -> NBFit

Fit a negative-binomial (NB2) GLLVM by maximising the **variational** lower bound
([`nb_marginal_loglik_va`](@ref)) over `[β; vec(Λ); log r]` with L-BFGS, jointly
estimating the dispersion `r` — the VA counterpart of [`fit_nb_gllvm`](@ref)
(which maximises the Laplace marginal). `Y` is a p×n integer count matrix; `K` the
latent dimension. Same warm start as the Laplace driver (empirical log-mean
intercepts + SVD loadings + a moderate `r₀`) and finite-difference gradient. The
returned `NBFit`'s `loglik` field holds the maximised ELBO (a lower bound on the
true log-marginal), so it is directly comparable across VA fits but sits slightly
below the Laplace `loglik` for the same data. The ELBO is derived for the log link
only: any other `link` throws `ArgumentError`.
"""
function fit_nb_gllvm_va(Y::AbstractMatrix{<:Integer}; K::Integer,
        link::Link = LogLink(),
        g_tol::Real = 1e-5, iterations::Integer = 500,
        maxiter::Integer = 100, tol::Real = 1e-9)
    _va_require_canonical_link("fit_nb_gllvm_va", link, LogLink())
    p, n = size(Y)
    rr = rr_theta_len(p, K)

    Zemp = [linkfun(link, max(Y[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc)
    kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    logr0 = log(10.0)

    θ0 = vcat(β0, pack_lambda(Λ0), logr0)
    x, w = _gauss_hermite(20)
    # Combined objective/gradient: ONE inner-solve pass per evaluation. The outer
    # gradient is exact via the envelope theorem (∂ELBO/∂(m,v)=0 at the inner
    # optimum), eliminating the finite-difference factor of ~2·length(θ).
    function fg!(F, G, θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        Λ2 = Λ .^ 2
        r = exp(θ[p + rr + 1])
        elbo, M, V = _va_negbin_solve_all(Y, Λ, Λ2, β, r, x, w;
                                          maxiter = maxiter, tol = tol)
        if G !== nothing
            _va_negbin_outer_grad!(G, Y, Λ, Λ2, β, r, M, V, x, w)
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
    r̂ = exp(θ̂[p + rr + 1])
    return NBFit(β̂, Λ̂, r̂, link, _fit_verdict(res)...)
end
