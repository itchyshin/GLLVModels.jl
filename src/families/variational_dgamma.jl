# Gaussian-variational (VA / ELBO) marginal for the Delta-Gamma two-part GLLVM
# (occurrence Bernoulli × positive Gamma, log-link mean). Companion to
# families/twopart.jl (the Laplace path) and families/variational_gamma.jl (the
# closed-form Gamma VA). With the v1 convention Λ_z = 0 the occurrence part is
# CONSTANT in the latent z — π_t = logistic(β^z_t) — so the only z-dependence is
# through the positive part's η^c. The positive Gamma log-density is linear in η^c
# and in e^{−η^c}, both of which have closed-form Gaussian expectations, so (as for
# the Gamma VA) the Delta-Gamma ELBO is CLOSED FORM (no Gauss–Hermite).
#
# Gamma(shape α, scale μ/α), μ = e^{η^c} (E[y|y>0]=μ, Var=μ²/α; matches twopart.jl):
#   log p(y>0 | η, α) = log π + (α−1) log y − y α e^{−η^c} − α η^c + α log α − logΓ(α),
#   log p(y=0)        = log(1 − π).
# Under q(z_s)=N(m_s, diag(v_s)) the positive predictor is η^c_ts ~ N(μη_t, σ²_t),
#   μη_t = β^c_t + (Λc m_s)_t,   σ²_t = Σ_k Λc_tk² v_sk,   E_q[e^{−η^c_t}] = e^{−μη_t+σ²_t/2},
# so the per-trait positive expectation is closed form. Per site,
#   ELBO_s = Σ_t L_t − KL_s,   KL_s = ½ Σ_k (v_sk + m_sk² − 1 − log v_sk),
#   L_t = (y_t==0) ? log(1−π_t)
#                  : log(π_t) + [(α−1) log y_t − y_t α e^{−μη_t+σ²_t/2} − α μη_t + α log α − logΓ(α)].
# As Λc→0 (σ²→0, μη=β^c, optimal q = prior m=0,v=1,KL=0) the ELBO reduces EXACTLY to
# the independent two-part Delta-Gamma loglik. The variational params (m_s, v_s) are
# profiled per site by minimising the negative ELBO over [m; logv] with L-BFGS (K small).

# Per-site Delta-Gamma ELBO at variational params ψ = [m (K); logv (K)].
function _va_site_dgamma_elbo(ψ::AbstractVector, y::AbstractVector,
        Λc::AbstractMatrix, Λc2::AbstractMatrix, βz::AbstractVector,
        βc::AbstractVector, α::Real)
    p, K = size(Λc)
    m  = @view ψ[1:K]
    lv = @view ψ[(K + 1):(2K)]
    v  = exp.(lv)
    σ2 = Λc2 * v
    μη = βc .+ Λc * m
    ℓ = zero(eltype(ψ))
    @inbounds for t in 1:p
        π = inv(one(βz[t]) + exp(-βz[t]))                  # occurrence prob (constant in z)
        π = clamp(π, 1e-12, 1 - 1e-12)
        if y[t] > 0
            w = exp(_clamp_eta(-μη[t] + 0.5 * σ2[t]))      # E_q[e^{−η^c_t}]
            ℓ += log(π) + (α - 1) * log(y[t]) - y[t] * α * w -
                 α * μη[t] + α * log(α) - loggamma(α)
        else
            ℓ += log1p(-π)
        end
    end
    kl = 0.5 * sum(v .+ m .^ 2 .- 1.0 .- lv)
    return ℓ - kl
end

# In-place gradient of the negative per-site Delta-Gamma ELBO at ψ = [m; logv].
# Recomputes w_t = E_q[e^{−η^c_t}] exactly as _va_site_dgamma_elbo does. The
# occurrence part and y=0 species are constant in z, so the sums run ONLY over
# species with y_t>0. The (positive) ELBO gradient is
#   ∂ELBO/∂m_k  = α·Σ_{t:y_t>0} Λc_tk·(y_t·w_t − 1) − m_k
#   ∂ELBO/∂lv_k = −½·α·v_k·Σ_{t:y_t>0} Λc_tk²·y_t·w_t − ½·v_k + ½
# and the objective is −ELBO, so G holds the negation of both. (_clamp_eta's
# derivative is treated as 1; the clamp is inactive for benign data.)
function _va_site_dgamma_grad!(G::AbstractVector, ψ::AbstractVector, y::AbstractVector,
        Λc::AbstractMatrix, Λc2::AbstractMatrix, βz::AbstractVector,
        βc::AbstractVector, α::Real)
    p, K = size(Λc)
    m  = @view ψ[1:K]
    lv = @view ψ[(K + 1):(2K)]
    v  = exp.(lv)
    σ2 = Λc2 * v
    μη = βc .+ Λc * m
    @inbounds for k in 1:K
        gm = zero(eltype(ψ)); gv = zero(eltype(ψ))
        for t in 1:p
            y[t] > 0 || continue
            w = exp(_clamp_eta(-μη[t] + 0.5 * σ2[t]))      # E_q[e^{−η^c_t}]
            gm += Λc[t, k] * (y[t] * w - 1)
            gv += Λc2[t, k] * y[t] * w
        end
        dELBO_dm  = α * gm - m[k]
        dELBO_dlv = -0.5 * α * v[k] * gv - 0.5 * v[k] + 0.5
        G[k]      = -dELBO_dm
        G[K + k]  = -dELBO_dlv
    end
    return G
end

# Profile (m_s, v_s) for one site by minimising the negative ELBO over [m; logv].
function _va_site_dgamma(y::AbstractVector, Λc::AbstractMatrix, Λc2::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector, α::Real;
        maxiter::Integer = 100, tol::Real = 1e-9)
    K = size(Λc, 2)
    negelbo(ψ) = -_va_site_dgamma_elbo(ψ, y, Λc, Λc2, βz, βc, α)
    g!(G, ψ) = _va_site_dgamma_grad!(G, ψ, y, Λc, Λc2, βz, βc, α)
    res = Optim.optimize(negelbo, g!, zeros(2K), Optim.LBFGS(),
                         Optim.Options(g_tol = tol, iterations = maxiter))
    return -Optim.minimum(res)
end

"""
    delta_gamma_marginal_loglik_va(Y, Λc, βz, βc, α; maxiter=100, tol=1e-9) -> Float64

Gaussian-variational (VA) log-marginal lower bound (ELBO) over the `n` sites
(columns) of a Delta-Gamma two-part GLLVM: occurrence probability
`π = logistic(β^z)` (intercept-only, `Λ_z = 0` — constant in the latent z) times a
positive Gamma with mean `μ = exp(β^c + Λ_c z)` and shape `α > 0`. `Y` is p×n with
`0` for absences and positive reals for the positive part, `Λc` is p×K, `βz`/`βc`
length-p. The per-site posterior `q(z_s)=N(m_s, diag(v_s))` is profiled out; the
Delta-Gamma ELBO is closed form (no quadrature). The value is a **lower bound** on
the true log-marginal; as `Λ_c → 0` it equals the independent two-part Delta-Gamma
loglik exactly. Companion to [`delta_gamma_marginal_loglik_laplace`](@ref).
"""
function delta_gamma_marginal_loglik_va(Y::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector, α::Real;
        maxiter::Integer = 100, tol::Real = 1e-9)
    size(Λc, 1) == size(Y, 1) == length(βz) == length(βc) ||
        throw(DimensionMismatch("Λc, Y, βz, βc must share p = $(size(Y,1)) rows"))
    α > 0 || throw(ArgumentError("shape α must be > 0"))
    Λc2 = Λc .^ 2
    acc = 0.0
    @inbounds for s in axes(Y, 2)
        acc += _va_site_dgamma(view(Y, :, s), Λc, Λc2, βz, βc, float(α);
                               maxiter = maxiter, tol = tol)
    end
    return acc
end

"""
    fit_delta_gamma_gllvm_va(Y; K, link=LogLink(), …) -> DeltaGammaFit

Fit a Delta-Gamma two-part GLLVM by maximising the **variational** lower bound
([`delta_gamma_marginal_loglik_va`](@ref)) over `[βz; βc; vec(Λc); log α]` with
L-BFGS, jointly estimating the shape `α` — the VA counterpart of
[`fit_delta_gamma_gllvm`](@ref) (which maximises the Laplace marginal). Same warm
start as the Laplace fit (occurrence logits from the empirical presence rate, `βc`
from `log` mean positives, `Λc` from the SVD of positive-part log-residuals, a
method-of-moments `α₀`) and finite-difference gradient, with the same `log α` clamp
to `[log(1e-3), log(1e3)]` used by [`fit_gamma_gllvm_va`](@ref) to avoid the
catastrophic-cancellation runaway in `α log α − logΓ(α)`. The returned `DeltaGammaFit`'s
`loglik` field holds the maximised ELBO (a lower bound on the true log-marginal).
"""
function fit_delta_gamma_gllvm_va(Y::AbstractMatrix{<:Real}; K::Integer,
        link::Link = LogLink(),
        g_tol::Real = 1e-5, iterations::Integer = 500,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, n = size(Y)
    _check_twopart_support("fit_delta_gamma_gllvm_va", Y, _tp_positive_ok, "0 or a finite positive real")
    rr = rr_theta_len(p, K)

    βz0 = Vector{Float64}(undef, p); βc0 = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        npres = count(>(0), view(Y, t, :))
        pr = clamp((npres + 0.5) / (n + 1), 1e-3, 1 - 1e-3)
        βz0[t] = log(pr / (1 - pr))
        s = 0.0; c = 0
        for j in 1:n
            if Y[t, j] > 0
                s += Y[t, j]; c += 1
            end
        end
        βc0[t] = c == 0 ? 0.0 : log(max(s / c, 1e-6))
    end
    # method-of-moments shape from standardised positives r = y/μ̂ (mean≈1, Var≈1/α)
    sumsq = 0.0; nres = 0
    @inbounds for t in 1:p
        μt = exp(βc0[t])
        for j in 1:n
            if Y[t, j] > 0
                r = Y[t, j] / μt - 1.0; sumsq += r^2; nres += 1
            end
        end
    end
    α0 = nres > 1 ? clamp((nres - 1) / sumsq, 0.1, 100.0) : 1.0
    Zc = [Y[t, j] > 0 ? log(max(Y[t, j], 1e-6)) - βc0[t] : 0.0 for t in 1:p, j in 1:n]
    F = svd(Zc); kk = min(K, length(F.S))
    Λc0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λc0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end

    θ0 = vcat(βz0, βc0, pack_lambda(Λc0), log(α0))
    # Shape bounds: α·logα − logΓ(α) suffers catastrophic cancellation at very large
    # α, which lets the optimiser run α away to nonsense. Confine the shape to a
    # generous, numerically safe range (same trick as fit_gamma_gllvm_va).
    logαlo, logαhi = log(1e-3), log(1e3)
    function negelbo(θ)
        βz = θ[1:p]; βc = θ[(p + 1):(2p)]
        Λc = unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
        α = exp(clamp(θ[2p + rr + 1], logαlo, logαhi))
        v = try
            -delta_gamma_marginal_loglik_va(Y, Λc, βz, βc, α; maxiter = maxiter, tol = tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negelbo, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    βz = θ̂[1:p]; βc = θ̂[(p + 1):(2p)]
    Λc = unpack_lambda(θ̂[(2p + 1):(2p + rr)], p, K)
    α = exp(clamp(θ̂[2p + rr + 1], logαlo, logαhi))
    return DeltaGammaFit(βz, βc, Λc, α, _fit_verdict(res)...)
end
