# Gaussian-variational (VA / EVA-style) marginal for the non-Gaussian GLLVM.
#
# Companion to families/laplace.jl. Where Laplace plugs in the posterior mode and a
# Hessian curvature, VA fits a Gaussian posterior q(z_s) = N(m_s, diag(v_s)) per
# site against the evidence lower bound (ELBO) — a guaranteed lower bound on the
# true log-marginal, and gllvm's default estimator (more stable than Laplace on the
# dispersion/multimodal cells documented in ROADMAP).
#
# Under q the linear predictor is Gaussian: η_ts ~ N(μ_ts, σ²_ts) with
#   μ_ts = β_t + (Λ m_s)_t,   σ²_ts = Σ_k Λ_tk² v_sk.
# Per site,
#   ELBO_s = Σ_t E_q[log p(y_ts | η_ts)] − KL_s,
#   KL_s   = ½ Σ_k (v_sk + m_sk² − 1 − log v_sk).
# For the Poisson/log family the expectation is closed-form (η ~ N(μ,σ²) ⇒
# E_q[e^η] = e^{μ+σ²/2}):
#   E_q[log p] = Σ_t [ y_t μ_t − e^{μ_t+σ²_t/2} − log Γ(y_t+1) ].
#
# This file implements the Poisson VA marginal (increment 1). The variational
# params are profiled per site by coordinate ascent: a Newton step on m (the ELBO
# is concave in m), then the exact stationary update for v,
#   v_k = 1 / (1 + Σ_t Λ_tk² λ_t),   λ_t = e^{μ_t+σ²_t/2},
# which keeps v ∈ (0, 1] (posterior variance ≤ the unit prior). As Λ→0 the solver
# returns m=0, v=1 and the ELBO reduces EXACTLY to the independent-Poisson loglik.

"""
    _gauss_hermite(G) -> (x, w)

`G`-point Gauss–Hermite nodes `x` and weights `w` (for the weight `e^{-t²}`, so
`∫ e^{-t²} f(t) dt ≈ Σ_j w_j f(x_j)` and `Σ_j w_j = √π`) via the Golub–Welsch
algorithm: the nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix
with zero diagonal and off-diagonals `√(j/2)`, and the weights are `√π` times the
squared first components of the corresponding eigenvectors. Shared by the
non-conjugate VA families (Binomial, NB) for their `E_q` quadrature.
"""
function _gauss_hermite(G::Integer)
    G ≥ 1 || throw(ArgumentError("_gauss_hermite: need G ≥ 1, got $G"))
    G == 1 && return ([0.0], [sqrt(π)])
    off = [sqrt(j / 2) for j in 1:(G - 1)]
    E = eigen(SymTridiagonal(zeros(G), off))
    return (E.values, sqrt(π) .* (E.vectors[1, :] .^ 2))
end

# Per-site Poisson ELBO at the profiled variational optimum (m_s, v_s).
function _va_site_poisson(y::AbstractVector, Λ::AbstractMatrix, Λ2::AbstractMatrix,
        β::AbstractVector; maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    m = zeros(K)
    v = ones(K)
    for _ in 1:maxiter
        σ2 = Λ2 * v
        μ  = β .+ Λ * m
        λ  = exp.(_clamp_eta.(μ .+ 0.5 .* σ2))
        # Newton step on m: gradient Λ'(y−λ)−m, negative Hessian Λ'diag(λ)Λ + I.
        g  = Λ' * (y .- λ) .- m
        A  = Symmetric(Λ' * (λ .* Λ) + I)
        Δ  = _safe_solve(A, g)
        (Δ === nothing || !all(isfinite, Δ)) && break
        m  = m .+ Δ
        # exact stationary update for v at the new m.
        σ2 = Λ2 * v
        μ  = β .+ Λ * m
        λ  = exp.(_clamp_eta.(μ .+ 0.5 .* σ2))
        vnew = 1.0 ./ (1.0 .+ (Λ2' * λ))
        dv = maximum(abs, vnew .- v)
        v  = vnew
        (maximum(abs, Δ) < tol && dv < tol) && break
    end
    σ2 = Λ2 * v
    μ  = β .+ Λ * m
    λ  = exp.(_clamp_eta.(μ .+ 0.5 .* σ2))
    ℓ = 0.0
    @inbounds for t in 1:p
        ℓ += y[t] * μ[t] - λ[t] - loggamma(y[t] + 1)
    end
    kl = 0.5 * sum(v .+ m .^ 2 .- 1.0 .- log.(v))
    return ℓ - kl
end

"""
    poisson_marginal_loglik_va(Y, Λ, β; maxiter=100, tol=1e-9) -> Float64

Gaussian-variational (VA) log-marginal lower bound (ELBO) over the `n` sites
(columns) of a Poisson GLLVM with log link — `Y` the p×n integer count matrix, `Λ`
p×K, `β` length-p. The per-site variational posterior `q(z_s)=N(m_s, diag(v_s))` is
profiled out by coordinate ascent. The returned value is a **lower bound** on the
true log-marginal (≤ it for any `q`); as `Λ→0` it equals the independent-Poisson
loglik exactly. Companion to [`poisson_marginal_loglik_laplace`](@ref).
"""
function poisson_marginal_loglik_va(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector; maxiter::Integer = 100, tol::Real = 1e-9)
    size(Λ, 1) == size(Y, 1) == length(β) ||
        throw(DimensionMismatch("Λ, Y, β must share p = $(size(Y,1)) rows"))
    Λ2 = Λ .^ 2
    acc = 0.0
    @inbounds for s in axes(Y, 2)
        acc += _va_site_poisson(view(Y, :, s), Λ, Λ2, β; maxiter = maxiter, tol = tol)
    end
    return acc
end

# The VA (ELBO) objectives are closed-form / Gauss-Hermite expressions written for ONE
# link each (logit for Beta and Binomial, log for the rest) and carry no `link` argument.
# The fitters keep a `link` keyword for signature parity with the Laplace fitters, but any
# other link used to be accepted and silently ignored: the fit optimised the canonical-link
# ELBO, then stored the requested link on the result. Refuse it instead.
function _va_require_canonical_link(fname::AbstractString, link::Link, canonical::Link)
    typeof(link) === typeof(canonical) || throw(ArgumentError(
        "$fname: the variational (ELBO) objective is derived for the " *
        "$(nameof(typeof(canonical))) only; got link = $(nameof(typeof(link))). " *
        "Use the Laplace fitter for other links, or omit `link`."))
    return nothing
end

"""
    fit_poisson_gllvm_va(Y; K, β_init=nothing, Λ_init=nothing, …) -> PoissonFit

Fit a Poisson GLLVM by maximising the **variational** lower bound
([`poisson_marginal_loglik_va`](@ref)) over `[β; vec(Λ)]` with L-BFGS — the VA
counterpart of [`fit_poisson_gllvm`](@ref) (which maximises the Laplace marginal).
Same warm start (empirical log-mean intercepts + SVD loadings) and finite-difference
gradient. The returned `PoissonFit`'s `loglik` field holds the maximised ELBO (a
lower bound on the true log-marginal), so it is directly comparable across VA fits
but sits slightly below the Laplace `loglik` for the same data.
"""
function fit_poisson_gllvm_va(Y::AbstractMatrix{<:Integer}; K::Integer,
        β_init = nothing, Λ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, n = size(Y)
    rr = rr_theta_len(p, K)
    link = LogLink()

    Zemp = [linkfun(link, max(Y[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    β0 = β_init === nothing ? vec(sum(Zemp; dims = 2)) ./ n : collect(float.(β_init))
    Λ0 = if Λ_init === nothing
        Zc = Zemp .- β0
        F = svd(Zc); kk = min(K, length(F.S))
        L = zeros(p, K)
        @inbounds for j in 1:kk
            L[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
        end
        L
    else
        collect(float.(Λ_init))
    end

    θ0 = vcat(β0, pack_lambda(Λ0))
    function negelbo(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        v = try
            -poisson_marginal_loglik_va(Y, Λ, β; maxiter = maxiter, tol = tol)
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
    return PoissonFit(β̂, Λ̂, link, _fit_verdict(res)...)
end

