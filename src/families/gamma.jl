# Gamma (positive continuous) family pieces for the generic Laplace core
# (src/families/laplace.jl). y_t > 0; mean μ = exp(η) (log link), shape α;
# the per-observation law is Gamma(shape α, scale μ/α), so E[y] = μ and
# Var = μ²/α. The shape `α` is carried in the family marker `Gamma(α, ·)` —
# only its `α` field is read.
#
# Score/weight wrt η (Gamma GLM, variance function V(μ) = μ²/α):
#   s = α (y − μ) / μ² · dμ/dη
#   W = α (dμ/dη)² / μ²          (expected information ⇒ W ≥ 0)
_clamp_mu(::Gamma, μ) = max(μ, 1e-12)
_glm_score(f::Gamma, μ, n, me, y) = f.α * (y - μ) / μ^2 * me
_glm_weight(f::Gamma, μ, n, me)   = f.α * me^2 / μ^2
_glm_logpdf(f::Gamma, μ, n, y)    = logpdf(Gamma(f.α, μ / f.α), y)

# ---------------------------------------------------------------------------
# Log-det curvature default for Gamma/log (2026-08-25). PER-FAMILY, deliberately
# not a global flip.
#
# TMB Laplaces with the OBSERVED joint Hessian, which for Gamma at the log link
# is `α·y/μ` — not the Fisher weight `α`, which is exactly its expectation
# (substitute y = E[y] = μ). This was instance 8 of the curvature fault class,
# and the worst-placed one: `fit_gllvm(Y; family = Gamma())` reaches it, because
# Gamma is deliberately excluded from the `disp_group = :species` auto-coerce at
# `fit_gllvm.jl:141-145` ("Gamma unchanged").
#
# Flipped for Gamma ALONE, on family-specific evidence rather than as part of a
# global default change: measured over 12 seeds against 8001-node quadrature,
# `:observed` is closer 12/12, with 20-60× smaller error. The same measurement
# says Beta is slightly WORSE and GP-1 substantially worse, so those keep the
# Fisher default until they have evidence of their own.
#
# This obliges `laplace_grad.jl`'s Gamma log-det weight to move in the same
# commit — otherwise the analytic gradient stops being the gradient of this
# objective and degrades silently rather than erroring.
_default_hessian(::Gamma, ::LogLink) = :observed

# Analytic observed curvature for Gamma/log: −∂²ℓ/∂η² = α·y/μ.
#
# The generic nested-ForwardDiff fallback reproduces this to ~1e-16 relative
# (locked in the oracle tests), but "to 1e-16" is not "identically". The shared
# path and the grouped path must agree EXACTLY, because
# `test_grouped_dispersion_beta_gamma.jl:49` pins "constant αvec ≡ shared α" at
# atol 1e-10 — and AD-vs-closed-form round-off, accumulated over p×n cells and
# through a log-det, lands at ~1.3e-10. That is round-off, not a semantic
# difference, but the correct response is to make both paths use ONE formula
# rather than to widen a tolerance.
#
# Same reasoning as the DeltaGamma fix, which delegates to this same weight
# rather than re-deriving it: one formula, one place. Delegating here keeps that
# property and takes the AD call out of the hot path as a side benefit.
_glm_obs_weight(f::Gamma, μ, n, me, y, link::LogLink, η) =
    _gamma_grouped_laplace_weight(:observed, f, μ, me, y, link)

"""
    gamma_marginal_loglik_laplace(Y, Λ, β, α; link=LogLink(), kwargs...) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a Gamma GLLVM with
shape `α` — responses `Y > 0`, mean `μ = exp(η)` (log link), per-observation
`Gamma(α, μ/α)` (`Var = μ²/α`). A thin wrapper over the family-generic
`marginal_loglik_laplace` with the `Gamma(α, ·)` marker.
"""
gamma_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector,
        α::Real; link::Link = LogLink(), kwargs...) =
    marginal_loglik_laplace(Gamma(float(α), 1.0), Y, ones(Int, size(Y)), Λ, β, link; kwargs...)

# ---------------------------------------------------------------------------
# Fit driver (Gamma family slice 2).
# ---------------------------------------------------------------------------

"""
    GammaFit

Result of [`fit_gamma_gllvm`](@ref): intercepts `β` (length p), loadings `Λ` (p×K),
the estimated shape `α` (Var = μ²/α), the `link`, the maximised Laplace
`loglik`, the optimiser `converged` flag, and `iterations`.
"""
struct GammaFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    α::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    alpha_lv::Union{Nothing, Matrix{Float64}}
    theta_packed::Vector{Float64}
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
end

# Positional compatibility constructor (2026-08-28): every pre-existing
# construction site builds a default-curvature fit; the `hessian` field
# records the objective identity so `confint`/bootstrap can rebuild THE
# SAME objective instead of guessing (the audit's confint-consistency class).
GammaFit(β, Λ, α, link, loglik, converged, iterations, alpha_lv, theta_packed) =
    GammaFit(β, Λ, α, link, loglik, converged, iterations, alpha_lv, theta_packed, _default_hessian(Gamma(1.0, 1.0), link))

GammaFit(β::Vector{Float64}, Λ::Matrix{Float64}, α::Float64, link::Link,
         loglik::Float64, converged::Bool, iterations::Int) =
    GammaFit(β, Λ, α, link, loglik, converged, iterations, nothing, Float64[])

function Base.show(io::IO, f::GammaFit)
    p, K = size(f.Λ)
    print(io, "GammaFit(p=", p, ", K=", K, ", α=", round(f.α; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          f.alpha_lv === nothing ? "" : ", X_lv=true",
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    gamma_lv_nll_packed(params, Y, p, K, link; X_lv, q_lv, kwargs...) -> Real

Negative Laplace log-likelihood for the predictor-informed latent-score Gamma
model. Parameter layout: `params[1:p]` = β; next `q_lv*K` = `alpha_lv`; next
`rr` = packed loadings `Λ`; last entry = `log α` (shape). The predictor mean
enters the Laplace core as the offset `Λ * alpha_lv' * X_lv[s]`.
"""
function gamma_lv_nll_packed(params::AbstractVector, Y::AbstractMatrix,
        p::Integer, K::Integer, link::Link;
        X_lv::AbstractMatrix, q_lv::Integer,
        mask = nothing, offset = nothing,
        maxiter::Integer = 100, tol::Real = 1e-9)
    size(Y, 1) == p ||
        throw(ArgumentError("Y first dim ($(size(Y, 1))) must equal p ($p)"))
    n = size(Y, 2)
    size(X_lv, 1) == n ||
        throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n_sites ($n)"))
    size(X_lv, 2) == q_lv ||
        throw(ArgumentError("X_lv second dim ($(size(X_lv, 2))) must equal q_lv ($q_lv)"))
    q_lv > 0 || throw(ArgumentError("q_lv must be positive"))

    rr = rr_theta_len(p, K)
    n_expected = p + q_lv * K + rr + 1
    length(params) == n_expected || throw(ArgumentError(
        "params length ($(length(params))) must equal $n_expected " *
        "(p=$p + alpha_lv=$(q_lv * K) + rr=$rr + log_alpha=1)"))

    cursor = 0
    β = @view params[(cursor + 1):(cursor + p)]
    cursor += p
    alpha_vec = @view params[(cursor + 1):(cursor + q_lv * K)]
    alpha_lv = reshape(alpha_vec, q_lv, K)
    cursor += q_lv * K
    θ_rr = @view params[(cursor + 1):(cursor + rr)]
    Λ = unpack_lambda(θ_rr, p, K)
    cursor += rr
    α = exp(params[cursor + 1])

    lv_offset = _lv_mean_eta(Λ, X_lv, alpha_lv)
    off = offset === nothing ? lv_offset : offset .+ lv_offset
    return -gamma_marginal_loglik_laplace(Y, Λ, β, α; link = link, mask = mask, offset = off,
                                          maxiter = maxiter, tol = tol)
end

"""
    fit_gamma_gllvm(Y; K, link=LogLink(), α_init=nothing, …) -> GammaFit

Fit a Gamma GLLVM by L-BFGS over `[β; vec(Λ); log α]` on the Laplace marginal
(`gamma_marginal_loglik_laplace`), jointly estimating the shape `α`
(`Var = μ²/α`). `Y` is a p×n matrix of positive reals; `K` the latent
dimension. Analytic gradient on the no-mask/no-offset path with the log link, with a
finite-difference fallback (masked or offset fits, and any other `link`, use finite
differences of the objective for that link); warm start = link-scale row-means as
intercepts (log row-means for the default log link) + SVD of the row-centred
link-scale Y as loadings + `logα₀ = log(2.0)`.

`hessian` selects the Laplace log-det curvature only (`:fisher` expected /
`:observed` joint — TMB's choice); the inner mode search is always
Fisher-scored. Default: Gamma/log default `:observed` (2026-08-25 flip). Omitting it is exactly the pre-kwarg
behaviour.
"""
function fit_gamma_gllvm(Y::AbstractMatrix; K::Integer,
        link::Link = LogLink(), mask = nothing, offset = nothing,
        gradient::Symbol = :analytic,
        hessian::Symbol = _default_hessian(Gamma(1.0, 1.0), link),
        β_init = nothing, Λ_init = nothing, α_init = nothing,
        X_lv::Union{Nothing, AbstractMatrix} = nothing,
        alpha_lv_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    rr = rr_theta_len(p, K)
    hessian in (:fisher, :observed) || throw(ArgumentError(
        "fit_gamma_gllvm: hessian must be :fisher or :observed; got :$hessian"))
    if X_lv !== nothing && hessian !== _default_hessian(Gamma(1.0, 1.0), link)
        throw(ArgumentError("fit_gamma_gllvm: a non-default hessian is not yet supported " *
                            "together with X_lv (the packed X_lv objective does not thread it)"))
    end

    # Predictor-informed latent-score mean (Design 73 / gllvmTMB C1): X_lv (n×q_lv)
    # activates joint estimation of alpha_lv with the shared shape α, via the
    # parameter-dependent offset Λ * alpha_lv' * X_lv[s]. Point-estimate only.
    q_lv = 0
    X_lv_fit = nothing
    if X_lv !== nothing
        K > 0 || throw(ArgumentError("X_lv requires a positive latent dimension K"))
        size(X_lv, 1) == n ||
            throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n_sites ($n)"))
        q_lv = size(X_lv, 2)
        q_lv > 0 || throw(ArgumentError("X_lv must have at least one predictor column"))
        X_lv_fit = Matrix{Float64}(X_lv)
        if alpha_lv_init !== nothing
            size(alpha_lv_init, 1) == q_lv ||
                throw(ArgumentError(
                    "alpha_lv_init first dim ($(size(alpha_lv_init, 1))) must equal size(X_lv, 2) ($q_lv)"))
            size(alpha_lv_init, 2) == K ||
                throw(ArgumentError(
                    "alpha_lv_init second dim ($(size(alpha_lv_init, 2))) must equal K ($K)"))
        end
    elseif alpha_lv_init !== nothing
        throw(ArgumentError("alpha_lv_init requires X_lv"))
    end

    msk = _resolve_obs_mask(mask, Y)                  # NA handling
    Yc  = _sanitize_missing(Y, 1.0)                   # positive placeholder

    # link-scale warm start (η = g(y)); for the default log link this is exactly `log.(max.(Yc, 1e-6))`
    Zemp = [linkfun(link, max(Yc[t, i], 1e-6)) for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)           # offset (η = β + offset + Λz)
    _mask_warmstart!(Zemp, msk)
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
    logα0 = α_init === nothing ? log(2.0) : log(float(α_init))

    # alpha_lv warm start: least-squares regression of the initial PPCA scores on X_lv.
    alpha0 = if X_lv_fit === nothing
        nothing
    elseif alpha_lv_init === nothing
        F = svd(Zc)
        kk = min(K, length(F.S))
        scores0 = zeros(Float64, n, K)
        @inbounds for j in 1:kk
            scores0[:, j] = sqrt(n) .* F.V[:, j]
        end
        X_lv_fit \ scores0
    else
        Matrix{Float64}(alpha_lv_init)
    end

    θ0 = vcat(β0, pack_lambda(Λ0), logα0)
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        α = exp(θ[p + rr + 1])
        v = try
            -gamma_marginal_loglik_laplace(Yc, Λ, β, α; link = link, mask = msk,
                                           offset = offset, hessian = hessian,
                                           maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = if X_lv_fit !== nothing
        θ0_lv = vcat(β0, vec(alpha0), pack_lambda(Λ0), logα0)
        negll_lv = θ -> begin
            v = try
                gamma_lv_nll_packed(θ, Yc, p, K, link;
                                    X_lv = X_lv_fit, q_lv = q_lv,
                                    mask = msk, offset = offset,
                                    maxiter = newton_maxiter, tol = newton_tol)
            catch
                return 1e12
            end
            return isfinite(v) ? v : 1e12
        end
        Optim.optimize(negll_lv, θ0_lv, ls, opts; autodiff = :finite)
    elseif gradient === :analytic && offset === nothing && link isa LogLink &&
           hessian === _default_hessian(Gamma(1.0, 1.0), link)
        # `gamma_laplace_grad` is derived for the log link only; any other link takes the
        # :finite path below, which differentiates the actual objective.
        ag = θ -> begin
            β = θ[1:p]; Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K); av = exp(θ[p + rr + 1])
            try -gamma_laplace_grad(Yc, Λ, β, av; mask = msk) catch; nothing end
        end
        _optimize_with_analytic(negll, ag, θ0, ls, opts)
    else
        Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    end
    θ̂ = Optim.minimizer(res)
    if X_lv_fit !== nothing
        cursor = 0
        β̂ = collect(θ̂[(cursor + 1):(cursor + p)])
        cursor += p
        alpha_hat = reshape(collect(θ̂[(cursor + 1):(cursor + q_lv * K)]), q_lv, K)
        cursor += q_lv * K
        Λ̂ = unpack_lambda(@view(θ̂[(cursor + 1):(cursor + rr)]), p, K)
        cursor += rr
        α̂ = exp(θ̂[cursor + 1])
        return GammaFit(β̂, Λ̂, α̂, link, _fit_verdict(res)...,
                       alpha_hat, collect(Float64, θ̂), hessian)
    else
        β̂ = θ̂[1:p]
        Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
        α̂ = exp(θ̂[p + rr + 1])
        return GammaFit(β̂, Λ̂, α̂, link, _fit_verdict(res)..., nothing, Float64[], hessian)
    end
end
