# Beta (proportions in (0,1)) family pieces for the generic Laplace core
# (src/families/laplace.jl). y_t ∈ (0,1); mean μ = linkinv(link, η) (logit link),
# precision φ; the per-observation law is Beta(μφ, (1−μ)φ), Var = μ(1−μ)/(1+φ).
# The precision φ is carried in the family marker `Beta(φ, ·)` — only its `α`
# field is read as φ.
#
# Score/weight wrt η (Ferrari & Cribari-Neto 2004 beta regression):
#   y*  = logit(y),   μ* = ψ(μφ) − ψ((1−μ)φ)
#   s   = φ (y* − μ*) · dμ/dη
#   W   = φ² [ψ′(μφ) + ψ′((1−μ)φ)] · (dμ/dη)²        (expected information ⇒ W ≥ 0)
# with ψ = digamma, ψ′ = trigamma.
_clamp_mu(::Beta, μ) = clamp(μ, 1e-6, 1 - 1e-6)

# Boundary guard: clamp y into [1e-12, 1-1e-12], matching gllvmTMB's y_safe
# (gllvmTMB.cpp, Beta branch) so exact 0/1 proportions do not give log(0) (#147).
# A no-op for interior y.
_clamp_y(::Beta, y) = clamp(y, 1e-12, 1 - 1e-12)

function _glm_score(f::Beta, μ, n, me, y)
    φ = f.α
    y = _clamp_y(f, y)
    ystar = log(y) - log1p(-y)                      # logit(y)
    μstar = digamma(μ * φ) - digamma((1 - μ) * φ)
    return φ * (ystar - μstar) * me
end

function _glm_weight(f::Beta, μ, n, me)
    φ = f.α
    ν = trigamma(μ * φ) + trigamma((1 - μ) * φ)
    return φ^2 * ν * me^2
end

_glm_logpdf(f::Beta, μ, n, y) = _beta_logpdf(μ * f.α, (1 - μ) * f.α, _clamp_y(f, y))

# Beta(a, b) log-density, a = μφ, b = (1−μ)φ. `logpdf(Beta(a, b), y)` forms
# `loggamma(a+b) − loggamma(a) − loggamma(b)` from three terms of size φ log φ, so at φ ≈ 1e12
# Float64 leaves an error of ~1e-3 per observation and at φ ≈ 1e18 an error of hundreds (measured
# against BigFloat). A fit that ran to such a precision then reported a large positive
# log-likelihood (#1143). When BOTH a and b are at least `_BETA_STIRLING_FROM` the three loggammas
# are replaced by their Stirling series; with s = a + b the large terms cancel analytically:
#   log f = a [log(y s / a) − u] + b [log((1−y) s / b) − w] + ½ log(a b / (s 2π)) − log y − log(1−y)
#           + S(s) − S(a) − S(b),     S(x) = 1/(12x) − 1/(360x³) + 1/(1260x⁵) − 1/(1680x⁷),
# where u = (y s − a)/a and w = −(y s − a)/b (so a u + b w = 0, which is what lets the kernel be
# written as a [log(1+u) − u] + b [log(1+w) − w]). Both brackets are ≤ 0 and are evaluated by
# `_log1pmx` without cancellation near y = a/s. The numerator y s − a is formed with an fma and the
# rounding error of s = a + b, so it is accurate when y is within a few standard deviations of a/s.
# Below the threshold the original call is used unchanged, so ordinary fits are bit-identical.
const _BETA_STIRLING_FROM = 1.0e3

# log1p(x) − x for x > −1 without cancellation, given r = 1 + x supplied accurately by the caller:
# for |x| < 1/2 the series 2 Σ_{k≥3,odd} z^k/k − 2z²/(1−z) in z = x/(2+x) (|z| ≤ 1/3), else log(r) − x.
@inline function _log1pmx(x, r)
    abs(x) < 0.5 || return log(r) - x
    z = x / (2 + x)
    z2 = z * z
    term = z * z2          # z³
    acc = zero(z)
    k = 3
    while k <= 41
        acc += term / k
        term *= z2
        k += 2
    end
    return 2 * acc - 2 * z2 / (1 - z)
end

@inline _stirling_corr(x) = inv(12x) - inv(360 * x^3) + inv(1260 * x^5) - inv(1680 * x^7)

_beta_stirling_applies(a, b) = a >= _BETA_STIRLING_FROM && b >= _BETA_STIRLING_FROM

function _beta_logpdf(a, b, y)
    _beta_stirling_applies(a, b) || return logpdf(Beta(a, b), y)
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)            # exact rounding error of s = a + b
    n = fma(y, s, -a) + y * e                # y (a + b) − a
    u = n / a
    w = -n / b
    ω = 1 - y
    kernel = a * _log1pmx(u, y * s / a) + b * _log1pmx(w, ω * s / b)
    return kernel + 0.5 * log(a * (b / s) / (2π)) - log(y) - log1p(-y) +
           _stirling_corr(s) - _stirling_corr(a) - _stirling_corr(b)
end

"""
    beta_marginal_loglik_laplace(Y, Λ, β, φ; link=LogitLink(), kwargs...) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a Beta GLLVM with
precision `φ` — responses `Y ∈ (0,1)`, mean `μ = logistic(η)`, per-observation
`Beta(μφ, (1−μ)φ)` (`Var = μ(1−μ)/(1+φ)`). A thin wrapper over the family-generic
`marginal_loglik_laplace` with the `Beta(φ, ·)` marker.
"""
beta_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector,
        φ::Real; link::Link = LogitLink(), kwargs...) =
    marginal_loglik_laplace(Beta(float(φ), 1.0), Y, ones(Int, size(Y)), Λ, β, link; kwargs...)

# ---------------------------------------------------------------------------
# Fit driver (Beta family slice 2).
# ---------------------------------------------------------------------------

"""
    BetaFit

Result of [`fit_beta_gllvm`](@ref): intercepts `β` (length p), loadings `Λ` (p×K),
the estimated precision `φ` (Var = μ(1−μ)/(1+φ)), the `link`, the maximised Laplace
`loglik`, the optimiser `converged` flag, and `iterations`.
`offset` is the p×n training offset the fit was made with (`nothing` when it had none);
[`predict`](@ref), [`getLV`](@ref) and `residuals` use it by default.
"""
struct BetaFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    φ::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    alpha_lv::Union{Nothing, Matrix{Float64}}
    theta_packed::Vector{Float64}
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    offset::Union{Nothing,Matrix{Float64}}   # training offset (p×n); `nothing` = none
end

# Pre-offset compat tier (10 positional args): no stored training offset.
BetaFit(β, Λ, φ, link, loglik, converged, iterations, alpha_lv, theta_packed, hessian) =
    BetaFit(β, Λ, φ, link, loglik, converged, iterations, alpha_lv, theta_packed, hessian, nothing)

# Positional compatibility constructor (2026-08-28): every pre-existing
# construction site builds a default-curvature fit; the `hessian` field
# records the objective identity so `confint`/bootstrap can rebuild THE
# SAME objective instead of guessing (the audit's confint-consistency class).
BetaFit(β, Λ, φ, link, loglik, converged, iterations, alpha_lv, theta_packed) =
    BetaFit(β, Λ, φ, link, loglik, converged, iterations, alpha_lv, theta_packed, _default_hessian(Beta(1.0, 1.0), link))

BetaFit(β::Vector{Float64}, Λ::Matrix{Float64}, φ::Float64, link::Link,
        loglik::Float64, converged::Bool, iterations::Int) =
    BetaFit(β, Λ, φ, link, loglik, converged, iterations, nothing, Float64[])

function Base.show(io::IO, f::BetaFit)
    p, K = size(f.Λ)
    print(io, "BetaFit(p=", p, ", K=", K, ", φ=", round(f.φ; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          f.alpha_lv === nothing ? "" : ", X_lv=true",
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    beta_lv_nll_packed(params, Y, p, K, link; X_lv, q_lv, kwargs...) -> Real

Negative Laplace log-likelihood for the predictor-informed latent-score Beta
model. Parameter layout: `params[1:p]` = β; next `q_lv*K` = `alpha_lv`; next
`rr` = packed loadings `Λ`; last entry = `log φ` (precision). The predictor mean
enters the Laplace core as the offset `Λ * alpha_lv' * X_lv[s]`.
"""
function beta_lv_nll_packed(params::AbstractVector, Y::AbstractMatrix,
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
        "(p=$p + alpha_lv=$(q_lv * K) + rr=$rr + log_phi=1)"))

    cursor = 0
    β = @view params[(cursor + 1):(cursor + p)]
    cursor += p
    alpha_vec = @view params[(cursor + 1):(cursor + q_lv * K)]
    alpha_lv = reshape(alpha_vec, q_lv, K)
    cursor += q_lv * K
    θ_rr = @view params[(cursor + 1):(cursor + rr)]
    Λ = unpack_lambda(θ_rr, p, K)
    cursor += rr
    φ = exp(params[cursor + 1])

    lv_offset = _lv_mean_eta(Λ, X_lv, alpha_lv)
    off = offset === nothing ? lv_offset : offset .+ lv_offset
    return -beta_marginal_loglik_laplace(Y, Λ, β, φ; link = link, mask = mask, offset = off,
                                         maxiter = maxiter, tol = tol)
end

# Beta/logit Laplace log-det defaults to the OBSERVED conditional curvature
# (decision A, 2026-08-27, on the 900-cell campaign: observed's estimates beat
# Fisher's in 100% of medium/strong cells; the reported-loglik accuracy cost
# — Fisher approximated the VALUE better in 83% of cells — was accepted
# explicitly for TMB parity and estimator quality). Delegates to the
# hand-derived grouped formula. The analytic gradient's log-det weight
# (`laplace_grad.jl`) moves in the same commit — the coupled-change rule.
_glm_obs_weight(f::Beta, μ, n, me, y, link::LogitLink, η) =
    _beta_grouped_laplace_weight(:observed, f, μ, me, y, link, η)
_default_hessian(::Beta, ::LogitLink) = :observed

"""
    fit_beta_gllvm(Y; K, link=LogitLink(), φ_init=nothing, …) -> BetaFit

Fit a Beta GLLVM by L-BFGS over `[β; vec(Λ); log φ]` on the Laplace marginal
(`beta_marginal_loglik_laplace`), jointly estimating the precision `φ`
(`Var = μ(1−μ)/(1+φ)`). `Y` is a p×n matrix of proportions in (0,1); `K` the latent
dimension. The default analytic Laplace gradient is used on the plain
no-mask/no-offset path with the logit link, with an internal finite-difference
fallback; masked or offset fits, and any other `link`, use finite differences of
the objective for that link. Warm start = empirical logit-mean intercepts +
an SVD loadings init + a moderate `φ₀`.

`hessian` selects the Laplace log-det curvature only (`:fisher` expected /
`:observed` joint — TMB's choice); the inner mode search is always
Fisher-scored. Default: Beta/logit default `:fisher`. Omitting it is exactly the pre-kwarg
behaviour.
"""
function fit_beta_gllvm(Y::AbstractMatrix; K::Integer,
        link::Link = LogitLink(), mask = nothing, offset = nothing,
        gradient::Symbol = :analytic,
        hessian::Symbol = _default_hessian(Beta(1.0, 1.0), link),
        β_init = nothing, Λ_init = nothing, φ_init = nothing,
        X_lv::Union{Nothing, AbstractMatrix} = nothing,
        alpha_lv_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, mask = mask,
                               caller = "fit_beta_gllvm")
    rr = rr_theta_len(p, K)
    # `hessian` selects the log-det curvature ONLY (the mode search is always
    # Fisher-scored — see families/laplace.jl). Default = the family's registered
    # default, so omitting the kwarg is bit-for-bit the pre-kwarg behaviour.
    hessian in (:fisher, :observed) || throw(ArgumentError(
        "fit_beta_gllvm: hessian must be :fisher or :observed; got :$hessian"))
    if X_lv !== nothing && hessian !== _default_hessian(Beta(1.0, 1.0), link)
        throw(ArgumentError("fit_beta_gllvm: a non-default hessian is not yet supported " *
                            "together with X_lv (the packed X_lv objective does not thread it)"))
    end

    # Predictor-informed latent-score mean (Design 73 / gllvmTMB C1): X_lv (n×q_lv)
    # activates joint estimation of alpha_lv with the shared precision φ, via the
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
    _check_response_support(:beta, Y, msk; caller = "fit_beta_gllvm")
    Yc  = _sanitize_unobserved(Y, msk, 0.5)           # in-(0,1) placeholder

    Zemp = [linkfun(link, clamp(float(Yc[t, i]), 1e-6, 1 - 1e-6)) for t in 1:p, i in 1:n]
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
    logφ0 = φ_init === nothing ? log(10.0) : log(float(φ_init))

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

    θ0 = vcat(β0, pack_lambda(Λ0), logφ0)
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        φ = exp(θ[p + rr + 1])
        v = try
            -beta_marginal_loglik_laplace(Yc, Λ, β, φ; link = link, mask = msk,
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
        θ0_lv = vcat(β0, vec(alpha0), pack_lambda(Λ0), logφ0)
        negll_lv = θ -> begin
            v = try
                beta_lv_nll_packed(θ, Yc, p, K, link;
                                   X_lv = X_lv_fit, q_lv = q_lv,
                                   mask = msk, offset = offset,
                                   maxiter = newton_maxiter, tol = newton_tol)
            catch
                return 1e12
            end
            return isfinite(v) ? v : 1e12
        end
        Optim.optimize(negll_lv, θ0_lv, ls, opts; autodiff = :finite)
    elseif gradient === :analytic && offset === nothing && link isa LogitLink &&
           hessian === _default_hessian(Beta(1.0, 1.0), link)
        # The analytic gradient implements the DEFAULT curvature's objective (the two
        # moved together in the 2026-08-25 role-separation work). A non-default
        # `hessian` therefore falls through to :finite below, which differentiates the
        # actual objective and cannot desynchronise from it. The same holds for the
        # link: `beta_laplace_grad` is derived for the logit link only (its score,
        # weight and log-det are hand-written in `μ(1−μ)`), so any other link also
        # takes the :finite path.
        ag = θ -> begin
            β = θ[1:p]; Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K); fv = exp(θ[p + rr + 1])
            try -beta_laplace_grad(Yc, Λ, β, fv; mask = msk) catch; nothing end
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
        φ̂ = exp(θ̂[cursor + 1])
        return BetaFit(β̂, Λ̂, φ̂, link, _fit_verdict(res)...,
                       alpha_hat, collect(Float64, θ̂), hessian, _stored_offset(offset))
    else
        β̂ = θ̂[1:p]
        Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
        φ̂ = exp(θ̂[p + rr + 1])
        return BetaFit(β̂, Λ̂, φ̂, link, _fit_verdict(res)..., nothing, Float64[], hessian,
                       _stored_offset(offset))
    end
end
