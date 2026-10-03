# Negative-binomial (NB2) family pieces for the generic Laplace core
# (src/families/laplace.jl). y_t ~ NegBinomial(mean μ_t, dispersion r); μ = exp(η)
# (log link), Var = μ + μ²/r. As r → ∞ the NB collapses to Poisson. The dispersion
# `r` is carried in the family marker `NegativeBinomial(r, ·)` — only its `r` field
# is used. Evaluate the density directly from μ: forming p = r/(r+μ)
# loses the mean as p rounds to one near the Poisson limit.
#
# Score/weight wrt η (with V(μ) = μ + μ²/r the NB2 variance):
#   s = (y − μ)/V · dμ/dη,   W = (dμ/dη)²/V   (expected-information ⇒ W ≥ 0).
_clamp_mu(::NegativeBinomial, μ) = max(μ, 1e-12)
_glm_score(f::NegativeBinomial, μ, n, me, y) = (y - μ) / (μ + μ^2 / f.r) * me
_glm_weight(f::NegativeBinomial, μ, n, me)   = me^2 / (μ + μ^2 / f.r)
function _nb2_logpdf_mean(μ, r, y::Int)
    y < 0 && return oftype(μ + r, -Inf)
    q = log1p(μ / r)
    y == 0 && return -r * q
    # The rising-factorial series is shared with truncated NB2. Its explicit
    # remainder bound avoids cancellation at large r without an O(y) loop.
    rise, admissible = _truncnb2_logrise_series(r, y)
    if admissible
        return rise + y * log(μ) - loggamma(float(y) + 1) - r * q - y * q
    end
    return -r * q + y * (log(μ) - log(r) - q) -
           log(r + y) - logbeta(r, float(y) + 1)
end
_glm_logpdf(f::NegativeBinomial, μ, n, y) = _nb2_logpdf_mean(μ, f.r, Int(y))

# Observed conditional curvature for NB2/log: −∂²ℓ/∂η² = μ·r·(r+y)/(r+μ)².
# (Derivative of the score r(y−μ)/(r+μ) wrt η with μ = e^η; differs from the
# Fisher weight μr/(r+μ) whenever y ≠ μ — log is non-canonical for NB2.)
# Divide through by r² before evaluation; the original products overflow even
# when the size and limiting curvature are both finite.
_nb2_observed_weight(μ, r, y) = μ * (1 + y / r) / (1 + μ / r)^2
_glm_obs_weight(f::NegativeBinomial, μ, n, me, y, link::LogLink, η) =
    _nb2_observed_weight(μ, f.r, y)

"""
    nb_marginal_loglik_laplace(Y, Λ, β, r; link=LogLink(), kwargs...) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a negative-binomial
(NB2) GLLVM with dispersion `r` (`Var = μ + μ²/r`) — a thin wrapper over the
family-generic `marginal_loglik_laplace` with `NegativeBinomial(r, ·)`. `Y` is the
p×n integer count matrix; `Λ` p×K; `β` length-p. As `r → ∞` this tends to the
Poisson marginal.
"""
nb_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector,
        r::Real; link::Link = LogLink(), kwargs...) =
    marginal_loglik_laplace(NegativeBinomial(float(r), 0.5), Y, ones(Int, size(Y)), Λ, β, link; kwargs...)

# ---------------------------------------------------------------------------
# Fit driver (NB family slice 2).
# ---------------------------------------------------------------------------

"""
    NBFit

Result of [`fit_nb_gllvm`](@ref): intercepts `β` (length p), loadings `Λ` (p×K),
the estimated dispersion `r` (Var = μ + μ²/r), the `link`, the maximised Laplace
`loglik`, the optimiser `converged` flag, and `iterations`. Fits using `X_lv`
additionally retain `alpha_lv`, the raw latent-axis coefficients for the
predictor-informed score mean; use [`extract_lv_effects`](@ref) for the
rotation-stable trait-scale product `Λ * alpha_lv'`.
"""
struct NBFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    r::Float64
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
NBFit(β, Λ, r, link, loglik, converged, iterations, alpha_lv, theta_packed) =
    NBFit(β, Λ, r, link, loglik, converged, iterations, alpha_lv, theta_packed, _default_hessian(NegativeBinomial(1.0, 0.5), link))

NBFit(β::Vector{Float64}, Λ::Matrix{Float64}, r::Float64, link::Link,
      loglik::Float64, converged::Bool, iterations::Int) =
    NBFit(β, Λ, r, link, loglik, converged, iterations, nothing, Float64[])

function Base.show(io::IO, f::NBFit)
    p, K = size(f.Λ)
    print(io, "NBFit(p=", p, ", K=", K, ", r=", round(f.r; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          f.alpha_lv === nothing ? "" : ", X_lv=true",
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    nb_lv_nll_packed(params, Y, p, K, link; X_lv, q_lv, kwargs...) -> Real

Negative Laplace log-likelihood for the predictor-informed latent-score NB2
model. Parameter layout: `params[1:p]` = β; next `q_lv*K` = `alpha_lv` (q_lv×K);
next `rr` = packed loadings `Λ`; last entry = `log r` (shared dispersion). The
predictor mean enters the Laplace core as the parameter-dependent offset
`Λ * alpha_lv' * X_lv[s]` (the same offset trick as the Poisson/binomial routes).
"""
function nb_lv_nll_packed(params::AbstractVector, Y::AbstractMatrix,
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
        "(p=$p + alpha_lv=$(q_lv * K) + rr=$rr + log_r=1)"))

    cursor = 0
    β = @view params[(cursor + 1):(cursor + p)]
    cursor += p
    alpha_vec = @view params[(cursor + 1):(cursor + q_lv * K)]
    alpha_lv = reshape(alpha_vec, q_lv, K)
    cursor += q_lv * K
    θ_rr = @view params[(cursor + 1):(cursor + rr)]
    Λ = unpack_lambda(θ_rr, p, K)
    cursor += rr
    r = exp(params[cursor + 1])

    lv_offset = _lv_mean_eta(Λ, X_lv, alpha_lv)
    off = offset === nothing ? lv_offset : offset .+ lv_offset
    return -nb_marginal_loglik_laplace(Y, Λ, β, r; link = link, mask = mask, offset = off,
                                       maxiter = maxiter, tol = tol)
end

# NB2/log Laplace log-det defaults to the OBSERVED conditional curvature
# W_obs = μ·r·(r+y)/(r+μ)², matching TMB / gllvmTMB. CHANGED 2026-08-27 on the
# curvature-adjudication campaign (900 cells): observed's estimates land closer
# to the exact-marginal optimum in 100% of medium/strong cells AND its
# objective value approximates the exact marginal better in 87% of cells —
# both metrics agree for NB2 (unlike Beta/NB1/Student-t, which stay Fisher
# pending the maintainer's call on the estimator-vs-reporting trade-off).
# This obliges `laplace_grad.jl`'s NB2 log-det weight to move in the same
# commit — otherwise the analytic gradient stops being the gradient of this
# objective and degrades silently rather than erroring.
_default_hessian(::NegativeBinomial, ::LogLink) = :observed

"""
    fit_nb_gllvm(Y; K, link=LogLink(), mask=nothing, r_init=nothing, …) -> NBFit

Fit a negative-binomial (NB2) GLLVM by L-BFGS over `[β; vec(Λ); log r]` on the
Laplace marginal (`nb_marginal_loglik_laplace`), jointly estimating the dispersion
`r`. `Y` is a p×n integer count matrix (may contain `missing`); `K` the latent
dimension. The default analytic Laplace gradient is used on the plain
no-mask/no-offset path, with an internal finite-difference fallback; masked or
offset fits use finite differences. Warm start = empirical log-mean intercepts +
an SVD loadings init + a moderate `r₀`.

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or `missing` entries in
`Y`; masked cells are dropped from the marginal and the warm start, so the fit
depends only on the observed cells.

`hessian` selects the Laplace log-det curvature only (`:fisher` expected /
`:observed` joint — TMB's choice); the inner mode search is always
Fisher-scored. Default: NB2/log default `:observed` (changed 2026-08-27 on the
curvature-adjudication campaign evidence). Omitting it is exactly the default-path
behaviour.

`converged` is forced `false` when the fitted `r` lies outside `[1e-6, 1e6]` (the
Poisson limit above, unidentified extreme overdispersion below), matching the grouped
NB fitters; the reported `loglik` is unchanged.
"""
function fit_nb_gllvm(Y::AbstractMatrix; K::Integer,
        link::Link = LogLink(), mask = nothing, offset = nothing,
        gradient::Symbol = :analytic,
        hessian::Symbol = _default_hessian(NegativeBinomial(1.0, 0.5), link),
        β_init = nothing, Λ_init = nothing, r_init = nothing,
        X_lv::Union{Nothing, AbstractMatrix} = nothing,
        alpha_lv_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, mask = mask,
                               caller = "fit_nb_gllvm")
    rr = rr_theta_len(p, K)
    hessian in (:fisher, :observed) || throw(ArgumentError(
        "fit_nb_gllvm: hessian must be :fisher or :observed; got :$hessian"))
    if X_lv !== nothing && hessian !== _default_hessian(NegativeBinomial(1.0, 0.5), link)
        throw(ArgumentError("fit_nb_gllvm: a non-default hessian is not yet supported " *
                            "together with X_lv (the packed X_lv objective does not thread it)"))
    end

    # Predictor-informed latent-score mean (Design 73 / gllvmTMB C1): X_lv (n×q_lv)
    # activates joint estimation of alpha_lv with the shared NB2 dispersion r, via
    # the parameter-dependent offset Λ * alpha_lv' * X_lv[s]. Point-estimate only.
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

    # NA handling: observation mask + sanitized counts (see fit_poisson_gllvm).
    msk = mask === nothing ? (any(ismissing, Y) ? observed_mask(Y) : nothing) : mask
    Yc = Integer.(_sanitize_missing(Y, 0))

    Zemp = [linkfun(link, max(Yc[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)           # offset (η = β + offset + Λz)
    if msk !== nothing
        @inbounds for t in 1:p
            cnt = count(view(msk, t, :))
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
    logr0 = r_init === nothing ? log(10.0) : log(float(r_init))

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

    θ0 = vcat(β0, pack_lambda(Λ0), logr0)
    N1 = ones(Int, size(Yc))                     # unit trials, hoisted out of the per-eval closure
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        r = exp(θ[p + rr + 1])
        v = try
            -marginal_loglik_laplace(NegativeBinomial(float(r), 0.5), Yc, N1, Λ, β, link;
                                     hessian = hessian,
                                     mask = msk, offset = offset,
                                     maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = if X_lv_fit !== nothing
        # Predictor-informed latent-score route: joint (β, alpha_lv, Λ, log r) by
        # finite differences — the offset depends jointly on Λ and alpha_lv.
        θ0_lv = vcat(β0, vec(alpha0), pack_lambda(Λ0), logr0)
        negll_lv = θ -> begin
            v = try
                nb_lv_nll_packed(θ, Yc, p, K, link;
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
           hessian === _default_hessian(NegativeBinomial(1.0, 0.5), link)
        # `nb_laplace_grad` is derived for the log link only; any other link takes the
        # :finite path below, which differentiates the actual objective.
        ag = θ -> begin
            β = θ[1:p]; Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K); rv = exp(θ[p + rr + 1])
            try -nb_laplace_grad(Yc, Λ, β, rv; mask = msk) catch; nothing end
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
        r̂ = exp(θ̂[cursor + 1])
        return NBFit(β̂, Λ̂, r̂, link, _nb_shared_r_verdict(res, r̂)...,
                     alpha_hat, collect(Float64, θ̂), hessian)
    else
        β̂ = θ̂[1:p]
        Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
        r̂ = exp(θ̂[p + rr + 1])
        return NBFit(β̂, Λ̂, r̂, link, _nb_shared_r_verdict(res, r̂)..., nothing, Float64[], hessian)
    end
end

# Shared-r dispersion verdict. Maintainer decision 2026-09-29 ("NB upper end: warn only
# everywhere"): a fitted r ABOVE 1e6 is the Poisson limit. There the likelihood is flat in
# log r, but the other estimates are the Poisson fit's, so the fit only WARNS and keeps
# Optim's verdict (gllvmTMB 0.7.1 itself puts one trait's dispersion above 1e6 on 4 of 5
# ordinary per-trait NB fits, so flagging this end would mark most normal fits as failed).
# Measured on origin/main 0ce4a35aa: Poisson data fitted with fit_nb_gllvm reached r = 1.7e7
# (fixture test/fixtures/nb_shared_r_boundary.toml) with no message at all. A fitted r
# BELOW 1e-6 (unidentified extreme overdispersion) still forces converged = false. The
# loglik and iteration count stay as `_fit_verdict` computed them. Returns the triple in
# `_fit_verdict`'s order, ready for `NBFit`.
_nb_shared_r_lower(r::Real) = r < 1e-6

function _nb_shared_r_verdict(res, r::Real)
    loglik, conv, iters = _fit_verdict(res)
    if r > 1e6
        @warn "NB2 shared-dispersion fit reached the Poisson limit (r = $(round(r; sigdigits = 4)) above 1e6): the data show no overdispersion; the other estimates are those of a Poisson fit and are unaffected." maxlog=1
    end
    lower = _nb_shared_r_lower(r)
    lower && @warn "NB2 shared-dispersion fit reached the lower boundary (r = $(round(r; sigdigits = 4)) below 1e-6); the overdispersion is not identified on this data, so converged is set to false." maxlog=1
    return (loglik, conv && !lower, iters)
end
