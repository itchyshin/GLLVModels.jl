# Beta-hurdle (hurdle-beta) two-part GLLVM family.
#
# A hurdle model on [0,1): the zero/one "absence" process is Bernoulli with
# occurrence probability π = logistic(η^z), and conditional on presence (y > 0)
# the positive part is Beta(μφ, (1−μ)φ) with mean μ = logistic(η^c) and shared
# precision φ (Var = μ(1−μ)/(1+φ)).
#
# Joint log-density:
#   y = 0:  log(1−π)
#   y > 0:  log(π) + logpdf(Beta(μφ, (1−μ)φ), y)
#
# Score / Fisher weight wrt η^c for the positive-part (Ferrari & Cribari-Neto
# 2004 beta regression, logit-link, dμ/dη = μ(1−μ)):
#   y* = logit(y),  μ* = ψ(μφ) − ψ((1−μ)φ)
#   s^c = φ(y* − μ*) · μ(1−μ)          (∂logf/∂η^c)
#   W^c = φ²[ψ′(μφ) + ψ′((1−μ)φ)] · (μ(1−μ))²     (expected Fisher info ≥ 0)
#
# The substrate is the shared two-part Newton from src/families/twopart.jl:
# the occurrence part (Λ_z = 0 by default in v1) uses the logistic Bernoulli
# pieces, and the positive part uses the Beta GLM score/weight — exactly as
# DeltaGamma uses Gamma pieces.
#
# References:
#   Ferrari & Cribari-Neto 2004 (beta regression, Technometrics)
#   Ospina & Ferrari 2010 (inflated beta distributions)

# ---------------------------------------------------------------------------
# BetaHurdle marker — carries the shared Beta precision φ.
# ---------------------------------------------------------------------------

"""
    BetaHurdle(φ = 5.0)

Marker for the Beta-hurdle two-part family: Bernoulli occurrence × positive Beta
with shared precision `φ` (mean `μ = logistic(η^c)`, `Var = μ(1−μ)/(1+φ)`).
Use with the unified entry point:

```julia
fit_gllvm(Y; family = BetaHurdle(), K = 2)
fit_gllvm(Y; family = BetaHurdle(80.0), K = 2)   # same fit — φ is a tag payload
gllvm(@formula(y ~ 1), Y, site_data; family = BetaHurdle(), K = 2)
```

The marker's `φ` field is a **tag payload** — it is never read by
[`fit_gllvm`](@ref) / [`fit_beta_hurdle_gllvm`](@ref); the shared precision is
always estimated. There is no `φ_init` keyword. `BetaHurdle()` is the public
convenience (`5.0` matches the named fitter's MoM fallback). Named fitter
[`fit_beta_hurdle_gllvm`](@ref) remains available. +X, `disp_group`, and
`row_eff` are not admitted on this surface.
"""
struct BetaHurdle
    φ::Float64
end

# Public-call convenience (mirrors `HurdleNB()` / `COMPoisson()`): φ is never
# read on the `fit_gllvm` route. 5.0 is the fitter's MoM fallback.
BetaHurdle() = BetaHurdle(5.0)

# Two-part pieces dispatched on BetaHurdle. `_tp_pieces` is defined in
# src/families/twopart.jl (loaded before this file).
function _tp_pieces(f::BetaHurdle, y, ηz, ηc)
    # Occurrence block (identical for every two-part family).
    π  = inv(one(ηz) + exp(-ηz))                # logistic(η^z)
    Wz = π * (one(π) - π)                       # Bernoulli Fisher weight
    if y > 0
        φ  = f.φ
        # Positive-part linear predictor → logit-link mean.
        μ  = inv(one(ηc) + exp(-ηc))            # logistic(η^c) ∈ (0,1)
        μ  = clamp(μ, 1e-6, 1 - 1e-6)
        me = μ * (one(μ) - μ)                   # dμ/dη = μ(1−μ)  (logit link)
        yc = clamp(float(y), 1e-6, 1 - 1e-6)   # guard against exact 0/1 data
        ystar = log(yc) - log1p(-yc)            # logit(y)
        μstar = digamma(μ * φ) - digamma((one(μ) - μ) * φ)
        sc    = φ * (ystar - μstar) * me        # ∂logf/∂η^c
        ν     = trigamma(μ * φ) + trigamma((one(μ) - μ) * φ)
        Wc    = φ^2 * ν * me^2                  # expected Fisher info wrt η^c
        logf  = log(π) + logpdf(Beta(μ * φ, (one(μ) - μ) * φ), yc)
        return (one(π) - π, sc, Wz, Wc, logf)
    else
        return (-π, zero(ηc), Wz, zero(ηc), log1p(-π))
    end
end

# ---------------------------------------------------------------------------
# Public marginal log-likelihood.
# ---------------------------------------------------------------------------

"""
    beta_hurdle_marginal_loglik_laplace(Y, Λc, βz, βc, φ; Λz=nothing, kwargs...) -> Float64

Total two-part Laplace log-marginal for a Beta-hurdle GLLVM: occurrence probability
`π = logistic(β^z + Λ_z z)` (intercept-only by default, `Λ_z = 0`) times a positive
Beta with mean `μ = logistic(β^c + Λ_c z)` and precision `φ`
(`Var = μ(1−μ)/(1+φ)`). `Y` is p×n with `0` for absences and values in `(0,1)` for
the positive part. With `Λ_c = 0` (and `Λ_z = 0`) this reduces exactly to the
independent per-cell two-part log-likelihood (the key Λ=0 anchor).
"""
function beta_hurdle_marginal_loglik_laplace(Y::AbstractMatrix, Λc::AbstractMatrix,
        βz::AbstractVector, βc::AbstractVector, φ::Real;
        Λz::Union{Nothing, AbstractMatrix} = nothing, kwargs...)
    p, K = size(Λc)
    Λz_ = Λz === nothing ? zeros(p, K) : Λz
    return twopart_marginal_loglik_laplace(BetaHurdle(float(φ)), Y, Λz_, Λc, βz, βc; kwargs...)
end

# ---------------------------------------------------------------------------
# Fit result struct + display.
# ---------------------------------------------------------------------------

"""
    BetaHurdleFit

Result of [`fit_beta_hurdle_gllvm`](@ref): occurrence logits `βz` (length p),
positive-part logit-mean intercepts `βc` (length p), positive-part loadings `Λc`
(p×K), the shared Beta precision `φ` (`Var = μ(1−μ)/(1+φ)`), the maximised
`loglik`, `converged`, and `iterations`. (`Λz = 0` — occurrence is intercept-only.)
"""
struct BetaHurdleFit
    βz::Vector{Float64}
    βc::Vector{Float64}
    Λc::Matrix{Float64}
    φ::Float64
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::BetaHurdleFit)
    p, K = size(f.Λc)
    print(io, "BetaHurdleFit(p=", p, ", K=", K, ", φ=", round(f.φ; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

# ---------------------------------------------------------------------------
# Post-fit API — getLoadings / getLV / predict / residuals / _nparams / show.
# These follow the DeltaGammaFit pattern exactly (see src/postfit.jl); they
# live here so the file is self-contained and postfit.jl needs no edits.
# ---------------------------------------------------------------------------

_loadings(fit::BetaHurdleFit) = fit.Λc
_loglik(fit::BetaHurdleFit)   = fit.loglik

function _nparams(fit::BetaHurdleFit)
    p, K = size(fit.Λc)
    return 2p + (p * K - div(K * (K - 1), 2)) + 1   # βz + βc + Λc + φ
end

"""
    getLV(fit::BetaHurdleFit, Y; rotate=true) -> n×K matrix

Conditional latent scores for a Beta-hurdle fit: the per-site two-part Laplace mode
`ẑₛ` (occurrence intercept-only, so only the positive part loads on `z`). `Y` is
p×n with `0` for absences and values in `(0,1)` for the positive part;
`rotate=true` applies the canonical [`rotation`](@ref).
"""
function getLV(fit::BetaHurdleFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y); K = size(fit.Λc, 2)
    fam = BetaHurdle(fit.φ)
    Λz  = zeros(p, K)
    Z   = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _twopart_mode(fam, view(Y, :, s), Λz, fit.Λc, fit.βz, fit.βc)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λc) : Zt
end

"""
    predict(fit::BetaHurdleFit, Y; type=:response) -> p×n matrix

In-sample predictions at the Laplace mode. `type=:link` is the positive-part logit
predictor `η^c = β^c + Λ_c ẑ`; `:occurrence` the presence probability
`π = logistic(β^z)`; `:positive` the conditional positive mean `μ = logistic(η^c)`;
`:response` the unconditional mean `π · μ`.
"""
function predict(fit::BetaHurdleFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:response, :occurrence, :positive, :link) ||
        throw(ArgumentError(
            "type must be :response, :occurrence, :positive, or :link; got :$type"))
    p, n = size(Y)
    Z  = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'                    # p×n
    type === :link && return ηc
    π = inv.(1 .+ exp.(-fit.βz))                  # length p
    type === :occurrence && return repeat(π, 1, n)
    μ = inv.(1 .+ exp.(-ηc))                       # logistic(η^c), p×n
    type === :positive && return μ
    return π .* μ
end

"""
    residuals(fit::BetaHurdleFit, Y; rng=Random.default_rng()) -> p×n matrix

Dunn–Smyth randomized quantile residuals for the Beta-hurdle fit: `Φ⁻¹(u)` with
`u = (1−π) + π·G(y)` for `y > 0` (`G` the Beta CDF at the fitted parameters) and
`u` uniform on `[0, 1−π]` for `y = 0` — ≈ N(0,1) under a correct model (pass a
fixed `rng` to reproduce). The Beta CDF is continuous on `(0,1)` so the
randomization is only needed for the point mass at zero.
"""
function residuals(fit::BetaHurdleFit, Y::AbstractMatrix{<:Real};
                   rng::AbstractRNG = Random.default_rng())
    p, n = size(Y); φ = fit.φ
    Z  = getLV(fit, Y; rotate = false)
    ηc = fit.βc .+ fit.Λc * Z'
    π  = inv.(1 .+ exp.(-fit.βz))
    R  = Matrix{Float64}(undef, p, n)
    @inbounds for s in 1:n, t in 1:p
        πt = π[t]
        if Y[t, s] > 0
            μ = inv(1 + exp(-ηc[t, s]))
            μ = clamp(μ, 1e-6, 1 - 1e-6)
            d = Beta(μ * φ, (1 - μ) * φ)
            yc = clamp(float(Y[t, s]), 1e-6, 1 - 1e-6)
            u  = (1 - πt) + πt * cdf(d, yc)
        else
            u = (1 - πt) * rand(rng)
        end
        R[t, s] = quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))
    end
    return R
end

function Base.show(io::IO, ::MIME"text/plain", fit::BetaHurdleFit)
    p, K = size(fit.Λc)
    println(io, "Beta-hurdle GLLVM fit (two-part)")
    println(io, "  responses p = ", p, ", latent factors K = ", K,
            ", precision φ = ", round(fit.φ; sigdigits = 4))
    println(io, "  logLik = ", round(fit.loglik; sigdigits = 7),
            ", AIC = ", round(aic(fit); sigdigits = 7))
    print(io,   "  converged = ", fit.converged, " (", fit.iterations, " iterations)")
end

# ---------------------------------------------------------------------------
# Fit driver.
# ---------------------------------------------------------------------------

"""
    fit_beta_hurdle_gllvm(Y; K, …) -> BetaHurdleFit

Fit a Beta-hurdle two-part GLLVM by L-BFGS over `[βz; βc; vec(Λc); log φ]` on the
two-part Laplace marginal ([`beta_hurdle_marginal_loglik_laplace`](@ref)), with
`Λz = 0` (per-species occurrence intercept), jointly estimating the precision `φ`.
`Y` is p×n with `0` for absences and values in `(0,1)` otherwise. Finite-difference
gradient; warm start = `logit(empirical P(y>0))` occurrence intercepts +
`logit` mean of the positive values as logit-mean intercepts + SVD of the
logit-scale positive residuals as loadings + a method-of-moments `φ₀` from the
positive-part empirical variance.

`hessian` selects the two-part Laplace log-det curvature (`:observed` default /
`:fisher`); the mode search does not depend on it. NOTE (2026-08-28): for this
family the observed count-part weight is not yet specialised, so both selectors
currently produce the identical objective (the `TWOPART_KNOWN_OPEN` census gap;
DeltaGamma is the only two-part family whose observed weight is implemented).
Exposing the kwarg is the measurement prerequisite for closing that gap.
"""
function fit_beta_hurdle_gllvm(Y::AbstractMatrix{<:Real}; K::Integer,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_beta_hurdle_gllvm: hessian must be :observed or :fisher; got :$hessian"))
    rr = rr_theta_len(p, K)

    # --- Warm start --------------------------------------------------------
    βz0 = Vector{Float64}(undef, p)
    βc0 = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        npres = count(>(0), view(Y, t, :))
        pr    = clamp((npres + 0.5) / (n + 1), 1e-3, 1 - 1e-3)
        βz0[t] = log(pr / (1 - pr))
        # logit-mean of positive values
        s = 0.0; c = 0
        for j in 1:n
            if Y[t, j] > 0
                yc  = clamp(float(Y[t, j]), 1e-6, 1 - 1e-6)
                s  += log(yc) - log1p(-yc)    # logit(y)
                c  += 1
            end
        end
        βc0[t] = c == 0 ? 0.0 : s / c         # mean logit(y) among positives
    end

    # Method-of-moments φ from the positive-part logit residuals:
    # Var(logit(y)) ≈ 1/(φ·μ(1−μ)) in the limit; use a simple moment estimate.
    sumsq = 0.0; nres = 0
    @inbounds for t in 1:p
        μt = inv(1 + exp(-βc0[t]))
        for j in 1:n
            if Y[t, j] > 0
                yc    = clamp(float(Y[t, j]), 1e-6, 1 - 1e-6)
                ystar = log(yc) - log1p(-yc)    # logit(y)
                r     = ystar - βc0[t]           # logit residual
                sumsq += r^2
                nres  += 1
            end
        end
    end
    # logit-scale variance ≈ 1/φ (at central μ=0.5 roughly); clamp to sensible range.
    φ0 = nres > 1 ? clamp((nres - 1) / sumsq, 0.1, 100.0) : 5.0

    # SVD of positive-part logit residuals for loadings warm start.
    Zc = [Y[t, j] > 0 ?
              (log(clamp(float(Y[t, j]), 1e-6, 1 - 1e-6)) -
               log1p(-clamp(float(Y[t, j]), 1e-6, 1 - 1e-6))) - βc0[t] :
              0.0
          for t in 1:p, j in 1:n]
    F  = svd(Zc); kk = min(K, length(F.S))
    Λc0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λc0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end

    # --- Objective (negative Laplace marginal) -----------------------------
    θ0 = vcat(βz0, βc0, pack_lambda(Λc0), log(φ0))
    function negll(θ)
        βz = θ[1:p]
        βc = θ[(p + 1):(2p)]
        Λc = unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
        φ  = exp(θ[2p + rr + 1])
        v  = try
            -beta_hurdle_marginal_loglik_laplace(Y, Λc, βz, βc, φ; hessian = hessian,
                                                 maxiter = newton_maxiter,
                                                 tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end

    # --- L-BFGS optimisation -----------------------------------------------
    ls  = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls,
                         Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂  = Optim.minimizer(res)
    βz = θ̂[1:p]
    βc = θ̂[(p + 1):(2p)]
    Λc = unpack_lambda(θ̂[(2p + 1):(2p + rr)], p, K)
    φ  = exp(θ̂[2p + rr + 1])
    return BetaHurdleFit(βz, βc, Λc, φ, _fit_verdict(res)...)
end
