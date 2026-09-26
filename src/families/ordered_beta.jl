# Ordered-beta family (Kubinec 2023) for the GLLVModels Laplace path.
#
# Responses y ∈ [0,1] with point masses at exactly 0 and 1 plus a continuous Beta
# interior — proportion / cover data. One latent linear predictor η drives all
# three regions via two ordered cutpoints c0 < c1 and a Beta precision φ:
#
#     P(y=0)     = 1 − σ(η − c0)            = σ(c0 − η)
#     P(0<y<1)   = σ(η − c0) − σ(η − c1)
#     P(y=1)     = σ(η − c1)
#     interior density: [σ(η−c0) − σ(η−c1)] · Beta(y; μφ, (1−μ)φ),  μ = σ(η),
#
# so
#     log p(y|η) = (y==0) ? log σ(c0−η)
#                : (y==1) ? log σ(η−c1)
#                : log(σ(η−c0) − σ(η−c1)) + logpdf(Beta(μφ,(1−μ)φ), y).
#
# The link here is identity-on-η (the family marker carries c0, c1, φ; μ = σ(η)
# is formed inside the pieces), so this file runs its OWN per-site Laplace,
# mirroring `_laplace_mode` / `laplace_loglik_site` from families/laplace.jl. The
# per-trait score s_t = ∂log p/∂η and weight W_t = −∂²log p/∂η² are obtained by
# ForwardDiff on the scalar map η → log p (lower risk than the messy three-branch
# closed form), with W_t clamped to ≥ 1e-8 for SPD.

"""
    OrderedBeta(c0, c1, φ)
    OrderedBeta()

Ordered-beta family marker (Kubinec 2023) for proportions / cover in `[0,1]`
with point masses at 0 and 1. `c0 < c1` are the ordered cutpoints and `φ` is
the Beta precision of the (0,1) interior.

```julia
fit_gllvm(Y; family = OrderedBeta(), K = 2)
fit_gllvm(Y; family = OrderedBeta(0.0, 2.0, 3.0), K = 2)  # same — tags never read
gllvm(@formula(y ~ 1), Y, site_data; family = OrderedBeta(), K = 2)
```

The marker's `c0`, `c1`, and `φ` fields are **tag payloads** — they are never
read by [`fit_gllvm`](@ref) / [`fit_ordered_beta_gllvm`](@ref); all three are
always estimated. They are not used as `c0_init` / `c1_init` / `φ_init`. This
is the opposite of [`StudentTFamily`](@ref), whose `ν` is structural and held
fixed, and of Ordinal's `τ₁ = 0` pin (this family has no twin pin).
`OrderedBeta()` is the public convenience (`-1.0, 1.0, 10.0` matches the named
fitter's own defaults).

Named fitter [`fit_ordered_beta_gllvm`](@ref) remains available. +X,
`disp_group`, and `row_eff` are not admitted on this surface.
"""
struct OrderedBeta <: Distribution{Univariate, Continuous}
    c0::Float64
    c1::Float64
    φ::Float64
end

# Public-call convenience (mirrors `COMPoisson()` / `BetaHurdle()`): c0, c1, φ
# are never read on the `fit_gllvm` route. Defaults match the named fitter.
OrderedBeta() = OrderedBeta(-1.0, 1.0, 10.0)

# logistic σ(x), numerically safe at large |x|.
_ob_logistic(x) = x ≥ 0 ? inv(one(x) + exp(-x)) : (e = exp(x); e / (one(x) + e))
# log σ(x) = −log(1 + e^{−x}), numerically safe.
_ob_logsigmoid(x) = -log1p(exp(-abs(x))) + (x < 0 ? x : zero(x))

# Stable log(1 - exp(x)) for x ≤ 0. Standard split at log(2): for x close to 0
# (1-exp(x)) loses precision via log1p, so use log(-expm1(x)) there instead.
_ob_log1mexp(x) = x < -log(2) ? log1p(-exp(x)) : log(-expm1(x))

const _OB_MU_LO = 1e-12
const _OB_MU_HI = 1 - 1e-12

"""
    ordered_beta_logp(y, η, c0, c1, φ) -> Float64

Scalar ordered-beta conditional log-density log p(y|η) for one trait. `y == 0`
and `y == 1` hit the point masses; `0 < y < 1` adds the interior Beta log-density
with `μ = σ(η)` clamped to (1e-12, 1−1e-12).
"""
function ordered_beta_logp(y, η, c0, c1, φ)
    if y == 0
        return _ob_logsigmoid(c0 - η)
    elseif y == 1
        return _ob_logsigmoid(η - c1)
    else
        # interior mass: log(σ(η−c0) − σ(η−c1)); since c0 < c1, σ(η−c0) > σ(η−c1).
        #
        # FIXED 2026-08-26 (root cause: docs/dev-log/check-log.md). The naive
        # `log(σ(η−c0) − σ(η−c1))` underflows to log(0.0) = -Inf whenever both raw
        # sigmoids round to 1.0 in Float64 (η ≳ 37 at this c0/c1, an ordinary value
        # reachable by the Laplace mode solve, not a corner case). `_ob_logsigmoid` is
        # already stable and already used by the boundary branches above; the interior
        # branch simply never got the same treatment. Rewritten via the identity
        # log(σ(a) − σ(b)) = logσ(a) + log1mexp(logσ(b) − logσ(a)) for a > b, whose
        # argument to log1mexp is ≤ 0 by construction (logσ is monotone in its argument
        # and a > b), so it never faces the cancellation that broke the naive form.
        # Verified against the naive computation to < 1.5e-15 everywhere the naive form
        # is itself accurate (see docs/dev-log/pending/ordered-beta-logmass-fix.jl).
        loga = _ob_logsigmoid(η - c0)
        logb = _ob_logsigmoid(η - c1)
        logmass = loga + _ob_log1mexp(logb - loga)
        μ = clamp(_ob_logistic(η), _OB_MU_LO, _OB_MU_HI)
        return logmass + logpdf(Beta(μ * φ, (one(μ) - μ) * φ), y)
    end
end

# Per-trait score s_t = ∂log p/∂η and weight W_t = −∂²log p/∂η², via ForwardDiff
# on the scalar map η → log p. W clamped to ≥ 1e-8 to keep Λ'WΛ + I SPD.
function _ob_score_weight(y, η, c0, c1, φ)
    f  = ηv -> ordered_beta_logp(y, ηv, c0, c1, φ)
    g  = ηv -> ForwardDiff.derivative(f, ηv)
    s  = g(η)
    W  = -ForwardDiff.derivative(g, η)
    return s, max(W, 1e-8)
end

# Site log-posterior q(z) = Σ_t log p(y_t|η_t) − ½z'z (mask drops a term
# entirely, matching the score/weight masking below). Used only by the damped
# mode search's step-halving line search (#501), mirroring
# `_laplace_mode_logpost` / `_grouped_laplace_mode_logpost`.
function _ordered_beta_logpost(y::AbstractVector, Λ::AbstractMatrix, β::AbstractVector,
        c0::Real, c1::Real, φ::Real, z::AbstractVector; mask = nothing)
    η = β .+ Λ * z
    q = -0.5 * dot(z, z)
    @inbounds for t in eachindex(y)
        (mask === nothing || mask[t]) || continue
        q += ordered_beta_logp(y[t], η[t], c0, c1, φ)
    end
    return q
end

# Inner Laplace mode-finder for one site (damped Newton on the negative second
# derivative; #501). Mirrors `_laplace_mode` from families/laplace.jl and the
# `_gamma_grouped_mode`/`_nb1_grouped_mode` pattern (#479/#503): returns `(z,
# converged)`. `mask` (length-p Bool, or `nothing` = all observed) drops missing
# responses: a masked entry contributes zero score and zero weight, so it
# neither pulls the mode nor enters the Hessian.
#
# Before #501 this loop ran undamped Newton and returned whatever `z` it held
# at `maxiter`, converged or not (the #479/#484/#503 defect class). The per-site
# conditional density here is a nonconvex mixture of two point masses and an
# interior Beta piece, so an undamped step can overshoot across a local ridge
# between two competing modes; the class-audit and independent verification
# (issue #501, `obeta-verify-reproduce.md`) measured the returned z at such a
# point having a central finite-difference gradient that scales as 1/h across
# five step sizes spanning four orders of magnitude — the signature of a jump
# discontinuity in the OUTER marginal negative log-likelihood, not a smooth
# steep slope, which was tripping the outer optimiser's own x/f convergence
# test (the #485-class failure) rather than genuine multimodality.
#
# Now a step that lowers the per-site log-posterior is halved (so small and
# accepted full steps are bit-identical to the old loop), and `converged` is
# true only when the FULL proposed step (not a halved one) is below `tol` — the
# old loop's own stopping test. Unlike Gamma/NB1's LogLink fallback, there is no
# second, alternate curvature to retry with here (`_ob_score_weight` is the
# single, already-clamped-positive weight this family has); a site that still
# fails after the default budget is retried once with a 20x iteration budget
# (mirrors #509's Student-t / #507's NB1 review retry), restarting from z = 0,
# before the caller gives up and reports -Inf.
function _ordered_beta_mode_search(y::AbstractVector, Λ::AbstractMatrix, β::AbstractVector,
        c0::Real, c1::Real, φ::Real; mask = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    z = zeros(K)
    for _ in 1:maxiter
        η = β .+ Λ * z
        s = Vector{Float64}(undef, p)
        W = Vector{Float64}(undef, p)
        @inbounds for t in 1:p
            if mask !== nothing && !mask[t]
                s[t] = 0.0; W[t] = 0.0           # masked ⇒ no contribution
                continue
            end
            st, Wt = _ob_score_weight(y[t], η[t], c0, c1, φ)
            s[t] = st
            W[t] = Wt
        end
        A = Symmetric(Λ' * (W .* Λ) + I)
        Δ = _safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && return z, false
        maximum(abs, Δ) < tol && return z .+ Δ, true
        if norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        else
            q0 = _ordered_beta_logpost(y, Λ, β, c0, c1, φ, z; mask = mask)
            if isfinite(q0)
                accepted = false
                step = 1.0
                for _half in 1:30
                    ztrial = z .+ step .* Δ
                    q1 = _ordered_beta_logpost(y, Λ, β, c0, c1, φ, ztrial; mask = mask)
                    if isfinite(q1) && q1 >= q0
                        z = ztrial
                        accepted = true
                        break
                    end
                    step *= 0.5
                end
                accepted || return z, false
            else
                z = z .+ Δ
            end
        end
    end
    return z, false
end

# Public per-site mode-finder. Retries `_ordered_beta_mode_search` with a 20x
# iteration budget (restarting from z = 0) before giving up — mirrors #509's
# Student-t and #507's NB1 review retry: a genuinely converging site can still
# need more than the default `maxiter = 100` damped steps under
# ill-conditioned curvature. Runs only where the default budget failed, so a
# site that converges within `maxiter` keeps its original path and iteration
# count. Returns `(z, converged)`; used both by `_ordered_beta_loglik_site`
# (below) and by `getLV`/`predict` (further down), which take only `z` — a
# call at converged `(Λ, β, c0, c1, φ)` lands on the mode this same function
# certified during fitting.
function _ordered_beta_mode(y::AbstractVector, Λ::AbstractMatrix, β::AbstractVector,
        c0::Real, c1::Real, φ::Real; mask = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    z, ok = _ordered_beta_mode_search(y, Λ, β, c0, c1, φ; mask = mask, maxiter = maxiter, tol = tol)
    ok || ((z, ok) = _ordered_beta_mode_search(y, Λ, β, c0, c1, φ;
                                               mask = mask, maxiter = 20 * maxiter, tol = tol))
    return z, ok
end

# Per-site Laplace log-marginal:
#   log p(y_s) ≈ ℓ(ẑ) − ½ẑ'ẑ − ½logdet(Λ'WΛ + I).
# `mask` drops the masked entries from the score/weight (via the mode search)
# and from the conditional log-density sum. A mode search that does not converge
# (default budget, then a 20x-budget retry; see `_ordered_beta_mode`)
# must not produce a finite value: `-Inf` makes the fitter's own 1e12 sentinel
# fire instead of scoring a garbage surface (#501, the #479 precedent).
function _ordered_beta_loglik_site(y::AbstractVector, Λ::AbstractMatrix,
        β::AbstractVector, c0::Real, c1::Real, φ::Real;
        mask = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    z, ok = _ordered_beta_mode(y, Λ, β, c0, c1, φ; mask = mask, maxiter = maxiter, tol = tol)
    ok || return -Inf
    η = β .+ Λ * z
    ℓ = 0.0
    W = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        if mask !== nothing && !mask[t]
            W[t] = 0.0                           # masked ⇒ no Hessian weight, no logpdf
            continue
        end
        ℓ += ordered_beta_logp(y[t], η[t], c0, c1, φ)
        _, Wt = _ob_score_weight(y[t], η[t], c0, c1, φ)
        W[t] = Wt
    end
    A = Symmetric(Λ' * (W .* Λ) + I)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

"""
    ordered_beta_marginal_loglik_laplace(Y, Λ, β, c0, c1, φ; mask=nothing, maxiter=100, tol=1e-9) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of an ordered-beta GLLVM.
`Y` is a p×n matrix of responses in `[0,1]` (with exact 0s and 1s allowed); `Λ`
p×K loadings; `β` length-p intercepts; `c0 < c1` the ordered cutpoints; `φ` the
Beta precision. Runs its own per-site Laplace (identity-on-η link). At `Λ = 0`
this reduces exactly to the sum of the independent ordered-beta `logp`.

`mask` (p×n Bool, or `nothing`) marks observed cells — masked (missing) responses
are dropped per site from the score, the Hessian weight, and the log-density sum,
so the marginal is over the observed entries only (invariant to the masked-cell
placeholder).
"""
function ordered_beta_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, c0::Real, c1::Real, φ::Real;
        mask = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask === nothing ? nothing : view(mask, :, i)
        acc += _ordered_beta_loglik_site(view(Y, :, i), Λ, β, c0, c1, φ;
                                         mask = mi, maxiter = maxiter, tol = tol)
    end
    return acc
end

# ---------------------------------------------------------------------------
# Fit driver.
# ---------------------------------------------------------------------------

"""
    OrderedBetaFit

Result of [`fit_ordered_beta_gllvm`](@ref): intercepts `β` (length p), loadings
`Λ` (p×K), the ordered cutpoints `c0 < c1`, the Beta precision `φ`, the maximised
Laplace `loglik`, the optimiser `converged` flag, and `iterations`.
"""
struct OrderedBetaFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    c0::Float64
    c1::Float64
    φ::Float64
    loglik::Float64
    converged::Bool
    iterations::Int
end

# ---------------------------------------------------------------------------
# Post-fit ordination: getLV / predict. The link is identity-on-η (the family
# marker carries c0, c1, φ; μ = σ(η)), so the per-site mode is this file's own
# `_ordered_beta_mode`, and :mean returns the interior Beta mean μ = σ(η).
# ---------------------------------------------------------------------------

_loadings(fit::OrderedBetaFit) = fit.Λ
_loglik(fit::OrderedBetaFit)   = fit.loglik

# Free params: β (p) + reduced loadings Λ + cutpoints (c0, c1) + Beta precision φ.
function _nparams(fit::OrderedBetaFit)
    p, K = size(fit.Λ)
    return p + (p * K - div(K * (K - 1), 2)) + 3       # β + Λ + c0 + c1 + φ
end

"""
    getLV(fit::OrderedBetaFit, Y; rotate=true) -> n×K matrix

Conditional latent-variable scores for an ordered-beta fit: the per-site Laplace
mode `ẑₛ` (`_ordered_beta_mode`) at the fitted `(Λ, β)`, cutpoints `c0 < c1`, and
precision `φ`. `Y` is the `p×n` matrix of responses in `[0,1]`; `rotate=true`
applies the canonical [`rotation`](@ref).
"""
function getLV(fit::OrderedBetaFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y)
    K = size(fit.Λ, 2)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        z, _ = _ordered_beta_mode(view(Y, :, s), fit.Λ, fit.β, fit.c0, fit.c1, fit.φ)
        Z[:, s] = z
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::OrderedBetaFit, Y; type=:mean) -> p×n matrix

In-sample fitted values at the Laplace mode `ẑ` (see [`getLV`](@ref)): `type=:link`
returns the linear predictor `η = β + Λ ẑ`; `type=:mean` returns the interior Beta
mean `μ = logistic(η)` (η clamped). Note `:mean` is the *conditional Beta mean of
the (0,1) interior*, not the unconditional `E[y]` over the full zero/interior/one
mixture (which would weight by the point-mass probabilities).
"""
function predict(fit::OrderedBetaFit, Y::AbstractMatrix{<:Real}; type::Symbol = :mean)
    type in (:link, :mean) ||
        throw(ArgumentError("type must be :link or :mean; got :$type"))
    Z = getLV(fit, Y; rotate = false)                 # n×K
    η = fit.β .+ fit.Λ * Z'                            # p×n
    type === :link && return η
    return _ob_logistic.(_clamp_eta.(η))
end

function Base.show(io::IO, f::OrderedBetaFit)
    p, K = size(f.Λ)
    print(io, "OrderedBetaFit(p=", p, ", K=", K,
          ", c0=", round(f.c0; sigdigits = 4),
          ", c1=", round(f.c1; sigdigits = 4),
          ", φ=", round(f.φ; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_ordered_beta_gllvm(Y; K, c0_init=-1.0, c1_init=1.0, φ_init=nothing, …) -> OrderedBetaFit

Fit an ordered-beta GLLVM by L-BFGS on the Laplace marginal
(`ordered_beta_marginal_loglik_laplace`), jointly estimating the cutpoints
`c0 < c1` (parameterised `c1 = c0 + exp(Δ)` to keep the order) and the Beta
precision `φ`. `Y` is a p×n matrix of responses in `[0,1]`; `K` the latent
dimension. The optimiser θ = `[β(p); pack_lambda(Λ)(rr); c0; Δ; log φ]`. Finite-
difference gradient; warm start = empirical logit-mean intercepts (interior
values only) + an SVD loadings init + a moderate `φ₀`, mirroring `fit_beta_gllvm`.

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or simply include
`missing` entries in `Y` — either way the masked cells are dropped from the
marginal *and* from the warm start, so the fit depends only on the observed cells
(it is invariant to whatever sits in the masked positions).
"""
function fit_ordered_beta_gllvm(Y::AbstractMatrix; K::Integer,
        c0_init::Real = -1.0, c1_init::Real = 1.0, mask = nothing,
        β_init = nothing, Λ_init = nothing, φ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    rr = rr_theta_len(p, K)

    # NA handling: derive the observation mask (explicit `mask`, else from `missing`)
    # and a sanitized response matrix with an in-(0,1) placeholder in masked cells.
    msk = _resolve_obs_mask(mask, Y)
    Yc = _sanitize_missing(Y, 0.5)

    # warm start: empirical logit-mean over interior values (fall back to clamp).
    Zemp = [log(clamp(float(Yc[t, i]), 1e-3, 1 - 1e-3) /
                (1 - clamp(float(Yc[t, i]), 1e-3, 1 - 1e-3))) for t in 1:p, i in 1:n]
    _mask_warmstart!(Zemp, msk)
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
    c0_0  = float(c0_init)
    Δ0    = log(max(float(c1_init) - c0_0, 1e-3))      # c1 = c0 + exp(Δ)

    θ0 = vcat(β0, pack_lambda(Λ0), c0_0, Δ0, logφ0)
    function negll(θ)
        β  = θ[1:p]
        Λ  = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        c0 = θ[p + rr + 1]
        c1 = c0 + exp(θ[p + rr + 2])
        φ  = exp(θ[p + rr + 3])
        v = try
            -ordered_beta_marginal_loglik_laplace(Yc, Λ, β, c0, c1, φ;
                                                  mask = msk, maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂  = Optim.minimizer(res)
    β̂  = θ̂[1:p]
    Λ̂  = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    c0̂ = θ̂[p + rr + 1]
    c1̂ = c0̂ + exp(θ̂[p + rr + 2])
    φ̂  = exp(θ̂[p + rr + 3])
    return OrderedBetaFit(β̂, Λ̂, c0̂, c1̂, φ̂, _fit_verdict(res)...)
end
