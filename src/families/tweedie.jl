# Tweedie (compound Poisson–Gamma, power 1 < p < 2) family pieces for the
# generic Laplace core (src/families/laplace.jl). Responses y ≥ 0: an exact
# point mass at 0 plus a positive continuous part — the standard model for
# biomass / abundance with true zeros. Mean μ = exp(η) (log link), dispersion
# φ > 0, power p ∈ (1, 2), variance function V(μ) = μ^p (Var = φ μ^p).
#
# Exponential-dispersion form (Dunn & Smyth 2005). The μ-dependent kernel of
# log f is  (1/φ)·[ y·μ^{1-p}/(1-p) − μ^{2-p}/(2-p) ]  (μ-free normaliser aside),
# so the Laplace score/weight wrt η need NO series:
#   score  s = (y − μ) μ^{1-p} / φ · (dμ/dη / μ)            (= (y−μ)μ^{1−p}/φ at log link)
#   weight W = (dμ/dη)² / (φ μ^p)                            (expected info = μ^{2−p}/φ)
# The normalising constant a(y, φ, p) is an infinite series (μ-free); it enters
# only the conditional log-density `_glm_logpdf`.

# A plain marker — NOT a Distributions type — carrying the dispersion φ and the
# power p. Used only by the dedicated Tweedie pieces below.
struct TweedieED
    φ::Float64
    p::Float64
end

_clamp_mu(::TweedieED, μ) = max(μ, 1e-12)

# Score/weight wrt η (log link ⇒ me = dμ/dη = μ). General me forms below.
_glm_score(f::TweedieED, μ, n, me, y) = me * (y - μ) / (f.φ * μ^f.p)
_glm_weight(f::TweedieED, μ, n, me)   = me^2 / (f.φ * μ^f.p)

# Observed conditional curvature for TweedieED/log: −∂²ℓ/∂η² = μ^(1-p)·[(2-p)·μ +
# (p-1)·y] / φ. Derivation: the μ-dependent kernel of log f (see the module header)
# is K(η) = (1/φ)[y·μ^(1-p)/(1-p) − μ^(2-p)/(2-p)], μ = e^η; the normalising series
# `_tweedie_logA(y,φ,p)` is μ-FREE (its arguments are y, φ, p only), so it
# contributes zero η-curvature and the claim reduces to differentiating K twice.
# dK/dη = (y·μ^(1-p) − μ^(2-p))/φ (= _glm_score at me=μ, matching the existing
# slot); d²K/dη² = μ^(1-p)·[y(1-p) − (2-p)μ]/φ, so W_obs = −d²K/dη² as above.
# FD-verified against the FULL `_glm_logpdf` (kernel + series, y > 0 and y = 0
# separately) to ≤ 3e-7 relative gap across φ ∈ {0.4,1,2.5}, p ∈ {1.2,…,1.9},
# η ∈ [-1,2], y ∈ {0,…,12.4} (scratchpad/fd_probe_tweedie_probit.jl, 2026-08-28).
# Symmetry check holds: at y = μ this collapses to μ^(2-p)/φ = _glm_weight
# (since (2-p)+(p-1) = 1). ALWAYS NON-NEGATIVE: μ, φ > 0, p ∈ (1,2) ⇒ (2-p) > 0,
# (p-1) > 0, y ≥ 0, so both summands of the bracket are ≥ 0 — unlike Beta/
# Student-t there is no PD-guard-relevant sign-changing region for this family.
_default_hessian(::TweedieED, ::LogLink) = :observed
_glm_obs_weight(f::TweedieED, μ, n, me, y, link::LogLink, η) =
    μ^(1.0 - f.p) * ((2.0 - f.p) * μ + (f.p - 1.0) * y) / f.φ

# Numerically-safe log-sum-exp over a vector of log-weights.
@inline function _tweedie_logsumexp(logw::AbstractVector)
    m = maximum(logw)
    (isfinite(m) || return m)
    s = 0.0
    @inbounds for lw in logw
        s += exp(lw - m)
    end
    return m + log(s)
end

# log a(y, φ, p): the μ-free Dunn–Smyth normalising series (1 < p < 2, y > 0).
#   α = (2-p)/(1-p)  (α < 0 here)
#   logW_j = j·[ -α·log y + α·log(p-1) - (1-α)·log φ - log(2-p) ]
#            - logΓ(j+1) - logΓ(-jα)
#   log a  = -log y + logsumexp_j logW_j
# The summand peaks near j* ≈ y^{2-p} / (φ (2-p)); we sum a window around j*,
# expanding until the boundary terms fall ≳ 37 below the running max.
function _tweedie_logA(y::Float64, φ::Float64, p::Float64)
    α = (2.0 - p) / (1.0 - p)              # < 0 for 1 < p < 2
    # Per-j linear coefficient of the leading term.
    a = -α * log(y) + α * log(p - 1.0) - (1.0 - α) * log(φ) - log(2.0 - p)
    logW(j) = j * a - loggamma(j + 1.0) - loggamma(-j * α)

    jstar = max(1, round(Int, y^(2.0 - p) / (φ * (2.0 - p))))
    drop = 37.0
    cap = 5000
    W = 1
    local lo, hi, m, terms
    while true
        lo = max(1, jstar - W)
        hi = jstar + W
        terms = Float64[logW(float(j)) for j in lo:hi]
        m = maximum(terms)
        edge = max(terms[1], terms[end])
        if (m - edge) ≥ drop || W ≥ cap
            break
        end
        W *= 2
    end
    return -log(y) + _tweedie_logsumexp(terms)
end

"""
    tweedie_logpdf(y, μ, φ, p) -> Float64

Scalar Tweedie (compound Poisson–Gamma, power `1 < p < 2`) log-density in the
exponential-dispersion form (Dunn & Smyth 2005), with mean `μ`, dispersion `φ`
and `Var = φ μ^p`. Handles the exact point mass at `y = 0` and the positive
continuous part (the latter via the μ-free normalising series `log a(y,φ,p)`).
"""
function tweedie_logpdf(y::Real, μ::Real, φ::Real, p::Real)
    y = float(y); μ = float(μ); φ = float(φ); p = float(p)
    μ = max(μ, 1e-12)
    if y == 0.0
        # series term is 0, log a(0) = 0
        return -μ^(2.0 - p) / (φ * (2.0 - p))
    else
        kernel = (y * μ^(1.0 - p) / (1.0 - p) - μ^(2.0 - p) / (2.0 - p)) / φ
        return kernel + _tweedie_logA(y, φ, p)
    end
end

_glm_logpdf(f::TweedieED, μ, n, y) = tweedie_logpdf(y, μ, f.φ, f.p)

"""
    tweedie_cdf(y, μ, φ, p) -> Float64

Tweedie (compound Poisson–Gamma, `1 < p < 2`) CDF `P(Y ≤ y)` for `y ≥ 0`: the exact
atom `P(Y = 0) = exp(−μ^{2−p}/(φ(2−p)))` plus, for `y > 0`, the integral of the
positive continuous density (`exp(tweedie_logpdf)`) over `(0, y]` by composite
Simpson quadrature. Used by the Dunn–Smyth residual.
"""
function tweedie_cdf(y::Real, μ::Real, φ::Real, p::Real)
    y = float(y); μ = float(μ); φ = float(φ); p = float(p)
    F0 = exp(tweedie_logpdf(0.0, μ, φ, p))
    y <= 0 && return F0
    return clamp(F0 + _tweedie_cdf_pos(y, μ, φ, p), 0.0, 1.0)
end

# Integral of the positive-part Tweedie density over (0, y] by composite Simpson's
# rule (the density is smooth and bounded on (0, y]; the `m`-point grid is ample for
# the residual PIT). The 0 endpoint is finite (the density → 0 there) but evaluated
# at a tiny ε to avoid log(0).
function _tweedie_cdf_pos(y::Float64, μ::Float64, φ::Float64, p::Float64)
    m = 200                                   # even → m+1 grid points
    h = y / m
    ε = 1e-12 * y
    f(t) = exp(tweedie_logpdf(max(t, ε), μ, φ, p))
    s = f(0.0) + f(y)
    @inbounds for i in 1:(m - 1)
        s += (isodd(i) ? 4.0 : 2.0) * f(i * h)
    end
    return s * h / 3.0
end

"""
    tweedie_marginal_loglik_laplace(Y, Λ, β, φ, p; mask=nothing, link=LogLink(), kwargs...) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a Tweedie GLLVM with
dispersion `φ` and power `p ∈ (1,2)` — responses `Y ≥ 0` with a point mass at 0,
mean `μ = exp(η)` (log link), `Var = φ μ^p`. A thin wrapper over the
family-generic `marginal_loglik_laplace` with the `TweedieED(φ, p)` marker.

`mask` (p×n Bool, or `nothing`) marks observed cells — masked (missing) responses
are dropped per site from the marginal (gllvm-style NA handling), so the value is
invariant to whatever placeholder sits in the masked cells of `Y`.
"""
tweedie_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector,
        φ::Real, p::Real; mask = nothing, link::Link = LogLink(), maxiter::Integer = 100,
        tol::Real = 1e-9, kwargs...) =
    marginal_loglik_laplace(TweedieED(float(φ), float(p)), Y, ones(Int, size(Y)),
                            Λ, β, link; mask = mask, maxiter = maxiter, tol = tol, kwargs...)

# ---------------------------------------------------------------------------
# Fit driver.
# ---------------------------------------------------------------------------

# Value the packed objective returns when the Laplace marginal cannot be
# evaluated. It must be finite so the line search retreats rather than stalling
# on a NaN, but it is a failure marker, not a log-likelihood: `_tweedie_verdict`
# refuses to report a point still sitting on it.
const _TWEEDIE_FAIL_PENALTY = 1e12

# |ξ| beyond this puts the power within ~2e-9 of 1 or 2 — a run that has
# diverged along the (φ, power) ridge, not an estimate on the open interval the
# `TweedieFit` docstring promises. Any genuine Tweedie power (1.1–1.9) sits at
# |ξ| < 2.2, so the bound never fires on a real optimum.
const _TWEEDIE_XI_MAX = 20.0

# Data-scaled offset for the log warm start. Responses have an exact atom at 0,
# and `log(max(y, 1e-6))` maps every structural zero to −13.8 whatever the data
# scale: that drags the intercepts far below the log-mean and inflates the SVD
# loadings, landing the start on a part of the Laplace marginal from which the
# optimiser cannot recover. Offsetting by a fraction of the mean *positive*
# observed response keeps the zeros on the data's own scale.
function _tweedie_log_offset(Yc::AbstractMatrix, msk)
    tot = 0.0
    cnt = 0
    @inbounds for i in eachindex(Yc)
        (msk === nothing || msk[i]) || continue
        Yc[i] > 0 || continue
        tot += Yc[i]
        cnt += 1
    end
    cnt == 0 && return 1e-6
    return 0.1 * (tot / cnt)
end

"""
    _tweedie_verdict(optim_converged, gres, nll, ξ, g_tol) -> (converged, loglik, reason)

Convergence contract for [`fit_tweedie_gllvm`](@ref) and
[`fit_tweedie_gllvm_grouped`](@ref). `Optim`'s own verdict is
necessary but not sufficient here: on this objective it fires `f_converged` on a
relative step test while the gradient residual is still ~1e15 (the optimiser
stalled), and `g_converged` on the flat failure plateau where the
finite-difference gradient is exactly zero. Three additional checks gate the
reported flag, and the failure sentinel is never reported as a log-likelihood:

- `:objective_failed` — the objective at the returned point is still at
  `_TWEEDIE_FAIL_PENALTY`; the marginal was never evaluated successfully.
  `loglik` is `-Inf`, not `-1e12`.
- `:power_at_boundary` — `|ξ| > _TWEEDIE_XI_MAX`, i.e. the power has run to the
  closed end of `(1, 2)`.
- `:gradient_not_small` — the gradient residual is large relative to the
  objective's own scale, so the point is not a stationary point.
"""
function _tweedie_verdict(optim_converged::Bool, gres::Real, nll::Real, ξ::Real, g_tol::Real)
    (isfinite(nll) && nll < _TWEEDIE_FAIL_PENALTY) ||
        return (false, -Inf, :objective_failed)
    abs(ξ) <= _TWEEDIE_XI_MAX || return (false, -float(nll), :power_at_boundary)
    (isfinite(gres) && gres <= max(g_tol, g_tol * abs(nll))) ||
        return (false, -float(nll), :gradient_not_small)
    return (optim_converged, -float(nll), :ok)
end

# The per-species grouped contract has one unconstrained power coordinate per
# response. Reuse the scalar objective/gradient checks, but require every
# coordinate to remain interior; one boundary coordinate makes the whole
# parameter vector an invalid Tweedie estimate.
function _tweedie_verdict(optim_converged::Bool, gres::Real, nll::Real,
                          ξ::AbstractVector, g_tol::Real)
    (isfinite(nll) && nll < _TWEEDIE_FAIL_PENALTY) ||
        return (false, -Inf, :objective_failed)
    all(isfinite, ξ) && all(abs.(ξ) .<= _TWEEDIE_XI_MAX) ||
        return (false, -float(nll), :power_at_boundary)
    (isfinite(gres) && gres <= max(g_tol, g_tol * abs(nll))) ||
        return (false, -float(nll), :gradient_not_small)
    return (optim_converged, -float(nll), :ok)
end

"""
    TweedieFit

Result of [`fit_tweedie_gllvm`](@ref): intercepts `β` (length p), loadings `Λ`
(p×K), the estimated dispersion `φ` and power `p ∈ (1,2)` (`Var = φ μ^p`), the
`link`, the maximised Laplace `loglik`, the optimiser `converged` flag, and
`iterations`.
"""
struct TweedieFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    φ::Float64
    p::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
end

# Positional compatibility constructor (2026-08-28): every pre-existing
# construction site builds a default-curvature fit; the `hessian` field
# records the objective identity so `confint`/bootstrap can rebuild THE
# SAME objective instead of guessing (the audit's confint-consistency class).
TweedieFit(β, Λ, φ, p, link, loglik, converged, iterations) =
    TweedieFit(β, Λ, φ, p, link, loglik, converged, iterations, _default_hessian(TweedieED(1.0, 1.5), link))

function Base.show(io::IO, f::TweedieFit)
    pp, K = size(f.Λ)
    print(io, "TweedieFit(p=", pp, ", K=", K,
          ", φ=", round(f.φ; sigdigits = 4),
          ", power=", round(f.p; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

# TweedieED opts into the damped mode-search backtracking (#623). The 2026-08-27
# opt-in was reverted because the merit function evaluated the full log-density,
# whose series normaliser made the "Tweedie engine health" testset take 48m20s.
# The merit now uses only the μ-dependent kernel below: the series term c(y, φ)
# is constant in μ, so it cancels in every q1 >= q0 comparison and the accepted
# steps are the same as with the full density. Undamped, the K = 2 search
# diverged: on the #623 fixture it returned ẑ = [-5.2e10, 8.4e8] where the mode
# is [-1.975, 0.389], and the site's Laplace value was off by 1.35e21.
_laplace_mode_should_backtrack(::TweedieED) = true
function _laplace_mode_merit_term(f::TweedieED, μ, n, y)
    μ = max(μ, 1e-12)                       # as in `tweedie_logpdf`
    return (y * μ^(1 - f.p) / (1 - f.p) - μ^(2 - f.p) / (2 - f.p)) / f.φ
end
_laplace_mode_step_weight(f::TweedieED, μ, n, me, y, link::LogLink, η) =
    ismissing(y) ? _glm_weight(f, μ, n, me) :
    max(_glm_obs_weight(f, μ, n, me, y, link, η), _glm_weight(f, μ, n, me))

"""
    fit_tweedie_gllvm(Y; K, link=LogLink(), φ_init=1.0, p_init=1.5, …) -> TweedieFit

Fit a Tweedie GLLVM by L-BFGS over `[β; pack_lambda(Λ); log φ; ξ]` on the Laplace
marginal (`tweedie_marginal_loglik_laplace`), jointly estimating the dispersion
`φ` and power `p`. The power is mapped to `(1,2)` by `p = 1 + 1/(1+exp(-ξ))`
(so `ξ = 0 ⇒ p = 1.5`). `Y` is a p×n matrix of non-negative reals (a point mass
at 0 allowed); `K` the latent dimension. Finite-difference gradient; warm start =
log row-means of `Y + c` as intercepts + SVD of row-centred log-`(Y + c)` as
loadings + `logφ₀ = log(φ_init)`, `ξ₀ = logit(p_init − 1)`, where the offset
`c = 0.1 · mean(Y[Y > 0])` keeps the exact zeros on the data's own scale.

`converged` is not `Optim`'s verdict alone: it additionally requires that the
Laplace marginal was evaluated successfully at the returned point, that the
power is strictly interior to `(1, 2)`, and that the gradient residual is small
relative to the objective's scale (see `_tweedie_verdict`). A fit that fails any
of these reports `converged = false` rather than advertising a stalled,
boundary, or unevaluable point as a maximum.

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or simply include
`missing` entries in `Y` — either way the masked cells are dropped from the
marginal *and* from the warm start, so the fit depends only on the observed cells
(it is invariant to whatever sits in the masked positions).

`hessian` selects the Laplace log-det curvature only (`:fisher` expected /
`:observed` joint — TMB's choice); the inner mode search is always
Fisher-scored. Tweedie/log defaults to `:observed` so its Laplace curvature
matches TMB/`gllvmTMB`. Omitting it is exactly the default-path behaviour.
"""
function fit_tweedie_gllvm(Y::AbstractMatrix; K::Integer,
        link::Link = LogLink(), φ_init::Real = 1.0, p_init::Real = 1.5, mask = nothing,
        hessian::Symbol = _default_hessian(TweedieED(1.0, 1.5), link),
        β_init = nothing, Λ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p_sp, n = size(Y)
    rr = rr_theta_len(p_sp, K)
    hessian in (:fisher, :observed) || throw(ArgumentError(
        "fit_tweedie_gllvm: hessian must be :fisher or :observed; got :$hessian"))

    # NA handling: derive the observation mask (explicit `mask`, else from `missing`)
    # and a sanitized response matrix with a safe placeholder (0) in the masked cells.
    msk = _resolve_obs_mask(mask, Y)
    Yc = float.(_sanitize_missing(Y, 0))

    Zemp = log.(Yc .+ _tweedie_log_offset(Yc, msk))
    _mask_warmstart!(Zemp, msk)
    β0 = β_init === nothing ? vec(sum(Zemp; dims = 2)) ./ n : collect(float.(β_init))
    Λ0 = if Λ_init === nothing
        Zc = Zemp .- β0
        F = svd(Zc)
        kk = min(K, length(F.S))
        L = zeros(p_sp, K)
        @inbounds for j in 1:kk
            L[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
        end
        L
    else
        collect(float.(Λ_init))
    end
    logφ0 = log(float(φ_init))
    ξ0 = log((float(p_init) - 1.0) / (2.0 - float(p_init)))   # logit(p_init - 1)

    θ0 = vcat(β0, pack_lambda(Λ0), logφ0, ξ0)
    function negll(θ)
        β = θ[1:p_sp]
        Λ = unpack_lambda(θ[(p_sp + 1):(p_sp + rr)], p_sp, K)
        φ = exp(θ[p_sp + rr + 1])
        ξ = θ[p_sp + rr + 2]
        pw = 1.0 + 1.0 / (1.0 + exp(-ξ))
        v = try
            -tweedie_marginal_loglik_laplace(Yc, Λ, β, φ, pw;
                                             hessian = hessian,
                                             mask = msk, link = link,
                                             maxiter = newton_maxiter, tol = newton_tol)
        catch
            return _TWEEDIE_FAIL_PENALTY
        end
        return isfinite(v) ? v : _TWEEDIE_FAIL_PENALTY
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p_sp]
    Λ̂ = unpack_lambda(θ̂[(p_sp + 1):(p_sp + rr)], p_sp, K)
    φ̂ = exp(θ̂[p_sp + rr + 1])
    ξ̂ = θ̂[p_sp + rr + 2]
    p̂ = 1.0 + 1.0 / (1.0 + exp(-ξ̂))
    conv, loglik, reason = _tweedie_verdict(Optim.converged(res), Optim.g_residual(res),
                                            Optim.minimum(res), ξ̂, g_tol)
    if reason === :objective_failed
        @warn "fit_tweedie_gllvm: the Laplace marginal could not be evaluated at any \
               accepted point; returning converged = false and loglik = -Inf. Try a \
               different `p_init` / `φ_init`, or check `Y` for extreme values."
    elseif reason === :power_at_boundary
        @warn "fit_tweedie_gllvm: the power ran to the boundary of (1, 2) \
               (p̂ = $(p̂), φ̂ = $(φ̂)); the fit is flagged as not converged."
    end
    return TweedieFit(β̂, Λ̂, φ̂, p̂, link, loglik, conv, Optim.iterations(res), hessian)
end
