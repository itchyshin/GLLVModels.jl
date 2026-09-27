# Conway–Maxwell–Poisson (CMP / COM-Poisson) family for the GLLVModels Laplace path.
#
# A two-parameter count distribution that handles BOTH under- and over-dispersion
# — a capability gllvmTMB does NOT offer. In the *rate* parameterisation,
#
#     P(y) = λ^y / ((y!)^ν · Z(λ,ν)),   y = 0,1,2,…,
#     Z(λ,ν) = Σ_{j=0}^∞ λ^j / (j!)^ν     (the normalising constant),
#
# where ν > 0 is the dispersion exponent:
#     ν = 1 ⇒ Poisson(λ)                  (Z = e^λ; THE correctness anchor),
#     ν > 1 ⇒ underdispersion,
#     ν < 1 ⇒ overdispersion.
# NOTE: λ is the CMP *rate*, NOT the mean (E[y] ≠ λ except at ν=1). This is the
# standard rate form; `predict(..., type=:rate)` therefore returns λ, not E[y].
#
# Link: log link on the rate, λ = exp(η), η = β_t + (Λz_s)_t.
#
# A single latent η drives λ; the family marker carries only the scalar dispersion
# ν. This file runs its OWN per-site Laplace (mirroring ordered_beta.jl /
# beta_binomial.jl): the per-trait score s_t = ∂log p/∂η and weight
# W_t = −∂²log p/∂η² are obtained by ForwardDiff on the scalar map η → log p
# (no hand-derived CMP derivatives), with W_t clamped to ≥ 1e-8 for SPD.
#
# Z(λ,ν) has no closed form for general ν, so its log is computed by summing the
# series to convergence (terms decay fast once j > λ^{1/ν}); the sum is written
# generically so it is ForwardDiff-Dual-safe (η is a Dual during differentiation):
# log(j!) = loggamma(j+1) (SpecialFunctions), and the accumulators take the type
# of η. Truncate when the added term is < 1e-12 of the running sum (hard cap
# 10_000, mirroring the NB1 Fisher-sum idiom in families/negbin1.jl).

"""
    COMPoisson(ν = 1.0)

Conway–Maxwell–Poisson family marker (rate parameterisation). `ν > 0` is the
dispersion exponent: `ν = 1` ⇒ Poisson, `ν > 1` ⇒ underdispersion, `ν < 1` ⇒
overdispersion. Named to avoid colliding with any `Distributions` type. Use with
the unified entry point:

```julia
fit_gllvm(Y; family = COMPoisson(), K = 2)
fit_gllvm(Y; family = COMPoisson(2.0), K = 2)   # same fit — ν is a tag payload
gllvm(@formula(y ~ 1), Y, site_data; family = COMPoisson(), K = 2)
```

The marker's `ν` field is a **tag payload** — it is never read by
[`fit_gllvm`](@ref) / [`fit_compoisson_gllvm`](@ref); the shared dispersion
exponent is always estimated. This is the opposite of [`StudentTFamily`](@ref),
whose `ν` is structural and held fixed. Pass `COMPoisson()` so the public call
need not invent an exponent that is never read from the family instance.
`COMPoisson(1.0)` is the Poisson-anchor dummy (the fitter's own `ν_init`
default).

Named fitter [`fit_compoisson_gllvm`](@ref) remains available. +X,
`disp_group`, and `row_eff` are not admitted on this surface.
"""
struct COMPoisson <: Distribution{Univariate, Discrete}
    ν::Float64
end

# Public-call convenience (mirrors `NB1()` / `DeltaGamma()`): ν is never read
# on the `fit_gllvm` route. 1.0 is the Poisson anchor and the fitter default.
COMPoisson() = COMPoisson(1.0)

const _CMP_LOGZ_TOL = 1e-12
const _CMP_LOGZ_CAP = 10_000

"""
    compoisson_logz(logλ, ν) -> (type of logλ)

Log normalising constant `log Z(λ,ν)` with `λ = exp(logλ)`, computed by summing
`Z = Σ_j exp(j·logλ − ν·loggamma(j+1))` to convergence. Written generically so it
is ForwardDiff-Dual-safe (accumulators take the type of `logλ`). The series is
truncated once an added term is `< 1e-12` of the running sum (hard cap 10_000
terms). At `ν = 1` this returns `λ` exactly (`Z = e^λ`), the Poisson anchor.

When the series' mode `j* = λ^{1/ν}` exceeds 80% of the term cap (where the
truncated sum would be silently low), the function instead returns the
Shmueli et al. (2005) asymptotic
`ν·λ^{1/ν} − ((ν−1)/(2ν))·log λ − ((ν−1)/2)·log 2π − ½·log ν` — exact at
`ν = 1`, `O(1/j*)` relative error otherwise, validated against the series on
the crossover band (rel. err. ≤ 3e-8 for ν ∈ {1.3, 2}).
"""
function compoisson_logz(logλ, ν)
    # float(...): integer arguments must not make T an Int — the asymptotic
    # guard below computes T(0.8) (InexactError for Int T; regression caught
    # by the 2026-08-28 pre-commit review).
    T = float(promote_type(typeof(logλ), typeof(ν)))
    # ASYMPTOTIC BRANCH (2026-08-28, closes the flagged _CMP_LOGZ_CAP
    # limitation): the series' mode sits at j* ≈ λ^{1/ν} = exp(logλ/ν). Once
    # j* approaches the term cap the truncated sum is silently low by orders
    # of magnitude, so beyond 80% of the cap switch to Shmueli et al. (2005,
    # JRSS-C 54:127-142, eq. 2):
    #     log Z ≈ ν·λ^{1/ν} − ((ν−1)/(2ν))·log λ − ((ν−1)/2)·log 2π − ½·log ν,
    # exact at ν = 1 (log Z = λ, the Poisson anchor) and accurate to O(1/j*)
    # relative error at large j* — validated against the series itself on the
    # crossover band j* ∈ [2000, 8000] (test_com_poisson_logz_asymptotic).
    # Dual-safe: arithmetic only, in T.
    jstar = exp(logλ / ν)
    if jstar > T(0.8) * _CMP_LOGZ_CAP
        return ν * jstar - ((ν - one(T)) / (2ν)) * logλ -
               ((ν - one(T)) / 2) * log(2 * T(π)) - log(T(ν)) / 2
    end
    # FIXED 2026-08-26 (root cause: docs/dev-log/check-log.md). The series is
    # Σⱼ exp(logtermⱼ), logtermⱼ = j·logλ − ν·loggamma(j+1), which peaks near
    # j ≈ λ^{1/ν}. NEAR THAT PEAK `logtermⱼ` can be in the thousands even though the
    # properly-normalised `log Z` is small — e.g. logλ=8, ν=1: logterm₂₉₈₁ ≈ 2972,
    # so `exp(logterm)` alone overflows Float64 (max exponent ~709) long before the
    # naive accumulator `Z += exp(logterm)` ever reaches its own convergence check.
    # This is reachable from an ordinary Newton step in the Laplace mode solve, not a
    # constructed corner case (measured: η went from 1.8 to 8.0 in one step on real
    # data). Rewritten as a streaming/online log-sum-exp: track the running maximum
    # `m` and the running sum `S = Σ exp(logtermⱼ − m)` of terms seen so far, rescaling
    # `S` whenever a new term exceeds the current max. `m + log(S)` is then `log Z`
    # without ever exponentiating an unnormalised term. Verified against the naive
    # computation to machine precision everywhere the naive form is itself accurate,
    # and correct (cross-checked against the closed-form Poisson identity
    # log Z(λ, ν=1) = λ) at the value that produced the reported bug — see
    # docs/dev-log/pending/compoisson-logz-fix.jl.
    m = zero(T)
    S = one(T)
    j = 0
    @inbounds while j < _CMP_LOGZ_CAP
        j += 1
        logterm = j * logλ - ν * loggamma(T(j + 1))
        if logterm > m
            S = S * exp(m - logterm) + one(T)
            m = logterm
        else
            S += exp(logterm - m)
        end
        exp(logterm - m) < _CMP_LOGZ_TOL * S && break
    end
    return m + log(S)
end

# The former KNOWN LIMITATION (cap too small once the mode λ^{1/ν} exceeds it,
# silently returning a value low by orders of magnitude) is CLOSED 2026-08-28
# by the asymptotic branch above: past 80% of the cap the Shmueli expansion
# takes over, so no reachable logλ silently truncates.

"""
    compoisson_logpdf(y, η, ν) -> Float64

Scalar CMP log-pmf `log P(y | λ=exp(η), ν)` for one trait, in the rate form
`log p = y·log λ − ν·loggamma(y+1) − log Z(λ,ν)`, with `log λ = η` clamped via
`_clamp_eta` (so `λ ≤ exp(30)`). Dual-safe (`compoisson_logz` accumulates in the
type of `η`). At `ν = 1` this equals the Poisson(λ) log-pmf.
"""
function compoisson_logpdf(y, η, ν)
    logλ = _clamp_eta(η)
    logZ = compoisson_logz(logλ, ν)
    return y * logλ - ν * loggamma(oftype(logλ, y) + 1) - logZ
end

# Per-trait score s_t = ∂log p/∂η and weight W_t = −∂²log p/∂η², via ForwardDiff
# on the scalar map η → log p. W clamped to ≥ 1e-8 to keep Λ'WΛ + I SPD.
function _cmp_score_weight(y, η, ν)
    f = ηv -> compoisson_logpdf(y, ηv, ν)
    g = ηv -> ForwardDiff.derivative(f, ηv)
    s = g(η)
    W = -ForwardDiff.derivative(g, η)
    return s, max(W, 1e-8)
end

# Per-site log-posterior ℓ(z) = Σ_t compoisson_logpdf(y_t, η_t, ν) − ½z'z (mask-aware).
# Used only by `_compoisson_mode`'s damped-step accept/reject test, below.
function _compoisson_mode_logpost(y::AbstractVector, Λ::AbstractMatrix, β::AbstractVector,
        ν::Real, z::AbstractVector; mask = nothing)
    η = β .+ Λ * z
    ℓ = 0.0
    @inbounds for t in eachindex(y)
        (mask !== nothing && !mask[t]) && continue
        ℓ += compoisson_logpdf(y[t], η[t], ν)
    end
    return ℓ - 0.5 * dot(z, z)
end

# Inner Laplace mode-finder for one site (Newton on the negative second
# derivative). Mirrors `_beta_binomial_mode` / `_ordered_beta_mode`. `mask`
# (length-p Bool, or `nothing` = all observed) drops missing responses: a masked
# entry contributes zero score and zero Fisher weight, so it neither pulls the
# mode nor enters the Hessian (mirrors `_laplace_mode` in families/laplace.jl).
#
# Returns `(z, converged)` (#503, following the #479/#480/#484/#500/#507/#509
# pattern). Before this fix the loop ran an undamped Newton step every
# iteration and returned whatever `z` it held at `maxiter`, converged or not —
# measured 7/266 stress-probe sites non-mode (class-wide audit, `origin/claude/
# lane-true-parity-finish-20260925:docs/dev-log/core070/class-audit-20260924/`,
# re-measured against `#503`). `W` is always clamped to `>= 1e-8`
# (`_cmp_score_weight`), so `A = Λ'WΛ + I` is SPD by construction and every
# Newton step solves in a well-defined direction — but the clamp only floors
# the curvature used for the step, it does not certify that the step is a
# descent step on the true per-site log-posterior, so an undamped step can
# still overshoot into a far worse `z` (measured: a tiny, unrelated parameter
# perturbation — 1e-5 relative on `ν` — flips a healthy site from `z=[1.88,
# -0.94]`, log-posterior -13.76, to a divergent `z=[1.46, 3.42]`, log-posterior
# -10590, that gets WORSE with more undamped iterations; this is the mechanism
# behind the sibling-screen `sibling-screen-2026-09-26.md` COM-Poisson anomaly,
# see that report and this PR's body — not a kink in `compoisson_logz`, whose
# value is continuous to ~1e-9 relative error across its branch boundary,
# checked directly and too small to explain a 1e8-1e14 apparent FD gradient).
# Now a step that lowers the per-site log-posterior is halved (the generic
# core's `_laplace_mode` rule, reused here as `_compoisson_mode_logpost`), so
# small steps and accepted full steps are bit-identical to the old loop.
# `converged` requires BOTH the full proposed step below `tol` AND the
# log-posterior gradient itself below `grad_tol`: mirrors `_studentt_grouped_
# mode`'s (#509) extra gradient check — a large, well-conditioned `A` (large
# `W` at an extreme count) can solve `AΔ = g` with a tiny `Δ` while `g` itself
# is still far from zero.
function _compoisson_mode(y::AbstractVector, Λ::AbstractMatrix, β::AbstractVector,
        ν::Real; mask = nothing, maxiter::Integer = 100, tol::Real = 1e-9,
        grad_tol::Real = 1e-6)
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
            st, Wt = _cmp_score_weight(y[t], η[t], ν)
            s[t] = st
            W[t] = Wt
        end
        g = Λ' * s .- z
        A = Symmetric(Λ' * (W .* Λ) + I)
        Δ = _safe_solve(A, g)
        (Δ === nothing || !all(isfinite, Δ)) && return z, false
        maximum(abs, Δ) < tol && maximum(abs, g) < grad_tol && return z .+ Δ, true
        if norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        else
            q0 = _compoisson_mode_logpost(y, Λ, β, ν, z; mask = mask)
            if isfinite(q0)
                accepted = false
                step = 1.0
                for _half in 1:30
                    ztrial = z .+ step .* Δ
                    q1 = _compoisson_mode_logpost(y, Λ, β, ν, ztrial; mask = mask)
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

# Per-site Laplace log-marginal:
#   log p(y_s) ≈ ℓ(ẑ) − ½ẑ'ẑ − ½logdet(Λ'WΛ + I).
# `mask` drops the masked entries from the score/weight (via `_compoisson_mode`)
# and from the conditional log-density sum.
#
# `-Inf` on a search that cannot certify a stationary point (#503; the fitter's
# objective then returns its 1e12 sentinel instead of a garbage finite value).
# The default budget is retried once at 20x `maxiter` (same `tol`), restarting
# from `z=0` — mirrors #507/#509's retry: a genuinely healthy but
# ill-conditioned site can need more damped-Newton steps than the default
# budget without ever being a diverging one.
function _compoisson_loglik_site(y::AbstractVector, Λ::AbstractMatrix,
        β::AbstractVector, ν::Real; mask = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    z, ok = _compoisson_mode(y, Λ, β, ν; mask = mask, maxiter = maxiter, tol = tol)
    ok || ((z, ok) = _compoisson_mode(y, Λ, β, ν; mask = mask, maxiter = 20 * maxiter, tol = tol))
    ok || return -Inf
    η = β .+ Λ * z
    ℓ = 0.0
    W = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        if mask !== nothing && !mask[t]
            W[t] = 0.0                           # masked ⇒ no Hessian weight, no logpdf
            continue
        end
        ℓ += compoisson_logpdf(y[t], η[t], ν)
        _, Wt = _cmp_score_weight(y[t], η[t], ν)
        W[t] = Wt
    end
    A = Symmetric(Λ' * (W .* Λ) + I)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

"""
    compoisson_marginal_loglik_laplace(Y, Λ, β, ν; mask=nothing, link=LogLink(), maxiter=100, tol=1e-9) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a Conway–Maxwell–Poisson
GLLVM with dispersion `ν` (rate parameterisation, log link `λ = exp(η)`). `Y` is the
p×n integer count matrix; `Λ` p×K loadings; `β` length-p intercepts. Runs its own
per-site Laplace (single latent η). At `Λ = 0` this reduces exactly to the sum of
the independent CMP `logp`; at `ν = 1` it equals the Poisson marginal (the anchor).

`mask` (p×n Bool, or `nothing`) marks observed cells — masked (missing) responses
are dropped per site from the score, the Hessian weight, and the log-density sum,
so the marginal is over the observed entries only (the value is invariant to
whatever placeholder sits in the masked cells of `Y`).

The `link` keyword is accepted for interface symmetry with the other count
families but is fixed to the log link on the rate (`λ = exp(η)`); a non-log link is
not meaningful in the rate form.
"""
function compoisson_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, ν::Real; mask = nothing, link::Link = LogLink(),
        maxiter::Integer = 100, tol::Real = 1e-9)
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask === nothing ? nothing : view(mask, :, i)
        acc += _compoisson_loglik_site(view(Y, :, i), Λ, β, ν;
                                       mask = mi, maxiter = maxiter, tol = tol)
    end
    return acc
end

# ---------------------------------------------------------------------------
# Fit driver.
# ---------------------------------------------------------------------------

"""
    COMPoissonFit

Result of [`fit_compoisson_gllvm`](@ref): intercepts `β` (length p), loadings `Λ`
(p×K), the `link` (always the log link on the rate), the dispersion exponent `ν`
(`ν=1` Poisson, `ν>1` under-, `ν<1` overdispersion), the maximised Laplace
`loglik`, the optimiser `converged` flag, and `iterations`.
"""
struct COMPoissonFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    link::Link
    ν::Float64
    loglik::Float64
    converged::Bool
    iterations::Int
end

# ---------------------------------------------------------------------------
# Post-fit ordination: getLV / predict. A single latent η drives λ (the family
# marker carries ν), so the per-site mode is this file's own `_compoisson_mode`,
# and :rate returns the CMP rate λ = exp(η) (NOT the mean).
# ---------------------------------------------------------------------------

_loadings(fit::COMPoissonFit) = fit.Λ
_loglik(fit::COMPoissonFit)   = fit.loglik

# Free params: β (p) + reduced loadings Λ + dispersion ν.
function _nparams(fit::COMPoissonFit)
    p, K = size(fit.Λ)
    return p + (p * K - div(K * (K - 1), 2)) + 1       # β + Λ + ν
end

"""
    getLV(fit::COMPoissonFit, Y; rotate=true) -> n×K matrix

Conditional latent-variable scores for a CMP fit: the per-site Laplace mode `ẑₛ`
(`_compoisson_mode`) at the fitted `(Λ, β)` and dispersion `ν`. `Y` is the `p×n`
integer count matrix; `rotate=true` applies the canonical [`rotation`](@ref).

A site whose mode search does not certify a stationary point at the default
budget is retried at 20x the iterations (mirrors `_compoisson_loglik_site`);
unlike that function there is no `-Inf` sentinel here (a latent score, not a
log-likelihood), so the best `z` the retried search found is still returned.
"""
function getLV(fit::COMPoissonFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y)
    K = size(fit.Λ, 2)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        y = view(Y, :, s)
        z, ok = _compoisson_mode(y, fit.Λ, fit.β, fit.ν)
        ok || ((z, ok) = _compoisson_mode(y, fit.Λ, fit.β, fit.ν; maxiter = 2000))
        Z[:, s] = z
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::COMPoissonFit, Y; type=:rate) -> p×n matrix

In-sample fitted values at the Laplace mode `ẑ` (see [`getLV`](@ref)): `type=:link`
returns the linear predictor `η = β + Λ ẑ`; `type=:rate` returns the CMP **rate**
`λ = exp(η)` (η clamped). NOTE: in the rate parameterisation `λ` is NOT the mean
`E[y]` (they coincide only at `ν=1`); there is no closed-form mean, so `predict`
returns the rate, not the expectation.
"""
function predict(fit::COMPoissonFit, Y::AbstractMatrix{<:Real}; type::Symbol = :rate)
    type in (:link, :rate) ||
        throw(ArgumentError("type must be :link or :rate; got :$type"))
    Z = getLV(fit, Y; rotate = false)                 # n×K
    η = fit.β .+ fit.Λ * Z'                            # p×n
    type === :link && return η
    return exp.(_clamp_eta.(η))
end

function Base.show(io::IO, f::COMPoissonFit)
    p, K = size(f.Λ)
    print(io, "COMPoissonFit(p=", p, ", K=", K, ", link=", nameof(typeof(f.link)),
          ", ν=", round(f.ν; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_compoisson_gllvm(Y; K, link=LogLink(), ν_init=1.0, …) -> COMPoissonFit

Fit a Conway–Maxwell–Poisson GLLVM by L-BFGS over `[β; vec(Λ); log ν]` on the
Laplace marginal ([`compoisson_marginal_loglik_laplace`](@ref)), jointly estimating
the dispersion exponent `ν > 0` (optimised on the log scale). `Y` is a p×n integer
count matrix; `K` the latent dimension. Finite-difference gradient (the Laplace
inner mode-finder is not forward-AD-friendly). Warm start = empirical log-rate
intercepts (`log` of per-species `(mean + 0.5)`) + an SVD (PPCA-style) loadings
init + `ν₀ = ν_init` (default 1, i.e. Poisson).

The log link on the rate is fixed; `link` is accepted for interface symmetry.

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or simply include
`missing` entries in `Y` — either way the masked cells are dropped from the
marginal *and* from the warm start, so the fit depends only on the observed cells
(it is invariant to whatever sits in the masked positions).
"""
function fit_compoisson_gllvm(Y::AbstractMatrix; K::Integer,
        link::Link = LogLink(), ν_init::Real = 1.0, mask = nothing,
        β_init = nothing, Λ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    rr = rr_theta_len(p, K)

    # NA handling: derive the observation mask (explicit `mask`, else from `missing`)
    # and a sanitized count matrix with a safe placeholder (0) in the masked cells.
    msk = _resolve_obs_mask(mask, Y)
    Yc = Integer.(_sanitize_missing(Y, 0))

    # warm start: empirical log-rate intercepts (log of per-species (mean+0.5)) +
    # SVD (PPCA-like) loadings. Masked cells are overwritten with their row's
    # observed mean so neither the intercept nor the SVD sees the placeholder.
    Zemp = [log(max(float(Yc[t, i]) + 0.5, 1e-4)) for t in 1:p, i in 1:n]
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
    logν0 = log(float(ν_init))

    θ0 = vcat(β0, pack_lambda(Λ0), logν0)
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        ν = exp(θ[p + rr + 1])
        v = try
            -compoisson_marginal_loglik_laplace(Yc, Λ, β, ν; mask = msk, link = link,
                                                maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    ν̂ = exp(θ̂[p + rr + 1])
    return COMPoissonFit(β̂, Λ̂, link, ν̂, _fit_verdict(res)...)
end
