using SpecialFunctions: logbeta

# Zero-truncated NB2 family for the generic Laplace core.
#
# Twin gllvmTMB family_id 11 (`truncated_nbinom2()`, log link; y ≥ 1 strictly):
#   ℓ = log NB2(y; μ, φ) − log(1 − p0),   μ = exp(η), p0 = (φ/(μ+φ))^φ,
#   φ = exp(log_phi_truncnb2) per trait on the twin.
# Arc1 Julia = shared scalar r ≡ φ, pack [β; pack(Λ); log r] (length p+rr+1).
# Arc1b = per-trait r_t ≡ twin log_phi_truncnb2, pack [β; pack(Λ); log r_1…log r_p]
#   (length p+rr+p); rvec = exp.(θ[tail]); fams = TruncatedNegBin2.(rvec);
#   mode via `_grouped_laplace_mode` (do not edit grouped_dispersion.jl).
# Score / weight (log link) — NB2 chain-rule factor a = r_t/(r_t+μ) is REQUIRED:
#   μ_tr = μ/(1−p0);  Var_tr = (V+μ²)/(1−p0) − μ_tr²,  V = μ + μ²/r_t;
#   s = a · (y − μ_tr);  W = a² · Var_tr.
# (Do NOT omit `a`; Sol ceiling 2026-08-15: bare (y−μ_tr) mismatches dℓ/dη.)
# Cite: gllvmTMB `src/gllvmTMB.cpp` fid==11; Identity 2026-08-15-truncated-nbinom2-identity.md.

"""
    TruncatedNegBin2(r)
    TruncatedNegBin2()

Marker for zero-truncated NB2 (support `{1,2,…}`; log link on the *untruncated*
mean `μ = exp(η)`; dispersion `r` with `Var = μ + μ²/r` ≡ twin `φ`).
`TruncatedNegBin2()` uses a warm-start default `r = 10` for `fit_gllvm` dispatch;
the fitter jointly estimates shared `r`. Distinct from Distributions.jl
`NegativeBinomial` and from the hurdle occurrence×truncated two-part family.
"""
struct TruncatedNegBin2{T<:Real}
    r::T
end
TruncatedNegBin2() = TruncatedNegBin2{Float64}(10.0)

_clamp_mu(::TruncatedNegBin2, μ) = max(μ, 1e-12)

# Truncated mean / variance helpers (untruncated μ > 0, r > 0).
function _truncnb2_mean_var(μ, r)
    # Keep the mean information when r/(r+μ) would round to one.
    logp0 = -r * log1p(μ / r)
    p0 = exp(logp0)
    denom = -expm1(logp0)
    μtr = μ / denom
    var_tr = μtr * (1 + μ / r - μ * p0 / denom)
    return μtr, var_tr
end

# NB2 mean-to-η factor a = r/(r+μ). Ordinary NB2 log-link score is a·(y−μ);
# truncated replaces μ with μ_tr. General link: ∂ℓ/∂η = a · (y − μ_tr)/μ · me.
function _glm_score(f::TruncatedNegBin2, μ, n, me, y)
    r = f.r
    μtr, _ = _truncnb2_mean_var(μ, r)
    a = inv(1 + μ / r)
    return a * (y - μtr) / μ * me
end

function _glm_weight(f::TruncatedNegBin2, μ, n, me)
    r = f.r
    _, var_tr = _truncnb2_mean_var(μ, r)
    a = inv(1 + μ / r)
    return (a * me / μ)^2 * var_tr
end

# Exact negative conditional curvature −∂²ℓ/∂η² for zero-truncated NB2 at the LOG
# link, where ℓ = log NB2(y; μ, r) − log(1 − p₀) and p₀ = (r/(r+μ))^r.
#
# This is the curvature TMB's Laplace uses (observed joint Hessian). It differs from
# the Fisher weight above because the NB2 term is **y-dependent**:
#
#   −∂²ℓ_nb/∂η²    = μ r (y + r) / (μ + r)²                        (y enters here)
#   −∂²ℓ_trunc/∂η² = −p₀A²/(1−p₀)² + [p₀/(1−p₀)]·μr²/(μ+r)²,   A = −μr/(μ+r)
#
# Substituting E[y] = μ in the first term recovers μr/(μ+r), the untruncated NB2
# Fisher weight — which is precisely why Fisher ≢ observed here, unlike truncated
# Poisson (fid 10), where y enters η linearly and the two coincide pointwise. That
# distinction is why the fid-10 cell paid legitimately through the Fisher core while
# fid 11 could not.
#
# Verified against ForwardDiff to a max relative error of 1.8e-13 over 125 (μ, r, y)
# cells spanning μ ∈ [0.5, 25], r ∈ [0.3, 50], y ∈ [1, 40].
#
# Log link only: the twin restricts fid 11 to the log link (`R/fit-multi.R:844-845`)
# and the Julia fitters enforce `LogLink` as well.
function _truncnb2_observed_weight(f::TruncatedNegBin2, μ, y, link::Link)
    link isa LogLink || throw(ArgumentError(
        "hessian=:observed for truncated_nbinom2 is supported only with LogLink()"))
    r = f.r
    μtr, var_tr = _truncnb2_mean_var(μ, r)
    a = inv(1 + μ / r)
    # Exact observed = Fisher + response-dependent score-factor derivative.
    # Do not form 1-a, which loses the small correction at large r.
    return a^2 * (var_tr + (μ / r) * (y - μtr))
end

# Dispatch helper, mirroring `_nb_grouped_laplace_weight` in grouped_dispersion.jl.
# ---------------------------------------------------------------------------
# Curvature contract wiring (2026-08-25).
#
# This family's OWN kernels already default to `:observed` — it was among the
# first fixed (instance 2, on main via #263). But the GENERIC core
# (`families/laplace.jl`) was never told, so `_default_hessian` fell through to
# the global `:fisher` and the two routes returned DIFFERENT log-likelihoods for
# the same model. Measured on a p=5, K=1, n=40 fixture (seed 77):
#
#     generic core                 = -408.8988683230   (:fisher, by fallthrough)
#     truncated_nbinom2 own kernel = -408.9397531377   (:observed, by its default)
#     abs Δ                        =  4.088e-02
#
# and the own kernel forced to `:fisher` reproduces the core EXACTLY, which is
# what proves the curvature default was the only difference.
#
# Same class of defect as the R bridge routing `family = "gamma"` and
# `["gamma", …]` to different kernels: one model, two answers, depending on
# which entry point the caller happened to use.
_default_hessian(::TruncatedNegBin2, ::LogLink) = :observed

# Analytic override so the generic core and this family's own kernel compute the
# SAME formula rather than AD-vs-closed-form. One formula, one place — the same
# reasoning as the Gamma override and the DeltaGamma fix.
_glm_obs_weight(f::TruncatedNegBin2, μ, n, me, y, link::LogLink, η) =
    _truncnb2_laplace_weight(:observed, f, μ, me, y, link)

function _truncnb2_laplace_weight(hessian::Symbol, f::TruncatedNegBin2, μ, me, y,
        link::Link)
    hessian === :fisher && return _glm_weight(f, μ, 1, me)
    hessian === :observed || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    return _truncnb2_observed_weight(f, μ, y, link)
end

# For large r, differentiating a beta normalizer can subtract nearly equal
# digammas. Expand sum(k=0:y-1) log1p(k/r) with a bounded alternating remainder.
# Five power sums are closed form, so work does not grow with the count y.
function _truncnb2_logrise_series(r, y::Int)
    y == 1 && return zero(r), true
    n = oftype(r, y - 1)
    x = n / r
    first = y * x / 2
    admissible = x <= 0.01 && y * x^6 / 6 <= eps(typeof(float(r))) * max(one(r), abs(first))
    admissible || return zero(r), false
    z = inv(n)
    value = y * x * (1/2 - x * (2 + z) / 12 + x^2 * (1 + z) / 12 -
                    x^3 * (2 + z) * (3 + 3z - z^2) / 120 +
                    x^4 * (1 + z) * (2 + 2z - z^2) / 60)
    return value, true
end

function _glm_logpdf(f::TruncatedNegBin2, μ, n, y)
    yi = Int(y)
    yi < 1 && return oftype(μ, -Inf)
    r = f.r
    log1pmur = log1p(μ / r)
    logp0 = -r * log1pmur
    log_nz = log(-expm1(logp0))
    rise, series = _truncnb2_logrise_series(r, yi)
    if series
        return rise + yi * log(μ) - loggamma(float(yi) + 1) +
               logp0 - yi * log1pmur - log_nz
    end
    # Algebra of Distributions.NegativeBinomial's beta normalizer, with log
    # probabilities computed from μ/r rather than a rounded success probability.
    return logp0 + yi * (log(μ) - log(r) - log1pmur) -
           log(r + yi) - logbeta(r, float(yi) + 1) - log_nz
end

_laplace_mode_should_backtrack(::TruncatedNegBin2) = true

# ---------------------------------------------------------------------------
# Laplace breakdown guard (Julia-side; the same design as ZI_LAPLACE_EIGMIN_FLOOR on
# the zi_* route, PR #557).
#
# The observed truncated-NB2 curvature `_truncnb2_observed_weight` is negative for
# small r (for example -0.033 at r = 0.2, y = 1, mu = 7.4), and it grows more negative
# as r -> 0. The site precision A = I + Λ' diag(W) Λ can then fall towards 0 while the
# mode search still converges, and -1/2 logdet(A) inflates the Laplace value. A PD
# check cannot catch this: at a certified mode A is the negative Hessian of the site
# log-posterior and is positive semi-definite anyway; the failure is PD but
# near-singular. Impossible-loglik audit (2026-09-27, F1), p = 4, n = 150, K = 1,
# r = 0.3: 3 of 12 default fits reported converged = true with a smallest site
# eigenvalue of 8e-6 to 1.3e-4 and a Laplace value 40 to 139 units above the exact
# marginal (4001-point quadrature). On one draw the breakdown point was the global
# maximum of the Laplace objective (-1663.1 vs -1686.3 at the healthy optimum, whose
# exact marginal is 55 units higher).
#
# Floor = 0.1, as on the zi_* route. Healthy truncated-NB2 optima measured 0.79 to
# 9.6 (audit) and the sweep in docs/dev-log/decisions/
# 2026-09-27-truncnb2-laplace-breakdown-guard.md, so the guard does not touch them.
#
# The floor is safe for the optimiser only together with the "within 10% of the
# floor -> converged = false" rule in fit_truncated_nbinom2_gllvm: Optim reports
# converged = true for a fit stalled at the wall (a line search into the 1e12
# sentinel becomes a zero step), so that rule is load-bearing. Do not relax it as
# redundant.
# ---------------------------------------------------------------------------
# The public marginal functions stay unguarded unless `eigmin_floor` is passed.
const TRUNCNB2_LAPLACE_EIGMIN_FLOOR = 0.1

"""
    truncated_nbinom2_marginal_loglik_laplace(Y, Λ, β, r; link=LogLink(),
                                              hessian=:observed, kwargs...) -> Float64

Laplace log-marginal for a zero-truncated NB2 GLLVM with shared dispersion `r`.
`Y` must be integer counts with every observed cell `≥ 1`.

`hessian=:observed` (the default) uses TMB's observed Laplace curvature;
`hessian=:fisher` retains the expected-information approximation.
`eigmin_floor` (default `-Inf`, unguarded) returns `-Inf` when any site's Laplace
precision has an eigenvalue below it; see `GLLVModels.TRUNCNB2_LAPLACE_EIGMIN_FLOOR` = 0.1.

Implemented as the **equal-`r_t` special case** of
[`truncated_nbinom2_pertrait_marginal_loglik_laplace`](@ref) rather than through the
generic Laplace core. Two reasons (2026-08-24):

1. The generic core hard-codes the **Fisher** weight with no `hessian` keyword, so
   routing through it would leave the shared route on a different objective from TMB —
   and from the per-trait route, which now defaults to `:observed`.
2. It makes *"equal `r_t` reduces to shared `r`"* true **by construction** instead of
   an invariant that has to be asserted and can silently break. That invariant
   (`test/test_truncated_nbinom2.jl` "Arc1b: equal r_t reduces to shared-r ll") is
   exactly what caught the asymmetry when only the per-trait route was converted.

`laplace.jl` is deliberately untouched (Arc1b amendment fences it).
"""
truncated_nbinom2_marginal_loglik_laplace(Y::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, r::Real;
        link::Link = LogLink(), kwargs...) =
    truncated_nbinom2_pertrait_marginal_loglik_laplace(
        Y, Λ, β, fill(float(r), size(Λ, 1)); link = link, kwargs...)

"""
    TruncatedNegBin2Fit

Result of [`fit_truncated_nbinom2_gllvm`](@ref): intercepts `β`, loadings `Λ`,
shared dispersion `r` (`Var = μ + μ²/r` ≡ twin `φ`), link, loglik, convergence.
`min_site_eigen` is the smallest eigenvalue of the per-site Laplace precision
`I + Λ' diag(W) Λ` at the optimum; a fit within 10% of the breakdown guard
(`GLLVModels.TRUNCNB2_LAPLACE_EIGMIN_FLOOR` = 0.1) is reported with `converged = false`.
"""
struct TruncatedNegBin2Fit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    r::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    theta_packed::Vector{Float64}
    min_site_eigen::Float64
end

function Base.show(io::IO, f::TruncatedNegBin2Fit)
    p, K = size(f.Λ)
    print(io, "TruncatedNegBin2Fit(p=", p, ", K=", K,
          ", r=", round(f.r; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

# Verdict at the dispersion boundary for the truncated-NB2 fitters. Any r below 1e-6
# (degenerate: extreme overdispersion) makes the fit not converged, with a warning.
# Any r above 1e6 (the Poisson limit) only warns: r is not identified there, but the
# rest of the fit is usually sound, and reporting it as not converged would flag
# ordinary fits (trait 5 of the seed-58 parity data ends at r = 9.5e9 while the
# log-likelihood matches gllvmTMB to 8e-7).
function _truncnb2_dispersion_verdict(converged::Bool, r::AbstractVector{<:Real},
                                      who::AbstractString)
    low = findall(<(1e-6), r)
    high = findall(>(1e6), r)
    isempty(high) || @warn "$who: r for trait(s) $high is above 1e6 (the Poisson limit); " *
        "r is not identified for them on this data, but the fit is otherwise reported as is."
    if !isempty(low)
        @warn "$who: r for trait(s) $low is below 1e-6 (the dispersion boundary, " *
              "extreme overdispersion); the fit is degenerate and reported as not converged."
        converged = false
    end
    return converged
end

"""
    fit_truncated_nbinom2_gllvm(Y; K, link=LogLink(), …) -> TruncatedNegBin2Fit

Fit a zero-truncated NB2 GLLVM by Laplace + LBFGS (finite-difference outer
gradient) over `[β; pack(Λ); log r]` (shared scalar `r`; length `p+rr+1`).
Twin per-trait `log_phi_truncnb2` is [`fit_truncated_nbinom2_gllvm_pertrait`](@ref)
(Arc1b). Twin-aligned: log link on untruncated `μ`, support `y ≥ 1`.
Throws if any observed cell is `< 1`.

`eigmin_floor` sets the Laplace breakdown guard
(`GLLVModels.TRUNCNB2_LAPLACE_EIGMIN_FLOOR` = 0.1; `-Inf` disables it). A fit that
ends at the guard is retried once with a moment-based start for `r`; if it still ends
there, the higher-loglik of the two fits is reported with `converged = false` and a
warning. An estimate of `r` below 1e-6 (the dispersion boundary: extreme
overdispersion) is reported with `converged = false` and a warning; above 1e6 (the
Poisson limit, where `r` is not identified) the fit only warns.
"""
function fit_truncated_nbinom2_gllvm(Y::AbstractMatrix; K::Integer,
        link::Link = LogLink(), mask = nothing, offset = nothing,
        hessian::Symbol = :observed,
        eigmin_floor::Real = TRUNCNB2_LAPLACE_EIGMIN_FLOOR,
        β_init = nothing, Λ_init = nothing, r_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    link isa LogLink || throw(ArgumentError(
        "fit_truncated_nbinom2_gllvm: only LogLink is supported (twin truncated_nbinom2)"))
    # Validated up front, NOT inside negll: that objective wraps its body in a
    # try/catch converting any throw into 1e12, which would launder a typo'd symbol
    # into a converged-looking garbage fit.
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_truncated_nbinom2_gllvm: hessian must be :observed or :fisher; got :$hessian"))
    p, n = size(Y)
    rr = rr_theta_len(p, K)
    msk = mask === nothing ? (any(ismissing, Y) ? observed_mask(Y) : nothing) : mask
    Yc = Integer.(_sanitize_missing(Y, 1))   # placeholder 1 for masked (never enters ℓ)
    @inbounds for t in 1:p, s in 1:n
        (msk !== nothing && !msk[t, s]) && continue
        Yc[t, s] < 1 && throw(ArgumentError(
            "truncated_nbinom2 requires y ≥ 1; found y=$(Yc[t, s]) at ($t,$s)"))
    end

    Zemp = [linkfun(link, max(Float64(Yc[t, i]), 1.0)) for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)
    if msk !== nothing
        @inbounds for t in 1:p
            obs = view(msk, t, :)
            cnt = count(obs)
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

    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        r = exp(θ[p + rr + 1])
        v = try
            -truncated_nbinom2_marginal_loglik_laplace(Yc, Λ, β, r;
                                     link = link, mask = msk, offset = offset,
                                     hessian = hessian, eigmin_floor = eigmin_floor,
                                     maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    # One L-BFGS run from a start; returns the optimum, Optim's verdict and the
    # smallest site eigenvalue there.
    function run_from(Λstart, logrstart)
        res = Optim.optimize(negll, vcat(β0, pack_lambda(Λstart), logrstart), ls, opts;
                             autodiff = :finite)
        θ̂ = collect(Float64, Optim.minimizer(res))
        β̂ = θ̂[1:p]
        Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
        r̂ = exp(θ̂[p + rr + 1])
        loglik, conv, iters = _fit_verdict(res)
        mineig = _truncnb2_min_site_eigen(Yc, Λ̂, β̂, fill(r̂, p); link = link,
                     mask = msk, offset = offset, hessian = hessian,
                     maxiter = newton_maxiter, tol = newton_tol)
        return (β = β̂, Λ = Λ̂, r = r̂, θ = θ̂, loglik = loglik, converged = conv,
                iters = iters, mineig = mineig)
    end
    # An optimum within 10% of the floor sits at the guard. This rule is load-bearing,
    # not belt-and-braces: when a line search runs into the 1e12 sentinel, L-BFGS with
    # BackTracking takes a zero step and Optim reports converged = true AT the wall.
    # Without this rule those fits would be reported converged at a Laplace value that
    # is not a usable log-likelihood.
    at_guard(f) = isfinite(eigmin_floor) && f.mineig < 1.1 * eigmin_floor
    f = run_from(Λ0, logr0)
    if at_guard(f)
        # One retry, from the same loadings with r from a moment estimate of the counts
        # (below) instead of the default 10; kept if it ends off the guard. If both end
        # at the guard, the one with the higher loglik is reported (flagged below): the
        # first fit can end at the wall far below the retry (-3659 vs -2008 on a r = 0.05
        # draw with r̂ -> 6e-42 on the first). The r reset is what matters: from
        # r = 10 the default start runs to the breakdown region, and a loadings-x-0.1
        # start with r = 10 fell into poor basins on all 3 audit draws. #557's
        # loadings-x-0.1 start with the moment r reached the healthy optimum on Julia
        # 1.10 but stopped at a genuine local maximum 19 units lower on one audit draw on
        # Julia 1.13; the unshrunk loadings reached it on both. Rates: the decisions note
        # 2026-09-27-truncnb2-laplace-breakdown-guard.md.
        f2 = run_from(Λ0, _truncnb2_moment_logr(Yc, msk))
        (!at_guard(f2) || f2.loglik > f.loglik) && (f = f2)
    end
    converged = f.converged
    if converged && at_guard(f)
        converged = false
        @warn "fit_truncated_nbinom2_gllvm: the optimum sits at the Laplace breakdown " *
              "guard (smallest site precision eigenvalue $(round(f.mineig; sigdigits = 3)), " *
              "floor $(eigmin_floor)). The Laplace approximation is not reliable here and " *
              "the fit is reported as not converged."
    end
    # Dispersion boundary (2026-09-29). r < 1e-6 is a degenerate fit (extreme
    # overdispersion; r = 3.1e-46 with one count of 10^13), yet Optim reports
    # converged = true, so it is reported as not converged. r > 1e6 is the Poisson
    # limit: r is not identified, but the fit itself is usually sound (a trait whose
    # extra variance the latent variable absorbs), so it only warns. Unlike the NB2
    # grouped fitters' `_dispersion_group_boundary`, which flags both ends.
    converged = _truncnb2_dispersion_verdict(converged, [f.r], "fit_truncated_nbinom2_gllvm")
    return TruncatedNegBin2Fit(f.β, f.Λ, f.r, link, f.loglik, converged, f.iters,
                               f.θ, f.mineig)
end

# log of a shared-r start for the breakdown retry: the geometric mean over traits of a
# per-trait NB2 moment estimate m² / (v − m) on the observed counts, each clamped to
# [0.2, 20] (the clamp of the zi_* route's phi start). Ignores the truncation: it only
# needs to put r on the right side of 1, and small-r data are strongly overdispersed.
function _truncnb2_moment_logr(Y::AbstractMatrix, mask)
    p, n = size(Y)
    acc = 0.0; cnt = 0
    for t in 1:p
        ys = [Float64(Y[t, i]) for i in 1:n if mask === nothing || mask[t, i]]
        length(ys) >= 2 || continue
        m = sum(ys) / length(ys)
        v = sum(abs2, ys .- m) / (length(ys) - 1)
        acc += log(clamp(m^2 / max(v - m, 1e-3 * m), 0.2, 20.0)); cnt += 1
    end
    return cnt == 0 ? log(10.0) : acc / cnt
end

# ---------------------------------------------------------------------------
# Arc1b — per-trait r_t ≡ twin log_phi_truncnb2
# pack [β; pack(Λ); log r_1 … log r_p], length p+rr+p
# ---------------------------------------------------------------------------

function _truncnb2_pertrait_loglik_site(fams::AbstractVector, y::AbstractVector,
        n::AbstractVector, Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, hessian::Symbol = :observed,
        eigmin_floor::Real = -Inf, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    # NOTE: the mode solve stays on the Fisher weight (`_grouped_laplace_mode`), which
    # is a Fisher-scoring iteration. That affects only HOW the mode is found, not the
    # objective — the log-det below is what defines the Laplace approximation, and it
    # is the term that must carry TMB's observed curvature.
    z = _grouped_laplace_mode(fams, y, n, Λ, β, link;
                              mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    η  = _clamp_eta.(β .+ off .+ Λ * z)
    μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
    me = mu_eta.(Ref(link), η)
    W  = _truncnb2_laplace_weight.(Ref(hessian), fams, μ, me, y, Ref(link))
    if mask !== nothing
        W = ifelse.(mask, W, 0.0)
    end
    A = Symmetric(Λ' * (W .* Λ) + I)
    # Laplace breakdown guard (see TRUNCNB2_LAPLACE_EIGMIN_FLOOR). Skipped entirely at
    # the default -Inf, so the unguarded value is unchanged bit for bit.
    (isfinite(eigmin_floor) && eigmin(A) < eigmin_floor) && return -Inf
    ℓ = 0.0
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _glm_logpdf(fams[t], μ[t], n[t], y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

# Smallest per-site Laplace precision eigenvalue min_s eigmin(I + Λ' diag(W_s) Λ) at the
# given parameters, with the same mode, weights and mask as the site kernel above.
function _truncnb2_min_site_eigen(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, rvec::AbstractVector; link::Link = LogLink(),
        mask = nothing, offset = nothing, hessian::Symbol = :observed,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p = size(Λ, 1)
    fams = TruncatedNegBin2.(float.(rvec))
    n1 = ones(Int, p)
    m = Inf
    @inbounds for i in axes(Y, 2)
        mi = mask   === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        y = view(Y, :, i)
        z = _grouped_laplace_mode(fams, y, n1, Λ, β, link;
                                  mask = mi, offset = oi, maxiter = maxiter, tol = tol)
        η  = _clamp_eta.(β .+ (oi === nothing ? false : oi) .+ Λ * z)
        μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
        me = mu_eta.(Ref(link), η)
        W  = _truncnb2_laplace_weight.(Ref(hessian), fams, μ, me, y, Ref(link))
        mi === nothing || (W = ifelse.(mi, W, 0.0))
        m = min(m, eigmin(Symmetric(Λ' * (W .* Λ) + I)))
    end
    return m
end

"""
    truncated_nbinom2_pertrait_marginal_loglik_laplace(Y, Λ, β, rvec; link=LogLink(), kwargs...) -> Float64

Laplace log-marginal for a zero-truncated NB2 GLLVM with **per-trait**
dispersion `rvec` (length p; `r_t` ≡ twin `φ_t = exp(log_phi_truncnb2[t])`).
Equal `r_t` reduces to the shared-`r` [`truncated_nbinom2_marginal_loglik_laplace`](@ref).
`Y` must be integer counts with every observed cell `≥ 1`. Mode-finding reuses
`_grouped_laplace_mode` with `fams = TruncatedNegBin2.(rvec)`.
"""
function truncated_nbinom2_pertrait_marginal_loglik_laplace(Y::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, rvec::AbstractVector;
        link::Link = LogLink(), mask = nothing, offset = nothing, kwargs...)
    p = size(Λ, 1)
    length(rvec) == p || throw(ArgumentError(
        "length(rvec)=$(length(rvec)) must equal p=$p"))
    N1 = ones(Int, size(Y))
    fams = TruncatedNegBin2.(float.(rvec))
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask   === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        acc += _truncnb2_pertrait_loglik_site(fams, view(Y, :, i), view(N1, :, i),
                                              Λ, β, link; mask = mi, offset = oi, kwargs...)
    end
    return acc
end

"""
    TruncatedNegBin2PerTraitFit

Result of [`fit_truncated_nbinom2_gllvm_pertrait`](@ref): intercepts `β`,
loadings `Λ`, per-trait dispersion `r` (length p; `Var_t = μ_t + μ_t²/r_t`
≡ twin `φ_t`), link, loglik, convergence.
"""
struct TruncatedNegBin2PerTraitFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    r::Vector{Float64}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    theta_packed::Vector{Float64}
end

function Base.show(io::IO, f::TruncatedNegBin2PerTraitFit)
    p, K = size(f.Λ)
    print(io, "TruncatedNegBin2PerTraitFit(p=", p, ", K=", K,
          ", r∈[", round(minimum(f.r); sigdigits = 4), ", ",
          round(maximum(f.r); sigdigits = 4), "]",
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_truncated_nbinom2_gllvm_pertrait(Y; K, link=LogLink(), hessian=:observed, …)
        -> TruncatedNegBin2PerTraitFit

Fit a zero-truncated NB2 GLLVM with **per-trait** dispersion by Laplace + LBFGS
over `[β; pack(Λ); log r_1 … log r_p]` (length `p+rr+p`). Twin-aligned:
`r_t` ≡ `φ_t = exp(log_phi_truncnb2[t])`; log link on untruncated `μ`;
support `y ≥ 1`. Score keeps `a = r_t/(r_t+μ)` (Sol 2026-08-15).
Throws if any observed cell is `< 1`. If any `r_t` ends below 1e-6 (the dispersion
boundary: extreme overdispersion), the fit is reported with `converged = false` and a
warning; an `r_t` above 1e6 (the Poisson limit, not identified) only warns.

`hessian=:observed` (the default) uses the exact conditional truncated-NB2/log
curvature that TMB's Laplace objective uses; `hessian=:fisher` retains the
expected-information approximation.

!!! note "Why the default is `:observed` (2026-08-24)"
    Both truncated-NB2 routes previously used the Fisher weight with no way to
    select otherwise, so the Laplace log-det term was built from expected rather
    than observed information — **a different objective from TMB's**, which made a
    twin parity cell for fid 11 meaningless. Unlike truncated Poisson (fid 10),
    where `y` enters `η` linearly so the two curvatures coincide pointwise, the
    NB2 curvature is y-dependent through `−(y+r)·log(μ+r)`; the difference is real
    and is the same class of fault fixed for NB1 the same day. The observed weight
    is `_truncnb2_observed_weight`, verified against ForwardDiff to 1.8e-13.
"""
function fit_truncated_nbinom2_gllvm_pertrait(Y::AbstractMatrix; K::Integer,
        link::Link = LogLink(), mask = nothing, offset = nothing,
        hessian::Symbol = :observed,
        β_init = nothing, Λ_init = nothing, r_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    link isa LogLink || throw(ArgumentError(
        "fit_truncated_nbinom2_gllvm_pertrait: only LogLink is supported (twin truncated_nbinom2)"))
    # Validate here, NOT inside negll: the objective wraps its body in a try/catch that
    # converts any throw into 1e12, so a typo'd symbol would otherwise be swallowed and
    # silently return a garbage fit instead of failing loudly.
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_truncated_nbinom2_gllvm_pertrait: hessian must be :observed or :fisher; got :$hessian"))
    p, n = size(Y)
    rr = rr_theta_len(p, K)
    msk = mask === nothing ? (any(ismissing, Y) ? observed_mask(Y) : nothing) : mask
    Yc = Integer.(_sanitize_missing(Y, 1))
    @inbounds for t in 1:p, s in 1:n
        (msk !== nothing && !msk[t, s]) && continue
        Yc[t, s] < 1 && throw(ArgumentError(
            "truncated_nbinom2 requires y ≥ 1; found y=$(Yc[t, s]) at ($t,$s)"))
    end

    Zemp = [linkfun(link, max(Float64(Yc[t, i]), 1.0)) for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)
    if msk !== nothing
        @inbounds for t in 1:p
            obs = view(msk, t, :)
            cnt = count(obs)
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
    logr0 = if r_init === nothing
        fill(log(10.0), p)
    elseif r_init isa AbstractVector
        length(r_init) == p || throw(ArgumentError(
            "length(r_init)=$(length(r_init)) must equal p=$p"))
        log.(float.(r_init))
    else
        fill(log(float(r_init)), p)
    end

    θ0 = vcat(β0, pack_lambda(Λ0), logr0)
    length(θ0) == p + rr + p || throw(ArgumentError(
        "per-trait pack length $(length(θ0)) ≠ p+rr+p=$(p + rr + p)"))
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        rvec = exp.(θ[(p + rr + 1):(p + rr + p)])
        v = try
            -truncated_nbinom2_pertrait_marginal_loglik_laplace(Yc, Λ, β, rvec;
                    link = link, mask = msk, offset = offset, hessian = hessian,
                    maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    r̂ = exp.(θ̂[(p + rr + 1):(p + rr + p)])
    loglik, converged, iters = _fit_verdict(res)
    # Dispersion boundary (2026-09-29): see `fit_truncated_nbinom2_gllvm`.
    converged = _truncnb2_dispersion_verdict(converged, r̂,
                                             "fit_truncated_nbinom2_gllvm_pertrait")
    return TruncatedNegBin2PerTraitFit(β̂, Λ̂, r̂, link, loglik, converged, iters,
                                       collect(Float64, θ̂))
end
