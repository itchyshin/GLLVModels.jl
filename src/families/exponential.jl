# Exponential (positive continuous, no dispersion) family. y_t > 0; mean μ = exp(η)
# (log link), so the per-observation law is Exponential(μ) — i.e. Gamma(shape 1,
# scale μ): E[y]=μ, Var = μ². It is the dispersion-free special case of the Gamma
# family (α ≡ 1), so its score/weight are the Gamma GLM pieces at α = 1:
#   s = (y − μ)/μ² · dμ/dη,   W_fisher = (dμ/dη)²/μ².
#
# CURVATURE — corrected 2026-08-24. `_glm_weight` below is the **expected** (Fisher)
# information; at the log link it is the CONSTANT 1, independent of y. TMB's Laplace
# uses the **observed** joint Hessian, which for Exponential/log is
#
#     −∂²ℓ/∂η² = y / μ          (verified against ForwardDiff to 1.9e-16)
#
# E[y] = μ recovers 1, so the shipped Fisher value was exactly the *expectation* of
# the correct one — the signature of this fault class (six instances repo-wide; see
# docs/dev-log/check-log.md 2026-08-24). The error is not marginal: at μ = 0.5, y = 9
# the correct weight is 18, not 1.
#
# The generic core (`src/families/laplace.jl`) hard-codes the Fisher weight, carries
# no `hessian` keyword, and is fenced by the Arc1b amendment. So the marginal below
# routes through the **Gamma grouped** kernel at α ≡ 1, which already implements the
# observed weight (`_gamma_grouped_laplace_weight` = `α·y/μ`) and accepts `hessian`.
# Verified equivalent at α = 1: `_glm_logpdf` agrees to 4.4e-16, `_glm_score` exactly.
# `_glm_weight` is retained for `hessian = :fisher` and for the generic-core path.
_clamp_mu(::Exponential, μ) = max(μ, 1e-12)
_glm_score(::Exponential, μ, n, me, y) = (y - μ) / μ^2 * me
_glm_weight(::Exponential, μ, n, me)   = me^2 / μ^2
_glm_logpdf(::Exponential, μ, n, y)    = logpdf(Exponential(μ), y)

# Observed conditional curvature for Exponential/log: ℓ = −η − y·e^{−η}, so
# −∂²ℓ/∂η² = y·e^{−η} = y/μ (the Gamma α=1 special case of α·y/μ).
_glm_obs_weight(::Exponential, μ, n, me, y, link::LogLink, η) = y / μ

# The family-level default, DECLARED (2026-08-27, adversarial-audit fix): the
# user-facing fitter has defaulted `:observed` since 2026-08-24, but only via
# its own signature — the `_default_hessian` registry still said `:fisher`,
# so the covariates/quadratic/mixed/SPDE/phylo-GLM/coevolution kernels ran
# Exponential on the OTHER curvature. Declaring it here makes every route
# agree with the shipped default (the same cross-route coherence the NB2 flip
# enforced) and satisfies the census's has-observed ⇒ declared-default rule.
_default_hessian(::Exponential, ::LogLink) = :observed

"""
    exponential_marginal_loglik_laplace(Y, Λ, β; link=LogLink(), hessian=:observed,
                                        kwargs...) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of an Exponential GLLVM —
responses `Y > 0`, mean `μ = exp(η)` (log link), per-observation `Exponential(μ)`
(`Var = μ²`).

`hessian=:observed` (the default) uses TMB's observed Laplace curvature `y/μ`;
`hessian=:fisher` retains the expected-information approximation (constant `1` at the
log link), which is what this path used unconditionally before 2026-08-24.

Evaluated through the **Gamma grouped** kernel at `α ≡ 1` rather than the generic
Laplace core, because only the former carries an observed-curvature implementation
and a `hessian` keyword. Exponential is exactly Gamma(shape 1), verified: `_glm_logpdf`
agrees to 4.4e-16 and `_glm_score` agrees exactly at `α = 1`.
"""
function exponential_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector; link::Link = LogLink(), hessian::Symbol = :observed, kwargs...)
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "exponential_marginal_loglik_laplace: hessian must be :observed or :fisher; got :$hessian"))
    if hessian === :fisher
        # Route the Fisher branch through the ORIGINAL generic core, not through the
        # Gamma grouped kernel. Both give the same value at a fixed (Λ, β) — verified
        # bit-for-bit — but they use DIFFERENT mode solvers: the generic core carries
        # restart/backtracking safety (`_laplace_mode_should_backtrack`) that
        # `_grouped_laplace_mode` does not. Under optimisation that difference is not
        # cosmetic: routing Fisher through the grouped kernel let ‖Λ‖ run away to ~960
        # against a true 0.38 on a p=5, K=1, n=80 fixture. Preserving the original path
        # here keeps `:fisher` EXACTLY the pre-2026-08-24 behaviour, so this change is a
        # corrected default rather than an altered capability.
        # `hessian = :fisher` is passed EXPLICITLY, never left to the core's
        # default. If that default is ever flipped to `:observed`, this branch
        # would otherwise start computing observed curvature while still calling
        # itself `:fisher` — silently destroying the bit-for-bit guarantee the
        # comment above promises, with no test able to see it.
        return marginal_loglik_laplace(Exponential(1.0), Y, ones(Int, size(Y)), Λ, β,
                                       link; hessian = :fisher, kwargs...)
    end
    # RE-ROUTED 2026-08-27: through the generic core, retiring the Gamma
    # grouped-kernel detour. The detour existed (2026-08-24) because only that
    # kernel carried an observed implementation; the generic core has taken
    # `hessian = :observed` since the kwarg work (PR #268). The detour's own
    # per-site Newton loop is UNDAMPED and diverges at moderate parameters on
    # exponential-scale data — measured: at a healthy fisher-fit optimum
    # (p=10, n=150 fixture) it returned −5.0e23 against an exact marginal of
    # −1717.6, while the generic core returned −1716.1. That garbage surface
    # is what drove `fit_exponential_gllvm(hessian = :observed)` fits to
    # ‖Λ̂‖ ~ 10³ with `converged = true` in 87/150 curvature-adjudication
    # campaign cells. The undamped per-site loops remaining in
    # `grouped_dispersion.jl` are recorded engine debt (Arc 2).
    return marginal_loglik_laplace(Exponential(1.0), Y, ones(Int, size(Y)), Λ, β,
                                   link; hessian = :observed, kwargs...)
end

# ---------------------------------------------------------------------------
# Fit driver.
# ---------------------------------------------------------------------------

"""
    ExponentialFit

Result of [`fit_exponential_gllvm`](@ref): intercepts `β` (length p), loadings `Λ`
(p×K), the `link`, the maximised Laplace `loglik`, the optimiser `converged` flag,
and `iterations`. (No dispersion — the Exponential has `Var = μ²` fixed.)
`offset` is the p×n training offset the fit was made with (`nothing` when it had none);
[`predict`](@ref), [`getLV`](@ref) and `residuals` use it by default.
"""
struct ExponentialFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    offset::Union{Nothing,Matrix{Float64}}   # training offset (p×n); `nothing` = none
end

# Pre-offset compat tier (7 positional args): no stored training offset.
ExponentialFit(β, Λ, link, loglik, converged, iterations, hessian) =
    ExponentialFit(β, Λ, link, loglik, converged, iterations, hessian, nothing)

# Positional compatibility constructor (2026-08-28): every pre-existing
# construction site builds a default-curvature fit; the `hessian` field
# records the objective identity so `confint`/bootstrap can rebuild THE
# SAME objective instead of guessing (the audit's confint-consistency class).
ExponentialFit(β, Λ, link, loglik, converged, iterations) =
    ExponentialFit(β, Λ, link, loglik, converged, iterations, _default_hessian(Exponential(1.0), link))

function Base.show(io::IO, f::ExponentialFit)
    p, K = size(f.Λ)
    print(io, "ExponentialFit(p=", p, ", K=", K, ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_exponential_gllvm(Y; K, link=LogLink(), mask=nothing, offset=nothing, …) -> ExponentialFit

Fit an Exponential GLLVM by L-BFGS over `[β; vec(Λ)]` on the Laplace marginal
(`exponential_marginal_loglik_laplace`). `Y` is a p×n matrix of positive reals;
`K` the latent dimension. Finite-difference gradient; warm start = log row-means
as intercepts + SVD of row-centred log-Y as loadings.

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or `missing` entries
in `Y`; masked cells are dropped from the marginal and the warm start, so the fit
depends only on the observed cells (gllvm-style NA handling).
"""
function fit_exponential_gllvm(Y::AbstractMatrix; K::Integer,
        link::Link = LogLink(), mask = nothing, offset = nothing,
        hessian::Symbol = :observed,
        β_init = nothing, Λ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    # Validated up front: the objective below wraps its body in a try/catch that turns
    # any throw into a large penalty, which would launder a typo'd symbol into a
    # converged-looking garbage fit.
    hessian in (:observed, :fisher) || throw(ArgumentError(
        "fit_exponential_gllvm: hessian must be :observed or :fisher; got :$hessian"))
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, mask = mask,
                               caller = "fit_exponential_gllvm")
    rr = rr_theta_len(p, K)

    msk = _resolve_obs_mask(mask, Y)                  # NA handling
    Yc  = _sanitize_missing(Y, 1.0)                   # positive placeholder

    Zemp = log.(max.(Yc, 1e-6))
    offset === nothing || (Zemp .-= offset)           # offset (η = β + offset + Λz)
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

    θ0 = vcat(β0, pack_lambda(Λ0))
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        v = try
            -exponential_marginal_loglik_laplace(Yc, Λ, β; link = link, mask = msk,
                                                 offset = offset, hessian = hessian,
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
    return ExponentialFit(β̂, Λ̂, link, _fit_verdict(res)..., hessian, _stored_offset(offset))
end
