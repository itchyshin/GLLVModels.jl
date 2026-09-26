# Fixed-effect covariates (Xβ) for the non-Gaussian Laplace families.
#
# The non-Gaussian Laplace path (src/families/laplace.jl) has no covariate term:
# its linear predictor is η_{ts} = β_t + (Λ z_s)_t. This file adds the
# environmental fixed-effect surface that gllvmTMB centres on — an additive
# offset o_{ts} = Σ_k X[t,s,k]·γ_k on the linear predictor:
#
#     η_{ts} = β_t + o_{ts} + (Λ z_s)_t
#
# mirroring the Gaussian engine's X::(p,n,q) / coefficient contract (src/likelihood.jl,
# src/fit.jl). Because the offset is constant in z, the per-observation score and
# Fisher weight wrt η are unchanged, so the existing family pieces
# (`_glm_score` / `_glm_weight` / `_glm_logpdf` / `_clamp_mu`) are reused verbatim.
#
# Design choice: a STANDALONE offset-aware per-site Laplace (`_laplace_site_off`)
# rather than editing the shared core, so families without covariates are byte-for-
# byte unaffected. The shared `_default_link`/links/packing helpers are reused.
#
# Coefficient convention: `γ` (length q) is SHARED across species; the 3-D `X`
# encodes species- and site-varying covariates exactly as the Gaussian path does
# (a per-species response is a block-expanded column of `X`). The first slice
# fits with covariates and recovers `γ`; post-fit predict/CI integration for the
# `GllvmCovFit` type is a documented follow-up.

# --- offset-aware per-site Laplace (mirrors families/laplace.jl, with η0 = β + offset) ---

# One pass of the per-site mode search. Returns `(z, converged)`. `step_weight` picks
# the curvature that sets the step, `:fisher` (expected, never negative) or `:observed`.
# It never changes the mode, which is the fixed point of `Λ's − z = 0` whatever W is.
#
# Before 2026-09-25 this loop took full Fisher-scoring steps with no damping and
# returned whatever `z` it held when it stopped, converged or not. Gamma/log's Fisher
# weight is the constant α, blind to y/μ, so the step overshoots when y/μ is far from 1:
# on one site of fit_gllvm_cov's own warm start (the #479 data) it ran out to z ≈ 1e11
# and the site returned a FINITE −7.2e22, which slipped past the fitter's 1e12 sentinel
# and sent the line search to a log α so low that exp(log α) == 0.0.
#
# Now a step that lowers the per-site log-posterior is halved (the generic core's
# `_laplace_mode` rule, so small steps and accepted full steps are bit-identical to the
# old loop), and `converged` is true only when the full proposed step is below `tol`,
# the old loop's own stopping test. A heavily halved step does NOT count.
function _laplace_mode_off_pass(family, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, η0::AbstractVector, link::Link, step_weight::Symbol;
        mask = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    z = zeros(K)
    for _ in 1:maxiter
        η  = _clamp_eta.(η0 .+ Λ * z)
        μ  = _clamp_mu.(Ref(family), linkinv.(Ref(link), η))
        me = mu_eta.(Ref(link), η)
        s  = _glm_score.(Ref(family), μ, n, me, y)
        W  = if step_weight === :fisher
            _glm_weight.(Ref(family), μ, n, me)
        else
            # Masked cells hold a placeholder `y`, so their observed weight is not evaluated.
            [(mask === nothing || mask[t]) ?
                _glm_obs_weight(family, μ[t], n[t], me[t], y[t], link, η[t]) : 0.0 for t in 1:p]
        end
        if mask !== nothing
            s = ifelse.(mask, s, 0.0)            # masked ⇒ no contribution
            W = ifelse.(mask, W, 0.0)
        end
        # A negative (or NaN) observed weight can make Λ'WΛ + I indefinite, and then the
        # step need not point uphill. Give up rather than take it.
        step_weight === :fisher || all(w -> w >= 0, W) || return z, false
        A  = Symmetric(Λ' * (W .* Λ) + I)
        Δ  = _safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && return z, false
        maximum(abs, Δ) < tol && return z .+ Δ, true
        if norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        else
            q0 = _laplace_mode_logpost(family, y, n, Λ, η0, link, z; mask = mask)
            if isfinite(q0)
                accepted = false
                step = 1.0
                for _half in 1:30
                    ztrial = z .+ step .* Δ
                    q1 = _laplace_mode_logpost(family, y, n, Λ, η0, link, ztrial; mask = mask)
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

# Per-site mode search with a convergence report. Returns `(z, converged)`.
# Fisher scoring runs first, so every site that converged before keeps its old path.
# Fallback: Fisher scoring converges only linearly when it is not Newton (Gamma/log,
# Exponential/log, NB2/log), and even with halving it left sites unconverged after 100
# iterations on healthy Gamma datasets at the fitter's warm start (measured: 6 of 4000
# site evaluations on the #479 sibling screen, 2 of them on healthy datasets 4 and 5).
# A damped Newton search on the observed curvature converged at all 4000, in at most 6
# iterations. It runs only where Fisher failed and the two curvatures differ; if it also
# fails, the Fisher pass's last iterate is returned, flagged unconverged.
function _laplace_mode_off_conv(family, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, η0::AbstractVector, link::Link;
        mask = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    z, ok = _laplace_mode_off_pass(family, y, n, Λ, η0, link, :fisher;
                                   mask = mask, maxiter = maxiter, tol = tol)
    (ok || _glm_weight_matches_observed(family, link)) && return z, ok
    z2, ok2 = _laplace_mode_off_pass(family, y, n, Λ, η0, link, :observed;
                                     mask = mask, maxiter = maxiter, tol = tol)
    return ok2 ? (z2, true) : (z, false)
end

# The mode alone, for the getLV-style callers (covariate, species-covariate,
# fourth-corner, row-effect and constrained getLV). Unchanged contract: a plain score,
# with no convergence flag, even at a site whose search did not converge.
function _laplace_mode_off(family, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, η0::AbstractVector, link::Link;
        mask = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    return first(_laplace_mode_off_conv(family, y, n, Λ, η0, link;
                                        mask = mask, maxiter = maxiter, tol = tol))
end

function _laplace_site_off(family, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, η0::AbstractVector, link::Link;
        mask = nothing, hessian::Symbol = _default_hessian(family, link),
        maxiter::Integer = 100, tol::Real = 1e-9)
    (hessian === :fisher || hessian === :observed) || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    p = size(Λ, 1)
    # The MODE SEARCH is Fisher first, with an observed-curvature fallback
    # (`_laplace_mode_off_conv`); only the log-det takes the selector. Same role
    # separation as the generic core. A site whose search does not converge returns
    # -Inf, never a value computed at an unconverged mode, so the fitters' objective
    # returns its 1e12 sentinel instead of a garbage surface.
    z, ok = _laplace_mode_off_conv(family, y, n, Λ, η0, link; mask = mask, maxiter = maxiter, tol = tol)
    ok || return -Inf
    η  = _clamp_eta.(η0 .+ Λ * z)
    μ  = _clamp_mu.(Ref(family), linkinv.(Ref(link), η))
    me = mu_eta.(Ref(link), η)
    # This kernel builds its OWN Λ'WΛ + I and its own logdet, so the generic
    # core's selector never reached it. Without this, a family whose default is
    # `:observed` (Gamma/log since 2026-08-25) disagrees with every flipped path
    # — measured at 0.238 on `test_gamma_x_identity.jl:35` before this change.
    # Masked cells: the observed weight reads `y`, which is a PLACEHOLDER there,
    # so it must not be evaluated — see the same note in `laplace.jl`.
    W  = if hessian === :fisher || _glm_weight_matches_observed(family, link)
        _glm_weight.(Ref(family), μ, n, me)
    else
        [(mask === nothing || mask[t]) ?
            _glm_obs_weight(family, μ[t], n[t], me[t], y[t], link, η[t]) : 0.0
         for t in 1:p]
    end
    if mask !== nothing
        W = ifelse.(mask, W, 0.0)
    end
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = 0.0
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _glm_logpdf(family, μ[t], n[t], y[t])
    end
    # PD guard, keyed on the weight's sign — see `laplace.jl` for the reasoning.
    if any(w -> w < zero(w), W)
        F = cholesky(A; check = false)
        issuccess(F) || return oftype(ℓ, -Inf)
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

# Total Laplace log-marginal with an additive per-(t,s) offset matrix `O` (p×n):
# η0_s = β + O[:, s]. `mask` (p×n Bool, or `nothing`) drops missing responses per
# site (gllvm-style NA handling), mirroring `marginal_loglik_laplace`.
function _marginal_loglik_offset(family, Y::AbstractMatrix, N::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, O::AbstractMatrix, link::Link;
        mask = nothing, kwargs...)
    acc = 0.0
    @inbounds for s in axes(Y, 2)
        η0 = β .+ view(O, :, s)
        mi = mask === nothing ? nothing : view(mask, :, s)
        acc += _laplace_site_off(family, view(Y, :, s), view(N, :, s), Λ, η0, link;
                                 mask = mi, kwargs...)
    end
    return acc
end

# Offset matrix O[t,s] = Σ_k X[t,s,k]·γ_k from X::(p,n,q) and γ::length-q.
function _build_offset(X::AbstractArray{<:Real, 3}, γ::AbstractVector)
    p, n, q = size(X)
    length(γ) == q || throw(DimensionMismatch("γ has length $(length(γ)) but X has q = $q covariates"))
    return reshape(reshape(X, p * n, q) * γ, p, n)
end

# --- family-specific bits (isolated so the fitter stays generic) ---
_cov_default_link(::Poisson)          = LogLink()
_cov_default_link(::NegativeBinomial) = LogLink()
_cov_default_link(::NB1)              = LogLink()
_cov_default_link(::Binomial)         = LogitLink()
_cov_default_link(::Beta)             = LogitLink()
_cov_default_link(::Gamma)            = LogLink()
_cov_default_link(::Exponential)      = LogLink()

_cov_has_disp(::Poisson)          = false
_cov_has_disp(::Binomial)         = false
_cov_has_disp(::Exponential)      = false
_cov_has_disp(::NegativeBinomial) = true
_cov_has_disp(::NB1)              = true
_cov_has_disp(::Beta)             = true
_cov_has_disp(::Gamma)            = true

_cov_disp_init(::NegativeBinomial) = 10.0
_cov_disp_init(::NB1)              = 1.0
_cov_disp_init(::Beta)             = 10.0
_cov_disp_init(::Gamma)            = 2.0
_cov_disp_init(f)                  = 1.0

# Rebuild the family marker carrying the current dispersion (only the relevant
# field is read by the marginal pieces).
_cov_family(::Poisson, d)          = Poisson()
_cov_family(::Binomial, d)         = Binomial()
_cov_family(::NegativeBinomial, d) = NegativeBinomial(d, 0.5)
_cov_family(::NB1, d)              = NB1(float(d))
_cov_family(::Beta, d)             = Beta(d, 1.0)
_cov_family(::Gamma, d)            = Gamma(d, 1.0)
_cov_family(::Exponential, d)      = Exponential(1.0)

# CI term name for the dispersion parameter (if any).
_cov_dispname(::NegativeBinomial) = "r"
_cov_dispname(::NB1)              = "phi"
_cov_dispname(::Beta)             = "phi"
_cov_dispname(::Gamma)            = "alpha"
_cov_dispname(f)                  = "disp"
# Draw one response from `family` (carrying its dispersion) at mean `μ`; `nt` is
# the Binomial trial count (ignored otherwise). Used by predict-side simulation
# and the bootstrap CI.
_cov_sample(::Poisson, μ, nt, rng)          = rand(rng, Poisson(max(μ, 1e-12)))
_cov_sample(f::NegativeBinomial, μ, nt, rng) = (m = max(μ, 1e-12); rand(rng, NegativeBinomial(f.r, f.r / (f.r + m))))
_cov_sample(f::NB1, μ, nt, rng) = begin
    m = max(μ, 1e-12)
    rand(rng, NegativeBinomial(m / f.φ, 1 / (1 + f.φ)))
end
_cov_sample(::Binomial, μ, nt, rng)         = rand(rng, Binomial(nt, clamp(μ, 1e-12, 1 - 1e-12)))
function _cov_sample(f::Beta, μ, nt, rng)
    m = clamp(μ, 1e-6, 1 - 1e-6)
    return clamp(rand(rng, Beta(m * f.α, (1 - m) * f.α)), 1e-6, 1 - 1e-6)
end
_cov_sample(f::Gamma, μ, nt, rng)           = rand(rng, Gamma(f.α, max(μ, 1e-12) / f.α))
_cov_sample(::Exponential, μ, nt, rng)      = rand(rng, Exponential(max(μ, 1e-12)))
# Domain-safe placeholder for masked (missing) Y cells, per family. The masked
# cells are dropped from every likelihood contribution and overwritten in the warm
# start (`_mask_warmstart!`), so the placeholder only has to keep the score/weight
# broadcast in-domain (the values there are computed then zeroed).
_cov_placeholder(::Beta) = 0.5                       # in (0,1)
_cov_placeholder(f)      = 0.0                        # counts / Gamma (Zemp uses max(·,ε))

# Link-scale latent proxy for the warm start (per family).
function _cov_zemp(family, Y::AbstractMatrix, N::AbstractMatrix, link::Link)
    p, n = size(Y)
    if family isa Poisson || family isa NegativeBinomial || family isa NB1
        return [linkfun(link, max(Y[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    elseif family isa Binomial
        return [linkfun(link, clamp((Y[t, i] + 0.5) / (N[t, i] + 1), 1e-4, 1 - 1e-4)) for t in 1:p, i in 1:n]
    elseif family isa Beta
        return [linkfun(link, clamp(float(Y[t, i]), 1e-6, 1 - 1e-6)) for t in 1:p, i in 1:n]
    else  # Gamma
        return [linkfun(link, max(float(Y[t, i]), 1e-6)) for t in 1:p, i in 1:n]
    end
end

"""
    GllvmCovFit

Result of [`fit_gllvm_cov`](@ref): a GLLVM fit with fixed-effect covariates. Fields:
`family` (the Distributions marker), per-species intercepts `β` (length p), shared
covariate coefficients `γ` (length q, including fixed-zero entries), `γ_fixed`
(Bool vector marking fixed-zero entries), loadings `Λ` (p×K), `dispersion`
(`r`/`φ`/`α`, or `NaN` when the family has none), `link`, the maximised Laplace
`loglik`, `converged`, and `iterations`.
"""
struct GllvmCovFit
    # Distributions markers (Poisson, NB2, …) plus custom NB1(φ) for the
    # shared-φ + X opt-in; public twin default under X is NB1GroupedCovFit.
    family::Any
    β::Vector{Float64}
    γ::Vector{Float64}
    γ_fixed::Vector{Bool}
    Λ::Matrix{Float64}
    dispersion::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::GllvmCovFit)
    p, K = size(f.Λ); q = length(f.γ)
    print(io, "GllvmCovFit(", nameof(typeof(f.family)), ", p=", p, ", q=", q, ", K=", K)
    any(f.γ_fixed) && print(io, ", fixed γ=", count(f.γ_fixed))
    isnan(f.dispersion) || print(io, ", disp=", round(f.dispersion; sigdigits = 4))
    print(io, ", loglik=", round(f.loglik; sigdigits = 7), f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_gllvm_cov(Y; family, X, K, link=nothing, N=nothing, γ_fixed=nothing, …) -> GllvmCovFit

Fit a non-Gaussian GLLVM **with fixed-effect covariates** by L-BFGS over
`[β; γ; vec(Λ); (log-dispersion)]` on the offset-augmented Laplace marginal, where
the linear predictor is `η_{ts} = β_t + Σ_k X[t,s,k]·γ_k + (Λ z_s)_t`.

`family` is a `Distributions` marker — `Poisson()`, `NegativeBinomial()`,
`Binomial()`, `Beta()`, or `Gamma()` — or the custom `NB1` marker —
and dispatches the marginal (the dispersion, where present, is a **shared
scalar** jointly estimated). For NB2/Beta/Gamma/NB1 the public and bridge
default under X is per-trait dispersion via
[`fit_nb_gllvm_grouped_cov`](@ref) / [`fit_beta_gllvm_grouped_cov`](@ref) /
[`fit_gamma_gllvm_grouped_cov`](@ref) / [`fit_nb1_gllvm_grouped_cov`](@ref);
this shared-dispersion path remains the explicit opt-in. `X` is the `(p, n, q)`
covariate array
(same contract as the Gaussian engine); `γ` (length q) are coefficients shared
across species (encode species-specific responses by block-expanding `X`). `Y` is
`p × n`; `N` supplies Binomial trial counts (default all-ones). Finite-difference
gradient. `γ_fixed` optionally fixes selected covariate coefficients to zero;
pass a Bool vector of length `size(X, 3)`, an integer index vector, or a Dict
index=>0.

```julia
# Poisson abundance with one site covariate, shared coefficient:
X = reshape(repeat(temp', p), p, n, 1)            # X[t,s,1] = temp[s]
fit = fit_gllvm_cov(Y; family = Poisson(), X = X, K = 2)
fit.γ            # estimated environmental coefficient(s)
```

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or simply include
`missing` entries in `Y` — either way the masked cells are dropped from the
marginal *and* from the warm start, so the fit depends only on the observed cells.
"""
function fit_gllvm_cov(Y::AbstractMatrix; family, X::AbstractArray{<:Real, 3},
        K::Integer, link::Union{Nothing, Link} = nothing,
        N::Union{Nothing, AbstractMatrix} = nothing, mask = nothing,
        γ_fixed = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q) = ($p, $n, q); got $(size(X))"))
    q_full = size(X, 3)
    γ_fixed_mask = _fixed_zero_mask(γ_fixed, q_full, "γ_fixed")
    X_fit, _ = _slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    rr = rr_theta_len(p, K)
    lk = link === nothing ? _cov_default_link(family) : link
    Nm = N === nothing ? fill(1, p, n) : N
    has_disp = _cov_has_disp(family)

    # NA handling: derive the observation mask and a sanitized response matrix.
    msk = _resolve_obs_mask(mask, Y)
    Yc = _sanitize_missing(Y, _cov_placeholder(family))

    Zemp = _cov_zemp(family, Yc, Nm, lk)
    _mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end

    θ0 = has_disp ? vcat(β0, zeros(q), pack_lambda(Λ0), log(_cov_disp_init(family))) :
                    vcat(β0, zeros(q), pack_lambda(Λ0))
    function negll(θ)
        β = θ[1:p]
        γ = θ[(p + 1):(p + q)]
        Λ = unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        disp = has_disp ? exp(θ[p + q + rr + 1]) : NaN
        O = _build_offset(X_fit, γ)
        v = try
            # Built inside the `try`: a line-search step can push log-dispersion so low
            # that exp(...) == 0.0, and Distributions then throws DomainError (Gamma,
            # Beta and NegativeBinomial all check their parameter). That point must get
            # the sentinel, not end the fit.
            fam = _cov_family(family, disp)
            -_marginal_loglik_offset(fam, Yc, Nm, Λ, β, O, lk;
                                     mask = msk, maxiter = newton_maxiter, tol = newton_tol)
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
    γ̂_free = θ̂[(p + 1):(p + q)]
    γ̂ = collect(Float64, _expand_fixed_zero(γ̂_free, γ_fixed_mask))
    Λ̂ = unpack_lambda(θ̂[(p + q + 1):(p + q + rr)], p, K)
    disp̂ = has_disp ? exp(θ̂[p + q + rr + 1]) : NaN
    return GllvmCovFit(family, β̂, γ̂, collect(Bool, γ_fixed_mask), Λ̂, disp̂, lk, _fit_verdict(res)...)
end
