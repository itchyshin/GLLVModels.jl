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
    return ℓ - 0.5 * dot(z, z) - 0.5 * _laplace_logdet(A)
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

# --- exact gradient for the covariate fitters (2026-10-01) ---
#
# Until this change `fit_gllvm_cov` and `fit_nb_gllvm_grouped_cov` drove L-BFGS with
# `autodiff = :finite`: 2·nθ full marginal evaluations (each n per-site Newton mode
# searches) per gradient, plus finite-difference noise that kept the fits from reaching
# `g_tol` (fungi 100 sites x 10 species ran to the 500-iteration cap, unconverged).
#
# This is the `src/laplace_grad.jl` construction with an offset: solve each site's mode
# ẑ once at the primal θ (the SAME mode search the objective uses), then evaluate the
# site marginal at the differentiable one-Newton-step point
#       z(θ) = ẑ + H(ẑ,θ)⁻¹ (Λ's(ẑ;θ) − ẑ),
# with H = Λ'W_obsΛ + I the exact Hessian of the per-site log-posterior, and take
# ForwardDiff through it. z(θ̂) = ẑ and dz/dθ is the implicit dẑ/dθ, so the result is
# the exact gradient of the objective, log-det and implicit terms included. The log-det
# weight is the one the objective's own kernel uses (`_laplace_site_off` for
# `fit_gllvm_cov`, `_nb_grouped_loglik_site` for the grouped NB2 route).
#
# `fams` is indexable per species (a vector), so the one kernel serves the shared-
# dispersion route (all entries equal) and the per-trait-dispersion NB2 route.

# Exact per-cell curvature −∂²ℓ/∂η² of the coded log-density (the implicit step's
# weight). Equal to the Fisher weight where the family declares so.
_cov_obs_weight(f, μ, n, me, y, link::Link, η) =
    _glm_weight_matches_observed(f, link) ? _glm_weight(f, μ, n, me) :
                                            _glm_obs_weight(f, μ, n, me, y, link, η)

# Differentiable per-site Laplace marginal at the implicit-step point. `η0 = β + O[:, s]`
# and `Λ` may carry duals; `ẑ` is the concrete (Float64) converged mode. `logdet_w(t, μ,
# me, η)` returns the objective's log-det weight for cell t.
function _cov_site_diffable(fams, y::AbstractVector, n::AbstractVector, Λ::AbstractMatrix,
        η0::AbstractVector, link::Link, ẑ::AbstractVector, logdet_w; mask = nothing)
    p = size(Λ, 1)
    obs(t) = mask === nothing || mask[t]
    η  = _clamp_eta.(η0 .+ Λ * ẑ)
    μ  = [_clamp_mu(fams[t], linkinv(link, η[t])) for t in 1:p]
    me = mu_eta.(Ref(link), η)
    T  = eltype(μ)
    s  = [obs(t) ? _glm_score(fams[t], μ[t], n[t], me[t], y[t]) : zero(T) for t in 1:p]
    Wo = [obs(t) ? _cov_obs_weight(fams[t], μ[t], n[t], me[t], y[t], link, η[t]) : zero(T)
          for t in 1:p]
    H  = Λ' * (Wo .* Λ) + I
    z  = ẑ .+ (Matrix(H) \ (Λ' * s .- ẑ))

    ηz  = _clamp_eta.(η0 .+ Λ * z)
    μz  = [_clamp_mu(fams[t], linkinv(link, ηz[t])) for t in 1:p]
    mez = mu_eta.(Ref(link), ηz)
    Tz  = eltype(μz)
    W   = [obs(t) ? logdet_w(t, μz[t], mez[t], ηz[t]) : zero(Tz) for t in 1:p]
    A   = Matrix(Λ' * (W .* Λ) + I)
    ℓ = zero(Tz)
    @inbounds for t in 1:p
        obs(t) || continue
        ℓ += _glm_logpdf(fams[t], μz[t], n[t], y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * _laplace_logdet(A)
end

# Optimiser for the exact-gradient route: dense BFGS, not L-BFGS. These objectives are
# ill-conditioned (fungi 100 x 10: R's own cond(H) is 1.2e8, with near-separated species
# pushing intercepts and loadings far out). With the exact gradient, L-BFGS (m = 10) crawled
# along the flat valley and hit the 500-iteration cap short of the optimum (negll 531.74 after
# 500 iterations, 531.30 after 3000, against 530.484071); dense BFGS reaches 530.484071
# (R's value, to 1e-8) in 190 iterations. The inverse-Hessian cost is O(nθ²) per step,
# negligible next to the per-site mode searches at the θ sizes these fitters meet.
_COV_BFGS() = Optim.BFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))

# Families whose marker can carry a ForwardDiff dual dispersion (NB1's field is
# Float64-typed, so NB1 keeps the finite-difference gradient).
_cov_ad_ok(::NB1) = false
_cov_ad_ok(f)     = true

# −∇θ of `fit_gllvm_cov`'s objective, θ = [β; γ_free; pack(Λ); (log disp)], or
# `nothing` when any site's mode search fails (the objective is the 1e12 sentinel there)
# or the AD pass errors; `_optimize_with_analytic` then falls back to a central finite
# difference at that θ.
function _cov_negll_grad(family, Yc, Nm, X_fit, θ, p, q, K, rr, has_disp, lk, msk,
                         maxiter, tol, prior_nll = nothing)
    try
        n = size(Yc, 2)
        β = θ[1:p]; γ = θ[(p + 1):(p + q)]
        Λ = unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        fam = _cov_family(family, has_disp ? exp(θ[p + q + rr + 1]) : NaN)
        O = _build_offset(X_fit, γ)
        ẑs = Vector{Vector{Float64}}(undef, n)
        for s in 1:n
            mi = msk === nothing ? nothing : view(msk, :, s)
            z, ok = _laplace_mode_off_conv(fam, view(Yc, :, s), view(Nm, :, s), Λ,
                                           β .+ view(O, :, s), lk;
                                           mask = mi, maxiter = maxiter, tol = tol)
            ok || return nothing
            ẑs[s] = z
        end
        hess = _default_hessian(fam, lk)
        function marg(θd)
            βd = θd[1:p]; γd = θd[(p + 1):(p + q)]
            Λd = unpack_lambda(θd[(p + q + 1):(p + q + rr)], p, K)
            fd = has_disp ? _cov_family(family, exp(θd[p + q + rr + 1])) : fam
            fams = fill(fd, p)
            fisher = hess === :fisher || _glm_weight_matches_observed(fd, lk)
            Od = _build_offset(X_fit, γd)
            acc = zero(eltype(θd))
            for s in 1:n
                y = view(Yc, :, s); nt = view(Nm, :, s)
                mi = msk === nothing ? nothing : view(msk, :, s)
                ldw = (t, μ, me, η) -> fisher ? _glm_weight(fd, μ, nt[t], me) :
                                                _glm_obs_weight(fd, μ, nt[t], me, y[t], lk, η)
                acc += _cov_site_diffable(fams, y, nt, Λd, βd .+ view(Od, :, s), lk, ẑs[s],
                                          ldw; mask = mi)
            end
            return acc
        end
        g = -ForwardDiff.gradient(marg, θ)
        prior_nll === nothing || (g .+= ForwardDiff.gradient(prior_nll, θ))
        return all(isfinite, g) ? g : nothing
    catch
        return nothing
    end
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

# #1200: when the objective sits on the 1e12 failure sentinel at the starting point
# (one site's Laplace mode search fails at the family's default start dispersion),
# the optimiser has no gradient and exits at iteration 0 with logLik = -Inf. Retry the
# start with a smaller dispersion (3x, 10x, 30x below the default; a smaller Beta/NB
# precision gives flatter, easier mode searches). A start that already evaluates
# finite is never touched, so every fit that worked before is bit-identical.
function _rescue_disp_start!(negll, θ0::AbstractVector, has_disp::Bool, family)
    has_disp || return θ0
    negll(θ0) < 1e12 && return θ0
    ld0 = θ0[end]
    for f in (3.0, 10.0, 30.0)
        θ0[end] = ld0 - log(f)
        negll(θ0) < 1e12 && return θ0
    end
    θ0[end] = ld0
    return θ0
end

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
(`r`/`φ`/`α`, or `NaN` when the family has none), `link`, the data-only Laplace
`loglik`, `converged`, and `iterations`. `prior_metadata` is `nothing` without priors; with
priors it records the input table, resolved targets, separate penalties, and the combined
penalized objective. `convergence` records the objective, gradient, boundary, and Hessian
checks used for the final status.
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
    prior_metadata::Any
    convergence::FitConvergence
end

function Base.getproperty(f::GllvmCovFit, name::Symbol)
    name === :convergence && return getfield(f, :convergence)
    name in _GLLVM_CONVERGENCE_PROPERTIES &&
        return getproperty(getfield(f, :convergence), name)
    return getfield(f, name)
end

function Base.propertynames(::GllvmCovFit, private::Bool = false)
    return (fieldnames(GllvmCovFit)..., _GLLVM_CONVERGENCE_PROPERTIES...)
end

function Base.show(io::IO, f::GllvmCovFit)
    p, K = size(f.Λ); q = length(f.γ)
    print(io, "GllvmCovFit(", nameof(typeof(f.family)), ", p=", p, ", q=", q, ", K=", K)
    any(f.γ_fixed) && print(io, ", fixed γ=", count(f.γ_fixed))
    isnan(f.dispersion) || print(io, ", disp=", round(f.dispersion; sigdigits = 4))
    f.prior_metadata === nothing || print(io, ", prior-penalized")
    print(io, ", loglik=", round(f.loglik; sigdigits = 7), f.converged ? "" : ", NOT CONVERGED", ")")
end

function _gllv_cov_prior_nll(spec, value)
    z = (value - spec.location) / spec.scale
    if spec.kind === :normal
        return log(spec.scale) + 0.5 * log(2π) + 0.5 * z^2
    end
    df = spec.df
    return log(spec.scale) + loggamma(df / 2) + 0.5 * log(df * π) -
           loggamma((df + 1) / 2) + ((df + 1) / 2) * log1p(z^2 / df)
end

function _gllv_cov_prior_penalties(θ, p, q, K, rr, fixed_priors, loading_ridge)
    fixed = zero(eltype(θ))
    for spec in fixed_priors
        index = spec.parameter_role === :response_intercept ? spec.parameter_index : p + spec.parameter_index
        fixed += _gllv_cov_prior_nll(spec, θ[index])
    end
    loading = if isfinite(loading_ridge)
        λ = inv(loading_ridge)^2
        Λ = unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        0.5 * λ * sum(abs2, Λ)
    else
        zero(eltype(θ))
    end
    return fixed, loading
end

"""
    fit_gllvm_cov(Y; family, X, K, link=nothing, N=nothing, γ_fixed=nothing,
                  coefficient_names=nothing, response_names=nothing, priors=nothing,
                  loading_ridge=nothing, …) -> GllvmCovFit

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
`p × n`; `N` supplies Binomial trial counts (default all-ones). Exact gradient
(ForwardDiff through the implicit one-Newton-step construction of
`src/laplace_grad.jl`), with a central finite difference at any θ where it is not
available; NB1 keeps the finite-difference gradient. `γ_fixed` optionally fixes selected covariate coefficients to zero;
pass a Bool vector of length `size(X, 3)`, an integer index vector, or a Dict
index=>0.

For `class="fixef"`, `coef` may name either a shared covariate coefficient in
`coefficient_names` or a response-specific intercept in `response_names`. A
numeric response matrix has no row names, so direct calls default to
`trait_1`, …, `trait_p`; supply explicit names to target particular responses.
Formula calls use the same names, and the long-format formula method uses its
sorted species levels.

For the Binomial Laplace route, `priors` accepts a Tables.jl table with named
`class="fixef"` rows using `normal(location,scale)` or
`student_t(location,scale,df)` on `dpar="mu", scale="link"`. It may also
contain one `class="loading", coef="all"` row using `ridge(lambda)` on
`scale="loading_coef"`. Direct calls resolve names against
`coefficient_names`, which defaults to `x1`, `x2`, …; formula calls provide the
model-matrix names. Response intercept selectors resolve against `response_names`.
Other families and routes reject this table. Prior fits set
`prior_metadata` and retain the data-only `loglik`; their `confint` route is
refused because penalized curvature is not calibrated uncertainty.

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
        γ_fixed = nothing, coefficient_names = nothing, response_names = nothing, priors = nothing,
        loading_ridge = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q) = ($p, $n, q); got $(size(X))"))
    q_full = size(X, 3)
    γ_fixed_mask = _fixed_zero_mask(γ_fixed, q_full, "γ_fixed")
    cnames = coefficient_names === nothing ? ["x$i" for i in 1:q_full] : String.(coefficient_names)
    length(cnames) == q_full || throw(DimensionMismatch(
        "coefficient_names must have length $q_full (got $(length(cnames)))"))
    all(name -> !isempty(name), cnames) || throw(ArgumentError("coefficient_names must not be empty"))
    length(unique(cnames)) == length(cnames) || throw(ArgumentError("coefficient_names must be unique"))
    rnames = response_names === nothing ? ["trait_$i" for i in 1:p] : String.(response_names)
    length(rnames) == p || throw(DimensionMismatch(
        "response_names must have length $p (got $(length(rnames)))"))
    all(name -> !isempty(name), rnames) || throw(ArgumentError("response_names must not be empty"))
    length(unique(rnames)) == length(rnames) || throw(ArgumentError("response_names must be unique"))
    table_controls = priors === nothing ? nothing : _gllv_prior_table_controls(priors)
    priors !== nothing && loading_ridge !== nothing && throw(ArgumentError(
        "pass the loading-ridge row in priors instead of combining priors with loading_ridge"))
    any_prior = table_controls !== nothing || loading_ridge !== nothing
    any_prior && !(family isa Binomial) && throw(ArgumentError(
        "GLLVModels coefficient/loading priors are supported only by the Binomial covariate Laplace route"))
    resolved_fixed_priors = NamedTuple[]
    resolved_fixed_targets = NamedTuple[]
    resolved_targets = NamedTuple[]
    if table_controls !== nothing
        free_indices = findall(!, γ_fixed_mask)
        seen_fixed_targets = Set{Tuple{Symbol,Int}}()
        for spec in table_controls.coefficient_priors
            targets = if spec.coef == "all"
                vcat([(role=:response_intercept, index=i, coef=string(rnames[i], "::(Intercept)"),
                       response=rnames[i]) for i in eachindex(rnames)],
                     [(role=:covariate, index=i, coef=cnames[j], response=nothing)
                      for (i, j) in enumerate(free_indices)])
            else
                response_index = findfirst(==(spec.coef), rnames)
                coefficient_index = findfirst(==(spec.coef), cnames)
                response_index !== nothing && coefficient_index !== nothing && throw(ArgumentError(
                    "fixed-effect prior target `$(spec.coef)` is ambiguous between a response intercept and a covariate coefficient"))
                if response_index !== nothing
                    [(role=:response_intercept, index=response_index,
                      coef=string(rnames[response_index], "::(Intercept)"),
                      response=rnames[response_index])]
                elseif coefficient_index !== nothing
                    γ_fixed_mask[coefficient_index] && throw(ArgumentError(
                        "fixed-effect prior target `$(spec.coef)` is fixed at zero and cannot be estimated"))
                    free_index = findfirst(==(coefficient_index), free_indices)
                    [(role=:covariate, index=free_index, coef=cnames[coefficient_index],
                      response=nothing)]
                else
                    throw(ArgumentError("fixed-effect prior target `$(spec.coef)` is absent from both response_names and coefficient_names"))
                end
            end
            for target in targets
                key = (target.role, target.index)
                key in seen_fixed_targets && throw(ArgumentError(
                    "prior_table contains duplicate or overlapping fixed-effect target `$(target.coef)`"))
                push!(seen_fixed_targets, key)
                push!(resolved_fixed_priors, merge(spec, (; coef=target.coef,
                    parameter_index=target.index, parameter_role=target.role)))
                if target.role === :response_intercept
                    push!(resolved_fixed_targets, (; class="fixef", coef=target.coef,
                        dpar="mu", scale="link", parameter_role=target.role,
                        response=target.response))
                else
                    push!(resolved_fixed_targets, (; class="fixef", coef=target.coef,
                        dpar="mu", scale="link", parameter_role=target.role))
                end
            end
        end
        append!(resolved_targets, resolved_fixed_targets)
        append!(resolved_targets, filter(t -> t.class == "loading", table_controls.resolved_targets))
        any(t -> t.class == "loading", resolved_targets) && K == 0 &&
            throw(ArgumentError("loading prior requires K > 0"))
    end
    tau_loading = if priors !== nothing
        table_controls.loading_ridge
    elseif loading_ridge === nothing
        Inf
    else
        value = Float64(loading_ridge)
        ((isinf(value) && value > 0) || (isfinite(value) && value > 0)) ||
            throw(ArgumentError("loading_ridge must be positive finite or Inf"))
        value
    end
    X_fit, _ = _slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    rr = rr_theta_len(p, K)
    lk = link === nothing ? _cov_default_link(family) : link
    Nm = N === nothing ? fill(1, p, n) : N
    has_disp = _cov_has_disp(family)

    # NA handling: derive the observation mask and a sanitized response matrix.
    msk = _resolve_obs_mask(mask, Y)
    _check_finite_covariates(X, "fit_gllvm_cov"; obs = msk)
    _check_family_response_support(family, Y, msk; caller = "fit_gllvm_cov", N = N)
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

    Λ0 = _lower_triangular_loadings(Λ0)   # #865: pack_lambda truncates the SVD's upper triangle
    θ0 = has_disp ? vcat(β0, zeros(q), pack_lambda(Λ0), log(_cov_disp_init(family))) :
                    vcat(β0, zeros(q), pack_lambda(Λ0))
    function data_negll(θ)
        β = θ[1:p]
        γ = θ[(p + 1):(p + q)]
        Λ = unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        disp = has_disp ? exp(θ[p + q + rr + 1]) : NaN
        O = _build_offset(X_fit, γ)
        _eta_cap_guarded(lk) && _eta_above_clamp(β, O; mask = msk) && return 1e12   # #869: η-clamp plateau
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
    prior_nll = any_prior ? θ -> begin
        fixed_penalty, loading_penalty = _gllv_cov_prior_penalties(
            θ, p, q, K, rr, resolved_fixed_priors, tau_loading)
        fixed_penalty + loading_penalty
    end : nothing
    negll(θ) = begin
        value = data_negll(θ)
        value >= 1e12 && return value
        prior_nll === nothing ? value : value + prior_nll(θ)
    end
    _rescue_disp_start!(negll, θ0, has_disp, family)   # #1200
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = if _cov_ad_ok(family)
        grad = θ -> _cov_negll_grad(family, Yc, Nm, X_fit, θ, p, q, K, rr, has_disp, lk,
                                    msk, newton_maxiter, newton_tol, prior_nll)
        _optimize_with_analytic(negll, grad, θ0, _COV_BFGS(), opts)
    else
        Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    end
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    γ̂_free = θ̂[(p + 1):(p + q)]
    γ̂ = collect(Float64, _expand_fixed_zero(γ̂_free, γ_fixed_mask))
    Λ̂ = unpack_lambda(θ̂[(p + q + 1):(p + q + rr)], p, K)
    disp̂ = has_disp ? exp(θ̂[p + q + rr + 1]) : NaN
    loglik, conv, iters = _fit_verdict(res)
    prior_metadata = if any_prior
        data_nll = data_negll(θ̂)
        fixed_penalty, loading_penalty = _gllv_cov_prior_penalties(
            θ̂, p, q, K, rr, resolved_fixed_priors, tau_loading)
        total_objective = data_nll + fixed_penalty + loading_penalty
        loglik = -data_nll
        metadata = (; input_table = deepcopy(priors), resolved_targets,
           coefficient_names = copy(cnames),
           fixed_effect_penalty = Float64(fixed_penalty),
           loading_penalty = Float64(loading_penalty),
           total_objective = Float64(total_objective),
           penalized_loglik = -Float64(total_objective))
        any(t -> hasproperty(t, :response), resolved_targets) ?
            merge(metadata, (; response_names = copy(rnames))) : metadata
    else
        nothing
    end
    eta_cap = _eta_cap_guarded(lk) && _eta_cap_contact(
        β̂, _build_offset(X_fit, γ̂_free); mask = msk)
    _eta_cap_verdict(conv, lk, β̂, _build_offset(X_fit, γ̂_free); mask = msk,
        caller = "fit_gllvm_cov")
    boundary_params = eta_cap ? ["mu:eta_cap"] : String[]
    loading_scale = max(1.0, maximum(abs, Λ̂; init = 0.0))
    for j in 1:min(K, p)
        abs(Λ̂[j, j]) <= 1e-8 * loading_scale && push!(boundary_params, "loading_factor[$j]")
    end
    if has_disp && (!isfinite(disp̂) || disp̂ <= sqrt(eps(Float64)))
        push!(boundary_params, string(_cov_dispname(family), ":boundary"))
    end
    data_nll_at_fit = data_negll(θ̂)
    fixed_penalty, loading_penalty = _gllv_cov_prior_penalties(
        θ̂, p, q, K, rr, resolved_fixed_priors, tau_loading)
    log_prior = -Float64(fixed_penalty + loading_penalty)
    failed_objective = _nll_failed(negll(θ̂))
    loglik = _nll_failed(data_nll_at_fit) ? -Inf : -Float64(data_nll_at_fit)
    prior_targets = [string(hasproperty(t, :class) ? t.class : "prior", ":",
                           hasproperty(t, :coef) ? t.coef : "")
                     for t in resolved_targets]
    if isfinite(tau_loading) && !any(t -> hasproperty(t, :class) && t.class == "loading", resolved_targets)
        push!(prior_targets, "loading:all")
    end
    gradient = failed_objective ? fill(NaN, length(θ̂)) : _fit_objective_gradient(negll, θ̂)
    hessian = failed_objective ? fill(NaN, length(θ̂), length(θ̂)) : _fit_objective_hessian(negll, θ̂)
    hessian_unpenalised = failed_objective ? fill(NaN, length(θ̂), length(θ̂)) :
        _fit_objective_hessian(data_negll, θ̂)
    lower_boundary = has_disp && disp̂ <= sqrt(eps(Float64)) ? [p + q + rr + 1] : Int[]
    lower_boundary_scores = Dict{Int,Float64}()
    if !isempty(lower_boundary) && !failed_objective
        index = only(lower_boundary)
        lower_boundary_scores[index] = _fit_natural_positive_score(gradient[index], disp̂)
    end
    projected_gradient = failed_objective ? gradient :
        _fit_projected_gradient(gradient; lower = lower_boundary,
            lower_scores = lower_boundary_scores)
    diagnostics = _fit_convergence(optimizer_finished = Optim.converged(res) && !eta_cap,
        objective = failed_objective ? -Inf : loglik + log_prior,
        logLik = loglik, log_prior = log_prior,
        penalized = any_prior, gradient = gradient, projected_gradient = projected_gradient,
        hessian = hessian,
        hessian_unpenalised = hessian_unpenalised, prior_targets = prior_targets,
        boundary_params = boundary_params, boundary_check_available = true,
        relative_gradient_tol = g_tol)
    conv = diagnostics.status === :converged
    if prior_metadata !== nothing
        prior_metadata = merge(prior_metadata, (; logLik=loglik,
            log_prior, objective=loglik + log_prior))
    end
    return GllvmCovFit(family, β̂, γ̂, collect(Bool, γ_fixed_mask), Λ̂, disp̂,
        lk, loglik, conv, iters, prior_metadata, diagnostics)
end
