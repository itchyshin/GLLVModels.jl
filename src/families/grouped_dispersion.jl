# Grouped / species-specific dispersion for the negative binomial (NB2) — gllvm's
# `disp.group`. Each species t carries its own dispersion r_{g(t)} for a group
# assignment g: 1..p → 1..G, so overdispersion can vary across species (or groups
# NOTE (2026-08-25): `hessian` selects the LOG-DET weight ONLY. The Newton mode
# search in every grouped site kernel is Fisher-scored unconditionally, because
# expected information is >= 0 and so keeps `Λ'WΛ + I` SPD, making every step a
# descent step. Previously this one symbol drove BOTH roles, which meant the
# fitters — which default to `:observed` — ran an unguarded, possibly-indefinite
# Newton (Beta's observed weight is measurably negative). The converged mode is
# unaffected either way: it is the fixed point of `Λ's − z = 0`, which does not
# involve W at all. Exception (part of #503): where Fisher scoring fails at an NB2
# site under LogLink, `_nb_grouped_loglik_site` falls back to observed-curvature
# Newton, which is exact and SPD for NB2/log.
#
# of species) instead of one shared r. With G = 1 this reduces EXACTLY to the
# shared-dispersion NB2 fit (`fit_nb_gllvm`): both routes default to
# hessian=:observed (TMB's Laplace curvature) since 2026-08-27, when the shared
# NB2 default flipped on the curvature-adjudication campaign evidence;
# hessian=:fisher selects the previous expected-information objective on both.
#
# Implementation note: the generic Laplace core (families/laplace.jl) broadcasts a
# SINGLE family marker over species (`Ref(family)`). Per-species dispersion instead
# needs a per-species marker, so this is a small isolated parallel of the core's
# site routine that broadcasts a length-p VECTOR of `NegativeBinomial(r_t)` markers
# — reusing the exact same NB `_glm_score`/`_glm_weight`/`_glm_logpdf`/`_clamp_mu`
# pieces. The shared families' hot path is left untouched.

# Exact negative conditional curvature for NB2/log. TMB's Laplace objective uses
# this observed Hessian rather than the expected Fisher information:
# -∂²ℓ/∂η² = r * μ * (r + y) / (r + μ)^2.
function _nb_grouped_laplace_weight(hessian::Symbol, f::NegativeBinomial, μ, me, y, link::Link)
    hessian === :fisher && return _glm_weight(f, μ, 1, me)
    hessian === :observed || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    link isa LogLink || throw(ArgumentError(
        "hessian=:observed is currently supported only for NB2 with LogLink()"))
    return _nb2_observed_weight(μ, f.r, y)
end

# Per-site mode search for the NB2 grouped kernel (part of #503; the #507 NB1 pattern).
# Returns `(z, converged)`. `step_weight` picks the curvature that sets the Newton step;
# it never changes the mode, which is the fixed point of `Λ's − z = 0` whatever W is.
#
# Before this fix the kernel ran undamped Fisher scoring and returned whatever `z` it
# held when the loop stopped, converged or not. Where a count sits far above its mean
# the NB2 Fisher weight μr/(r+μ) under-states the observed curvature μr(r+y)/(r+μ)² by
# the factor (r+y)/(r+μ), so the Fisher step overshoots and can settle into a 2-cycle.
# The site then returned a finite value away from the mode. On a p=20, n=300, K_true=3
# NB fixture (test/fixtures/nb2_grouped_seed1_Y.toml) this steered L-BFGS to poor optima
# that reported converged = true.
#
# Now a step that lowers the per-site log-posterior is halved (the generic core's
# `_laplace_mode` rule, so small steps and accepted full steps are bit-identical to the
# old loop), and `converged` is true only when the full proposed step is below `tol`,
# which is the old loop's own stopping test.
function _nb_grouped_mode(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link, step_weight::Symbol;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    z = zeros(K)
    for _ in 1:maxiter
        η  = _clamp_eta.(β .+ off .+ Λ * z)
        μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
        me = mu_eta.(Ref(link), η)
        s  = _glm_score.(fams, μ, n, me, y)
        W  = _nb_grouped_laplace_weight.(Ref(step_weight), fams, μ, me, y, Ref(link))
        if mask !== nothing
            s = ifelse.(mask, s, 0.0)
            W = ifelse.(mask, W, 0.0)
        end
        A  = Symmetric(Λ' * (W .* Λ) + I)
        Δ  = _safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && return z, false
        maximum(abs, Δ) < tol && return z .+ Δ, true
        if norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        else
            q0 = _grouped_laplace_mode_logpost(fams, y, n, Λ, β, link, z;
                                               mask = mask, offset = offset)
            if isfinite(q0)
                accepted = false
                step = 1.0
                for _half in 1:30
                    ztrial = z .+ step .* Δ
                    q1 = _grouped_laplace_mode_logpost(fams, y, n, Λ, β, link, ztrial;
                                                       mask = mask, offset = offset)
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

# The NB2 site-mode chain shared by the likelihood and getLV (#503 follow-up): damped
# Fisher, observed-Newton under LogLink, then a 20x Fisher retry. Returns (z, converged).
function _nb_grouped_site_mode(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    z, ok = _nb_grouped_mode(fams, y, n, Λ, β, link, :fisher;
                             mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    # Fallback 1 (part of #503). Under LogLink the observed NB2 weight μr(r+y)/(r+μ)²
    # is positive, so this is an exact, always-SPD Newton step on the concave per-site
    # log-posterior. It converges in a few steps where Fisher scoring 2-cycles
    # (measured: 5 steps at the fixture site, where Fisher had not settled after 2000).
    # It runs only where Fisher failed. Sites where the old loop converged keep its
    # path; the damped search's full-step certification moves their values by about
    # 1e-10 to 1e-8.
    if !ok && link isa LogLink
        z, ok = _nb_grouped_mode(fams, y, n, Λ, β, link, :observed;
                                 mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    end
    # Fallback 2 (mirrors #507's NB1 retry): other links have no observed weight here,
    # so retry Fisher scoring with a 20x budget from z = 0.
    ok || ((z, ok) = _nb_grouped_mode(fams, y, n, Λ, β, link, :fisher;
                                      mask = mask, offset = offset,
                                      maxiter = 20 * maxiter, tol = tol))
    return z, ok
end

# Per-site Laplace log-marginal with per-species NB dispersion markers `fams`.
function _nb_grouped_loglik_site(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, hessian::Symbol = :observed,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    # Role separation (2026-08-25). The MODE SEARCH is Fisher-scored first:
    # `:fisher` here is not the caller's selector, which governs only the post-loop
    # log-det below. Expected information is >= 0, so `Λ'WΛ + I` is SPD by
    # construction. The converged mode is unchanged either way: it is the fixed point
    # of `Λ's − z = 0`, which does not involve W at all. W only sets the step.
    z, ok = _nb_grouped_site_mode(fams, y, n, Λ, β, link;
                                  mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    # A search that did not converge must not produce a finite value. -Inf makes the
    # fitters' objective return its 1e12 sentinel instead of a garbage surface (#503,
    # the #479 pattern).
    ok || return -Inf
    η  = _clamp_eta.(β .+ off .+ Λ * z)
    μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
    me = mu_eta.(Ref(link), η)
    W  = _nb_grouped_laplace_weight.(Ref(hessian), fams, μ, me, y, Ref(link))
    if mask !== nothing
        W = ifelse.(mask, W, 0.0)
    end
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = 0.0
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _glm_logpdf(fams[t], μ[t], n[t], y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

# Per-species log-posterior for one site: the backtracking merit function.
# Mirrors `_laplace_mode_logpost` (families/laplace.jl) with a fams vector.
function _grouped_laplace_mode_logpost(fams::AbstractVector, y::AbstractVector,
        n::AbstractVector, Λ::AbstractMatrix, β::AbstractVector, link::Link,
        z::AbstractVector; mask = nothing, offset = nothing)
    p = size(Λ, 1)
    off = offset === nothing ? false : offset
    η = _clamp_eta.(β .+ off .+ Λ * z)
    μ = _clamp_mu.(fams, linkinv.(Ref(link), η))
    q = -0.5 * dot(z, z)
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        q += _laplace_mode_merit_term(fams[t], μ[t], n[t], y[t])
    end
    return q
end

# Gradient of that per-site log-posterior, `Λ's − z`: the overshoot test's probe.
function _grouped_laplace_mode_grad(fams::AbstractVector, y::AbstractVector,
        n::AbstractVector, Λ::AbstractMatrix, β::AbstractVector, link::Link,
        z::AbstractVector; mask = nothing, offset = nothing)
    off = offset === nothing ? false : offset
    η  = _clamp_eta.(β .+ off .+ Λ * z)
    μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
    me = mu_eta.(Ref(link), η)
    s  = _glm_score.(fams, μ, n, me, y)
    mask === nothing || (s = ifelse.(mask, s, 0.0))
    return Λ' * s .- z
end

function _grouped_laplace_mode(fams::AbstractVector, y::AbstractVector,
        n::AbstractVector, Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    length(fams) == p || throw(ArgumentError("length(fams)=$(length(fams)) must equal p=$p"))
    length(y) == p || throw(ArgumentError("length(y)=$(length(y)) must equal p=$p"))
    length(n) == p || throw(ArgumentError("length(n)=$(length(n)) must equal p=$p"))
    K == 0 && return zeros(Float64, 0)
    off = offset === nothing ? false : offset
    z = zeros(K)
    # Restart/backtracking safety, mirrored from the generic `_laplace_mode`
    # (2026-08-27). Without it an undamped Newton overshoot let ‖Λ‖ run away
    # (~960 against a true 0.38 on the Exponential :observed route, which
    # evaluates through this kernel; 75% of the curvature-adjudication
    # campaign's Exponential cells produced garbage estimates the same way).
    # Well-behaved cells are BIT-IDENTICAL: small steps and accepted full
    # steps update `z .+ Δ` exactly as before; only steps that DECREASE the
    # per-site log-posterior are halved.
    backtrack = any(_laplace_mode_should_backtrack, fams)
    restarted = false
    @inbounds for _ in 1:maxiter
        η  = _clamp_eta.(β .+ off .+ Λ * z)
        μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
        me = mu_eta.(Ref(link), η)
        s  = _glm_score.(fams, μ, n, me, y)
        W  = _glm_weight.(fams, μ, n, me)
        if mask !== nothing
            s = ifelse.(mask, s, 0.0)
            W = ifelse.(mask, W, 0.0)
        end
        A  = Symmetric(Λ' * (W .* Λ) + I)
        g  = Λ' * s .- z
        Δ  = _safe_solve(A, g)
        if Δ === nothing || !all(isfinite, Δ)
            if !restarted
                z = zeros(K)
                restarted = true
                continue
            end
            break
        end
        step_taken = 1.0
        if !backtrack
            z = z .+ Δ
        else
            # A trial step must pass two tests; each failure halves it.
            # (1) Large steps only: it must not lower the per-site log-posterior.
            # (2) Overshoot guard (2026-09-28): the gradient along Δ at the trial
            #     point must not have turned past −½ of its starting value, i.e. the
            #     step may not exceed 1.5 Newton steps along Δ. Test (1) alone let
            #     Fisher scoring 2-cycle around the mode wherever the observed
            #     curvature is about twice the Fisher weight W: an overshoot of just
            #     under 2 still raises the log-posterior, and small steps skipped (1)
            #     entirely. The loop then stopped wherever `maxiter` fell (truncated
            #     NB2, y = 121 at μ ≈ 27.5: off by 1.3e-3, a 5.5e-4 jump in the
            #     Laplace objective for a 1e-5 step in log r). With (2) each accepted
            #     step contracts the error by at least ½ on a locally quadratic
            #     log-posterior. A full step that passes is bit-identical to the old
            #     update.
            small = norm(Δ) <= 1e-3 * (1 + norm(z))
            q0 = small ? NaN : _grouped_laplace_mode_logpost(fams, y, n, Λ, β, link, z;
                                                             mask = mask, offset = offset)
            if !small && !isfinite(q0)
                z = z .+ Δ
            else
                gΔ = dot(g, Δ)
                accepted = false
                for _half in 1:30
                    ztrial = step_taken == 1.0 ? z .+ Δ : z .+ step_taken .* Δ
                    ok = true
                    if !small
                        q1 = _grouped_laplace_mode_logpost(fams, y, n, Λ, β, link, ztrial;
                                                           mask = mask, offset = offset)
                        ok = isfinite(q1) && q1 >= q0
                    end
                    if ok && gΔ > 0
                        g1Δ = dot(_grouped_laplace_mode_grad(fams, y, n, Λ, β, link, ztrial;
                                                             mask = mask, offset = offset), Δ)
                        ok = isfinite(g1Δ) && g1Δ >= -0.5 * gΔ
                    end
                    if ok
                        z = ztrial
                        accepted = true
                        break
                    end
                    step_taken *= 0.5
                end
                # As before: a large step with no acceptable length ends the search;
                # a small one takes the last (tiny) trial, which trips the stop test.
                if !accepted
                    small || break
                    z = z .+ step_taken .* Δ
                end
            end
        end
        step_taken * maximum(abs, Δ) < tol && break
    end
    return z
end

# getLV must return the z at which the fit's Laplace objective was evaluated. Families
# whose likelihood kernel has its own mode chain (NB2, NB1, Gamma, Beta) reuse it; where
# that chain fails (the likelihood returned -Inf there) keep the generic kernel's z.
# Everything else keeps `_grouped_laplace_mode`.
_grouped_site_mode(fams::AbstractVector{<:NegativeBinomial}, a...; kw...) = _nb_grouped_site_mode(fams, a...; kw...)
_grouped_site_mode(fams::AbstractVector{<:NB1}, a...; kw...)              = _nb1_grouped_site_mode(fams, a...; kw...)
_grouped_site_mode(fams::AbstractVector{<:Gamma}, a...; kw...)            = _gamma_grouped_site_mode(fams, a...; kw...)
_grouped_site_mode(fams::AbstractVector{<:Beta}, a...; kw...)             = _beta_grouped_site_mode(fams, a...; kw...)
_grouped_site_mode(fams::AbstractVector, a...; kw...)                     = (_grouped_laplace_mode(fams, a...; kw...), true)
function _grouped_getLV_mode(fams, y, n, Λ, β, link; mask = nothing, offset = nothing)
    size(Λ, 2) == 0 && return zeros(Float64, 0)
    z, ok = _grouped_site_mode(fams, y, n, Λ, β, link; mask = mask, offset = offset)
    return ok ? z : _grouped_laplace_mode(fams, y, n, Λ, β, link; mask = mask, offset = offset)
end

function _grouped_getLV(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector,
        link::Link, fams::AbstractVector; N = nothing, rotate::Bool = true,
        mask = nothing, offset = nothing)
    p, n = size(Y)
    length(fams) == p || throw(ArgumentError("length(fams)=$(length(fams)) must equal p=$p"))
    Nm = N === nothing ? ones(Int, p, n) : N
    size(Nm) == (p, n) || throw(ArgumentError("N must have size $(p)×$(n); got $(size(Nm))"))
    K = size(Λ, 2)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        oi = offset === nothing ? nothing : view(offset, :, s)
        Z[:, s] = _grouped_getLV_mode(fams, view(Y, :, s), view(Nm, :, s),
                                      Λ, β, link; mask = mi, offset = oi)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(Λ) : Zt
end

# The offset `getLV` / `predict` add for a grouped fit that keeps its training offset in
# `fit.offset` (NBGroupedFit, NB1GroupedFit, BetaGroupedFit, GammaGroupedFit; #788). An
# explicit matrix whose size is not that of Y is a DimensionMismatch, as before the fit kept
# its offset; otherwise the rule of `_laplace_prediction_offset` (src/postfit.jl): a Y of
# the training size uses the stored offset, new units need their own offset (a p×n matrix,
# a scalar or a length-p vector), and an offset fit refuses new units without one.
function _grouped_prediction_offset(fit, Y::AbstractMatrix, offset, mask,
                                    caller::AbstractString)
    offset isa AbstractMatrix && size(offset) != size(Y) &&
        throw(DimensionMismatch("offset must have size $(size(Y)); got $(size(offset))"))
    return _laplace_prediction_offset(fit.offset, Y, offset, mask, caller)
end

"""
    nb_grouped_marginal_loglik_laplace(Y, Λ, β, rvec; link=LogLink(), mask=nothing,
                                       offset=nothing, hessian=:observed, kwargs...) -> Float64

Total Laplace log-marginal of a negative-binomial GLLVM with **per-species**
dispersion `rvec` (length p; `Var_t = μ_t + μ_t²/rvec[t]`). `Y` is the p×n integer
count matrix; `Λ` p×K; `β` length-p. With a constant `rvec = fill(r, p)` this
equals the shared-dispersion [`nb_marginal_loglik_laplace`](@ref) to machine
precision — both default `hessian=:observed` (TMB's conditional NB2/log
Hessian) since 2026-08-27; `hessian=:fisher` selects the previous
expected-information objective on both routes.

Returns `-Inf` when a site's mode search does not converge (damped Fisher scoring,
then observed-curvature Newton under `LogLink`, then a 20x Fisher retry), so a
caller's objective sees a failure rather than a value away from the mode.
"""
function nb_grouped_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, rvec::AbstractVector; link::Link = LogLink(),
        mask = nothing, offset = nothing, hessian::Symbol = :observed, kwargs...)
    p = size(Λ, 1)
    length(rvec) == p || throw(ArgumentError("length(rvec)=$(length(rvec)) must equal p=$p"))
    N = ones(Int, size(Y))
    fams = [NegativeBinomial(float(rvec[t]), 0.5) for t in 1:p]
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask   === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        acc += _nb_grouped_loglik_site(fams, view(Y, :, i), view(N, :, i), Λ, β, link;
                                       mask = mi, offset = oi, hessian = hessian, kwargs...)
    end
    return acc
end

# Shared per-group dispersion-boundary rule for the four grouped-dispersion
# families in this file (NB2 `r_group`, NB1/Beta/Gamma per-group dispersion) —
# T14 F1, 2026-09-02 (docs/dev-log/core070/t14-nb2-wald-nan-diagnosis.md). Mirrors
# `_studentt_nu_boundary` (ν > 1e6, `studentt.jl:267-268`) and Tweedie's
# `_TWEEDIE_XI_MAX`: a fitted per-group dispersion outside `[1e-6, 1e6]` means that
# group's curvature direction in the joint Wald Hessian is numerically flat
# (Fisher information ~0 relative to the other groups) — the same order of
# magnitude R's `sdreport` treats as a degenerate block. Each family's own
# limiting interpretation of the two ends (all on the SAME 1e-6/1e6 magnitude as
# Student-t/Tweedie, for cross-family consistency — nothing in any of these four
# parameterisations demands a different constant):
#   - NB2 `r_group` (`Var = μ + μ²/r`): r > 1e6 is the Poisson limit (no
#     overdispersion left to estimate); r < 1e-6 is extreme overdispersion,
#     equally unidentified from finite data.
#   - NB1 `φ` (`Var = μ(1+φ)`): φ < 1e-6 is the Poisson limit; φ > 1e6 is
#     numerically flat overdispersion (Fisher information in log φ vanishes as
#     φ → ∞, mirroring r → ∞ for NB2).
#   - Beta `φ` (`Var = μ(1-μ)/(1+φ)`): φ < 1e-6 is the maximal-variance
#     (near-Bernoulli) limit; φ > 1e6 is the near-deterministic collapse
#     (Var → 0), which is identified, so grouped Beta fits flag only the lower
#     end (`_dispersion_group_lower_boundary` below).
#   - Gamma `α` (`Var = μ²/α`): α < 1e-6 is extreme overdispersion (Var → ∞);
#     α > 1e6 is the near-deterministic collapse (Var → 0), which is identified,
#     so grouped Gamma fits flag only the lower end.
_dispersion_group_boundary(dvec::AbstractVector{<:Real}) =
    Bool[(d > 1e6) || (d < 1e-6) for d in dvec]

# Lower end only, for the Gamma shape α and the Beta precision φ (maintainer
# decision 2026-09-29). Their upper end is the near-deterministic collapse
# (Var → 0), which the data identify rather than a flat-likelihood limit: on data
# drawn with true α = 1e8 or φ = 1e8 the grouped fits estimate 8.9e7 to 1.3e8
# (test/fixtures/gamma_beta_upper_boundary.toml), so flagging α, φ > 1e6 reported
# a well-identified optimum as not converged. NB2 r and NB1 φ keep both ends in the
# `dispersion_boundary` field (`_dispersion_group_boundary`). For `converged`, the
# Poisson limit only warns (maintainer decision 2026-09-29: gllvmTMB itself puts a
# trait's dispersion there on about 4 of 5 ordinary NB fits). That limit is the upper
# end for NB2 r (so the grouped NB2 fitters use this helper) but the lower end for NB1 φ
# (so the grouped NB1 fitters test `φ > 1e6` instead).
# The bootstrap adapters already leave the Gamma/Beta upper end unflagged (#565, #568).
_dispersion_group_lower_boundary(dvec::AbstractVector{<:Real}) =
    Bool[d < 1e-6 for d in dvec]

# NB2 grouped fits can stall with a group's log r out at the Poisson boundary, where
# the likelihood is nearly flat, well below a better point (#477). From the returned
# point, restart with the boundary groups at r = 1: all of them together and, when
# there are several, each one on its own (a group can genuinely belong at the
# boundary while another does not). Keep the best result, and only if it lowers the
# negative log-likelihood by more than 1e-6. Fits that never reach the boundary are
# returned unchanged. The likelihood can have several maxima on small data, so this
# is a better local search, not a guarantee of the global maximum.
# `grad` (optional, −∇negll closure or `nothing`): when given, the restarts use the same
# exact-gradient optimiser as the main fit (`_optimize_with_analytic`).
function _nb_boundary_restart(negll, res, ls, opts, first_log_r::Integer; grad = nothing)
    θ = Optim.minimizer(res)
    bd = findall(_dispersion_group_boundary(exp.(θ[first_log_r:end])))
    isempty(bd) && return res
    trials = length(bd) == 1 ? [bd] : vcat([bd], [[g] for g in bd])
    best = res
    for groups in trials
        θs = copy(θ)
        θs[first_log_r - 1 .+ groups] .= 0.0
        trial = grad === nothing ?
            Optim.optimize(negll, θs, ls, opts; autodiff = :finite) :
            _optimize_with_analytic(negll, grad, θs, ls, opts)
        Optim.minimum(trial) < Optim.minimum(best) - 1e-6 && (best = trial)
    end
    return best
end

# Toward the Poisson limit the likelihood in log r flattens but keeps rising, so
# L-BFGS can crawl along that ridge until the iteration cap and report a fit that is
# not converged (#615: per-species fits at p = 12, n = 60 stopped at r = 1.8e4 and
# 9.5e5 after 500 iterations). Only for such a fit, run up to `rounds` more fits from
# the current point. Each round fixes the groups with r > 1e3 at r = 1e10 and refits
# the other parameters. In the first round, when several groups qualify, it also tries
# each one alone; every round finally tries a plain continuation with nothing fixed.
# A round keeps its first refit that is no worse than the current point (by 1e-6) and
# either converged or strictly better, and the rounds stop once the fit converges or
# a round keeps nothing. Extra rounds matter at p = 24 (#615): there the first refit
# often stopped at the cap just short of convergence, and a species below 1e3 could
# keep crawling up the ridge. A group fixed at 1e10 reports r_group = 1e10, which the
# fitters already treat as the Poisson limit (warn only).
# `grad` (optional, full-θ exact gradient) is restricted to the free coordinates in
# every round. Returns `(θ, nll, converged, iterations)`; a converged fit is returned unchanged.
# The refits use dense BFGS (`alg`), not the caller's `ls`: near the optimum L-BFGS
# crawls (#615: 17 common-r p = 24 fits stayed non-converged after three L-BFGS
# rounds, while BFGS from the same points converged on 17 of 17 within 126
# iterations). `alg` matches `_COV_BFGS()`, so the exact-gradient route is unchanged.
# The limit is a keyword so NB1 can use it too: there the Poisson limit is φ → 0, so the
# NB1 fitters pass `at_limit = <(1e-3)` and `limit_log = log(1e-10)`. `conv_test`
# judges each fit (the NB1 fitters add their #485 gradient test to Optim's verdict).
function _nb_poisson_ridge_polish(negll, res, ls, opts, first_log_r::Integer;
                                  rounds::Integer = 3, grad = nothing,
                                  alg = Optim.BFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)),
                                  at_limit = >(1e3), limit_log = log(1e10),
                                  conv_test = Optim.converged)
    θ = Optim.minimizer(res)
    f0 = Optim.minimum(res)
    conv = conv_test(res)
    iters = Optim.iterations(res)
    (conv || _nll_failed(f0)) && return (θ, f0, conv, iters)
    for round in 1:rounds
        cand = findall(at_limit, exp.(θ[first_log_r:end]))
        trials = round == 1 && length(cand) > 1 ? vcat([cand], [[g] for g in cand]) : [cand]
        isempty(cand) || push!(trials, Int[])
        kept = false
        for groups in trials
            fixed = first_log_r - 1 .+ groups
            free = setdiff(eachindex(θ), fixed)
            θs = copy(θ)
            θs[fixed] .= limit_log
            sub(x) = negll(setindex!(copy(θs), x, free))
            # With an exact full-θ gradient, restrict it to the free coordinates.
            trial = grad === nothing ?
                Optim.optimize(sub, θs[free], alg, opts; autodiff = :finite) :
                _optimize_with_analytic(sub, x -> begin
                        g = grad(setindex!(copy(θs), x, free))
                        g === nothing ? nothing : g[free]
                    end, θs[free], alg, opts)
            f1 = Optim.minimum(trial)
            c1 = conv_test(trial)
            if f1 <= f0 + 1e-6 && (c1 || f1 < f0)
                θs[free] = Optim.minimizer(trial)
                θ, f0, conv = θs, f1, c1
                iters += Optim.iterations(trial)
                kept = true
                break
            end
        end
        (conv || !kept) && break
    end
    return (θ, f0, conv, iters)
end

"""
    NBGroupedFit

Result of [`fit_nb_gllvm_grouped`](@ref): intercepts `β` (length p), loadings `Λ`
(p×K), the per-group dispersion vector `r_group` (length G), the species→group map
`group` (length p), the `link`, the maximised Laplace `loglik`, `converged`, and
`iterations`. The per-species dispersion is `r_group[group[t]]`. `dispersion_boundary`
(length G, T14 F1, 2026-09-02) flags groups whose fitted `r_group[g]` fell outside
`[1e-6, 1e6]` (see `_dispersion_group_boundary`) — the Poisson limit at the upper
end, extreme unidentified overdispersion at the lower end. Only the lower end
(`r_group[g] < 1e-6`) forces `converged` to `false`; an `r_group[g] > 1e6` (the
Poisson limit, which is a normal outcome) only emits a warning and leaves `converged`
at the optimizer verdict (maintainer decision 2026-09-29).
`offset` is the p×n training offset the fit was made with (`nothing` when the fit had
none); [`getLV`](@ref), [`predict`](@ref) and [`residuals`](@ref) use it by default.
"""
struct NBGroupedFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    r_group::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    dispersion_boundary::Vector{Bool}   # per-group Poisson-limit / degenerate flag (T14 F1)
    offset::Union{Nothing,Matrix{Float64}}   # training offset (p×n); `nothing` = none (#788)
end

# Positional compatibility constructors (2026-08-28 hessian; T14 F1 dispersion_boundary,
# 2026-09-02): every pre-existing construction site builds a default-curvature fit; the
# `hessian` field records the objective identity so `confint`/bootstrap can rebuild THE
# SAME objective instead of guessing (the audit's confint-consistency class).
# `dispersion_boundary` is DERIVED from `r_group` for backward-compatible call sites,
# mirroring `StudentTFit`'s `_studentt_nu_boundary` compat cascade (studentt.jl:283-289).
NBGroupedFit(β, Λ, r_group, group, link, loglik, converged, iterations) =
    NBGroupedFit(β, Λ, r_group, group, link, loglik, converged, iterations, :observed)
NBGroupedFit(β, Λ, r_group, group, link, loglik, converged, iterations, hessian::Symbol) =
    NBGroupedFit(β, Λ, r_group, group, link, loglik, converged, iterations, hessian,
                _dispersion_group_boundary(r_group))
# Pre-offset compat tier (10 positional args, #788): no stored training offset.
NBGroupedFit(β, Λ, r_group, group, link, loglik, converged, iterations, hessian, dispersion_boundary) =
    NBGroupedFit(β, Λ, r_group, group, link, loglik, converged, iterations, hessian, dispersion_boundary, nothing)

function Base.show(io::IO, f::NBGroupedFit)
    p, K = size(f.Λ)
    print(io, "NBGroupedFit(p=", p, ", K=", K, ", G=", length(f.r_group),
          ", r_group=", round.(f.r_group; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          any(f.dispersion_boundary) ?
              ", dispersion at boundary (groups $(findall(f.dispersion_boundary)))" : "",
          f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::NBGroupedFit) = fit.Λ
_loglik(fit::NBGroupedFit)   = fit.loglik

# Free params: β (p) + reduced loadings Λ + one dispersion per group (G).
function _nparams(fit::NBGroupedFit)
    p, K = size(fit.Λ)
    return p + rr_theta_len(p, K) + length(fit.r_group)   # β + Λ + G dispersions r
end

"""
    getLV(fit::NBGroupedFit, Y; N=nothing, rotate=true, mask=nothing, offset=nothing) -> n×K matrix

Conditional latent-variable scores for a grouped-dispersion NB2 fit, using the
per-trait dispersion `r_group[group[t]]` in the same Laplace mode equations as
the grouped likelihood.

On a fit made with an `offset` the mode search uses it (η = β + offset + Λz): the
stored training offset (`fit.offset`) when `Y` has the training size, otherwise the
`offset` you pass (a p×n matrix, a scalar or a length-p vector). New units from an
offset fit without an `offset` are refused, as gllvmTMB refuses `newdata` that lacks
the offset variable.
"""
function getLV(fit::NBGroupedFit, Y::AbstractMatrix{<:Integer};
               N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
               rotate::Bool = true, mask = nothing, offset = nothing)
    p = size(Y, 1)
    offset = _grouped_prediction_offset(fit, Y, offset, mask, "getLV")
    rvec = [fit.r_group[fit.group[t]] for t in 1:p]
    fams = [NegativeBinomial(float(rvec[t]), 0.5) for t in 1:p]
    return _grouped_getLV(Y, fit.Λ, fit.β, fit.link, fams;
                          N = N, rotate = rotate, mask = mask, offset = offset)
end

# Warm start shared by the grouped NB2 / Beta / NB1 fitters: `β_init` and `Λ_init` when
# given (the `fit_nb_gllvm` convention, src/families/negbin.jl), otherwise empirical
# link-scale means and SVD loadings of the centred link-scale data. As in `fit_nb_gllvm`,
# a supplied `β_init` also centres the data the default loadings are taken from.
function _grouped_warm_start(Zemp::AbstractMatrix, K::Integer, β_init, Λ_init)
    p, n = size(Zemp)
    if β_init !== nothing
        length(β_init) == p || throw(ArgumentError(
            "β_init has length $(length(β_init)); expected p = $p"))
    end
    if Λ_init !== nothing
        size(Λ_init) == (p, K) || throw(ArgumentError(
            "Λ_init has size $(size(Λ_init)); expected (p, K) = ($p, $K)"))
    end
    β0 = β_init === nothing ? vec(sum(Zemp; dims = 2)) ./ n : collect(float.(β_init))
    Λ_init === nothing || return β0, collect(float.(Λ_init))
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    return β0, Λ0
end

"""
    fit_nb_gllvm_grouped(Y; K, group, link=LogLink(), mask=nothing, offset=nothing,
                         hessian=:observed, β_init=nothing, Λ_init=nothing, …) -> NBGroupedFit

Fit a negative-binomial GLLVM with grouped / species-specific dispersion (gllvm's
`disp.group`): species `t` shares dispersion `r_group[group[t]]`. `group` is a
length-p vector of group ids (relabelled to `1..G` internally). L-BFGS over
`[β; vec(Λ); log r_1 … log r_G]`; finite-difference gradient; warm start from
empirical log-means + SVD loadings + a moderate per-group `r₀`. If a group's `r`
ends at the Poisson boundary (outside `[1e-6, 1e6]`), the fit restarts from that
point with the boundary groups at `r = 1` (together, and each on its own when there
are several) and keeps the best restart only if its log-likelihood is higher by
more than `1e-6`; otherwise the boundary fit stands and is flagged in
`dispersion_boundary` (which flags both ends; only `r < 1e-6` also makes `converged`
false, `r > 1e6` warns). `iterations` then counts the kept run only. On small data the
likelihood can have several maxima, so this improves the local search but does not
guarantee the global maximum. The per-site mode search is damped and falls back to
an exact observed-curvature Newton search where Fisher scoring does not converge; a
site that still fails makes the objective return its failure sentinel. With one group this
matches [`fit_nb_gllvm`](@ref) when `hessian=:fisher`. `hessian=:observed` (the
default) uses the exact conditional NB2/log curvature used by TMB's Laplace
objective; set `hessian=:fisher` to retain the expected-information approximation.
`β_init` (length p) and `Λ_init` (p×K) replace the default warm start when given,
as in [`fit_nb_gllvm`](@ref); `r_group` still starts at `r₀ = 10`.
"""
function fit_nb_gllvm_grouped(Y::AbstractMatrix; K::Integer, group::AbstractVector{<:Integer},
        link::Link = LogLink(), mask = nothing, offset = nothing,
        hessian::Symbol = :observed, β_init = nothing, Λ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, mask = mask,
                               caller = "fit_nb_gllvm_grouped")
    length(group) == p || throw(ArgumentError("length(group)=$(length(group)) must equal p=$p"))
    rr = rr_theta_len(p, K)
    # relabel groups to 1..G, build species→group index
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]

    msk = _resolve_obs_mask(mask, Y)
    Yc = Integer.(_sanitize_missing(Y, 0))
    Zemp = [linkfun(link, max(Yc[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)
    _mask_warmstart!(Zemp, msk)
    β0, Λ0 = _grouped_warm_start(Zemp, K, β_init, Λ_init)
    θ0 = vcat(β0, pack_lambda(Λ0), fill(log(10.0), G))

    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        rg = exp.(θ[(p + rr + 1):(p + rr + G)])
        rvec = [rg[gidx[t]] for t in 1:p]
        v = try
            -nb_grouped_marginal_loglik_laplace(Yc, Λ, β, rvec; link = link, mask = msk,
                                                offset = offset, hessian = hessian,
                                                maxiter = newton_maxiter,
                                                tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    res = _nb_boundary_restart(negll, res, ls, opts, p + rr + 1)
    θ̂, nll, conv0, iters0 = _nb_poisson_ridge_polish(negll, res, ls, opts, p + rr + 1)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    r̂g = exp.(θ̂[(p + rr + 1):(p + rr + G)])
    boundary = _dispersion_group_boundary(r̂g)   # flags BOTH ends; feeds `dispersion_boundary` and the interval code
    lowb = _dispersion_group_lower_boundary(r̂g)   # only the lower end blocks `converged` (upper end: warn only)
    any(lowb) && @warn "NB2 grouped-dispersion fit reached the per-group lower boundary (r_group below 1e-6) for group(s) $(findall(lowb)); those groups' overdispersion is extreme and not identified from this data; converged is false for this fit." maxlog=1
    any(>(1e6), r̂g) && @warn "NB2 grouped-dispersion fit has r_group above 1e6 for group(s) $(findall(>(1e6), r̂g)); the group's dispersion is at the Poisson limit (no overdispersion left to estimate). The fit's other estimates are unaffected, and converged is not affected by this." maxlog=1
    loglik, conv, iters = _fit_verdict(nll, conv0, iters0)
    return NBGroupedFit(β̂, Λ̂, r̂g, gidx, link, loglik, conv && !any(lowb), iters, hessian,
                        boundary, _stored_offset(offset))
end

"""
    NBGroupedCovFit

Result of [`fit_nb_gllvm_grouped_cov`](@ref): per-trait intercepts `β`, shared
covariate coefficients `γ` (with `γ_fixed` zero mask), loadings `Λ`, per-group
dispersion `r_group`, species→group map `group`, `link`, maximised Laplace
`loglik`, `converged`, and `iterations`. Linear predictor
`η = β + Xγ + Λz` with species dispersion `r_group[group[t]]`. `dispersion_boundary`
(length G, T14 F1, 2026-09-02) flags groups whose fitted `r_group[g]` fell outside
`[1e-6, 1e6]` (see `_dispersion_group_boundary`); only the lower end (`< 1e-6`)
forces `converged` to `false`, while `r_group[g] > 1e6` (the Poisson limit) only warns.
"""
struct NBGroupedCovFit
    β::Vector{Float64}
    γ::Vector{Float64}
    γ_fixed::Vector{Bool}
    Λ::Matrix{Float64}
    r_group::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    dispersion_boundary::Vector{Bool}   # per-group Poisson-limit / degenerate flag (T14 F1)
end

# Positional compatibility constructors (2026-08-28 hessian; T14 F1 dispersion_boundary,
# 2026-09-02): see NBGroupedFit above.
NBGroupedCovFit(β, γ, γ_fixed, Λ, r_group, group, link, loglik, converged, iterations) =
    NBGroupedCovFit(β, γ, γ_fixed, Λ, r_group, group, link, loglik, converged, iterations, :observed)
NBGroupedCovFit(β, γ, γ_fixed, Λ, r_group, group, link, loglik, converged, iterations,
                hessian::Symbol) =
    NBGroupedCovFit(β, γ, γ_fixed, Λ, r_group, group, link, loglik, converged, iterations,
                    hessian, _dispersion_group_boundary(r_group))

function Base.show(io::IO, f::NBGroupedCovFit)
    p, K = size(f.Λ); q = length(f.γ)
    print(io, "NBGroupedCovFit(p=", p, ", q=", q, ", K=", K, ", G=", length(f.r_group),
          ", r_group=", round.(f.r_group; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          any(f.dispersion_boundary) ?
              ", dispersion at boundary (groups $(findall(f.dispersion_boundary)))" : "",
          f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::NBGroupedCovFit) = fit.Λ
_loglik(fit::NBGroupedCovFit)   = fit.loglik

function _nparams(fit::NBGroupedCovFit)
    p, K = size(fit.Λ)
    return p + count(!, fit.γ_fixed) + rr_theta_len(p, K) + length(fit.r_group)
end

"""
    getLV(fit::NBGroupedCovFit, Y, X; rotate=true, mask=nothing) -> n×K matrix

Conditional latent scores at `η = β + Xγ + Λz` with per-trait NB2 dispersion.
"""
function getLV(fit::NBGroupedCovFit, Y::AbstractMatrix{<:Integer},
               X::AbstractArray{<:Real, 3};
               rotate::Bool = true, mask = nothing)
    p = size(Y, 1)
    rvec = [fit.r_group[fit.group[t]] for t in 1:p]
    fams = [NegativeBinomial(float(rvec[t]), 0.5) for t in 1:p]
    O = _build_offset(X, fit.γ)
    return _grouped_getLV(Y, fit.Λ, fit.β, fit.link, fams;
                          rotate = rotate, mask = mask, offset = O)
end

# −∇θ of `fit_nb_gllvm_grouped_cov`'s objective, θ = [β; γ_free; pack(Λ); log r_1..r_G],
# or `nothing` (finite-difference fallback) when a site's mode search fails or the AD
# pass errors. Mode: the objective's own chain (`_nb_grouped_site_mode`); log-det weight:
# `_nb_grouped_laplace_weight` at the objective's `hessian`.
function _nb_grouped_cov_negll_grad(Yc, X_fit, θ, p, q, K, rr, G, gidx, link, msk,
                                    hessian, maxiter, tol)
    try
        n = size(Yc, 2)
        N1 = ones(Int, p)
        β = θ[1:p]; γ = θ[(p + 1):(p + q)]
        Λ = unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        rg = exp.(θ[(p + q + rr + 1):(p + q + rr + G)])
        fams = [NegativeBinomial(float(rg[gidx[t]]), 0.5) for t in 1:p]
        O = _build_offset(X_fit, γ)
        ẑs = Vector{Vector{Float64}}(undef, n)
        for s in 1:n
            mi = msk === nothing ? nothing : view(msk, :, s)
            z, ok = _nb_grouped_site_mode(fams, view(Yc, :, s), N1, Λ, β, link;
                                          mask = mi, offset = view(O, :, s),
                                          maxiter = maxiter, tol = tol)
            ok || return nothing
            ẑs[s] = z
        end
        function marg(θd)
            βd = θd[1:p]; γd = θd[(p + 1):(p + q)]
            Λd = unpack_lambda(θd[(p + q + 1):(p + q + rr)], p, K)
            rgd = exp.(θd[(p + q + rr + 1):(p + q + rr + G)])
            famsd = [NegativeBinomial(rgd[gidx[t]], 0.5) for t in 1:p]
            Od = _build_offset(X_fit, γd)
            acc = zero(eltype(θd))
            for s in 1:n
                y = view(Yc, :, s)
                mi = msk === nothing ? nothing : view(msk, :, s)
                ldw = (t, μ, me, η) -> _nb_grouped_laplace_weight(hessian, famsd[t], μ, me,
                                                                  y[t], link)
                acc += _cov_site_diffable(famsd, y, N1, Λd, βd .+ view(Od, :, s), link,
                                          ẑs[s], ldw; mask = mi)
            end
            return acc
        end
        g = ForwardDiff.gradient(marg, θ)
        return all(isfinite, g) ? -g : nothing
    catch
        return nothing
    end
end

# Shared-slope least squares start: Z[t,i] ≈ β_t + Σ_k γ_k X[t,i,k]. Per-trait centring removes
# β; γ solves the pooled normal equations; returns (β, γ, residual matrix).
function _cov_ols_start(Z::AbstractMatrix, X::AbstractArray{<:Real, 3})
    p, n = size(Z); q = size(X, 3)
    zbar = vec(sum(Z; dims = 2)) ./ n
    Zc = Z .- zbar
    q == 0 && return zbar, Float64[], Zc
    Xbar = sum(X; dims = 2) ./ n
    Xc = X .- Xbar
    A = zeros(q, q); b = zeros(q)
    @inbounds for k in 1:q
        b[k] = sum(view(Xc, :, :, k) .* Zc)
        for l in k:q
            A[k, l] = A[l, k] = sum(view(Xc, :, :, k) .* view(Xc, :, :, l))
        end
    end
    γ = (Symmetric(A) + 1e-8 * I) \ b
    β = zbar .- [sum(γ[k] * Xbar[t, 1, k] for k in 1:q) for t in 1:p]
    R = copy(Zc)
    @inbounds for k in 1:q
        R .-= γ[k] .* view(Xc, :, :, k)
    end
    return β, γ, R
end


"""
    fit_nb_gllvm_grouped_cov(Y; X, K, group=1:p, link=LogLink(), mask=nothing,
                             γ_fixed=nothing, hessian=:observed, …) -> NBGroupedCovFit

Fit a negative-binomial GLLVM with **grouped / per-trait dispersion** and
**shared site covariates** `X` (`p×n×q`). Working vector
`[β; γ_free; pack(Λ); log r_1 … log r_G]`; offset `O = Xγ` is passed into the
grouped Laplace marginal. Default `hessian=:observed` matches TMB; identity
checks against shared [`fit_gllvm_cov`](@ref) should force `hessian=:fisher`.
`starts = :both` (default) fits from two starting points, the trait-mean start with `γ = 0`
and a covariate-adjusted least-squares start, and keeps the higher log-likelihood (ties keep
the first; differences under 1e-6 are ties); `starts = :default` runs only the first, at about half the cost. The NB2
covariate objective has several local optima and neither start wins everywhere.
Groups that end at the Poisson boundary get the same restart as
[`fit_nb_gllvm_grouped`](@ref) (together and each on its own, kept only if better).
Public / bridge default under X for NB2 (twin API B). Keep `fit_gllvm_cov` for
the shared-`r` + X opt-in.
"""
function fit_nb_gllvm_grouped_cov(Y::AbstractMatrix; X::AbstractArray{<:Real, 3},
        K::Integer, group::AbstractVector{<:Integer} = collect(1:size(Y, 1)),
        link::Link = LogLink(), mask = nothing, γ_fixed = nothing,
        hessian::Symbol = :observed, starts::Symbol = :both,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    starts in (:both, :default) ||
        throw(ArgumentError("starts must be :both or :default; got :$starts"))
    p, n = size(Y)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q) = ($p, $n, q); got $(size(X))"))
    length(group) == p || throw(ArgumentError("length(group)=$(length(group)) must equal p=$p"))
    q_full = size(X, 3)
    γ_fixed_mask = _fixed_zero_mask(γ_fixed, q_full, "γ_fixed")
    X_fit, _ = _slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    rr = rr_theta_len(p, K)
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]

    msk = _resolve_obs_mask(mask, Y)
    Yc = Integer.(_sanitize_missing(Y, 0))
    Zemp = [linkfun(link, max(Yc[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    _mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(β0, zeros(q), pack_lambda(Λ0), fill(log(10.0), G))

    function negll(θ)
        β = θ[1:p]
        γ = θ[(p + 1):(p + q)]
        Λ = unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        rg = exp.(θ[(p + q + rr + 1):(p + q + rr + G)])
        rvec = [rg[gidx[t]] for t in 1:p]
        O = _build_offset(X_fit, γ)
        v = try
            -nb_grouped_marginal_loglik_laplace(Yc, Λ, β, rvec; link = link, mask = msk,
                                                offset = O, hessian = hessian,
                                                maxiter = newton_maxiter,
                                                tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    # Exact gradient (implicit one-Newton-step + ForwardDiff, `_cov_site_diffable` in
    # covariates.jl) under LogLink, the only link whose observed log-det weight the
    # objective defines; other links keep the finite-difference gradient.
    grad = link isa LogLink ?
        (θ -> _nb_grouped_cov_negll_grad(Yc, X_fit, θ, p, q, K, rr, G, gidx, link, msk,
                                         hessian, newton_maxiter, newton_tol)) : nothing
    # Dense BFGS on the exact-gradient route (see `_COV_BFGS` in covariates.jl for why).
    ls_ad = grad === nothing ? ls : _COV_BFGS()
    function _fit_from(θstart)
        r = grad === nothing ? Optim.optimize(negll, θstart, ls, opts; autodiff = :finite) :
                               _optimize_with_analytic(negll, grad, θstart, ls_ad, opts)
        r = _nb_boundary_restart(negll, r, ls_ad, opts, p + q + rr + 1; grad = grad)
        return _nb_poisson_ridge_polish(negll, r, ls_ad, opts, p + q + rr + 1; grad = grad)
    end
    θ̂, nll, conv0, iters0 = _fit_from(θ0)
    if starts === :both && q > 0
        # The objective is multimodal and the start picks the basin. Second start: shared-slope
        # least squares for β, γ and loadings from the SVD of what is left (gllvmTMB's start).
        # Keep the better fit; differences under 1e-6 count as ties and keep the default-start fit.
        β1, γ1, Zc1 = _cov_ols_start(Zemp, X_fit)
        F1 = svd(Zc1); kk1 = min(K, length(F1.S))
        Λ1 = zeros(p, K)
        @inbounds for j in 1:kk1
            Λ1[:, j] = F1.U[:, j] .* (F1.S[j] / sqrt(n))
        end
        θ1 = vcat(β1, γ1, pack_lambda(Λ1), fill(log(10.0), G))
        θ̂2, nll2, conv2, iters2 = _fit_from(θ1)
        if isfinite(nll2) && nll2 < nll - 1e-6
            θ̂, nll, conv0, iters0 = θ̂2, nll2, conv2, iters2
        end
    end
    β̂ = θ̂[1:p]
    γ̂_free = θ̂[(p + 1):(p + q)]
    γ̂ = collect(Float64, _expand_fixed_zero(γ̂_free, γ_fixed_mask))
    Λ̂ = unpack_lambda(θ̂[(p + q + 1):(p + q + rr)], p, K)
    r̂g = exp.(θ̂[(p + q + rr + 1):(p + q + rr + G)])
    boundary = _dispersion_group_boundary(r̂g)   # flags BOTH ends; feeds `dispersion_boundary` and the interval code
    lowb = _dispersion_group_lower_boundary(r̂g)   # only the lower end blocks `converged` (upper end: warn only)
    any(lowb) && @warn "NB2 grouped-cov fit reached the per-group lower boundary (r_group below 1e-6) for group(s) $(findall(lowb)); those groups' overdispersion is extreme and not identified from this data; converged is false for this fit." maxlog=1
    any(>(1e6), r̂g) && @warn "NB2 grouped-cov fit has r_group above 1e6 for group(s) $(findall(>(1e6), r̂g)); the group's dispersion is at the Poisson limit (no overdispersion left to estimate). The fit's other estimates are unaffected, and converged is not affected by this." maxlog=1
    loglik, conv, iters = _fit_verdict(nll, conv0, iters0)
    return NBGroupedCovFit(β̂, γ̂, collect(Bool, γ_fixed_mask), Λ̂, r̂g, gidx, link,
                           loglik, conv && !any(lowb), iters, hessian, boundary)
end

# ===========================================================================
# Beta family — grouped / species-specific precision φ (gllvm's disp.group with
# disp.formula = NULL). Each species t carries its own precision φ_{g(t)}, so the
# Var = μ(1−μ)/(1+φ) overdispersion can vary across species (or groups). With
# G = 1 this reduces EXACTLY to the shared-precision Beta fit. The precision φ is
# carried in the family marker `Beta(φ, ·)` — only its `α` field is read as φ.
# This mirrors the NB grouped path above; the shared Beta hot path (beta.jl) is
# left untouched.
# ===========================================================================

# Exact negative conditional curvature for Beta/logit. TMB's Laplace objective
# uses this observed Hessian rather than the expected Fisher information.
function _beta_grouped_laplace_weight(hessian::Symbol, f::Beta, μ, me, y, link::Link, η)
    hessian === :fisher && return _glm_weight(f, μ, 1, me)
    hessian === :observed || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    link isa LogitLink || throw(ArgumentError(
        "hessian=:observed is currently supported only for Beta with LogitLink()"))
    φ = f.α
    ystar = log(y) - log1p(-y)
    μstar = digamma(μ * φ) - digamma((1 - μ) * φ)
    ν = trigamma(μ * φ) + trigamma((1 - μ) * φ)
    μeta2 = me * (1 - 2μ)
    return φ^2 * ν * me^2 - φ * (ystar - μstar) * μeta2
end

# Damped per-site mode search for the Beta grouped kernel (#503 class). Before this, the
# kernel ran undamped Fisher scoring and scored whatever `z` it held at `maxiter`,
# converged or not. Where the observed curvature exceeds about twice the Fisher
# curvature (responses near 0 or 1), the step overshoots into a 2-cycle: on one measured
# site Fisher alternated between z = −0.318 and −0.983 around a mode at −0.555, and the
# site's Laplace value was off by up to 15 log-likelihood units.
#
# Same rule as `_gamma_grouped_mode` (#479): a step that lowers the per-site
# log-posterior is halved (small steps and accepted full steps are bit-identical to the
# old loop), and `converged` is true only when a full proposed step is below `tol`.
# `step_weight` is `:fisher` or `:dominant`. The observed Beta weight can be negative,
# so the NB2/Gamma observed-Newton fallback is not safe here; `:dominant` uses
# max(observed, Fisher) per cell instead, which is positive (so `Λ'WΛ + I` is SPD) and
# never understates the curvature that caused the overshoot. W only sets the step: the
# mode is the fixed point of `Λ's − z = 0` whatever W is.
function _beta_grouped_mode(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link, step_weight::Symbol;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    z = zeros(K)
    for _ in 1:maxiter
        η  = _clamp_eta.(β .+ off .+ Λ * z)
        μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
        me = mu_eta.(Ref(link), η)
        s  = _glm_score.(fams, μ, n, me, y)
        W  = _beta_grouped_laplace_weight.(Ref(:fisher), fams, μ, me, y, Ref(link), η)
        if step_weight === :dominant
            Wo = _beta_grouped_laplace_weight.(Ref(:observed), fams, μ, me, y, Ref(link), η)
            W  = max.(W, Wo)
        end
        if mask !== nothing
            s = ifelse.(mask, s, 0.0)
            W = ifelse.(mask, W, 0.0)
        end
        A  = Symmetric(Λ' * (W .* Λ) + I)
        Δ  = _safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && return z, false
        maximum(abs, Δ) < tol && return z .+ Δ, true
        if norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        else
            q0 = _grouped_laplace_mode_logpost(fams, y, n, Λ, β, link, z;
                                               mask = mask, offset = offset)
            if isfinite(q0)
                accepted = false
                step = 1.0
                for _half in 1:30
                    ztrial = z .+ step .* Δ
                    q1 = _grouped_laplace_mode_logpost(fams, y, n, Λ, β, link, ztrial;
                                                       mask = mask, offset = offset)
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

# The Beta site-mode chain shared by the likelihood and getLV. Returns (z, converged).
function _beta_grouped_site_mode(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    z, ok = _beta_grouped_mode(fams, y, n, Λ, β, link, :fisher;
                               mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    # Fallback 1: max(observed, Fisher) weight, defined for LogitLink only.
    if !ok && link isa LogitLink
        z, ok = _beta_grouped_mode(fams, y, n, Λ, β, link, :dominant;
                                   mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    end
    # Fallback 2 (mirrors #507): a genuinely slow site gets a 20x Fisher budget.
    ok || ((z, ok) = _beta_grouped_mode(fams, y, n, Λ, β, link, :fisher;
                                        mask = mask, offset = offset,
                                        maxiter = 20 * maxiter, tol = tol))
    return z, ok
end

# Per-site Laplace log-marginal with per-species Beta precision markers `fams`.
function _beta_grouped_loglik_site(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, hessian::Symbol = :observed,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    # Role separation (2026-08-25). The MODE SEARCH is Fisher-scored first; the
    # caller's `hessian` governs only the post-loop log-det below. The observed
    # weight CAN be negative (measured: Beta at φ=12, η=−1.2, y=0.87 gives −1.218),
    # which is why the step never uses it alone.
    z, ok = _beta_grouped_site_mode(fams, y, n, Λ, β, link;
                                    mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    # A search that did not converge must not produce a finite value. -Inf makes the
    # fitters' objective return its 1e12 sentinel instead of a garbage surface.
    ok || return -Inf
    η  = _clamp_eta.(β .+ off .+ Λ * z)
    μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
    me = mu_eta.(Ref(link), η)
    W  = _beta_grouped_laplace_weight.(Ref(hessian), fams, μ, me, y, Ref(link), η)
    if mask !== nothing
        W = ifelse.(mask, W, 0.0)
    end
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = 0.0
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _glm_logpdf(fams[t], μ[t], n[t], y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

"""
    beta_grouped_marginal_loglik_laplace(Y, Λ, β, φvec; link=LogitLink(), mask=nothing,
                                         offset=nothing, hessian=:observed, kwargs...) -> Float64

Total Laplace log-marginal of a Beta GLLVM with **per-species** precision `φvec`
(length p; `Var_t = μ_t(1−μ_t)/(1+φvec[t])`). `Y` is the p×n matrix of proportions
in (0,1); `Λ` p×K; `β` length-p. With a constant `φvec = fill(φ, p)` this equals the
shared-precision [`beta_marginal_loglik_laplace`](@ref) to machine precision when
`hessian=:fisher` (the default). `hessian=:observed` uses the conditional
Beta/logit Hessian used by TMB's Laplace objective.
"""
function beta_grouped_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, φvec::AbstractVector; link::Link = LogitLink(),
        mask = nothing, offset = nothing, hessian::Symbol = :observed, kwargs...)
    p = size(Λ, 1)
    length(φvec) == p || throw(ArgumentError("length(φvec)=$(length(φvec)) must equal p=$p"))
    N = ones(Int, size(Y))
    fams = [Beta(float(φvec[t]), 1.0) for t in 1:p]
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask   === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        acc += _beta_grouped_loglik_site(fams, view(Y, :, i), view(N, :, i), Λ, β, link;
                                         mask = mi, offset = oi, hessian = hessian, kwargs...)
    end
    return acc
end

"""
    BetaGroupedFit

Result of [`fit_beta_gllvm_grouped`](@ref): intercepts `β` (length p), loadings `Λ`
(p×K), the per-group precision vector `φ` (length G), the species→group map `group`
(length p), the `link`, the maximised Laplace `loglik`, `converged`, and
`iterations`. The per-species precision is `φ[group[t]]`. `dispersion_boundary`
(length G, T14 F1, 2026-09-02) flags groups whose fitted `φ[g]` fell below
`1e-6` (see `_dispersion_group_lower_boundary`), the near-Bernoulli
maximal-variance limit. A large `φ` (the near-deterministic end) is identified by
the data and is not flagged (2026-09-29); `converged` is forced `false` whenever
any group is flagged, and is
`true` only when the optimizer's gradient criterion was met (#480).
`offset` is the p×n training offset the fit was made with (`nothing` when the fit had
none); [`getLV`](@ref) uses it by default.
"""
struct BetaGroupedFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    φ::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    dispersion_boundary::Vector{Bool}   # per-group near-Bernoulli (lower-end) flag (T14 F1)
    offset::Union{Nothing,Matrix{Float64}}   # training offset (p×n); `nothing` = none (#788)
end

# Positional compatibility constructors (2026-08-28 hessian; T14 F1 dispersion_boundary,
# 2026-09-02): see NBGroupedFit above.
BetaGroupedFit(β, Λ, φ, group, link, loglik, converged, iterations) =
    BetaGroupedFit(β, Λ, φ, group, link, loglik, converged, iterations, :observed)
BetaGroupedFit(β, Λ, φ, group, link, loglik, converged, iterations, hessian::Symbol) =
    BetaGroupedFit(β, Λ, φ, group, link, loglik, converged, iterations, hessian,
                   _dispersion_group_lower_boundary(φ))
# Pre-offset compat tier (10 positional args, #788): no stored training offset.
BetaGroupedFit(β, Λ, φ, group, link, loglik, converged, iterations, hessian, dispersion_boundary) =
    BetaGroupedFit(β, Λ, φ, group, link, loglik, converged, iterations, hessian, dispersion_boundary, nothing)

function Base.show(io::IO, f::BetaGroupedFit)
    p, K = size(f.Λ)
    print(io, "BetaGroupedFit(p=", p, ", K=", K, ", G=", length(f.φ),
          ", φ=", round.(f.φ; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          any(f.dispersion_boundary) ?
              ", dispersion at boundary (groups $(findall(f.dispersion_boundary)))" : "",
          f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::BetaGroupedFit) = fit.Λ
_loglik(fit::BetaGroupedFit)   = fit.loglik

# Free params: β (p) + reduced loadings Λ + one precision per group (G).
function _nparams(fit::BetaGroupedFit)
    p, K = size(fit.Λ)
    return p + rr_theta_len(p, K) + length(fit.φ)   # β + Λ + G precisions φ
end

"""
    getLV(fit::BetaGroupedFit, Y; rotate=true, mask=nothing, offset=nothing) -> n×K matrix

Conditional latent-variable scores for a grouped-precision Beta fit, using the
per-trait precision `φ[group[t]]` in the same Laplace mode equations as the
grouped likelihood.

On a fit made with an `offset` the mode search uses it (η = β + offset + Λz): the
stored training offset (`fit.offset`) when `Y` has the training size, otherwise the
`offset` you pass (a p×n matrix, a scalar or a length-p vector). New units from an
offset fit without an `offset` are refused, as gllvmTMB refuses `newdata` that lacks
the offset variable.
"""
function getLV(fit::BetaGroupedFit, Y::AbstractMatrix{<:Real};
               rotate::Bool = true, mask = nothing, offset = nothing)
    p = size(Y, 1)
    offset = _grouped_prediction_offset(fit, Y, offset, mask, "getLV")
    φvec = [fit.φ[fit.group[t]] for t in 1:p]
    fams = [Beta(float(φvec[t]), 1.0) for t in 1:p]
    return _grouped_getLV(Y, fit.Λ, fit.β, fit.link, fams;
                          rotate = rotate, mask = mask, offset = offset)
end

# Beta grouped fits can stop on a zero-length line-search step, which Optim counts as
# "the objective did not change" (f_converged) and reports as converged, while the
# gradient is still large (#480; screen dataset d05: Optim g residual 9.06 against
# g_tol 1e-5, logLik 265.976 where a fresh start reaches 272.609). When the run stops
# without Optim's gradient criterion, restart once from the warm start with every
# log φ = 0 and once from the returned point, and keep the best run only if it lowers
# the negative log-likelihood by more than 1e-6. A run that meets the gradient
# criterion is returned as it is unless it sits on a precision plateau
# (`_beta_grouped_phi_plateau`, below). `first_log_phi` indexes the first log φ in θ; the
# log φ block runs to the end of θ.
# Scale-aware gradient test, as in `_tweedie_verdict`: the residual is judged against
# `g_tol` scaled by the objective's own size, so a caller's g_tol below the
# finite-difference noise floor does not turn a stationary point into a non-converged fit.
_beta_grouped_g_met(res, g_tol) = (gres = Optim.g_residual(res);
    isfinite(gres) && gres <= max(g_tol, g_tol * abs(Optim.minimum(res))))

# Large-precision Beta fits (φ above about 1e5, near-deterministic proportions) can never
# meet `_beta_grouped_g_met`: the intercept curvature grows like φ (measured 1.1e9 at
# φ = 9.6e7), so a stationary point to within the objective's own noise (about 1e-6)
# still shows a raw finite-difference gradient of 1 to 10. There the gradient is judged in
# standard-error units instead, |g_i| / sqrt(H_ii), against the same scale-aware
# threshold `max(g_tol, g_tol * |nll|)`, after a diagonal Newton polish that is accepted
# only when it lowers the objective. `_beta_grouped_curvature_probe` widens each
# coordinate's step (1e-6 to 0.1) until the symmetric rise clears 1e-4, well above the
# noise, so g_i and H_ii are measured rather than noise; a coordinate whose symmetric
# rise never clears it (flat, or falling on both sides) fails the test. Measured on the #620
# beta_huge fixture: 0.0153 before the polish, 2.5e-4 after two steps (threshold 0.0152).
# A caller's tiny g_tol still fails (d01, g_tol = 1e-12: 1.9e-4 against 2.7e-10).
function _beta_grouped_curvature_probe(f, θ, f0)
    d = length(θ)
    g = fill(NaN, d); H = fill(NaN, d)
    e = zeros(d)
    for i in 1:d
        h = 1e-6
        while h <= 0.1
            e[i] = h
            fp, fm = f(θ .+ e), f(θ .- e)
            e[i] = 0.0
            rise = (fp + fm) / 2 - f0
            if rise >= 1e-4
                g[i] = (fp - fm) / (2h); H[i] = 2rise / h^2
                break
            end
            h *= 10
        end
        isnan(H[i]) && return g, H, false
    end
    return g, H, true
end

function _beta_grouped_scaled_polish(f, θ, f0, thr; maxit::Integer = 10)
    for _ in 1:maxit
        g, H, ok = _beta_grouped_curvature_probe(f, θ, f0)
        ok || return θ, f0, false      # a coordinate that is flat or not convex
        maximum(abs.(g) ./ sqrt.(H)) <= thr && return θ, f0, true
        step = g ./ H
        t = 1.0; moved = false
        for _ in 1:20
            ft = f(θ .- t .* step)
            if isfinite(ft) && ft < f0
                θ = θ .- t .* step; f0 = ft; moved = true
                break
            end
            t /= 2
        end
        moved || return θ, f0, false
    end
    return θ, f0, false
end

# The Beta grouped verdict: `_fit_verdict`, then the #480 gradient test, then (only when
# that fails) the standard-error-scaled test above. Returns the point to report.
function _beta_grouped_verdict(negll, res, g_tol)
    loglik, conv, iters = _fit_verdict(res)
    θ̂ = Optim.minimizer(res)
    (!conv || _beta_grouped_g_met(res, g_tol)) && return θ̂, loglik, conv, iters
    f0 = Optim.minimum(res)
    θp, fp, ok = _beta_grouped_scaled_polish(negll, θ̂, f0, max(g_tol, g_tol * abs(f0)))
    return ok ? (θp, -fp, true, iters) : (θ̂, loglik, false, iters)
end

# A group precision far above the rest is the flat-plateau sign #480 described: as φ grows
# the Beta tends to a point mass and that group's log-φ gradient goes to zero, so L-BFGS can
# stop there and still meet the gradient test. Measured on the #480 screen dataset d05
# after the Beta kernel fix: a stationary point at logLik 269.30 with φ5 ≈ 1139 (about
# 550x the median, log-φ5 gradient exactly 0), where the restart reaches 272.61. The
# restart keeps a run only if it is better, so a false trigger costs time, not accuracy.
function _beta_grouped_phi_plateau(θ, first_log_phi::Integer)
    lφ = θ[first_log_phi:end]
    length(lφ) >= 2 || return false
    return maximum(lφ) - median(lφ) > log(100)
end

function _beta_grouped_gradient_restart(negll, res, θ_warm, ls, opts, first_log_phi::Integer)
    _beta_grouped_g_met(res, Optim.g_tol(res)) &&
        !_beta_grouped_phi_plateau(Optim.minimizer(res), first_log_phi) && return res
    θa = copy(θ_warm)
    θa[first_log_phi:end] .= 0.0
    best = res
    for θs in (θa, copy(Optim.minimizer(res)))
        trial = Optim.optimize(negll, θs, ls, opts; autodiff = :finite)
        Optim.minimum(trial) < Optim.minimum(best) - 1e-6 && (best = trial)
    end
    return best
end

"""
    fit_beta_gllvm_grouped(Y; K, group, link=LogitLink(), mask=nothing, offset=nothing,
                           hessian=:observed, β_init=nothing, Λ_init=nothing, …) -> BetaGroupedFit

Fit a Beta GLLVM with grouped / species-specific precision (gllvm's `disp.group`):
species `t` shares precision `φ[group[t]]`. `group` is a length-p vector of group
ids (relabelled to `1..G` internally; default `1:p` = per-species). L-BFGS over
`[β; vec(Λ); log φ_1 … log φ_G]`; finite-difference gradient; warm start from
empirical logit-mean intercepts + SVD loadings + a moderate per-group `φ₀`.
`converged` is `true` only when the optimizer's gradient criterion (`g_tol`) is met.
If the first run stops without it, or stops with one group's precision more than 100
times the median (a flat-plateau stationary point), the fit restarts once from the warm
start with every `φ = 1` and once from the returned point, and keeps the best run only
if its log-likelihood is higher by more than `1e-6`; `iterations` then counts the kept run
only. This improves the local search but does not guarantee the global maximum. With one
group this matches [`fit_beta_gllvm`](@ref). `hessian=:observed` (the default)
uses the exact conditional Beta/logit curvature used by TMB's Laplace objective;
set `hessian=:fisher` to retain the expected-information approximation.
`β_init` (length p) and `Λ_init` (p×K) replace the default warm start when given,
as in [`fit_nb_gllvm`](@ref); the per-group `φ` still starts at `φ₀ = 10`.
"""
function fit_beta_gllvm_grouped(Y::AbstractMatrix; K::Integer,
        group::AbstractVector{<:Integer} = collect(1:size(Y, 1)),
        link::Link = LogitLink(), mask = nothing, offset = nothing,
        hessian::Symbol = :observed, β_init = nothing, Λ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, mask = mask,
                               caller = "fit_beta_gllvm_grouped")
    length(group) == p || throw(ArgumentError("length(group)=$(length(group)) must equal p=$p"))
    rr = rr_theta_len(p, K)
    # relabel groups to 1..G, build species→group index
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]

    msk = _resolve_obs_mask(mask, Y)
    Yc  = _sanitize_missing(Y, 0.5)
    Zemp = [linkfun(link, clamp(float(Yc[t, i]), 1e-6, 1 - 1e-6)) for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)
    _mask_warmstart!(Zemp, msk)
    β0, Λ0 = _grouped_warm_start(Zemp, K, β_init, Λ_init)
    θ0 = vcat(β0, pack_lambda(Λ0), fill(log(10.0), G))

    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        φg = exp.(θ[(p + rr + 1):(p + rr + G)])
        φvec = [φg[gidx[t]] for t in 1:p]
        v = try
            -beta_grouped_marginal_loglik_laplace(Yc, Λ, β, φvec; link = link, mask = msk,
                                                  offset = offset, hessian = hessian,
                                                  maxiter = newton_maxiter,
                                                  tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    res = _beta_grouped_gradient_restart(negll, res, θ0, ls, opts, p + rr + 1)
    θ̂, loglik, conv, iters = _beta_grouped_verdict(negll, res, g_tol)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    φ̂g = exp.(θ̂[(p + rr + 1):(p + rr + G)])
    boundary = _dispersion_group_lower_boundary(φ̂g)
    any(boundary) && @warn "Beta grouped-dispersion fit reached the per-group lower boundary (φ below 1e-6) for group(s) $(findall(boundary)); those groups' precision is at the near-Bernoulli limit on this data, and optimizer convergence flags are unreliable for them." maxlog=1
    return BetaGroupedFit(β̂, Λ̂, φ̂g, gidx, link, loglik, conv && !any(boundary), iters, hessian,
                          boundary, _stored_offset(offset))
end

"""
    BetaGroupedCovFit

Result of [`fit_beta_gllvm_grouped_cov`](@ref): per-trait intercepts `β`, shared
covariate coefficients `γ` (with `γ_fixed` zero mask), loadings `Λ`, per-group
precision `φ`, species→group map `group`, `link`, maximised Laplace `loglik`,
`converged`, and `iterations`. Linear predictor `η = β + Xγ + Λz` with species
precision `φ[group[t]]`. `dispersion_boundary` (length G, T14 F1, 2026-09-02)
flags groups whose fitted `φ[g]` fell below `1e-6` (see
`_dispersion_group_lower_boundary`; a large `φ` is identified and not flagged,
2026-09-29); `converged` is forced `false` whenever any group
is flagged, and is `true` only when the optimizer's gradient criterion was met (#480).
"""
struct BetaGroupedCovFit
    β::Vector{Float64}
    γ::Vector{Float64}
    γ_fixed::Vector{Bool}
    Λ::Matrix{Float64}
    φ::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    dispersion_boundary::Vector{Bool}   # per-group near-Bernoulli (lower-end) flag (T14 F1)
end

# Positional compatibility constructors (2026-08-28 hessian; T14 F1 dispersion_boundary,
# 2026-09-02): see NBGroupedFit above.
BetaGroupedCovFit(β, γ, γ_fixed, Λ, φ, group, link, loglik, converged, iterations) =
    BetaGroupedCovFit(β, γ, γ_fixed, Λ, φ, group, link, loglik, converged, iterations, :observed)
BetaGroupedCovFit(β, γ, γ_fixed, Λ, φ, group, link, loglik, converged, iterations,
                  hessian::Symbol) =
    BetaGroupedCovFit(β, γ, γ_fixed, Λ, φ, group, link, loglik, converged, iterations,
                      hessian, _dispersion_group_lower_boundary(φ))

function Base.show(io::IO, f::BetaGroupedCovFit)
    p, K = size(f.Λ); q = length(f.γ)
    print(io, "BetaGroupedCovFit(p=", p, ", q=", q, ", K=", K, ", G=", length(f.φ),
          ", φ=", round.(f.φ; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          any(f.dispersion_boundary) ?
              ", dispersion at boundary (groups $(findall(f.dispersion_boundary)))" : "",
          f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::BetaGroupedCovFit) = fit.Λ
_loglik(fit::BetaGroupedCovFit)   = fit.loglik

function _nparams(fit::BetaGroupedCovFit)
    p, K = size(fit.Λ)
    return p + count(!, fit.γ_fixed) + rr_theta_len(p, K) + length(fit.φ)
end

"""
    getLV(fit::BetaGroupedCovFit, Y, X; rotate=true, mask=nothing) -> n×K matrix

Conditional latent scores at `η = β + Xγ + Λz` with per-trait Beta precision.
"""
function getLV(fit::BetaGroupedCovFit, Y::AbstractMatrix{<:Real},
               X::AbstractArray{<:Real, 3};
               rotate::Bool = true, mask = nothing)
    p = size(Y, 1)
    φvec = [fit.φ[fit.group[t]] for t in 1:p]
    fams = [Beta(float(φvec[t]), 1.0) for t in 1:p]
    O = _build_offset(X, fit.γ)
    return _grouped_getLV(Y, fit.Λ, fit.β, fit.link, fams;
                          rotate = rotate, mask = mask, offset = O)
end

"""
    fit_beta_gllvm_grouped_cov(Y; X, K, group=1:p, link=LogitLink(), mask=nothing,
                               γ_fixed=nothing, hessian=:observed, …) -> BetaGroupedCovFit

Fit a Beta GLLVM with **grouped / per-trait precision** and **shared site
covariates** `X` (`p×n×q`). Working vector `[β; γ_free; pack(Λ); log φ_1 … log φ_G]`;
offset `O = Xγ` is passed into the grouped Laplace marginal. Default
`hessian=:observed` matches TMB; identity checks against shared
[`fit_gllvm_cov`](@ref) should force `hessian=:fisher`. `converged` requires the
optimizer's gradient criterion, and a run that stops without it gets the same
restart as [`fit_beta_gllvm_grouped`](@ref) (warm start with `γ = 0` and every
`φ = 1`, and the returned point; kept only if better by more than `1e-6`). Public / bridge default
under X for Beta (twin API B). Keep `fit_gllvm_cov` for the shared-`φ` + X opt-in.
"""
function fit_beta_gllvm_grouped_cov(Y::AbstractMatrix; X::AbstractArray{<:Real, 3},
        K::Integer, group::AbstractVector{<:Integer} = collect(1:size(Y, 1)),
        link::Link = LogitLink(), mask = nothing, γ_fixed = nothing,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q) = ($p, $n, q); got $(size(X))"))
    length(group) == p || throw(ArgumentError("length(group)=$(length(group)) must equal p=$p"))
    q_full = size(X, 3)
    γ_fixed_mask = _fixed_zero_mask(γ_fixed, q_full, "γ_fixed")
    X_fit, _ = _slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    rr = rr_theta_len(p, K)
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]

    msk = _resolve_obs_mask(mask, Y)
    Yc = _sanitize_missing(Y, 0.5)
    Zemp = [linkfun(link, clamp(float(Yc[t, i]), 1e-6, 1 - 1e-6)) for t in 1:p, i in 1:n]
    _mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(β0, zeros(q), pack_lambda(Λ0), fill(log(10.0), G))

    function negll(θ)
        β = θ[1:p]
        γ = θ[(p + 1):(p + q)]
        Λ = unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        φg = exp.(θ[(p + q + rr + 1):(p + q + rr + G)])
        φvec = [φg[gidx[t]] for t in 1:p]
        O = _build_offset(X_fit, γ)
        v = try
            -beta_grouped_marginal_loglik_laplace(Yc, Λ, β, φvec; link = link, mask = msk,
                                                  offset = O, hessian = hessian,
                                                  maxiter = newton_maxiter,
                                                  tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    res = _beta_grouped_gradient_restart(negll, res, θ0, ls, opts, p + q + rr + 1)
    θ̂, loglik, conv, iters = _beta_grouped_verdict(negll, res, g_tol)
    β̂ = θ̂[1:p]
    γ̂_free = θ̂[(p + 1):(p + q)]
    γ̂ = collect(Float64, _expand_fixed_zero(γ̂_free, γ_fixed_mask))
    Λ̂ = unpack_lambda(θ̂[(p + q + 1):(p + q + rr)], p, K)
    φ̂g = exp.(θ̂[(p + q + rr + 1):(p + q + rr + G)])
    boundary = _dispersion_group_lower_boundary(φ̂g)
    any(boundary) && @warn "Beta grouped-cov fit reached the per-group lower boundary (φ below 1e-6) for group(s) $(findall(boundary)); those groups' precision is at the near-Bernoulli limit on this data, and optimizer convergence flags are unreliable for them." maxlog=1
    return BetaGroupedCovFit(β̂, γ̂, collect(Bool, γ_fixed_mask), Λ̂, φ̂g, gidx, link,
                             loglik, conv && !any(boundary), iters, hessian, boundary)
end

# ===========================================================================
# Gamma family — grouped / species-specific shape α (gllvm's disp.group with
# disp.formula = NULL). Each species t carries its own shape α_{g(t)}, so the
# Var = μ²/α overdispersion can vary across species (or groups). With G = 1 and
# hessian=:fisher this reduces EXACTLY to the shared-shape Gamma fit. The fit
# default hessian=:observed is the TMB Laplace curvature (different objective).
# The shape α is carried in the family marker `Gamma(α, ·)` — only its `α` field
# is read. This mirrors the NB/Beta grouped paths; the shared Gamma hot path
# (gamma.jl) is left untouched.
# ===========================================================================

# Exact negative conditional curvature for Gamma/log. TMB's Laplace objective
# uses this observed Hessian rather than the expected Fisher information:
# -∂²ℓ/∂η² = α * y / μ  (log link).
function _gamma_grouped_laplace_weight(hessian::Symbol, f::Gamma, μ, me, y, link::Link)
    hessian === :fisher && return _glm_weight(f, μ, 1, me)
    hessian === :observed || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    link isa LogLink || throw(ArgumentError(
        "hessian=:observed is currently supported only for Gamma with LogLink()"))
    return f.α * y / μ
end

# Per-site mode search for the Gamma grouped kernel (#479). Returns `(z, converged)`.
# `step_weight` picks the curvature that sets the Newton step. It never changes the
# mode, which is the fixed point of `Λ's − z = 0` whatever W is.
#
# Before #479 the kernel ran undamped Fisher scoring and returned whatever `z` it held
# when the loop stopped, converged or not. Gamma/log's Fisher weight is the constant α,
# blind to y/μ, so the step overshoots when y/μ is far from 1: on one site of the
# fitter's own warm start it cycled out to z ≈ 1e11 and the kernel returned a FINITE
# −1.4e23. That value slipped past the fitter's 1e12 sentinel and stopped L-BFGS at
# iteration 2, 43 log-likelihood units below the optimum, with converged = true.
#
# Now a step that lowers the per-site log-posterior is halved (the generic core's
# `_laplace_mode` rule, so small steps and accepted full steps are bit-identical to the
# old loop), and `converged` is true only when the full proposed step is below `tol`,
# which is the old loop's own stopping test. A heavily halved step does NOT count.
function _gamma_grouped_mode(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link, step_weight::Symbol;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    z = zeros(K)
    for _ in 1:maxiter
        η  = _clamp_eta.(β .+ off .+ Λ * z)
        μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
        me = mu_eta.(Ref(link), η)
        s  = _glm_score.(fams, μ, n, me, y)
        W  = _gamma_grouped_laplace_weight.(Ref(step_weight), fams, μ, me, y, Ref(link))
        if mask !== nothing
            s = ifelse.(mask, s, 0.0)
            W = ifelse.(mask, W, 0.0)
        end
        A  = Symmetric(Λ' * (W .* Λ) + I)
        Δ  = _safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && return z, false
        maximum(abs, Δ) < tol && return z .+ Δ, true
        if norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        else
            q0 = _grouped_laplace_mode_logpost(fams, y, n, Λ, β, link, z;
                                               mask = mask, offset = offset)
            if isfinite(q0)
                accepted = false
                step = 1.0
                for _half in 1:30
                    ztrial = z .+ step .* Δ
                    q1 = _grouped_laplace_mode_logpost(fams, y, n, Λ, β, link, ztrial;
                                                       mask = mask, offset = offset)
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

# The Gamma site-mode chain shared by the likelihood and getLV (#503 follow-up).
function _gamma_grouped_site_mode(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    z, ok = _gamma_grouped_mode(fams, y, n, Λ, β, link, :fisher;
                                mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    # Fallback (#479). Fisher scoring converges only linearly for Gamma/log and can
    # take far longer than `maxiter` even with halving (measured: 1532 iterations on the
    # #479 site at the fitter's warm start, contraction factor about 0.991). Under LogLink the observed weight α·y/μ is >= 0 for every admissible
    # response (masked cells are zeroed), so `Λ'WΛ + I` stays SPD and the step is exact
    # damped Newton on a strictly concave per-site objective (measured: at most 6
    # iterations on every site probed across the sibling screen's 10 datasets). It
    # runs only where the Fisher search failed, so every site that converged before
    # keeps its old path.
    if !ok && link isa LogLink
        z, ok = _gamma_grouped_mode(fams, y, n, Λ, β, link, :observed;
                                    mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    end
    return z, ok
end

# Per-site Laplace log-marginal with per-species Gamma shape markers `fams`.
function _gamma_grouped_loglik_site(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, hessian::Symbol = :observed,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    # Role separation (2026-08-25). The MODE SEARCH is Fisher-scored first:
    # `:fisher` here is not the caller's selector, which governs only the post-loop
    # log-det below. Expected information is >= 0, so `Λ'WΛ + I` is SPD by
    # construction. The converged mode is unchanged either way: it is the fixed point
    # of `Λ's − z = 0`, which does not involve W at all. W only sets the step.
    z, ok = _gamma_grouped_site_mode(fams, y, n, Λ, β, link;
                                     mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    # A search that did not converge must not produce a finite value. -Inf makes the
    # fitters' objective return its 1e12 sentinel instead of a garbage surface.
    ok || return -Inf
    η  = _clamp_eta.(β .+ off .+ Λ * z)
    μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
    me = mu_eta.(Ref(link), η)
    W  = _gamma_grouped_laplace_weight.(Ref(hessian), fams, μ, me, y, Ref(link))
    if mask !== nothing
        W = ifelse.(mask, W, 0.0)
    end
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = 0.0
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _glm_logpdf(fams[t], μ[t], n[t], y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

"""
    gamma_grouped_marginal_loglik_laplace(Y, Λ, β, αvec; link=LogLink(), mask=nothing,
                                          offset=nothing, hessian=:fisher, kwargs...) -> Float64

Total Laplace log-marginal of a Gamma GLLVM with **per-species** shape `αvec`
(length p; `Var_t = μ_t²/αvec[t]`). `Y` is the p×n matrix of positive reals; `Λ`
p×K; `β` length-p. With a constant `αvec = fill(α, p)` this equals the shared-shape
[`gamma_marginal_loglik_laplace`](@ref) to machine precision when
`hessian=:fisher`. `hessian=:observed` (the default) uses the conditional
Gamma/log Hessian used by TMB's Laplace objective. If any site's mode search fails
to converge to `tol` (default 1e-9), the total is `-Inf`, never a value computed at
an unconverged mode (#479). The search is Fisher scoring with step halving for up to
`maxiter` steps (default 100) and, under `LogLink`, a damped Newton search of up to
`maxiter` more.
"""
function gamma_grouped_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, αvec::AbstractVector; link::Link = LogLink(),
        mask = nothing, offset = nothing, hessian::Symbol = :observed, kwargs...)
    p = size(Λ, 1)
    length(αvec) == p || throw(ArgumentError("length(αvec)=$(length(αvec)) must equal p=$p"))
    N = ones(Int, size(Y))
    fams = [Gamma(float(αvec[t]), 1.0) for t in 1:p]
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask   === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        acc += _gamma_grouped_loglik_site(fams, view(Y, :, i), view(N, :, i), Λ, β, link;
                                         mask = mi, offset = oi, hessian = hessian, kwargs...)
    end
    return acc
end

"""
    GammaGroupedFit

Result of [`fit_gamma_gllvm_grouped`](@ref): intercepts `β` (length p), loadings `Λ`
(p×K), the per-group shape vector `α` (length G), the species→group map `group`
(length p), the `link`, the maximised Laplace `loglik`, `converged`, and
`iterations`. The per-species shape is `α[group[t]]`. `dispersion_boundary`
(length G, T14 F1, 2026-09-02) flags groups whose fitted `α[g]` fell below
`1e-6` (see `_dispersion_group_lower_boundary`), the extreme-overdispersion
limit. A large `α` (the near-deterministic end) is identified by the data and is
not flagged (2026-09-29); `converged` is forced `false` whenever any group is
flagged.
`offset` is the p×n training offset the fit was made with (`nothing` when the fit had
none); [`getLV`](@ref) uses it by default.
"""
struct GammaGroupedFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    α::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    dispersion_boundary::Vector{Bool}   # per-group extreme-overdispersion (lower-end) flag (T14 F1)
    offset::Union{Nothing,Matrix{Float64}}   # training offset (p×n); `nothing` = none (#788)
end

# Positional compatibility constructors (2026-08-28 hessian; T14 F1 dispersion_boundary,
# 2026-09-02): see NBGroupedFit above.
GammaGroupedFit(β, Λ, α, group, link, loglik, converged, iterations) =
    GammaGroupedFit(β, Λ, α, group, link, loglik, converged, iterations, :observed)
GammaGroupedFit(β, Λ, α, group, link, loglik, converged, iterations, hessian::Symbol) =
    GammaGroupedFit(β, Λ, α, group, link, loglik, converged, iterations, hessian,
                    _dispersion_group_lower_boundary(α))
# Pre-offset compat tier (10 positional args, #788): no stored training offset.
GammaGroupedFit(β, Λ, α, group, link, loglik, converged, iterations, hessian, dispersion_boundary) =
    GammaGroupedFit(β, Λ, α, group, link, loglik, converged, iterations, hessian, dispersion_boundary, nothing)

function Base.show(io::IO, f::GammaGroupedFit)
    p, K = size(f.Λ)
    print(io, "GammaGroupedFit(p=", p, ", K=", K, ", G=", length(f.α),
          ", α=", round.(f.α; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          any(f.dispersion_boundary) ?
              ", dispersion at boundary (groups $(findall(f.dispersion_boundary)))" : "",
          f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::GammaGroupedFit) = fit.Λ
_loglik(fit::GammaGroupedFit)   = fit.loglik

# Free params: β (p) + reduced loadings Λ + one shape per group (G).
function _nparams(fit::GammaGroupedFit)
    p, K = size(fit.Λ)
    return p + rr_theta_len(p, K) + length(fit.α)   # β + Λ + G shapes α
end

"""
    getLV(fit::GammaGroupedFit, Y; rotate=true, mask=nothing, offset=nothing) -> n×K matrix

Conditional latent-variable scores for a grouped-shape Gamma fit, using the
per-trait shape `α[group[t]]` in the same Laplace mode equations as the grouped
likelihood.

On a fit made with an `offset` the mode search uses it (η = β + offset + Λz): the
stored training offset (`fit.offset`) when `Y` has the training size, otherwise the
`offset` you pass (a p×n matrix, a scalar or a length-p vector). New units from an
offset fit without an `offset` are refused, as gllvmTMB refuses `newdata` that lacks
the offset variable.
"""
function getLV(fit::GammaGroupedFit, Y::AbstractMatrix{<:Real};
               rotate::Bool = true, mask = nothing, offset = nothing)
    p = size(Y, 1)
    offset = _grouped_prediction_offset(fit, Y, offset, mask, "getLV")
    αvec = [fit.α[fit.group[t]] for t in 1:p]
    fams = [Gamma(float(αvec[t]), 1.0) for t in 1:p]
    return _grouped_getLV(Y, fit.Λ, fit.β, fit.link, fams;
                          rotate = rotate, mask = mask, offset = offset)
end

"""
    fit_gamma_gllvm_grouped(Y; K, group, link=LogLink(), mask=nothing, offset=nothing,
                            hessian=:observed, …) -> GammaGroupedFit

Fit a Gamma GLLVM with grouped / species-specific shape (gllvm's `disp.group`):
species `t` shares shape `α[group[t]]`. `group` is a length-p vector of group ids
(relabelled to `1..G` internally; default `1:p` = per-species). L-BFGS over
`[β; vec(Λ); log α_1 … log α_G]`; finite-difference gradient; warm start from log
row-means as intercepts + SVD of row-centred log-Y as loadings + a moderate per-group
`α₀`. With one group and `hessian=:fisher` this matches [`fit_gamma_gllvm`](@ref).
`hessian=:observed` (the default) is the TMB Laplace curvature — a different
objective; set `hessian=:fisher` to retain the expected-information approximation.
"""
function fit_gamma_gllvm_grouped(Y::AbstractMatrix; K::Integer,
        group::AbstractVector{<:Integer} = collect(1:size(Y, 1)),
        link::Link = LogLink(), mask = nothing, offset = nothing,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, mask = mask,
                               caller = "fit_gamma_gllvm_grouped")
    length(group) == p || throw(ArgumentError("length(group)=$(length(group)) must equal p=$p"))
    rr = rr_theta_len(p, K)
    # relabel groups to 1..G, build species→group index
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]

    msk = _resolve_obs_mask(mask, Y)
    Yc  = _sanitize_missing(Y, 1.0)
    Zemp = log.(max.(Yc, 1e-6))
    offset === nothing || (Zemp .-= offset)
    _mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(β0, pack_lambda(Λ0), fill(log(2.0), G))

    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        αg = exp.(θ[(p + rr + 1):(p + rr + G)])
        αvec = [αg[gidx[t]] for t in 1:p]
        v = try
            -gamma_grouped_marginal_loglik_laplace(Yc, Λ, β, αvec; link = link, mask = msk,
                                                   offset = offset, hessian = hessian,
                                                   maxiter = newton_maxiter,
                                                   tol = newton_tol)
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
    α̂g = exp.(θ̂[(p + rr + 1):(p + rr + G)])
    boundary = _dispersion_group_lower_boundary(α̂g)
    any(boundary) && @warn "Gamma grouped-dispersion fit reached the per-group lower boundary (α below 1e-6) for group(s) $(findall(boundary)); those groups' shape is at the extreme-overdispersion limit on this data, and optimizer convergence flags are unreliable for them." maxlog=1
    loglik, conv, iters = _fit_verdict(res)
    return GammaGroupedFit(β̂, Λ̂, α̂g, gidx, link, loglik, conv && !any(boundary), iters, hessian,
                           boundary, _stored_offset(offset))
end

"""
    GammaGroupedCovFit

Result of [`fit_gamma_gllvm_grouped_cov`](@ref): per-trait intercepts `β`, shared
covariate coefficients `γ` (with `γ_fixed` zero mask), loadings `Λ`, per-group
shape `α`, species→group map `group`, `link`, maximised Laplace `loglik`,
`converged`, and `iterations`. Linear predictor `η = β + Xγ + Λz` with species
shape `α[group[t]]` (`Var = μ²/α`). `dispersion_boundary` (length G, T14 F1,
2026-09-02) flags groups whose fitted `α[g]` fell below `1e-6` (see
`_dispersion_group_lower_boundary`; a large `α` is identified and not flagged,
2026-09-29); `converged` is forced `false` whenever any group is flagged.
"""
struct GammaGroupedCovFit
    β::Vector{Float64}
    γ::Vector{Float64}
    γ_fixed::Vector{Bool}
    Λ::Matrix{Float64}
    α::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    dispersion_boundary::Vector{Bool}   # per-group extreme-overdispersion (lower-end) flag (T14 F1)
end

# Positional compatibility constructors (2026-08-28 hessian; T14 F1 dispersion_boundary,
# 2026-09-02): see NBGroupedFit above.
GammaGroupedCovFit(β, γ, γ_fixed, Λ, α, group, link, loglik, converged, iterations) =
    GammaGroupedCovFit(β, γ, γ_fixed, Λ, α, group, link, loglik, converged, iterations, :observed)
GammaGroupedCovFit(β, γ, γ_fixed, Λ, α, group, link, loglik, converged, iterations,
                   hessian::Symbol) =
    GammaGroupedCovFit(β, γ, γ_fixed, Λ, α, group, link, loglik, converged, iterations,
                       hessian, _dispersion_group_lower_boundary(α))

function Base.show(io::IO, f::GammaGroupedCovFit)
    p, K = size(f.Λ); q = length(f.γ)
    print(io, "GammaGroupedCovFit(p=", p, ", q=", q, ", K=", K, ", G=", length(f.α),
          ", α=", round.(f.α; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          any(f.dispersion_boundary) ?
              ", dispersion at boundary (groups $(findall(f.dispersion_boundary)))" : "",
          f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::GammaGroupedCovFit) = fit.Λ
_loglik(fit::GammaGroupedCovFit)   = fit.loglik

function _nparams(fit::GammaGroupedCovFit)
    p, K = size(fit.Λ)
    return p + count(!, fit.γ_fixed) + rr_theta_len(p, K) + length(fit.α)
end

"""
    getLV(fit::GammaGroupedCovFit, Y, X; rotate=true, mask=nothing) -> n×K matrix

Conditional latent scores at `η = β + Xγ + Λz` with per-trait Gamma shape.
"""
function getLV(fit::GammaGroupedCovFit, Y::AbstractMatrix{<:Real},
               X::AbstractArray{<:Real, 3};
               rotate::Bool = true, mask = nothing)
    p = size(Y, 1)
    αvec = [fit.α[fit.group[t]] for t in 1:p]
    fams = [Gamma(float(αvec[t]), 1.0) for t in 1:p]
    O = _build_offset(X, fit.γ)
    return _grouped_getLV(Y, fit.Λ, fit.β, fit.link, fams;
                          rotate = rotate, mask = mask, offset = O)
end

"""
    fit_gamma_gllvm_grouped_cov(Y; X, K, group=1:p, link=LogLink(), mask=nothing,
                                γ_fixed=nothing, hessian=:observed, …) -> GammaGroupedCovFit

Fit a Gamma GLLVM with **grouped / per-trait shape** and **shared site
covariates** `X` (`p×n×q`). Working vector `[β; γ_free; pack(Λ); log α_1 … log α_G]`;
offset `O = Xγ` is passed into the grouped Laplace marginal. Default
`hessian=:observed` matches TMB; identity checks against shared
[`fit_gllvm_cov`](@ref) should force `hessian=:fisher`. Public / bridge default
under X for Gamma (twin API B; decision 2026-08-03). Keep [`fit_gllvm_cov`](@ref)
for the shared-`α` + X opt-in. Identity checks against shared cov should use
`group = ones(Int, p)` (G=1).
"""
function fit_gamma_gllvm_grouped_cov(Y::AbstractMatrix; X::AbstractArray{<:Real, 3},
        K::Integer, group::AbstractVector{<:Integer} = collect(1:size(Y, 1)),
        link::Link = LogLink(), mask = nothing, γ_fixed = nothing,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q) = ($p, $n, q); got $(size(X))"))
    length(group) == p || throw(ArgumentError("length(group)=$(length(group)) must equal p=$p"))
    q_full = size(X, 3)
    γ_fixed_mask = _fixed_zero_mask(γ_fixed, q_full, "γ_fixed")
    X_fit, _ = _slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    rr = rr_theta_len(p, K)
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]

    msk = _resolve_obs_mask(mask, Y)
    Yc = _sanitize_missing(Y, 1.0)
    Zemp = log.(max.(Yc, 1e-6))
    _mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(β0, zeros(q), pack_lambda(Λ0), fill(log(2.0), G))

    function negll(θ)
        β = θ[1:p]
        γ = θ[(p + 1):(p + q)]
        Λ = unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        αg = exp.(θ[(p + q + rr + 1):(p + q + rr + G)])
        αvec = [αg[gidx[t]] for t in 1:p]
        O = _build_offset(X_fit, γ)
        v = try
            -gamma_grouped_marginal_loglik_laplace(Yc, Λ, β, αvec; link = link, mask = msk,
                                                   offset = O, hessian = hessian,
                                                   maxiter = newton_maxiter,
                                                   tol = newton_tol)
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
    α̂g = exp.(θ̂[(p + q + rr + 1):(p + q + rr + G)])
    boundary = _dispersion_group_lower_boundary(α̂g)
    any(boundary) && @warn "Gamma grouped-cov fit reached the per-group lower boundary (α below 1e-6) for group(s) $(findall(boundary)); those groups' shape is at the extreme-overdispersion limit on this data, and optimizer convergence flags are unreliable for them." maxlog=1
    loglik, conv, iters = _fit_verdict(res)
    return GammaGroupedCovFit(β̂, γ̂, collect(Bool, γ_fixed_mask), Λ̂, α̂g, gidx, link,
                              loglik, conv && !any(boundary), iters, hessian, boundary)
end

# ===========================================================================
# NB1 family — grouped / species-specific dispersion φ (gllvm's disp.group with
# disp.formula = NULL, `family = negative.binomial1`). Each species t carries its
# own LINEAR-variance dispersion φ_{g(t)}, so the overdispersion Var = μ(1+φ) can
# vary across species (or groups). With G = 1 and hessian=:fisher this reduces
# EXACTLY to the shared-dispersion NB1 fit. The fit/cov default hessian=:observed
# is the TMB Laplace curvature (different objective). Dispersion is carried in
# the family marker `NB1(φ)`. The shared NB1 hot path (negbin1.jl) is left
# untouched for Fisher-only callers.
# ===========================================================================

# Exact negative conditional curvature for NB1/log. With r = μ/φ:
# ∂ℓ/∂μ = (1/φ)[ψ(y+r) − ψ(r) − log(1+φ)],  ∂²ℓ/∂μ² = (1/φ²)[ψ'(y+r) − ψ'(r)].
# Under η = log μ: W = −∂²ℓ/∂η² = −μ·(∂ℓ/∂μ) − μ²·(∂²ℓ/∂μ²).
function _nb1_grouped_laplace_weight(hessian::Symbol, f::NB1, μ, me, y, link::Link)
    hessian === :fisher && return _glm_weight(f, μ, 1, me)
    hessian === :observed || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    link isa LogLink || throw(ArgumentError(
        "hessian=:observed is currently supported only for NB1 with LogLink()"))
    φ = float(f.φ)
    r = μ / φ
    d1, d2 = _nb1_rise_diffs(r, Int(y))   # stable toward φ → 0 (negbin1.jl)
    s_μ = (d1 - log1p(φ)) / φ
    return -μ * s_μ - (μ / φ)^2 * d2
end

# Per-site mode search for the NB1 grouped kernel (#503, the #479 pattern). Returns
# `(z, converged)`. `step_weight` picks the curvature that sets the Newton step; it
# never changes the mode, which is the fixed point of `Λ's − z = 0` whatever W is.
#
# Before this fix the kernel ran undamped Fisher scoring (the loop below, unmodified
# apart from the halving) and returned whatever `z` it held when the loop stopped,
# converged or not — the same defect #479 fixed for Gamma. The audit's stress probe
# (Λ scaled up to 3x, warm start perturbed ±50%) measured a 3.6% rate (108/3000, later
# re-measured 20/600) of sites where the returned z had |grad log-posterior| > 1e-4
# while a from-scratch restart converged cleanly, yet the kernel still returned a
# finite site log-likelihood.
#
# Now a step that lowers the per-site log-posterior is halved (the generic core's
# `_laplace_mode` rule, so small steps and accepted full steps are bit-identical to the
# old loop), and `converged` is true only when the full proposed step is below `tol`,
# which is the old loop's own stopping test. A heavily halved step does NOT count.
function _nb1_grouped_mode(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link, step_weight::Symbol;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    z = zeros(K)
    for _ in 1:maxiter
        η  = _clamp_eta.(β .+ off .+ Λ * z)
        μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
        me = mu_eta.(Ref(link), η)
        s  = _glm_score.(fams, μ, n, me, y)
        W  = _nb1_grouped_laplace_weight.(Ref(step_weight), fams, μ, me, y, Ref(link))
        if mask !== nothing
            s = ifelse.(mask, s, 0.0)
            W = ifelse.(mask, W, 0.0)
        end
        A  = Symmetric(Λ' * (W .* Λ) + I)
        Δ  = _safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && return z, false
        maximum(abs, Δ) < tol && return z .+ Δ, true
        if norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        else
            q0 = _grouped_laplace_mode_logpost(fams, y, n, Λ, β, link, z;
                                               mask = mask, offset = offset)
            if isfinite(q0)
                accepted = false
                step = 1.0
                for _half in 1:30
                    ztrial = z .+ step .* Δ
                    q1 = _grouped_laplace_mode_logpost(fams, y, n, Λ, β, link, ztrial;
                                                       mask = mask, offset = offset)
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

# The NB1 site-mode chain shared by the likelihood and getLV (#503 follow-up).
function _nb1_grouped_site_mode(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    z, ok = _nb1_grouped_mode(fams, y, n, Λ, β, link, :fisher;
                              mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    # Larger-budget retry (review of #507, mirrors #509's Student-t fallback). A
    # genuinely converging site can still need more than the default
    # `maxiter = 100` Fisher-scored steps under ill-conditioned curvature
    # (measured: reviewer's stress probe found 2/200 stationary, negative-definite
    # sites failing at `maxiter = 100` that converge cleanly, to the same `z`, at
    # `maxiter = 2000`). Retry Fisher scoring with a 20x iteration budget, still
    # restarting from z = 0, before falling back to the `:observed` direction below —
    # it runs only where the default budget failed, so every site that converged
    # within `maxiter` keeps its original path and iteration count.
    ok || ((z, ok) = _nb1_grouped_mode(fams, y, n, Λ, β, link, :fisher;
                                       mask = mask, offset = offset,
                                       maxiter = 20 * maxiter, tol = tol))
    # Fallback (#503, the #479 pattern). If Fisher scoring does not converge, retry
    # with the observed weight, but only under LogLink — the only link this kernel's
    # :observed weight supports (see `_nb1_grouped_laplace_weight`). Unlike Gamma's
    # fallback, NB1's observed curvature is not established to be sign-definite in
    # general, so this is offered only as a second candidate DIRECTION: the
    # step-halving accept test inside `_nb1_grouped_mode` still requires the trial
    # point to raise the per-site log-posterior over its own q0, so an indefinite or
    # wrong-signed step is rejected (falls through to non-convergence), never accepted.
    if !ok && link isa LogLink
        z, ok = _nb1_grouped_mode(fams, y, n, Λ, β, link, :observed;
                                  mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    end
    return z, ok
end

# Per-site Laplace log-marginal with per-species NB1 dispersion markers `fams`.
function _nb1_grouped_loglik_site(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, hessian::Symbol = :observed,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    # Role separation (2026-08-25). The MODE SEARCH is Fisher-scored first:
    # `:fisher` here is not the caller's selector, which governs only the post-loop
    # log-det below. Expected information is >= 0, so `Λ'WΛ + I` is SPD by
    # construction. The converged mode is unchanged either way: it is the fixed point
    # of `Λ's − z = 0`, which does not involve W at all. W only sets the step.
    z, ok = _nb1_grouped_site_mode(fams, y, n, Λ, β, link;
                                   mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    # A search that did not converge must not produce a finite value. -Inf makes the
    # fitters' objective return its 1e12 sentinel instead of a garbage surface (#503,
    # the #479 pattern).
    ok || return -Inf
    η  = _clamp_eta.(β .+ off .+ Λ * z)
    μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
    me = mu_eta.(Ref(link), η)
    W  = _nb1_grouped_laplace_weight.(Ref(hessian), fams, μ, me, y, Ref(link))
    if mask !== nothing
        W = ifelse.(mask, W, 0.0)
    end
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = 0.0
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _glm_logpdf(fams[t], μ[t], n[t], y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

"""
    nb1_grouped_marginal_loglik_laplace(Y, Λ, β, φvec; link=LogLink(), mask=nothing,
                                        offset=nothing, hessian=:observed, kwargs...) -> Float64

Total Laplace log-marginal of a negative-binomial type-1 (NB1) GLLVM with
**per-species** dispersion `φvec` (length p; linear variance `Var_t = μ_t(1+φvec[t])`).
`Y` is the p×n integer count matrix; `Λ` p×K; `β` length-p. With a constant
`φvec = fill(φ, p)` and `hessian=:fisher` this equals the shared-dispersion
[`nb1_marginal_loglik_laplace`](@ref) to machine precision. `hessian=:observed`
uses the conditional observed curvature (TMB Laplace).
"""
function nb1_grouped_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, φvec::AbstractVector; link::Link = LogLink(),
        mask = nothing, offset = nothing, hessian::Symbol = :observed, kwargs...)
    p = size(Λ, 1)
    length(φvec) == p || throw(ArgumentError("length(φvec)=$(length(φvec)) must equal p=$p"))
    N = ones(Int, size(Y))
    fams = [NB1(float(φvec[t])) for t in 1:p]
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask   === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        acc += _nb1_grouped_loglik_site(fams, view(Y, :, i), view(N, :, i), Λ, β, link;
                                        mask = mi, offset = oi, hessian = hessian, kwargs...)
    end
    return acc
end

"""
    NB1GroupedFit

Result of [`fit_nb1_gllvm_grouped`](@ref): intercepts `β` (length p), loadings `Λ`
(p×K), the per-group dispersion vector `φ` (length G), the species→group map `group`
(length p), the `link`, the maximised Laplace `loglik`, `converged`, and
`iterations`. The per-species dispersion is `φ[group[t]]` (linear variance
`Var_t = μ_t(1+φ[group[t]])`). `dispersion_boundary` (length G, T14 F1,
2026-09-02) flags groups whose fitted `φ[g]` fell outside `[1e-6, 1e6]` (see
`_dispersion_group_boundary`) — the Poisson limit at the lower end, numerically
flat overdispersion at the upper end. Only the upper end (`φ[g] > 1e6`) forces
`converged` to `false`; `φ[g] < 1e-6`, the NB1 Poisson limit, only warns (maintainer
decision 2026-09-29: the Poisson limit warns only).
`offset` is the p×n training offset the fit was made with (`nothing` when the fit had
none); [`getLV`](@ref) uses it by default.
"""
struct NB1GroupedFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    φ::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    dispersion_boundary::Vector{Bool}   # per-group Poisson-limit / flat-overdispersion flag (T14 F1)
    offset::Union{Nothing,Matrix{Float64}}   # training offset (p×n); `nothing` = none (#788)
end

# Positional compatibility constructors (2026-08-28 hessian; T14 F1 dispersion_boundary,
# 2026-09-02): see NBGroupedFit above.
NB1GroupedFit(β, Λ, φ, group, link, loglik, converged, iterations) =
    NB1GroupedFit(β, Λ, φ, group, link, loglik, converged, iterations, :observed)
NB1GroupedFit(β, Λ, φ, group, link, loglik, converged, iterations, hessian::Symbol) =
    NB1GroupedFit(β, Λ, φ, group, link, loglik, converged, iterations, hessian,
                 _dispersion_group_boundary(φ))
# Pre-offset compat tier (10 positional args, #788): no stored training offset.
NB1GroupedFit(β, Λ, φ, group, link, loglik, converged, iterations, hessian, dispersion_boundary) =
    NB1GroupedFit(β, Λ, φ, group, link, loglik, converged, iterations, hessian, dispersion_boundary, nothing)

function Base.show(io::IO, f::NB1GroupedFit)
    p, K = size(f.Λ)
    print(io, "NB1GroupedFit(p=", p, ", K=", K, ", G=", length(f.φ),
          ", φ=", round.(f.φ; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          any(f.dispersion_boundary) ?
              ", dispersion at boundary (groups $(findall(f.dispersion_boundary)))" : "",
          f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::NB1GroupedFit) = fit.Λ
_loglik(fit::NB1GroupedFit)   = fit.loglik

# Free params: β (p) + reduced loadings Λ + one dispersion per group (G).
function _nparams(fit::NB1GroupedFit)
    p, K = size(fit.Λ)
    return p + rr_theta_len(p, K) + length(fit.φ)   # β + Λ + G dispersions φ
end

"""
    getLV(fit::NB1GroupedFit, Y; N=nothing, rotate=true, mask=nothing, offset=nothing) -> n×K matrix

Conditional latent-variable scores for a grouped-dispersion NB1 fit, using the
per-trait linear-variance dispersion `φ[group[t]]` in the same Laplace mode
equations as the grouped likelihood.

On a fit made with an `offset` the mode search uses it (η = β + offset + Λz): the
stored training offset (`fit.offset`) when `Y` has the training size, otherwise the
`offset` you pass (a p×n matrix, a scalar or a length-p vector). New units from an
offset fit without an `offset` are refused, as gllvmTMB refuses `newdata` that lacks
the offset variable.
"""
function getLV(fit::NB1GroupedFit, Y::AbstractMatrix{<:Integer};
               N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
               rotate::Bool = true, mask = nothing, offset = nothing)
    p = size(Y, 1)
    offset = _grouped_prediction_offset(fit, Y, offset, mask, "getLV")
    φvec = [fit.φ[fit.group[t]] for t in 1:p]
    fams = [NB1(float(φvec[t])) for t in 1:p]
    return _grouped_getLV(Y, fit.Λ, fit.β, fit.link, fams;
                          N = N, rotate = rotate, mask = mask, offset = offset)
end

# NB1 grouped fits can stop on a zero-length line-search step, which Optim counts as
# "the objective did not change" (f_converged) and reports as converged, while the
# gradient is still large (#485). Scale-aware gradient test, as in `_tweedie_verdict`
# and `_beta_grouped_g_met`: the residual is judged against `g_tol` scaled by the
# objective's own size, so a caller's g_tol below the finite-difference noise floor
# does not turn a stationary point into a non-converged fit. A gradient residual at or
# above ~1e11 is what a finite-difference step looks like when it lands on the 1e12
# objective-failure penalty — that fails this test on scale alone and is never a pass.
_nb1_grouped_g_met(res, g_tol) = (gres = Optim.g_residual(res);
    isfinite(gres) && gres <= max(g_tol, g_tol * abs(Optim.minimum(res))))

"""
    fit_nb1_gllvm_grouped(Y; K, group, link=LogLink(), mask=nothing, offset=nothing,
                          hessian=:observed, β_init=nothing, Λ_init=nothing, …) -> NB1GroupedFit

Fit a negative-binomial type-1 (NB1) GLLVM with grouped / species-specific
dispersion (gllvm's `disp.group`): species `t` shares dispersion `φ[group[t]]`
(linear variance `Var_t = μ_t(1+φ[group[t]])`). `group` is a length-p vector of
group ids (relabelled to `1..G` internally; default `1:p` = per-species). L-BFGS over
`[β; vec(Λ); log φ_1 … log φ_G]`; finite-difference gradient; warm start from
empirical log-mean intercepts + SVD loadings + a moderate per-group `φ₀`. With one
group this matches [`fit_nb1_gllvm`](@ref) when `hessian=:fisher`.
`hessian=:observed` (the default) uses the conditional NB1/log curvature that TMB's
Laplace objective uses; set `hessian=:fisher` to retain the expected-information
approximation.

!!! note "Why the default is `:observed` (2026-08-24)"
    This keyword was previously absent here, so the fit silently inherited the
    `:fisher` default of `nb1_grouped_marginal_loglik_laplace` — a *different
    objective* from TMB's, as the header of this file already warned. The symptom
    was a stable but strictly worse optimum: on a p=5, K=1, n=120 fixture the no-X
    route returned a log-likelihood 0.115 below the twin, while
    [`fit_nb1_gllvm_grouped_cov`](@ref) with an all-zero `X` — the same model, but
    already defaulting to `:observed` — matched `gllvmTMB` to 1.3e-8. Aligning this
    default with the NB2 ([`fit_nb_gllvm_grouped`](@ref)) and Beta siblings closes
    the gap. Caught by `test/parity/test_nox_dispersion_parity.jl`.
`β_init` (length p) and `Λ_init` (p×K) replace the default warm start when given,
as in [`fit_nb_gllvm`](@ref); the per-group `φ` still starts at `φ₀ = 1`.
"""
function fit_nb1_gllvm_grouped(Y::AbstractMatrix; K::Integer,
        group::AbstractVector{<:Integer} = collect(1:size(Y, 1)),
        link::Link = LogLink(), mask = nothing, offset = nothing,
        hessian::Symbol = :observed, β_init = nothing, Λ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, mask = mask,
                               caller = "fit_nb1_gllvm_grouped")
    length(group) == p || throw(ArgumentError("length(group)=$(length(group)) must equal p=$p"))
    rr = rr_theta_len(p, K)
    # relabel groups to 1..G, build species→group index
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]

    msk = _resolve_obs_mask(mask, Y)
    Yc = Integer.(_sanitize_missing(Y, 0))
    Zemp = [linkfun(link, max(Yc[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)
    _mask_warmstart!(Zemp, msk)
    β0, Λ0 = _grouped_warm_start(Zemp, K, β_init, Λ_init)
    θ0 = vcat(β0, pack_lambda(Λ0), fill(log(1.0), G))

    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        φg = exp.(θ[(p + rr + 1):(p + rr + G)])
        φvec = [φg[gidx[t]] for t in 1:p]
        v = try
            -nb1_grouped_marginal_loglik_laplace(Yc, Λ, β, φvec; link = link, mask = msk,
                                                 offset = offset, hessian = hessian,
                                                 maxiter = newton_maxiter,
                                                 tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    # NB1's Poisson limit is φ → 0: a non-converged fit crawling there is refit with those
    # groups fixed at φ = 1e-10 (the NB2 polish mirrored; #615 NB1 follow-up).
    θ̂, nll, conv0, iters0 = _nb_poisson_ridge_polish(negll, res, ls, opts, p + rr + 1;
        at_limit = <(1e-3), limit_log = log(1e-10),
        conv_test = r -> Optim.converged(r) && _nb1_grouped_g_met(r, g_tol))
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    φ̂g = exp.(θ̂[(p + rr + 1):(p + rr + G)])
    boundary = _dispersion_group_boundary(φ̂g)   # flags BOTH ends; feeds `dispersion_boundary` and the interval code
    # NB1 (Var = μ(1+φ)) has its Poisson limit at the LOWER end, φ → 0, so the maintainer
    # decision ("the Poisson limit warns only", 2026-09-29) applies there; φ above 1e6 is
    # extreme overdispersion the data do not identify and still blocks `converged`.
    lowb = φ̂g .> 1e6
    any(lowb) && @warn "NB1 grouped-dispersion fit has φ above 1e6 for group(s) $(findall(lowb)); that overdispersion is numerically flat and not identified from this data; converged is false for this fit." maxlog=1
    any(<(1e-6), φ̂g) && @warn "NB1 grouped-dispersion fit has φ below 1e-6 for group(s) $(findall(<(1e-6), φ̂g)); the group's dispersion is at the Poisson limit (no overdispersion left to estimate). The fit's other estimates are unaffected, and converged is not affected by this." maxlog=1
    loglik, conv, iters = _fit_verdict(nll, conv0, iters0)   # conv0 includes the #485 gradient test
    return NB1GroupedFit(β̂, Λ̂, φ̂g, gidx, link, loglik, conv && !any(lowb), iters, hessian,
                         boundary, _stored_offset(offset))
end

"""
    NB1GroupedCovFit

Result of [`fit_nb1_gllvm_grouped_cov`](@ref): per-trait intercepts `β`, shared
covariate coefficients `γ` (with `γ_fixed` zero mask), loadings `Λ`, per-group
linear-variance dispersion `φ`, species→group map `group`, `link`, maximised
Laplace `loglik`, `converged`, and `iterations`. Linear predictor
`η = β + Xγ + Λz` with species dispersion `φ[group[t]]` (`Var = μ(1+φ)`).
`dispersion_boundary` (length G, T14 F1, 2026-09-02) flags groups whose fitted
`φ[g]` fell outside `[1e-6, 1e6]` (see `_dispersion_group_boundary`); only the upper
end (`> 1e6`, unidentified overdispersion) forces `converged` to `false`, while
`φ[g] < 1e-6` (the NB1 Poisson limit) only warns.
"""
struct NB1GroupedCovFit
    β::Vector{Float64}
    γ::Vector{Float64}
    γ_fixed::Vector{Bool}
    Λ::Matrix{Float64}
    φ::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    dispersion_boundary::Vector{Bool}   # per-group Poisson-limit / flat-overdispersion flag (T14 F1)
end

# Positional compatibility constructors (2026-08-28 hessian; T14 F1 dispersion_boundary,
# 2026-09-02): see NBGroupedFit above.
NB1GroupedCovFit(β, γ, γ_fixed, Λ, φ, group, link, loglik, converged, iterations) =
    NB1GroupedCovFit(β, γ, γ_fixed, Λ, φ, group, link, loglik, converged, iterations, :observed)
NB1GroupedCovFit(β, γ, γ_fixed, Λ, φ, group, link, loglik, converged, iterations,
                 hessian::Symbol) =
    NB1GroupedCovFit(β, γ, γ_fixed, Λ, φ, group, link, loglik, converged, iterations,
                     hessian, _dispersion_group_boundary(φ))

function Base.show(io::IO, f::NB1GroupedCovFit)
    p, K = size(f.Λ); q = length(f.γ)
    print(io, "NB1GroupedCovFit(p=", p, ", q=", q, ", K=", K, ", G=", length(f.φ),
          ", φ=", round.(f.φ; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          any(f.dispersion_boundary) ?
              ", dispersion at boundary (groups $(findall(f.dispersion_boundary)))" : "",
          f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::NB1GroupedCovFit) = fit.Λ
_loglik(fit::NB1GroupedCovFit)   = fit.loglik

function _nparams(fit::NB1GroupedCovFit)
    p, K = size(fit.Λ)
    return p + count(!, fit.γ_fixed) + rr_theta_len(p, K) + length(fit.φ)
end

"""
    getLV(fit::NB1GroupedCovFit, Y, X; rotate=true, mask=nothing) -> n×K matrix

Conditional latent scores at `η = β + Xγ + Λz` with per-trait NB1 φ.
"""
function getLV(fit::NB1GroupedCovFit, Y::AbstractMatrix{<:Integer},
               X::AbstractArray{<:Real, 3};
               rotate::Bool = true, mask = nothing)
    p = size(Y, 1)
    φvec = [fit.φ[fit.group[t]] for t in 1:p]
    fams = [NB1(float(φvec[t])) for t in 1:p]
    O = _build_offset(X, fit.γ)
    return _grouped_getLV(Y, fit.Λ, fit.β, fit.link, fams;
                          rotate = rotate, mask = mask, offset = O)
end

"""
    fit_nb1_gllvm_grouped_cov(Y; X, K, group=1:p, link=LogLink(), mask=nothing,
                              γ_fixed=nothing, hessian=:observed, …) -> NB1GroupedCovFit

Fit an NB1 GLLVM with **grouped / per-trait linear-variance φ** and **shared site
covariates** `X` (`p×n×q`). Working vector
`[β; γ_free; pack(Λ); log φ_1 … log φ_G]`; offset `O = Xγ` is passed into the
grouped Laplace marginal. Default `hessian=:observed` matches TMB; identity
checks against shared [`fit_gllvm_cov`](@ref) / G=1 should force
`hessian=:fisher`. Public / bridge default under X for NB1 (twin API B;
decision 2026-08-05). Keep [`fit_gllvm_cov`](@ref) for the shared-`φ` + X
opt-in. Identity checks against shared cov should use `group = ones(Int, p)`.
"""
function fit_nb1_gllvm_grouped_cov(Y::AbstractMatrix; X::AbstractArray{<:Real, 3},
        K::Integer, group::AbstractVector{<:Integer} = collect(1:size(Y, 1)),
        link::Link = LogLink(), mask = nothing, γ_fixed = nothing,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q) = ($p, $n, q); got $(size(X))"))
    length(group) == p || throw(ArgumentError("length(group)=$(length(group)) must equal p=$p"))
    q_full = size(X, 3)
    γ_fixed_mask = _fixed_zero_mask(γ_fixed, q_full, "γ_fixed")
    X_fit, _ = _slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    rr = rr_theta_len(p, K)
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]

    msk = _resolve_obs_mask(mask, Y)
    Yc = Integer.(_sanitize_missing(Y, 0))
    Zemp = [linkfun(link, max(Yc[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    _mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(β0, zeros(q), pack_lambda(Λ0), fill(log(1.0), G))

    function negll(θ)
        β = θ[1:p]
        γ = θ[(p + 1):(p + q)]
        Λ = unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        φg = exp.(θ[(p + q + rr + 1):(p + q + rr + G)])
        φvec = [φg[gidx[t]] for t in 1:p]
        O = _build_offset(X_fit, γ)
        v = try
            -nb1_grouped_marginal_loglik_laplace(Yc, Λ, β, φvec; link = link, mask = msk,
                                                 offset = O, hessian = hessian,
                                                 maxiter = newton_maxiter,
                                                 tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    # NB1's Poisson limit is φ → 0: a non-converged fit crawling there is refit with those
    # groups fixed at φ = 1e-10 (the NB2 polish mirrored; #615 NB1 follow-up).
    θ̂, nll, conv0, iters0 = _nb_poisson_ridge_polish(negll, res, ls, opts, p + q + rr + 1;
        at_limit = <(1e-3), limit_log = log(1e-10))
    β̂ = θ̂[1:p]
    γ̂_free = θ̂[(p + 1):(p + q)]
    γ̂ = collect(Float64, _expand_fixed_zero(γ̂_free, γ_fixed_mask))
    Λ̂ = unpack_lambda(θ̂[(p + q + 1):(p + q + rr)], p, K)
    φ̂g = exp.(θ̂[(p + q + rr + 1):(p + q + rr + G)])
    boundary = _dispersion_group_boundary(φ̂g)   # flags BOTH ends; feeds `dispersion_boundary` and the interval code
    # NB1 (Var = μ(1+φ)) has its Poisson limit at the LOWER end, φ → 0, so the maintainer
    # decision ("the Poisson limit warns only", 2026-09-29) applies there; φ above 1e6 is
    # extreme overdispersion the data do not identify and still blocks `converged`.
    lowb = φ̂g .> 1e6
    any(lowb) && @warn "NB1 grouped-cov fit has φ above 1e6 for group(s) $(findall(lowb)); that overdispersion is numerically flat and not identified from this data; converged is false for this fit." maxlog=1
    any(<(1e-6), φ̂g) && @warn "NB1 grouped-cov fit has φ below 1e-6 for group(s) $(findall(<(1e-6), φ̂g)); the group's dispersion is at the Poisson limit (no overdispersion left to estimate). The fit's other estimates are unaffected, and converged is not affected by this." maxlog=1
    loglik, conv, iters = _fit_verdict(nll, conv0, iters0)
    return NB1GroupedCovFit(β̂, γ̂, collect(Bool, γ_fixed_mask), Λ̂, φ̂g, gidx, link,
                            loglik, conv && !any(lowb), iters, hessian, boundary)
end

# ===========================================================================
# Tweedie family — grouped / species-specific dispersion φ (gllvm's disp.group with
# disp.formula = NULL). Each species t carries its own dispersion φ_{g(t)}, so the
# Var = φ μ^power overdispersion can vary across species (or groups). The POWER
# p ∈ (1,2) is SHARED (a single global power, matching gllvm — `disp.formula`
# governs the dispersion only). The dispersion φ and shared power are carried in
# the family marker `TweedieED(φ, power)`. This mirrors the NB2 grouped path above;
# the shared Tweedie hot path (tweedie.jl) is left untouched.
#
# FIXED DEFECT (2026-08-28): `fit_tweedie_gllvm` (shared route) flipped its
# default log-det curvature to `:observed` on the maintainer's decision
# (docs/dev-log/decisions/2026-08-28-arc-decision-batch.md), but
# `_tweedie_grouped_loglik_site` had NO `hessian` selector at all — it was
# unconditionally Fisher. So with G = 1 and a constant `φvec`, the grouped
# route no longer reduced to the shared route's DEFAULT objective; it reduced
# to the shared route called with `hessian = :fisher` explicitly. Aligned here
# with the same-shaped NB2/Beta/NB1/Gamma grouped work: `hessian::Symbol`
# threaded through the LOG-DET only (role separation below — the Newton mode
# search stays Fisher-scored unconditionally, matching every sibling). Default
# is `:observed`, using the tweedie.jl `_glm_obs_weight(::TweedieED, …)`
# observed curvature (FD-verified, always non-negative for p ∈ (1,2), y ≥ 0 —
# no PD-guard sign concern for this family, so no backtracking gate is needed
# here, matching the caution in the maintainer's brief). With G = 1 and a
# constant `φvec`, this grouped route now reduces EXACTLY to the shared route
# under ITS default (`hessian = :observed`), and under `hessian = :fisher` to
# the shared route with `:fisher`. See `docs/src/response-families.md` and
# `docs/src/gllvmtmb-parity.md` for the (now-updated) user-facing note.
# ===========================================================================

# Observed vs expected log-det weight selector, mirroring
# `_nb_grouped_laplace_weight` / `_nb1_grouped_laplace_weight` /
# `_beta_grouped_laplace_weight`. `η` is required by `_glm_obs_weight`'s
# TweedieED/LogLink signature (tweedie.jl).
function _tweedie_grouped_laplace_weight(hessian::Symbol, f::TweedieED, μ, me, y, link::Link, η)
    hessian === :fisher && return _glm_weight(f, μ, 1, me)
    hessian === :observed || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    link isa LogLink || throw(ArgumentError(
        "hessian=:observed is currently supported only for Tweedie with LogLink()"))
    return _glm_obs_weight(f, μ, 1, me, y, link, η)
end

# Per-site Laplace log-marginal with per-species Tweedie dispersion markers `fams`.
function _tweedie_grouped_loglik_site(fams::AbstractVector, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, hessian::Symbol = :observed,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    z = zeros(K)
    local A
    for _ in 1:maxiter
        η  = _clamp_eta.(β .+ off .+ Λ * z)
        μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
        me = mu_eta.(Ref(link), η)
        s  = _glm_score.(fams, μ, n, me, y)
        # Role separation (mirrors NB2/Beta/NB1 above). The MODE SEARCH is
        # Fisher-scored, ALWAYS — `:fisher` here is not the caller's selector.
        # Expected information is >= 0, so `Λ'WΛ + I` is SPD by construction
        # and every Newton step is a descent step. The selector still governs
        # the post-loop log-det below, which is the only role that needs the
        # observed curvature. The converged mode is unchanged either way: it
        # is the fixed point of `Λ's − z = 0`, which does not involve W at all.
        W  = _tweedie_grouped_laplace_weight.(Ref(:fisher), fams, μ, me, y, Ref(link), η)
        if mask !== nothing
            s = ifelse.(mask, s, 0.0)
            W = ifelse.(mask, W, 0.0)
        end
        A  = Symmetric(Λ' * (W .* Λ) + I)
        Δ  = _safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z  = z .+ Δ
        maximum(abs, Δ) < tol && break
    end
    η  = _clamp_eta.(β .+ off .+ Λ * z)
    μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
    me = mu_eta.(Ref(link), η)
    W  = _tweedie_grouped_laplace_weight.(Ref(hessian), fams, μ, me, y, Ref(link), η)
    if mask !== nothing
        W = ifelse.(mask, W, 0.0)
    end
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = 0.0
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _glm_logpdf(fams[t], μ[t], n[t], y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

"""
    tweedie_grouped_marginal_loglik_laplace(Y, Λ, β, φvec, power; link=LogLink(),
                                            mask=nothing, offset=nothing, kwargs...) -> Float64

Total Laplace log-marginal of a Tweedie GLLVM with **per-species** dispersion `φvec`
(length p) and a single SHARED `power` ∈ (1,2) (`Var_t = φvec[t]·μ_t^power`). `Y` is
the p×n matrix of non-negative reals (point mass at 0 allowed); `Λ` p×K; `β` length-p.
With a constant `φvec = fill(φ, p)` (same `power`) and `hessian=:fisher` this
equals the shared-dispersion [`tweedie_marginal_loglik_laplace`](@ref) to
machine precision; `hessian=:observed` (the default) uses the conditional
observed curvature (TMB Laplace) that [`fit_tweedie_gllvm`](@ref) defaults to.
"""
function tweedie_grouped_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, φvec::AbstractVector, power::Real; link::Link = LogLink(),
        mask = nothing, offset = nothing, hessian::Symbol = :observed, kwargs...)
    p = size(Λ, 1)
    length(φvec) == p || throw(ArgumentError("length(φvec)=$(length(φvec)) must equal p=$p"))
    N = ones(Int, size(Y))
    fams = [TweedieED(float(φvec[t]), float(power)) for t in 1:p]
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask   === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        acc += _tweedie_grouped_loglik_site(fams, view(Y, :, i), view(N, :, i), Λ, β, link;
                                            mask = mi, offset = oi, hessian = hessian, kwargs...)
    end
    return acc
end

# R's Tweedie engine carries `logit_p_tweedie` per trait.  Keep the scalar
# method above for the historical shared-power contract and make the trait
# vector a separate dispatch path; this avoids scalar/vector branches in the
# site likelihood itself.
function tweedie_grouped_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, φvec::AbstractVector, power::AbstractVector; link::Link = LogLink(),
        mask = nothing, offset = nothing, hessian::Symbol = :observed, kwargs...)
    p = size(Λ, 1)
    length(φvec) == p || throw(ArgumentError("length(φvec)=$(length(φvec)) must equal p=$p"))
    length(power) == p || throw(ArgumentError("length(power)=$(length(power)) must equal p=$p"))
    all(x -> isfinite(x) && 1.0 < x < 2.0, power) ||
        throw(ArgumentError("each Tweedie power must be finite and strictly between 1 and 2"))
    N = ones(Int, size(Y))
    fams = [TweedieED(float(φvec[t]), float(power[t])) for t in 1:p]
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask   === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        acc += _tweedie_grouped_loglik_site(fams, view(Y, :, i), view(N, :, i), Λ, β, link;
                                            mask = mi, offset = oi, hessian = hessian, kwargs...)
    end
    return acc
end

"""
    TweedieGroupedFit

Result of [`fit_tweedie_gllvm_grouped`](@ref): intercepts `β` (length p), loadings `Λ`
(p×K), the per-group dispersion vector `φ` (length G), the SHARED `power` ∈ (1,2), the
species→group map `group` (length p), the `link`, the maximised Laplace `loglik`,
`converged`, and `iterations`. The per-species dispersion is `φ[group[t]]`
(`Var_t = φ[group[t]]·μ_t^power`). `converged` uses the same `_tweedie_verdict`
contract as [`fit_tweedie_gllvm`](@ref): a successfully evaluated objective, a
strictly interior power, and a gradient residual small relative to the
objective's scale.
"""
struct TweedieGroupedFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    φ::Vector{Float64}
    power::Float64
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    power_fixed::Bool # true only when `power = p0` removed all power coordinates
end

# Positional compatibility constructor (2026-08-28): see NBGroupedFit above.
# Defaults to `:observed` (2026-08-28 alignment), matching the NB2/Beta/NB1/
# Gamma grouped siblings and `fit_tweedie_gllvm`'s own default.
TweedieGroupedFit(β, Λ, φ, power, group, link, loglik, converged, iterations) =
    TweedieGroupedFit(β, Λ, φ, power, group, link, loglik, converged, iterations, :observed, false)
TweedieGroupedFit(β, Λ, φ, power, group, link, loglik, converged, iterations, hessian) =
    TweedieGroupedFit(β, Λ, φ, power, group, link, loglik, converged, iterations, hessian, false)

function Base.show(io::IO, f::TweedieGroupedFit)
    p, K = size(f.Λ)
    print(io, "TweedieGroupedFit(p=", p, ", K=", K, ", G=", length(f.φ),
          ", φ=", round.(f.φ; sigdigits = 4),
          ", power=", round(f.power; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::TweedieGroupedFit) = fit.Λ
_loglik(fit::TweedieGroupedFit)   = fit.loglik

# Free params: β (p) + reduced loadings Λ + one dispersion per group (G) + an
# estimated shared power only. Fixed power is model identity, never inferred
# from the numerical value.
function _nparams(fit::TweedieGroupedFit)
    p, K = size(fit.Λ)
    return p + rr_theta_len(p, K) + length(fit.φ) + (fit.power_fixed ? 0 : 1)
end

"""
    TweediePerTraitPowerFit

Result of `fit_tweedie_gllvm_grouped(...; power_group=:species)`.  The grouped
dispersion vector `φ` is indexed by `group`; `power[t]` is the separately
estimated Tweedie power for species `t`.
"""
struct TweediePerTraitPowerFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    φ::Vector{Float64}
    power::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol
end

_loadings(fit::TweediePerTraitPowerFit) = fit.Λ
_loglik(fit::TweediePerTraitPowerFit) = fit.loglik
function _nparams(fit::TweediePerTraitPowerFit)
    p, K = size(fit.Λ)
    return p + rr_theta_len(p, K) + length(fit.φ) + length(fit.power)
end
StatsAPI.dof(fit::TweediePerTraitPowerFit) = _nparams(fit)

struct _TweediePowerSpec
    values::Vector{Float64}
    nfree::Int
    fixed::Bool
    mode::Symbol
end

function _tweedie_power_spec(power::Union{Nothing,Real}, power_group::Symbol, p::Integer)
    power_group in (:shared, :species) || throw(ArgumentError(
        "power_group must be :shared or :species; got :$power_group"))
    p > 0 || throw(ArgumentError("number of species must be positive"))
    if power !== nothing
        isfinite(power) && 1.0 < power < 2.0 || throw(ArgumentError(
            "fixed Tweedie power must be finite and strictly between 1 and 2"))
        return _TweediePowerSpec(fill(float(power), p), 0, true, :fixed)
    elseif power_group === :shared
        return _TweediePowerSpec(fill(1.5, p), 1, false, :shared)
    else
        return _TweediePowerSpec(fill(1.5, p), p, false, :species)
    end
end

function _tweedie_power_init(power_init::Union{Real,AbstractVector}, p::Integer, mode::Symbol)
    values = if power_init isa Real
        fill(float(power_init), p)
    else
        length(power_init) == p || throw(ArgumentError(
            "length(power_init)=$(length(power_init)) must equal p=$p"))
        collect(float.(power_init))
    end
    all(x -> isfinite(x) && 1.0 < x < 2.0, values) || throw(ArgumentError(
        "each power_init value must be finite and strictly between 1 and 2"))
    mode === :shared && !all(==(values[1]), values) && throw(ArgumentError(
        "power_init must be scalar or constant when power_group=:shared"))
    return values
end

_tweedie_xi(p::Real) = log((float(p) - 1.0) / (2.0 - float(p)))
_tweedie_power(ξ::Real) = 1.0 + 1.0 / (1.0 + exp(-ξ))

"""
    fit_tweedie_gllvm_grouped(Y; K, group, power=nothing, power_group=:shared,
                              power_init=1.5, link=LogLink(), …)

Fit a Tweedie GLLVM with grouped / species-specific dispersion (gllvm's `disp.group`):
species `t` shares dispersion `φ[group[t]]`. `group` is a length-p vector of group
ids (relabelled to `1..G` internally; default `1:p` = per-species). The three power
contracts are explicit: `power=p0` fixes a common `p0 ∈ (1,2)`; with
`power=nothing`, `power_group=:shared` estimates one power (the historical Julia
grouped model), while `power_group=:species` estimates a vector `p_t` and matches
the frozen gllvmTMB default parameterisation. Invalid controls are rejected before
fitting, including an invalid `power_group` paired with fixed `power`.

L-BFGS uses `[β; vec(Λ); log φ_1 … log φ_G; ξ…]`; every free power uses
`p = 1 + 1/(1+exp(-ξ))` (so `ξ = 0 ⇒ p = 1.5`). Finite-difference gradient; warm start from log
row-means of `Y + c` as intercepts + SVD of row-centred log-`(Y + c)` as loadings
+ a moderate per-group `φ₀` + `ξ₀ = logit(power_init − 1)`, where
`c = 0.1 · mean(Y[Y > 0])` keeps the exact zeros on the data's own scale (the
same offset as [`fit_tweedie_gllvm`](@ref)).

`converged` is not `Optim`'s verdict alone: it uses `_tweedie_verdict`, so a
fit that stalls, sits on the failure sentinel, or runs the power to the closed
end of `(1, 2)` reports `converged = false` rather than advertising that point
as a maximum. With one group and `hessian=:observed` (the default) this
matches [`fit_tweedie_gllvm`](@ref)'s own default; `hessian=:fisher` selects
the previous expected-information objective on both.
"""
function fit_tweedie_gllvm_grouped(Y::AbstractMatrix{<:Real}; K::Integer,
        group::AbstractVector{<:Integer} = collect(1:size(Y, 1)),
        power::Union{Nothing,Real} = nothing, power_group::Symbol = :shared,
        power_init::Union{Real,AbstractVector} = 1.5,
        link::Link = LogLink(), mask = nothing, offset = nothing,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, mask = mask,
                               caller = "fit_tweedie_gllvm_grouped")
    length(group) == p || throw(ArgumentError("length(group)=$(length(group)) must equal p=$p"))
    hessian in (:fisher, :observed) || throw(ArgumentError(
        "fit_tweedie_gllvm_grouped: hessian must be :fisher or :observed; got :$hessian"))
    spec = _tweedie_power_spec(power, power_group, p)
    power0 = spec.fixed ? spec.values : _tweedie_power_init(power_init, p, spec.mode)
    rr = rr_theta_len(p, K)
    # relabel groups to 1..G, build species→group index
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]

    msk = _resolve_obs_mask(mask, Y)
    Yc  = _sanitize_missing(Y, 1e-6)
    Zemp = log.(Yc .+ _tweedie_log_offset(Yc, msk))
    offset === nothing || (Zemp .-= offset)
    _mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    ξ0 = spec.nfree == 0 ? Float64[] :
         spec.mode === :shared ? [_tweedie_xi(power0[1])] : _tweedie_xi.(power0)
    θ0 = vcat(β0, pack_lambda(Λ0), fill(log(1.0), G), ξ0)

    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        φg = exp.(θ[(p + rr + 1):(p + rr + G)])
        φvec = [φg[gidx[t]] for t in 1:p]
        ξ = spec.nfree == 0 ? Float64[] : @view θ[(p + rr + G + 1):end]
        pw = if spec.nfree == 0
            spec.values
        elseif spec.mode === :shared
            fill(_tweedie_power(ξ[1]), p)
        else
            _tweedie_power.(ξ)
        end
        v = try
            -tweedie_grouped_marginal_loglik_laplace(Yc, Λ, β, φvec, pw; link = link,
                                                     mask = msk, offset = offset,
                                                     hessian = hessian,
                                                     maxiter = newton_maxiter,
                                                     tol = newton_tol)
        catch
            return _TWEEDIE_FAIL_PENALTY
        end
        return isfinite(v) ? v : _TWEEDIE_FAIL_PENALTY
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    φ̂g = exp.(θ̂[(p + rr + 1):(p + rr + G)])
    ξ̂ = spec.nfree == 0 ? Float64[] : @view θ̂[(p + rr + G + 1):end]
    p̂ = if spec.nfree == 0
        spec.values
    elseif spec.mode === :shared
        fill(_tweedie_power(ξ̂[1]), p)
    else
        _tweedie_power.(ξ̂)
    end
    verdict_xi = spec.nfree == 0 ? 0.0 : spec.mode === :shared ? ξ̂[1] : ξ̂
    conv, loglik, reason = _tweedie_verdict(Optim.converged(res), Optim.g_residual(res),
                                            Optim.minimum(res), verdict_xi, g_tol)
    if reason === :objective_failed
        @warn "fit_tweedie_gllvm_grouped: the Laplace marginal could not be evaluated at any \
               accepted point; returning converged = false and loglik = -Inf. Try a \
               different `power_init`, or check `Y` for extreme values."
    elseif reason === :power_at_boundary
        @warn "fit_tweedie_gllvm_grouped: the power ran to the boundary of (1, 2) \
               (p̂ = $(p̂), φ̂ = $(φ̂g)); the fit is flagged as not converged."
    end
    if spec.mode === :species && !spec.fixed
        return TweediePerTraitPowerFit(β̂, Λ̂, φ̂g, collect(p̂), gidx, link, loglik, conv,
                                       Optim.iterations(res), hessian)
    end
    return TweedieGroupedFit(β̂, Λ̂, φ̂g, p̂[1], gidx, link, loglik, conv,
                             Optim.iterations(res), hessian, spec.fixed)
end
