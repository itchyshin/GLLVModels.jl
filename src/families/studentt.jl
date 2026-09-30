# Student-t (heavy-tailed continuous) family pieces for the generic Laplace core
# (src/families/laplace.jl). y_t ∈ ℝ; location η (IDENTITY link, so μ = η), scale
# σ > 0 and finite fixed degrees of freedom ν > 0 (estimated ν > 1): the law is the
# location–scale t, (y − η)/σ ~ t_ν. A numeric `nu` fixes ν; `nu = nothing`
# estimates it. The scale σ is always estimated on a log scale, and
# `disp_group = :species` estimates one scale and (when ν is free) one ν per
# trait. This implementation lives in this file, not `grouped_dispersion.jl`.
# The conditional density is
#
#   p(y | η) = Γ((ν+1)/2) / (Γ(ν/2) √(νπ) σ) · (1 + (y−η)²/(ν σ²))^{−(ν+1)/2},
#
# i.e. a Gaussian-tailed model robustified against outliers; as ν → ∞ it tends to
# Normal(η, σ²). The marker `StudentTFamily(ν, σ)` carries a fixed numerical ν
# for an individual likelihood evaluation; the fitter supplies the estimated
# value when `nu = nothing`.
#
# Score/weight wrt η (identity link ⇒ dμ/dη = me = 1). The robust t-score is
#   r = y − η,   s_η = (ν+1) r / (ν σ² + r²)
# (the score down-weights large residuals — the bounded-influence property of the
# t; Lange, Little & Taylor 1989 JASA). The expected (Fisher) information wrt η is
#   I_η = (ν+1) / ((ν+3) σ²)
# (a constant, the standard location-t information; Lange et al. 1989 eq. for the
# scaled-t score variance). Using the EXPECTED information as the Fisher-scoring
# weight keeps W ≥ 0 (the OBSERVED Hessian of a t is non-monotone and can go
# negative for |r| large, which would break the SPD Newton step), so the generic
# mode-finder in laplace.jl stays well-conditioned:
#   _glm_score  = s_η · me = (ν+1) r / (ν σ² + r²)        (me = 1)
#   _glm_weight = I_η · me² = (ν+1) / ((ν+3) σ²)          (me = 1, ⇒ W ≥ 0)
#
# `_glm_logpdf` uses a stable normalizer so ForwardDiff Duals flow
# cleanly through both η (via the residual r = y − η) and log σ (via σ in the aux),
# which is what makes the generic scalar-aux implicit-gradient path AD-clean.

"""
    StudentTFamily(ν = nothing, σ = 1.0)

Student-t (heavy-tailed continuous) family marker: location–scale t with
finite degrees of freedom `ν > 0` (or `nothing` before fitting) and scale `σ > 0`,
identity link (location `μ = η`),
so `(y − η)/σ ~ t_ν`. Used as the family argument to the generic Laplace core and
to [`fit_gllvm`](@ref):

```julia
fit_gllvm(Y; family = StudentTFamily(), K = 2)      # estimate ν (default)
fit_gllvm(Y; family = StudentTFamily(7.0), K = 2)   # fix a lighter tail
```

The two fields play **different** roles on the public route. `ν` is structural: it
defines the likelihood and a numeric value fixes it, while `ν = nothing` asks the
fitter to estimate it. `σ` is a **tag payload** —
the scale is always estimated, so `StudentTFamily(4.0, 1.0)` and
`StudentTFamily(4.0, 9.0)` give the same fit; pass `σ_init` to
[`fit_studentt_gllvm`](@ref) to seed it. Internally the Laplace kernels construct
their own per-iteration `StudentTFamily(ν, σ)` markers, which is what the `σ`
field is for. As `ν → ∞` the family tends to `Normal(η, σ²)`.
"""
struct StudentTFamily{N<:Union{Real,Nothing}, T<:Real}
    ν::N
    σ::T
end
StudentTFamily(ν::Real, σ::Real) = (νσ = promote(float(ν), float(σ)); StudentTFamily{typeof(νσ[1]), typeof(νσ[2])}(νσ[1], νσ[2]))
StudentTFamily(ν::Nothing, σ::Real) = StudentTFamily{Nothing, typeof(float(σ))}(nothing, float(σ))
StudentTFamily(ν::Real) = StudentTFamily(float(ν), 1.0)
StudentTFamily(::Nothing) = StudentTFamily(nothing, 1.0)
StudentTFamily() = StudentTFamily(nothing, 1.0)

const StudentT = StudentTFamily

default_link(::StudentTFamily) = IdentityLink()

# Location is unconstrained ⇒ no μ clamp (identity link, μ = η ∈ ℝ).
_clamp_mu(::StudentTFamily, μ) = μ

# Robust t-score wrt η: (ν+1)(y−μ)/(ν σ² + (y−μ)²), times me (= 1 for identity).
function _glm_score(f::StudentTFamily, μ, n, me, y)
    r = y - μ
    return (f.ν + one(f.ν)) * r / (f.ν * f.σ^2 + r^2) * me
end

# Expected (Fisher) information wrt η: (ν+1)/((ν+3) σ²), times me² (= 1). W ≥ 0.
_glm_weight(f::StudentTFamily, μ, n, me) =
    (f.ν + one(f.ν)) / ((f.ν + 3 * one(f.ν)) * f.σ^2) * me^2

# Damped mode search without the small-step bypass (#623). The Fisher weight
# above is (ν+3)/ν times smaller than the observed curvature at small residuals,
# so for ν < 3 a full Fisher step overshoots the mode by more than a factor of 2
# and the undamped search cycles: on the #623 fixture (ν = 1.5, σ = 0.3) it
# alternated between -0.047 and 0.212 around the mode 0.074. Step halving on the
# log posterior stops the cycle; `_laplace_mode_robust` makes it apply to small
# steps too and extrapolates along non-concave ridges (see laplace.jl).
_laplace_mode_should_backtrack(::StudentTFamily) = true
_laplace_mode_robust(::StudentTFamily) = true

# Two-peak joints (#626). The Student-t log joint in z is non-concave wherever a
# residual exceeds σ√ν, and there it can have a second, higher peak, at which the
# outlying trait is fitted and the others are treated as outliers instead. A local
# Newton search from z = 0 reaches the nearer peak: on the #623 fixture
# `studentt_K1_true`, sites 54 and 103 stopped 0.24 and 2.85 below the global one.
# So after the search, the observed trait with the largest residual beyond σ√ν
# gives one extra start, the point reached by moving along its loading row until
# that trait fits exactly, and the higher peak is kept. It only runs where such a
# residual exists (a concave joint has one peak), and a restart must beat the first
# peak by more than rounding level, so single-peak sites keep the same mode.
function _laplace_mode_alt_starts(f::StudentTFamily, z, y, n, Λ, β, link::IdentityLink;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    off = offset === nothing ? false : offset
    η = β .+ off .+ Λ * z
    thr = f.σ * sqrt(f.ν)
    # One restart, from the most outlying trait: restarting from every outlier cost
    # 41% more time in test_studentt.jl (58 s -> 82 s) for no extra fixture site.
    tbest = 0
    rmax = thr
    @inbounds for t in eachindex(y)
        (mask === nothing || mask[t]) || continue
        r = abs(y[t] - η[t])
        r > rmax && any(!iszero, view(Λ, t, :)) && (tbest = t; rmax = r)
    end
    tbest == 0 && return z
    q0 = _laplace_mode_logpost(f, y, n, Λ, β, link, z; mask = mask, offset = offset)
    isfinite(q0) || return z
    λ = Λ[tbest, :]
    zt = _laplace_mode(f, y, n, Λ, β, link; mask = mask, offset = offset,
                       maxiter = maxiter, tol = tol,
                       z0 = z .+ λ .* ((y[tbest] - η[tbest]) / dot(λ, λ)), alt_starts = false)
    qt = _laplace_mode_logpost(f, y, n, Λ, β, link, zt; mask = mask, offset = offset)
    return (isfinite(qt) && qt > q0 + 1e-10 * (1 + abs(q0))) ? zt : z
end
_laplace_mode_step_weight(f::StudentTFamily, μ, n, me, y, link::IdentityLink, η) =
    ismissing(y) ? _glm_weight(f, μ, n, me) :
    max(_glm_obs_weight(f, μ, n, me, y, link, η), _glm_weight(f, μ, n, me))

# Closed-form location–scale t log-density:
#   ℓ = logΓ((ν+1)/2) − logΓ(ν/2) − ½log(νπ) − log σ − (ν+1)/2 · log(1 + r²/(ν σ²)).
# Limit the fixed-order series to Float64, including nested ForwardDiff Duals.
# BigFloat and BigFloat-backed Duals retain their caller-selected precision.
@inline _studentt_float64_backed(::Any) = false
@inline _studentt_float64_backed(::Float64) = true
@inline _studentt_float64_backed(x::ForwardDiff.Dual) = _studentt_float64_backed(ForwardDiff.value(x))

function _studentt_log_normalizer(ν)
    if _studentt_float64_backed(ν) && ν >= 64
        # DLMF 5.11.8, subtracting h=1/2 and h=0 at z=ν/2.
        # First omitted term is 691/(88ν^11), about 1.1e-19 at ν=64.
        # Direct differentiation of this expression avoids digamma cancellation.
        u = inv(ν)
        u2 = u*u
        o = one(ν)
        correction = u * (-o/4 + u2*(o/24 + u2*(-o/20 + u2*(17o/112 - u2*31o/36))))
        return -log(2*oftype(ν, π))/2 + correction
    end
    half = (ν + one(ν))/2
    return loggamma(half) - loggamma(ν/2) - log(ν*oftype(half, π))/2
end

function _glm_logpdf(f::StudentTFamily, μ, n, y)
    ν = f.ν
    σ = f.σ
    r = y - μ
    half = (ν + one(ν)) / 2
    return _studentt_log_normalizer(ν) - log(σ) -
           half * log1p(r^2 / (ν * σ^2))
end

"""
    studentt_marginal_loglik_laplace(Y, Λ, β, σ; ν=4.0, link=IdentityLink(), kwargs...) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a Student-t GLLVM with
degrees of freedom `ν` and scale `σ` (`(y − η)/σ ~ t_ν`, identity link) — a
thin wrapper over the family-generic `marginal_loglik_laplace` or grouped site
evaluator. `Y` is the p×n response matrix; `Λ` p×K; `β` length-p. `σ` and `ν`
may each be a scalar `Real` or a length-p `AbstractVector`. As `ν → ∞` this tends
to the Gaussian marginal.
"""
studentt_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector,
        σ::Real; ν::Real = 4.0, link::Link = IdentityLink(), kwargs...) =
    marginal_loglik_laplace(StudentTFamily(ν, σ), Y, ones(Int, size(Y)), Λ, β, link; kwargs...)

function studentt_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector,
        σ::Union{Real, AbstractVector}; ν::Union{Real, AbstractVector} = 4.0, link::Link = IdentityLink(),
        mask = nothing, offset = nothing, kwargs...)
    p = size(Λ, 1)
    if σ isa AbstractVector
        length(σ) == p || throw(ArgumentError("length(σ)=$(length(σ)) must equal p=$p"))
    end
    if ν isa AbstractVector
        length(ν) == p || throw(ArgumentError("length(ν)=$(length(ν)) must equal p=$p"))
    end
    fams = if σ isa AbstractVector && ν isa AbstractVector
        [StudentTFamily(ν[t], σ[t]) for t in 1:p]
    elseif σ isa AbstractVector
        [StudentTFamily(ν, σ[t]) for t in 1:p]
    elseif ν isa AbstractVector
        [StudentTFamily(ν[t], σ) for t in 1:p]
    else
        [StudentTFamily(ν, σ) for t in 1:p]
    end
    N = ones(Int, size(Y))
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask   === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        acc += _studentt_grouped_loglik_site(fams, view(Y, :, i), view(N, :, i), Λ, β, link;
                                             mask = mi, offset = oi, kwargs...)
    end
    return acc
end

# ---------------------------------------------------------------------------
# Per-trait dispersion substrate (2026-08-28, gllvmTMB.cpp:1184 `log_sigma_student`,
# length n_traits — measured baseline Δ logLik = +1.070 at df fixed = ν on both
# sides, docs/dev-log/decisions/2026-08-28-studentt-parameterisation.md). Mirrors
# the established `grouped_dispersion.jl` pattern (Gamma/Beta/NB2/NB1): the
# family-generic single-marker core in `families/laplace.jl` broadcasts ONE
# family via `Ref(family)`, so per-species dispersion needs its own site kernel
# that broadcasts a length-p VECTOR of `StudentTFamily` markers instead — reusing
# the exact same `_glm_score`/`_glm_weight`/`_glm_logpdf`/`_clamp_mu`/
# `_glm_obs_weight` pieces already defined above. The shared-σ path
# (`studentt_marginal_loglik_laplace(..., σ::Real; ...)` above) is left
# byte-for-byte untouched, which is what keeps `disp_group = :shared` (the
# default) bit-identical to the pre-existing fitter.
function _studentt_grouped_laplace_weight(hessian::Symbol, f::StudentTFamily, μ, me, y, link::Link, η)
    hessian === :fisher && return _glm_weight(f, μ, 1, me)
    hessian === :observed || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    link isa IdentityLink || throw(ArgumentError(
        "hessian=:observed is currently supported only for Student-t with IdentityLink()"))
    return _glm_obs_weight(f, μ, 1, me, y, link, η)
end

# Per-site mode search for the Student-t grouped kernel (#503, following the
# #479/#480/#484/#500 pattern). Returns `(z, converged)`. Mode search is ALWAYS
# Fisher-scored (role separation, 2026-08-25 convention shared with
# Gamma/Beta/NB2 grouped kernels): expected information is >= 0, so `Λ'WΛ + I`
# is SPD by construction, independent of the caller's `hessian` (which only
# selects the post-loop log-det curvature, in `_studentt_grouped_loglik_site`
# below). There is no observed-curvature Newton fallback here, unlike Gamma's
# LogLink route: the observed Student-t curvature is genuinely negative for
# |r| > σ√ν (documented above at `_glm_obs_weight`), so it cannot be used as a
# guaranteed-descent step the way Gamma/log's α·y/μ can.
#
# Before this fix the loop below ran undamped Fisher scoring and returned
# whatever `z` it held when `maxiter` was reached, converged or not — the same
# defect class as #479 (Gamma) and #500 (twopart): a step that overshoots the
# per-site log-posterior is never rejected, so the mode search can leave a
# site far from its stationary point while still returning a finite value.
# Now a step that lowers the per-site log-posterior is halved (the generic
# core's `_laplace_mode`/`_grouped_laplace_mode` rule, reusing
# `_grouped_laplace_mode_logpost` from `grouped_dispersion.jl`, included
# before this file), so small steps and accepted full steps are bit-identical
# to the old loop. `converged` is true only when the FULL proposed step (not a
# halved one) is below `tol` AND the log-posterior gradient itself is below
# `grad_tol` (see the comment at the check, below): mirrors
# `_gamma_grouped_mode`'s "a heavily halved step does not count as converged"
# rule, plus the extra gradient check this family's constant Fisher weight
# needs.
function _studentt_grouped_mode(fams::AbstractVector{<:StudentTFamily}, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9,
        grad_tol::Real = 1e-6)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    z = zeros(K)
    for _ in 1:maxiter
        η  = _clamp_eta.(β .+ off .+ Λ * z)
        μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
        me = mu_eta.(Ref(link), η)
        s  = _glm_score.(fams, μ, n, me, y)
        W  = _glm_weight.(fams, μ, n, me)
        if mask !== nothing
            s = ifelse.(mask, s, 0.0)
            W = ifelse.(mask, W, 0.0)
        end
        # Log-posterior gradient wrt z at the CURRENT z (before this step) —
        # the same quantity `_safe_solve` is asked to zero. Checked directly,
        # not only via the step size below: Student-t's Fisher weight
        # (ν+1)/((ν+3)σ²) is a CONSTANT, blind to the residual, so on an
        # ill-scaled site `A = Λ'WΛ + I` can be large enough that a tiny
        # Newton step `Δ` solves `AΔ = g` even while `g` itself is still far
        # from zero — measured on the #503 stress probe: 25/500 sites hit
        # `maximum(abs, Δ) < tol` this way while sitting up to 13.7
        # log-posterior units from the true mode. A step-size-only test (the
        # rule every OTHER grouped kernel uses, since their curvature tracks
        # the response) is not sufficient here; requiring the gradient itself
        # to be small is what catches these.
        g  = Λ' * s .- z
        A  = Symmetric(Λ' * (W .* Λ) + I)
        Δ  = _safe_solve(A, g)
        (Δ === nothing || !all(isfinite, Δ)) && return z, false
        maximum(abs, Δ) < tol && maximum(abs, g) < grad_tol && return z .+ Δ, true
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

# Per-site Laplace log-marginal with per-species Student-t markers `fams` (each
# entry shares ν, differs only in σ). PD guard mirrors the generic single-family
# core (`families/laplace.jl`'s `laplace_loglik_site`): the observed Student-t
# curvature is genuinely negative for |r| > σ√ν (documented above at
# `_glm_obs_weight`), so `A` is not SPD by construction here the way it is for
# Gamma/log — the guard is load-bearing, not defensive.
function _studentt_grouped_loglik_site(fams::AbstractVector{<:StudentTFamily}, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, hessian::Symbol = :observed,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    z, ok = _studentt_grouped_mode(fams, y, n, Λ, β, link;
                                   mask = mask, offset = offset, maxiter = maxiter, tol = tol)
    # Fallback (measured, mirrors #479's Gamma fallback in spirit though not in
    # mechanism): damped Fisher scoring is only linearly convergent here (no
    # PSD observed-curvature Newton fallback exists for Student-t, unlike
    # Gamma/log), so a genuinely healthy site can still need more than the
    # default `maxiter = 100` steps to reach `tol`. Measured on the #503 stress
    # probe: a site with gradient norm <= 1e-4 at its own warm start (i.e. one
    # a from-scratch restart confirms is already near the true mode) needed up
    # to several hundred damped-Fisher steps to certify convergence at
    # `tol = 1e-9`. Retrying with a 20x iteration budget (still Fisher-scored,
    # still damped, restarting from z = 0) resolves every such site in the
    # stress probe; it runs only where the first pass failed, so every site
    # that converged within the default budget keeps its original path and
    # iteration count.
    ok || ((z, ok) = _studentt_grouped_mode(fams, y, n, Λ, β, link;
                                            mask = mask, offset = offset,
                                            maxiter = 20 * maxiter, tol = tol))
    # A search that did not converge must not produce a finite value. -Inf makes the
    # fitters' objective return its 1e12 sentinel instead of a garbage surface (#479
    # precedent). The nu-boundary guard in `fit_studentt_gllvm` is a separate,
    # fit-level check (an ESTIMATED ν running to the flat Gaussian-limit boundary)
    # and is untouched by this change.
    ok || return -Inf
    η  = _clamp_eta.(β .+ off .+ Λ * z)
    μ  = _clamp_mu.(fams, linkinv.(Ref(link), η))
    me = mu_eta.(Ref(link), η)
    W  = _studentt_grouped_laplace_weight.(Ref(hessian), fams, μ, me, y, Ref(link), η)
    if mask !== nothing
        W = ifelse.(mask, W, 0.0)
    end
    A  = Symmetric(Λ' * (W .* Λ) + I)
    ℓ = 0.0
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _glm_logpdf(fams[t], μ[t], n[t], y[t])
    end
    if any(w -> w < zero(w), W)
        F = cholesky(A; check = false)
        issuccess(F) || return oftype(ℓ, -Inf)
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

# ---------------------------------------------------------------------------
# Fit driver.
# ---------------------------------------------------------------------------

"""
    StudentTFit

Result of [`fit_studentt_gllvm`](@ref): intercepts `β` (length p), loadings `Λ`
(p×K), degrees of freedom `ν` (`Float64` or length-p `Vector{Float64}`), whether
`ν` was estimated (`estimated_nu`), and estimated scale `σ`
(`(y − η)/σ ~ t_ν`; a `Float64` under `disp_group == :shared`, or a length-p `Vector{Float64}`
under `disp_group == :species`), the `link` (always `IdentityLink()`), the
maximised Laplace `loglik`, the optimiser `converged` flag, `iterations`, the
`hessian` curvature selector, and `disp_group` (`:shared` default or
`:species`).
"""
struct StudentTFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    ν::Union{Float64, Vector{Float64}}
    σ::Union{Float64, Vector{Float64}}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    disp_group::Symbol
    estimated_nu::Bool
    # Flat Gaussian-limit boundary honesty (panel 2026-09-01): true when an
    # ESTIMATED ν reached the ν→∞ boundary (any ν > 1e6, the same rule the
    # parity fixture diagnoses). Additive; `converged` semantics unchanged.
    nu_boundary::Bool
end

_studentt_nu_boundary(estimated::Bool, ν) =
    estimated && any(>(1e6), ν isa Real ? (ν,) : ν)

# Positional compatibility constructors (2026-08-28): every pre-existing
# construction site builds a default-curvature, shared-dispersion fit; the
# `hessian` field records the objective identity so `confint`/bootstrap can
# rebuild THE SAME objective instead of guessing (the audit's
# confint-consistency class); `disp_group` mirrors the delta fitters'
# precedent (`DeltaLogNormalFit`).
StudentTFit(β, Λ, ν, σ, link, loglik, converged, iterations) =
    StudentTFit(β, Λ, ν, σ, link, loglik, converged, iterations,
               _default_hessian(StudentTFamily(4.0, 1.0), link), :shared, false)
StudentTFit(β, Λ, ν, σ, link, loglik, converged, iterations, hessian::Symbol) =
    StudentTFit(β, Λ, ν, σ, link, loglik, converged, iterations, hessian, :shared, false)
StudentTFit(β, Λ, ν, σ, link, loglik, converged, iterations, hessian::Symbol,
            disp_group::Symbol) =
    StudentTFit(β, Λ, ν, σ, link, loglik, converged, iterations, hessian,
                disp_group, false)
# 11-positional compatibility (pre-nu_boundary sites): derive the flag.
StudentTFit(β, Λ, ν, σ, link, loglik, converged, iterations, hessian::Symbol,
            disp_group::Symbol, estimated_nu::Bool) =
    StudentTFit(β, Λ, ν, σ, link, loglik, converged, iterations, hessian,
                disp_group, estimated_nu, _studentt_nu_boundary(estimated_nu, ν))

function Base.show(io::IO, f::StudentTFit)
    p, K = size(f.Λ)
    σstr = f.σ isa Real ? string(round(f.σ; sigdigits = 4)) : "per-trait"
    νstr = f.ν isa Real ? string(round(f.ν; sigdigits = 4)) : "per-trait"
    print(io, "StudentTFit(p=", p, ", K=", K, ", ν=", νstr,
          f.estimated_nu ? " (estimated)" : " (fixed)",
          ", σ=", σstr,
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.nu_boundary ? ", ν at Gaussian-limit boundary" : "",
          f.converged ? "" : ", NOT CONVERGED", ")")
end

# Student-t/identity Laplace log-det defaults to the OBSERVED conditional
# curvature (decision A, 2026-08-27 — campaign: estimator preference 100% in
# every regime; reported-loglik cost accepted). With r = y − μ (identity link,
# me = 1): −∂²ℓ/∂η² = (ν+1)(νσ² − r²)/(νσ² + r²)². GENUINELY NEGATIVE for
# |r| > σ√ν — the assembly-level PD guard in `marginal_loglik_laplace` handles
# that (measured, load-bearing; never clamp here). No analytic-gradient
# coupling: this fitter is finite-difference only.
function _glm_obs_weight(f::StudentTFamily, μ, n, me, y, link::IdentityLink, η)
    r = y - μ
    νσ² = f.ν * f.σ^2
    return (f.ν + 1) * (νσ² - r^2) / (νσ² + r^2)^2
end
_default_hessian(::StudentTFamily, ::IdentityLink) = :observed

"""
    fit_studentt_gllvm(Y; K, nu=nothing, link=IdentityLink(), σ_init=nothing, nu_init=nothing, …) -> StudentTFit

Fit a Student-t GLLVM by L-BFGS over `[β; vec(Λ); log σ; log(ν-1)]` on the Laplace
marginal (`studentt_marginal_loglik_laplace`). When `nu === nothing` (default),
the degrees of freedom `ν` are estimated jointly per-trait (`ν_j = 1 + exp(θ_{ν,j}) > 1`),
matching `gllvmTMB`. A finite positive number (e.g. `nu = 4.0`) or a length-`p`
vector of finite positive values fixes `nu`. Nonfinite values are rejected before
reading responses. Fixed values `0 < nu <= 1` are a Julia extension; the frozen
R 0.7.0 constructor requires `df > 1`. The Gaussian limit does not admit `nu = Inf`.
`Y` is a p×n response matrix; `K` the latent dimension.

Initial values: `σ₀ = 1.0`, `ν₀ = 3.0` (`log(ν₀ - 1) = log(2.0)`), matching gllvmTMB.
`hessian` selects the Laplace log-det curvature (`:observed` default / `:fisher`).
`disp_group` selects `:species` (per-trait dispersion) or `:shared`.
"""
function fit_studentt_gllvm(Y::AbstractMatrix{<:Real}; K::Integer,
        nu::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        link::Link = IdentityLink(),
        hessian::Symbol = _default_hessian(StudentTFamily(4.0, 1.0), link),
        disp_group::Symbol = :shared,
        β_init = nothing, Λ_init = nothing, σ_init = nothing, nu_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    if nu !== nothing
        if nu isa Real
            isfinite(nu) && nu > 0 || throw(ArgumentError("Student-t degrees of freedom nu must be finite and > 0; got $nu"))
        else
            length(nu) == p || throw(ArgumentError("length(nu)=$(length(nu)) must equal p=$p"))
            all(x -> isfinite(x) && x > 0, nu) || throw(ArgumentError("All Student-t degrees of freedom nu must be finite and > 0"))
        end
    end
    hessian in (:fisher, :observed) || throw(ArgumentError(
        "fit_studentt_gllvm: hessian must be :fisher or :observed; got :$hessian"))
    disp_group in (:shared, :species) || throw(ArgumentError(
        "fit_studentt_gllvm: disp_group must be :shared or :species; got :$disp_group"))
    rr = rr_theta_len(p, K)

    Zemp = float.(Y)                                   # identity link ⇒ Z = Y
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
    # Initial values: σ_0 = 1.0, ν_0 = 3.0 (matching gllvmTMB).
    σ0 = σ_init === nothing ? 1.0 : float(σ_init)
    σ0vec = σ_init === nothing ? fill(1.0, p) : (σ_init isa AbstractVector ? float.(σ_init) : fill(float(σ_init), p))
    ndisp = disp_group === :shared ? 1 : p
    logσ0 = disp_group === :shared ? [log(σ0)] : log.(σ0vec)

    log_nu_minus_1_0 = if nu === nothing
        if nu_init === nothing
            fill(log(2.0), ndisp)
        else
            if disp_group === :shared
                [log(float(nu_init) - 1.0)]
            else
                nu_init isa AbstractVector ? log.(float.(nu_init) .- 1.0) : fill(log(float(nu_init) - 1.0), p)
            end
        end
    else
        Float64[]
    end
    ν_fixed = nu === nothing ? nothing : (nu isa Real ? float(nu) : float.(nu))

    θ0 = if nu === nothing
        vcat(β0, pack_lambda(Λ0), logσ0, log_nu_minus_1_0)
    else
        vcat(β0, pack_lambda(Λ0), logσ0)
    end

    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        σ = disp_group === :shared ? exp(θ[p + rr + 1]) : exp.(θ[(p + rr + 1):(p + rr + ndisp)])
        ν = if nu === nothing
            disp_group === :shared ? (1.0 + exp(θ[p + rr + ndisp + 1])) : (1.0 .+ exp.(θ[(p + rr + ndisp + 1):(p + rr + 2 * ndisp)]))
        else
            ν_fixed
        end
        v = try
            -studentt_marginal_loglik_laplace(Y, Λ, β, σ; ν = ν, link = link,
                                              hessian = hessian,
                                              maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    # An estimated ν can run past an interior optimum to the flat ν→∞ limit: the
    # per-trait ν profile can have an interior peak and a lower rise towards the
    # Gaussian limit, and L-BFGS from ν₀ = 3 may land on the wrong side (near-Gaussian
    # parity diagnostic, Julia 1.13 draw: ν₁ → 5e9 at logLik −1430.162, while the
    # interior optimum ν₁ = 17.7 has −1430.097). So when any estimated ν reaches the
    # boundary, restart those traits warm from ν = 20 and ν = 50 and keep the best
    # optimum. Fits whose ν stays finite are untouched.
    if nu === nothing
        iν = (p + rr + ndisp + 1):(p + rr + 2 * ndisp)
        θb = Optim.minimizer(res)
        at_bound = findall(>(1e6), 1.0 .+ exp.(θb[iν]))
        for ν_r in (isempty(at_bound) ? () : (20.0, 50.0))
            θr = copy(θb)
            θr[iν[at_bound]] .= log(ν_r - 1.0)
            res_r = Optim.optimize(negll, θr, ls, opts; autodiff = :finite)
            Optim.minimum(res_r) < Optim.minimum(res) - 1e-8 && (res = res_r)
        end
    end
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    σ̂ = disp_group === :shared ? exp(θ̂[p + rr + 1]) : exp.(θ̂[(p + rr + 1):(p + rr + ndisp)])
    ν̂ = if nu === nothing
        disp_group === :shared ? (1.0 + exp(θ̂[p + rr + ndisp + 1])) : (1.0 .+ exp.(θ̂[(p + rr + ndisp + 1):(p + rr + 2 * ndisp)]))
    else
        ν_fixed
    end
    estimated = nu === nothing
    boundary = _studentt_nu_boundary(estimated, ν̂)
    boundary && @warn "Student-t estimated ν reached the flat Gaussian-limit boundary (ν > 1e6); the Student model is not distinguishable from Gaussian on this data, and optimizer convergence flags are unreliable at this boundary. Consider a fixed ν or the Gaussian family." maxlog=1
    loglik, conv, iters = _fit_verdict(res)
    # Boundary honesty wiring (mirrors `_tweedie_verdict`'s `:power_at_boundary` rule,
    # families/tweedie.jl: a family parameter that has run to the edge of its domain
    # forces `converged = false` regardless of what Optim itself reported — Optim
    # cannot tell a stationary point from a flat plateau by gradient alone). Forcing
    # `converged` here — rather than leaving it `true` alongside `nu_boundary` — is
    # what makes the fit fail the GENERIC `sanity_multi`/`gllvmTMB_diagnose` gate
    # (diagnostics.jl), which reads `fit.converged` and nothing family-specific: no
    # diagnostics.jl edit needed, and no existing gate is weakened, only tightened.
    conv = conv && !boundary
    return StudentTFit(β̂, Λ̂, ν̂, σ̂, link, loglik, conv, iters, hessian,
                       disp_group, estimated, boundary)
end
