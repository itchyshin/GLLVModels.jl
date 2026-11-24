# Screen Optim's raw result before it becomes user-visible fields.
#
# THE DEFECT THIS EXISTS TO PREVENT. Objective closures across this package return a large
# FINITE sentinel when they cannot be evaluated (`catch; return 1e12`, or
# `isfinite(v) ? v : 1e12`). That is correct *inside* the closure — it is a barrier keeping
# the line search out of a bad region. The defect is what happens next:
# `Optim.minimum(res)` carries the sentinel out, the finite-difference gradient of the flat
# plateau is exactly zero so `Optim.converged(res)` fires `g_converged` at iteration 0, and
# the constructor stores `-1.0e12` as `loglik` with `converged = true`. `Base.show` gates
# its NOT-CONVERGED tag on `converged == false`, so it prints as a clean fit, and
# `aic`/`bic` turn it into a finite ~2e12 criterion.
#
# Measured before this helper existed:
#   * one zero cell in Gamma data       → converged=true, loglik=-1.0e12, aic=2.0e12
#   * NB grouped_cov with a non-LogLink → converged=true, loglik=-1.0e12, iterations=0,
#     100% of calls under the documented default `hessian = :observed` — and the package's
#     own `ArgumentError("hessian=:observed is currently supported only for NB2 with
#     LogLink()")` was being swallowed to produce it.
#
# WHY ONE SHARED HELPER rather than a verdict function per fitter. `_tweedie_verdict`
# (families/tweedie.jl:186) and `_phylo_verdict` (fit_phylo.jl:92) are family-specific
# because they carry extra tests — Tweedie's ξ-boundary and gradient-scale checks, for
# instance. These sites need only one question: "is this the plateau?". Copying ninety
# hand-written verdict functions is the drift machine that produced this class in the first
# place.
#
# CONVENTION: failure reports `loglik = -Inf` and `converged = false`, matching
# `_phylo_verdict`. `-Inf` cannot masquerade as a finite AIC and it trips every downstream
# `isfinite` check — including existing test assertions that currently pass on −1e12.

const _NLL_SENTINEL = 1e12

# A decade below the sentinel and far above any reachable real negative log-likelihood.
# Matches `_FD_FAIL_THRESHOLD` (confint_family.jl:1865) so one threshold covers both the
# fitting and the interval machinery. Also catches near-plateau stalls and clamped values
# well above 1e11 that are equally not log-likelihoods.
const _NLL_FAIL_THRESHOLD = 1e11

_nll_failed(nll::Real) = !isfinite(nll) || nll >= _NLL_FAIL_THRESHOLD

"""Diagnostics used by the staged fit-convergence contract.

`objective` is maximised and equals `logLik + log_prior`; `logLik` is always
the unpenalised data objective. `pd_hessian` describes the objective used for
the fit, while `pd_hessian_unpenalised` records data-only curvature. A fit can
remain converged when only the penalised Hessian is positive definite; in that
case `prior_held_params` names the prior targets that supplied the missing
curvature.
"""
struct FitConvergence
    status::Symbol
    optimizer_finished::Bool
    relative_gradient::Float64
    raw_relative_gradient::Float64
    pd_hessian::Bool
    penalized::Bool
    objective::Float64
    logLik::Float64
    log_prior::Float64
    pd_hessian_unpenalised::Bool
    prior_held_params::Vector{String}
    se_type::Symbol
    boundary_check_available::Bool
    boundary_params::Vector{String}
end

const _GLLVM_CONVERGENCE_PROPERTIES = (:status, :optimizer_finished, :relative_gradient,
    :raw_relative_gradient, :pd_hessian, :penalized, :objective, :logLik, :log_prior,
    :pd_hessian_unpenalised, :prior_held_params, :se_type,
    :boundary_check_available, :boundary_params)

# Used only by legacy/approximate result constructors that cannot supply the
# optimizer, gradient, boundary scan, and Hessian evidence. Their convergence
# is therefore unavailable and must fail closed.
_fit_unavailable_convergence(logLik::Real) = FitConvergence(:degenerate, false,
    Inf, Inf, false, false, Float64(logLik), Float64(logLik), 0.0, false,
    String[], :ml, false, String[])

"""Project a minimization gradient onto lower/upper-bound KKT cones.

`lower` and `upper` contain one-based gradient indices at active bounds. For a
lower bound the valid score is nonnegative; for an upper bound it is
nonpositive. `lower_scores` and `upper_scores` optionally provide the score in
the natural constrained coordinate when the optimizer uses a transformation
(for example, a log-SD coordinate).
"""
function _fit_projected_gradient(gradient::AbstractVector; lower = (), upper = (),
        lower_scores = Dict(), upper_scores = Dict())
    projected = Float64.(gradient)
    lower_indices = Int.(collect(lower))
    upper_indices = Int.(collect(upper))
    all(i -> 1 <= i <= length(projected), vcat(lower_indices, upper_indices)) ||
        throw(ArgumentError("boundary gradient index is out of range"))
    isempty(intersect(lower_indices, upper_indices)) ||
        throw(ArgumentError("a boundary gradient index cannot be both lower and upper"))
    for (indices, scores, project) in ((lower_indices, lower_scores, min),
                                        (upper_indices, upper_scores, max))
        for index in indices
            score = Float64(get(scores, index, projected[index]))
            isfinite(score) || throw(ArgumentError("boundary score must be finite"))
            projected[index] = project(score, 0.0)
        end
    end
    return projected
end

"""Convert a log-coordinate score to a finite natural-positive-coordinate score."""
function _fit_natural_positive_score(log_score::Real, value::Real)
    isfinite(log_score) || return Float64(log_score)
    value > 0 || return copysign(floatmax(Float64), Float64(log_score))
    score = Float64(log_score) / Float64(value)
    return isfinite(score) ? score : copysign(floatmax(Float64), Float64(log_score))
end

"""Return whether a finite symmetric Hessian is positive definite."""
function _fit_hessian_pd(H)
    H === nothing && return false
    H isa AbstractMatrix || return false
    size(H, 1) == size(H, 2) || return false
    all(isfinite, H) || return false
    isapprox(H, H'; rtol = 1e-8, atol = 1e-10) || return false
    try
        cholesky(Symmetric((H + H') / 2); check = true)
        return true
    catch e
        e isa PosDefException || rethrow()
        return false
    end
end

"""Central finite-difference gradient for a scalar objective."""
function _fit_fd_gradient(objective, θ::AbstractVector; step::Float64 = 1e-5)
    g = Vector{Float64}(undef, length(θ))
    for j in eachindex(θ)
        h = step * max(1.0, abs(Float64(θ[j])))
        plus = Float64.(θ); minus = Float64.(θ)
        plus[j] += h; minus[j] -= h
        fp, fm = objective(plus), objective(minus)
        (fp isa Real && fm isa Real && isfinite(fp) && isfinite(fm)) || return fill(NaN, length(θ))
        g[j] = (fp - fm) / (2h)
    end
    return g
end

"""Finite-difference a gradient to obtain a symmetric Hessian."""
function _fit_fd_hessian_from_gradient(gradient, θ::AbstractVector; step::Float64 = 1e-4)
    d = length(θ)
    H = Matrix{Float64}(undef, d, d)
    for j in eachindex(θ)
        h = step * max(1.0, abs(Float64(θ[j])))
        plus = Float64.(θ); minus = Float64.(θ)
        plus[j] += h; minus[j] -= h
        gp, gm = gradient(plus), gradient(minus)
        (gp isa AbstractVector && gm isa AbstractVector && length(gp) == d &&
            length(gm) == d && all(isfinite, gp) && all(isfinite, gm)) ||
            return fill(NaN, d, d)
        H[:, j] .= (gp .- gm) ./ (2h)
    end
    return (H + H') / 2
end

function _fit_hessian_from_gradient(gradient, θ::AbstractVector; fd_step::Float64 = 1e-4)
    H = try
        ForwardDiff.jacobian(gradient, Float64.(θ))
    catch
        nothing
    end
    if H === nothing || size(H) != (length(θ), length(θ)) || !all(isfinite, H)
        H = _fit_fd_hessian_from_gradient(gradient, θ; step = fd_step)
    else
        H = (H + H') / 2
    end
    return H
end

function _fit_objective_gradient(objective, θ::AbstractVector)
    g = try
        ForwardDiff.gradient(objective, Float64.(θ))
    catch
        nothing
    end
    if g === nothing || length(g) != length(θ) || !all(isfinite, g)
        return _fit_fd_gradient(objective, θ)
    end
    return g
end

function _fit_fd_hessian(objective, θ::AbstractVector; step::Float64 = 1e-4)
    d = length(θ)
    H = Matrix{Float64}(undef, d, d)
    x = Float64.(θ)
    f0 = objective(x)
    f0 isa Real && isfinite(f0) || return fill(NaN, d, d)
    hs = step .* max.(1.0, abs.(x))
    for i in 1:d
        xp, xm = copy(x), copy(x)
        xp[i] += hs[i]; xm[i] -= hs[i]
        fp, fm = objective(xp), objective(xm)
        (fp isa Real && fm isa Real && isfinite(fp) && isfinite(fm)) || return fill(NaN, d, d)
        H[i, i] = (fp - 2 * f0 + fm) / hs[i]^2
        for j in (i + 1):d
            xpp, xpm, xmp, xmm = copy(x), copy(x), copy(x), copy(x)
            xpp[i] += hs[i]; xpp[j] += hs[j]
            xpm[i] += hs[i]; xpm[j] -= hs[j]
            xmp[i] -= hs[i]; xmp[j] += hs[j]
            xmm[i] -= hs[i]; xmm[j] -= hs[j]
            vals = (objective(xpp), objective(xpm), objective(xmp), objective(xmm))
            all(v -> v isa Real && isfinite(v), vals) || return fill(NaN, d, d)
            Hij = (vals[1] - vals[2] - vals[3] + vals[4]) / (4hs[i] * hs[j])
            H[i, j] = Hij
            H[j, i] = Hij
        end
    end
    return H
end

function _fit_objective_hessian(objective, θ::AbstractVector)
    H = try
        ForwardDiff.hessian(objective, Float64.(θ))
    catch
        nothing
    end
    if H !== nothing && size(H) == (length(θ), length(θ)) && all(isfinite, H) &&
       isapprox(H, H'; rtol = 1e-8, atol = 1e-10)
        return Matrix(Symmetric((H + H') / 2))
    end
    return _fit_fd_hessian(objective, θ)
end

"""Create one fit verdict from objective, gradient, curvature and boundary evidence.

This is the common gate for fitters that provide those quantities. Missing or
invalid curvature is classified as `:degenerate`; it is never silently treated
as a successful fit. `relative_gradient` is max-absolute gradient divided by
`max(1, abs(objective))`, using `projected_gradient` when supplied. The raw
optimizer-coordinate norm remains available as `raw_relative_gradient`.
"""
function _fit_convergence(; optimizer_finished::Bool, objective::Real, logLik::Real,
        log_prior::Real = 0.0, penalized::Bool = !iszero(log_prior),
        gradient, projected_gradient = nothing, hessian, hessian_unpenalised = hessian,
        prior_targets = String[], boundary_params = String[],
        boundary_check_available::Bool = false,
        relative_gradient_tol::Real = 1e-5)
    isfinite(relative_gradient_tol) && relative_gradient_tol > 0 ||
        throw(ArgumentError("relative_gradient_tol must be positive and finite"))
    grad = try Float64.(gradient) catch; Float64[] end
    projected = projected_gradient === nothing ? grad :
        try Float64.(projected_gradient) catch; Float64[] end
    finite = isfinite(objective) && isfinite(logLik) && isfinite(log_prior) &&
        !isempty(grad) && all(isfinite, grad) && length(projected) == length(grad) &&
        all(isfinite, projected)
    scale = max(1.0, abs(Float64(objective)))
    raw_relative_gradient = finite ? maximum(abs, grad) / scale : Inf
    relative_gradient = finite ? maximum(abs, projected) / scale : Inf
    pd = _fit_hessian_pd(hessian)
    pd0 = _fit_hessian_pd(hessian_unpenalised)
    held = pd && !pd0 && penalized ? String.(prior_targets) : String[]
    boundary = String.(boundary_params)
    status = if !finite
        :degenerate
    elseif !optimizer_finished || relative_gradient > relative_gradient_tol
        :not_converged
    elseif !isempty(boundary)
        :boundary
    elseif !boundary_check_available
        :degenerate
    elseif !pd && isempty(held)
        :degenerate
    else
        :converged
    end
    return FitConvergence(status, optimizer_finished, relative_gradient,
        raw_relative_gradient, pd, penalized,
        Float64(objective), Float64(logLik), Float64(log_prior), pd0, held,
        penalized ? :penalised : :ml, boundary_check_available, boundary)
end

"""
    _fit_verdict(res) -> (loglik, converged, iterations)
    _fit_verdict(nll, converged, iterations) -> (loglik, converged, iterations)

Convert an `Optim` result into user-visible fields, refusing to report a failure sentinel
as a log-likelihood. A run that ended on the penalty plateau returns
`(-Inf, false, iterations)` regardless of what `Optim.converged` claims — the optimiser
cannot tell the difference, because the gradient of a constant is exactly zero.

The three-argument form is for fitters that construct their result without an `Optim`
object (an `iterations = 0` early return, or a grid search keeping a best record).
"""
function _fit_verdict(res)
    return _fit_verdict(Optim.minimum(res), Optim.converged(res), Optim.iterations(res))
end

"""Evaluate the universal convergence contract when a fitter supplies its objective.

Routes that call `_fit_verdict(res)` without their objective can only retain the
legacy optimizer/sentinel check. New and migrated fitters should pass the exact
minimization objective here, plus their route-specific boundary evidence.
"""
function _fit_objective_verdict(res, objective; boundary_params = String[],
        boundary_check_available::Bool = false, relative_gradient_tol::Real = 1e-5)
    diagnostics = _fit_objective_diagnostics(Optim.minimizer(res), Optim.converged(res),
        Optim.iterations(res), objective; boundary_params = boundary_params,
        boundary_check_available = boundary_check_available,
        relative_gradient_tol = relative_gradient_tol)
    return diagnostics.logLik, diagnostics.status === :converged, Optim.iterations(res)
end

function _fit_objective_diagnostics(θ̂::AbstractVector, optimizer_finished::Bool,
        iterations::Integer, objective; boundary_params = String[],
        boundary_check_available::Bool = false, relative_gradient_tol::Real = 1e-5)
    nll = try objective(θ̂) catch; Inf end
    failed = !(nll isa Real) || _nll_failed(nll)
    gradient = failed ? fill(NaN, length(θ̂)) : _fit_objective_gradient(objective, θ̂)
    hessian = failed ? fill(NaN, length(θ̂), length(θ̂)) :
        _fit_objective_hessian(objective, θ̂)
    ll = failed ? -Inf : -Float64(nll)
    diagnostics = _fit_convergence(optimizer_finished = optimizer_finished,
        objective = ll, logLik = ll, gradient = gradient, hessian = hessian,
        boundary_params = boundary_params,
        boundary_check_available = boundary_check_available,
        relative_gradient_tol = relative_gradient_tol)
    return diagnostics
end

function _fit_objective_verdict(θ̂::AbstractVector, optimizer_finished::Bool,
        iterations::Integer, objective; boundary_params = String[],
        boundary_check_available::Bool = false, relative_gradient_tol::Real = 1e-5)
    diagnostics = _fit_objective_diagnostics(θ̂, optimizer_finished, iterations, objective;
        boundary_params = boundary_params, boundary_check_available = boundary_check_available,
        relative_gradient_tol = relative_gradient_tol)
    return diagnostics.logLik, diagnostics.status === :converged, Int(iterations)
end

function _fit_verdict(nll::Real, converged::Bool, iterations::Integer)
    _nll_failed(nll) && return (-Inf, false, Int(iterations))
    return (-Float64(nll), converged, Int(iterations))
end
