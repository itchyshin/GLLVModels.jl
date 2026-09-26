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
#
# THE SECOND DEFECT (#485). `Optim.converged(res) = x_converged || f_converged ||
# g_converged`, and this package's callers all leave `x_abstol = x_reltol = f_abstol =
# f_reltol = 0.0` (Optim 1.13.3 defaults). A single zero-length line-search step then
# trips `x_converged`/`f_converged` trivially — `norm(x - x_previous) <= 0` is exactly
# true when the step underflowed — even while the gradient residual sits orders of
# magnitude above `g_tol`. Measured on `fit_nb1_gllvm_grouped` (no-X, 3/3 seeded
# datasets): `converged = true` with Optim gradient residuals of 5.68, 2.14e2 and 1.79e1
# against `g_tol = 1e-5`. `_fit_verdict(res)` now additionally requires the gradient
# criterion, scale-aware as `_tweedie_verdict` (families/tweedie.jl) and the Beta
# grouped verdict (families/grouped_dispersion.jl, `_beta_grouped_g_met`, #480/#483)
# already judge it: `gres <= max(g_tol, g_tol * |nll|)`. A caller's `g_tol` set below
# the finite-difference noise floor still passes at a genuine stationary point, and a
# `x`/`f`-only stall at a large gradient no longer reports `converged = true`.

const _NLL_SENTINEL = 1e12

# A decade below the sentinel and far above any reachable real negative log-likelihood.
# Matches `_FD_FAIL_THRESHOLD` (confint_family.jl:1865) so one threshold covers both the
# fitting and the interval machinery. Also catches near-plateau stalls and clamped values
# well above 1e11 that are equally not log-likelihoods.
const _NLL_FAIL_THRESHOLD = 1e11

_nll_failed(nll::Real) = !isfinite(nll) || nll >= _NLL_FAIL_THRESHOLD

# Scale-aware gradient criterion, the same rule `_tweedie_verdict` and the Beta grouped
# verdict (`_beta_grouped_g_met`) already use: judge the residual against `g_tol` scaled
# by the objective's own size, not `g_tol` alone.
_gradient_criterion_met(res) =
    (gres = Optim.g_residual(res); g_tol = Optim.g_tol(res);
     isfinite(gres) && gres <= max(g_tol, g_tol * abs(Optim.minimum(res))))

"""
    _fit_verdict(res) -> (loglik, converged, iterations)
    _fit_verdict(nll, converged, iterations) -> (loglik, converged, iterations)

Convert an `Optim` result into user-visible fields, refusing to report a failure sentinel
as a log-likelihood. A run that ended on the penalty plateau returns
`(-Inf, false, iterations)` regardless of what `Optim.converged` claims — the optimiser
cannot tell the difference, because the gradient of a constant is exactly zero.
`converged` also requires the scale-aware gradient criterion
(`_gradient_criterion_met`): `Optim.converged` alone can fire on a zero-length
x/f line-search step whose gradient residual is still far above `g_tol` (#485).

The three-argument form is for fitters that construct their result without an `Optim`
object (an `iterations = 0` early return, or a grid search keeping a best record); it
has no `Optim` result to judge the gradient from, so it keeps the caller's `converged`
as given.
"""
function _fit_verdict(res)
    conv = Optim.converged(res) && _gradient_criterion_met(res)
    return _fit_verdict(Optim.minimum(res), conv, Optim.iterations(res))
end

function _fit_verdict(nll::Real, converged::Bool, iterations::Integer)
    _nll_failed(nll) && return (-Inf, false, Int(iterations))
    return (-Float64(nll), converged, Int(iterations))
end
