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

function _fit_verdict(nll::Real, converged::Bool, iterations::Integer)
    _nll_failed(nll) && return (-Inf, false, Int(iterations))
    return (-Float64(nll), converged, Int(iterations))
end
