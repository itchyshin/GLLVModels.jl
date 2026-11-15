# Shared helper for the phylo x NB2 and phylo x Binomial B_eta_realized canaries (refs #1048).
#
# The phylogenetic variance is not identified in these designs (free species intercepts absorb it, and
# the canary data draw no phylogenetic effect), so the fit may run sigma2_phy to the floating-point
# floor (about 8e-308). There `Q ./ sigma2` overflows, the objective returns its 1e12 failure value and
# the optimiser stops with `stop_reason == :gradient_at_failure_value` although the likelihood is flat.
# A canary may therefore accept that stop as a boundary fit, and only when ALL five conditions hold
# (no source clamp; the sentinel label itself is a separate follow-up):
#   (i)   the stop reason is the failure-value stop;
#   (ii)  sigma2_phy is at or below the floor `_XLV_SIGMA2_FLOOR`;
#   (iii) the final log-likelihood is finite and at least the value at the start;
#   (iv)  the other packed parameters are finite and their (plain-model) gradient is small;
#   (v)   the log-likelihood matches a reference fit of the model with sigma2_phy fixed at 0 (the plain
#         predictor-informed LV model), within `tol_ref`.
# The reference fit and the gradient are only computed once (i) and (ii) hold, so the branch costs
# nothing when sigma2_phy is off the floor.
using GLLVModels

const _XLV_SIGMA2_FLOOR = 1e-300

function _xlv_boundary_conditions(fit; nll_plain::Function, theta_start_plain::AbstractVector,
        tol_grad::Real = 1e-2, tol_ref::Real = 5e-3)
    stop_failure = fit.stop_reason === :gradient_at_failure_value
    at_floor = isfinite(fit.sigma2_phy) && fit.sigma2_phy <= _XLV_SIGMA2_FLOOR
    if !(stop_failure && at_floor)
        return (stop_failure = stop_failure, at_floor = at_floor, best_seen = false,
                others_ok = false, matches_ref = false)
    end
    theta_plain = collect(fit.theta_packed[1:(end - 1)])      # drop log sigma2 (last coordinate)
    ll_start = -nll_plain(collect(theta_start_plain))
    best_seen = isfinite(fit.loglik) && fit.loglik >= ll_start - 1e-8
    g = map(eachindex(theta_plain)) do i
        h = 1e-5 * max(1.0, abs(theta_plain[i]))
        tp = copy(theta_plain); tp[i] += h
        tm = copy(theta_plain); tm[i] -= h
        (nll_plain(tp) - nll_plain(tm)) / (2h)
    end
    others_ok = all(isfinite, theta_plain) && all(isfinite, g) && maximum(abs, g) <= tol_grad
    res = GLLVModels.Optim.optimize(nll_plain, theta_plain, GLLVModels.Optim.LBFGS(),
        GLLVModels.Optim.Options(g_tol = 1e-6, iterations = 3000); autodiff = :finite)
    ll_ref = -GLLVModels.Optim.minimum(res)
    matches_ref = isfinite(ll_ref) && abs(fit.loglik - ll_ref) <= tol_ref
    return (stop_failure = stop_failure, at_floor = at_floor, best_seen = best_seen,
            others_ok = others_ok, matches_ref = matches_ref)
end

"""
    _xlv_converged_or_boundary(fit; nll_plain, theta_start_plain) -> Bool

`fit.converged`, or the boundary branch described at the top of this file.
"""
function _xlv_converged_or_boundary(fit; kwargs...)
    fit.converged && return true
    return all(values(_xlv_boundary_conditions(fit; kwargs...)))
end
