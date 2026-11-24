# Post-fit convergence contract for `fit_gaussian_gllvm` when a phylogenetic
# loading block is present (`K_phy > 0`). Covers #1163 (#505 class): Optim can
# report convergence at `Λ_phy ≈ 0` with the same marginal likelihood as the
# no-phylo-block model while a better interior point exists.

const _GAUSSIAN_NEWTON_SE_TOL = 1e-3
const _GAUSSIAN_LAMBDA_PHY_ZERO_TOL = 1e-6

function _gaussian_newton_se_max(fit::GllvmFit, y,
                                 X::Union{Nothing, AbstractArray{<:Real, 3}},
                                 Σ_phy::Union{Nothing, AbstractMatrix})
    θ̂ = fit.pars.θ_packed
    nll = _confint_reconstruct_nll(fit, y, X, Σ_phy)
    g = ForwardDiff.gradient(nll, θ̂)
    all(isfinite, g) || return Inf
    H = try
        ForwardDiff.hessian(nll, θ̂)
    catch err
        err isa InterruptException && rethrow()
        return Inf
    end
    all(isfinite, H) || return Inf
    pins = _lambda_constraint_pinned_theta_indices(fit)
    free = setdiff(1:length(θ̂), pins)
    isempty(free) && return 0.0
    _, se_free, pd = _gaussian_wald_covariance_from_hessian(H[free, free], length(free))
    pd || return Inf
    F = cholesky(Symmetric(H[free, free]); check = true)
    step = F \ g[free]
    m = 0.0
    @inbounds for (r, j) in enumerate(free)
        se = se_free[r]
        (isfinite(se) && se > 0) || return Inf
        m = max(m, abs(step[r]) / se)
    end
    return m
end

"""
    _gaussian_gllvm_convergence_verdict(model, pars, y, X, Σ_phy, res, g_tol)
        -> (converged::Bool, stopping_reason::Symbol)

Shinichi convergence rule for Gaussian GLLVM fits with `K_phy > 0`. Failure
reasons are chosen in order: `:not_converged`, `:boundary`, `:degenerate`
(first match wins).
"""
function _gaussian_gllvm_convergence_verdict(model::GllvmModel, pars::NamedTuple,
        y::AbstractMatrix, X, Σ_phy, res, g_tol::Real)
    loglik, _, _ = _fit_verdict(res)
    isfinite(loglik) || return (false, :not_converged)
    optim_converged = Optim.converged(res)
    gres = Optim.g_residual(res)
    if !optim_converged || !isfinite(gres) || gres > g_tol
        return (false, :not_converged)
    end
    Λ_phy = pars.Λ_phy
    if Λ_phy !== nothing && norm(Λ_phy) < _GAUSSIAN_LAMBDA_PHY_ZERO_TOL
        return (false, :boundary)
    end
    fit_stub = GllvmFit(model, pars, loglik, Optim.iterations(res), false, res,
                        0.0, nothing, nothing, :unknown)
    _, _, _, _, se_all, pd = _confint_gaussian_wald_covariance(fit_stub, y, X, Σ_phy)
    if !pd || any(!isfinite, se_all)
        return (false, :degenerate)
    end
    newton_max = _gaussian_newton_se_max(fit_stub, y, X, Σ_phy)
    if !isfinite(newton_max) || newton_max > _GAUSSIAN_NEWTON_SE_TOL
        return (false, :not_converged)
    end
    return (true, :converged)
end