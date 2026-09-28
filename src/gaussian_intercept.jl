# Per-trait intercepts for the public Gaussian route.
#
# `fit_gaussian_gllvm(Y; K)` documents `X = nothing` as a zero mean, and the R
# bridge relies on that by centring Y itself. The public entry point
# `fit_gllvm(Y; family = Normal(), K)` must instead estimate one intercept per
# trait, as every other family on that entry point does and as gllvmTMB's
# `value ~ 0 + trait + latent(...)` does. This file supplies that route and the
# tag that lets post-fit helpers apply the intercepts when `X` is omitted.

# X[t, s, t] = 1: one intercept column per trait, identical at every site.
function _trait_intercept_design(p::Integer, n::Integer)
    X = zeros(Float64, p, n, p)
    @inbounds for t in 1:p, s in 1:n
        X[t, s, t] = 1.0
    end
    return X
end

# X[t, s, 1] = 1: one intercept shared by every trait. Phylogenetic fits use
# this, because the phylogenetic effect J_n ⊗ B is constant across sites within
# a species and a free per-species intercept would absorb it.
_common_intercept_design(p::Integer, n::Integer) = ones(Float64, p, n, 1)

_has_trait_intercepts(fit::GllvmFit) =
    get(fit.pars, :mean_design, nothing) === :trait_intercepts

_has_common_intercept(fit::GllvmFit) =
    get(fit.pars, :mean_design, nothing) === :common_intercept

# True when the fit estimated an intercept design without a caller-supplied X.
_has_intercept_design(fit::GllvmFit) = _has_trait_intercepts(fit) || _has_common_intercept(fit)

# The intercept design of such a fit at `n` sites.
_intercept_design(fit::GllvmFit, n::Integer) =
    _has_common_intercept(fit) ? _common_intercept_design(fit.model.p, n) :
                                 _trait_intercept_design(fit.model.p, n)

# The per-trait intercept values (length p) of such a fit.
_intercept_mean(fit::GllvmFit) =
    _has_common_intercept(fit) ? fill(fit.pars.β[1], fit.model.p) : fit.pars.β

# The mean design a post-fit routine should use: the caller's `X` when given,
# else the intercept design for a fit that estimated intercepts (`n` sites),
# else `nothing` (zero mean).
_mean_X(fit::GllvmFit, X, n) =
    X === nothing && n !== nothing && _has_intercept_design(fit) ?
        _intercept_design(fit, n) : X

# _fit_gaussian_trait_intercepts(Y; K, kwargs...) -> GllvmFit
#
# Gaussian GLLVM with one estimated intercept per trait: the model behind
# `fit_gllvm(Y; family = Normal(), K)` when no `X` is supplied. It calls
# `fit_gaussian_gllvm` with the design `X[t, s, t] = 1`, so the
# intercepts `fit.pars.β` (length `p`) are optimised jointly with `Λ` and `σ_eps`
# and are counted by `nobs`/`aic`/`bic`. With complete data they equal the trait
# means of `Y`.
#
# The fit records `pars.mean_design = :trait_intercepts`, so `getLV`, `predict`,
# `residuals` and `simulate` apply the intercepts when called without `X`.
# Supplying `X` defines the complete mean instead (no intercept is added) and is
# passed through unchanged. `lambda_constraint` still requires a zero-mean fit
# and is passed through unchanged. A phylogenetic fit (`Σ_phy` supplied) gets one
# common intercept instead (`pars.mean_design = :common_intercept`, `β` length 1).
function _fit_gaussian_trait_intercepts(Y::AbstractMatrix; K::Integer, kwargs...)
    if get(kwargs, :X, nothing) !== nothing || get(kwargs, :lambda_constraint, nothing) !== nothing
        return fit_gaussian_gllvm(Y; K = K, kwargs...)
    end
    p, n = size(Y)
    rest = Base.structdiff((; kwargs...), NamedTuple{(:X,)})
    # The phylogenetic effect J_n ⊗ B is constant across sites within a species,
    # so a free per-species intercept would absorb it and drive σ_phy to zero.
    common = get(kwargs, :Σ_phy, nothing) !== nothing
    X = common ? _common_intercept_design(p, n) : _trait_intercept_design(p, n)
    fit = fit_gaussian_gllvm(Y; K = K, X = X, rest...)
    pars = merge(fit.pars, (mean_design = common ? :common_intercept : :trait_intercepts,))
    return GllvmFit(fit.model, pars, fit.logLik, fit.n_iter, fit.converged,
                    fit.optim_result, fit.cputime, fit.integration)
end
