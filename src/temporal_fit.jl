# Fitter for the temporal source (slice 1): the separate door
# `fit_temporal_gllvm`. No edit to formula.jl (spec section 3.5, step 1).

"""
    TemporalGaussianFit

Result of [`fit_temporal_gllvm`](@ref). Fields:

- `beta`, `coefficient_names`: fixed-effect coefficients of the mean formula;
- `sigma_eps`: residual SD;
- `time_parameter` (`:phi` for AR1, `:ou_rate` for OU) and `time_value`
  (`phi = (1 - 1e-6) tanh(theta)` or `kappa = exp(theta)`);
- `loadings`: `Lambda_temporal` (`p × rank`; raw, no sign flip) for `:dep`
  and `:latent`, else `nothing`; `psi`: temporal diagonal variances
  `exp(2 theta_diag)` for `:indep` and `:latent` with `unique = true`, else
  `nothing`; `Sigma_T`: the trait covariance at one state;
- `parameters`, `parameter_names`: the optimizer vector in gllvmTMB's
  `opt\$par` order and R's `names(opt\$par)`;
- `loglik`, `converged`, `gradient_norm`, `hessian_min_eigenvalue`,
  `hessian_positive_definite`, `iterations`, `stopping_reason`;
- `term`, `spec`, `y`, `X`, `data`, `formula`, `trait`: what a refit needs.

`converged` requires the optimizer verdict and a fresh gradient check. A
converged fit does not establish identification: an unreplicated
`temporal_indep` fit separates `psi` from `sigma_eps^2` only through the
temporal correlation, as in gllvmTMB.
"""
struct TemporalGaussianFit <: StatsAPI.StatisticalModel
    beta::Vector{Float64}
    coefficient_names::Vector{String}
    sigma_eps::Float64
    time_parameter::Symbol
    time_value::Float64
    loadings::Union{Nothing,Matrix{Float64}}
    psi::Union{Nothing,Vector{Float64}}
    Sigma_T::Matrix{Float64}
    parameters::Vector{Float64}
    parameter_names::Vector{String}
    loglik::Float64
    converged::Bool
    gradient_norm::Float64
    hessian_min_eigenvalue::Float64
    hessian_positive_definite::Bool
    iterations::Int
    stopping_reason::Symbol
    term::TemporalTerm
    spec::TemporalSpec
    y::Vector{Float64}
    X::Matrix{Float64}
    data::NamedTuple
    formula::Any
    trait::Symbol
    g_tol::Float64
    max_iterations::Int
end

function _temporal_design(formula, cols::NamedTuple)
    formula isa StatsModels.FormulaTerm ||
        throw(ArgumentError("formula must be a StatsModels formula such as @formula(value ~ 0 + trait)"))
    lhs = formula.lhs
    lhs isa StatsModels.Term ||
        throw(ArgumentError("the temporal fit needs one response column on the left-hand side"))
    response = lhs.sym
    sch = StatsModels.schema(formula, cols)
    applied = StatsModels.apply_schema(formula, sch, StatsModels.StatisticalModel)
    _, X = StatsModels.modelcols(applied, cols)
    names = StatsModels.coefnames(applied.rhs)
    names = names isa AbstractString ? [String(names)] : String.(collect(names))
    return response, Matrix{Float64}(X isa AbstractVector ? reshape(X, :, 1) : X), names
end

function _temporal_start(y, X, L::TemporalLayout)
    theta = zeros(L.total)
    beta = X \ y
    theta[L.beta] .= beta
    r = y .- X * beta
    s = length(r) > 1 ? std(r) : 0.0
    theta[L.log_sigma] = log(max(isfinite(s) ? s : 0.0, 1e-3))  # .gllvmTMB_log_sigma_eps_start
    theta[L.time] = 0.0
    if L.rank > 0
        theta[L.rr] .= init_theta_rr(L.p, L.rank)                # init_rr_theta: 0.5 diag, 0 lower
    end
    theta[L.diag] .= 0.0
    return theta
end

"""
    fit_temporal_gllvm(long_data; formula, temporal, trait = :trait,
                       structure = Expr[], start = nothing, g_tol = 1e-6,
                       iterations = 500)

Fit a Gaussian model with one temporal covariance source to long data (one
row per series, occasion, trait and optional replicate), by exact maximum
likelihood. Twin of gllvmTMB P1's `gllvmTMB(value ~ 0 + trait +
temporal_*(0 + trait | series, time = ...), family = gaussian())` for the
temporal source alone.

`formula` is the mean model, for example `@formula(value ~ 0 + trait)`;
`temporal` is a [`TemporalTerm`](@ref) from [`temporal_indep`](@ref),
[`temporal_dep`](@ref) or [`temporal_latent`](@ref). `long_data` is any
Tables.jl table. The marginal covariance of the response is
`Z (K ⊗ Sigma_T) Z' + sigma_eps^2 I`, with `K` block diagonal over series
(`phi^|t - u|` for AR1, `exp(-kappa |t - u|)` for OU) and `Sigma_T` given
by the mode. The data are validated as gllvmTMB's temporal pre-pass does, and
refusals throw [`TemporalContractError`](@ref) with gllvmTMB's message.

The parameter vector follows gllvmTMB's `opt\$par` order and names
(`b_fix`, `log_sigma_eps`, `theta_temporal_time`, `theta_temporal_rr`,
`theta_temporal_diag`); `start` supplies it in that order, and
`iterations = 0` evaluates a fit at `start`. Optimisation is Optim LBFGS
with ForwardDiff gradients; the final gradient and Hessian are recomputed.

Not available in this version: ordinary `unit` / `unit_obs` terms beside the
temporal source, cross-source cells (kernel, phylo, animal, spatial), the wide
`traits()` form, offsets, non-Gaussian families and the R bridge.
[`extract_temporal`](@ref) returns the fitted temporal parameters.
"""
function fit_temporal_gllvm(long_data; formula, temporal, trait::Symbol=:trait,
        structure=Expr[], start=nothing, g_tol::Real=1e-6, iterations::Integer=500)
    temporal isa TemporalTerm ||
        throw(ArgumentError("temporal must be a TemporalTerm from temporal_indep, temporal_dep or temporal_latent"))
    isfinite(g_tol) && g_tol > 0 || throw(ArgumentError("g_tol must be finite and positive"))
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative"))
    cols = Tables.columntable(long_data)
    formula isa StatsModels.FormulaTerm && formula.lhs isa StatsModels.Term ||
        throw(ArgumentError("formula must be a StatsModels formula with one response column, such as @formula(value ~ 0 + trait)"))
    spec = _parse_temporal_term(temporal, cols; trait=trait, response=formula.lhs.sym,
        structure=structure)
    response, X, coef_names = _temporal_design(formula, cols)
    y = Float64.(cols[response])
    L = TemporalLayout(size(X, 2), spec)
    theta0 = if start === nothing
        _temporal_start(y, X, L)
    else
        length(start) == L.total ||
            throw(DimensionMismatch("start has $(length(start)) coordinates; expected $(L.total)"))
        all(x -> x isa Real && isfinite(x), start) || throw(ArgumentError("start must be finite and real"))
        Float64.(collect(start))
    end
    rows = _temporal_series_rows(spec)
    objective(t) = temporal_marginal_nll(t, y, X, spec; series_rows=rows)
    isfinite(objective(theta0)) || throw(ArgumentError("start produces an invalid covariance"))
    estimate, iters, opt_converged = iterations == 0 ? (theta0, 0, true) :
        _temporal_optimize(objective, theta0, Float64(g_tol), Int(iterations))
    return _temporal_fit_result(estimate, y, X, spec, temporal, cols, formula, trait,
        coef_names, rows; iterations=iters, optimizer_converged=opt_converged,
        g_tol=Float64(g_tol), max_iterations=Int(iterations))
end

# LBFGS with the default Hager-Zhang line search; a boundary fit can send a
# trial step to a singular covariance (objective Inf), which that line search
# rejects with an assertion. Then rerun from the same start with cubic
# backtracking, which steps back from an infinite trial value.
function _temporal_optimize(objective, theta0, g_tol, iterations)
    options = Optim.Options(g_tol=g_tol, iterations=iterations)
    res = try
        Optim.optimize(objective, theta0, Optim.LBFGS(), options; autodiff=:forward)
    catch e
        e isa AssertionError || e isa DomainError || rethrow()
        nothing
    end
    if res === nothing || !isfinite(Optim.minimum(res))
        res = Optim.optimize(objective, theta0,
            Optim.LBFGS(linesearch=Optim.LineSearches.BackTracking(order=3)), options;
            autodiff=:forward)
    end
    return Optim.minimizer(res), Optim.iterations(res), Optim.converged(res)
end

function _temporal_fit_result(estimate, y, X, spec, term, cols, formula, trait, coef_names,
        rows; iterations, optimizer_converged, g_tol, max_iterations)
    L = TemporalLayout(size(X, 2), spec)
    objective(t) = temporal_marginal_nll(t, y, X, spec; series_rows=rows)
    value = objective(estimate)
    gradient_norm = Inf
    min_eig = NaN
    if isfinite(value)
        g = ForwardDiff.gradient(objective, estimate)
        gradient_norm = all(isfinite, g) ? maximum(abs, g) : Inf
        H = ForwardDiff.hessian(objective, estimate)
        min_eig = all(isfinite, H) ? eigmin(Symmetric(H)) : NaN
    end
    pd = isfinite(min_eig) && min_eig > 0
    converged = optimizer_converged && isfinite(value) && isfinite(gradient_norm) &&
        gradient_norm <= g_tol
    reason = converged ? :converged : !isfinite(value) ? :invalid_final :
        (max_iterations > 0 && iterations >= max_iterations) ? :iteration_limit :
        :gradient_not_converged
    Sigma, Lambda, psi = _temporal_trait_block(L, estimate)
    theta_time = estimate[L.time]
    return TemporalGaussianFit(collect(estimate[L.beta]), coef_names,
        exp(estimate[L.log_sigma]), spec.structure === :ar1 ? :phi : :ou_rate,
        _temporal_time_value(spec.structure, theta_time),
        Lambda === nothing ? nothing : Matrix{Float64}(Lambda),
        psi === nothing ? nothing : Vector{Float64}(psi), Matrix{Float64}(Sigma),
        collect(Float64, estimate), _temporal_parameter_names(L), -value, converged,
        gradient_norm, min_eig, pd, iterations, reason, term, spec, y, X, cols, formula,
        trait, g_tol, max_iterations)
end

"""Return the fixed-effect coefficients of a `TemporalGaussianFit`."""
coef(f::TemporalGaussianFit) = copy(f.beta)
"""Return the exact marginal log-likelihood of a `TemporalGaussianFit`."""
loglikelihood(f::TemporalGaussianFit) = f.loglik
"""Return the number of response rows of a `TemporalGaussianFit` (gllvmTMB's `nobs`)."""
nobs(f::TemporalGaussianFit) = length(f.y)
"""Return the number of free outer coordinates of a `TemporalGaussianFit` (`length(opt\$par)` in gllvmTMB)."""
dof(f::TemporalGaussianFit) = length(f.parameters)
"""Return `-2 loglik + 2 dof` for a `TemporalGaussianFit`."""
aic(f::TemporalGaussianFit) = -2 * loglikelihood(f) + 2 * dof(f)
"""Return `-2 loglik + dof log(nobs)` for a `TemporalGaussianFit`."""
bic(f::TemporalGaussianFit) = -2 * loglikelihood(f) + dof(f) * log(nobs(f))

function Base.show(io::IO, f::TemporalGaussianFit)
    print(io, "TemporalGaussianFit(temporal_", f.spec.mode, ", :", f.spec.structure, ", ",
        f.spec.workflow, ", ", length(f.spec.traits), " traits, ",
        length(unique(f.spec.pair_table.series)), " series, ", length(f.spec.pair_table.time),
        " states, ", f.time_parameter, "=", f.time_value, ", loglik=", f.loglik,
        ", status=", f.stopping_reason, ")")
end
