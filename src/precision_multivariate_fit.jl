# Candidate internal fitter for the sparse multivariate PrecisionPhy kernel.
# The public bridge/exports own admission; this file only owns complete
# Gaussian data, fixed sigma2_phy = 1, finite-difference optimization, and
# observed-marginal interval targets.

using LinearAlgebra
using Optim

# Retain the compatibility name, but use the shared objective-failure contract.
const _PMV_PENALTY = _NLL_SENTINEL
_pmv_valid_objective(value) = !_fd_failed(value)

@inline function _pmv_mode(mode::Symbol)
    mode in (:barelowrank, :explicitunique) ||
        throw(ArgumentError("mode must be :barelowrank or :explicitunique"))
    return mode
end

@inline function _pmv_residual_mode(residual_mode::Symbol)
    residual_mode in (:trait, :shared) ||
        throw(ArgumentError("residual_mode must be :trait or :shared"))
    return residual_mode
end

@inline function _pmv_layout(d::Integer, rank::Integer, mode::Symbol, q::Integer;
        residual_mode::Symbol = :trait)
    _pmv_mode(mode)
    _pmv_residual_mode(residual_mode)
    _warn_covariance_redundancy(:phylo,
        rr_theta_len(d, rank) + (mode === :explicitunique ? d : 0), d * (d + 1) ÷ 2)
    rr = rr_theta_len(d, rank)
    unique_range = mode === :explicitunique ? ((q + rr + 1):(q + rr + d)) : (1:0)
    residual_start = mode === :explicitunique ? q + rr + d + 1 : q + rr + 1
    residual_count = residual_mode === :shared ? 1 : d
    return (rr = rr, loading = (q + 1):(q + rr), unique = unique_range,
            residual = residual_start:(residual_start + residual_count - 1),
            total = residual_start + residual_count - 1)
end

"""Unpack the internal `[beta; rr; log_sd_U?; log_sd_eps]` coordinates."""
function _precision_multivariate_unpack(theta::AbstractVector, d::Integer,
        rank::Integer, mode::Symbol, q::Integer; residual_mode::Symbol = :trait)
    1 <= rank <= d || throw(ArgumentError("rank must lie in 1:d"))
    q >= 0 || throw(ArgumentError("mean-design column count must be nonnegative"))
    layout = _pmv_layout(d, rank, mode, q; residual_mode = residual_mode)
    length(theta) == layout.total ||
        throw(DimensionMismatch("packed parameter length $(length(theta)) differs from expected $(layout.total)"))
    all(isfinite, theta) || throw(ArgumentError("packed parameters must be finite"))
    # Preserve a possible Dual element type for the interval-target closures.
    # The sparse likelihood itself is deliberately Float64-only, but its
    # derived target maps must remain differentiable by ForwardDiff.
    beta = collect(view(theta, 1:q))
    loading = unpack_lambda(view(theta, layout.loading), d, rank)
    unique = mode === :explicitunique ? exp.(2 .* view(theta, layout.unique)) : nothing
    residual_raw = exp.(2 .* view(theta, layout.residual))
    residual = residual_mode === :shared ? fill(only(residual_raw), d) : collect(residual_raw)
    all(isfinite, residual) && all(>(0), residual) ||
        throw(ArgumentError("residual variances are not finite positive values"))
    unique === nothing || (all(isfinite, unique) && all(>(0), unique)) ||
        throw(ArgumentError("unique phylogenetic variances are not finite positive values"))
    return (beta = beta, loading = loading,
            phylo_unique_variance = unique,
            residual_variance = residual, layout = layout, residual_mode = residual_mode)
end

"""
    _precision_multivariate_nll(Y, phy, theta; rank, mode, species_id, mean_design=nothing)

Internal complete-data negative marginal log likelihood. `Y` is traits x
observations; the residual is transposed only at the sparse-kernel boundary.
The phylogenetic scale is deliberately fixed at one.
"""
function _precision_multivariate_nll(Y::AbstractMatrix, phy::PrecisionPhy,
        theta::AbstractVector; rank::Integer, mode::Symbol,
        residual_mode::Symbol = :trait,
        species_id::AbstractVector{<:Integer} = collect(1:phy.n_leaves),
        mean_design = nothing)
    d, m = size(Y)
    d > 0 && m > 0 || throw(ArgumentError("Y must have positive trait and observation dimensions"))
    all(isfinite, Y) || throw(ArgumentError("Y must be finite"))
    D = mean_design === nothing ? _trait_mean_design(d, m) : mean_design
    size(D, 1) == d * m || throw(DimensionMismatch("mean_design must have d*m rows"))
    q = size(D, 2)
    u = try
        _precision_multivariate_unpack(theta, d, rank, mode, q;
            residual_mode = residual_mode)
    catch err
        err isa ArgumentError || err isa DimensionMismatch || rethrow()
        return _NLL_SENTINEL
    end
    residual = Matrix{Float64}(Y) .- reshape(D * u.beta, d, m)
    value = try
        -multivariate_phylo_precision_loglik(residual', phy, u.loading,
            u.residual_variance; sigma2_phy = 1.0,
            phylo_unique_variance = u.phylo_unique_variance,
            species_id = species_id)
    catch err
        err isa PosDefException || err isa ArgumentError || rethrow()
        _NLL_SENTINEL
    end
    return _pmv_valid_objective(value) ? value : _NLL_SENTINEL
end

function _pmv_fd_gradient!(storage::AbstractVector, objective::Function,
        theta::AbstractVector)
    @inbounds for j in eachindex(theta)
        h = cbrt(eps(Float64)) * max(1.0, abs(theta[j]))
        plus = copy(theta); minus = copy(theta)
        plus[j] += h; minus[j] -= h
        fp, fm = objective(plus), objective(minus)
        # A failed arm is not a zero slope.  NaN forces both Optim and the
        # post-fit convergence gate to surface an invalid finite-difference
        # stencil rather than certifying a boundary as stationary.
        storage[j] = _pmv_valid_objective(fp) && _pmv_valid_objective(fm) ?
            (fp - fm) / (2h) : NaN
    end
    return storage
end

function _pmv_hessian_diagnostics(objective::Function, theta::AbstractVector)
    H = try
        _fd_hessian(objective, theta)
    catch err
        err isa InterruptException && rethrow()
        fill(NaN, length(theta), length(theta))
    end
    all(isfinite, H) || return (minimum = NaN, positive_definite = false, condition = NaN)
    F = cholesky(Symmetric(H); check = false)
    issuccess(F) || return (minimum = eigmin(Symmetric(H)), positive_definite = false, condition = Inf)
    return (minimum = eigmin(Symmetric(H)), positive_definite = true, condition = cond(H))
end

"""Multivariate Gaussian result from the explicit `fit_gllvm(...; phylo=precision)` route."""
struct PrecisionMultivariateFit <: StatsAPI.StatisticalModel
    beta::Vector{Float64}
    loading::Matrix{Float64}
    phylo_unique_variance::Union{Nothing,Vector{Float64}}
    residual_variance::Vector{Float64}
    mode::Symbol
    rank::Int
    phy::PrecisionPhy
    species_id::Vector{Int}
    parameters::Vector{Float64}
    loglik::Float64
    converged::Bool
    gradient_norm::Float64
    hessian_min_eigenvalue::Float64
    hessian_positive_definite::Bool
    hessian_condition_number::Float64
    iterations::Int
    stopping_reason::Symbol
    response::Matrix{Float64}
    mean_design::Matrix{Float64}
    response_shape::Tuple{Int,Int}
    coefficient_names::Vector{Union{String,Symbol}}
    residual_mode::Symbol
    parameter_labels::Vector{String}
end

function _pmv_parameter_labels(q::Integer, d::Integer, rank::Integer,
        mode::Symbol, residual_mode::Symbol)
    layout = _pmv_layout(d, rank, mode, q; residual_mode = residual_mode)
    labels = ["beta[$j]" for j in 1:q]
    append!(labels, ["phylo.loading[$j]" for j in 1:layout.rr])
    mode === :explicitunique && append!(labels,
        ["log_sd_phylo_unique[$j]" for j in 1:d])
    append!(labels, residual_mode === :shared ? ["log_sd_residual_shared"] :
        ["log_sd_residual[$j]" for j in 1:d])
    return labels
end

# Existing direct fixture constructors retain their trait-residual semantics.
function PrecisionMultivariateFit(beta, loading, unique, residual, mode, rank,
        phy, species_id, parameters, loglik, converged, gradient_norm,
        hessian_minimum, hessian_pd, hessian_condition, iterations, reason,
        response, design, shape, names)
    return PrecisionMultivariateFit(beta, loading, unique, residual, mode, rank,
        phy, species_id, parameters, loglik, converged, gradient_norm,
        hessian_minimum, hessian_pd, hessian_condition, iterations, reason,
        response, design, shape, names, :trait,
        _pmv_parameter_labels(length(beta), size(loading, 1), rank, mode, :trait))
end

"""
    fit_precision_multivariate(Y, phy; rank=1, mode=:barelowrank,
                               residual_mode=:trait,
                               species_id=collect(1:phy.n_leaves), X=nothing,
                               coefficient_names=nothing, start=nothing,
                               g_tol=1e-5, iterations=400)

Fit the candidate complete Gaussian multivariate phylogenetic model with
fixed `sigma2_phy=1`. `Y` is traits x observations. `X`, when supplied, uses
the existing complete trait-major `p*n x q` / `p x n x q` mean-design helper.
`residual_mode=:trait` estimates one observation residual variance per trait;
`:shared` estimates one common variance, using a single log-SD coordinate.
The latter matches the residual layout of the simplest frozen-R phylogenetic
model, but does not by itself establish paired likelihood agreement.
The optimizer and post-fit diagnostics use finite differences because CHOLMOD
does not accept automatic-differentiation numbers.
"""
function fit_precision_multivariate(Y::AbstractMatrix{<:Real}, phy::PrecisionPhy;
        rank::Integer = 1, mode::Symbol = :barelowrank,
        residual_mode::Symbol = :trait,
        species_id::AbstractVector{<:Integer} = collect(1:phy.n_leaves),
        X = nothing, coefficient_names = nothing, start = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 400)
    phy = _validate_precision_fit_input(phy)
    d, m = size(Y)
    d > 0 && m >= 2 || throw(ArgumentError("fitting needs at least one trait and two observations"))
    1 <= rank <= d || throw(ArgumentError("rank must lie in 1:d"))
    _pmv_mode(mode)
    _pmv_residual_mode(residual_mode)
    all(isfinite, Y) || throw(ArgumentError("Y must be finite and complete"))
    length(species_id) == m && all(i -> 1 <= i <= phy.n_leaves, species_id) ||
        throw(ArgumentError("species_id must map every observation to a valid tip"))
    isfinite(g_tol) && g_tol > 0 || throw(ArgumentError("g_tol must be finite and positive"))
    iterations >= 0 || throw(ArgumentError("iterations must be nonnegative"))
    data = Matrix{Float64}(Y)
    default_means = X === nothing
    D = default_means ? _trait_mean_design(d, m) : _source_mean_design(X, d, m)
    q = size(D, 2)
    names = _source_coefficient_names(coefficient_names, q;
        default_trait_names = default_means)
    layout = _pmv_layout(d, rank, mode, q; residual_mode = residual_mode)
    beta0 = collect(Float64, D \ vec(data))
    residual0 = reshape(vec(data) - D * beta0, d, m)
    trait_var = vec(sum(abs2, residual0; dims = 2)) ./ max(m - 1, 1)
    trait_var .= max.(trait_var, 1e-3)
    theta0 = if start === nothing
        base = vcat(beta0, init_theta_rr(d, rank))
        mode === :explicitunique && append!(base, log.(sqrt.(0.15 .* trait_var)))
        if residual_mode === :shared
            push!(base, log(sqrt(0.85 * sum(trait_var) / d)))
        else
            append!(base, log.(sqrt.(0.85 .* trait_var)))
        end
        base
    else
        length(start) == layout.total ||
            throw(DimensionMismatch("start has $(length(start)) coordinates; expected $(layout.total)"))
        all(x -> x isa Real && isfinite(x), start) ||
            throw(ArgumentError("start must be finite and real"))
        Float64.(start)
    end
    objective = theta -> _precision_multivariate_nll(data, phy, theta;
        rank = rank, mode = mode, residual_mode = residual_mode,
        species_id = species_id, mean_design = D)
    _pmv_valid_objective(objective(theta0)) ||
        throw(ArgumentError("start produces an invalid marginal objective"))
    gradient! = (storage, theta) -> _pmv_fd_gradient!(storage, objective, theta)
    result = Optim.optimize(objective, gradient!, theta0,
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)),
        Optim.Options(g_tol = Float64(g_tol), iterations = Int(iterations)))
    estimate = collect(Float64, Optim.minimizer(result))
    objective_value = objective(estimate)
    gradient = zeros(Float64, length(estimate))
    _pmv_valid_objective(objective_value) && _pmv_fd_gradient!(gradient, objective, estimate)
    gradient_norm = _pmv_valid_objective(objective_value) && all(isfinite, gradient) ? maximum(abs, gradient) : Inf
    hessian = _pmv_valid_objective(objective_value) ? _pmv_hessian_diagnostics(objective, estimate) :
        (minimum = NaN, positive_definite = false, condition = NaN)
    converged = Optim.converged(result) && _pmv_valid_objective(objective_value) && gradient_norm <= g_tol
    reason = converged ? :converged : !_pmv_valid_objective(objective_value) ? :invalid_final :
        Optim.iterations(result) >= iterations ? :iteration_limit : :gradient_not_converged
    u = _precision_multivariate_unpack(estimate, d, rank, mode, q;
        residual_mode = residual_mode)
    loglik = _pmv_valid_objective(objective_value) ? -objective_value : NaN
    return PrecisionMultivariateFit(u.beta, u.loading,
        u.phylo_unique_variance === nothing ? nothing : collect(u.phylo_unique_variance),
        u.residual_variance, mode, Int(rank), phy, collect(Int, species_id), estimate,
        loglik, converged, gradient_norm, hessian.minimum,
        hessian.positive_definite, hessian.condition, Optim.iterations(result), reason,
        data, Matrix{Float64}(D), (d, m), names, residual_mode,
        _pmv_parameter_labels(q, d, rank, mode, residual_mode))
end

function _pmv_targets(fit::PrecisionMultivariateFit)
    d, _ = fit.response_shape
    q = length(fit.beta)
    layout = _pmv_layout(d, fit.rank, fit.mode, q;
        residual_mode = fit.residual_mode)
    targets = NamedTuple[]
    for j in 1:q
        push!(targets, (name = "beta[$j]", value = theta -> theta[j], transform = :identity))
    end
    unpack = theta -> _precision_multivariate_unpack(theta, d, fit.rank, fit.mode, q;
        residual_mode = fit.residual_mode)
    for j in 1:d
        push!(targets, (name = "phylo_cov[$j,$j]", value = theta -> begin
            u = unpack(theta); sum(abs2, view(u.loading, j, :)) +
                (u.phylo_unique_variance === nothing ? 0.0 : u.phylo_unique_variance[j])
        end, transform = :log))
        for i in (j + 1):d
            push!(targets, (name = "phylo_cov[$i,$j]", value = theta -> begin
                u = unpack(theta); dot(view(u.loading, i, :), view(u.loading, j, :))
            end, transform = :identity))
        end
        index = fit.residual_mode === :shared ? first(layout.residual) :
            first(layout.residual) + j - 1
        label = fit.residual_mode === :shared ? "residual_var_shared[$j]" :
            "residual_var[$j]"
        push!(targets, (name = label, value = theta -> exp(2 * theta[index]),
            transform = :log))
    end
    return targets
end

"""
    _pmv_phylogenetic_signal(fit)

Report why this candidate fitter does not yet expose the frozen R
species-level latent-variance fraction. Observation residual variance is not
the species-level non-phylogenetic component in that estimand. A per-tip
phylogenetic share of observation variance would be a different quantity and
must not silently substitute for R's `extract_phylo_signal()` default.
"""
function _pmv_phylogenetic_signal(::PrecisionMultivariateFit)
    return (status = :estimand_not_admitted,
        definition = "frozen R default: species-level latent phylogenetic variance / total species-level latent variance; excludes observation residual",
        message = "phylogenetic signal is not admitted: the species-level non-phylogenetic decomposition and matching extractor have not been paired; observation residual psi is not a substitute")
end

"""
    precision_multivariate_intervals(fit; level=.95, gradient_tolerance=1e-4)

Return full observed-marginal transformed-Wald intervals for the fixed effects,
rotation-invariant phylogenetic trait covariance and observation residual
variances of a [`PrecisionMultivariateFit`](@ref). The retained responses,
precision and complete design define the objective. Nuisance coordinates are
included in the information matrix; no conditional random-effect curvature is
substituted. Inspect overall `status` and each row in `intervals` before using
endpoints. Nonstationarity, non-identification and failed curvature remain
explicit diagnostics. This supplies neither a profile fallback nor an admitted
phylogenetic-signal estimate, R parity, or nominal coverage certification.
"""
function precision_multivariate_intervals(fit::PrecisionMultivariateFit;
        level::Real = .95, gradient_tolerance::Real = 1e-4)
    p = size(fit.response, 1)
    redundant = rr_theta_len(p, fit.rank) + (fit.mode === :explicitunique ? p : 0) > p * (p + 1) ÷ 2
    objective = theta -> _precision_multivariate_nll(fit.response, fit.phy, theta;
        rank = fit.rank, mode = fit.mode, residual_mode = fit.residual_mode,
        species_id = fit.species_id,
        mean_design = fit.mean_design)
    return _marginal_target_intervals(objective, fit.parameters, _pmv_targets(fit);
        structural_redundancy=redundant,
        converged = fit.converged, level = level, gradient_tolerance = gradient_tolerance)
end

coef(fit::PrecisionMultivariateFit) = copy(fit.beta)
loglikelihood(fit::PrecisionMultivariateFit) = fit.loglik
nobs(fit::PrecisionMultivariateFit) = length(fit.response)
dof(fit::PrecisionMultivariateFit) = length(fit.parameters)
