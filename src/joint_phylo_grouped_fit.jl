# Private outer optimizer for one joint phylogenetic-plus-ordinary Gaussian
# marginal objective. Public routing and bridge admission remain elsewhere.

using LinearAlgebra
using Optim
using SparseArrays

_jpgf_valid(value) = !_fd_failed(value)

function _joint_phylo_grouped_fit_layout(p::Integer, rank::Integer,
        phylo_mode::Symbol, q::Integer, terms::Vector{GroupingTerm};
        residual_mode::Symbol = :trait)
    1 <= rank <= p || throw(ArgumentError("rank must lie in 1:p"))
    _pmv_mode(phylo_mode)
    _pmv_residual_mode(residual_mode)
    q >= 0 || throw(ArgumentError("mean-design column count must be nonnegative"))
    all(term -> term.mode === :indep && !term.common, terms) || throw(ArgumentError(
        "joint ordinary terms must use mode=:indep and common=false"))
    rr = rr_theta_len(p, rank)
    phylo_coordinates = rr + (phylo_mode === :explicitunique ? p : 0)
    phylo_coordinates <= p * (p + 1) ÷ 2 || throw(ArgumentError(
        "phylogenetic factor-plus-unique coordinates are structurally non-identifiable"))
    loading = (q + 1):(q + rr)
    unique = phylo_mode === :explicitunique ? ((q + rr + 1):(q + rr + p)) : (1:0)
    residual_start = phylo_mode === :explicitunique ? q + rr + p + 1 : q + rr + 1
    residual_count = residual_mode === :shared ? 1 : p
    residual = residual_start:(residual_start + residual_count - 1)
    ordinary_start = last(residual) + 1
    ordinary = UnitRange{Int}[]
    for _ in terms
        push!(ordinary, ordinary_start:(ordinary_start + p - 1))
        ordinary_start += p
    end
    return (loading = loading, unique = unique, residual = residual,
        ordinary = ordinary, total = ordinary_start - 1)
end

function _joint_phylo_grouped_fit_unpack(theta::AbstractVector, p::Integer,
        rank::Integer, phylo_mode::Symbol, q::Integer, terms::Vector{GroupingTerm};
        residual_mode::Symbol = :trait)
    layout = _joint_phylo_grouped_fit_layout(p, rank, phylo_mode, q, terms;
        residual_mode = residual_mode)
    length(theta) == layout.total || throw(DimensionMismatch(
        "packed parameter length $(length(theta)) differs from expected $(layout.total)"))
    all(isfinite, theta) || throw(ArgumentError("packed parameters must be finite"))
    beta = collect(view(theta, 1:q))
    loading = unpack_lambda(view(theta, layout.loading), p, rank)
    unique = phylo_mode === :explicitunique ? exp.(2 .* view(theta, layout.unique)) : nothing
    residual_raw = exp.(2 .* view(theta, layout.residual))
    residual = residual_mode === :shared ? fill(only(residual_raw), p) : residual_raw
    ordinary = [exp.(2 .* view(theta, range)) for range in layout.ordinary]
    all(isfinite, residual) && all(>(0), residual) || throw(ArgumentError(
        "residual variances are not finite positive values"))
    unique === nothing || (all(isfinite, unique) && all(>(0), unique)) || throw(ArgumentError(
        "phylogenetic unique variances are not finite positive values"))
    all(v -> all(isfinite, v) && all(>(0), v), ordinary) || throw(ArgumentError(
        "ordinary variances are not finite positive values"))
    return (beta = beta, loading = loading,
        phylo_unique_variance = unique, residual_variance = collect(residual),
        ordinary_variances = [collect(v) for v in ordinary], layout = layout)
end

function _joint_phylo_grouped_nll(Y::AbstractMatrix, phy::PrecisionPhy,
        terms::Vector{GroupingTerm}, incidences::AbstractVector;
        rank::Integer, phylo_mode::Symbol, residual_mode::Symbol = :trait,
        species_id::AbstractVector{<:Integer} = collect(1:phy.n_leaves),
        mean_design = nothing)
    p, n = size(Y)
    p > 0 && n > 0 || throw(ArgumentError("Y must have positive trait and observation dimensions"))
    data = try
        Matrix{Float64}(Y)
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError("Y must convert to Float64"))
    end
    all(isfinite, data) || throw(ArgumentError("Y must be finite"))
    D = try
        mean_design === nothing ? _trait_mean_design(p, n) : Matrix{Float64}(mean_design)
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError("mean_design must convert to Float64"))
    end
    size(D, 1) == p * n || throw(DimensionMismatch("mean_design rows must equal p*n"))
    all(isfinite, D) || throw(ArgumentError("mean_design must be finite"))
    LinearAlgebra.rank(D) == size(D, 2) || throw(ArgumentError("mean_design must have full column rank"))
    q = size(D, 2)
    length(terms) == length(incidences) || throw(DimensionMismatch(
        "one incidence matrix is required per ordinary term"))
    phy_snapshot = deepcopy(phy)
    species_snapshot = collect(Int, species_id)
    terms_snapshot = copy(terms)
    incidence_snapshot = copy.(incidences)
    _joint_phylo_grouped_fit_layout(p, rank, phylo_mode, q, terms_snapshot;
        residual_mode = residual_mode)
    return function (theta)
        unpacked = try
            _joint_phylo_grouped_fit_unpack(theta, p, rank, phylo_mode, q, terms_snapshot;
                residual_mode = residual_mode)
        catch error
            error isa InterruptException && rethrow()
            error isa ArgumentError || error isa DimensionMismatch || rethrow()
            return _NLL_SENTINEL
        end
        residualized = data .- reshape(D * unpacked.beta, p, n)
        value = try
            -joint_phylo_grouped_gaussian_loglik(residualized, phy_snapshot, unpacked.loading,
                unpacked.residual_variance; phylo_unique_variance = unpacked.phylo_unique_variance,
                species_id = species_snapshot, ordinary_incidences = incidence_snapshot,
                ordinary_trait_variances = unpacked.ordinary_variances)
        catch error
            error isa InterruptException && rethrow()
            error isa ArgumentError || error isa DimensionMismatch || error isa PosDefException || rethrow()
            return _NLL_SENTINEL
        end
        return _jpgf_valid(value) ? value : _NLL_SENTINEL
    end
end

function _joint_phylo_grouped_fit_labels(coefficient_names, p::Integer, rank::Integer,
        phylo_mode::Symbol, terms::Vector{GroupingTerm}; residual_mode::Symbol = :trait)
    labels = String.(coefficient_names)
    append!(labels, ["phylo.loading[$j]" for j in 1:rr_theta_len(p, rank)])
    phylo_mode === :explicitunique && append!(labels,
        ["phylo.log_sd_unique[$j]" for j in 1:p])
    append!(labels, residual_mode === :shared ? ["log_sd_psi_shared"] :
        ["log_sd_psi[$j]" for j in 1:p])
    for term in terms
        append!(labels, ["$(term.name).log_sd[$j]" for j in 1:p])
    end
    return labels
end

"""Joint Gaussian precision/grouping result returned by [`fit_gllvm`](@ref)."""
struct JointPhyloGroupedGaussianFit <: StatsAPI.StatisticalModel
    beta::Vector{Float64}
    loading::Matrix{Float64}
    phylo_unique_variance::Union{Nothing,Vector{Float64}}
    phylo_covariance::Matrix{Float64}
    residual_variance::Vector{Float64}
    ordinary_covariances::Vector{Matrix{Float64}}
    terms::Vector{GroupingTerm}
    phy::PrecisionPhy
    species_id::Vector{Int}
    parameters::Vector{Float64}
    parameter_labels::Vector{String}
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
    incidences::Vector{SparseMatrixCSC{Float64,Int}}
    phylo_mode::Symbol
    rank::Int
    residual_mode::Symbol
end

# Positional constructions that pre-date `residual_mode` (26 arguments) keep working
# and mean the per-trait residual layout.
JointPhyloGroupedGaussianFit(args::Vararg{Any,26}) =
    JointPhyloGroupedGaussianFit(args..., :trait)

"""
    fit_joint_phylo_grouped_gaussian(Y, phy; rank=1, phylo_mode=:barelowrank,
        terms, unit=nothing, unit_obs=nothing, cluster=nothing, cluster2=nothing,
        species_id=collect(1:phy.n_leaves), X=nothing, residual_mode=:trait, ...)

Internal optimizer for a complete-data Gaussian model with one joint marginal likelihood
for a fixed-scale phylogenetic low-rank covariance plus ordinary independent
grouping sources. All ordinary terms must be `mode=:indep, common=false`.
Use `fit_gllvm(...; phylo=precision, grouping=terms)` for the public route.
`residual_mode=:trait` (default) estimates one observation residual variance per
trait. `residual_mode=:shared` estimates one common residual variance (a single
log-SD coordinate, `p - 1` fewer parameters), which is the residual layout of
R's Gaussian route; `start` must then have the shorter length. The fit records
its mode and every post-fit path rebuilds the objective with it. Agreement with
R under `:shared` has not been measured. This optimizer does not establish
qualification or R parity.
"""
function fit_joint_phylo_grouped_gaussian(Y::AbstractMatrix{<:Real}, phy::PrecisionPhy;
        rank::Integer = 1, phylo_mode::Symbol = :barelowrank, terms,
        unit = nothing, unit_obs = nothing, cluster = nothing, cluster2 = nothing,
        species_id::AbstractVector{<:Integer} = collect(1:phy.n_leaves),
        X = nothing, coefficient_names = nothing, start = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 200, residual_mode::Symbol = :trait)
    _pmv_residual_mode(residual_mode)
    phy = _validate_precision_fit_input(phy)
    p, n = size(Y)
    p > 0 && n >= 2 || throw(ArgumentError("fitting needs at least one trait and two observations"))
    isfinite(g_tol) && g_tol > 0 || throw(ArgumentError("g_tol must be finite and positive"))
    iterations >= 0 || throw(ArgumentError("iterations must be nonnegative"))
    all(term -> term isa GroupingTerm, terms) || throw(ArgumentError(
        "terms must contain GroupingTerm objects"))
    termvec = GroupingTerm[terms...]
    isempty(termvec) && throw(ArgumentError("at least one ordinary independent grouping term is required"))
    all(term -> term.mode === :indep && !term.common, termvec) || throw(ArgumentError(
        "joint ordinary terms must use mode=:indep and common=false"))
    data = try
        Matrix{Float64}(Y)
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError("Y must convert to Float64"))
    end
    all(isfinite, data) || throw(ArgumentError("Y must be finite and complete"))
    length(species_id) == n && all(i -> 1 <= i <= phy.n_leaves, species_id) ||
        throw(ArgumentError("species_id must map every observation to a valid tip"))
    label_vectors = _grouped_labels(n, termvec; unit = unit, unit_obs = unit_obs,
        cluster = cluster, cluster2 = cluster2)
    incidences = [_grouped_incidence(values, n) for values in label_vectors]
    D = X === nothing ? _trait_mean_design(p, n) : _source_mean_design(X, p, n)
    D64 = try
        Matrix{Float64}(D)
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError("mean design must convert to Float64"))
    end
    all(isfinite, D64) || throw(ArgumentError("mean design must be finite"))
    LinearAlgebra.rank(D64) == size(D64, 2) || throw(ArgumentError("mean design must have full column rank"))
    q = size(D64, 2)
    names = _source_coefficient_names(coefficient_names, q; default_trait_names = X === nothing)
    layout = _joint_phylo_grouped_fit_layout(p, rank, phylo_mode, q, termvec;
        residual_mode = residual_mode)
    beta0 = collect(Float64, D64 \ vec(data))
    residual0 = reshape(vec(data) - D64 * beta0, p, n)
    trait_variance = vec(sum(abs2, residual0; dims = 2)) ./ max(n - 1, 1)
    trait_variance .= max.(trait_variance, 1e-3)
    theta0 = if start === nothing
        theta = vcat(beta0, init_theta_rr(p, rank))
        phylo_mode === :explicitunique && append!(theta, log.(sqrt.(0.15 .* trait_variance)))
        if residual_mode === :shared
            push!(theta, log(sqrt(0.70 * sum(trait_variance) / p)))
        else
            append!(theta, log.(sqrt.(0.70 .* trait_variance)))
        end
        for _ in termvec
            append!(theta, log.(sqrt.(0.15 .* trait_variance)))
        end
        theta
    else
        length(start) == layout.total || throw(DimensionMismatch(
            "start has $(length(start)) coordinates; expected $(layout.total)"))
        all(x -> x isa Real && isfinite(x), start) || throw(ArgumentError(
            "start must be finite and real"))
        try
            Float64.(start)
        catch error
            error isa InterruptException && rethrow()
            throw(ArgumentError("start must convert to Float64"))
        end
    end
    phy_snapshot = deepcopy(phy)
    species_snapshot = collect(Int, species_id)
    terms_snapshot = copy(termvec)
    incidences_snapshot = copy.(incidences)
    objective = _joint_phylo_grouped_nll(data, phy_snapshot, terms_snapshot, incidences_snapshot;
        rank = rank, phylo_mode = phylo_mode, residual_mode = residual_mode,
        species_id = species_snapshot, mean_design = D64)
    _jpgf_valid(objective(theta0)) || throw(ArgumentError(
        "start produces an invalid joint marginal objective"))
    gradient! = (storage, theta) -> _pmv_fd_gradient!(storage, objective, theta)
    result = Optim.optimize(objective, gradient!, theta0,
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)),
        Optim.Options(g_tol = Float64(g_tol), iterations = Int(iterations)))
    estimate = collect(Float64, Optim.minimizer(result))
    objective_value = objective(estimate)
    gradient = zeros(Float64, length(estimate))
    _jpgf_valid(objective_value) && _pmv_fd_gradient!(gradient, objective, estimate)
    gradient_norm = _jpgf_valid(objective_value) && all(isfinite, gradient) ?
        maximum(abs, gradient) : Inf
    hessian = _jpgf_valid(objective_value) ? _pmv_hessian_diagnostics(objective, estimate) :
        (minimum = NaN, positive_definite = false, condition = NaN)
    converged = Optim.converged(result) && _jpgf_valid(objective_value) && gradient_norm <= g_tol
    reason = converged ? :converged : !_jpgf_valid(objective_value) ? :invalid_final :
        Optim.iterations(result) >= iterations ? :iteration_limit : :gradient_not_converged
    unpacked = _joint_phylo_grouped_fit_unpack(estimate, p, rank, phylo_mode, q, terms_snapshot;
        residual_mode = residual_mode)
    phylo_covariance = unpacked.loading * unpacked.loading' +
        (unpacked.phylo_unique_variance === nothing ? zeros(Float64, p, p) :
            Matrix(Diagonal(unpacked.phylo_unique_variance)))
    ordinary_covariances = [Matrix(Diagonal(v)) for v in unpacked.ordinary_variances]
    component_diagnostics = _joint_covariance_identification(phy_snapshot,
        species_snapshot, terms_snapshot, incidences_snapshot, unpacked.loading;
        phylo_unique_variance=unpacked.phylo_unique_variance)
    if component_diagnostics.reason !== :identified
        @warn "joint phylo grouped covariance components are not separately identifiable; the fixed-parameter marginal kernel remains evaluable, but component interpretation and observed-marginal component inference are unavailable" reason=component_diagnostics.reason labels=component_diagnostics.labels aliases=component_diagnostics.aliases
    end
    return JointPhyloGroupedGaussianFit(collect(unpacked.beta), Matrix{Float64}(unpacked.loading),
        unpacked.phylo_unique_variance === nothing ? nothing :
            collect(Float64, unpacked.phylo_unique_variance),
        Matrix{Float64}(phylo_covariance), collect(Float64, unpacked.residual_variance),
        ordinary_covariances, terms_snapshot, phy_snapshot,
        species_snapshot, estimate, _joint_phylo_grouped_fit_labels(names, p, rank,
            phylo_mode, terms_snapshot; residual_mode = residual_mode),
        _jpgf_valid(objective_value) ? -objective_value : NaN,
        converged, gradient_norm, hessian.minimum, hessian.positive_definite,
        hessian.condition, Optim.iterations(result), reason, copy(data), copy(D64), (p, n), names,
        incidences_snapshot, phylo_mode, Int(rank), residual_mode)
end

function _joint_phylo_grouped_component_identification(fit::JointPhyloGroupedGaussianFit)
    return _joint_covariance_identification(fit.phy, fit.species_id, fit.terms,
        fit.incidences, fit.loading; phylo_unique_variance=fit.phylo_unique_variance)
end

function _joint_phylo_grouped_targets(fit::JointPhyloGroupedGaussianFit)
    p, _ = fit.response_shape
    q = length(fit.beta)
    unpack = theta -> _joint_phylo_grouped_fit_unpack(theta, p, fit.rank,
        fit.phylo_mode, q, fit.terms; residual_mode = fit.residual_mode)
    targets = NamedTuple[]
    for j in 1:q
        push!(targets, (name = "beta[$j]", value = theta -> theta[j], transform = :identity))
    end
    for j in 1:p
        push!(targets, (name = "phylo_cov[$j,$j]", value = theta -> begin
            u = unpack(theta)
            sum(abs2, view(u.loading, j, :)) +
                (u.phylo_unique_variance === nothing ? 0.0 : u.phylo_unique_variance[j])
        end, transform = :log))
        for i in (j + 1):p
            push!(targets, (name = "phylo_cov[$i,$j]", value = theta -> begin
                u = unpack(theta); dot(view(u.loading, i, :), view(u.loading, j, :))
            end, transform = :identity))
        end
        push!(targets, (name = fit.residual_mode === :shared ?
            "residual_var_shared[$j]" : "residual_var[$j]", value = theta ->
            unpack(theta).residual_variance[j], transform = :log))
        for source in eachindex(fit.terms)
            push!(targets, (name = "$(fit.terms[source].name)_var[$j]", value = theta ->
                unpack(theta).ordinary_variances[source][j], transform = :log))
        end
    end
    return targets
end

"""
    joint_phylo_grouped_intervals(fit; level=.95, gradient_tolerance=1e-4)

Return observed-marginal transformed-Wald intervals for fixed effects,
phylogenetic trait covariance, ordinary group variances and observation residual
variances in a [`JointPhyloGroupedGaussianFit`](@ref). The full joint likelihood
includes nuisance parameters. Inspect `status` and individual `intervals` rows;
boundary, stationarity and curvature failures are not repaired with a ridge or
conditional-field information. This is not a profile fallback, a
phylogenetic-signal extractor, or a coverage/R-parity certification.
"""
function joint_phylo_grouped_intervals(fit::JointPhyloGroupedGaussianFit;
        level::Real = .95, gradient_tolerance::Real = 1e-4)
    # Keep the early component-identification exits honest: they must validate
    # the same inference controls as `_marginal_target_intervals` would.
    isfinite(level) && 0 < level < 1 ||
        throw(ArgumentError("level must lie in (0,1)"))
    isfinite(gradient_tolerance) && gradient_tolerance > 0 ||
        throw(ArgumentError("gradient_tolerance must be positive and finite"))
    objective = _joint_phylo_grouped_nll(fit.response, fit.phy, fit.terms, fit.incidences;
        rank = fit.rank, phylo_mode = fit.phylo_mode, residual_mode = fit.residual_mode,
        species_id = fit.species_id, mean_design = fit.mean_design)
    targets = _joint_phylo_grouped_targets(fit)
    component = _joint_phylo_grouped_component_identification(fit)
    component.reason === :invalid && return _joint_phylo_grouped_unavailable_intervals(
        targets, :invalid_covariance_tangent)
    # Structural component aliases are independent of numerical stationarity.
    # Report them before the generic helper's convergence gate.
    component.reason === :nonidentifiable && return _joint_phylo_grouped_unavailable_intervals(
        targets, :nonidentifiable)
    return _marginal_target_intervals(objective, fit.parameters,
        targets; converged = fit.converged, level = level,
        gradient_tolerance = gradient_tolerance,
        structural_redundancy = fit.phylo_mode === :explicitunique &&
            rr_theta_len(first(fit.response_shape), fit.rank) + first(fit.response_shape) >
            first(fit.response_shape) * (first(fit.response_shape) + 1) ÷ 2)
end

function _joint_phylo_grouped_unavailable_intervals(targets, status::Symbol)
    return (status=status, covariance=nothing, gradient_norm=NaN, condition_number=NaN,
        intervals=[(name=String(target.name), estimate=NaN, lower=NaN,
            upper=NaN, se_transformed=NaN, transform=target.transform,
            method=:unavailable, status=status) for target in targets])
end

coef(fit::JointPhyloGroupedGaussianFit) = copy(fit.beta)
loglikelihood(fit::JointPhyloGroupedGaussianFit) = fit.loglik
nobs(fit::JointPhyloGroupedGaussianFit) = length(fit.response)
dof(fit::JointPhyloGroupedGaussianFit) = length(fit.parameters)
