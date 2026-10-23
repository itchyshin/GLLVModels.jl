# Destination-B Gaussian named-grouping implementation. Qualification and
# non-Gaussian/public bridge expansion remain separately gated.

const _GROUPING_TERM_NAMES = (:unit, :unit_obs, :cluster, :cluster2)

function _grouped_stopping_reason(converged, valid, used_iterations, limit, gradient_norm, g_tol)
    converged && return :converged
    !valid && return :invalid_final
    used_iterations >= limit && return :iteration_limit
    gradient_norm > g_tol && return :gradient_not_converged
    return :optimizer_not_converged
end

"""
    GroupingTerm(name; mode=:latent, rank=1, unique=false, common=false)

Select a shared random-effect covariance for `:unit`, `:unit_obs`, `:cluster`,
or `:cluster2`. Use with `fit_gllvm(...; grouping=[...], unit=labels, ...)`.
`:latent` uses a rank-restricted trait factor, optionally with trait-specific
unique variance; `:indep` uses independent trait variances; `:dep` uses a full
trait covariance. `common=true` ties independent/unique variances across traits.
Only latent terms take `rank`. `cluster2` requires `mode=:indep`.

Identifiers specify sharing, not covariance: each selected term also needs its
labels. `unit_obs` labels must be globally nested within supplied `unit` labels;
`cluster` may cross units. Current public fitting support is Gaussian with one
shared residual variance. No frozen-R parity or recovery qualification is implied.
"""
struct GroupingTerm
    name::Symbol
    mode::Symbol
    rank::Int
    unique::Bool
    common::Bool
end

function GroupingTerm(name::Symbol; mode::Symbol = :latent, rank = nothing,
        unique::Bool = false, common::Bool = false)
    name in _GROUPING_TERM_NAMES ||
        throw(ArgumentError("grouping term name must be :unit, :unit_obs, :cluster, or :cluster2"))
    mode in (:latent, :indep, :dep) ||
        throw(ArgumentError("grouping mode must be :latent, :indep, or :dep"))
    name === :cluster2 && mode !== :indep &&
        throw(ArgumentError("cluster2 supports only mode=:indep"))
    if mode === :latent
        rank === nothing && (rank = 1)
        rank isa Integer && !(rank isa Bool) && rank > 0 ||
            throw(ArgumentError("latent rank must be a positive integer"))
        common && !unique &&
            throw(ArgumentError("common=true requires latent unique=true"))
        return GroupingTerm(name, mode, Int(rank), unique, common)
    elseif mode === :indep
        rank === nothing || throw(ArgumentError("rank is only used with mode=:latent"))
        unique && throw(ArgumentError("unique is encoded by mode=:indep, not its unique flag"))
        return GroupingTerm(name, mode, 0, false, common)
    else
        rank === nothing || throw(ArgumentError("rank is only used with mode=:latent"))
        (unique || common) && throw(ArgumentError("dep terms do not take unique or common"))
        return GroupingTerm(name, mode, 0, false, false)
    end
end

"""
    GroupedGaussianFit

Joint Gaussian grouping result returned by `fit_gllvm` with explicit `grouping`.
Retains fixed effects, trait covariance per selected term, a shared residual SD,
optimizer/curvature diagnostics, and response/design/incidence provenance for
[`grouped_gaussian_intervals`](@ref). Point convergence and valid uncertainty
are separate: inspect `converged` and `hessian_positive_definite` independently.
"""
struct GroupedGaussianFit
    beta::Vector{Float64}
    sigma_eps::Float64
    term_covariances::Vector{Matrix{Float64}}
    terms::Vector{GroupingTerm}
    parameters::Vector{Float64}
    loglik::Float64
    converged::Bool
    gradient_norm::Float64
    hessian_min_eigenvalue::Float64
    hessian_positive_definite::Bool
    iterations::Int
    stopping_reason::Symbol
    mean_design::Matrix{Float64}
    coefficient_names::Vector{Union{String,Symbol}}
    parameter_labels::Vector{String}
    response_shape::Tuple{Int,Int}
    response::Matrix{Float64}
    incidences::Vector{SparseMatrixCSC{Float64,Int}}
end

function _grouping_term_nparams(term::GroupingTerm, p::Integer)
    if term.mode === :latent
        return rr_theta_len(p, term.rank) + (term.unique ? (term.common ? 1 : p) : 0)
    elseif term.mode === :indep
        return term.common ? 1 : p
    end
    return rr_theta_len(p, p)
end

function _grouped_identification_diagnostics(terms, p::Integer)
    covariance_dimension = p * (p + 1) ÷ 2
    # This necessary dimension check detects structural redundancy. Passing
    # it is not proof of identification: incidence aliasing and boundaries
    # still need separate checks on the fitted marginal objective.
    return [(source=term.name, parameter_count=_grouping_term_nparams(term, p),
             covariance_dimension=covariance_dimension)
            for term in terms if _grouping_term_nparams(term, p) > covariance_dimension]
end

function _grouped_warn_identification(terms, p::Integer)
    for note in _grouped_identification_diagnostics(terms, p)
        _warn_covariance_redundancy(note.source, note.parameter_count, note.covariance_dimension)
    end
    return nothing
end

function _warn_covariance_redundancy(source, parameter_count, covariance_dimension)
    parameter_count > covariance_dimension || return nothing
    @warn "Covariance is structurally non-identifiable in its current coordinates; loadings and unique variances cannot be separated. Total covariance may still be identifiable. Use a non-redundant supported covariance representation; full-coordinate Wald intervals may be unavailable." source parameter_count covariance_dimension
    return nothing
end

function _grouped_term_unpack(theta::AbstractVector, p::Integer,
        terms::Vector{GroupingTerm})
    T = eltype(theta)
    loads = Matrix{T}[]
    uniques = Union{Nothing,Vector{T}}[]
    covariances = Matrix{T}[]
    offset = 0
    for term in terms
        if term.mode === :latent
            count = rr_theta_len(p, term.rank)
            L = unpack_lambda(view(theta, offset + 1:offset + count), p, term.rank)
            offset += count
            d = nothing
            if term.unique
                nd = term.common ? 1 : p
                raw = exp.(2 .* view(theta, offset + 1:offset + nd))
                d = term.common ? fill(only(raw), p) : collect(raw)
                offset += nd
            end
            push!(loads, L)
            push!(uniques, d)
            push!(covariances, L * L' + (d === nothing ? zeros(T, p, p) : Diagonal(d)))
        elseif term.mode === :indep
            nd = term.common ? 1 : p
            raw = exp.(2 .* view(theta, offset + 1:offset + nd))
            d = term.common ? fill(only(raw), p) : collect(raw)
            offset += nd
            push!(loads, zeros(T, p, 1))
            push!(uniques, d)
            push!(covariances, Matrix(Diagonal(d)))
        else
            count = rr_theta_len(p, p)
            L = unpack_lambda(view(theta, offset + 1:offset + count), p, p)
            offset += count
            push!(loads, L)
            push!(uniques, nothing)
            push!(covariances, L * L')
        end
    end
    return loads, uniques, covariances, offset
end

function _grouped_incidence(values::AbstractVector, n::Integer)
    length(values) == n || throw(DimensionMismatch("grouping labels must have one value per observation"))
    !isempty(values) && all(x -> !ismissing(x), values) ||
        throw(ArgumentError("grouping labels must be nonempty and contain no missing values"))
    codes, levels = _code_grouping(values)
    return sparse(collect(1:n), codes, ones(Float64, n), n, length(levels))
end

function _validate_unit_obs_nesting(unit::AbstractVector, unit_obs::AbstractVector)
    length(unit) == length(unit_obs) || throw(DimensionMismatch("unit and unit_obs must have equal length"))
    parent = Dict{Any,Any}()
    for (u, o) in zip(unit, unit_obs)
        if haskey(parent, o) && parent[o] != u
            throw(ArgumentError("unit_obs must be globally nested: each unit_obs level maps to one unit"))
        end
        parent[o] = u
    end
    return nothing
end

function _grouped_labels(n::Integer, terms::Vector{GroupingTerm}; unit=nothing,
        unit_obs=nothing, cluster=nothing, cluster2=nothing)
    isempty(terms) && throw(ArgumentError("at least one explicit GroupingTerm is required; labels alone add no random term"))
    labels = Dict{Symbol,Any}(:unit => unit, :unit_obs => unit_obs,
        :cluster => cluster, :cluster2 => cluster2)
    selected = Set(term.name for term in terms)
    length(selected) == length(terms) || throw(ArgumentError("grouping term names must be unique"))
    for (name, values) in labels
        values === nothing && continue
        values isa AbstractVector || throw(ArgumentError("$name labels must be an AbstractVector"))
        length(values) == n || throw(DimensionMismatch("$name labels must have one value per observation"))
        if !(name in selected) && !(name === :unit && (:unit_obs in selected))
            throw(ArgumentError("$name labels do not add a random term; select GroupingTerm(:$name, ...) explicitly"))
        end
    end
    for term in terms
        labels[term.name] === nothing &&
            throw(ArgumentError("GroupingTerm(:$(term.name)) requires corresponding labels"))
    end
    if :unit_obs in selected
        unit === nothing && throw(ArgumentError("unit_obs requires unit labels for global nesting validation"))
        _validate_unit_obs_nesting(unit, unit_obs)
    end
    return [labels[term.name] for term in terms]
end

function _grouped_fd_gradient(f, theta::Vector{Float64}; step::Float64 = 1e-5)
    g = similar(theta)
    for j in eachindex(theta)
        h = step * max(1.0, abs(theta[j]))
        plus = copy(theta); minus = copy(theta)
        plus[j] += h; minus[j] -= h
        fp, fm = f(plus), f(minus)
        g[j] = isfinite(fp) && isfinite(fm) && !_nll_failed(fp) && !_nll_failed(fm) ?
            (fp - fm) / (2h) : NaN
    end
    return g
end

function _grouped_fd_hessian(f, theta::Vector{Float64}; step::Float64 = 1e-4)
    # A finite failure sentinel is not likelihood information. Reject it at
    # every stencil point, including mixed derivatives.
    safe = value -> begin
        result = f(value)
        isfinite(result) && !_nll_failed(result) ? result : NaN
    end
    d = length(theta); H = Matrix{Float64}(undef, d, d); f0 = safe(theta)
    for j in 1:d
        hj = step * max(1.0, abs(theta[j]))
        ej = zeros(Float64, d); ej[j] = hj
        H[j, j] = (safe(theta + ej) - 2 * f0 + safe(theta - ej)) / hj^2
        for i in 1:(j - 1)
            hi = step * max(1.0, abs(theta[i]))
            ei = zeros(Float64, d); ei[i] = hi
            value = (safe(theta + ei + ej) - safe(theta + ei - ej) -
                     safe(theta - ei + ej) + safe(theta - ei - ej)) / (4hi * hj)
            H[i, j] = value; H[j, i] = value
        end
    end
    return H
end

"""
    _grouped_fd_hessian_from_gradient(gradfn, theta; step=1e-5) -> Matrix{Float64}

The S9 (D-274) Hessian: finite-differences a GRADIENT function `gradfn`
(`theta -> Vector{Float64}` or `nothing` on failure) in `d = length(theta)`
gradient calls, rather than `_grouped_fd_hessian`'s `O(d^2)` objective calls.
Central difference per coordinate, `H[:, j] = (gradfn(theta+h*e_j) -
gradfn(theta-h*e_j)) / (2h)`, then SYMMETRISED as `(H + H') / 2` -- the two
finite-difference columns need not agree exactly off-diagonal even though the
true Hessian does. Returns a matrix of `NaN` (the same failure sentinel
`_grouped_fd_hessian` uses) if `gradfn` returns `nothing` or a non-finite
vector at either stencil point of any coordinate, so callers can fall back to
`_grouped_fd_hessian` exactly as they already do for a failed value.
"""
function _grouped_fd_hessian_from_gradient(gradfn, theta::Vector{Float64}; step::Float64 = 1e-5)
    d = length(theta)
    fail = fill(NaN, d, d)
    H = Matrix{Float64}(undef, d, d)
    for j in 1:d
        h = step * max(1.0, abs(theta[j]))
        plus = copy(theta); minus = copy(theta)
        plus[j] += h; minus[j] -= h
        gp = gradfn(plus)
        gm = gradfn(minus)
        (gp === nothing || gm === nothing || !all(isfinite, gp) || !all(isfinite, gm)) && return fail
        H[:, j] = (gp .- gm) ./ (2h)
    end
    return (H .+ H') ./ 2
end

function _grouped_initial_parameters(data::Matrix{Float64}, D::Matrix{Float64},
        terms::Vector{GroupingTerm})
    p = size(data, 1)
    theta = collect(D \ vec(data))
    for term in terms
        if term.mode === :latent
            append!(theta, init_theta_rr(p, term.rank))
            term.unique && append!(theta, fill(log(0.25), term.common ? 1 : p))
        elseif term.mode === :indep
            append!(theta, fill(log(0.25), term.common ? 1 : p))
        else
            append!(theta, init_theta_rr(p, p))
        end
    end
    residual = vec(data) - D * view(theta, 1:size(D, 2))
    push!(theta, log(max(std(residual) / 2, 0.1)))
    return theta
end

function _grouped_parameter_labels(coefficient_names, p::Integer,
        terms::Vector{GroupingTerm})
    labels = String.(coefficient_names)
    for term in terms
        prefix = String(term.name)
        if term.mode === :latent
            append!(labels, ["$(prefix).loading[$j]" for j in 1:rr_theta_len(p, term.rank)])
            if term.unique
                append!(labels, term.common ? ["$(prefix).log_sd_unique_common"] :
                    ["$(prefix).log_sd_unique[$j]" for j in 1:p])
            end
        elseif term.mode === :indep
            append!(labels, term.common ? ["$(prefix).log_sd_common"] :
                ["$(prefix).log_sd[$j]" for j in 1:p])
        else
            append!(labels, ["$(prefix).loading[$j]" for j in 1:rr_theta_len(p, p)])
        end
    end
    push!(labels, "log_sigma_eps")
    return labels
end

"""Rebuild the full packed observed-marginal objective for interval integration."""
function _grouped_gaussian_objective(data::Matrix{Float64}, D::Matrix{Float64},
        terms::Vector{GroupingTerm}, incidences::Vector{SparseMatrixCSC{Float64,Int}})
    p, n = size(data)
    size(D, 1) == p * n || throw(DimensionMismatch("mean design rows must equal p*n"))
    length(terms) == length(incidences) || throw(DimensionMismatch("one incidence matrix is required per term"))
    q = size(D, 2)
    source_coordinates = sum(term -> _grouping_term_nparams(term, p), terms; init=0)
    zeros_mean = zeros(Float64, p)
    return function (value)
        length(value) == q + source_coordinates + 1 || return _NLL_SENTINEL
        all(isfinite, value) || return _NLL_SENTINEL
        try
            gamma = view(value, 1:q)
            loads, uniques, _, used = _grouped_term_unpack(view(value, q + 1:q + source_coordinates), p, terms)
            used == source_coordinates || return _NLL_SENTINEL
            sigma_eps = exp(value[end])
            adjusted = reshape(vec(data) - D * gamma, p, n)
            answer = _grouped_gaussian_factor_nll(adjusted, zeros_mean, incidences,
                loads, sigma_eps; uniques=uniques)
            return isfinite(answer) ? answer : _NLL_SENTINEL
        catch
            return _NLL_SENTINEL
        end
    end
end

"""
    fit_grouped_gaussian(Y; terms, unit=nothing, unit_obs=nothing,
                         cluster=nothing, cluster2=nothing, X=nothing, ...)

Internal joint Gaussian fitter used by the public explicit-grouping route.
Grouping labels supply incidence only; each random covariance requires a
`GroupingTerm`. The objective is the sparse grouped-factor marginal, optimised
with central finite-difference derivatives because CHOLMOD does not accept
forward-mode dual numbers. Use `fit_gllvm` or the grouping formula route for
public fitting and `grouped_gaussian_intervals` for marginal interval diagnostics.
This development implementation does not establish R parity or recovery.
"""
function fit_grouped_gaussian(Y::AbstractMatrix{<:Real}; terms,
        unit=nothing, unit_obs=nothing, cluster=nothing, cluster2=nothing,
        X=nothing, coefficient_names=nothing, start=nothing,
        g_tol::Real=1e-5, iterations::Integer=100)
    p, n = size(Y)
    p > 0 && n >= 2 || throw(ArgumentError("grouped fitting needs at least one trait and two observations"))
    all(isfinite, Y) || throw(ArgumentError("grouped fitting requires finite complete responses"))
    isfinite(g_tol) && g_tol > 0 || throw(ArgumentError("g_tol must be finite and positive"))
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative"))
    all(term -> term isa GroupingTerm, terms) ||
        throw(ArgumentError("terms must contain GroupingTerm objects"))
    termvec = GroupingTerm[terms...]
    _grouped_warn_identification(termvec, p)
    label_vectors = _grouped_labels(n, termvec; unit=unit, unit_obs=unit_obs,
        cluster=cluster, cluster2=cluster2)
    incidences = [_grouped_incidence(values, n) for values in label_vectors]
    data = try
        Matrix{Float64}(Y)
    catch
        throw(ArgumentError("responses must be representable as Float64"))
    end
    all(isfinite, data) || throw(ArgumentError("responses exceed Float64 range"))
    default_means = X === nothing
    D = default_means ? _trait_mean_design(p, n) : _source_mean_design(X, p, n)
    q = size(D, 2)
    names = _source_coefficient_names(coefficient_names, q; default_trait_names=default_means)
    source_coordinates = sum(term -> _grouping_term_nparams(term, p), termvec; init=0)
    total = q + source_coordinates + 1
    theta = if start === nothing
        _grouped_initial_parameters(data, D, termvec)
    else
        length(start) == total || throw(DimensionMismatch("start has $(length(start)) coordinates; expected $total"))
        all(x -> x isa Real && isfinite(x), start) || throw(ArgumentError("start must be finite and real"))
        Float64.(start)
    end
    objective = _grouped_gaussian_objective(data, D, termvec, incidences)
    isfinite(objective(theta)) && !_nll_failed(objective(theta)) ||
        throw(ArgumentError("start produces an invalid grouped marginal objective"))
    gradient! = (storage, value) -> (storage .= _grouped_fd_gradient(objective, value))
    result = Optim.optimize(objective, gradient!, theta, Optim.LBFGS(),
        Optim.Options(g_tol=Float64(g_tol), iterations=Int(iterations)))
    estimate = collect(Optim.minimizer(result))
    value = objective(estimate)
    valid = isfinite(value) && !_nll_failed(value)
    gradient = valid ? _grouped_fd_gradient(objective, estimate) : fill(Inf, length(estimate))
    gradient_norm = all(isfinite, gradient) ? maximum(abs, gradient) : Inf
    H = valid ? _grouped_fd_hessian(objective, estimate) : fill(NaN, length(estimate), length(estimate))
    min_eigenvalue = all(isfinite, H) ? eigmin(Symmetric(H)) : NaN
    pd_hessian = isfinite(min_eigenvalue) && min_eigenvalue > 0
    converged = Optim.converged(result) && valid && gradient_norm <= g_tol
    reason = _grouped_stopping_reason(converged, valid, Optim.iterations(result),
        iterations, gradient_norm, g_tol)
    _, _, covariances, _ = _grouped_term_unpack(view(estimate, q + 1:q + source_coordinates), p, termvec)
    return GroupedGaussianFit(collect(estimate[1:q]), exp(estimate[end]), covariances,
        termvec, estimate, valid ? -value : -Inf, converged, gradient_norm,
        min_eigenvalue, pd_hessian, Optim.iterations(result), reason, D, names,
        _grouped_parameter_labels(names, p, termvec), (p, n), copy(data), copy.(incidences))
end

function _grouped_primary_targets(fit::GroupedGaussianFit)
    p, _ = fit.response_shape
    q = size(fit.mean_design, 2)
    source_coordinates = length(fit.parameters) - q - 1
    unpack = theta -> _grouped_term_unpack(view(theta, q + 1:q + source_coordinates),
        p, fit.terms)
    targets = NamedTuple[]
    for j in 1:q
        index = j
        push!(targets, (name=Symbol("fixed.$(fit.parameter_labels[index])"),
            value=theta -> theta[index], transform=:identity))
    end
    for term_index in eachindex(fit.terms)
        prefix = String(fit.terms[term_index].name)
        covariance = theta -> unpack(theta)[3][term_index]
        for j in 1:p
            index = j
            push!(targets, (name=Symbol("$(prefix).variance[$index]"),
                value=theta -> covariance(theta)[index, index], transform=:log))
            push!(targets, (name=Symbol("$(prefix).sd[$index]"),
                value=theta -> sqrt(covariance(theta)[index, index]), transform=:log))
        end
        if fit.terms[term_index].mode !== :indep
            for column in 2:p, row in 1:(column - 1)
                i, j = row, column
                push!(targets, (name=Symbol("$(prefix).covariance[$i,$j]"),
                    value=theta -> covariance(theta)[i, j], transform=:identity))
            end
        end
        if fit.terms[term_index].mode === :latent && fit.terms[term_index].unique
            for j in 1:p
                index = j
                push!(targets, (name=Symbol("$(prefix).unique_variance[$index]"),
                    value=theta -> unpack(theta)[2][term_index][index], transform=:log))
            end
        end
    end
    push!(targets, (name=:residual_variance,
        value=theta -> exp(2theta[end]), transform=:log))
    return targets
end

"""
    grouped_gaussian_intervals(Y, fit; unit=nothing, unit_obs=nothing,
                               cluster=nothing, cluster2=nothing, ...)

Primary observed-marginal intervals for a `GroupedGaussianFit`. This is
a thin target/packing adapter over `_marginal_target_intervals`; it does not
repair curvature or provide profile intervals for boundary cases.
"""
function grouped_gaussian_intervals(Y::AbstractMatrix{<:Real}, fit::GroupedGaussianFit;
        unit=nothing, unit_obs=nothing, cluster=nothing, cluster2=nothing,
        level::Real=0.95, gradient_tolerance::Real=1e-4)
    p, n = fit.response_shape
    size(Y) == (p, n) || throw(DimensionMismatch("Y shape must match the fitted response shape"))
    data = try
        Matrix{Float64}(Y)
    catch
        throw(ArgumentError("responses must be representable as Float64"))
    end
    all(isfinite, data) || throw(ArgumentError("responses must be finite"))
    label_vectors = _grouped_labels(n, fit.terms; unit=unit, unit_obs=unit_obs,
        cluster=cluster, cluster2=cluster2)
    incidences = [_grouped_incidence(values, n) for values in label_vectors]
    data == fit.response ||
        throw(ArgumentError("Y must exactly match the response used to fit grouped uncertainty"))
    length(incidences) == length(fit.incidences) &&
        all(incidences[i] == fit.incidences[i] for i in eachindex(incidences)) ||
        throw(ArgumentError("grouping labels must induce the same incidences used to fit grouped uncertainty"))
    objective = _grouped_gaussian_objective(fit.response, fit.mean_design,
        fit.terms, fit.incidences)
    return _marginal_target_intervals(objective, fit.parameters,
        _grouped_primary_targets(fit);
        converged=fit.converged, level=level,
        structural_redundancy=!isempty(_grouped_identification_diagnostics(fit.terms, p)),
        gradient_tolerance=gradient_tolerance)
end
