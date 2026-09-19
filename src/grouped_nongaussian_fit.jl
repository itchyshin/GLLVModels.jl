# Private Destination-B grouped non-Gaussian outer fit candidate.  The parent
# integration owner controls inclusion, exports, and any public surface.

"""
    GroupedNonGaussianFit

Joint-Laplace fit returned by `fit_gllvm` for explicit non-Gaussian grouping.
Stores fixed effects, per-term trait covariances, complete marginal parameter
coordinates, convergence/curvature diagnostics, and owned response/design data.
Beta precision and NB2 size default to one value per trait; `dispersion=:shared`
selects one scalar. Point convergence does not establish interval feasibility.
"""
struct GroupedNonGaussianFit
    beta::Vector{Float64}
    family::Any
    dispersion::Union{Nothing,Float64,Vector{Float64}}
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
    trials::Matrix{Float64}
    incidences::Vector{SparseMatrixCSC{Float64,Int}}
    family_kind::Symbol
    dispersion_mode::Symbol
    inner_status::Symbol
end

function _grouped_nongaussian_kind(family)
    family isa Poisson && return :poisson
    family isa Binomial && return :binomial
    if family isa Beta
        family.β == 1.0 ||
            throw(ArgumentError("Beta grouped Laplace family must be Beta(phi, 1.0)"))
        isfinite(family.α) && family.α > 0 ||
            throw(ArgumentError("Beta precision phi must be finite and positive"))
        return :beta
    end
    if family isa NegativeBinomial
        family.p == 0.5 ||
            throw(ArgumentError("NB2 grouped Laplace family must be NegativeBinomial(r, 0.5)"))
        isfinite(family.r) && family.r > 0 ||
            throw(ArgumentError("NB2 size r must be finite and positive"))
        return :nb2
    end
    throw(ArgumentError("grouped non-Gaussian Laplace supports Poisson, Binomial, Beta(phi, 1.0), and NegativeBinomial(r, 0.5)"))
end

_grouped_nongaussian_link(::Val{:poisson}) = LogLink()
_grouped_nongaussian_link(::Val{:binomial}) = LogitLink()
_grouped_nongaussian_link(::Val{:beta}) = LogitLink()
_grouped_nongaussian_link(::Val{:nb2}) = LogLink()

function _grouped_nongaussian_dispersion_mode(kind::Symbol, dispersion::Symbol)
    dispersion in (:shared, :trait) ||
        throw(ArgumentError("dispersion must be :shared or :trait"))
    return kind in (:beta, :nb2) ? dispersion : :none
end

function _grouped_nongaussian_internal_dispersion_mode(kind::Symbol, mode::Symbol)
    if kind in (:beta, :nb2)
        mode in (:shared, :trait) ||
            throw(ArgumentError("Beta/NB2 internal dispersion mode must be :shared or :trait"))
        return mode
    end
    # The standalone objective historically defaulted to :trait.  Poisson and
    # Binomial have no dispersion coordinate, so normalize both public modes
    # (and the explicit internal spelling) to the no-op representation.
    mode in (:none, :shared, :trait) ||
        throw(ArgumentError("Poisson/Binomial internal dispersion mode must be :none, :shared, or :trait"))
    return :none
end

_grouped_nongaussian_dispersion_count(kind::Symbol, mode::Symbol, p::Integer) =
    kind in (:beta, :nb2) ? (mode === :shared ? 1 : p) : 0

function _grouped_nongaussian_dispersion_indices(q::Integer, source_coordinates::Integer,
        kind::Symbol, mode::Symbol, p::Integer)
    count = _grouped_nongaussian_dispersion_count(kind, mode, p)
    return (q + source_coordinates + 1):(q + source_coordinates + count)
end

function _grouped_nongaussian_family(kind::Symbol, value::AbstractVector,
        dispersion_indices::UnitRange{Int}, p::Integer, n::Integer, mode::Symbol)
    kind === :poisson && return Poisson()
    kind === :binomial && return Binomial()
    length(dispersion_indices) == _grouped_nongaussian_dispersion_count(kind, mode, p) ||
        throw(ArgumentError("dispersion coordinate layout is invalid"))
    values = exp.(view(value, dispersion_indices))
    all(isfinite, values) && all(>(0.0), values) || return nothing
    marker = kind === :beta ? (d -> Beta(d, 1.0)) :
        kind === :nb2 ? (d -> NegativeBinomial(d, 0.5)) : nothing
    marker === nothing && throw(ArgumentError("unsupported grouped non-Gaussian family kind"))
    mode === :shared && return marker(only(values))
    trait_markers = marker.(values)
    return [trait_markers[trait] for trait in 1:p, observation in 1:n] |> vec
end

function _grouped_nongaussian_dispersion(value::AbstractVector,
        dispersion_indices::UnitRange{Int}, kind::Symbol, mode::Symbol)
    kind in (:beta, :nb2) || return nothing
    natural = exp.(view(value, dispersion_indices))
    return mode === :shared ? only(natural) : collect(natural)
end

function _grouped_nongaussian_marker_start(family, kind::Symbol, p::Integer,
        mode::Symbol)
    value = kind === :beta ? family.α : kind === :nb2 ? family.r : nothing
    value === nothing && return Float64[]
    return fill(log(value), _grouped_nongaussian_dispersion_count(kind, mode, p))
end

function _grouped_nongaussian_parameter_labels(coefficient_names, p::Integer,
        terms::Vector{GroupingTerm}, kind::Symbol, mode::Symbol)
    labels = _grouped_parameter_labels(coefficient_names, p, terms)
    pop!(labels) # Gaussian residual scale is absent in the conditional families.
    if kind === :beta
        append!(labels, mode === :shared ? ["log_phi"] : ["log_phi[$trait]" for trait in 1:p])
    elseif kind === :nb2
        append!(labels, mode === :shared ? ["log_r"] : ["log_r[$trait]" for trait in 1:p])
    end
    return labels
end

function _grouped_laplace_trait_factors(load::AbstractMatrix, unique)
    L = Matrix{Float64}(load)
    columns = Matrix{Float64}[]
    all(iszero, L) || push!(columns, L)
    if unique !== nothing
        d = Float64.(unique)
        all(isfinite, d) && all(>=(0.0), d) ||
            throw(ArgumentError("grouped unique variances must be finite and non-negative"))
        positive = findall(>(0.0), d)
        isempty(positive) || begin
            U = zeros(Float64, length(d), length(positive))
            for (column, trait) in enumerate(positive)
                U[trait, column] = sqrt(d[trait])
            end
            push!(columns, U)
        end
    end
    isempty(columns) && return zeros(Float64, size(L, 1), 0)
    return reduce(hcat, columns)
end

"""
    _grouped_laplace_design(incidences, loads; uniques=nothing)

Build one global sparse random-effect design for the joint Laplace kernel. Its
response rows follow `vec(Y)` and its column blocks follow selected grouping
terms. This is the same `Z_s \\otimes L_s^*` representation as the grouped
Gaussian factor kernel, not scalar row effects.
"""
function _grouped_laplace_design(incidences::Vector{SparseMatrixCSC{Float64,Int}},
        loads::Vector{<:AbstractMatrix}; uniques=nothing)
    length(incidences) == length(loads) ||
        throw(DimensionMismatch("one grouped factor matrix is required per incidence"))
    source_uniques = uniques === nothing ? fill(nothing, length(loads)) : uniques
    length(source_uniques) == length(loads) ||
        throw(DimensionMismatch("one unique-variance vector is required per grouped factor"))
    blocks = SparseMatrixCSC{Float64,Int}[]
    for source in eachindex(loads)
        Lstar = _grouped_laplace_trait_factors(loads[source], source_uniques[source])
        size(Lstar, 2) == 0 && continue
        push!(blocks, grouped_trait_design(incidences[source], Lstar))
    end
    isempty(blocks) && return spzeros(Float64, size(incidences[1], 1) * size(loads[1], 1), 0)
    return sparse(reduce(hcat, blocks))
end

function _grouped_nongaussian_initial_parameters(data::Matrix{Float64},
        trials::Matrix{Float64}, D::Matrix{Float64}, terms::Vector{GroupingTerm},
        kind::Symbol, family, mode::Symbol)
    p, n = size(data)
    q = size(D, 2)
    theta = zeros(Float64, q)
    if D == _trait_mean_design(p, n)
        for trait in 1:p
            values = view(data, trait, :)
            if kind === :poisson || kind === :nb2
                theta[trait] = log(max(mean(values), 0.1))
            elseif kind === :binomial
                probability = (sum(values) + 0.5) / (sum(view(trials, trait, :)) + 1.0)
                theta[trait] = log(probability / (1.0 - probability))
            else
                probability = clamp(mean(values), 0.01, 0.99)
                theta[trait] = log(probability / (1.0 - probability))
            end
        end
    end
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
    append!(theta, _grouped_nongaussian_marker_start(family, kind, p, mode))
    return theta
end

"""
    _grouped_nongaussian_objective(...; warm_start_inner=false)

`warm_start_inner` (S7c, default `false` — opt-in): when `true`, the closure
keeps the last successfully-converged inner Laplace mode `b` (across calls to
the SAME closure instance) and seeds the next call's `joint_grouped_laplace_loglik`
with it via `b_init`, instead of starting cold from `zeros(m)` every time. One
outer optimisation (Nelder-Mead simplex evaluations plus FD-gradient/-Hessian
stencils) calls this closure with many nearby `value`s, so the previous mode
is typically a very short walk from the next one's. This changes ONLY the
Newton starting point, never the converged answer (see
`joint_grouped_laplace_loglik`'s docstring and
`test/test_grouped_laplace_identity.jl --gate warm_identity`). A non-`:ok`
result leaves the cache untouched (never poisons the next call with a bad
guess); `destination_b_fixed_effects.jl`'s call site does not pass this
keyword and stays cold, unaffected.
"""
function _grouped_nongaussian_objective(data::Matrix{Float64}, trials::Matrix{Float64},
        D::Matrix{Float64}, terms::Vector{GroupingTerm},
        incidences::Vector{SparseMatrixCSC{Float64,Int}}, kind::Symbol;
        dispersion_mode::Symbol=:trait, inner_maxiter::Integer, inner_tol::Float64,
        warm_start_inner::Bool=false)
    p, n = size(data)
    q = size(D, 2)
    mode = _grouped_nongaussian_internal_dispersion_mode(kind, dispersion_mode)
    source_coordinates = sum(term -> _grouping_term_nparams(term, p), terms; init=0)
    dispersion_indices = _grouped_nongaussian_dispersion_indices(q, source_coordinates,
        kind, mode, p)
    expected = q + source_coordinates + length(dispersion_indices)
    b_cache = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    return function (value)
        length(value) == expected && all(isfinite, value) || return _NLL_SENTINEL
        try
            gamma = collect(view(value, 1:q))
            loads, uniques, _, used = _grouped_term_unpack(
                view(value, q + 1:q + source_coordinates), p, terms)
            used == source_coordinates || return _NLL_SENTINEL
            family = _grouped_nongaussian_family(kind, value, dispersion_indices, p, n, mode)
            family === nothing && return _NLL_SENTINEL
            W = _grouped_laplace_design(incidences, loads; uniques=uniques)
            b_init = warm_start_inner ? b_cache[] : nothing
            result = joint_grouped_laplace_loglik(family, vec(data), vec(trials), D, gamma, W;
                link=_grouped_nongaussian_link(Val(kind)), maxiter=inner_maxiter, tol=inner_tol,
                b_init=b_init)
            if warm_start_inner && result.status === :ok
                b_cache[] = copy(result.mode)
            end
            return result.status === :ok && result.converged && isfinite(result.loglik) ?
                -result.loglik : _NLL_SENTINEL
        catch
            return _NLL_SENTINEL
        end
    end
end

function _grouped_nongaussian_trials(Y::AbstractMatrix, N, kind::Symbol)
    if kind === :binomial
        N === nothing && throw(ArgumentError("Binomial grouped Laplace fitting requires N trials"))
        size(N) == size(Y) || throw(DimensionMismatch("N must have the same shape as Y"))
        trials = try
            Matrix{Float64}(N)
        catch
            throw(ArgumentError("N must be representable as Float64"))
        end
        all(isfinite, trials) || throw(ArgumentError("N must be finite"))
        return trials
    end
    N === nothing || throw(ArgumentError("N trials are only used with Binomial grouped Laplace fitting"))
    return ones(Float64, size(Y))
end

"""
    fit_grouped_nongaussian(Y; family, terms, unit=nothing, unit_obs=nothing,
                             cluster=nothing, cluster2=nothing, N=nothing, ...)

Internal joint outer fit for grouped Poisson, Binomial, Beta, and NB2 models.
It reuses `GroupingTerm` packing and one global joint-Laplace random-effect
mode. Public fitting uses `fit_gllvm` or the grouping formula route; marginal
interval diagnostics use `grouped_nongaussian_intervals`. The development route
does not establish frozen-R parity, recovery, or coverage qualification.

`warm_start_inner` (S7c, default `true`): each of the many inner Laplace-fit
calls the outer optimiser makes (Nelder-Mead simplex evaluations plus the FD
gradient/Hessian stencils) seeds its Newton solve from the previous call's
converged mode instead of `zeros(m)`, since nearby outer-parameter values
share a nearby mode. This changes only how fast each inner solve converges,
never the converged answer (see `joint_grouped_laplace_loglik`'s `b_init`
docstring); pass `warm_start_inner=false` to recover the pre-S7c cold-start
behaviour.
"""
function fit_grouped_nongaussian(Y::AbstractMatrix{<:Real}; family, terms,
        unit=nothing, unit_obs=nothing, cluster=nothing, cluster2=nothing,
        N=nothing, X=nothing, coefficient_names=nothing, start=nothing,
        dispersion::Symbol=:trait,
        g_tol::Real=1e-4, iterations::Integer=100,
        inner_maxiter::Integer=100, inner_tol::Real=1e-8,
        warm_start_inner::Bool=true)
    p, n = size(Y)
    p > 0 && n >= 2 || throw(ArgumentError("grouped fitting needs at least one trait and two observations"))
    all(isfinite, Y) || throw(ArgumentError("grouped fitting requires finite complete responses"))
    isfinite(g_tol) && g_tol > 0 || throw(ArgumentError("g_tol must be finite and positive"))
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative"))
    inner_maxiter >= 0 || throw(ArgumentError("inner_maxiter must be non-negative"))
    isfinite(inner_tol) && inner_tol > 0 || throw(ArgumentError("inner_tol must be finite and positive"))
    kind = _grouped_nongaussian_kind(family)
    mode = _grouped_nongaussian_dispersion_mode(kind, dispersion)
    all(term -> term isa GroupingTerm, terms) ||
        throw(ArgumentError("terms must contain GroupingTerm objects"))
    termvec = GroupingTerm[terms...]
    _grouped_warn_identification(termvec, p)
    labels = _grouped_labels(n, termvec; unit=unit, unit_obs=unit_obs,
        cluster=cluster, cluster2=cluster2)
    incidences = [_grouped_incidence(values, n) for values in labels]
    data = try
        Matrix{Float64}(Y)
    catch
        throw(ArgumentError("responses must be representable as Float64"))
    end
    all(isfinite, data) || throw(ArgumentError("responses exceed Float64 range"))
    trials = _grouped_nongaussian_trials(data, N, kind)
    default_means = X === nothing
    D = default_means ? _trait_mean_design(p, n) : _source_mean_design(X, p, n)
    q = size(D, 2)
    names = _source_coefficient_names(coefficient_names, q; default_trait_names=default_means)
    source_coordinates = sum(term -> _grouping_term_nparams(term, p), termvec; init=0)
    dispersion_indices = _grouped_nongaussian_dispersion_indices(q, source_coordinates,
        kind, mode, p)
    total = q + source_coordinates + length(dispersion_indices)
    theta = if start === nothing
        _grouped_nongaussian_initial_parameters(data, trials, D, termvec, kind, family, mode)
    else
        length(start) == total || throw(DimensionMismatch("start has $(length(start)) coordinates; expected $total"))
        all(x -> x isa Real && isfinite(x), start) || throw(ArgumentError("start must be finite and real"))
        Float64.(start)
    end
    objective = _grouped_nongaussian_objective(data, trials, D, termvec, incidences, kind;
        dispersion_mode=mode, inner_maxiter=Int(inner_maxiter), inner_tol=Float64(inner_tol),
        warm_start_inner=warm_start_inner)
    initial_value = objective(theta)
    isfinite(initial_value) && !_nll_failed(initial_value) ||
        throw(ArgumentError("start produces an invalid grouped Laplace objective"))
    # The joint Laplace domain can invalidate a finite-difference neighbour.
    # `_grouped_fd_gradient` correctly marks such a stencil NaN; feeding it to
    # a line-search method would turn a rejected stencil into an Optim error.
    # Use a value-only outer search, then require a valid FD gradient/Hessian
    # for convergence and observed-marginal inference below.
    result = Optim.optimize(objective, theta, Optim.NelderMead(),
        Optim.Options(g_tol=Float64(g_tol), iterations=Int(iterations)))
    # A valid simplex endpoint can still be non-stationary because Nelder-Mead
    # stops on objective/simplex geometry, not this fitter's FD gradient norm.
    # Refine only when every initial BFGS stencil is valid. Invalid stencils
    # retain the value-only result and cannot be smuggled into a line search.
    candidate = collect(Optim.minimizer(result))
    candidate_gradient = _grouped_fd_gradient(objective, candidate)
    if all(isfinite, candidate_gradient)
        gradient! = (storage, value) -> (storage .= _grouped_fd_gradient(objective, value))
        refined = try
            Optim.optimize(objective, gradient!, candidate, Optim.BFGS(),
                Optim.Options(g_tol=Float64(g_tol), iterations=Int(iterations)))
        catch
            nothing
        end
        if refined !== nothing
            refined_estimate = collect(Optim.minimizer(refined))
            refined_value = objective(refined_estimate)
            if isfinite(refined_value) && !_nll_failed(refined_value) &&
                    refined_value <= objective(candidate)
                result = refined
            end
        end
    end
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
    loads, uniques, covariances, _ = _grouped_term_unpack(
        view(estimate, q + 1:q + source_coordinates), p, termvec)
    final_family = _grouped_nongaussian_family(kind, estimate, dispersion_indices, p, n, mode)
    final_W = _grouped_laplace_design(incidences, loads; uniques=uniques)
    inner = final_family === nothing ? nothing : joint_grouped_laplace_loglik(
        final_family, vec(data), vec(trials), D, collect(view(estimate, 1:q)), final_W;
        link=_grouped_nongaussian_link(Val(kind)), maxiter=Int(inner_maxiter), tol=Float64(inner_tol))
    inner_status = inner === nothing ? :invalid_family : inner.status
    natural_dispersion = _grouped_nongaussian_dispersion(estimate, dispersion_indices, kind, mode)
    return GroupedNonGaussianFit(collect(estimate[1:q]), final_family, natural_dispersion, covariances,
        termvec, estimate, valid ? -value : -Inf, converged, gradient_norm,
        min_eigenvalue, pd_hessian, Optim.iterations(result), reason, D, names,
        _grouped_nongaussian_parameter_labels(names, p, termvec, kind, mode), (p, n),
        copy(data), copy(trials), copy.(incidences), kind, mode, inner_status)
end

function _grouped_nongaussian_primary_targets(fit::GroupedNonGaussianFit)
    p, _ = fit.response_shape
    q = size(fit.mean_design, 2)
    source_coordinates = sum(term -> _grouping_term_nparams(term, p), fit.terms; init=0)
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
    dispersion_count = _grouped_nongaussian_dispersion_count(fit.family_kind,
        fit.dispersion_mode, p)
    if fit.family_kind === :beta
        for trait in 1:dispersion_count
            index = length(fit.parameters) - dispersion_count + trait
            name = fit.dispersion_mode === :shared ? :beta_precision :
                Symbol("beta_precision[$trait]")
            push!(targets, (name=name, value=theta -> exp(theta[index]), transform=:log))
        end
    elseif fit.family_kind === :nb2
        for trait in 1:dispersion_count
            index = length(fit.parameters) - dispersion_count + trait
            name = fit.dispersion_mode === :shared ? :nb2_size :
                Symbol("nb2_size[$trait]")
            push!(targets, (name=name, value=theta -> exp(theta[index]), transform=:log))
        end
    end
    return targets
end

"""
    grouped_nongaussian_intervals(Y, fit; unit=nothing, unit_obs=nothing,
                                  cluster=nothing, cluster2=nothing, ...)

Observed-marginal Wald intervals for a `GroupedNonGaussianFit`, including all
nuisance coordinates. Targets include fixed effects, group covariance entries,
and fitted Beta precision or NB2 size. Suitable positive interior targets use
log-scale intervals. No profile fallback is implemented here.

Supplied response, trials and labels must reproduce the fit provenance exactly.
Omitted `dispersion` inherits the fitted mode; an explicit mismatch is rejected.
Inspect overall and per-target statuses: nonconvergence, structural redundancy
or boundaries can make some or all intervals unavailable.
"""
function grouped_nongaussian_intervals(Y::AbstractMatrix{<:Real}, fit::GroupedNonGaussianFit;
        unit=nothing, unit_obs=nothing, cluster=nothing, cluster2=nothing,
        N=nothing, dispersion::Union{Nothing,Symbol}=nothing,
        level::Real=0.95, gradient_tolerance::Real=1e-4,
        inner_maxiter::Integer=100, inner_tol::Real=1e-8)
    p, n = fit.response_shape
    size(Y) == (p, n) || throw(DimensionMismatch("Y shape must match the fitted response shape"))
    data = try
        Matrix{Float64}(Y)
    catch
        throw(ArgumentError("responses must be representable as Float64"))
    end
    all(isfinite, data) || throw(ArgumentError("responses must be finite"))
    requested_mode = dispersion === nothing ? fit.dispersion_mode :
        _grouped_nongaussian_dispersion_mode(fit.family_kind, dispersion)
    requested_mode == fit.dispersion_mode ||
        throw(ArgumentError("dispersion mode must match the grouped fit"))
    trials = _grouped_nongaussian_trials(data, N, fit.family_kind)
    labels = _grouped_labels(n, fit.terms; unit=unit, unit_obs=unit_obs,
        cluster=cluster, cluster2=cluster2)
    incidences = [_grouped_incidence(values, n) for values in labels]
    data == fit.response ||
        throw(ArgumentError("Y must exactly match the response used to fit grouped uncertainty"))
    trials == fit.trials ||
        throw(ArgumentError("N must exactly match the trials used to fit grouped uncertainty"))
    length(incidences) == length(fit.incidences) &&
        all(incidences[i] == fit.incidences[i] for i in eachindex(incidences)) ||
        throw(ArgumentError("grouping labels must induce the same incidences used to fit grouped uncertainty"))
    objective = _grouped_nongaussian_objective(fit.response, fit.trials, fit.mean_design,
        fit.terms, fit.incidences, fit.family_kind;
        dispersion_mode=fit.dispersion_mode,
        inner_maxiter=Int(inner_maxiter), inner_tol=Float64(inner_tol))
    return _marginal_target_intervals(objective, fit.parameters,
        _grouped_nongaussian_primary_targets(fit);
        structural_redundancy=!isempty(_grouped_identification_diagnostics(fit.terms, p)),
        converged=fit.converged, level=level, gradient_tolerance=gradient_tolerance)
end
