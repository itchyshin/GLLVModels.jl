# Shared table syntax for the GLLVM prior adapter. This file intentionally
# contains no model-family implementation: route admission happens separately.
import Tables

struct GLLVMPriorRow
    kind::Symbol
    class::Symbol
    coef::String
    dpar::String
    group::String
    scale::String
    trait::String
    kernel::String
    args::Vector{Float64}
end

struct GLLVMResolvedPrior
    kind::Symbol
    class::Symbol
    target::String
    index::Int
    scale::String
    args::Vector{Float64}
end

struct GLLVMPriorResolution
    rows::Vector{GLLVMResolvedPrior}
    response_names::Vector{String}
    table_loading_lambda::Union{Nothing,Float64}
end

function _prior_table_value(row, name::Symbol, default = nothing)
    hasproperty(row, name) || return default
    value = getproperty(row, name)
    return value === missing || value === nothing ? default : value
end

function _prior_table_number(text::AbstractString, name::AbstractString)
    value = tryparse(Float64, strip(text))
    value !== nothing && isfinite(value) || throw(ArgumentError(
        "prior_table: `$name` must be a finite number; got `$text`"))
    return value
end

function _parse_gllvm_prior_rows(table)
    Tables.istable(typeof(table)) || throw(ArgumentError(
        "priors must be a Tables.jl table with `prior`, `class`, and `coef` columns"))
    # `NamedTuple[]` is the row-oriented empty-table spelling used when a
    # caller builds priors conditionally. It has no schema to validate and
    # means the same no-prior fit as `priors = nothing`.
    table isa AbstractVector && isempty(table) && eltype(table) <: NamedTuple &&
        return GLLVMPriorRow[]
    # NamedTuple vectors are a convenient row-oriented input for mixed prior
    # classes. Their rows may omit selectors that are optional for that class
    # (for example, a `fixef` row has no `trait` or `kernel`, while a
    # `unique_variance` row does). Tables.rowtable expects one rectangular row
    # schema, so normalize the union of row fields and mark absent fields as
    # missing before handing each row to the class-specific defaults below.
    rows, column_names = if table isa AbstractVector && all(row -> row isa NamedTuple, table)
        names = Symbol[]
        for row in table, name in propertynames(row)
            name in names || push!(names, name)
        end
        normalized = [NamedTuple{Tuple(names)}(Tuple(
            hasproperty(row, name) ? getproperty(row, name) : missing for name in names))
            for row in table]
        normalized, names
    else
        columns = Tables.columns(table)
        names = collect(Tables.columnnames(columns))
        table_rows = try
            Tables.rowtable(table)
        catch err
            throw(ArgumentError("priors must be a row-oriented Tables.jl table: $(sprint(showerror, err))"))
        end
        table_rows, names
    end
    length(unique(column_names)) == length(column_names) ||
        throw(ArgumentError("prior_table: column names must be unique"))
    required_columns = (:prior, :class, :coef)
    all(in(column_names), required_columns) || throw(ArgumentError(
        "priors must include `prior`, `class`, and `coef` columns"))
    allowed_columns = (required_columns..., :dpar, :group, :scale, :trait, :kernel)
    all(in(allowed_columns), column_names) || throw(ArgumentError(
        "prior_table contains an unsupported column"))
    parsed = GLLVMPriorRow[]
    for (i, row) in enumerate(rows)
        prior = _prior_table_value(row, :prior)
        class_raw = _prior_table_value(row, :class)
        coef_raw = _prior_table_value(row, :coef, "")
        prior isa AbstractString || throw(ArgumentError("prior_table row $i: `prior` must be a string"))
        class_raw isa Union{Symbol,AbstractString} || throw(ArgumentError("prior_table row $i: `class` must be a string or Symbol"))
        coef_raw isa AbstractString || throw(ArgumentError("prior_table row $i: `coef` must be a string"))
        m = match(r"^\s*(normal|student_t|gamma|ridge)\s*\(\s*([^()]*)\s*\)\s*$"i, prior)
        m === nothing && throw(ArgumentError(
            "prior_table row $i: unsupported prior expression `$prior`; use normal(location, scale), student_t(location, scale, df), gamma(mean, shape), or ridge(lambda)"))
        kind = Symbol(lowercase(m.captures[1]))
        raw_args = split(m.captures[2], ',')
        expected = kind === :normal ? 2 : kind === :student_t ? 3 : kind === :gamma ? 2 : 1
        length(raw_args) == expected || throw(ArgumentError(
            "prior_table row $i: $kind expects $expected argument(s)"))
        args = [_prior_table_number(a, "prior argument") for a in raw_args]
        if kind === :normal
            args[2] > 0 || throw(ArgumentError("prior_table row $i: Normal scale must be positive"))
        elseif kind === :student_t
            args[2] > 0 && args[3] > 0 || throw(ArgumentError(
                "prior_table row $i: Student-t scale and degrees of freedom must be positive"))
        elseif kind === :gamma
            args[1] > 0 && args[2] > 1 || throw(ArgumentError(
                "prior_table row $i: Gamma mean must be positive and shape must exceed one"))
        else
            args[1] >= 0 || throw(ArgumentError("prior_table row $i: ridge lambda must be nonnegative"))
        end
        class = Symbol(lowercase(String(class_raw)))
        coef = strip(String(coef_raw))
        isempty(coef) && class !== :unique_variance && throw(ArgumentError(
            "prior_table row $i: `coef` cannot be empty for `$class`; use `all` explicitly"))
        dpar_default = class === :residual_sd ? "sigma" : class === :unique_variance ? "" : "mu"
        dpar = String(_prior_table_value(row, :dpar, dpar_default))
        group = String(_prior_table_value(row, :group, ""))
        scale_default = class === :fixef ? "link" : class === :loading ? "loading_coef" :
            class === :unique_variance ? "natural_sd" : ""
        scale = String(_prior_table_value(row, :scale, scale_default))
        trait = String(_prior_table_value(row, :trait, ""))
        kernel = String(_prior_table_value(row, :kernel, ""))
        push!(parsed, GLLVMPriorRow(kind, class, coef, dpar, group, scale, trait, kernel, args))
    end
    return parsed
end

function _gllv_prior_table_controls(table)
    rows = _parse_gllvm_prior_rows(table)
    isempty(rows) && throw(ArgumentError("prior table must not be empty"))
    coefficient_priors = NamedTuple[]
    resolved_targets = NamedTuple[]
    seen_coefficients = Set{String}()
    loading_lambda = nothing
    loading_ridge = Inf
    for (i, row) in enumerate(rows)
        isempty(row.trait) && isempty(row.kernel) || throw(ArgumentError(
            "prior_table row $i: `trait` and `kernel` selectors are supported only for unique-variance priors"))
        if row.class === :fixef
            row.kind in (:normal, :student_t) || throw(ArgumentError(
                "prior_table row $i: fixed-effect priors support Normal or Student-t"))
            !isempty(row.coef) || throw(ArgumentError(
                "prior_table row $i: fixed-effect target `coef` must be named or `all`"))
            row.dpar == "mu" || throw(ArgumentError("prior_table row $i: fixed effects require dpar=`mu`"))
            isempty(row.group) || throw(ArgumentError("prior_table row $i: fixed effects do not accept `group`"))
            row.scale == "link" || throw(ArgumentError("prior_table row $i: fixed effects require scale=`link`"))
            row.coef in seen_coefficients && throw(ArgumentError(
                "prior_table has duplicate fixed-effect target `$(row.coef)`"))
            push!(seen_coefficients, row.coef)
            push!(resolved_targets, (; class="fixef", coef=row.coef,
                dpar=row.dpar, scale=row.scale))
            location, scale = row.args[1:2]
            df = row.kind === :student_t ? row.args[3] : nothing
            push!(coefficient_priors, (; coef=row.coef, kind=row.kind,
                location, scale, df))
        elseif row.class === :loading
            row.kind === :ridge || throw(ArgumentError("prior_table row $i: loading priors support ridge(lambda) only"))
            row.coef == "all" || throw(ArgumentError("prior_table row $i: loading ridge requires coef=`all`"))
            row.dpar == "mu" || throw(ArgumentError("prior_table row $i: loading ridge requires dpar=`mu`"))
            isempty(row.group) || throw(ArgumentError("prior_table row $i: loading ridge does not accept `group`"))
            row.scale == "loading_coef" || throw(ArgumentError("prior_table row $i: loading ridge requires scale=`loading_coef`"))
            loading_lambda === nothing || throw(ArgumentError("prior_table may contain only one loading-ridge row"))
            loading_lambda = only(row.args)
            loading_ridge = iszero(loading_lambda) ? Inf : inv(sqrt(loading_lambda))
            push!(resolved_targets, (; class="loading", coef=row.coef,
                dpar=row.dpar, scale=row.scale))
        else
            throw(ArgumentError("prior_table row $i: `$(row.class)` priors are not supported by this Binomial covariate route"))
        end
    end
    return (; lambda=loading_lambda, loading_ridge, coefficient_priors,
        resolved_targets, dpar="mu", scale=isempty(resolved_targets) ? "link" :
            any(t -> t.class == "loading", resolved_targets) ? "loading_coef" : "link")
end

function _resolve_gaussian_source_prior_table(table, sources, coefficient_names, trait_names, p)
    rows = _parse_gllvm_prior_rows(table)
    isempty(rows) && return NamedTuple[], String[]
    isempty(sources) && throw(ArgumentError("source prior tables require at least one named source"))
    coefficients = String.(coefficient_names)
    length(unique(coefficients)) == length(coefficients) || throw(ArgumentError(
        "coefficient_names must be unique to resolve prior-table targets"))
    has_psi_rows = any(row -> row.class === :unique_variance, rows)
    if has_psi_rows
        trait_names === nothing && throw(ArgumentError(
            "trait_names must be supplied to resolve source Psi prior targets"))
        (trait_names isa AbstractVector || trait_names isa Tuple) || throw(ArgumentError(
            "trait_names must be a vector or tuple with one name per response trait"))
        length(trait_names) == p || throw(ArgumentError(
            "trait_names must contain exactly $p names, one per response trait"))
        all(t -> (t isa Symbol || t isa AbstractString) && !isempty(strip(String(t))), trait_names) ||
            throw(ArgumentError("trait_names must contain nonempty symbols or strings"))
        traits = String.(trait_names)
        length(unique(traits)) == p || throw(ArgumentError(
            "trait_names must contain one unique name per response trait for Psi priors"))
    else
        traits = trait_names === nothing ? nothing : String.(trait_names)
    end
    resolved = NamedTuple[]
    seen = Set{Tuple{Symbol,String}}()
    psi_rows = NamedTuple[]
    for (i, row) in enumerate(rows)
        if row.class === :fixef
            isempty(row.trait) && isempty(row.kernel) || throw(ArgumentError(
                "prior_table row $i: `trait` and `kernel` selectors are supported only for unique-variance priors"))
            row.dpar == "mu" || throw(ArgumentError("prior_table row $i: source fixed effects require dpar=`mu`"))
            isempty(row.group) || throw(ArgumentError("prior_table row $i: source fixed effects do not accept `group`"))
            row.scale == "link" || throw(ArgumentError("prior_table row $i: source fixed effects require scale=`link`"))
            row.kind in (:normal, :student_t) || throw(ArgumentError(
                "prior_table row $i: source fixed effects support Normal and Student-t only"))
            selected = row.coef == "all" ? collect(eachindex(coefficients)) : begin
                idx = findfirst(==(row.coef), coefficients)
                idx === nothing && throw(ArgumentError(
                    "prior_table row $i: unknown coefficient `$(row.coef)`; expected one of $(join(coefficients, ", ")) or `all`"))
                [idx]
            end
            isempty(selected) && throw(ArgumentError(
                "prior_table row $i: `coef=all` does not select any fixed-effect coefficients"))
            for idx in selected
                coefficient = coefficients[idx]
                key = (:fixef, coefficient)
                key in seen && throw(ArgumentError(
                    "prior_table has a duplicate or overlapping target `$(coefficient)`"))
                push!(seen, key)
                push!(resolved, (target=:fixef, kind=:fixed_effect, coefficient=Symbol(coefficient),
                    parameter_index=idx, distribution=row.kind, args=copy(row.args),
                    measure=:link, jacobian=false, once_per_target=true))
            end
        elseif row.class === :unique_variance
            row.kind === :gamma || throw(ArgumentError("prior_table row $i: unique_variance supports Gamma only"))
            isempty(row.coef) && isempty(row.dpar) || throw(ArgumentError(
                "prior_table row $i: unique_variance requires empty `coef` and `dpar`"))
            isempty(row.group) || throw(ArgumentError("prior_table row $i: unique_variance is selected by `trait` and `kernel`, not `group`"))
            row.scale == "natural_sd" || throw(ArgumentError(
                "prior_table row $i: unique_variance requires scale=`natural_sd`"))
            row.trait in traits || throw(ArgumentError("prior_table row $i: unknown trait `$(row.trait)`"))
            kernel = if isempty(row.kernel)
                length(sources) == 1 || throw(ArgumentError(
                    "prior_table row $i: `kernel` is required when multiple source kernels are present"))
                String(first(sources).name)
            else
                row.kernel
            end
            kernel in String.(getfield.(sources, :name)) || throw(ArgumentError(
                "prior_table row $i: unknown kernel `$(row.kernel)`"))
            key = (:unique_variance, "$(kernel)::$(row.trait)")
            key in seen && throw(ArgumentError("prior_table has a duplicate Psi target `$(key[2])`"))
            push!(seen, key)
            mean_sd, shape = row.args
            push!(psi_rows, (kernel=Symbol(kernel), trait=Symbol(row.trait), shape=shape,
                rate=shape / mean_sd))
        else
            throw(ArgumentError("prior_table row $i: `$(row.class)` is not supported by fit_gaussian_sources"))
        end
    end
    psi_targets = isempty(psi_rows) ? NamedTuple[] :
        _resolve_psi_sd_priors(psi_rows, sources, traits, length(traits), length(coefficients))
    append!(resolved, psi_targets)
    return resolved, traits
end

function _resolve_binomial_prior_table(table, p::Integer;
        response_names = nothing, legacy_loading_ridge::Real = Inf)
    isfinite(legacy_loading_ridge) && legacy_loading_ridge <= 0 &&
        throw(ArgumentError("loading_ridge must be positive or Inf"))
    names = response_names === nothing ? ["trait_$i" for i in 1:p] : String.(response_names)
    length(names) == p || throw(ArgumentError(
        "response_names must contain one name for each of the $p response rows"))
    all(n -> !isempty(strip(n)), names) || throw(ArgumentError("response_names cannot contain empty names"))
    length(unique(names)) == p || throw(ArgumentError("response_names must be unique"))
    rows = _parse_gllvm_prior_rows(table)
    resolved = GLLVMResolvedPrior[]
    used = Set{Tuple{Symbol,String}}()
    loading_lambda = nothing
    intercept_names = [string(n, "::(Intercept)") for n in names]
    for (row_index, row) in enumerate(rows)
        isempty(row.trait) && isempty(row.kernel) || throw(ArgumentError(
            "prior_table row $row_index: `trait` and `kernel` selectors are supported only for unique-variance priors"))
        if row.class === :fixef
            row.dpar == "mu" || throw(ArgumentError("prior_table row $row_index: GLLVM response fixed effects require dpar=`mu`"))
            isempty(row.group) || throw(ArgumentError("prior_table row $row_index: GLLVM response fixed effects do not accept `group`"))
            row.scale == "link" || throw(ArgumentError("prior_table row $row_index: response fixed effects require scale=`link`"))
            row.kind in (:normal, :student_t) || throw(ArgumentError(
                "prior_table row $row_index: response fixed effects support Normal and Student-t priors only"))
            selected = row.coef == "all" ? collect(eachindex(names)) : begin
                idx = findfirst(==(row.coef), intercept_names)
                idx === nothing && throw(ArgumentError(
                    "prior_table row $row_index: unknown response coefficient `$(row.coef)`; expected one of $(join(intercept_names, ", ")) or `all`"))
                [idx]
            end
            for idx in selected
                target = intercept_names[idx]
                key = (:fixef, target)
                key in used && throw(ArgumentError("prior_table has duplicate or overlapping selector for `$target`"))
                push!(used, key)
                push!(resolved, GLLVMResolvedPrior(row.kind, :fixef, target, idx, row.scale, copy(row.args)))
            end
        elseif row.class === :loading
            row.dpar == "mu" || throw(ArgumentError("prior_table row $row_index: loading rows require dpar=`mu`"))
            isempty(row.group) || throw(ArgumentError("prior_table row $row_index: loading rows do not accept `group`"))
            row.scale == "loading_coef" || throw(ArgumentError("prior_table row $row_index: loading rows require scale=`loading_coef`"))
            row.coef == "all" && row.kind === :ridge || throw(ArgumentError(
                "prior_table row $row_index: this route supports only one whole-loading ridge(lambda) row with coef=`all`"))
            loading_lambda === nothing || throw(ArgumentError("prior_table contains more than one loading ridge row"))
            isfinite(legacy_loading_ridge) && throw(ArgumentError(
                "a table loading ridge cannot be combined with the legacy finite loading_ridge keyword"))
            loading_lambda = row.args[1]
            push!(resolved, GLLVMResolvedPrior(:ridge, :loading, "all", 0, row.scale, copy(row.args)))
        else
            label = row.class === :unique_variance ? "Psi/unique_variance" :
                row.class === :correlation ? "correlation" : string(row.class)
            throw(ArgumentError("prior_table row $row_index: `$label` priors are not supported by this GLLVM Binomial Laplace route"))
        end
    end
    return GLLVMPriorResolution(resolved, names, loading_lambda)
end

function _resolve_gaussian_intercept_prior_table(table, p::Integer; response_names = nothing)
    names = response_names === nothing ? ["trait_$i" for i in 1:p] : String.(response_names)
    length(names) == p || throw(ArgumentError(
        "response_names must contain one name for each of the $p response rows"))
    all(n -> !isempty(strip(n)), names) || throw(ArgumentError("response_names cannot contain empty names"))
    length(unique(names)) == p || throw(ArgumentError("response_names must be unique"))
    rows = _parse_gllvm_prior_rows(table)
    resolved = GLLVMResolvedPrior[]
    used = Set{Tuple{Symbol,String}}()
    residual_seen = false
    intercept_names = [string(n, "::(Intercept)") for n in names]
    for (row_index, row) in enumerate(rows)
        isempty(row.trait) && isempty(row.kernel) || throw(ArgumentError(
            "prior_table row $row_index: `trait` and `kernel` selectors are supported only for unique-variance priors"))
        if row.class === :fixef
            row.dpar == "mu" || throw(ArgumentError("prior_table row $row_index: Gaussian mean coefficients require dpar=`mu`"))
            isempty(row.group) || throw(ArgumentError("prior_table row $row_index: Gaussian mean coefficients do not accept `group`"))
            row.scale == "link" || throw(ArgumentError("prior_table row $row_index: Gaussian mean coefficients require scale=`link`"))
            row.kind in (:normal, :student_t) || throw(ArgumentError(
                "prior_table row $row_index: Gaussian fixed effects support Normal and Student-t priors only"))
            selected = row.coef == "all" ? collect(eachindex(names)) : begin
                idx = findfirst(==(row.coef), intercept_names)
                idx === nothing && throw(ArgumentError(
                    "prior_table row $row_index: unknown Gaussian response coefficient `$(row.coef)`; expected one of $(join(intercept_names, ", ")) or `all`"))
                [idx]
            end
            for idx in selected
                target = intercept_names[idx]
                key = (:fixef, target)
                key in used && throw(ArgumentError("prior_table has duplicate or overlapping selector for `$target`"))
                push!(used, key)
                push!(resolved, GLLVMResolvedPrior(row.kind, :fixef, target, idx, row.scale, copy(row.args)))
            end
        elseif row.class === :residual_sd
            residual_seen && throw(ArgumentError("prior_table supports only one shared Gaussian residual-SD target"))
            row.kind === :gamma || throw(ArgumentError("prior_table row $row_index: residual_sd supports Gamma only"))
            isempty(row.group) || throw(ArgumentError("prior_table row $row_index: residual_sd does not accept `group`"))
            row.dpar == "sigma" || throw(ArgumentError("prior_table row $row_index: Gaussian residual SD requires dpar=`sigma`"))
            row.coef == "(Intercept)" || throw(ArgumentError("prior_table row $row_index: the intercept-only residual-SD target requires coef=`(Intercept)`"))
            row.scale == "natural_sd" || throw(ArgumentError("prior_table row $row_index: Gaussian residual SD requires scale=`natural_sd`"))
            residual_seen = true
            push!(resolved, GLLVMResolvedPrior(:gamma, :residual_sd, "sigma::(Intercept)", 0,
                row.scale, copy(row.args)))
        else
            label = row.class === :unique_variance ? "Psi/unique_variance" :
                row.class === :correlation ? "correlation" : string(row.class)
            throw(ArgumentError("prior_table row $row_index: `$label` priors are not supported by the Gaussian intercept-only route"))
        end
    end
    return GLLVMPriorResolution(resolved, names, nothing)
end

function _gllvm_prior_nll(resolution::GLLVMPriorResolution, beta, loadings)
    total = zero(eltype(beta))
    for row in resolution.rows
        if row.class === :fixef
            value = beta[row.index]
            d = row.kind === :normal ? Normal(row.args[1], row.args[2]) :
                Distributions.LocationScale(row.args[1], row.args[2], TDist(row.args[3]))
            total -= logpdf(d, value)
        elseif row.class === :loading
            total += 0.5 * row.args[1] * sum(abs2, loadings)
        end
    end
    return total
end

function _gllvm_gaussian_intercept_prior_nll(resolution::GLLVMPriorResolution, beta, sigma)
    components = _gllvm_gaussian_intercept_prior_components(resolution, beta, sigma)
    return components.fixed_effect + components.residual_sd
end

function _gllvm_gaussian_intercept_prior_components(resolution::GLLVMPriorResolution, beta, sigma)
    fixed_effect = zero(eltype(beta)) + zero(sigma)
    residual_sd = zero(eltype(beta)) + zero(sigma)
    for row in resolution.rows
        if row.class === :fixef
            value = beta[row.index]
            d = row.kind === :normal ? Normal(row.args[1], row.args[2]) :
                Distributions.LocationScale(row.args[1], row.args[2], TDist(row.args[3]))
            fixed_effect -= logpdf(d, value)
        elseif row.class === :residual_sd
            mean_sd, shape = row.args
            residual_sd -= logpdf(Gamma(shape, mean_sd / shape), sigma)
        end
    end
    return (; fixed_effect, residual_sd)
end

function _gllvm_gaussian_intercept_prior_objective(theta, data::AbstractMatrix,
        resolution::GLLVMPriorResolution)
    p, n = size(data)
    beta = view(theta, 1:p)
    sigma = exp(theta[p + 1])
    isfinite(sigma) && sigma > 0 || return convert(eltype(theta), _NLL_SENTINEL)
    residual = data .- reshape(beta, p, 1)
    return length(data) * (log(sigma) + 0.5log(2π)) +
        0.5sum(abs2, residual) / sigma^2 +
        _gllvm_gaussian_intercept_prior_nll(resolution, beta, sigma)
end

function _gllvm_prior_gradient!(gradient, resolution::GLLVMPriorResolution, beta, loadings, p::Int, rr::Int)
    for row in resolution.rows
        if row.class === :fixef
            δ = beta[row.index] - row.args[1]
            if row.kind === :normal
                gradient[row.index] += δ / row.args[2]^2
            else
                df, scale² = row.args[3], row.args[2]^2
                gradient[row.index] += (df + 1) * δ / (df * scale² + δ^2)
            end
        elseif row.class === :loading
            gradient[(p + 1):(p + rr)] .+= row.args[1] .* pack_lambda(loadings)
        end
    end
    return gradient
end

function _gllvm_prior_target_names(resolution::GLLVMPriorResolution)
    return [row.target for row in resolution.rows]
end

function _fit_gaussian_intercept_prior_table(Y::AbstractMatrix, resolution::GLLVMPriorResolution;
        g_tol::Real = 1e-6, iterations::Integer = 500, optimizer = nothing)
    p, n = size(Y)
    n >= 2 || throw(ArgumentError("the Gaussian intercept-only prior route needs at least two observations"))
    all(isfinite, Y) || throw(ArgumentError("the Gaussian intercept-only prior route requires finite complete responses"))
    data = Matrix{Float64}(Y)
    all(t -> sum(abs2, data[t, :] .- mean(view(data, t, :))) > 0, 1:p) ||
        throw(ArgumentError("each response must vary across observations in the Gaussian intercept-only prior route"))
    isfinite(g_tol) && g_tol > 0 || throw(ArgumentError("g_tol must be positive and finite"))
    iterations > 0 || throw(ArgumentError("iterations must be positive"))
    beta0 = vec(mean(data; dims = 2))
    residual = data .- beta0
    sigma0 = sqrt(sum(abs2, residual) / length(data))
    sigma0 > 0 && isfinite(sigma0) || throw(ArgumentError("the Gaussian intercept-only fit has no finite residual-SD start"))
    theta0 = vcat(beta0, log(sigma0))
    obs = length(data)
    function data_nll(theta)
        beta = view(theta, 1:p)
        sigma = exp(theta[p + 1])
        isfinite(sigma) && sigma > 0 || return convert(eltype(theta), _NLL_SENTINEL)
        resid = data .- reshape(beta, p, 1)
        return obs * (log(sigma) + 0.5log(2π)) + 0.5sum(abs2, resid) / sigma^2
    end
    objective(theta) = _gllvm_gaussian_intercept_prior_objective(theta, data, resolution)
    method = _optimizer_method(optimizer, Optim.LBFGS(); route = "fit_gllvm prior-table Gaussian intercept route")
    options = Optim.Options(g_tol = Float64(g_tol), iterations = Int(iterations), show_trace = false)
    started = time()
    result = Optim.optimize(objective, theta0, method, options; autodiff = :forward)
    elapsed = time() - started
    theta = Optim.minimizer(result)
    data_value = data_nll(theta)
    prior_value = objective(theta) - data_value
    prior_components = _gllvm_gaussian_intercept_prior_components(
        resolution, view(theta, 1:p), exp(theta[p + 1]))
    gradient = _fit_objective_gradient(objective, theta)
    hessian = _fit_objective_hessian(objective, theta)
    hessian_unpenalised = _fit_objective_hessian(data_nll, theta)
    sigma = exp(theta[p + 1])
    boundary = sigma <= 1e-8 * max(1.0, std(vec(data))) ? ["sigma"] : String[]
    lower_boundary = isempty(boundary) ? Int[] : [p + 1]
    lower_boundary_scores = isempty(boundary) ? Dict{Int,Float64}() :
        Dict(p + 1 => _fit_natural_positive_score(gradient[p + 1], sigma))
    projected_gradient = _fit_projected_gradient(gradient; lower = lower_boundary,
        lower_scores = lower_boundary_scores)
    diagnostics = _fit_convergence(optimizer_finished = Optim.converged(result),
        objective = -objective(theta), logLik = -data_value, log_prior = -prior_value,
        penalized = true, gradient = gradient, projected_gradient = projected_gradient,
        hessian = hessian,
        hessian_unpenalised = hessian_unpenalised,
        prior_targets = _gllvm_prior_target_names(resolution),
        boundary_params = boundary, boundary_check_available = true,
        relative_gradient_tol = g_tol)
    pars = (σ_eps = sigma, Λ = zeros(Float64, p, 0), β = collect(Float64, theta[1:p]),
        β_fixed = falses(p), alpha_lv = nothing, Λ_W = nothing, σ²_B = nothing,
        σ²_W = nothing, Λ_phy = nothing, σ_phy = nothing, Σ_phy = nothing,
        θ_packed = collect(Float64, theta), mean_design = :trait_intercepts,
        prior_resolution = resolution,
        prior_metadata = (; resolved_targets = copy(resolution.rows),
            fixed_effect_penalty = Float64(prior_components.fixed_effect),
            residual_sd_penalty = Float64(prior_components.residual_sd),
            total_penalty = Float64(prior_value),
            logLik = -Float64(data_value), log_prior = -Float64(prior_value),
            objective = -Float64(objective(theta))))
    return GllvmFit(GllvmModel(p, 0, 0, false, 0, false), pars, -Float64(data_value),
        Optim.iterations(result), diagnostics.status === :converged, result, elapsed,
        nothing, nothing, diagnostics)
end
