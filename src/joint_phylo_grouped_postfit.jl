# Post-fit surface for the private joint phylogenetic-plus-ordinary Gaussian
# record. Public routing and conditional random-effect prediction are excluded.

_joint_phylo_grouped_traits(fit::JointPhyloGroupedGaussianFit) = first(fit.response_shape)

"""
    joint_phylo_grouped_population_predict(fit[, design]; type=:response)

Return the Gaussian population fixed-effect mean `reshape(D*beta, p, n)` for
a private joint phylogenetic-plus-grouped fit. All phylogenetic and ordinary
random effects have mean zero: this equals the marginal mean for the Gaussian
identity link, but is not a conditional-mode/BLUP prediction. No predictive
variance is returned. `:link`, `:response`, and `:mean` coincide.
"""
function joint_phylo_grouped_population_predict(fit::JointPhyloGroupedGaussianFit,
        design = nothing; type::Symbol = :response, kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "unsupported prediction keyword(s): $(join(string.(keys(kwargs)), ", "))"))
    type in (:link, :response, :mean) || throw(ArgumentError(
        "type must be :link, :response, or :mean; got :$type"))
    D = design === nothing ? fit.mean_design : _destination_b_prediction_design(fit, design)
    size(D, 2) == length(fit.beta) || throw(DimensionMismatch(
        "mean design has incompatible coefficient width"))
    p = _joint_phylo_grouped_traits(fit)
    size(D, 1) > 0 && size(D, 1) % p == 0 || throw(DimensionMismatch(
        "mean design rows must be a positive multiple of trait count"))
    prediction = reshape(D * fit.beta, p, div(size(D, 1), p))
    all(isfinite, prediction) || throw(ArgumentError(
        "population prediction is nonfinite; rescale the supplied design"))
    return prediction
end

predict(fit::JointPhyloGroupedGaussianFit; kwargs...) =
    joint_phylo_grouped_population_predict(fit; kwargs...)
predict(fit::JointPhyloGroupedGaussianFit, design; kwargs...) =
    joint_phylo_grouped_population_predict(fit, design; kwargs...)

StatsAPI.fitted(fit::JointPhyloGroupedGaussianFit) =
    joint_phylo_grouped_population_predict(fit)

"""
    residuals(fit::JointPhyloGroupedGaussianFit) -> Matrix

Return raw observed-minus-population-mean residuals. These do not condition on
or predict retained phylogenetic/ordinary random effects.
"""
StatsAPI.residuals(fit::JointPhyloGroupedGaussianFit) =
    fit.response .- StatsAPI.fitted(fit)

function _joint_phylo_grouped_postfit_objective(fit::JointPhyloGroupedGaussianFit)
    return _joint_phylo_grouped_nll(fit.response, fit.phy, fit.terms, fit.incidences;
        rank = fit.rank, phylo_mode = fit.phylo_mode, species_id = fit.species_id,
        mean_design = fit.mean_design)
end

function _joint_phylo_grouped_fixed_information(fit::JointPhyloGroupedGaussianFit)
    component = _joint_phylo_grouped_component_identification(fit)
    component.reason === :invalid && return (status=:invalid_covariance_tangent,
        covariance=nothing, gradient_norm=NaN)
    # This exact structural result must not be masked by a stale/nonconverged
    # optimizer receipt in the generic observed-information helper.
    component.reason === :nonidentifiable && return (status=:nonidentifiable,
        covariance=nothing, gradient_norm=NaN)
    return _destination_b_fixed_effect_information(_joint_phylo_grouped_postfit_objective(fit),
        fit.parameters, length(fit.beta); converged = fit.converged,
        structural_redundancy = fit.phylo_mode === :explicitunique &&
            rr_theta_len(first(fit.response_shape), fit.rank) + first(fit.response_shape) >
            first(fit.response_shape) * (first(fit.response_shape) + 1) ÷ 2)
end

function _joint_phylo_grouped_require_fixed_covariance(fit::JointPhyloGroupedGaussianFit)
    result = _joint_phylo_grouped_fixed_information(fit)
    result.status === :available || throw(ArgumentError(
        "fixed-effect observed-marginal covariance unavailable: $(result.status)"))
    return result.covariance
end

"""
    StatsAPI.vcov(fit::JointPhyloGroupedGaussianFit)

Return the fixed-effect block of the inverse full observed marginal information,
including phylogenetic, residual, and ordinary-group nuisance coordinates. No
fixed-block-only inverse or diagonal fallback is used.
"""
StatsAPI.vcov(fit::JointPhyloGroupedGaussianFit) =
    _joint_phylo_grouped_require_fixed_covariance(fit)

"""
    StatsAPI.stderror(fit::JointPhyloGroupedGaussianFit)

Return square roots of the diagonal of the full-nuisance fixed-effect
covariance. It throws the same diagnostic as [`StatsAPI.vcov`](@ref) when
observed-marginal uncertainty is unavailable.
"""
StatsAPI.stderror(fit::JointPhyloGroupedGaussianFit) =
    sqrt.(diag(StatsAPI.vcov(fit)))

function _joint_phylo_grouped_covariance_parts(fit::JointPhyloGroupedGaussianFit,
        level::Symbol)
    p = _joint_phylo_grouped_traits(fit)
    if level === :phylo
        shared = Matrix(fit.loading * fit.loading')
        unique = fit.phylo_unique_variance === nothing ? zeros(Float64, p) :
            copy(fit.phylo_unique_variance)
        return (shared = shared, unique = unique,
            total = copy(fit.phylo_covariance), source = :phylogenetic)
    elseif level === :residual
        unique = copy(fit.residual_variance)
        return (shared = zeros(Float64, p, p), unique = unique,
            total = Matrix(Diagonal(unique)), source = :observation_residual)
    end
    index = findfirst(term -> term.name === level, fit.terms)
    index === nothing && throw(ArgumentError(
        "level must be :phylo, :residual, or a fitted ordinary term ($(join(string.(getfield.(fit.terms, :name)), ", ")))"))
    total = copy(fit.ordinary_covariances[index])
    return (shared = zeros(Float64, p, p), unique = collect(diag(total)),
        total = total, source = :ordinary)
end

"""
    extract_Sigma(fit::JointPhyloGroupedGaussianFit; level, part=:total)

Extract one explicitly labelled trait covariance source. `level=:phylo` returns
the phylogenetically correlated trait covariance; `:residual` returns the
independent observation residual covariance; an ordinary term name returns its
independent grouped trait covariance. Sources are never combined.
"""
function extract_Sigma(fit::JointPhyloGroupedGaussianFit; level::Symbol,
        part::Symbol = :total)
    parts = _joint_phylo_grouped_covariance_parts(fit, level)
    if part === :shared
        return (Sigma = parts.shared, level = level, part = part, source = parts.source)
    elseif part === :unique
        return (s = parts.unique, level = level, part = part, source = parts.source)
    elseif part === :total
        return (Sigma = parts.total, R = _cov2cor(parts.total), level = level,
            part = part, source = parts.source)
    end
    throw(ArgumentError("part must be one of :total, :shared, :unique; got :$part"))
end

function _joint_phylo_grouped_species_term(fit::JointPhyloGroupedGaussianFit,
        species_level)
    species_level isa Symbol || throw(ArgumentError(
        "species_level must be a Symbol naming a fitted ordinary grouping term"))
    index = findfirst(term -> term.name === species_level, fit.terms)
    index === nothing && throw(ArgumentError(
        "species_level must name a fitted ordinary grouping term ($(join(string.(getfield.(fit.terms, :name)), ", "))); got :$species_level"))
    n = length(fit.species_id)
    partition_error = ArgumentError("species_level=:$(species_level) must induce the " *
        "same observation partition as species_id; a term on another grouping " *
        "(for example a site term) is not species-level variance")
    rows, groups, weights = findnz(fit.incidences[index])
    group_of = zeros(Int, n)
    for (row, group, weight) in zip(rows, groups, weights)
        (weight == 1.0 && group_of[row] == 0) || throw(partition_error)
        group_of[row] = group
    end
    all(>(0), group_of) || throw(partition_error)
    species_to_group = Dict{Int,Int}()
    group_to_species = Dict{Int,Int}()
    for observation in 1:n
        species, group = fit.species_id[observation], group_of[observation]
        get!(species_to_group, species, group) == group &&
            get!(group_to_species, group, species) == species || throw(partition_error)
    end
    return index
end

"""
    extract_phylo_signal(fit::JointPhyloGroupedGaussianFit; species_level, ci = false)
        -> NamedTuple

Per-trait phylogenetic signal of a joint phylogenetic-plus-ordinary Gaussian
fit, the point estimate of `gllvmTMB::extract_phylo_signal()` (R/extract-omega.R:592-623
at the P1 pin). With `a[t] = diag(Sigma_phy)[t]` (shared plus phylogenetic unique
variance) and `u[t]` the diagonal of the ordinary term named by `species_level`,
`V_eta[t] = a[t] + u[t]`, `H2 = a / V_eta`, `Psi = u / V_eta` and `C2_non = 0`
(an ordinary diagonal term carries no shared latent part). Observation residual
variance and every ordinary term not on the species grouping are excluded from
`V_eta`, as R excludes them.

`species_level` is required. It must name an ordinary term whose observation
partition equals the partition given by `fit.species_id`, so a site term cannot
be mistaken for species-level variance. `ci = true` is refused. A fit whose
covariance components are not separately identifiable is refused, because the
split between `a`, `u` and the residual is then not determined by the data.

Returns the column table `(trait, H2, C2_non, Psi, V_eta)`. This method makes no
claim of numerical agreement with R: the fit itself differs (per-trait residual
variance here, one shared residual SD in R's Gaussian route).
"""
function extract_phylo_signal(fit::JointPhyloGroupedGaussianFit;
        species_level = nothing, ci::Bool = false)
    ci && throw(ArgumentError("phylogenetic signal intervals are outside the " *
        "joint phylo-grouped point-estimate scope (GJL-GATE-PHYLO-LATENT-SIGNAL-CI)"))
    species_level === nothing && throw(ArgumentError(
        "species_level is required to identify the species-grouped variance source"))
    index = _joint_phylo_grouped_species_term(fit, species_level)
    component = _joint_phylo_grouped_component_identification(fit)
    component.reason === :invalid && throw(ArgumentError(
        "cannot verify covariance-component identification " *
        "(GJL-GATE-PHYLO-LATENT-SIGNAL-IDENTIFICATION-INVALID)"))
    component.reason === :identified || throw(ArgumentError(
        "extract_phylo_signal requires separately identified covariance components " *
        "(GJL-GATE-PHYLO-LATENT-SIGNAL-IDENTIFICATION)"))
    a = collect(Float64, diag(fit.phylo_covariance))
    u = collect(Float64, diag(fit.ordinary_covariances[index]))
    v_eta = a .+ u
    ok = [isfinite(v) && v > 0 for v in v_eta]
    p = _joint_phylo_grouped_traits(fit)
    return (trait = ["trait$(t)" for t in 1:p],
        H2 = [ok[t] ? a[t] / v_eta[t] : NaN for t in 1:p],
        C2_non = zeros(p),
        Psi = [ok[t] ? u[t] / v_eta[t] : NaN for t in 1:p],
        V_eta = v_eta)
end

function _joint_phylo_grouped_fixed_summary(fit::JointPhyloGroupedGaussianFit,
        Y::AbstractMatrix)
    size(Y) == fit.response_shape && Matrix{Float64}(Y) == fit.response ||
        throw(ArgumentError("Y must exactly match the response retained by this fit"))
    result = _joint_phylo_grouped_fixed_information(fit)
    rows = NamedTuple[]
    if result.status === :available
        se = sqrt.(diag(result.covariance))
        for index in eachindex(fit.beta)
            estimate = fit.beta[index]
            push!(rows, (name = String(fit.coefficient_names[index]), estimate = estimate,
                se = se[index], lower = estimate - 1.959963984540054 * se[index],
                upper = estimate + 1.959963984540054 * se[index], status = :available,
                method = :observed_marginal_wald))
        end
    else
        for index in eachindex(fit.beta)
            push!(rows, (name = String(fit.coefficient_names[index]), estimate = fit.beta[index],
                se = NaN, lower = NaN, upper = NaN, status = result.status,
                method = :unavailable))
        end
    end
    return DestinationBFixedEffectSummary(rows, result.status, result.gradient_norm,
        fit.loglik, fit.converged, fit.gradient_norm, fit.hessian_positive_definite,
        fit.hessian_min_eigenvalue, fit.iterations, fit.stopping_reason)
end

"""
    summary(fit::JointPhyloGroupedGaussianFit, Y)

Return a structured 95% observed-marginal fixed-effect summary after checking
that `Y` exactly matches retained fit provenance. The stored `gradient_norm`
is distinct from the recomputed `inference_gradient_norm`; unavailable full
curvature remains a diagnostic rather than a fabricated interval.
"""
Base.summary(fit::JointPhyloGroupedGaussianFit, Y::AbstractMatrix) =
    _joint_phylo_grouped_fixed_summary(fit, Y)

function Base.summary(fit::JointPhyloGroupedGaussianFit)
    p, n = fit.response_shape
    component = _joint_phylo_grouped_component_identification(fit)
    return "Joint phylo-plus-grouped Gaussian fit (p=$p, n=$n, rank=$(fit.rank), terms=$(join(string.(getfield.(fit.terms, :name)), ",")), component_identification=$(component.reason), logLik=$(round(fit.loglik; sigdigits=5)))"
end

function Base.show(io::IO, ::MIME"text/plain", fit::JointPhyloGroupedGaussianFit)
    println(io, summary(fit))
    println(io, "  convergence = ", fit.converged, " (", fit.iterations,
        " iterations; reason = ", fit.stopping_reason, ")")
    println(io, "  covariance-component identification = ",
        _joint_phylo_grouped_component_identification(fit).reason)
    print(io, "  observed curvature PD = ", fit.hessian_positive_definite,
        "; min eigenvalue = ", round(fit.hessian_min_eigenvalue; sigdigits = 5),
        "; gradient norm = ", round(fit.gradient_norm; sigdigits = 5))
end
