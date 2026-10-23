# Private Destination-B post-fit surface for the grouped Gaussian and sparse
# multivariate-precision fit records.  Integration, exports, and public claims
# belong to the parent lane.

_destination_b_traits(fit) = first(fit.response_shape)

function _destination_b_prediction_design(fit, design)
    p = _destination_b_traits(fit)
    D = if design isa AbstractMatrix
        design
    elseif design isa AbstractArray && ndims(design) == 3
        size(design, 1) == p || throw(DimensionMismatch(
            "three-dimensional design first dimension must equal $p traits"))
        reshape(design, p * size(design, 2), size(design, 3))
    else
        throw(ArgumentError(
            "design must be a trait-major p*n × q matrix or p × n × q array"))
    end
    size(D, 2) == length(fit.beta) || throw(DimensionMismatch(
        "design has $(size(D, 2)) columns; fit requires $(length(fit.beta))"))
    size(D, 1) > 0 && size(D, 1) % p == 0 || throw(DimensionMismatch(
        "design rows must be a positive multiple of $p traits"))
    all(x -> x isa Real && isfinite(x), D) ||
        throw(ArgumentError("design must be finite and real"))
    Df = try
        Matrix{Float64}(D)
    catch
        throw(ArgumentError("design must be representable as Float64"))
    end
    all(isfinite, Df) || throw(ArgumentError("design exceeds the Float64 range"))
    return Df
end

"""
    destination_b_population_predict(fit[, design]; type=:response) -> Matrix

Population (fixed-effect-only) Gaussian mean prediction for a
`GroupedGaussianFit` or `PrecisionMultivariateFit`.  The retained or supplied
trait-major mean design is multiplied by `fit.beta`; all grouped and
phylogenetic effects are set to zero.  This is **not** a conditional prediction
or BLUP, because these fit records do not retain conditional random-effect
modes.  For these Gaussian fits `:link`, `:response`, and `:mean` are equal.
"""
function destination_b_population_predict(
        fit::Union{GroupedGaussianFit,PrecisionMultivariateFit}, design = nothing;
        type::Symbol = :response)
    type in (:link, :response, :mean) || throw(ArgumentError(
        "type must be :link, :response, or :mean; got :$type"))
    D = design === nothing ? fit.mean_design : _destination_b_prediction_design(fit, design)
    size(D, 2) == length(fit.beta) || throw(DimensionMismatch(
        "retained mean design has incompatible coefficient width"))
    p = _destination_b_traits(fit)
    size(D, 1) % p == 0 || throw(DimensionMismatch(
        "mean design rows must be divisible by the trait count"))
    return reshape(D * fit.beta, p, div(size(D, 1), p))
end

# Follow the existing `predict(fit, X)` population-prediction convention while
# keeping the longer name above visible in the private contract.
predict(fit::Union{GroupedGaussianFit,PrecisionMultivariateFit}; kwargs...) =
    destination_b_population_predict(fit; kwargs...)
predict(fit::Union{GroupedGaussianFit,PrecisionMultivariateFit}, design; kwargs...) =
    destination_b_population_predict(fit, design; kwargs...)

StatsAPI.fitted(fit::Union{GroupedGaussianFit,PrecisionMultivariateFit}) =
    destination_b_population_predict(fit)

"""
    residuals(fit::Union{GroupedGaussianFit,PrecisionMultivariateFit}) -> Matrix

Raw observed-minus-population-mean residuals.  They exclude conditional group,
phylogenetic, or latent-effect predictions and are neither standardized nor
BLUP residuals.
"""
StatsAPI.residuals(fit::Union{GroupedGaussianFit,PrecisionMultivariateFit}) =
    fit.response .- StatsAPI.fitted(fit)

StatsAPI.coef(fit::GroupedGaussianFit) = copy(fit.beta)
StatsAPI.loglikelihood(fit::GroupedGaussianFit) = fit.loglik
StatsAPI.nobs(fit::GroupedGaussianFit) = length(fit.response)
StatsAPI.dof(fit::GroupedGaussianFit) = length(fit.parameters)

function _destination_b_covariance_parts(fit::GroupedGaussianFit, level::Symbol)
    p = _destination_b_traits(fit)
    if level === :residual
        return zeros(Float64, p, p), fill(fit.sigma_eps^2, p),
            Matrix(Diagonal(fill(fit.sigma_eps^2, p)))
    end
    index = findfirst(term -> term.name === level, fit.terms)
    index === nothing && throw(ArgumentError(
        "level must name a fitted grouping term ($(join(string.(getfield.(fit.terms, :name)), ", "))) or :residual"))
    q = size(fit.mean_design, 2)
    source_coordinates = length(fit.parameters) - q - 1
    loads, uniques, covariances, used = _grouped_term_unpack(
        view(fit.parameters, (q + 1):(q + source_coordinates)), p, fit.terms)
    used == source_coordinates || throw(ArgumentError("grouped covariance packing is inconsistent"))
    unique = uniques[index] === nothing ? zeros(Float64, p) : collect(Float64, uniques[index])
    return Matrix(loads[index] * loads[index]'), unique, Matrix(covariances[index])
end

function _destination_b_covariance_parts(fit::PrecisionMultivariateFit, level::Symbol)
    p = _destination_b_traits(fit)
    if level === :residual
        return zeros(Float64, p, p), copy(fit.residual_variance),
            Matrix(Diagonal(fit.residual_variance))
    elseif level === :phylo || level === :phy
        shared = Matrix(fit.loading * fit.loading')
        unique = fit.phylo_unique_variance === nothing ? zeros(Float64, p) :
            copy(fit.phylo_unique_variance)
        return shared, unique, shared + Matrix(Diagonal(unique))
    end
    throw(ArgumentError("unsupported level :$(level); level must be :phy (alias :phylo) or :residual"))
end

"""
    extract_Sigma(fit::Union{GroupedGaussianFit,PrecisionMultivariateFit};
                  level, part=:total) -> NamedTuple

Extract a fitted trait covariance.  For grouped fits `level` is a selected
grouping-term name or `:residual`; for precision fits it is `:phy` (R's
level name; `:phylo` is accepted as an alias) or `:residual`.  `:shared` returns the low-rank covariance, `:unique` its diagonal
variance vector, and `:total` the sum with a correlation matrix.  These are
marginal trait covariances, not conditional random-effect covariances.
"""
function extract_Sigma(fit::Union{GroupedGaussianFit,PrecisionMultivariateFit};
        level::Symbol, part::Symbol = :total)
    shared, unique, total = _destination_b_covariance_parts(fit, level)
    if part === :shared
        return (Sigma = shared, level = level, part = part)
    elseif part === :unique
        return (s = unique, level = level, part = part)
    elseif part === :total
        return (Sigma = total, R = _cov2cor(total), level = level, part = part)
    end
    throw(ArgumentError("part must be one of :total, :shared, :unique; got :$part"))
end

function Base.summary(fit::GroupedGaussianFit)
    p, n = fit.response_shape
    return "Grouped Gaussian fit (p=$p, n=$n, terms=$(join(string.(getfield.(fit.terms, :name)), ",")), logLik=$(round(fit.loglik; sigdigits=5)))"
end

function Base.summary(fit::PrecisionMultivariateFit)
    p, n = fit.response_shape
    return "Multivariate precision Gaussian fit (p=$p, n=$n, rank=$(fit.rank), mode=$(fit.mode), residual=$(fit.residual_mode), logLik=$(round(fit.loglik; sigdigits=5)))"
end

function Base.show(io::IO, ::MIME"text/plain", fit::GroupedGaussianFit)
    println(io, summary(fit))
    println(io, "  convergence = ", fit.converged, " (", fit.iterations,
        " iterations; reason = ", fit.stopping_reason, ")")
    print(io, "  observed curvature PD = ", fit.hessian_positive_definite,
        "; min eigenvalue = ", round(fit.hessian_min_eigenvalue; sigdigits = 5),
        "; gradient norm = ", round(fit.gradient_norm; sigdigits = 5))
end

function Base.show(io::IO, ::MIME"text/plain", fit::PrecisionMultivariateFit)
    println(io, summary(fit))
    println(io, "  convergence = ", fit.converged, " (", fit.iterations,
        " iterations; reason = ", fit.stopping_reason, ")")
    print(io, "  observed curvature PD = ", fit.hessian_positive_definite,
        "; min eigenvalue = ", round(fit.hessian_min_eigenvalue; sigdigits = 5),
        "; condition = ", round(fit.hessian_condition_number; sigdigits = 5),
        "; gradient norm = ", round(fit.gradient_norm; sigdigits = 5))
end
