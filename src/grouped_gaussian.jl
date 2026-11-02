"""
    _grouped_gaussian_factor_nll(Y, beta, incidences, loadings, sigma_eps;
                                 uniques=nothing) -> Float64

Exact negative Gaussian marginal log-likelihood for fixed, additive grouped
factor sources. `Y` is traits × units, `beta` is a trait mean, and source `s`
has a sparse units × levels incidence matrix `incidences[s]` plus a direct
traits × rank loading matrix `loadings[s]`. Its level-specific factor scores
have independent unit-Normal priors. Sources may be shared or crossed.

`uniques[s]`, when supplied, is a non-negative traits-long vector of additional
unique variances for source `s`; only positive coordinates add a factor column,
so a bare low-rank block receives no diagonal ridge. This routine forms sparse
`W = [kron(Z_s, L_s^*)]` and sparse random-effect precision
`I + W'W / sigma_eps^2`; it never forms response covariance `V`.

The sparse CHOLMOD solve currently requires native `Float64` inputs. Other
scalar types (including AD duals) are rejected explicitly rather than coerced;
this internal kernel makes no sparse-factor AD or finite-difference claim.
"""
function _grouped_gaussian_factor_nll(Y::AbstractMatrix{<:Real},
        beta::AbstractVector{<:Real}, incidences::AbstractVector,
        loadings::AbstractVector, sigma_eps::Real; uniques=nothing)
    p, n = size(Y)
    p > 0 && n > 0 || throw(ArgumentError("Y must have positive trait and unit dimensions"))
    length(beta) == p || throw(DimensionMismatch("beta must have one entry per trait"))
    length(incidences) == length(loadings) ||
        throw(DimensionMismatch("one loading matrix is required per incidence source"))
    sigma_eps isa Float64 && isfinite(sigma_eps) && sigma_eps > 0 ||
        throw(ArgumentError("sparse grouped factor kernel requires a finite positive Float64 sigma_eps"))
    eltype(Y) === Float64 && eltype(beta) === Float64 ||
        throw(ArgumentError("sparse grouped factor kernel requires Float64 responses and means"))
    all(isfinite, Y) && all(isfinite, beta) ||
        throw(ArgumentError("Y and beta must be finite"))

    nsources = length(incidences)
    unique_values = if uniques === nothing
        fill(nothing, nsources)
    else
        length(uniques) == nsources ||
            throw(DimensionMismatch("one unique-variance vector is required per incidence source"))
        collect(uniques)
    end
    designs = SparseMatrixCSC{Float64,Int}[]
    for s in eachindex(incidences)
        Z = incidences[s]
        L = loadings[s]
        Z isa SparseMatrixCSC{Float64,Int} ||
            throw(ArgumentError("incidence matrices must be SparseMatrixCSC{Float64,Int}"))
        L isa AbstractMatrix{<:Real} && eltype(L) === Float64 ||
            throw(ArgumentError("loading matrices must have Float64 entries"))
        size(Z, 1) == n ||
            throw(DimensionMismatch("each incidence matrix must have one row per unit"))
        size(Z, 2) > 0 ||
            throw(DimensionMismatch("each incidence matrix must have at least one factor level"))
        size(L, 1) == p && size(L, 2) > 0 ||
            throw(DimensionMismatch("each loading matrix must be traits by positive factor rank"))
        all(isfinite, Z) && all(isfinite, L) ||
            throw(ArgumentError("incidence matrices and loadings must be finite"))

        d = unique_values[s]
        Lstar = if d === nothing
            L
        else
            d isa AbstractVector{<:Real} && length(d) == p && eltype(d) === Float64 ||
                throw(DimensionMismatch("each unique-variance vector must have one Float64 value per trait"))
            all(isfinite, d) && all(>=(0.0), d) ||
                throw(ArgumentError("unique variances must be finite and non-negative"))
            active = findall(>(0.0), d)
            if isempty(active)
                L
            else
                U = zeros(Float64, p, length(active))
                for (j, trait) in enumerate(active)
                    U[trait, j] = sqrt(d[trait])
                end
                hcat(L, U)
            end
        end
        push!(designs, sparse(kron(Z, Lstar)))
    end

    m = p * n
    sigma2 = sigma_eps^2
    residual = vec(Y .- beta)
    nsources == 0 && return (m * log(2pi) + m * log(sigma2) +
                              dot(residual, residual) / sigma2) / 2

    W = reduce(hcat, designs)
    random_dimension = size(W, 2)
    precision = spdiagm(0 => ones(Float64, random_dimension)) + (W' * W) / sigma2
    factor_precision = cholesky(Symmetric(precision))
    h = W' * residual / sigma2
    # Algebraically this equals r'r/sigma² - h'K⁻¹h, but that form subtracts
    # two enormous nearly equal values when residual noise is tiny. Evaluating
    # at the posterior random-effect mode is positive and stable instead.
    mode = factor_precision \ h
    mode_residual = residual - W * mode
    quadratic = dot(mode_residual, mode_residual) / sigma2 + dot(mode, mode)
    return (m * log(2pi) + m * log(sigma2) + logdet(factor_precision) + quadratic) / 2
end

"""
    _grouped_gaussian_nll(Y, beta, incidences, trait_covariances, sigma_eps) -> Float64

Compatibility form of [`_grouped_gaussian_factor_nll`](@ref) for fixed,
positive-definite trait covariance blocks. Each `B_s` is Cholesky-factorised
to a full-rank loading matrix before dispatch. The direct low-rank factor
interface is the model kernel; this convenience form cannot represent a bare
singular factor covariance.
"""
function _grouped_gaussian_nll(Y::AbstractMatrix{<:Real},
        beta::AbstractVector{<:Real}, incidences::AbstractVector,
        trait_covariances::AbstractVector, sigma_eps::Real)
    length(incidences) == length(trait_covariances) ||
        throw(DimensionMismatch("one trait covariance is required per incidence source"))
    p = size(Y, 1)
    factors = Matrix{Float64}[]
    for B in trait_covariances
        B isa AbstractMatrix{<:Real} && size(B) == (p, p) ||
            throw(DimensionMismatch("each trait covariance must be traits by traits"))
        B = _symmetric_admit(Matrix{Float64}(B), "trait covariance")
        factor = try
            cholesky(Symmetric(B))
        catch err
            err isa PosDefException || rethrow()
            throw(ArgumentError("trait covariances must be positive definite"))
        end
        push!(factors, Matrix(factor.L))
    end
    return _grouped_gaussian_factor_nll(Y, beta, incidences, factors, sigma_eps)
end
