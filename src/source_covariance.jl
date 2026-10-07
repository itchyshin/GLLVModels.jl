"""
    _gaussian_source_loglik(Y, beta, loadings, covariances, groups, sigma_eps)

Internal normalized Gaussian marginal log density with additive source covariance.
`Y` is traits × units, `beta` contains trait means, and each column of `loadings`
is one source's trait loading vector. `covariances[r]` is a known, exactly
symmetric positive-definite group covariance; `groups[i,r]` indexes the group
of unit `i` for source `r`. One or two sources are currently admitted.

Residual noise has independent variance `sigma_eps^2`. Source covariances are
projected through group incidence before trait loadings are applied. Repeated
groups are allowed. Inputs must be finite and the residual SD strictly positive.
No centering, ridge, covariance estimation, optimization or missing-data handling
is performed. This dense evaluator requires O((p*n)^2) memory and O((p*n)^3)
factorization work. It is not the matrix-normal coevolution model.

This evaluator uses the documented mathematical contract and current evidence
boundaries.
"""
function _gaussian_source_loglik(Y::AbstractMatrix{<:Real},
        beta::AbstractVector{<:Real}, loadings::AbstractMatrix{<:Real},
        covariances, groups::AbstractMatrix{<:Integer}, sigma_eps::Real)
    p, n = size(Y)
    p > 0 && n > 0 || throw(ArgumentError("Y must have traits and units"))
    nr = size(loadings, 2)
    nr in (1, 2) || throw(ArgumentError("one or two rank-one sources are supported"))
    length(beta) == p || throw(DimensionMismatch("beta must have one mean per trait"))
    size(loadings, 1) == p || throw(DimensionMismatch("loadings must have one row per trait"))
    length(covariances) == nr || throw(DimensionMismatch("one covariance is required per source"))
    size(groups) == (n, nr) || throw(DimensionMismatch("groups must be units by sources"))
    isfinite(sigma_eps) && sigma_eps > 0 || throw(ArgumentError("residual SD must be finite and positive"))
    all(isfinite, Y) && all(isfinite, beta) && all(isfinite, loadings) ||
        throw(ArgumentError("responses, means and loadings must be finite"))
    for r in 1:nr
        C = covariances[r]
        C isa AbstractMatrix{<:Real} || throw(ArgumentError("source covariance must be a real matrix"))
        size(C, 1) == size(C, 2) && size(C, 1) > 0 ||
            throw(DimensionMismatch("source covariance must be nonempty and square"))
        C = _symmetric_admit(C, "source covariance")
        all(g -> 1 <= g <= size(C, 1), view(groups, :, r)) ||
            throw(ArgumentError("group index outside source covariance"))
        cholesky(Symmetric(C)) # reject invalid source covariance even if noise masks it
    end
    T = promote_type(Float64, eltype(Y), eltype(beta), eltype(loadings),
                     typeof(sigma_eps), map(eltype, covariances)...)
    V = zeros(T, p*n, p*n)
    for k in axes(V, 1)
        V[k,k] = sigma_eps^2
    end
    for r in 1:nr
        C = covariances[r]
        for j in 1:n, i in 1:n
            cij = C[groups[i,r], groups[j,r]]
            for b in 1:p, a in 1:p
                V[a+p*(i-1), b+p*(j-1)] += loadings[a,r]*loadings[b,r]*cij
            end
        end
    end
    residual = vec(Y .- beta)
    factor = cholesky(Symmetric(V))
    return -(p*n*log(2pi) + logdet(factor) + dot(residual, factor\residual))/2
end
