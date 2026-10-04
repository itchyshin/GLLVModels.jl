# Accept covariance / precision matrices that are symmetric up to floating-point
# round-off; reject larger asymmetry with a measured error.

function _symmetric_admit(C::AbstractMatrix{<:Real}, label::AbstractString)
    all(isfinite, C) || throw(ArgumentError("$(label) must be finite"))
    δ = maximum(abs, C - C')
    scale = max(1.0, maximum(abs, C))
    if δ > 1e-10 * scale
        throw(ArgumentError(
            "$(label) must be symmetric within relative tolerance 1e-10 " *
            "(max |C - C'| = $(δ))"))
    end
    return issymmetric(C) ? C : (C + C') ./ 2
end

function _symmetric_sparse_admit(Q::AbstractMatrix, label::AbstractString)
    δ = norm(Q - Q')
    scale = max(1.0, norm(Q))
    if δ > 1e-10 * scale
        throw(ArgumentError(
            "$(label) must be symmetric within relative tolerance 1e-10 " *
            "(Frobenius |Q - Q'| = $(δ))"))
    end
    return Q
end
