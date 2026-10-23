# Packing/unpacking of reduced-rank Λ matrices matching gllvmTMB.cpp:343-376.
#
# A p × K loading matrix Λ with strict upper triangle = 0 is stored as a
# flat vector of length rr_theta_len(p, K) = p*K - K*(K-1)/2 .
#
# Layout (1-based):
#   θ[1:K]     = diag(Λ)  i.e. Λ[k, k] for k = 1:K
#   θ[K+1:end] = strict lower entries, packed column-by-column:
#                Λ[2,1], Λ[3,1], …, Λ[p,1],
#                Λ[3,2], …, Λ[p,2],
#                …,
#                Λ[K+1,K], …, Λ[p,K]
#
# Sign convention: diagonals are passed through unchanged (raw). The
# engine's positive-diagonal sign anchor is enforced at the parameter
# vector level (via parameter reparameterisation), not in the packing.

"""
    rr_theta_len(p::Integer, K::Integer) -> Int

Number of parameters needed to pack a p × K lower-triangular loading
matrix with zero strict upper triangle.
"""
function rr_theta_len(p::Integer, K::Integer)
    return Int(p) * Int(K) - Int(K) * (Int(K) - 1) ÷ 2
end

# Internal helper: 1-based position within θ of the strict-lower entry Λ[i, k]
# with k < i. Derived from gllvmTMB.cpp:365
#   lam_lower(j * p - (j + 1) * j / 2 + i - 1 - j)   // 0-based
# with j = k-1 (col, 0-based) and i_c = i-1 (row, 0-based), then shifted by K
# for the leading diagonal block.
@inline function _lower_index(p::Integer, K::Integer, i::Integer, k::Integer)
    # Position within the strict-lower block (1-based): (k-1)*p - k*(k-1)/2 + i - k
    return K + (k - 1) * p - k * (k - 1) ÷ 2 + i - k
end

"""
    unpack_lambda(θ::AbstractVector, p::Integer, K::Integer) -> AbstractMatrix

Inverse of `pack_lambda`. Returns a p × K matrix Λ with the diagonals and
strict-lower entries filled from `θ` and the strict upper triangle = 0.

AD-friendly: `eltype(θ)` is preserved, so `θ::Vector{<:ForwardDiff.Dual}`
returns a matrix of `Dual`s.
"""
function unpack_lambda(θ::AbstractVector, p::Integer, K::Integer)
    n = rr_theta_len(p, K)
    length(θ) == n || throw(ArgumentError(
        "unpack_lambda: θ has length $(length(θ)); expected $n for p=$p, K=$K"))
    T = eltype(θ)
    Λ = zeros(T, p, K)
    # Diagonals: θ[1:K] → Λ[k, k]
    @inbounds for k in 1:K
        Λ[k, k] = θ[k]
    end
    # Strict lower entries: column-by-column
    @inbounds for k in 1:K
        for i in (k + 1):p
            Λ[i, k] = θ[_lower_index(p, K, i, k)]
        end
    end
    return Λ
end

"""
    pack_lambda(Λ::AbstractMatrix) -> AbstractVector

Forward pack: given a p × K loading matrix with strict-upper = 0, return
the flat θ vector of length rr_theta_len(p, K).
"""
function pack_lambda(Λ::AbstractMatrix)
    p, K = size(Λ)
    T = eltype(Λ)
    θ = Vector{T}(undef, rr_theta_len(p, K))
    @inbounds for k in 1:K
        θ[k] = Λ[k, k]
    end
    @inbounds for k in 1:K
        for i in (k + 1):p
            θ[_lower_index(p, K, i, k)] = Λ[i, k]
        end
    end
    return θ
end

"""
    _lower_triangular_loadings(Λ::AbstractMatrix) -> Matrix

Rotate a p × K loading matrix (e.g. SVD loadings used as a start) to the
lower-triangular form that [`pack_lambda`](@ref) stores, leaving `ΛΛ'`
unchanged. `pack_lambda` drops the strict upper triangle, so packing raw SVD
loadings for K ≥ 2 silently changes `ΛΛ'`. Uses `Λ' = QR`, so `Λ Λ' = R'R`
and `R'` is lower-trapezoidal; columns are sign-flipped so the diagonal is
non-negative (#865). K < 2 is returned unchanged.
"""
function _lower_triangular_loadings(Λ::AbstractMatrix)
    p, K = size(Λ)
    K < 2 && return Matrix(Λ)               # K = 0, 1: already lower-triangular
    L = Matrix(qr(Matrix(Λ')).R')          # p × K, strict upper = 0
    @inbounds for k in 1:min(p, K)
        L[k, k] < 0 && (L[:, k] .*= -1)
    end
    return L
end

"""
    init_theta_rr(p::Integer, K::Integer) -> Vector{Float64}

Default initial values matching gllvmTMB::init_rr_theta (R/fit-multi.R:1291-1295):
  - diagonal entries initialized to 0.5
  - strict-lower entries initialized to 0
"""
function init_theta_rr(p::Integer, K::Integer)
    n = rr_theta_len(p, K)
    θ = zeros(Float64, n)
    @inbounds for k in 1:K
        θ[k] = 0.5
    end
    return θ
end
