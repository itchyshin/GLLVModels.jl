# Structure-aware factorisation for M = Λ Λ' + D with D = Diagonal(d).
#
# The Gaussian GLLVM marginal covariance has the form
#     M = Λ Λ' + diag(d)            (p × p, rank-K dense plus diagonal)
# with K ≪ p. Generic `cholesky(Symmetric(M))` is O(p³). Exploiting the
# rank-K structure via Woodbury (Golub & Van Loan §4.2):
#
#     M⁻¹ = D⁻¹ - D⁻¹ Λ (I_K + Λ' D⁻¹ Λ)⁻¹ Λ' D⁻¹
#     logdet(M) = sum(log d) + logdet(I_K + Λ' D⁻¹ Λ)        (mat. det. lemma)
#
# reduces every solve/logdet to a single K × K Cholesky of size K ≪ p plus
# O(p K) BLAS-2 work. Total work per solve: O(p K + K³).
#
# Storage: d ∈ ℝ^p, Λ ∈ ℝ^{p × K}, plus a K × K Cholesky.
# Compare to the dense Cholesky factor's p(p + 1)/2 entries.
#
# AD-friendliness: every kernel uses eltype-generic broadcasting / BLAS so
# that ForwardDiff.Dual (and similar AD element types) flow through. No
# Float64 hard-coding.

using LinearAlgebra
import LinearAlgebra: logdet, mul!, ldiv!

"""
    LowRankPlusDiagChol{T}

Factorisation of `M = Λ Λ' + Diagonal(d)` that stores three small pieces:

* `d::Vector{T}`              — the positive diagonal of `D`.
* `Λ::Matrix{T}`              — the `p × K` low-rank factor.
* `cholK::Cholesky{T, Matrix{T}}` — Cholesky of the `K × K` capacitance
  matrix `I_K + Λ' D⁻¹ Λ`.

Use [`low_rank_chol`](@ref) to construct, then `\\`, [`ldiv!`](@ref), or
`logdet` for the Woodbury-based operations.
"""
struct LowRankPlusDiagChol{T, MK <: AbstractMatrix{T}}
    d::Vector{T}
    Λ::Matrix{T}
    cholK::Cholesky{T, MK}
end

"""
    low_rank_chol(Λ::AbstractMatrix, d::AbstractVector)

Build the [`LowRankPlusDiagChol`](@ref) factorisation of
`M = Λ Λ' + Diagonal(d)`.

Promotes `eltype(Λ)` and `eltype(d)` so AD element types (e.g.
`ForwardDiff.Dual`) flow through. Constructs the `K × K` capacitance
matrix `I_K + Λ' D⁻¹ Λ` and factorises it once.
"""
function low_rank_chol(Λ::AbstractMatrix, d::AbstractVector)
    p, K = size(Λ)
    length(d) == p ||
        throw(DimensionMismatch("length(d) = $(length(d)) must equal size(Λ, 1) = $p"))
    all(>(0), d) ||
        throw(ArgumentError("low_rank_chol requires strictly positive diagonal d; " *
                            "got min = $(minimum(d))"))
    T = promote_type(eltype(Λ), eltype(d))
    dT = collect(T, d)                      # Vector{T}, owns its storage
    ΛT = Matrix{T}(Λ)                       # p × K, owns its storage
    # Scale rows of Λ by 1/d (broadcasting handles AD element types).
    DinvΛ = ΛT ./ dT
    # K × K capacitance matrix: I_K + Λ' D⁻¹ Λ.
    A_K = ΛT' * DinvΛ
    @inbounds for k in 1:K
        A_K[k, k] += one(T)
    end
    cholK = cholesky(Symmetric((A_K + A_K') ./ 2))
    return LowRankPlusDiagChol{T, typeof(cholK.factors)}(dT, ΛT, cholK)
end

# Trait queries -------------------------------------------------------------

Base.eltype(::Type{<:LowRankPlusDiagChol{T}}) where {T} = T
Base.eltype(F::LowRankPlusDiagChol) = eltype(typeof(F))
Base.size(F::LowRankPlusDiagChol)            = (length(F.d), length(F.d))
Base.size(F::LowRankPlusDiagChol, i::Integer) =
    i < 1 ? throw(BoundsError(size(F), i)) : (i <= 2 ? length(F.d) : 1)

# ---------------------------------------------------------------------------
# Solve M x = b via Woodbury.
#
#   M⁻¹ b = D⁻¹ b - D⁻¹ Λ (I + Λ' D⁻¹ Λ)⁻¹ Λ' D⁻¹ b
#         = D⁻¹ (b - Λ (cholK \ (Λ' (D⁻¹ b))))
#
# Steps:
#   1. Dinv_b = b ./ d                 — O(p)
#   2. y_K    = Λ' * Dinv_b            — O(p K)
#   3. solve  cholK y_K  (in place)    — O(K²)
#   4. out    = (b .- Λ * y_K) ./ d    — O(p K)
# ---------------------------------------------------------------------------

"""
    ldiv!(out, F::LowRankPlusDiagChol, b)
    ldiv!(out, F::LowRankPlusDiagChol, b, buf_K)

In-place Woodbury solve `out = M⁻¹ b` with `M = F.Λ F.Λ' + Diagonal(F.d)`.

If `buf_K` is omitted a fresh `K`-vector is allocated; pass one in to
make this fully allocation-free on the hot path.
"""
function ldiv!(out::AbstractVector, F::LowRankPlusDiagChol, b::AbstractVector)
    K   = size(F.Λ, 2)
    T   = promote_type(eltype(out), eltype(F), eltype(b))
    buf = Vector{T}(undef, K)
    return ldiv!(out, F, b, buf)
end

function ldiv!(out::AbstractVector, F::LowRankPlusDiagChol,
               b::AbstractVector, buf_K::AbstractVector)
    p, K = size(F.Λ)
    length(b)   == p || throw(DimensionMismatch("length(b) must equal p = $p"))
    length(out) == p || throw(DimensionMismatch("length(out) must equal p = $p"))
    length(buf_K) == K || throw(DimensionMismatch("length(buf_K) must equal K = $K"))

    # Step 1: Dinv_b = b ./ d   — write into out as working storage.
    @inbounds for t in 1:p
        out[t] = b[t] / F.d[t]
    end
    # Step 2: buf_K = Λ' * Dinv_b
    mul!(buf_K, F.Λ', out)
    # Step 3: in-place K × K Cholesky solve. After this `buf_K` holds
    # (I + Λ' D⁻¹ Λ)⁻¹ Λ' D⁻¹ b.
    ldiv!(F.cholK, buf_K)
    # Step 4: out = Dinv_b - D⁻¹ Λ buf_K. `out` already holds Dinv_b;
    # subtract D⁻¹ (Λ buf_K) via an explicit inner loop (BLAS gemv
    # would need a temporary p-vector and would not buy anything at
    # AD eltypes).
    @inbounds for t in 1:p
        s = zero(eltype(out))
        for k in 1:K
            s += F.Λ[t, k] * buf_K[k]
        end
        out[t] -= s / F.d[t]
    end
    return out
end

# Matrix RHS convenience: column-by-column.
function ldiv!(out::AbstractMatrix, F::LowRankPlusDiagChol, B::AbstractMatrix)
    size(out) == size(B) ||
        throw(DimensionMismatch("size(out) must equal size(B)"))
    size(B, 1) == length(F.d) ||
        throw(DimensionMismatch("size(B, 1) must equal p = $(length(F.d))"))
    K   = size(F.Λ, 2)
    T   = promote_type(eltype(out), eltype(F), eltype(B))
    buf = Vector{T}(undef, K)
    for j in 1:size(B, 2)
        ldiv!(view(out, :, j), F, view(B, :, j), buf)
    end
    return out
end

Base.:\(F::LowRankPlusDiagChol, b::AbstractVector) =
    ldiv!(similar(b, promote_type(eltype(F), eltype(b))), F, b)

Base.:\(F::LowRankPlusDiagChol, B::AbstractMatrix) =
    ldiv!(similar(B, promote_type(eltype(F), eltype(B))), F, B)

# ---------------------------------------------------------------------------
# logdet via the matrix determinant lemma:
#     logdet(D + Λ Λ') = logdet(D) + logdet(I_K + Λ' D⁻¹ Λ)
# ---------------------------------------------------------------------------

function LinearAlgebra.logdet(F::LowRankPlusDiagChol)
    # sum(log, F.d) + logdet(F.cholK) keeps eltype generic.
    return sum(log, F.d) + logdet(F.cholK)
end

# ---------------------------------------------------------------------------
# Matrix–vector product M * x (and M * X) — used for verification, not the
# hot path. Computed as M x = Λ (Λ' x) + d .* x to avoid materialising M.
# ---------------------------------------------------------------------------

function LinearAlgebra.mul!(out::AbstractVector, F::LowRankPlusDiagChol,
                            x::AbstractVector)
    p, K = size(F.Λ)
    length(x)   == p || throw(DimensionMismatch("length(x) must equal p = $p"))
    length(out) == p || throw(DimensionMismatch("length(out) must equal p = $p"))
    T = promote_type(eltype(out), eltype(F), eltype(x))
    tmp_K = Vector{T}(undef, K)
    mul!(tmp_K, F.Λ', x)                     # K-vector
    mul!(out, F.Λ, tmp_K)                    # p-vector: Λ (Λ' x)
    @inbounds for t in 1:p
        out[t] += F.d[t] * x[t]
    end
    return out
end

function LinearAlgebra.mul!(out::AbstractMatrix, F::LowRankPlusDiagChol,
                            X::AbstractMatrix)
    size(out) == size(X) ||
        throw(DimensionMismatch("size(out) must equal size(X)"))
    p = length(F.d)
    size(X, 1) == p ||
        throw(DimensionMismatch("size(X, 1) must equal p = $p"))
    for j in 1:size(X, 2)
        mul!(view(out, :, j), F, view(X, :, j))
    end
    return out
end

Base.:*(F::LowRankPlusDiagChol, x::AbstractVector) =
    mul!(similar(x, promote_type(eltype(F), eltype(x))), F, x)
Base.:*(F::LowRankPlusDiagChol, X::AbstractMatrix) =
    mul!(similar(X, promote_type(eltype(F), eltype(X))), F, X)
