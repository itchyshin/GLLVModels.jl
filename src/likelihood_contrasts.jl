# Closed-form Gaussian marginal log-likelihood on the Felsenstein
# contrast scale. The contrast matrix U (built in phylo_contrasts.jl) is
# the unique linear map that diagonalises the Brownian-motion phyloge-
# netic covariance: U Σ_phy U' = σ²_phy · diag(weights) for any tree.
#
# How exact equality with the dense path is achieved
# --------------------------------------------------
# Let T = [c_root'; U] be the full (p × p) basis change combining the
# root centroid c_root' (1 × p) with the (p − 1) × p contrast block U.
# As shown in phylo_contrasts.jl, T is bijective and volume-preserving:
# |det T| = 1, so log p(y) = log p(T y). Working in the transformed
# coordinates makes the Brownian-motion block of the covariance block-
# diagonal — under BM with diffusion variance σ²_phy,
#     T Σ_phy T' = σ²_phy · diag(t_root, weights)
# i.e. literally diagonal. Other contributions (Λ_B Λ_B', the σ²_eps
# diagonal, and Λ_phy / σ_phy multiplied into Σ_phy) become dense in the
# transformed basis but the rotation trick still applies — we compute
# the log-likelihood using two p × p Cholesky factorisations (of T A T'
# and T (A + n B) T') and the standard mean / centred decomposition
# (see dense path in likelihood.jl).
#
# What we get out of the contrast representation
# ----------------------------------------------
# 1. AD-friendly. Everything is dense linear algebra over the AD-tracked
#    parameters (σ_eps, σ²_phy, Λ_B, Λ_phy, σ_phy, β) with T precomputed
#    as Float64. ForwardDiff.Dual passes through without issue.
# 2. Numerically exact w.r.t. the dense `gaussian_marginal_loglik`
#    formula. Verified to ≲ 1e-10 in the accompanying tests.
# 3. The diagonalisation of Σ_phy is harvested when other contributions
#    are themselves trait-homogeneous (e.g. σ_eps I): the dense T A T'
#    matrix still needs forming, but the BM block enters as a clean
#    diagonal addition T (n σ²_phy V) T' = n σ²_phy diag(t_root, weights)
#    rather than as a dense V.
#
# Scope. Brownian motion only. For non-BM models (multi-rate, OU, EB),
# the contrast diagonalisation is partial (still sparse but not maximally
# so). The interface still accepts Λ_phy and σ_phy as trait-specific
# loadings on the BM axis(es) — those are arbitrary per-trait scalings,
# not new rate models.

using LinearAlgebra
using SparseArrays

"""
    gaussian_marginal_loglik_contrasts(y, Λ_B, σ_eps;
        X=nothing, β=nothing,
        Λ_phy=nothing, σ_phy=nothing,
        tree::AugmentedPhy, σ²_phy::Real=1.0) -> Real

Marginal log-likelihood of `y` (size p × n_sites) for the Gaussian
GLLVM with a Brownian-motion phylogenetic block, computed on the
Felsenstein-contrast scale.

The model is the same as the J3 dense path in `likelihood.jl`:
    y[t, s] = (Λ_B η_s)[t] + Σ_k Λ_phy[t, k] η_phy[k, t] + σ_phy[t] φ[t]
            + X[t,s,:]' β + ε[t, s]
with η_s ~ N(0, I_K_B), ε ~ N(0, σ²_eps), and the species-level phy
vectors η_phy[k, :] and φ ~ MVN(0, σ²_phy · V_tree) (independent across
the K_phy axes and across φ).

Σ_phy is NOT supplied as an argument — only the tree. The phylogenetic
covariance is taken to be `σ²_phy · V_tree` where `V_tree` is the BM
covariance implied by the tree (V_tree[i,j] = path length from root to
MRCA(i, j)). This is the standard BM assumption.

For BM, U Σ_phy U' = σ²_phy diag(weights). For non-BM rate models the
contrast diagonalisation no longer holds — use `Σ_phy = ...` with the
dense `gaussian_marginal_loglik` in that case.

Returns the FULL marginal log-likelihood of y (not the contrasts'
log-likelihood alone) — numerically identical to the dense path to
the limits of floating-point precision, for matched model specifications.

Matching caveat: when neither `Λ_phy` nor `σ_phy` is supplied, this function
adds a trait-homogeneous BM block (implicit loading `ones(p)`, covariance
`σ²_phy · V_tree`). That corresponds to the dense call
`gaussian_marginal_loglik(...; σ_phy = ones(p), Σ_phy = σ²_phy · V_tree)`,
NOT to the dense no-loadings call `gaussian_marginal_loglik(...; Σ_phy = ...)`,
which has no phylogenetic block and therefore returns a different value.
"""
function gaussian_marginal_loglik_contrasts(y::AbstractMatrix,
                                            Λ_B::AbstractMatrix,
                                            σ_eps::Real;
                                            X::Union{Nothing,AbstractArray{<:Real,3}} = nothing,
                                            β::Union{Nothing,AbstractVector} = nothing,
                                            Λ_phy::Union{Nothing,AbstractMatrix} = nothing,
                                            σ_phy::Union{Nothing,AbstractVector} = nothing,
                                            tree,
                                            σ²_phy::Real = 1.0)
    p, n = size(y)
    K    = size(Λ_B, 2)
    σ²   = σ_eps^2

    p == tree.n_leaves ||
        throw(ArgumentError("y first dim ($p) must equal tree.n_leaves " *
                            "($(tree.n_leaves))"))

    # Promote AD-aware element type.
    T_el = promote_type(eltype(y), eltype(Λ_B), typeof(σ²), typeof(float(σ²_phy)))
    if Λ_phy !== nothing
        size(Λ_phy, 1) == p ||
            throw(ArgumentError("Λ_phy first dim ($(size(Λ_phy, 1))) must equal p ($p)"))
        T_el = promote_type(T_el, eltype(Λ_phy))
    end
    if σ_phy !== nothing
        length(σ_phy) == p ||
            throw(ArgumentError("σ_phy length ($(length(σ_phy))) must equal p ($p)"))
        T_el = promote_type(T_el, eltype(σ_phy))
    end

    # Residual after fixed effects.
    if X === nothing && β === nothing
        resid = y
    else
        (X === nothing || β === nothing) &&
            throw(ArgumentError("Provide both X and β or neither"))
        q = size(X, 3)
        size(X, 1) == p || throw(ArgumentError("X first dim must equal p"))
        size(X, 2) == n || throw(ArgumentError("X second dim must equal n_sites"))
        length(β) == q  || throw(ArgumentError("β length must equal size(X, 3)"))
        Tres = promote_type(T_el, eltype(X), eltype(β))
        resid = Matrix{Tres}(undef, p, n)
        @inbounds for s in 1:n, t in 1:p
            μ_ts = zero(Tres)
            for k in 1:q
                μ_ts += X[t, s, k] * β[k]
            end
            resid[t, s] = y[t, s] - μ_ts
        end
        T_el = Tres
    end

    # ----- Precompute T = [c_root'; U] and the BM diagonal ----------------
    fc = felsenstein_contrasts(tree)
    U      = fc.U          # (p-1) × p, Float64 sparse
    w_vec  = fc.weights    # length p-1, Float64
    c_root = fc.c_root     # length p, Float64
    t_root = fc.t_root     # Float64

    # Apply T to the residual: z = T * resid (size p × n).
    # First row = c_root' * resid (per-site root centroid).
    # Rows 2:p = U * resid (per-site contrasts).
    z = Matrix{T_el}(undef, p, n)
    # Row 1: c_root' resid.
    @inbounds for s in 1:n
        acc = zero(T_el)
        for t in 1:p
            acc += c_root[t] * resid[t, s]
        end
        z[1, s] = acc
    end
    # Rows 2:p: U * resid. SparseMatrixCSC * Matrix promotes element
    # types correctly under ForwardDiff.
    Uz = U * resid                                 # (p-1) × n, T_el-typed
    @inbounds for s in 1:n, k in 1:(p - 1)
        z[k + 1, s] = Uz[k, s]
    end

    # Per-site cov in transformed basis Σ_z = T (Λ_B Λ_B' + σ²_eps I) T'.
    #   T Λ_B   = stacked transformed loadings (p × K)
    TLB = Matrix{T_el}(undef, p, K)
    @inbounds for k in 1:K
        # Row 1: c_root' * Λ_B[:, k]
        acc = zero(T_el)
        for t in 1:p
            acc += c_root[t] * Λ_B[t, k]
        end
        TLB[1, k] = acc
    end
    if K > 0
        ULB = U * Λ_B                              # (p-1) × K
        @inbounds for k in 1:K, j in 1:(p - 1)
            TLB[j + 1, k] = ULB[j, k]
        end
    end

    # T T' — needed for σ²_eps I in transformed basis.
    # T T' = [c_root' c_root, c_root' U'; U c_root, U U'].
    TTt = _build_TTt(T_el, U, c_root)

    # T (σ²_eps I) T' = σ²_eps * T T'.
    # T (Λ_B Λ_B') T' = TLB * TLB'.
    # Combined per-site: A_z = TLB TLB' + σ²_eps T T'.
    A_z = TLB * TLB'
    @inbounds for j in 1:p, i in 1:p
        A_z[i, j] += σ² * TTt[i, j]
    end

    # ----- Phylogenetic contribution ---------------------------------------
    # B (original basis) = (Λ_aug Λ_aug') ∘ Σ_phy where Σ_phy = σ²_phy · V.
    # In the transformed basis we need T B T'.
    # When K_aug == 0 (no Λ_phy and no σ_phy supplied), B = 0 and the
    # model has no phylogenetic block at all. Otherwise we materialise
    # B in the original basis (a p × p Hadamard product) and rotate to
    # T B T'. For p ≲ 10⁴ this is feasible; for larger p the dense
    # Σ_phy is impractical and the augmented-sparse path is the right
    # tool — but that path cannot be differentiated through.
    #
    # Special-case: if Λ_phy === nothing AND σ_phy === nothing, we
    # interpret "tree + σ²_phy" as a TRAIT-HOMOGENEOUS BM contribution
    # with implicit Λ_aug = ones(p). Then
    #     B = σ²_phy · V        (homogeneous BM across traits)
    #     T B T' = σ²_phy · diag(t_root, weights)
    # — literally diagonal, no need to materialise V. This is the
    # AD-friendly fast path the contrast representation was built for.
    has_phy_trait_specific = (Λ_phy !== nothing) || (σ_phy !== nothing)

    if !has_phy_trait_specific
        # Trait-homogeneous BM: T B T' is diagonal with entries
        # [σ²_phy · t_root, σ²_phy · weights[1], …].
        # No dense V needed.
        AnB_z = copy(A_z)
        AnB_z[1, 1] += n * float(σ²_phy) * t_root
        @inbounds for k in 1:(p - 1)
            AnB_z[k + 1, k + 1] += n * float(σ²_phy) * w_vec[k]
        end

        # m and Y_c on transformed scale.
        m_z = vec(sum(z, dims = 2)) ./ n
        Y_c_z = z .- reshape(m_z, p, 1)

        # logdet(A_z), logdet(A_z + n B_z), quadratic forms.
        cA   = cholesky(Symmetric((A_z + A_z') ./ 2))
        cAnB = cholesky(Symmetric((AnB_z + AnB_z') ./ 2))

        v_mean = cAnB \ m_z
        quad_mean = n * dot(m_z, v_mean)
        V_c = cA \ Y_c_z
        quad_centered = sum(Y_c_z .* V_c)

        logdet_Σ_full = logdet(cAnB) + (n - 1) * logdet(cA)
        quad = quad_mean + quad_centered

        # |det T| = 1 ⇒ no Jacobian correction.
        return -convert(T_el, 0.5) *
               (n * p * log(convert(T_el, 2π)) + logdet_Σ_full + quad)
    else
        # Trait-specific phy contribution: must materialise Σ_phy on the
        # original basis, form B = (Λ_aug Λ_aug') ∘ Σ_phy, then rotate.
        # Σ_phy is a (p × p) matrix; for p large this is the
        # complexity-dominating step.
        Σ_phy = _build_Σ_phy_dense(tree, σ²_phy, T_el)

        Λ_aug = if Λ_phy !== nothing && σ_phy !== nothing
            hcat(Matrix{T_el}(Λ_phy), Vector{T_el}(σ_phy))
        elseif Λ_phy !== nothing
            Matrix{T_el}(Λ_phy)
        else
            reshape(Vector{T_el}(σ_phy), p, 1)
        end
        B = (Λ_aug * Λ_aug') .* Σ_phy             # p × p

        # T B T' (dense, p × p). For now compute as T*B*T'.
        # Construct T as a (p × p) dense matrix once.
        Tmat = _build_T(T_el, U, c_root)
        TBTt = Tmat * B * Tmat'
        AnB_z = A_z .+ n .* TBTt

        m_z = vec(sum(z, dims = 2)) ./ n
        Y_c_z = z .- reshape(m_z, p, 1)

        cA   = cholesky(Symmetric((A_z + A_z') ./ 2))
        cAnB = cholesky(Symmetric((AnB_z + AnB_z') ./ 2))

        v_mean = cAnB \ m_z
        quad_mean = n * dot(m_z, v_mean)
        V_c = cA \ Y_c_z
        quad_centered = sum(Y_c_z .* V_c)

        logdet_Σ_full = logdet(cAnB) + (n - 1) * logdet(cA)
        quad = quad_mean + quad_centered

        return -convert(T_el, 0.5) *
               (n * p * log(convert(T_el, 2π)) + logdet_Σ_full + quad)
    end
end

# ---------------------------------------------------------------------------
# Internals: build T T', T, and Σ_phy without materialising larger objects.
# ---------------------------------------------------------------------------

# Build T T' (p × p) where T = [c_root'; U]. This is fully Float64 — the
# topology Jacobian, AD-tracked params don't enter.
function _build_TTt(::Type{T_el}, U::SparseMatrixCSC{Float64,Int},
                    c_root::Vector{Float64}) where {T_el}
    p = length(c_root)
    out = Matrix{T_el}(undef, p, p)
    # Row 1 / col 1: ⟨c_root, c_root⟩, ⟨c_root, U[k, :]⟩
    @inbounds out[1, 1] = T_el(dot(c_root, c_root))
    Uc = U * c_root                                # length p-1
    @inbounds for k in 1:(p - 1)
        out[1, k + 1] = T_el(Uc[k])
        out[k + 1, 1] = T_el(Uc[k])
    end
    # Block U U' (p-1 × p-1). Compute as sparse-dense product.
    # For sparse U the result U U' may be dense; we materialise it.
    UUt = Matrix(U * U')                           # (p-1) × (p-1) dense
    @inbounds for j in 1:(p - 1), i in 1:(p - 1)
        out[i + 1, j + 1] = T_el(UUt[i, j])
    end
    return out
end

# Build dense T (p × p): row 1 is c_root', rows 2:p are U.
function _build_T(::Type{T_el}, U::SparseMatrixCSC{Float64,Int},
                  c_root::Vector{Float64}) where {T_el}
    p = length(c_root)
    out = zeros(T_el, p, p)
    @inbounds for t in 1:p
        out[1, t] = T_el(c_root[t])
    end
    rows = rowvals(U)
    vals = nonzeros(U)
    @inbounds for j in 1:p
        for idx in nzrange(U, j)
            i = rows[idx]
            out[i + 1, j] = T_el(vals[idx])
        end
    end
    return out
end

# Build the dense Σ_phy from a tree by inverting Q on the augmented
# state (root-pinned), then selecting leaves. O(p³) — used only on
# the trait-specific path where Σ_phy must be materialised.
function _build_Σ_phy_dense(tree, σ²_phy, ::Type{T_el}) where {T_el}
    Σ = sigma_phy_dense(tree; σ²_phy = σ²_phy)
    return Matrix{T_el}(Σ)
end
