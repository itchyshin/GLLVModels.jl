# Gaussian marginal log-likelihood with augmented-state sparse phylogenetic
# precision. This is a parallel path to `gaussian_marginal_loglik` in
# `likelihood.jl` — same closed-form Gaussian model, but Σ_phy is never
# materialised as a dense (p × p) matrix. Instead the (2p − 1)-vector of
# augmented node values plays the role of Σ_phy, with internal nodes
# marginalised inside the sparse linear solve.
#
# Setup matches the J3 dense path:
#     y[t, s] = (Λ_B η_s)[t] + sum_k Λ_W[t,k] η_W[k, s]
#             + s_B[t, s] + s_W[t, s] + z_phy[t] + X[t,s,:]' β + ε[t,s]
#     A = Λ_B Λ_B' + Λ_W Λ_W' + diag(d_total)
# (the W-tier scores are shared by all traits of a unit, so Λ_W enters with
# its full cross-trait block, as in the C++ twin: issue #135. Below, Λ_W is
# stacked next to Λ_B and every "Λ_B" in the Woodbury algebra is that
# stacked p × (K_B + K_W) matrix.)
#     B = (Λ_aug Λ_aug') ∘ Σ_phy
#     Σ_y_full = I_n ⊗ A + J_n ⊗ B          (column-major vec)
# After the rotation trick used in the dense path, the marginal log-lik
# only needs A (which is rank-K_B Woodbury) and (A + n B) (which we
# tackle below). Λ_aug = hcat(Λ_phy, σ_phy) is the (p × K_aug) augmented
# loadings matrix; K_aug = K_phy + 1 if both are present.
#
# === Sparse representation of Σ_phy ===
#     Σ_phy = σ²_phy · S · Q_cond⁻¹ · S'
# where S extracts the p leaf rows from the (2p − 2)-vector of non-root
# augmented nodes and Q_cond is `phy.Q_topology` with the root row/col
# deleted (the constant-shift null vector pinned). The full topology Q is
# rank-deficient by one; dropping the root makes Q_cond positive definite.
#
# === Block-augmented saddle-point system ===
# Stack K_aug independent copies of the latent: z_full ∈ R^{K_aug(2p−2)},
# prior precision blockdiag(Q_cond/σ²_phy, …). The phy contribution to y
# is
#     y_phy[t,s] = sum_k λ_aug[t,k] · z^{(k)}[leaf(t)] = (D_K z_full)[t]
# where D_K is the (p × K_aug(2p−2)) selector-and-scale matrix that picks
# leaf positions per axis. Setting α = n σ²_phy, the marginal contribution
# to cov(y_full) is J_n ⊗ B with B = (Λ_aug Λ_aug') ∘ Σ_phy as expected.
#
# Solving (A + nB) v = m via the saddle-point auxiliary η = Q_full_cond⁻¹
# · D_K' · v:
#     A v + α · D_K η = m
#     Q_full_cond η = D_K' v
# Schur-complement v:
#     M_sad η = D_K' A⁻¹ m =: b
#     M_sad = Q_full_cond + α · D_K' A⁻¹ D_K
# Using Woodbury for A = D_total + Λ_B Λ_B':
#     D_K' A⁻¹ D_K = D_K' diag(d_total⁻¹) D_K
#                    − D_K' DinvΛ_B · cap⁻¹ · DinvΛ_B' D_K
# The diagonal part lives at the leaf positions per axis (and couples
# axis k to axis l at the SAME leaf via λ_k[t]·λ_l[t]/d_total[t]). The
# rank-K_B Woodbury correction is handled by an inner K_B × K_B dense
# solve. Define:
#     Q_eff = blockdiag(Q_cond) + α · (S_full)' diag(per-leaf coupling) S_full
#                                                                (SPARSE)
#     G     = D_K' · (D_total⁻¹ Λ_B)                          ((K_aug(2p−2)) × K_B)
# Then M_sad = Q_eff − α · G cap⁻¹ G'. CHOLMOD factorises Q_eff in O(p)
# (binary-tree sparsity → linear elimination tree). One K_B × K_B
# capacitance Cholesky completes the Woodbury.
#
# === Determinant identity (used to reach this layout) ===
# det(A + n B) = det(A) · det(I + n A⁻¹ B)
#              = det(A) · det(I + α A⁻¹ D_K Q_full_cond⁻¹ D_K')
#              = det(A) · det(Q_full_cond)⁻¹ · det(Q_full_cond + α D_K' A⁻¹ D_K)
#              = det(A) · det(Q_full_cond)⁻¹ · det(M_sad)
#              = det(A) · det(Q_full_cond)⁻¹ · det(Q_eff) ·
#                det(I_{K_B} − α · G' Q_eff⁻¹ G · cap⁻¹)
# Hence the closed-form
#     logdet(A + nB) = logdet(A) + logdet(Q_eff) − logdet(Q_full_cond)
#                      + logdet(cap − α · G' Q_eff⁻¹ G) − logdet(cap).
# Q_full_cond = blockdiag(Q_cond)^{K_aug}, so logdet(Q_full_cond) =
# K_aug · logdet(Q_cond). The σ²_phy factor enters only through α.
#
# === Note on AD ===
# CHOLMOD operates on Float64 / Float32 only — ForwardDiff.Dual element
# types are not supported. The sparse path therefore CASTS its inputs to
# Float64 for the sparse solve; users who need AD through Σ_phy
# parameters should fall back to the dense `gaussian_marginal_loglik`.
# The result is still returned in the input promoted eltype so that
# downstream code stays generic.

using SparseArrays
using LinearAlgebra

#     _phy_cond_and_leafpos(phy) :: (Q_cond, leaf_pos)
#
# Root-dropped topology precision plus each leaf's row/col position in it.
# Dispatches on the phylo representation: `AugmentedPhy` (root INCLUDED in
# `Q_topology`; dropped here) vs. `PrecisionPhy` (root already dropped at
# construction, `src/phylo_precision.jl`). Both feed the same downstream
# sparse-phylo likelihood kernel below.
function _phy_cond_and_leafpos(phy::AugmentedPhy)
    keep = filter(i -> i != phy.root_index, 1:phy.n_total)
    Q_cond = phy.Q_topology[keep, keep]            # SparseMatrixCSC{Float64,Int}
    # Position of each leaf in the (2p − 2)-vector of non-root nodes.
    leaf_pos = Vector{Int}(undef, phy.n_leaves)
    @inbounds for t in 1:phy.n_leaves
        lp = phy.leaf_indices[t]
        if phy.root_index < lp
            lp -= 1
        end
        leaf_pos[t] = lp
    end
    return Q_cond, leaf_pos
end

"""
    gaussian_marginal_loglik_sparse_phy(y, Λ_B, σ_eps;
        X=nothing, β=nothing,
        Λ_W=nothing, σ²_B=nothing, σ²_W=nothing,
        Λ_phy=nothing, σ_phy=nothing,
        phy::AugmentedPhy, σ²_phy::Real = 1.0)

Closed-form Gaussian marginal log-likelihood with the phylogenetic
covariance represented in **augmented-state sparse precision** form
instead of a dense `Σ_phy`. Numerically equivalent to
`gaussian_marginal_loglik(...; Σ_phy = dense)` where `dense` is the
explicit `σ²_phy · (S Q_cond⁻¹ S')` corresponding to `phy`, but scales
as O(p) thanks to the sparse Cholesky on the augmented precision.

Use this on phylogenies with hundreds to tens of thousands of species,
where forming or factorising the dense p × p Σ_phy is prohibitive.

# Evaluation-only (AD limitation)

This path is **evaluation-only**: CHOLMOD (Julia's sparse Cholesky) does
not support `ForwardDiff.Dual` element types, so inputs are cast to
`Float64` for the sparse solve. AD-based fitting (`fit_gaussian_gllvm`)
must therefore use the dense `gaussian_marginal_loglik` path. The sparse
path is intended for likelihood evaluation, simulation, and verification
on large trees — not for the optimiser's inner loop.
"""
function gaussian_marginal_loglik_sparse_phy(y::AbstractMatrix,
                                             Λ_B::AbstractMatrix,
                                             σ_eps::Real;
                                             X::Union{Nothing,AbstractArray{<:Real,3}} = nothing,
                                             β::Union{Nothing,AbstractVector} = nothing,
                                             Λ_W::Union{Nothing,AbstractMatrix} = nothing,
                                             σ²_B::Union{Nothing,AbstractVector} = nothing,
                                             σ²_W::Union{Nothing,AbstractVector} = nothing,
                                             Λ_phy::Union{Nothing,AbstractMatrix} = nothing,
                                             σ_phy::Union{Nothing,AbstractVector} = nothing,
                                             phy::Union{AugmentedPhy,PrecisionPhy},
                                             σ²_phy::Real = 1.0)
    p, n = size(y)
    σ²   = σ_eps^2

    p == phy.n_leaves ||
        throw(ArgumentError("y first dim ($p) must equal phy.n_leaves " *
                            "($(phy.n_leaves))"))

    # ----- 1. Residual after fixed effects ---------------------------------
    if X === nothing && β === nothing
        resid = y
    else
        (X === nothing || β === nothing) &&
            throw(ArgumentError("Provide both X and β or neither"))
        q = size(X, 3)
        size(X, 1) == p ||
            throw(ArgumentError("X first dim must equal p"))
        size(X, 2) == n ||
            throw(ArgumentError("X second dim must equal n_sites"))
        length(β) == q ||
            throw(ArgumentError("β length must equal size(X, 3)"))
        resid = Matrix{Float64}(undef, p, n)
        @inbounds for s in 1:n, t in 1:p
            μ_ts = 0.0
            for k in 1:q
                μ_ts += X[t, s, k] * β[k]
            end
            resid[t, s] = y[t, s] - μ_ts
        end
    end
    # CHOLMOD requires Float64; we cast eagerly for the sparse-solve path.
    # For AD callers, the dense `gaussian_marginal_loglik` is the
    # appropriate path.
    resid64 = Matrix{Float64}(resid)
    # W tier stacked next to Λ_B (full Λ_W Λ_W' block, issue #135); with no
    # W tier this is Λ_B alone, as before.
    Λ_B64   = Λ_W === nothing ? Matrix{Float64}(Λ_B) :
              hcat(Matrix{Float64}(Λ_B), Matrix{Float64}(Λ_W))
    K_B     = size(Λ_B64, 2)

    # ----- 2. Build d_total[t] = σ²_eps + σ²_B[t] + σ²_W[t]
    d_total = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        v = float(σ²)
        σ²_B !== nothing && (v += σ²_B[t])
        σ²_W !== nothing && (v += σ²_W[t])
        d_total[t] = v
    end

    # ----- 3. Build Λ_aug (p × K_aug) -------------------------------------
    if Λ_phy === nothing && σ_phy === nothing
        throw(ArgumentError("phy specified but no Λ_phy or σ_phy supplied"))
    end
    Λ_aug = if Λ_phy !== nothing && σ_phy !== nothing
        hcat(Matrix{Float64}(Λ_phy), Vector{Float64}(σ_phy))
    elseif Λ_phy !== nothing
        Matrix{Float64}(Λ_phy)
    else
        reshape(Vector{Float64}(σ_phy), p, 1)
    end
    K_aug = size(Λ_aug, 2)

    # ----- 4. m and Y_c (centred residual) --------------------------------
    m   = vec(sum(resid64, dims = 2)) ./ n         # length p
    Y_c = resid64 .- reshape(m, p, 1)              # p × n

    # ----- 5. Woodbury factorisation of A = D + Λ_B Λ_B' ------------------
    # cap = I_K + Λ_B' D⁻¹ Λ_B  (K_B × K_B)
    # A⁻¹ b = D⁻¹ b - D⁻¹ Λ_B cap⁻¹ Λ_B' D⁻¹ b
    d_inv  = 1.0 ./ d_total
    DinvΛB = d_inv .* Λ_B64                        # p × K_B
    cap    = Matrix(I + Λ_B64' * DinvΛB)
    chol_cap = cholesky(Symmetric((cap + cap') ./ 2))
    logdet_A = sum(log, d_total) + logdet(chol_cap)
    Ainv_m   = _woodbury_apply(d_inv, Λ_B64, DinvΛB, chol_cap, m)
    Ainv_Yc  = _woodbury_apply_matrix(d_inv, Λ_B64, DinvΛB, chol_cap, Y_c)

    # ----- 6. Build Q_cond (root row/col already removed) + leaf_pos ------
    # `phy` is either an AugmentedPhy (root INCLUDED; dropped here) or a
    # PrecisionPhy (root already dropped at construction — see
    # src/phylo_precision.jl). `_phy_cond_and_leafpos` dispatches on which.
    Q_cond, leaf_pos = _phy_cond_and_leafpos(phy)
    n_block = size(Q_cond, 1)                      # = 2p − 2

    chol_Qcond   = cholesky(Symmetric(Q_cond))
    logdet_Qcond = logdet(chol_Qcond)

    # ----- 7. Build Q_eff (block-augmented K_aug copies of Q_cond plus the
    # diagonal-of-A⁻¹ leaf coupling across axes) ---------------------------
    α = n * float(σ²_phy)
    total_size = K_aug * n_block

    I_q = Int[]
    J_q = Int[]
    V_q = Float64[]
    # K_aug × O(p) nnz from Q_cond copies
    sizehint!(I_q, K_aug * nnz(Q_cond) + K_aug * K_aug * p)
    sizehint!(J_q, K_aug * nnz(Q_cond) + K_aug * K_aug * p)
    sizehint!(V_q, K_aug * nnz(Q_cond) + K_aug * K_aug * p)

    # (a) K_aug diagonal copies of Q_cond
    rows = rowvals(Q_cond)
    vals = nonzeros(Q_cond)
    for k_blk in 1:K_aug
        offset = (k_blk - 1) * n_block
        for j in 1:n_block
            for idx in nzrange(Q_cond, j)
                i = rows[idx]
                push!(I_q, i + offset)
                push!(J_q, j + offset)
                push!(V_q, vals[idx])
            end
        end
    end
    # (b) Cross-block diagonal coupling at leaf positions: for each pair
    # (k, l), add α · (λ_k[t] λ_l[t] / d_total[t]) at row (leaf_pos[t] +
    # off_k), col (leaf_pos[t] + off_l). When k == l this also augments
    # the leaf diagonal of the k-th Q_cond copy, which is exactly the
    # δ_k = α · λ_k²/d_total contribution that restores PD-ness.
    @inbounds for k_blk in 1:K_aug, l_blk in 1:K_aug
        off_k = (k_blk - 1) * n_block
        off_l = (l_blk - 1) * n_block
        for t in 1:p
            v_couple = α * Λ_aug[t, k_blk] * Λ_aug[t, l_blk] / d_total[t]
            push!(I_q, leaf_pos[t] + off_k)
            push!(J_q, leaf_pos[t] + off_l)
            push!(V_q, v_couple)
        end
    end
    Q_eff = sparse(I_q, J_q, V_q, total_size, total_size)
    # SparseArrays.sparse sums duplicates by default, so leaf-diag entries
    # from Q_cond and from the (k==l) coupling combine correctly. The
    # matrix is symmetric by construction; ensure CHOLMOD takes the
    # symmetric upper triangle.
    chol_Q_eff   = cholesky(Symmetric(Q_eff))
    logdet_Q_eff = logdet(chol_Q_eff)

    # ----- 8. Build G ((K_aug · n_block) × K_B) --------------------------
    # G[(k-1)·n_block + leaf_pos[t], j] = λ_k[t] · d_inv[t] · Λ_B[t, j]
    G = zeros(Float64, total_size, K_B)
    @inbounds for k_blk in 1:K_aug
        offset = (k_blk - 1) * n_block
        for t in 1:p
            base = offset + leaf_pos[t]
            factor = Λ_aug[t, k_blk] * d_inv[t]
            for j in 1:K_B
                G[base, j] = factor * Λ_B64[t, j]
            end
        end
    end

    # ----- 9. Woodbury K_B × K_B correction -------------------------------
    # M_sad = Q_eff − α G cap⁻¹ G'
    # det(M_sad) = det(Q_eff) · det(I_K − α G' Q_eff⁻¹ G cap⁻¹)
    #            = det(Q_eff) · det(cap − α G' Q_eff⁻¹ G) / det(cap)
    # M_sad⁻¹ b = Q_eff⁻¹ b + Q_eff⁻¹ G (cap/α − G' Q_eff⁻¹ G)⁻¹ G' Q_eff⁻¹ b
    #          = Q_eff⁻¹ b + α · Q_eff⁻¹ G (cap − α G' Q_eff⁻¹ G)⁻¹ G' Q_eff⁻¹ b
    X_G  = chol_Q_eff \ G                          # (total_size) × K_B
    M_K  = G' * X_G                                # K_B × K_B
    S_K  = cap .- α .* M_K                         # K_B × K_B
    chol_S_K = cholesky(Symmetric((S_K + S_K') ./ 2))
    logdet_M_sad = logdet_Q_eff + logdet(chol_S_K) - logdet(chol_cap)

    logdet_Qfull_cond = K_aug * logdet_Qcond
    logdet_AnB = logdet_A + logdet_M_sad - logdet_Qfull_cond

    # ----- 10. Quadratic form for the mean component ---------------------
    # Solve M_sad η = b where b = D_K' A⁻¹ m  (a total_size-vector
    # concentrated at leaf positions of each axis).
    # b[(k-1)·n_block + leaf_pos[t]] = λ_k[t] · (A⁻¹ m)[t]
    b = zeros(Float64, total_size)
    @inbounds for k_blk in 1:K_aug
        offset = (k_blk - 1) * n_block
        for t in 1:p
            b[offset + leaf_pos[t]] = Λ_aug[t, k_blk] * Ainv_m[t]
        end
    end
    ξ0  = chol_Q_eff \ b
    yK  = chol_S_K \ (G' * ξ0)                     # (cap − α G' Q_eff⁻¹ G)⁻¹ G' ξ0
    ξ   = ξ0 .+ α .* (chol_Q_eff \ (G * yK))

    # v = A⁻¹ m − α A⁻¹ D_K η
    # m' v = m' A⁻¹ m − α · sum_k sum_t λ_k[t] · (A⁻¹ m)[t] · η[(k-1)n + leaf_pos[t]]
    #      = m' A⁻¹ m − α · b' η                (b matches the inner sum)
    quad_mean_inner = dot(m, Ainv_m) - α * dot(b, ξ)
    quad_mean       = n * quad_mean_inner

    # tr(Y_c' A⁻¹ Y_c)
    quad_centered = sum(Y_c .* Ainv_Yc)

    # ----- 11. Assemble log-likelihood -----------------------------------
    logdet_Σ_full = logdet_AnB + (n - 1) * logdet_A
    quad = quad_mean + quad_centered

    return -0.5 * (n * p * log(2π) + logdet_Σ_full + quad)
end

# ---------------------------------------------------------------------------
# Internal: Woodbury apply for A = D + Λ Λ' to a single vector / matrix.
# A⁻¹ b = D⁻¹ (b − Λ (cap \ (Λ' D⁻¹ b)))
# ---------------------------------------------------------------------------
@inline function _woodbury_apply(d_inv::AbstractVector,
                                  Λ::AbstractMatrix, DinvΛ::AbstractMatrix,
                                  chol_cap, b::AbstractVector)
    Dinv_b = d_inv .* b
    yK = Λ' * Dinv_b
    zK = chol_cap \ yK
    return Dinv_b .- (DinvΛ * zK)
end

@inline function _woodbury_apply_matrix(d_inv::AbstractVector,
                                         Λ::AbstractMatrix, DinvΛ::AbstractMatrix,
                                         chol_cap, B::AbstractMatrix)
    Dinv_B = d_inv .* B
    YK = Λ' * Dinv_B
    ZK = chol_cap \ YK
    return Dinv_B .- (DinvΛ * ZK)
end
