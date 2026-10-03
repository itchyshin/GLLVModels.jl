# Gaussian marginal log-likelihood with the phylogenetic covariance
# represented in the edge-node incidence form (`EdgePhy`).
#
# The model is identical to the J3 path in `likelihood.jl` and to the
# augmented-Q path in `likelihood_sparse_phy.jl`:
#
#     y[t, s] = (Λ_B η_s)[t] + sum_k Λ_W[t,k] η_W[k,s]
#             + s_B[t,s] + s_W[t,s] + z_phy[t]
#             + X[t,s,:]' β + ε[t,s]
#
# with marginal covariance over vec(y) (column-major)
#
#     Σ_y_full = I_n ⊗ A + J_n ⊗ B
#
# where A = Λ_B Λ_B' + Λ_W Λ_W' + diag(d_total) (W tier shared across traits
# within a unit, issue #135) and B = (Λ_phy_aug Λ_phy_aug') ⊙ Σ_phy,
# Λ_phy_aug = hcat(Λ_phy, σ_phy) (as in the dense path).
#
# What's different here is **how Σ_phy is produced**. The dense path takes
# Σ_phy from the caller as a p × p matrix. The edge-incidence path builds
# it from the tree:
#
#     Σ_phy[i, j] = σ²_phy · depth(MRCA(i, j))
#
# via O(p²) ancestor-chain traversal (see `sigma_phy_dense_edge`). The
# whole construction is differentiable w.r.t. σ²_phy and the branch
# lengths — no sparse Cholesky in the AD path.
#
# === AD properties ===
# Every operation in this function is `ForwardDiff.Dual`-compatible:
#   * the incidence matrix B is structural (Int) and does not carry duals;
#   * `branch_lengths` and `σ²_phy` flow through `sigma_phy_dense_edge`
#     linearly;
#   * the rest is the same dense-Cholesky rotation trick as the existing
#     J3 path in `likelihood.jl`, which already runs under Dual.
#
# This is the KEY advantage of the edge-incidence representation over the
# augmented sparse path: `gaussian_marginal_loglik_sparse_phy` casts to
# Float64 because CHOLMOD does not accept Dual, so AD is disabled.
# `gaussian_marginal_loglik_edge_phy` keeps the eltype generic throughout.
#
# === Scaling note ===
# Forming Σ_phy as a dense p × p matrix is O(p²) in time and storage. At
# p = 10_000 that is 800 MB of doubles — feasible for evaluation but not
# the linear-in-p ideal. A true O(p) AD path needs the matrix-free
# Σ_phy · v (via `Q_times_x` / `solve_Q`) plumbed into a Schur-complement
# saddle-point solve; that is left as a follow-up. The current path is
# correct and AD-friendly at every scale that fits in memory; the
# benchmark file reports honest timing numbers.

using LinearAlgebra

"""
    gaussian_marginal_loglik_edge_phy(y, Λ_B, σ_eps;
        X=nothing, β=nothing,
        Λ_W=nothing, σ²_B=nothing, σ²_W=nothing,
        Λ_phy=nothing, σ_phy=nothing,
        phy::EdgePhy, σ²_phy = 1.0)

Closed-form Gaussian marginal log-likelihood using the edge-node incidence
incidence representation of the phylogenetic precision (`phy::EdgePhy`).
The phylogenetic covariance Σ_phy is built from the tree topology in B
and the branch lengths, then plugged into the same rotation-trick path
the dense `gaussian_marginal_loglik` uses.

# Why this path?

This path is **AD-friendly**: B is structural (Int), branch lengths and
`σ²_phy` flow through ordinary element-wise ops, so `ForwardDiff.Dual`
element types pass through cleanly. The augmented-Q sparse path
(`gaussian_marginal_loglik_sparse_phy`) cannot do this — CHOLMOD rejects
Dual element types — and is therefore evaluation-only. This path enables
gradient-based fitting at moderate-to-large p.

# Numerical equivalence

For any tree, `Σ_phy` built here matches `sigma_phy_dense(phy_augmented)`
to floating-point precision, so this log-likelihood agrees with the
dense `gaussian_marginal_loglik(...; Σ_phy = sigma_phy_dense_edge(phy,
σ²_phy))` and with `gaussian_marginal_loglik_sparse_phy(...; phy =
augmented, σ²_phy = σ²_phy)` to ~1e-10 rtol.

# Cost

O(p²) — the dense rotation-trick step has not yet been replaced with the
matrix-free `Q_times_x` path. The headline AD advantage holds at any p
the dense path can handle.
"""
function gaussian_marginal_loglik_edge_phy(y::AbstractMatrix,
                                            Λ_B::AbstractMatrix,
                                            σ_eps::Real;
                                            X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                                            β::Union{Nothing, AbstractVector} = nothing,
                                            Λ_W::Union{Nothing, AbstractMatrix} = nothing,
                                            σ²_B::Union{Nothing, AbstractVector} = nothing,
                                            σ²_W::Union{Nothing, AbstractVector} = nothing,
                                            Λ_phy::Union{Nothing, AbstractMatrix} = nothing,
                                            σ_phy::Union{Nothing, AbstractVector} = nothing,
                                            phy::EdgePhy,
                                            σ²_phy::Real = 1.0)
    p, _ = size(y)
    p == phy.n_leaves ||
        throw(ArgumentError("y first dim ($p) must equal phy.n_leaves " *
                            "($(phy.n_leaves))"))

    # At least one phylogenetic-block component must be supplied — otherwise
    # the J3 path collapses and the dense routine would skip the phy block
    # entirely, which is not what the user asked for. Checked before the
    # O(p²) Σ_phy build below so the error path allocates nothing large.
    if Λ_phy === nothing && σ_phy === nothing
        throw(ArgumentError("phy specified but no Λ_phy or σ_phy supplied"))
    end

    # Σ_phy is built from the tree topology + σ²_phy + branch lengths.
    # AD-friendly: closed form, no sparse Cholesky on Duals.
    Σ_phy = sigma_phy_dense_edge(phy, σ²_phy)

    # Delegate to the dense J3 implementation — it is already
    # Dual-compatible and tested. Reach into the GLLVModels module since this
    # file is included outside the module on the PERF++++ branch (the
    # hard constraint forbids editing src/GLLVModels.jl).
    return GLLVModels.gaussian_marginal_loglik(y, Λ_B, σ_eps;
                                          X = X, β = β,
                                          Λ_W = Λ_W, σ²_B = σ²_B, σ²_W = σ²_W,
                                          Λ_phy = Λ_phy, σ_phy = σ_phy,
                                          Σ_phy = Σ_phy)
end
