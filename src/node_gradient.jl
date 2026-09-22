# Exact per-species phylogenetic gradient in the latent AUGMENTED-NODE frame.
#
# WHAT THIS FILE PROVIDES
# -----------------------
# The SAME phylo_unique Gaussian marginal likelihood the engine evaluates
# (`gaussian_marginal_loglik_sparse_phy` / `sparse_phy_value`), differentiated
# in the latent augmented-NODE frame:
#     y[:,s] = Λ_B η_s + diag(σ_phy) φ + ε_s,
#     φ ~ N(0, σ²_phy Σ_phy),  Σ_phy = S Q_cond⁻¹ S'  (tree, BM),
#     ε_s ~ N(0, σ²_eps I),    η_s ~ N(0, I_{K_B}).
#
# WHY THE NODE FRAME GIVES O(p) GRADIENTS ON ALL TREE SHAPES
# ----------------------------------------------------------
# In the augmented-NODE frame the latent is the (2p−2)-vector of non-root node
# values u with the SPARSE tree-structured prior precision Q_cond/σ²_phy. A
# tip's phylo effect is the SINGLE node value u_{leaf(t)} scaled by σ_phy[t] —
# NOT a sum over a path of edges. The variance-component envelope gives
#   ∂ℓ/∂σ_phy[t] = n[ n·cc_t·(Σ_φ Λ_φ cc)_t − (Σ_φ Λ_φ C⁻¹)_{tt} ],  cc=C⁻¹m,
# whose data term is one tree solve (O(p)) and whose trace term reduces EXACTLY
# to a NODE-DIAGONAL (`takahashi_diag`, O(nnz L)) plus a rank-K_B Woodbury
# correction. On a tree-augmented `chol_Q_eff` the factor stays sparse on every
# tree shape (balanced AND caterpillar), so the per-species gradient is O(p).
# The global gradients reuse the engine's O(p) machinery: dσ²_phy / dσ²_eps via
# the same-leaf Takahashi selected inverse, dΛ_B via the engine's low-rank
# algebra (`sparse_phy_grad`'s P_A Λ_B block).
#
# DEPENDENCIES (MUST be loaded before this file)
# -----------------------------------------------
# This file is `include`d by `src/GLLVModels.jl` AFTER `sparse_phy_grad.jl` (which
# itself `include`s `takahashi_selinv.jl`) and `sparse_phy.jl`. It relies on
# those already being in module scope:
#   * `SparsePhyState`, `build_sparse_phy_state`, `sparse_phy_value`,
#     `sparse_phy_grad` and the linear-operator helpers `_Cinv`, `_AinvM`,
#     `_MsadM`, `_DKt` (from `sparse_phy_grad.jl`).
#   * `takahashi_diag`, `takahashi_selinv` (from `takahashi_selinv.jl`).
#   * `AugmentedPhy` (from `sparse_phy.jl`).
#
# CONSTRAINT (CHOLMOD Float64-only — evaluation-only for ForwardDiff)
# ------------------------------------------------------------------
# The node-diagonal extraction uses `takahashi_diag(st.chol_Q_eff)`, and
# `chol_Q_eff` is a `SparseArrays.CHOLMOD.Factor{Float64}` — Float64-only, so
# `ForwardDiff.Dual` cannot flow through it. These gradients are therefore the
# ANALYTIC adjoint (a fittable replacement for AD), not an AD-differentiable
# objective: do not attempt to nest them inside ForwardDiff.

using LinearAlgebra
using SparseArrays

# ===========================================================================
# NODE-FRAME ANALYTIC GRADIENT (full phylo_unique model).
#   node_grad(st) -> (; dΛ_B, dσ²_eps, dσ²_phy, dσ_phy), all O(p) on a sparse
#   chol_Q_eff. Restricted to the phylo_unique single augmented column
#   (K_aug == 1), no separate Λ_phy.
# ===========================================================================

# Internal: per-species dσ_phy via the NODE-DIAGONAL + rank-K_B correction.
# `Qeff_diag` (the Takahashi selected-inverse diagonal of `st.chol_Q_eff`) is
# computed ONCE per gradient by the caller (`node_grad`) and passed in here —
# S7 item 3: this used to call `takahashi_diag(st.chol_Q_eff)` itself, a
# second, redundant O(nnz L) pass duplicating `_same_leaf_Msad_inv_diag`'s.
function node_dσ_phy(st::SparsePhyState, cc::AbstractVector, Qeff_diag::AbstractVector)
    p, n, K_B = st.p, st.n, st.K_B
    st.K_aug == 1 ||
        throw(ArgumentError("node_dσ_phy assumes phylo_unique (K_aug=1); got K_aug=$(st.K_aug)"))
    σ_phy = @view st.Λ_aug[:, 1]
    c = st.d_inv[1]                                   # 1/σ²_eps  (constant d_total)

    # NODE-DIAGONAL trace piece: (M_sad⁻¹)_{ll} = Q_eff⁻¹_{ll} + α (X_G S_K⁻¹ X_Gᵀ)_{ll}.
    leafrows = [st.leaf_pos[t] for t in 1:p]
    XGleaf = st.X_G[leafrows, :]                      # p × K_B (X_G stored on st)
    WR = st.chol_S_K \ XGleaf'                        # K_B × p  (= S_K⁻¹ X_G[leaf]ᵀ)
    Msadinv_ll = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        Msadinv_ll[t] = Qeff_diag[st.leaf_pos[t]] + st.α * dot(@view(XGleaf[t, :]), @view(WR[:, t]))
    end

    # rank-K_B correction: σ²_phy (S M_sad⁻¹ D_K' F)_{t,:}·F_{t,:},
    # F (p×K_B) with F Fᵀ = DinvΛB cap⁻¹ DinvΛBᵀ  ⇒  F = DinvΛB · chol_cap.L⁻ᵀ.
    F = collect((st.chol_cap.L \ st.DinvΛB')')        # p × K_B
    DKtF = zeros(Float64, st.total, K_B)              # D_K' F at leaf nodes, scaled by σ_phy
    @inbounds for a in 1:K_B, t in 1:p
        DKtF[st.leaf_pos[t], a] += σ_phy[t] * F[t, a]
    end
    MsadinvDKtF = _MsadM(st, DKtF)                    # total × K_B  (K_B sparse solves)
    SMsadDKtF = MsadinvDKtF[leafrows, :]              # p × K_B
    corr = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        corr[t] = dot(@view(SMsadDKtF[t, :]), @view(F[t, :]))
    end

    # trace term τ_t = (Σ_φ Λ_φ C⁻¹)_{tt}.
    τ = st.σ²_phy .* (c .* σ_phy .* Msadinv_ll .- corr)

    # data term (Σ_φ Λ_φ cc)_t = σ²_phy (S Q_cond⁻¹ S')(Λ_φ cc) — one tree solve.
    Λφcc = σ_phy .* cc
    rhs = zeros(Float64, st.nb)
    @inbounds for t in 1:p
        rhs[st.leaf_pos[t]] = Λφcc[t]
    end
    sol = st.chol_Qcond \ rhs
    ΣφΛφcc = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        ΣφΛφcc[t] = st.σ²_phy * sol[st.leaf_pos[t]]
    end

    dσ_phy = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        dσ_phy[t] = n * (n * cc[t] * ΣφΛφcc[t] - τ[t])
    end
    return dσ_phy
end

# Internal: same-leaf entries of M_sad⁻¹ (K_aug=1 ⇒ a length-p vector).
# `Qeff_diag` is computed once per gradient by the caller and passed in — see
# `node_dσ_phy` above.
function _same_leaf_Msad_inv_diag(st::SparsePhyState, Qeff_diag::AbstractVector)
    p = st.p; K_B = st.K_B
    leafrows = [st.leaf_pos[t] for t in 1:p]
    XGleaf = st.X_G[leafrows, :]
    WR = st.chol_S_K \ XGleaf'
    SL = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        l = st.leaf_pos[t]
        SL[t] = Qeff_diag[l] + st.α * dot(@view(XGleaf[t, :]), @view(WR[:, t]))
    end
    return SL
end

# Internal: scalar global gradients (dσ²_phy, dσ²_eps) via same-leaf Takahashi.
# `Qeff_diag` is computed once per gradient by the caller and passed in.
function node_scalar_grads(st::SparsePhyState, cc::AbstractVector, Ainv_Yc::AbstractMatrix,
                           Qeff_diag::AbstractVector)
    p, n, K_B = st.p, st.n, st.K_B
    σ_phy = @view st.Λ_aug[:, 1]
    SL = _same_leaf_Msad_inv_diag(st, Qeff_diag)      # (M_sad⁻¹)_{ll}, length p

    # dσ²_phy = ½ ⟨P_B, B⟩ / σ²_phy ; ⟨P_B,B⟩ = n[ −tr(C⁻¹B) + n cc'Bcc ].
    # tr(C⁻¹B) = σ²_phy tr(M_sad⁻¹ H), H = D_K'A⁻¹D_K = D_K'D⁻¹D_K − G cap⁻¹ G'.
    # tr(M_sad⁻¹ D_K'D⁻¹D_K) (K_aug=1) = Σ_t σ_phy[t]² d_inv[t] (M_sad⁻¹)_{ll}.
    tr_Msad_DtDinvD = 0.0
    @inbounds for t in 1:p
        tr_Msad_DtDinvD += SL[t] * (σ_phy[t]^2 * st.d_inv[t])
    end
    Msad_G = _MsadM(st, st.G)
    tr_Msad_GcapG = tr(st.chol_cap \ (st.G' * Msad_G))
    tr_Msad_H = tr_Msad_DtDinvD - tr_Msad_GcapG
    trCinvB = st.σ²_phy * tr_Msad_H
    Bcc = _Bphy_apply(st, cc)
    ccBcc = dot(cc, Bcc)
    dσ²_phy = 0.5 * (n * (-trCinvB + n * ccBcc)) / st.σ²_phy

    # dσ²_eps = ½ tr(P_A). trCinv = trAinv − α⟨LB,Z⟩, Z=D_K'A⁻²D_K. A⁻¹=cI−FFᵀ.
    trAinv = _trAinv(st)
    c = st.d_inv[1]
    F = collect((st.chol_cap.L \ st.DinvΛB')')        # p × K_B
    M_F = F' * F
    G_F = 2c .* Matrix{Float64}(I, K_B, K_B) .- M_F
    # (i) c² same-leaf term (K_aug=1): c² Σ_t σ_phy[t]² (M_sad⁻¹)_{ll}.
    sameleaf_c2 = 0.0
    @inbounds for t in 1:p
        sameleaf_c2 += σ_phy[t]^2 * SL[t]
    end
    sameleaf_c2 *= c^2
    # (ii) rank-K_B trace: tr(G_F · (Wᵀ M_sad⁻¹ W)), W = D_Kᵀ F.
    W = Matrix{Float64}(undef, st.total, K_B)
    @inbounds for a in 1:K_B
        W[:, a] = _DKt(st, @view F[:, a])
    end
    MW = _MsadM(st, W)
    WtMW = W' * MW
    lowrank = tr(G_F * WtMW)
    sum_LB_Z = sameleaf_c2 - lowrank
    trCinv = trAinv - st.α * sum_LB_Z
    trPA = -trCinv + n * dot(cc, cc) - (n - 1) * trAinv + sum(Ainv_Yc .^ 2)
    dσ²_eps = 0.5 * trPA

    return dσ²_phy, dσ²_eps
end

# Internal: dΛ_B (engine's O(p) block, P_A Λ_B).
function node_dΛ_B(st::SparsePhyState, cc::AbstractVector, Ainv_Yc::AbstractMatrix)
    p, n = st.p, st.n
    Cinv_LB = _CinvM(st, st.Λ_B)
    Ainv_LB = _AinvM(st, st.Λ_B)
    ccLB = cc * (cc' * st.Λ_B)
    AYcLB = Ainv_Yc * (Ainv_Yc' * st.Λ_B)
    return (-Cinv_LB) .+ n .* ccLB .- (n - 1) .* Ainv_LB .+ AYcLB
end

# S7 item 3 call-count evidence: `test/test_sparse_phy_identities.jl --gate
# gradient` (G7.2) resets this counter, calls `node_grad` once, and asserts it
# reads 1 — before the dedup, `node_dσ_phy` and `node_scalar_grads` (via
# `_same_leaf_Msad_inv_diag`) each called `takahashi_diag(st.chol_Q_eff)`
# independently, i.e. 2 calls per `node_grad` invocation.
const _NODE_GRAD_TAKAHASHI_CALLS = Ref(0)
_node_grad_takahashi_calls_reset!() = (_NODE_GRAD_TAKAHASHI_CALLS[] = 0; nothing)
_node_grad_takahashi_calls() = _NODE_GRAD_TAKAHASHI_CALLS[]

"""
    node_grad(st::SparsePhyState) -> (; dΛ_B, dσ²_eps, dσ²_phy, dσ_phy)

Full NODE-FRAME analytic gradient of the phylo_unique marginal log-likelihood
`sparse_phy_value(st)`. Every block is O(p) given a sparse `chol_Q_eff`:
`dσ_phy` from the node-diagonal `takahashi_diag` + rank-K_B Woodbury correction;
`dσ²_phy` / `dσ²_eps` from the same-leaf Takahashi selected inverse; `dΛ_B`
from the engine's low-rank algebra (identical to `sparse_phy_grad`'s `P_A Λ_B`
block).

Returns the same gradient as the engine's `sparse_phy_grad` for a phylo_unique
state (verified to machine precision, ≤1e-13), but extracts the tree-coupled
trace from a NODE DIAGONAL rather than a dense leaf-leaf block, so it stays
O(p) on every tree shape (balanced AND caterpillar).

Assumes the phylo_unique configuration (`st.K_aug == 1`): one shared per-trait
phylogenetic random effect with SDs `σ_phy = st.Λ_aug[:, 1]` and no separate
`Λ_phy` axis.

Evaluation-only for ForwardDiff: the node-diagonal uses a CHOLMOD Float64
factor (see file header).
"""
function node_grad(st::SparsePhyState)
    cc = _Cinv(st, st.m)
    Ainv_Yc = _AinvM(st, st.Y_c)
    # S7 item 3: computed ONCE per gradient and passed to both consumers
    # below (`node_dσ_phy` and `node_scalar_grads` via `_same_leaf_Msad_inv_diag`)
    # instead of each recomputing `takahashi_diag(st.chol_Q_eff)` independently.
    Qeff_diag = takahashi_diag(st.chol_Q_eff)
    _NODE_GRAD_TAKAHASHI_CALLS[] += 1
    dσ_phy = node_dσ_phy(st, cc, Qeff_diag)
    dσ²_phy, dσ²_eps = node_scalar_grads(st, cc, Ainv_Yc, Qeff_diag)
    dΛ_B = node_dΛ_B(st, cc, Ainv_Yc)
    return (; dΛ_B, dσ²_eps, dσ²_phy, dσ_phy)
end

"""
    node_dσ_phy_only(st::SparsePhyState) -> Vector{Float64}

Per-species `dσ_phy` block alone (length `p`), sharing the O(p) `cc = C⁻¹m`
solve. This is the headline node-diagonal object — the apples-to-apples
analogue of the edge-frame per-species gradient — isolated from the global
`dΛ_B` / `dσ²_eps` / `dσ²_phy` work for timing and scaling studies.
"""
node_dσ_phy_only(st::SparsePhyState) =
    node_dσ_phy(st, _Cinv(st, st.m), takahashi_diag(st.chol_Q_eff))

# ===========================================================================
# MATCHED single-trait per-species node gradient.
#   Model: single trait, fixed μ, σ²_phy folded into σ_phy, NO Λ_B:
#       Σ = σ²_eps I + Λ_φ Σ_phy Λ_φ,   Λ_φ = diag(σ_phy).
#   With A = σ²_eps I the rank-K_B Woodbury correction VANISHES, so the
#   per-species trace term is PURELY the NODE DIAGONAL (Λ̃⁻¹)_{leaf(t),leaf(t)}
#   of the node precision
#       Λ̃ = Q_cond + σ_eps⁻² S' Λ_φ² S       (sparse, tree-structured),
#   obtained by `takahashi_diag(chol(Λ̃))` in O(nnz L):
#       trace_t = 2 σ_eps⁻² σ_phy[t] (Λ̃⁻¹)_{leaf(t),leaf(t)}     (node diagonal)
#       dataq_t = 2 u_t (Σ_φ Λ_φ u)_t,   u = Σ⁻¹ (y − μ)         (two tree solves)
#       g[t]    = ½ (trace_t − dataq_t).
# ===========================================================================

"""
    NodePerSpecies

Pre-factorised node-frame solver for the MATCHED single-trait per-species
phylogenetic gradient (single trait, fixed mean `μ`, `σ²_phy` folded into the
per-tip `σ_phy`, no `Λ_B`). Holds the tree precision factorisation `chol_Qcond`
and the augmented node precision `Λ̃ = Q_cond + σ_eps⁻² S' Λ_φ² S` with its
Cholesky `cΛ̃`, so `grad_node_perspecies` / `node_blups` apply in O(p) per call.

Built by `build_node_perspecies`. The CHOLMOD factors are Float64-only
(evaluation-only for ForwardDiff; see file header).

Fields
------
* `phy`        – the `AugmentedPhy{Float64}` tree.
* `σ_phy`      – per-tip phylo SDs (length p), with `σ²_phy` already folded in.
* `σ²_eps`     – residual variance.
* `nb`         – number of non-root augmented nodes (= 2p − 2).
* `leaf_pos`   – maps tip t to its row in the root-dropped node frame.
* `chol_Qcond` – Cholesky of the root-dropped tree precision `Q_cond`.
* `Λ̃`         – augmented node precision `Q_cond + σ_eps⁻² S' Λ_φ² S`.
* `cΛ̃`        – Cholesky of `Λ̃`.
"""
# Parametric on the factor type `TF` (mirrors `SparsePhyState{TF}`): the
# CHOLMOD factor's index type is concrete per instance, so `st.cΛ̃ \ …` and
# `takahashi_diag(st.cΛ̃)` dispatch statically instead of at runtime.
struct NodePerSpecies{TF<:SparseArrays.CHOLMOD.Factor{Float64}}
    phy::AugmentedPhy{Float64}
    σ_phy::Vector{Float64}
    σ²_eps::Float64
    nb::Int
    leaf_pos::Vector{Int}
    chol_Qcond::TF
    Λ̃::SparseMatrixCSC{Float64,Int}
    cΛ̃::TF
end

"""
    build_node_perspecies(phy::AugmentedPhy{Float64}, σ_phy, σ²_eps) -> NodePerSpecies

Assemble a `NodePerSpecies` solver for the matched single-trait per-species
model. `σ_phy` is the length-`p` per-tip phylogenetic SD (with `σ²_phy` folded
in); `σ²_eps` is the residual variance. Builds the root-dropped tree precision
`Q_cond`, the augmented node precision `Λ̃ = Q_cond + σ_eps⁻² S' diag(σ_phy²) S`,
and their sparse Cholesky factors (O(p) on a tree).
"""
function build_node_perspecies(phy::AugmentedPhy{Float64}, σ_phy::AbstractVector,
                               σ²_eps::Real)
    keep = filter(i -> i != phy.root_index, 1:phy.n_total)
    Qc = phy.Q_topology[keep, keep]
    nb = size(Qc, 1); p = phy.n_leaves
    lp = Vector{Int}(undef, p)
    @inbounds for t in 1:p
        l = phy.leaf_indices[t]; lp[t] = phy.root_index < l ? l - 1 : l
    end
    inve = 1.0 / float(σ²_eps)
    Λ̃ = copy(Qc)
    @inbounds for t in 1:p
        Λ̃[lp[t], lp[t]] += inve * float(σ_phy[t])^2
    end
    return NodePerSpecies(phy, Vector{Float64}(σ_phy), float(σ²_eps), nb, lp,
                          cholesky(Symmetric(Qc)), Λ̃, cholesky(Symmetric(Λ̃)))
end

"""
    grad_node_perspecies(st::NodePerSpecies, y::AbstractVector, μ::Real) -> Vector{Float64}

Per-species gradient `∂negll/∂σ_phy[t]` (length `p`) of the matched
single-trait fixed-`μ` phylogenetic model at the data vector `y` (length `p`)
and mean `μ`. Each entry is

    g[t] = ½ (trace_t − dataq_t),
    trace_t = 2 σ_eps⁻² σ_phy[t] (Λ̃⁻¹)_{leaf(t),leaf(t)},
    dataq_t = 2 u_t (Σ_φ Λ_φ u)_t,   u = Σ⁻¹ (y − μ),

with the node-diagonal `(Λ̃⁻¹)_{ll}` from `takahashi_diag` (O(nnz L)) and `u`
via a Woodbury solve through `Λ̃`. O(p) given the `NodePerSpecies`
pre-factorisation. Matches the edge-frame per-species gradient to machine
precision while needing only a node diagonal (no ancestor–descendant
path-pairs).

Sign convention: returns `∂negll/∂σ_phy` — the gradient of the *negative*
log-likelihood (what an optimiser minimises; consumed by the O(p) single-trait
fitter). This is the OPPOSITE sign to [`node_grad`](@ref), whose blocks are
`∂loglik/∂θ`. Verified against central FD of `+negll` to `rel < 1e-6`.
"""
function grad_node_perspecies(st::NodePerSpecies, y::AbstractVector, μ::Real)
    p = st.phy.n_leaves; inve = 1.0 / st.σ²_eps; sp = st.σ_phy
    r = y .- μ
    # u = Σ⁻¹ r via Woodbury: Σ⁻¹ = inve I − inve² Λ_φ S Λ̃⁻¹ S' Λ_φ.
    Λφr = sp .* r
    rhs = zeros(Float64, st.nb)
    @inbounds for t in 1:p; rhs[st.leaf_pos[t]] = Λφr[t]; end
    sol = (st.cΛ̃ \ rhs)::Vector{Float64}     # assert: CHOLMOD `\` is not inferred
    u = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        u[t] = inve * r[t] - inve^2 * sp[t] * sol[st.leaf_pos[t]]
    end
    # Σ_φ Λ_φ u = S Q_cond⁻¹ S' (Λ_φ u).
    Λφu = sp .* u
    rhs2 = zeros(Float64, st.nb)
    @inbounds for t in 1:p; rhs2[st.leaf_pos[t]] = Λφu[t]; end
    sol2 = (st.chol_Qcond \ rhs2)::Vector{Float64}
    # node diagonal of Λ̃⁻¹ (selected-inverse diagonal), O(nnz L).
    dg = takahashi_diag(st.cΛ̃)
    g = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        trace_t = 2 * inve * sp[t] * dg[st.leaf_pos[t]]
        dataq_t = 2 * u[t] * sol2[st.leaf_pos[t]]
        g[t] = 0.5 * (trace_t - dataq_t)
    end
    return g
end

"""
    node_blups(st::NodePerSpecies, y::AbstractVector, μ::Real) -> (û, ẑ_tip)

Ancestral node posterior-mean BLUPs for the matched single-trait model. The
node posterior over `u` (non-root augmented nodes) has precision `Λ̃` and mean

    û = Λ̃⁻¹ (σ_eps⁻² S' Λ_φ (y − μ)),

and the tip phylo BLUP on the data scale is `ẑ_tip[t] = σ_phy[t] û[leaf(t)]`.
Returns `(û, ẑ_tip)` with `û` indexed in the root-dropped node ordering.

CAVEAT. Node-frame posterior-mean BLUPs `û` are exact (≤8e-16 vs the dense
reference). Edge-frame branch increments derived by differencing
(`û_child − û_parent`) differ from the edge-frame (P2) representation by a
`√σ²_phy`-scale convention; do not treat them as equivalent branch BLUPs.
"""
function node_blups(st::NodePerSpecies, y::AbstractVector, μ::Real)
    p = st.phy.n_leaves; inve = 1.0 / st.σ²_eps; sp = st.σ_phy
    Λφr = sp .* (y .- μ)
    rhs = zeros(Float64, st.nb)
    @inbounds for t in 1:p; rhs[st.leaf_pos[t]] = inve * Λφr[t]; end
    û = st.cΛ̃ \ rhs                              # node posterior mean (non-root)
    ẑ_tip = [sp[t] * û[st.leaf_pos[t]] for t in 1:p]
    return û, ẑ_tip
end
