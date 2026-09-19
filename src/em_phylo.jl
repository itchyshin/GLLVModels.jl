# Gradient-free EM for the Gaussian phylogenetic GLLVM (phylo_unique config).
#
# This is the phylo extension of `em_fa.jl`. It fits the SAME model that
# `fit_gaussian_gllvm(y; K, has_phy_unique = true, Σ_phy = Σ_phy)` fits via
# gradient-based optimisation, but with closed-form EM updates that need NO
# gradient — only fast linear solves. The phylo solves can be done with the
# dense Σ_phy (reference) or with the augmented-state sparse precision
# (`AugmentedPhy`, the O(p) path in `sparse_phy.jl`), sidestepping the
# CHOLMOD autodiff limitation that blocks the gradient-based fit on the fast
# path.
#
# ---------------------------------------------------------------------------
# Model (phylo_unique, K_B site factors + per-trait phylo random effect)
# ---------------------------------------------------------------------------
#   y[:, s] = Λ_B η_s + diag(σ_phy) φ + ε_s ,   s = 1, …, n
#   η_s ~ N(0, I_{K_B})        (per-site latent factors, independent)
#   φ   ~ N(0, Σ_phy)          (ONE shared phylo random effect, length p)
#   ε_s ~ N(0, σ²_eps I_p)
#
# Σ_phy (p × p) is the FIXED tree-derived species covariance. The free
# parameters are Λ_B (p × K_B), σ_eps (> 0), and the per-trait phylo SDs
# σ_phy (length p). With Λ_φ ≡ diag(σ_phy) the phylo block of cov(vec(y)) is
# J_n ⊗ B,  B = (σ_phy σ_phy') ∘ Σ_phy = Λ_φ Σ_phy Λ_φ, and the site block is
# I_n ⊗ A,  A = Λ_B Λ_B' + σ²_eps I. This is EXACTLY the dense J3
# phylo_unique covariance in `likelihood.jl`.
#
# ---------------------------------------------------------------------------
# EM (Rubin & Thayer 1982, generalised with a per-trait-shared latent φ)
# ---------------------------------------------------------------------------
# Missing data Z = (η_1, …, η_n, φ). All Gaussian ⇒ E-step is exact.
#
# E-step. Let m = mean_s(y_s), β = Λ_B' A⁻¹.
#   φ posterior (treat m as one obs with noise A/n, conjugate Gaussian):
#       V_φ = (Σ_phy⁻¹ + n Λ_φ A⁻¹ Λ_φ)⁻¹
#       μ_φ = V_φ (n Λ_φ A⁻¹ m)
#   Equivalent z-space form (z = Λ_φ φ, the phylo effect on the data scale):
#       μ_z = n B (A + n B)⁻¹ m        (the ancestral-state BLUP)
#       V_z = B − n B (A + n B)⁻¹ B
#   η_s posterior (law of total expectation over φ):
#       E[η_s|Y]    = β (y_s − Λ_φ μ_φ)
#       Cov(η_s|Y)  = (I − β Λ_B) + β Λ_φ V_φ Λ_φ β'
#       Cov(η_s,φ|Y) = − β Λ_φ V_φ
#
# M-step (closed form). Per trait t, regress y[t, :] on the latent design
# u_s = (η_s, φ[t]); coefficients (Λ_B[t, :], σ_phy[t]):
#       G_t = Σ_s E[u_s u_s'|Y]   ((K_B+1) × (K_B+1))
#       h_t = Σ_s E[u_s y[t,s]|Y] (K_B+1)
#       (Λ_B[t,:], σ_phy[t]) = G_t⁻¹ h_t
# σ²_eps = (Σ_{t,s} y[t,s]² − Σ_t θ_t' h_t) / (n p)   (WLS residual trace).
#
# Monotone non-decrease of the marginal log-lik is an EM invariant and is
# asserted by the caller / tests. The marginal log-lik itself is evaluated
# with the dense closed form `gaussian_marginal_loglik` so the EM trajectory
# is comparable to the gradient-based fit to machine precision.

using LinearAlgebra
using SparseArrays
using Statistics
using ForwardDiff

# Takahashi (1973) / Erisman–Tinney (1975) selected inverse — used by the
# sparse E-step (`_estep_sparse`) below to compute `diag(V_φ)` in O(p)
# instead of the dense `inv(cVφ)`'s O(p³). Defined in `src/takahashi_selinv.jl`.

# ---------------------------------------------------------------------------
# Sparse (A + n B)⁻¹ apply via the augmented-state saddle point.
# ---------------------------------------------------------------------------
# B = Λ_φ Σ_phy Λ_φ for phylo_unique (Λ_aug = σ_phy, a single column). The
# augmented precision represents Σ_phy = σ²_phy · S Q_cond⁻¹ S'. With α = n σ²_phy
# the system (A + n B) v = rhs is solved by the Schur complement (mirrors the
# determinant/quadratic machinery in `likelihood_sparse_phy.jl`):
#       M_sad η = D_K' A⁻¹ rhs ,    v = A⁻¹ (rhs − α D_K η)
#       M_sad   = Q_eff − α G cap⁻¹ G'
# Q_eff = Q_cond + α (S' diag(σ_phy²/d) S) (sparse, O(p) factorisation), G is
# the rank-K_B Woodbury coupling, cap = I + Λ_B' D⁻¹ Λ_B. This returns the
# ancestral-state BLUP machinery without ever forming the dense Σ_phy.

# ---------------------------------------------------------------------------
# CHOLMOD symbolic-reuse for `chol_Q_eff` across the EM iterations of ONE
# `em_fit_phylo` call (S7 item 4 — measured, not assumed: a throwaway
# before/after timing on the p=1000 fixture found CHOLMOD's symbolic analysis
# (AMD ordering + elimination tree) is ~70% of one fresh
# `cholesky(Symmetric(Q_eff))` factorisation, well above the ledger's 10%
# land/abandon threshold). `Q_eff`'s sparsity PATTERN depends only on the
# tree topology and K_B, both fixed for the lifetime of an EM fit — only the
# diagonal VALUES change as σ_phy / σ_eps update each M-step — so the
# symbolic step is redundant on every iteration after the first.
#
# Mirrors `_grouped_cached_cholesky!` (src/grouped_laplace.jl, S7b) exactly:
# pattern-checked on every call, falls back to a fresh factorisation (counted)
# if the pattern ever differs, never silently misfactorizes. Scope is ONE
# `chol_cache::Base.RefValue` per caller (`em_fit_phylo` builds one before its
# loop and threads it through every `_estep_sparse` call for that fit;
# `blup_phylo_sparse` and other one-shot callers pass `chol_cache = nothing`
# and get the old fresh-every-call behaviour).
mutable struct _EMPhyloCholStats
    calls::Int      # _em_phylo_cached_cholesky! invocations counted
    fresh::Int      # fresh cholesky(...) symbolic+numeric factorizations
    reused::Int     # cholesky!(...) numeric-only refactorizations (cache hit)
    fallback::Int   # a cached factor existed but its sparsity pattern changed
end

const _EM_PHYLO_CHOL_STATS = _EMPhyloCholStats(0, 0, 0, 0)

_em_phylo_chol_stats_reset!() = begin
    _EM_PHYLO_CHOL_STATS.calls = 0
    _EM_PHYLO_CHOL_STATS.fresh = 0
    _EM_PHYLO_CHOL_STATS.reused = 0
    _EM_PHYLO_CHOL_STATS.fallback = 0
    nothing
end

_em_phylo_chol_stats() = (calls = _EM_PHYLO_CHOL_STATS.calls, fresh = _EM_PHYLO_CHOL_STATS.fresh,
    reused = _EM_PHYLO_CHOL_STATS.reused, fallback = _EM_PHYLO_CHOL_STATS.fallback)

function _em_phylo_cached_cholesky!(cache::Base.RefValue,
        A::Symmetric{Float64, <:SparseMatrixCSC{Float64, Int}})
    _EM_PHYLO_CHOL_STATS.calls += 1
    S = parent(A)
    cached = cache[]
    if cached !== nothing
        factor, rowval, colptr = cached
        if size(factor) == size(S) && rowval == S.rowval && colptr == S.colptr
            _EM_PHYLO_CHOL_STATS.reused += 1
            return cholesky!(factor, A)
        end
        _EM_PHYLO_CHOL_STATS.fallback += 1
    end
    factor = cholesky(A)
    _EM_PHYLO_CHOL_STATS.fresh += 1
    cache[] = (factor, copy(S.rowval), copy(S.colptr))
    return factor
end

"""
    AnBSparseSolver

Pre-factorised augmented-state solver for `(A + n B)` where
`A = Λ_B Λ_B' + σ²_eps I` and `B = diag(σ_phy) Σ_phy diag(σ_phy)`, with
`Σ_phy` represented by an `AugmentedPhy` (sparse precision). Built once per
E-step; applies `(A + n B)⁻¹` to vectors in O(p) per solve.

Reuses the saddle-point factorisation strategy of
`gaussian_marginal_loglik_sparse_phy` (`likelihood_sparse_phy.jl`).
"""
struct AnBSparseSolver
    phy::GLLVModels.AugmentedPhy{Float64}
    n_block::Int
    leaf_pos::Vector{Int}
    d_total::Vector{Float64}
    d_inv::Vector{Float64}
    Λ_B::Matrix{Float64}
    DinvΛB::Matrix{Float64}
    chol_cap::Cholesky{Float64,Matrix{Float64}}
    chol_Q_eff::SparseArrays.CHOLMOD.Factor{Float64}
    G::Matrix{Float64}
    chol_S_K::Cholesky{Float64,Matrix{Float64}}
    σ_phy::Vector{Float64}
    α::Float64
    X_G::Matrix{Float64}   # chol_Q_eff \ G, stored once per factor (S7 item 1:
                            # `_estep_sparse` used to recompute this)
end

"""
    build_AnB_sparse(Λ_B, σ_eps, σ_phy, phy, n; σ²_phy=1.0, chol_cache=nothing) -> AnBSparseSolver

Factorise the augmented-state representation of `(A + n B)` for the
phylo_unique model. `phy::AugmentedPhy` supplies the sparse Σ_phy precision;
`σ²_phy` scales it (Σ_phy = σ²_phy · S Q_cond⁻¹ S').

`chol_cache`, when a `Base.RefValue` (see `_em_phylo_cached_cholesky!` above),
lets `chol_Q_eff` reuse a prior call's CHOLMOD symbolic analysis when the
sparsity pattern is unchanged (S7 item 4) — pass the SAME `Ref` across the
EM iterations of one fit. `nothing` (default) always factorises fresh.
"""
function build_AnB_sparse(Λ_B::AbstractMatrix, σ_eps::Real,
                          σ_phy::AbstractVector, phy::GLLVModels.AugmentedPhy,
                          n::Integer; σ²_phy::Real = 1.0,
                          chol_cache::Union{Nothing,Base.RefValue} = nothing)
    p   = phy.n_leaves
    K_B = size(Λ_B, 2)
    σ²  = float(σ_eps)^2
    Λ_B64 = Matrix{Float64}(Λ_B)
    σ_phy64 = Vector{Float64}(σ_phy)

    d_total = fill(σ², p)                      # A = Λ_B Λ_B' + σ²_eps I
    d_inv   = 1.0 ./ d_total
    DinvΛB  = d_inv .* Λ_B64                    # p × K_B
    cap     = Matrix(I + Λ_B64' * DinvΛB)       # K_B × K_B
    chol_cap = cholesky(Symmetric((cap + cap') ./ 2))

    keep    = filter(i -> i != phy.root_index, 1:phy.n_total)
    Q_cond  = phy.Q_topology[keep, keep]
    n_block = size(Q_cond, 1)
    leaf_pos = Vector{Int}(undef, p)
    @inbounds for t in 1:p
        lp = phy.leaf_indices[t]
        phy.root_index < lp && (lp -= 1)
        leaf_pos[t] = lp
    end

    α = n * float(σ²_phy)

    # Q_eff = Q_cond + α · (S' diag(σ_phy² / d_total) S)   (K_aug = 1 here)
    I_q = Int[]; J_q = Int[]; V_q = Float64[]
    rows = rowvals(Q_cond); vals = nonzeros(Q_cond)
    sizehint!(I_q, nnz(Q_cond) + p)
    sizehint!(J_q, nnz(Q_cond) + p)
    sizehint!(V_q, nnz(Q_cond) + p)
    for j in 1:n_block
        for idx in nzrange(Q_cond, j)
            push!(I_q, rows[idx]); push!(J_q, j); push!(V_q, vals[idx])
        end
    end
    @inbounds for t in 1:p
        push!(I_q, leaf_pos[t]); push!(J_q, leaf_pos[t])
        push!(V_q, α * σ_phy64[t]^2 * d_inv[t])
    end
    Q_eff = sparse(I_q, J_q, V_q, n_block, n_block)
    chol_Q_eff = chol_cache === nothing ?
        cholesky(Symmetric(Q_eff)) :
        _em_phylo_cached_cholesky!(chol_cache, Symmetric(Q_eff))

    # G[(leaf_pos[t]), j] = σ_phy[t] · d_inv[t] · Λ_B[t, j]
    G = zeros(Float64, n_block, K_B)
    @inbounds for t in 1:p
        factor = σ_phy64[t] * d_inv[t]
        for j in 1:K_B
            G[leaf_pos[t], j] = factor * Λ_B64[t, j]
        end
    end
    X_G = chol_Q_eff \ G
    M_K = G' * X_G
    S_K = cap .- α .* M_K
    chol_S_K = cholesky(Symmetric((S_K + S_K') ./ 2))

    return AnBSparseSolver(phy, n_block, leaf_pos, d_total, d_inv,
                           Λ_B64, DinvΛB, chol_cap, chol_Q_eff, G, chol_S_K,
                           σ_phy64, α, X_G)
end

# A⁻¹ b via Woodbury for A = D + Λ_B Λ_B'.
@inline function _Ainv(s::AnBSparseSolver, b::AbstractVector)
    Dinv_b = s.d_inv .* b
    return Dinv_b .- s.DinvΛB * (s.chol_cap \ (s.Λ_B' * Dinv_b))
end

"""
    solve_AnB(s::AnBSparseSolver, rhs) -> v

Apply `(A + n B)⁻¹` to `rhs` (length p) via the sparse augmented-state
saddle-point. O(p) given the pre-factorisation.
"""
function solve_AnB(s::AnBSparseSolver, rhs::AbstractVector)
    p = s.phy.n_leaves
    Ainv_rhs = _Ainv(s, rhs)
    # b = D_K' A⁻¹ rhs  (concentrated at leaf positions, scaled by σ_phy)
    b = zeros(Float64, s.n_block)
    @inbounds for t in 1:p
        b[s.leaf_pos[t]] = s.σ_phy[t] * Ainv_rhs[t]
    end
    ξ0 = s.chol_Q_eff \ b
    yK = s.chol_S_K \ (s.G' * ξ0)
    ξ  = ξ0 .+ s.α .* (s.chol_Q_eff \ (s.G * yK))   # M_sad⁻¹ b
    # v = A⁻¹ rhs − α A⁻¹ D_K ξ ;  (D_K ξ)[t] = σ_phy[t] ξ[leaf_pos[t]]
    DKξ = zeros(Float64, p)
    @inbounds for t in 1:p
        DKξ[t] = s.σ_phy[t] * ξ[s.leaf_pos[t]]
    end
    return Ainv_rhs .- s.α .* _Ainv(s, DKξ)
end

"""
    blup_phylo_sparse(y, Λ_B, σ_eps, σ_phy, phy; σ²_phy=1.0) -> μ_z

Ancestral-state BLUP of the phylo random effect on the data scale,
`μ_z = n B (A + n B)⁻¹ m` with `m = mean_s(y_s)`, computed via the sparse
augmented-state solve (no dense Σ_phy). `B v = Λ_φ Σ_phy Λ_φ v` is applied
through the same augmented machinery.
"""
function blup_phylo_sparse(y::AbstractMatrix, Λ_B::AbstractMatrix, σ_eps::Real,
                           σ_phy::AbstractVector, phy::GLLVModels.AugmentedPhy;
                           σ²_phy::Real = 1.0)
    p, n = size(y)
    s = build_AnB_sparse(Λ_B, σ_eps, σ_phy, phy, n; σ²_phy = σ²_phy)
    m = vec(sum(Matrix{Float64}(y), dims = 2)) ./ n
    w = solve_AnB(s, m)                          # (A + n B)⁻¹ m
    # μ_z = n B w ;  B w = Λ_φ Σ_phy Λ_φ w. Σ_phy x via augmented solve:
    # Σ_phy x = σ²_phy S Q_cond⁻¹ S' x  (S' x concentrated at leaf positions).
    Λφw = σ_phy .* w
    return μ_z_from_components(s, σ²_phy, Λφw, n)
end

# Helper: μ_z = n Λ_φ Σ_phy Λ_φ (A+nB)⁻¹ m, with Σ_phy applied via Q_cond.
# We need a Q_cond Cholesky distinct from Q_eff; build lazily here. To keep
# `AnBSparseSolver` lean we recompute the Σ_phy apply directly from phy.
function μ_z_from_components(s::AnBSparseSolver, σ²_phy::Real,
                            Λφw::AbstractVector, n::Integer)
    p = s.phy.n_leaves
    keep   = filter(i -> i != s.phy.root_index, 1:s.phy.n_total)
    Q_cond = s.phy.Q_topology[keep, keep]
    chol_Qcond = cholesky(Symmetric(Q_cond))
    rhs = zeros(Float64, s.n_block)
    @inbounds for t in 1:p
        rhs[s.leaf_pos[t]] = Λφw[t]
    end
    sol = chol_Qcond \ rhs
    Σφw = Vector{Float64}(undef, p)              # Σ_phy (Λ_φ w)
    @inbounds for t in 1:p
        Σφw[t] = σ²_phy * sol[s.leaf_pos[t]]
    end
    return n .* (s.σ_phy .* Σφw)                 # μ_z = n Λ_φ Σ_phy Λ_φ w
end

# ---------------------------------------------------------------------------
# Dense E-step + M-step (reference path; drives the EM fit).
# ---------------------------------------------------------------------------

# Dense E-step. Returns the sufficient statistics the M-step consumes plus
# the BLUPs. `A = Λ_B Λ_B' + σ²_eps I`, `B = (σ_phy σ_phy') ∘ Σ_phy`.
function _estep_dense(y::AbstractMatrix, Λ_B::AbstractMatrix, σ_eps::Real,
                      σ_phy::AbstractVector, Σ_phy::AbstractMatrix)
    p, n = size(y)
    K_B  = size(Λ_B, 2)
    σ²   = float(σ_eps)^2

    A  = Λ_B * Λ_B'
    @inbounds for t in 1:p
        A[t, t] += σ²
    end
    cA = cholesky(Symmetric((A + A') ./ 2))
    β  = Λ_B' / cA                                # K_B × p  (= Λ_B' A⁻¹)

    m   = vec(sum(y, dims = 2)) ./ n              # length p
    Λφ  = σ_phy                                   # diag(Λ_φ) as a vector

    # φ posterior: V_φ = (Σ_phy⁻¹ + n Λ_φ A⁻¹ Λ_φ)⁻¹, μ_φ = V_φ n Λ_φ A⁻¹ m.
    Ainv_Λφ = cA \ Diagonal(Λφ)                   # A⁻¹ Λ_φ  (p × p)
    Vφ_inv  = inv(Symmetric((Σ_phy + Σ_phy') ./ 2)) .+ n .* (Diagonal(Λφ) * Ainv_Λφ)
    cVφ     = cholesky(Symmetric((Vφ_inv + Vφ_inv') ./ 2))
    Vφ      = inv(cVφ)                            # p × p
    μ_φ     = Vφ * (n .* (Λφ .* (cA \ m)))        # length p

    # η posterior aggregated over sites.
    ImβΛ   = I - β * Λ_B                          # K_B × K_B  (= I − β Λ_B)
    βΛφ    = β .* reshape(Λφ, 1, p)               # K_B × p   (β Λ_φ, scale cols)
    βΛφVφ  = βΛφ * Vφ                             # K_B × p   (β Λ_φ V_φ)
    # E[η_s|Y] = β (y_s − Λ_φ μ_φ)
    zhat   = Λφ .* μ_φ                            # Λ_φ μ_φ  (= μ_z, BLUP)
    Eη     = β * (y .- reshape(zhat, p, 1))       # K_B × n
    sumEη  = vec(sum(Eη, dims = 2))               # K_B

    # Sufficient statistics for the M-step.
    # S_ηη = Σ_s E[η_s η_s'|Y] = n(I − βΛ_B) + n β Λ_φ V_φ Λ_φ β' + Eη Eη'
    S_ηη = n .* ImβΛ .+ n .* (βΛφVφ * βΛφ') .+ Eη * Eη'
    S_ηη = Symmetric((S_ηη + S_ηη') ./ 2)
    # E[φ[t]²|Y] = V_φ[t,t] + μ_φ[t]²
    Eφ2  = diag(Vφ) .+ μ_φ .^ 2                   # length p
    # Σ_s E[η_s φ[t]|Y] = sumEη μ_φ[t] − n (β Λ_φ V_φ)[:,t]
    #   stored as a K_B × p matrix C: C[:,t]
    C_ηφ = sumEη * μ_φ' .- n .* βΛφVφ             # K_B × p
    # Σ_s E[η_s y[t,s]|Y] = Σ_s E[η_s|Y] y[t,s]  →  K_B × p, col t
    H_ηy = Eη * y'                                # K_B × p (H_ηy[:,t] = Σ_s Eη_s y[t,s])

    return (; β, m, Eη, sumEη, S_ηη, Eφ2, C_ηφ, H_ηy, μ_φ, μ_z = zhat, Vφ)
end

# ---------------------------------------------------------------------------
# Sparse E-step: same sufficient statistics, but the per-trait variance
# `diag(V_φ)` is obtained via Takahashi-selected inverse on the augmented
# precision (O(p)) and the K_B-wide quantity `β Λ_φ V_φ` is obtained via
# K_B sparse solves with `M_sad`. The dense `inv(cVφ)` of `_estep_dense`
# (an O(p³) per-iteration call) is REMOVED in this path — that is the
# headline PERF gate of the EM swap.
#
# The augmented machinery is the SAME `AnBSparseSolver` already used for
# `solve_AnB` / `blup_phylo_sparse`; the extra ingredients here are
#   * `μ_φ` via the augmented-state posterior solve, and
#   * `diag(V_φ)` via Takahashi selected inverse of `Q_eff`, then the
#     rank-K_B Woodbury correction applied entry-by-entry at leaves.
#
# COST per E-step (with K_B small):
#   * `AnBSparseSolver` build (incl. K_B Q_eff solves for X_G)   O(K_B·p)
#   * 1 + K_B sparse solves for (μ_φ, β Λ_φ V_φ rows)             O(K_B·p)
#   * Takahashi selected inverse of Q_eff (diag)                  O(p)
#   * Rank-K_B Woodbury per leaf                                  O(K_B²·p)
# Total: O(K_B²·p). Linear in p — vs the dense E-step's O(p³).
# ---------------------------------------------------------------------------
function _estep_sparse(y::AbstractMatrix, Λ_B::AbstractMatrix, σ_eps::Real,
                      σ_phy::AbstractVector,
                      phy::GLLVModels.AugmentedPhy{Float64};
                      σ²_phy::Real = 1.0,
                      chol_cache::Union{Nothing,Base.RefValue} = nothing)
    p, n = size(y)
    p == phy.n_leaves ||
        throw(ArgumentError("y first dim ($p) must equal phy.n_leaves ($(phy.n_leaves))"))
    K_B = size(Λ_B, 2)
    Λ_B64 = Matrix{Float64}(Λ_B)
    Λφ = Vector{Float64}(σ_phy)
    α = n * float(σ²_phy)

    # Build the augmented saddle-point solver (shares its factorisation with
    # what `solve_AnB` uses; we reuse the constructed pieces directly).
    # `chol_cache` (S7 item 4), when passed, lets `chol_Q_eff` reuse the
    # CHOLMOD symbolic analysis across the EM iterations of one fit.
    s = build_AnB_sparse(Λ_B, σ_eps, Λφ, phy, n; σ²_phy = σ²_phy, chol_cache = chol_cache)
    nb       = s.n_block
    leaf_pos = s.leaf_pos
    d_inv    = s.d_inv
    DinvΛB   = s.DinvΛB
    chol_cap = s.chol_cap

    # `β · Λ_φ` (K_B × p), row k = (Λ_B' · A⁻¹ · diag(Λ_φ))[k, :]. Using the
    # Woodbury form A⁻¹ = D⁻¹ − DinvΛB · cap⁻¹ · DinvΛB':
    #   β · diag(Λ_φ) = (Λ_B' D⁻¹ − Λ_B' DinvΛB cap⁻¹ DinvΛB') · diag(Λ_φ)
    βΛφ = (Λ_B64' .* reshape(d_inv .* Λφ, 1, p)) .-
          (Λ_B64' * DinvΛB) * (chol_cap \ (DinvΛB' .* reshape(Λφ, 1, p)))

    # m and A⁻¹ m (a p-vector).
    m = vec(sum(Matrix{Float64}(y), dims = 2)) ./ n
    Ainv_m = _Ainv(s, m)
    Λφ_Ainv_m = Λφ .* Ainv_m

    # μ_φ = V_φ · (n Λ_φ A⁻¹ m). V_φ = σ²_phy · S · M_sad⁻¹ · S'.
    # Apply: M_sad⁻¹ lifted to nb-space at leaves, then restrict to leaves.
    rhs_aug = zeros(Float64, nb)
    @inbounds for t in 1:p
        rhs_aug[leaf_pos[t]] = n * Λφ_Ainv_m[t]
    end
    # M_sad⁻¹ rhs_aug via Woodbury:
    ξ0 = s.chol_Q_eff \ rhs_aug
    ξ  = ξ0 .+ α .* (s.chol_Q_eff \ (s.G * (s.chol_S_K \ (s.G' * ξ0))))
    μ_φ = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        μ_φ[t] = σ²_phy * ξ[leaf_pos[t]]
    end

    # β · Λ_φ · V_φ : K_B × p. Row k = (β · Λ_φ)[k, :] · V_φ. We have
    # (β · Λ_φ)' which is p × K_B; multiplying V_φ from the left = applying
    # σ²_phy · S · M_sad⁻¹ · S' to each of those K_B p-vectors. K_B solves.
    βΛφVφ = Matrix{Float64}(undef, K_B, p)
    @inbounds for k in 1:K_B
        rhs = zeros(Float64, nb)
        for t in 1:p
            rhs[leaf_pos[t]] = βΛφ[k, t]
        end
        η0 = s.chol_Q_eff \ rhs
        η  = η0 .+ α .* (s.chol_Q_eff \ (s.G * (s.chol_S_K \ (s.G' * η0))))
        for t in 1:p
            βΛφVφ[k, t] = σ²_phy * η[leaf_pos[t]]
        end
    end

    # diag(V_φ) via Takahashi-selected inverse of Q_eff + rank-K_B Woodbury.
    # `X_G = chol_Q_eff \ G` is now stored on `s` (built once inside
    # `build_AnB_sparse`, S7 item 1) instead of being recomputed here, and the
    # former p-fold loop of individual K_B × K_B solves (`chol_S_K \ collect(xg)`
    # per leaf) is collapsed into ONE multi-RHS solve of `chol_S_K` across all
    # p leaves at once.
    Qeff_diag = takahashi_diag(s.chol_Q_eff)             # length nb
    XG_leaf = Matrix{Float64}(undef, K_B, p)             # K_B × p, leaf rows of X_G'
    @inbounds for t in 1:p
        lp = leaf_pos[t]
        for k in 1:K_B
            XG_leaf[k, t] = s.X_G[lp, k]
        end
    end
    Y_leaf = s.chol_S_K \ XG_leaf                         # K_B × p, ONE multi-RHS solve
    diag_Vφ = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        lp = leaf_pos[t]
        acc = 0.0
        for k in 1:K_B
            acc += XG_leaf[k, t] * Y_leaf[k, t]
        end
        # Q_eff⁻¹[lp, lp] is `Qeff_diag[lp]` (selected inverse diagonal).
        diag_Vφ[t] = σ²_phy * (Qeff_diag[lp] + α * acc)
    end

    # ImβΛ = I - β · Λ_B   (K_B × K_B). β = Λ_B' A⁻¹ obtained via the
    # Woodbury/capacitance factorisation already built on `s` (S7 item 1):
    # no fresh dense p × p Cholesky of A = Λ_B Λ_B' + σ²·I here (compare the
    # dense reference E-step, `_estep_dense`, which still needs its own dense
    # `cA` for other quantities and so keeps the O(p³) Cholesky).
    F_A = GLLVModels.LowRankPlusDiagChol{Float64,Matrix{Float64}}(s.d_total, s.Λ_B, s.chol_cap)
    β = (F_A \ s.Λ_B)'                                    # K_B × p (= (A⁻¹ Λ_B)' = Λ_B' A⁻¹)
    ImβΛ = I - β * Λ_B64

    # μ_z = Λ_φ · μ_φ.
    zhat = Λφ .* μ_φ

    # Eη_s = β (y_s − Λ_φ μ_φ); sumEη and Eη Eη':
    Eη    = β * (y .- reshape(zhat, p, 1))               # K_B × n
    sumEη = vec(sum(Eη, dims = 2))                        # K_B

    # S_ηη = n(I − βΛ_B) + n · βΛφVφ · βΛφ' + Eη Eη'
    S_ηη = n .* ImβΛ .+ n .* (βΛφVφ * βΛφ') .+ Eη * Eη'
    S_ηη = Symmetric((S_ηη + S_ηη') ./ 2)
    Eφ2  = diag_Vφ .+ μ_φ .^ 2                            # length p
    C_ηφ = sumEη * μ_φ' .- n .* βΛφVφ                     # K_B × p
    H_ηy = Eη * Matrix{Float64}(y)'                        # K_B × p

    return (; β, m, Eη, sumEη, S_ηη, Eφ2, C_ηφ, H_ηy, μ_φ, μ_z = zhat)
end

# Dense M-step. Per-trait WLS for (Λ_B[t,:], σ_phy[t]); σ²_eps residual trace.
#
# For trait t the latent design is u_s = (η_s, φ[t]); the joint optimum of
# (Λ_B[t,:], σ_phy[t]) is the UNCONSTRAINED solution of the (K_B+1) normal
# equations G_t θ_t = h_t. This is the exact maximiser of the Q-function over
# those coordinates, so the EM step is monotone by construction.
#
# σ_phy is left SIGNED (no abs / no projection). The dense fit
# (`fit_gaussian_gllvm`) restricts σ_phy = exp(log_σ_phy) > 0; the two agree
# when the optimum is interior to the positive orthant (all σ_phy
# comfortably > 0), which is the regime this EM targets. A hard non-negativity
# projection is intentionally NOT used: clamping σ_phy[t] to 0 creates an
# absorbing boundary that traps EM away from an interior MLE, whereas naïve
# abs() overshoots the 0 boundary and breaks monotonicity. The honest scope is
# therefore "interior optimum"; the boundary case is documented as a known
# limitation. The reported σ_phy take the global sign convention σ_phy[1] ≥ 0
# (flipping ALL signs jointly is the only φ-orientation symmetry that leaves
# every B[t,t'] = σ_phy[t] σ_phy[t'] Σ_phy[t,t'] unchanged).
function _mstep_dense(y::AbstractMatrix, ss)
    p, n = size(y)
    K_B  = size(ss.H_ηy, 1)
    Λ_B_new   = Matrix{Float64}(undef, p, K_B)
    σ_phy_new = Vector{Float64}(undef, p)

    sy2 = sum(abs2, y)                            # Σ_{t,s} y[t,s]²
    quad_fit = 0.0                                # Σ_t θ_t' h_t

    S_ηη = Matrix(ss.S_ηη)
    @inbounds for t in 1:p
        # G_t ((K_B+1)×(K_B+1)): [[S_ηη, C_t]; [C_t', n Eφ2[t]]]
        Gt = Matrix{Float64}(undef, K_B + 1, K_B + 1)
        Gt[1:K_B, 1:K_B] .= S_ηη
        Gt[1:K_B, K_B+1]  .= ss.C_ηφ[:, t]
        Gt[K_B+1, 1:K_B]  .= ss.C_ηφ[:, t]
        Gt[K_B+1, K_B+1]   = n * ss.Eφ2[t]
        # h_t: [Σ_s Eη_s y[t,s]; μ_φ[t] Σ_s y[t,s]]
        ht = Vector{Float64}(undef, K_B + 1)
        ht[1:K_B] .= ss.H_ηy[:, t]
        ht[K_B+1]  = ss.μ_φ[t] * (n * ss.m[t])
        θt = Symmetric((Gt + Gt') ./ 2) \ ht
        Λ_B_new[t, :] .= θt[1:K_B]
        σ_phy_new[t]   = θt[K_B+1]
        quad_fit += dot(θt, ht)
    end

    σ²_eps_new = max((sy2 - quad_fit) / (n * p), eps())
    return Λ_B_new, sqrt(σ²_eps_new), σ_phy_new
end

# ---------------------------------------------------------------------------
# Observed information matrix via Supplemented EM (Meng & Rubin 1991 JASA).
#
# Louis (1982) defines the observed information at the MLE as
#       I_obs(θ̂) = I_complete(θ̂) − I_missing(θ̂),
# where I_complete = E[−∂²_θ log p(y, Z; θ) | y, θ̂]  (the expected complete-
# data information) and I_missing = Var[∂_θ log p(y, Z; θ̂) | y, θ̂] (the
# variance of the complete-data score under the posterior of the latents).
# At the EM stationary point E[∂_θ log p(y, Z; θ̂) | y, θ̂] = 0, so I_obs is the
# negative Hessian of the marginal log-likelihood. Computing I_missing
# analytically for our Gaussian phylo factor model requires fourth-moment
# posterior identities (Isserlis/Wick) that are TEDIOUS; we instead use
# Meng & Rubin's (1991) Supplemented EM (SEM) identity
#       I_obs(θ̂) = (I − DM(θ̂)) · I_complete(θ̂)
# where DM(θ̂) = ∂M/∂θ |_{θ̂} is the Jacobian of the EM map M = M_step ∘ E_step
# at the MLE. The covariance is
#       V = I_obs⁻¹ = I_complete⁻¹ · (I − DM)⁻¹.
# Both I_complete and DM are evaluated via ForwardDiff:
#   * I_complete = −∂²_θ Q(θ | θ̂) at θ = θ̂, where Q is the expected complete-
#     data log-likelihood. Q is a CLOSED-FORM quadratic in (Λ_B, σ_φ) plus
#     a log+quadratic piece in log σ_ε, using the sufficient statistics
#     produced by one E-step at θ̂. Differentiation is trivial.
#   * DM = ∂_θ M(θ) at θ = θ̂. M is the analytic E-step + M-step composition
#     (`_em_map_ad`), implemented in a type-generic form so ForwardDiff Duals
#     propagate through it.
# Parameterisation: θ = [log σ_ε; vec(Λ_B); σ_φ] (length p·K_B + p + 1). For
# K_B = 1 this matches the strict-lower-triangular packing the dense fitter
# (`gaussian_nll_packed`) uses, so SEs are directly comparable to confint.
# K_B > 1 requires a QR rotation to enforce strict-upper = 0 before packing;
# this path is currently restricted to K_B = 1 (the regime the gate covers).
#
# Refs: Louis (1982) JRSSB 44:226–233 (the observed-information identity).
#       Meng & Rubin (1991) JASA 86:899–909 (Supplemented EM).
# ---------------------------------------------------------------------------

# AD-friendly composite E-step + M-step on packed θ = [log σ_ε; vec(Λ_B); σ_φ].
# Returns the updated packed θ. Type-generic in `eltype(θ)` so ForwardDiff
# Duals propagate through both the E-step linear solves and the M-step WLS.
function _em_map_ad(θ::AbstractVector, y::AbstractMatrix{<:Real},
                    Σ_phy::AbstractMatrix, p::Integer, K_B::Integer)
    T = eltype(θ)
    n = size(y, 2)
    σ_eps = exp(θ[1])
    Λ_B = reshape(view(θ, 2:(1 + p * K_B)), p, K_B)
    σ_phy = view(θ, (2 + p * K_B):(1 + p * K_B + p))

    # ---- E-step (dense; type-generic) -------------------------------------
    σ² = σ_eps^2
    A  = Λ_B * Λ_B'
    @inbounds for t in 1:p
        A[t, t] += σ²
    end
    cA = cholesky(Symmetric((A + A') ./ 2))
    β  = Λ_B' / cA                                # K_B × p   (= Λ_B' A⁻¹)

    m   = vec(sum(y, dims = 2)) ./ n              # length p

    # φ posterior: V_φ = (Σ_phy⁻¹ + n Λ_φ A⁻¹ Λ_φ)⁻¹, μ_φ = V_φ n Λ_φ A⁻¹ m.
    Ainv_Λφ = cA \ Diagonal(σ_phy)                # A⁻¹ Λ_φ  (p × p)
    Vφ_inv  = inv(Symmetric((Σ_phy + Σ_phy') ./ 2)) .+
              n .* (Diagonal(σ_phy) * Ainv_Λφ)
    cVφ     = cholesky(Symmetric((Vφ_inv + Vφ_inv') ./ 2))
    Vφ      = inv(cVφ)                            # p × p
    μ_φ     = Vφ * (n .* (σ_phy .* (cA \ m)))     # length p

    # η posterior aggregated over sites.
    ImβΛ   = I - β * Λ_B
    βΛφ    = β .* reshape(σ_phy, 1, p)
    βΛφVφ  = βΛφ * Vφ
    zhat   = σ_phy .* μ_φ
    Eη     = β * (y .- reshape(zhat, p, 1))       # K_B × n
    sumEη  = vec(sum(Eη, dims = 2))

    S_ηη = n .* ImβΛ .+ n .* (βΛφVφ * βΛφ') .+ Eη * Eη'
    S_ηη = (S_ηη + S_ηη') ./ 2
    Eφ2  = [Vφ[t, t] + μ_φ[t]^2 for t in 1:p]
    C_ηφ = sumEη * μ_φ' .- n .* βΛφVφ             # K_B × p
    H_ηy = Eη * y'                                # K_B × p

    # ---- M-step (per-trait WLS; type-generic) -----------------------------
    sy2 = sum(abs2, y)
    Λ_B_new = Matrix{T}(undef, p, K_B)
    σ_phy_new = Vector{T}(undef, p)
    quad_fit = zero(T)
    @inbounds for t in 1:p
        Gt = Matrix{T}(undef, K_B + 1, K_B + 1)
        for j in 1:K_B, i in 1:K_B
            Gt[i, j] = S_ηη[i, j]
        end
        for k in 1:K_B
            Gt[k, K_B + 1] = C_ηφ[k, t]
            Gt[K_B + 1, k] = C_ηφ[k, t]
        end
        Gt[K_B + 1, K_B + 1] = n * Eφ2[t]
        ht = Vector{T}(undef, K_B + 1)
        for k in 1:K_B
            ht[k] = H_ηy[k, t]
        end
        ht[K_B + 1] = μ_φ[t] * (n * m[t])
        θt = Symmetric((Gt + Gt') ./ 2) \ ht
        for k in 1:K_B
            Λ_B_new[t, k] = θt[k]
        end
        σ_phy_new[t] = θt[K_B + 1]
        quad_fit += dot(θt, ht)
    end
    σ²_eps_new = (sy2 - quad_fit) / (n * p)
    σ_eps_new  = sqrt(σ²_eps_new)

    # Pack: [log σ_ε; vec(Λ_B); σ_φ]
    out = Vector{T}(undef, 1 + p * K_B + p)
    out[1] = log(σ_eps_new)
    for k in 1:K_B, i in 1:p
        out[1 + (k - 1) * p + i] = Λ_B_new[i, k]
    end
    @inbounds for t in 1:p
        out[1 + p * K_B + t] = σ_phy_new[t]
    end
    return out
end

# Expected complete-data log-likelihood Q(θ | θ̂), evaluated using the
# sufficient statistics from one E-step at θ̂. Closed-form quadratic in
# (Λ_B, σ_φ); log + quadratic in log σ_ε. Differentiating −Q via ForwardDiff
# gives I_complete at θ̂. Constants in Z = (η, φ) that do not depend on θ
# are dropped — they cancel in −∇²_θ Q.
function _Q_expected_complete(θ::AbstractVector, y::AbstractMatrix{<:Real},
                              ss, p::Integer, K_B::Integer)
    T = eltype(θ)
    n = size(y, 2)
    log_σ_eps = θ[1]
    σ_eps     = exp(log_σ_eps)
    Λ_B       = reshape(view(θ, 2:(1 + p * K_B)), p, K_B)
    σ_phy     = view(θ, (2 + p * K_B):(1 + p * K_B + p))

    sy2 = sum(abs2, y)
    cross = zero(T)
    quad  = zero(T)
    @inbounds for t in 1:p
        # Linear cross: Λ_B[t,:]' H_ηy[:,t] + σ_φ[t] μ_φ[t] (n m[t])
        c = zero(T)
        for k in 1:K_B
            c += Λ_B[t, k] * ss.H_ηy[k, t]
        end
        c += σ_phy[t] * ss.μ_φ[t] * (n * ss.m[t])
        cross += c

        # Quadratic: Λ_B[t,:]' S_ηη Λ_B[t,:] + 2 σ_φ[t] Λ_B[t,:]' C_ηφ[:,t] + n σ_φ[t]² Eφ2[t]
        q = zero(T)
        for j in 1:K_B, i in 1:K_B
            q += Λ_B[t, i] * ss.S_ηη[i, j] * Λ_B[t, j]
        end
        for k in 1:K_B
            q += 2 * σ_phy[t] * Λ_B[t, k] * ss.C_ηφ[k, t]
        end
        q += n * σ_phy[t]^2 * ss.Eφ2[t]
        quad += q
    end
    # Q(θ) = -np log σ_ε - (1 / 2σ_ε²) [sy2 - 2 cross + quad] + const
    return -n * p * log_σ_eps - (sy2 - 2 * cross + quad) / (2 * σ_eps^2)
end

# ---------------------------------------------------------------------------
# Public EM driver
# ---------------------------------------------------------------------------

"""
    EMPhyloFit

Result of `em_fit_phylo`. Fields:
* `Λ_B`        – fitted site loadings (p × K_B).
* `σ_eps`      – fitted residual SD.
* `σ_phy`      – fitted per-trait phylo SDs (length p).
* `logLik`     – final marginal log-likelihood (dense closed form).
* `n_iter`     – EM iterations run.
* `converged`  – whether the log-lik increment fell below `tol`.
* `loglik_trace` – log-lik at each iteration (monotone non-decreasing).
* `blup_phy`   – ancestral-state BLUP of the phylo effect on the data scale
                 (μ_z, length p) from the LAST E-step.
* `blup_phi`   – BLUP of the unit-scale phylo latent φ (μ_φ, length p).
* `fallback_used` – `true` iff this fit was returned by the SQUAREM
                 inferior-basin safety check after it fell back to plain EM
                 from the warm start (see `em_fit_phylo_squarem`). Always
                 `false` for `em_fit_phylo` and for an unguarded SQUAREM run.
"""
struct EMPhyloFit
    Λ_B::Matrix{Float64}
    σ_eps::Float64
    σ_phy::Vector{Float64}
    logLik::Float64
    n_iter::Int
    converged::Bool
    loglik_trace::Vector{Float64}
    blup_phy::Vector{Float64}
    blup_phi::Vector{Float64}
    fallback_used::Bool
    # Default `fallback_used = false` so existing 9-argument positional
    # constructions (in this file, em_squarem.jl, and the tests) are unchanged.
    function EMPhyloFit(Λ_B, σ_eps, σ_phy, logLik, n_iter, converged,
                        loglik_trace, blup_phy, blup_phi, fallback_used = false)
        new(Λ_B, σ_eps, σ_phy, logLik, n_iter, converged,
            loglik_trace, blup_phy, blup_phi, fallback_used)
    end
end

"""
    em_fit_phylo(y, K_B, Σ_phy;
                 λ_init=nothing, σ_eps_init=nothing, σ_phy_init=nothing,
                 tol=1e-9, max_iter=1000, assert_monotone=true,
                 phy=nothing, force_dense_estep=false) -> EMPhyloFit

Gradient-free EM fit of the Gaussian phylo_unique GLLVM: `K_B` site latent
factors plus one per-trait phylogenetic random effect with covariance
`(σ_phy σ_phy') ∘ Σ_phy`. Matches `fit_gaussian_gllvm(y; K = K_B,
has_phy_unique = true, Σ_phy = Σ_phy)`.

`y` is (p, n_sites). `Σ_phy` is the fixed (p × p) tree-derived species
covariance. Warm-started from PPCA (`ppca_init`) unless `λ_init`/`σ_eps_init`
are supplied. Returns an `EMPhyloFit` including the ancestral-state BLUPs
from the final E-step.

When `assert_monotone` (default), a log-lik DECREASE beyond `1e-7` triggers an
error — a monotone non-decrease is an EM invariant, so a decrease is a bug.

E-STEP PATH. The phylo block's per-trait posterior variance `diag(V_φ)` is the
only O(p³) ingredient of the E-step. When the tree (`phy::AugmentedPhy`) is
supplied, the E-step is routed BY DEFAULT through the augmented-state sparse
path (`_estep_sparse`), which obtains `diag(V_φ)` via the Takahashi (1973) /
Erisman–Tinney (1975) selected inverse in O(p) instead of the dense
`inv(cVφ)`'s O(p³). The two paths are exact-equivalent in floating point (the
sparse path is the same algebra, just refactored to never materialise dense
p × p inverses), so the MLE and BLUPs are identical — only the per-iteration
cost differs. Set `force_dense_estep = true` to use the dense E-step even when
`phy` is supplied (the dense path's BLAS is competitive at small p, and it is
the only option when `phy` is omitted, since an `AugmentedPhy` cannot be
recovered from the dense `Σ_phy` alone). When `phy === nothing` the dense path
is always used regardless of `force_dense_estep`.
"""
function em_fit_phylo(y::AbstractMatrix, K_B::Integer, Σ_phy::AbstractMatrix;
                      λ_init = nothing, σ_eps_init = nothing,
                      σ_phy_init = nothing,
                      tol = 1e-9, max_iter = 1000, assert_monotone = true,
                      phy::Union{Nothing,GLLVModels.AugmentedPhy{Float64}} = nothing,
                      force_dense_estep::Bool = false)
    p, n = size(y)
    K_B ≥ 1 || throw(ArgumentError("K_B must be ≥ 1"))
    K_B < p || throw(ArgumentError("EM requires K_B < p; got K_B=$K_B, p=$p"))
    size(Σ_phy) == (p, p) ||
        throw(ArgumentError("Σ_phy must be p × p; got $(size(Σ_phy)) for p=$p"))
    if phy !== nothing
        phy.n_leaves == p ||
            throw(ArgumentError("phy.n_leaves ($(phy.n_leaves)) must equal p ($p)"))
    end

    yf = Matrix{Float64}(y)
    # Sparse Takahashi E-step is the DEFAULT whenever the tree is available;
    # fall back to the dense E-step when `phy` is omitted (no AugmentedPhy to
    # build from the dense Σ_phy) or when `force_dense_estep` is requested.
    use_sparse = phy !== nothing && !force_dense_estep
    # S7 item 4: ONE cache, shared across every E-step call of this fit, so
    # `chol_Q_eff`'s CHOLMOD symbolic analysis is done once (iteration 1) and
    # reused via `cholesky!` thereafter — the sparsity pattern is fit-invariant
    # (tree topology + K_B fixed; only σ_phy/σ_eps values change per M-step).
    chol_cache = use_sparse ? Ref{Any}(nothing) : nothing
    estep = if use_sparse
        (LB, σe, σp) -> _estep_sparse(yf, LB, σe, σp, phy; σ²_phy = 1.0, chol_cache = chol_cache)
    else
        (LB, σe, σp) -> _estep_dense(yf, LB, σe, σp, Σ_phy)
    end
    # S7 item 2: the per-iteration monotonicity CHECK is numerically the same
    # closed-form Gaussian marginal log-lik regardless of which E-step ran,
    # but `gaussian_marginal_loglik`'s J3 phylo path (likelihood.jl:212-244)
    # forms TWO dense p × p Choleskys every call. When the sparse E-step is
    # selected, route the check through the O(p) sparse-precision path
    # instead (`gaussian_marginal_loglik_sparse_phy`, likelihood_sparse_phy.jl)
    # — numerically equivalent, see that function's docstring.
    loglik_check = if use_sparse
        (LB, σe, σp) -> GLLVModels.gaussian_marginal_loglik_sparse_phy(yf, LB, σe;
                                            σ_phy = σp, phy = phy, σ²_phy = 1.0)
    else
        (LB, σe, σp) -> GLLVModels.gaussian_marginal_loglik(yf, LB, σe;
                                            σ_phy = σp, Σ_phy = Σ_phy)
    end

    # ----- Warm start (PPCA for Λ_B, σ_eps; small phylo SD to start) -----
    if λ_init === nothing || σ_eps_init === nothing
        Λ0, σ0 = GLLVModels.ppca_init(yf, K_B)
        Λ_B   = λ_init === nothing ? Matrix{Float64}(Λ0) : Matrix{Float64}(λ_init)
        σ_eps = σ_eps_init === nothing ? float(σ0) : float(σ_eps_init)
    else
        Λ_B   = Matrix{Float64}(λ_init)
        σ_eps = float(σ_eps_init)
    end
    σ_phy = if σ_phy_init === nothing
        # Start the phylo SD from the marginal scale of the data.
        fill(0.1 * sqrt(mean(abs2, yf)), p)
    else
        Vector{Float64}(σ_phy_init)
    end

    loglik_trace = Float64[]
    loglik_prev  = -Inf
    converged    = false
    iters_run    = 0
    local blup_phy = zeros(Float64, p)
    local blup_phi = zeros(Float64, p)

    for iter in 1:max_iter
        iters_run = iter
        # Marginal log-lik at the CURRENT parameters (closed form — sparse
        # when the sparse E-step is selected, dense otherwise; see
        # `loglik_check` above), i.e. at the output of the previous M-step
        # ⇒ sequence is monotone.
        ll = loglik_check(Λ_B, σ_eps, σ_phy)
        push!(loglik_trace, ll)

        if iter > 1
            inc = ll - loglik_prev
            if assert_monotone && inc < -1e-7
                error("EM log-lik decreased by $(abs(inc)) at iter $iter " *
                      "(was $loglik_prev, now $ll) — EM monotonicity violated.")
            end
            if abs(inc) < tol
                converged = true
                # E-step once more to refresh BLUPs at the converged params.
                ss = estep(Λ_B, σ_eps, σ_phy)
                blup_phy = copy(ss.μ_z); blup_phi = copy(ss.μ_φ)
                break
            end
        end
        loglik_prev = ll

        ss = estep(Λ_B, σ_eps, σ_phy)
        blup_phy = copy(ss.μ_z); blup_phi = copy(ss.μ_φ)
        Λ_B, σ_eps, σ_phy = _mstep_dense(yf, ss)
    end

    ll_final = loglik_check(Λ_B, σ_eps, σ_phy)
    if !isempty(loglik_trace) && ll_final > loglik_trace[end]
        push!(loglik_trace, ll_final)
    end

    # Global φ-orientation convention: flipping ALL σ_phy signs jointly leaves
    # every B[t,t'] = σ_phy[t] σ_phy[t'] Σ_phy[t,t'] unchanged (and flips μ_φ,
    # leaving the data-scale BLUP μ_z = diag(σ_phy) μ_φ invariant). Anchor the
    # sign so the dominant-magnitude trait's σ_phy is ≥ 0, matching the dense
    # fit's σ_phy = exp(log_σ_phy) > 0 convention for interior optima.
    t_anchor = argmax(abs.(σ_phy))
    if σ_phy[t_anchor] < 0
        σ_phy = -σ_phy
        blup_phi = -blup_phi          # μ_z = diag(σ_phy) μ_φ unchanged
    end

    return EMPhyloFit(Λ_B, σ_eps, σ_phy, ll_final, iters_run, converged,
                      loglik_trace, blup_phy, blup_phi)
end

"""
    em_observed_information(emf, y, Σ_phy) -> NamedTuple

Compute the observed information matrix at the EM MLE via the Supplemented
EM identity (Meng & Rubin 1991 JASA), which evaluates Louis's (1982)
`I_obs = I_complete − I_missing` from the EM map's rate matrix and the
expected complete-data information:

        I_obs(θ̂) = (I − DM(θ̂)) · I_complete(θ̂) ,
        V(θ̂)    = I_obs⁻¹ = I_complete⁻¹ · (I − DM)⁻¹.

`I_complete` is the negative Hessian of the expected complete-data log-
likelihood `Q(θ | θ̂)` at `θ̂`; `DM` is the Jacobian of one EM step. Both
are computed via ForwardDiff on type-generic implementations of `Q` and
the EM map.

Parameterisation: `θ = [log σ_ε; vec(Λ_B); σ_φ]` (length `1 + p·K_B + p`).
For `K_B = 1` this matches the strict-lower-triangular packing that
`gaussian_nll_packed` / `confint` uses, so the standard errors returned
here are directly comparable to `confint(fit; y = …)`.

CURRENTLY RESTRICTED TO K_B = 1. For K_B > 1 the strict-upper triangle of
Λ_B is not enforced by EM (the M-step is rotation-equivariant); SEM on the
raw `vec(Λ_B)` parameterisation gives a singular `I_obs` along the
rotation directions. A QR-rotation onto the lower-triangular orbit before
packing would generalise this; postponed.

Returns a NamedTuple with fields:
  * `info::Matrix`         — observed information matrix `I_obs`
  * `cov::Matrix`          — asymptotic covariance `I_obs⁻¹`
  * `se::Vector`           — `sqrt.(diag(cov))` on the packed scale
  * `se_raw::Vector`       — SEs back-transformed to the raw scale via the
                             delta method (σ_ε via exp, others identity)
  * `term::Vector{String}` — parameter names matching `confint(fit).term`
                             when fit is a dense fit on the same model
  * `pd::Bool`             — whether `I_obs` is positive-definite

Refs: Louis (1982) JRSSB 44:226–233; Meng & Rubin (1991) JASA 86:899–909.
"""
function em_observed_information(emf::EMPhyloFit, y::AbstractMatrix,
                                 Σ_phy::AbstractMatrix)
    p, n = size(y)
    K_B  = size(emf.Λ_B, 2)
    K_B == 1 || throw(ArgumentError(
        "em_observed_information currently supports K_B = 1 only; got K_B=$K_B"))
    size(Σ_phy) == (p, p) ||
        throw(ArgumentError("Σ_phy must be p × p; got $(size(Σ_phy))"))

    yf = Matrix{Float64}(y)
    Σf = Matrix{Float64}(Σ_phy)

    # Pack the EM MLE: θ̂ = [log σ̂_ε; vec(Λ̂_B); σ̂_φ].
    θ̂ = vcat(log(emf.σ_eps), vec(emf.Λ_B), copy(emf.σ_phy))

    # Sufficient statistics from one E-step at θ̂ (drives Q).
    ss = _estep_dense(yf, emf.Λ_B, emf.σ_eps, emf.σ_phy, Σf)

    # I_complete = −∂² Q(θ | θ̂)|_{θ̂}  (Hessian of −Q via ForwardDiff).
    I_complete = ForwardDiff.hessian(
        θ -> -_Q_expected_complete(θ, yf, ss, p, K_B), θ̂)
    I_complete = (I_complete + I_complete') ./ 2

    # DM = ∂_θ M(θ)|_{θ̂}  (Jacobian of one EM step via ForwardDiff).
    DM = ForwardDiff.jacobian(
        θ -> _em_map_ad(θ, yf, Σf, p, K_B), θ̂)

    # I_obs = I_complete · (I − DM) = (I − DMᵀ) · I_complete  (Meng & Rubin
    # 1991, eq. 2.2.4-6). Both forms are symmetric because I_m = I_c·DM is
    # symmetric (the fraction-of-missing-information identity I_c⁻¹ I_m = DM).
    # We use the symmetric average for round-off; the constituent matrices are
    # symmetric in exact arithmetic.
    Ipar = size(DM, 1)
    A1   = I_complete * (Matrix{Float64}(I, Ipar, Ipar) - DM)
    I_obs = (A1 + A1') ./ 2                        # symmetrise round-off

    pd = true
    cov_ = try
        inv(Symmetric(I_obs))
    catch
        pd = false
        fill(NaN, Ipar, Ipar)
    end
    diag_cov = diag(cov_)
    if any(!isfinite(v) || v ≤ 0 for v in diag_cov)
        pd = false
    end

    se = [v > 0 ? sqrt(v) : NaN for v in diag_cov]

    # Raw-scale SEs (delta method): σ_ε = exp(log σ_ε) ⇒ SE_σ_ε = σ_ε · SE_log σ_ε.
    se_raw = copy(se)
    se_raw[1] = isfinite(se[1]) ? emf.σ_eps * se[1] : NaN

    # Term names (mirror confint's convention for the phylo_unique-only case).
    terms = String["sigma_eps"]
    for i in 1:p
        push!(terms, "Lambda_B[$i,1]")
    end
    for t in 1:p
        push!(terms, "sigma_phy[$t]")
    end

    return (info = I_obs, cov = cov_, se = se, se_raw = se_raw,
            term = terms, pd = pd)
end

const fit_em_phylo = em_fit_phylo
