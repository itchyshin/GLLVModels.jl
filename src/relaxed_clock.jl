# Relaxed-clock per-branch evolution-rate model on the edge-node incidence
# substrate.
#
# ===========================================================================
# THE QUESTION (personal communication via the maintainer)
# ===========================================================================
# The edge-incidence precision Q = B · W · Bᵀ writes the phylogenetic
# Brownian-motion precision as a weighted graph Laplacian of the tree with
# per-edge weights w_e on the diagonal of W:
#
#       W = diag(w_e),   w_e = 1 / (σ²_e · ℓ_e)
#
# where ℓ_e is the branch length and σ²_e is a PER-BRANCH evolution rate. In
# the single-rate model σ²_e ≡ σ²_phy for every edge. the conjecture:
# because the per-branch rates ARE the diagonal of W, the edge-incidence form
# is the natural substrate for estimating a SEPARATE rate per branch — a
# relaxed molecular clock — rather than only ancestral states.
#
# THE HONEST CAVEAT. A binary tree with p leaves has 2p − 2 branches but only
# p leaf observations. Each branch increment δ_e = x_child − x_parent is a
# SINGLE Gaussian draw N(0, σ²_e ℓ_e); from one trait realisation you cannot
# freely identify 2p − 2 variances from p numbers. Per-branch rates become
# estimable only as SHRINKAGE estimates under a hierarchical / relaxed-clock
# prior that ties them together, OR when several i.i.d. trait realisations
# share the same per-branch rates (so each branch increment is observed n
# times). This file implements BOTH the prior and the multi-replicate design
# and tests, empirically, whether the rates are recoverable.
#
# ===========================================================================
# THE MODEL (augmented-node Gaussian state space = the edge-incidence Q)
# ===========================================================================
# Latent node states x ∈ ℝ^{2p−1} (all nodes; root pinned). Per edge e
# (parent → child) the BM increment is
#
#       δ_e = x_child − x_parent ~ N(0, v_e),    v_e = σ²_e · ℓ_e .
#
# The joint log-density of the increments is exactly the edge-incidence
# quadratic form: Σ_e δ_e² / v_e = xᵀ (B W Bᵀ) x = xᵀ Q x, with the same
# B as `EdgePhy` and W = diag(1/v_e). Leaf observations (n i.i.d. replicates
# r = 1…n sharing the SAME per-branch rates):
#
#       y[t, r] = x_leaf(t), r  +  ε[t, r],   ε ~ N(0, σ²_eps) .
#
# HIERARCHICAL (relaxed-clock) PRIOR on the log-rates:
#
#       ρ_e := log σ²_e  ~  N(μ, τ²)   i.i.d. across edges.
#
# Free hyperparameters: (μ, τ², σ²_eps). Per-branch rates σ²_e are the random
# effects; we report their SHRINKAGE estimates (posterior modes).
#
# ===========================================================================
# THE ESTIMATOR (empirical Bayes via EM; cleanest tractable route)
# ===========================================================================
# Hard to integrate the node states AND the rates jointly in closed form, so
# we alternate (a generalised EM / empirical Bayes):
#
#   E-step (node states | rates).  With per-edge variances v_e fixed, the
#   node-state posterior given the n leaf replicates is Gaussian with
#   precision (per replicate)  Q(v) + Sᵀ diag(1/σ²_eps) S  on the non-root
#   nodes, where S selects leaves and the root is pinned to its posterior
#   mean. We need only the per-edge increment second moments
#       s_e = Σ_r E[ δ_{e,r}² | y ]  =  Σ_r ( (Bᵀ μ_r)_e² )  +  n · Var_post(δ_e)
#   (the conditional mean increment squared, summed over replicates, plus n×
#   the posterior increment variance). These are the sufficient statistics
#   the rate update consumes.
#
#   M-step part 1 (rates | node states, hyperparameters).  Given s_e and the
#   per-edge df = n, the per-branch log-rate ρ_e = log σ²_e has the penalised
#   1-D objective (drop constants)
#       g(ρ_e) = −(n/2)(ρ_e + log ℓ_e) − s_e/(2 ℓ_e) e^{−ρ_e}
#                − (ρ_e − μ)² / (2 τ²)
#   — a strictly concave function of ρ_e (the data term is concave, the prior
#   term is concave), solved to its unique mode by a few Newton steps. This IS
#   the shrinkage estimate: the data pulls ρ_e toward log(s_e/(n ℓ_e)), the
#   prior pulls it toward μ; the balance is set by n vs 1/τ².
#
#   M-step part 2 (hyperparameters).  μ, τ² are the (Laplace-)EB updates from
#   the rate posterior modes ρ̂_e and curvatures c_e = −g''(ρ̂_e):
#       μ  = mean_e ρ̂_e ;   τ² = mean_e [ (ρ̂_e − μ)² + 1/c_e ]
#   (the 1/c_e term is the Laplace posterior-variance correction — without it
#   τ² is biased downward). σ²_eps is the residual-variance update from the
#   leaf reconstruction.
#
# Convergence is monitored by the penalised marginal objective. This is the
# standard empirical-Bayes / Laplace recipe (e.g. the variance-component EB of
# Smyth 2004, Stat Appl Genet Mol Biol, applied per branch), specialised to
# the tree via the edge-incidence Q for the E-step.
#
# Every linear-algebra primitive is the edge-incidence Q = B W Bᵀ and its
# root-conditioned solve; no dense p × p covariance is formed beyond the small
# conditioned-node system, and the rate update is closed-form-per-edge.
#
# This file is self-contained and does NOT modify any existing src file.

using LinearAlgebra
using SparseArrays
using Random
using Statistics

# ---------------------------------------------------------------------------
# 0. Per-branch rate plumbing on an EdgePhy.
# ---------------------------------------------------------------------------

"""
    edge_W_diag(phy::EdgePhy, σ²_e::AbstractVector) -> Vector

Per-edge diagonal of `W` for the relaxed-clock model: `W[e,e] = 1/(σ²_e · ℓ_e)`.
`σ²_e` is the length-`n_edges` vector of per-branch rates (the diagonal that
the conjecture targets). The single-rate model is the special case
`σ²_e ≡ σ²_phy`.
"""
function edge_W_diag(phy::EdgePhy, σ²_e::AbstractVector)
    length(σ²_e) == phy.n_edges ||
        throw(ArgumentError("σ²_e length $(length(σ²_e)) must equal n_edges $(phy.n_edges)"))
    return 1.0 ./ (σ²_e .* phy.branch_lengths)
end

"""
    Q_perbranch(phy::EdgePhy, σ²_e::AbstractVector) -> SparseMatrixCSC

Materialise the per-branch precision `Q = B · diag(w_e) · Bᵀ` over all nodes
with `w_e = 1/(σ²_e ℓ_e)`. Sparse, ≈ 8p non-zeros (same sparsity as the
single-rate Q; only the edge weights change). Used for the conditioned-node
posterior solve in the E-step and for cross-checks.
"""
function Q_perbranch(phy::EdgePhy, σ²_e::AbstractVector)
    w = edge_W_diag(phy, σ²_e)
    Bw = phy.B * spdiagm(0 => w)            # (n_nodes × n_edges)
    Q = Bw * phy.B'
    return Q
end

# ---------------------------------------------------------------------------
# 1. DATA-GENERATING MODEL: simulate variable-rate Brownian motion.
# ---------------------------------------------------------------------------

"""
    simulate_relaxed_bm(phy, σ²_e_true, σ²_eps, n_rep; rng, root_value=0.0)
        -> (y, x_nodes, δ_true)

Simulate `n_rep` i.i.d. Brownian-motion trait realisations on `phy` with KNOWN
per-branch rates `σ²_e_true` (length `n_edges`) and i.i.d. observation noise
`N(0, σ²_eps)` at the leaves. The per-branch rates are the truth we try to
recover.

Generative recipe (the edge-incidence model, simulated forward):
  * root state = `root_value` (a constant offset; BM is identified only up to
    this, matching `Q · 1 = 0`);
  * for each edge e (pre-order, parent before child) draw the increment
        δ_e ~ N(0, σ²_e_true[e] · ℓ_e),  x_child = x_parent + δ_e;
  * leaf observation y[t, r] = x_leaf(t) + ε,  ε ~ N(0, σ²_eps).

Returns
  * `y`       :: p × n_rep leaf observations;
  * `x_nodes` :: n_nodes × n_rep latent node states (truth, incl. internals);
  * `δ_true`  :: n_edges × n_rep realised increments (truth).
"""
function simulate_relaxed_bm(phy::EdgePhy, σ²_e_true::AbstractVector,
                             σ²_eps::Real, n_rep::Integer;
                             rng::AbstractRNG = Random.default_rng(),
                             root_value::Real = 0.0)
    length(σ²_e_true) == phy.n_edges ||
        throw(ArgumentError("σ²_e_true length must equal n_edges $(phy.n_edges)"))
    n_rep ≥ 1 || throw(ArgumentError("n_rep must be ≥ 1"))
    p = phy.n_leaves
    x_nodes = zeros(Float64, phy.n_nodes, n_rep)
    δ_true  = zeros(Float64, phy.n_edges, n_rep)
    sd_e = sqrt.(σ²_e_true .* phy.branch_lengths)        # per-edge increment SD

    # Pre-order traversal: parent visited before children (root first).
    stack = [phy.root_index]
    @inbounds while !isempty(stack)
        u = pop!(stack)
        for v in phy.node_children[u]
            e = phy.node_edge[v]
            for r in 1:n_rep
                δ = sd_e[e] * randn(rng)
                δ_true[e, r] = δ
                x_nodes[v, r] = x_nodes[u, r] + δ
            end
            push!(stack, v)
        end
    end
    # Root offset (added uniformly; cancels in increments).
    if root_value != 0
        x_nodes .+= root_value
    end

    y = Matrix{Float64}(undef, p, n_rep)
    σ_eps = sqrt(σ²_eps)
    @inbounds for r in 1:n_rep, t in 1:p
        y[t, r] = x_nodes[phy.leaf_indices[t], r] + σ_eps * randn(rng)
    end
    return y, x_nodes, δ_true
end

# ---------------------------------------------------------------------------
# 2. E-STEP: node-state posterior given per-branch rates and σ²_eps.
# ---------------------------------------------------------------------------
# We pin the root (the BM offset is unidentified) and solve the Gaussian
# posterior over the remaining n_nodes − 1 nodes. The prior precision over
# all nodes is Q(v) = B diag(1/v) Bᵀ; the likelihood adds 1/σ²_eps to each
# leaf row. Per replicate r the posterior is
#       (Q_cond + Sᵀ (1/σ²_eps) S) μ_r = Sᵀ (1/σ²_eps) y_r        (root row dropped)
# and the per-edge increment second moments aggregate over replicates.
#
# We solve directly on the conditioned node system (sparse Cholesky). This is
# O(p) per solve on a tree; we batch all replicates as multiple right-hand
# sides. NOTE: the conditioned system uses node indices with the root removed,
# matching `solve_Q`'s convention.

struct _CondMap
    keep::Vector{Int}            # node indices retained (root dropped)
    pos::Vector{Int}             # pos[node] = row in conditioned system, 0 if dropped
    leaf_rows::Vector{Int}       # conditioned-row of each leaf t (length p)
end

function _cond_map(phy::EdgePhy)
    keep = filter(i -> i != phy.root_index, 1:phy.n_nodes)
    pos = zeros(Int, phy.n_nodes)
    for (r, node) in enumerate(keep)
        pos[node] = r
    end
    leaf_rows = [pos[phy.leaf_indices[t]] for t in 1:phy.n_leaves]
    return _CondMap(keep, pos, leaf_rows)
end

"""
    estep_edge_moments(phy, σ²_e, σ²_eps, y, cm) -> (s_e, recon, μ_nodes)

E-step on the edge-incidence substrate. Given per-branch rates `σ²_e`, noise
`σ²_eps`, and leaf data `y` (p × n_rep), return:

  * `s_e`     :: length-`n_edges` per-edge increment second moments
                 `s_e[e] = Σ_r E[δ_{e,r}² | y]`  (the rate-update sufficient
                 statistic): `Σ_r (conditional-mean δ)² + n_rep · Var_post(δ_e)`.
  * `recon`   :: Σ over (t, r) of `E[(y[t,r] − x_leaf)²]` — the σ²_eps update
                 residual (reconstruction sum of squares incl. posterior var).
  * `μ_nodes` :: n_nodes × n_rep posterior mean node states (root pinned to its
                 posterior mean; reported for diagnostics / BLUP of states).

All operations go through Q = B W Bᵀ (sparse). The per-edge posterior
variance Var_post(δ_e) = (Bᵀ P⁻¹ B)[e,e] where P is the conditioned posterior
precision; we obtain the needed diagonal via a selected solve (dense inverse
of the conditioned precision — fine at the prototype scale; the O(p) Takahashi
selected inverse is the production swap, identical to the §5.4 note).
"""
function estep_edge_moments(phy::EdgePhy, σ²_e::AbstractVector, σ²_eps::Real,
                            y::AbstractMatrix, cm::_CondMap)
    p, n_rep = size(y)
    nc = length(cm.keep)
    w = edge_W_diag(phy, σ²_e)                       # 1/(σ²_e ℓ_e)

    # Conditioned prior precision Q_cond = (B W Bᵀ)[keep, keep].
    Bw = phy.B * spdiagm(0 => w)
    Q_full = Bw * phy.B'
    Q_cond = Q_full[cm.keep, cm.keep]

    # Posterior precision P = Q_cond + Sᵀ (1/σ²_eps) S  (leaf diagonal bump).
    inv_eps = 1.0 / σ²_eps
    P = Matrix(Q_cond)
    @inbounds for t in 1:p
        P[cm.leaf_rows[t], cm.leaf_rows[t]] += inv_eps
    end
    Psym = Symmetric((P + P') ./ 2)
    # Use shift / positive-definite check with regularisation fallback for near-singular edge cases
    cP = try
        cholesky(Psym)
    catch
        cholesky(Psym + 1e-12 * I)
    end

    # RHS: Sᵀ (1/σ²_eps) y  (concentrated at leaf rows), all replicates batched.
    RHS = zeros(Float64, nc, n_rep)
    @inbounds for r in 1:n_rep, t in 1:p
        RHS[cm.leaf_rows[t], r] = inv_eps * y[t, r]
    end
    Mu_cond = cP \ RHS                               # nc × n_rep posterior means

    # Lift to full node space (root row = its posterior mean = 0 by pinning;
    # the offset is unidentified, so root stays at 0 — consistent with DGM up
    # to the common offset, which the increments ignore).
    μ_nodes = zeros(Float64, phy.n_nodes, n_rep)
    @inbounds for r in 1:n_rep
        for (rr, node) in enumerate(cm.keep)
            μ_nodes[node, r] = Mu_cond[rr, r]
        end
    end

    # Conditional-mean increments δ̄_{e,r} = (Bᵀ μ)_e and their second moments.
    Δbar = phy.B' * μ_nodes                          # n_edges × n_rep
    s_e = vec(sum(Δbar .^ 2; dims = 2))              # Σ_r δ̄²

    # Posterior increment variance per edge: Var(δ_e) = (Bᵀ P⁻¹ B)[e,e]
    # restricted to conditioned coordinates. B maps nodes→edges; the root row
    # of B contributes 0 to P⁻¹ (root pinned ⇒ zero variance), so we use the
    # conditioned rows of B.
    Bc = phy.B[cm.keep, :]                           # nc × n_edges
    Pinv = inv(cP)                                   # nc × nc (prototype dense)
    # diag(Bcᵀ Pinv Bc) without forming the full product:
    PinvBc = Pinv * Bc                               # nc × n_edges
    var_e = vec(sum(Bc .* PinvBc; dims = 1))         # length n_edges
    s_e .+= n_rep .* var_e

    # Reconstruction residual for σ²_eps: Σ_{t,r} E[(y − x_leaf)²]
    #   = Σ (y − μ_leaf)² + n_rep Σ_t Var_post(x_leaf_t).
    recon = 0.0
    @inbounds for r in 1:n_rep, t in 1:p
        d = y[t, r] - μ_nodes[phy.leaf_indices[t], r]
        recon += d * d
    end
    @inbounds for t in 1:p
        recon += n_rep * Pinv[cm.leaf_rows[t], cm.leaf_rows[t]]
    end

    return s_e, recon, μ_nodes
end

# ---------------------------------------------------------------------------
# 3. M-STEP part 1: per-branch log-rate shrinkage modes (penalised 1-D).
# ---------------------------------------------------------------------------
# Maximise, per edge, over ρ = log σ²_e:
#   g(ρ) = −(n/2)(ρ + log ℓ) − (s/(2ℓ)) e^{−ρ} − (ρ − μ)²/(2τ²)
# g'(ρ)  = −n/2 + (s/(2ℓ)) e^{−ρ} − (ρ − μ)/τ²
# g''(ρ) = −(s/(2ℓ)) e^{−ρ} − 1/τ²   (< 0 ⇒ strictly concave, unique mode)
# Newton from the data-only MLE log(s/(nℓ)) clamped; converges in a few steps.

"""
    shrink_logrates(s_e, df, ℓ, μ, τ²) -> (ρ̂, curv)

Per-edge shrinkage posterior modes `ρ̂_e = log σ̂²_e` and curvatures
`curv_e = −g''(ρ̂_e)` under the relaxed-clock prior `ρ_e ~ N(μ, τ²)`, given
per-edge sufficient statistics `s_e` (Σ_r E[δ²]) with `df` degrees of freedom
each and branch lengths `ℓ`. Closed-form Newton (strictly concave objective).
"""
function shrink_logrates(s_e::AbstractVector, df::Real, ℓ::AbstractVector,
                         μ::Real, τ²::Real)
    n_e = length(s_e)
    ρ̂ = Vector{Float64}(undef, n_e)
    curv = Vector{Float64}(undef, n_e)
    inv_τ² = 1.0 / τ²
    @inbounds for e in 1:n_e
        a = s_e[e] / (2 * ℓ[e])                       # = (s/(2ℓ))
        # data-only MLE: −n/2 + a e^{−ρ} = 0 ⇒ ρ = log(2a/n) = log(s/(nℓ)).
        ρ = a > 0 ? log(2a / df) : μ
        ρ = clamp(ρ, μ - 50, μ + 50)
        for _ in 1:100
            ex = exp(-ρ)
            g1 = -0.5 * df + a * ex - (ρ - μ) * inv_τ²
            g2 = -a * ex - inv_τ²                      # < 0
            step = g1 / g2
            ρ -= step
            abs(step) < 1e-12 && break
        end
        ρ̂[e] = ρ
        curv[e] = a * exp(-ρ) + inv_τ²                 # −g'' > 0
    end
    return ρ̂, curv
end

# ---------------------------------------------------------------------------
# 4. PUBLIC DRIVER: empirical-Bayes relaxed-clock fit.
# ---------------------------------------------------------------------------

"""
    RelaxedClockFit

Result of `fit_relaxed_clock`. Fields:
  * `σ²_e`       – per-branch rate shrinkage estimates (length n_edges).
  * `logrates`   – their logs ρ̂_e (the random effects on the prior scale).
  * `μ`, `τ²`    – fitted hyperparameters of `log σ²_e ~ N(μ, τ²)`.
  * `σ²_eps`     – fitted observation-noise variance.
  * `n_iter`     – EB iterations run.
  * `converged`  – whether the objective increment fell below `tol` AND the fit
                   did not exit on a boundary (see `tau2_boundary`,
                   `sigma_eps_boundary`).
  * `obj_trace`  – penalised objective per iteration.
  * `μ_nodes`    – posterior mean node states from the final E-step.
  * `tau2_boundary` – `true` when the free-τ² update ended clamped at
                   `τ²_floor` (rate dispersion within sampling noise of a single
                   rate). A pinned `fix_τ²` is never a boundary.
  * `sigma_eps_boundary` – `true` when the free-σ²_eps update ended clamped at
                   its data-relative floor (`1e-8 * mean(y.^2)`). A pinned
                   `fix_σ²_eps` is never a boundary.

A fit with either boundary flag set leaves `converged = false`: a clamp is not an
interior optimum.
"""
struct RelaxedClockFit
    σ²_e::Vector{Float64}
    logrates::Vector{Float64}
    μ::Float64
    τ²::Float64
    σ²_eps::Float64
    n_iter::Int
    converged::Bool
    obj_trace::Vector{Float64}
    μ_nodes::Matrix{Float64}
    tau2_boundary::Bool
    sigma_eps_boundary::Bool
end

# Pre-#876 positional constructor: boundary flags default to `false`.
RelaxedClockFit(σ²_e, logrates, μ, τ², σ²_eps, n_iter, converged, obj_trace,
                μ_nodes) =
    RelaxedClockFit(σ²_e, logrates, μ, τ², σ²_eps, n_iter, converged, obj_trace,
                    μ_nodes, false, false)

function Base.show(io::IO, f::RelaxedClockFit)
    print(io, "RelaxedClockFit(n_edges=", length(f.σ²_e), ", μ=", round(f.μ; digits = 4),
          ", τ²=", round(f.τ²; sigdigits = 4), ", σ²_eps=", round(f.σ²_eps; sigdigits = 4),
          ", n_iter=", f.n_iter, ", converged=", f.converged,
          ", tau2_boundary=", f.tau2_boundary,
          ", sigma_eps_boundary=", f.sigma_eps_boundary, ")")
end

# Data-relative σ²_eps floor: a fixed fraction of the response scale, so that
# rescaling `y` cannot change whether the residual sits on the floor. Zero or
# non-finite scale (constant-zero y) falls back to the absolute 1e-8.
function _relaxed_sigma_eps_floor(y::AbstractMatrix)
    sc = mean(abs2, y)
    return (isfinite(sc) && sc > 0) ? 1e-8 * sc : 1e-8
end

# Penalised EB objective (E-step lower bound surrogate): the per-edge data
# term at the current rates + the log-normal prior + the σ²_eps term. Used
# only to monitor monotonicity of the coordinate ascent, not for inference.
function _relaxed_obj(phy, σ²_e, σ²_eps, s_e, recon, μ, τ², n_rep, df)
    ℓ = phy.branch_lengths
    ρ = log.(σ²_e)
    val = 0.0
    @inbounds for e in 1:phy.n_edges
        v_e = σ²_e[e] * ℓ[e]
        val += -0.5 * df * log(v_e) - 0.5 * s_e[e] / v_e          # data
        val += -0.5 * log(2π * τ²) - 0.5 * (ρ[e] - μ)^2 / τ²      # prior
    end
    p = phy.n_leaves
    val += -0.5 * (n_rep * p) * log(σ²_eps) - 0.5 * recon / σ²_eps  # noise
    return val
end

"""
    fit_relaxed_clock(phy, y; max_iter=200, tol=1e-7,
                      τ²_floor=1e-4, fix_τ²=nothing, μ_init=nothing) -> RelaxedClockFit

Empirical-Bayes fit of the relaxed-clock per-branch rate model on the edge-
incidence substrate. `y` is p × n_rep leaf data (n_rep i.i.d. BM realisations
sharing the per-branch rates). Estimates the hyperparameters (μ, τ², σ²_eps)
and returns shrinkage estimates of every per-branch rate σ²_e.

`fix_τ²` pins the prior variance (sweep this to study how recovery degrades
as the prior tightens, τ² → 0, vs. relaxes, τ² → ∞). `fix_σ²_eps` pins the
observation-noise variance (useful when measurement error is known/estimated
separately — at small `n_rep` the per-branch BM variance and σ²_eps are weakly
separable, so pinning σ²_eps isolates the rate-recovery question). `τ²_floor`
guards the free-τ² path; the free-σ²_eps path is floored at `1e-8 * mean(y.^2)`.
If either free parameter exits clamped at its floor, the corresponding
`tau2_boundary` / `sigma_eps_boundary` field of the result is `true` and
`converged` is `false` (pinned parameters are never boundaries). The coordinate ascent alternates E-step (node-state
posterior via Q = B W Bᵀ), per-edge rate shrinkage, and hyperparameter updates.

The free-τ² update is **moment-matching / REML-style**: the dispersion of the
per-edge log-rate modes equals the prior variance plus the per-edge sampling
variance, so τ² = max(Var(ρ̂_e) − mean per-edge data variance, floor). This is
far less prone to the boundary-collapse pathology of the naive EM variance
update (which drives τ² → 0 whenever per-edge information is weak); it
collapses toward 0 ONLY when the observed dispersion is genuinely within
sampling noise of a single rate — which is itself the honest identifiability
signal at small `n_rep`.
"""
function fit_relaxed_clock(phy::EdgePhy, y::AbstractMatrix;
                           max_iter::Integer = 200, tol::Real = 1e-7,
                           τ²_floor::Real = 1e-4,
                           fix_τ²::Union{Nothing,Real} = nothing,
                           fix_σ²_eps::Union{Nothing,Real} = nothing,
                           μ_init::Union{Nothing,Real} = nothing,
                           σ²_eps_init::Union{Nothing,Real} = nothing,
                           verbose::Bool = false)
    p, n_rep = size(y)
    p == phy.n_leaves ||
        throw(ArgumentError("y first dim $p must equal n_leaves $(phy.n_leaves)"))
    cm = _cond_map(phy)
    df = float(n_rep)
    ℓ = phy.branch_lengths

    # Initialisation. A single-rate Brownian fit gives a sensible μ and σ²_eps
    # start: total leaf variance split between phylogenetic signal and noise.
    leaf_var = mean(vec(var(y; dims = 2, corrected = false)))
    σ²_eps = if fix_σ²_eps !== nothing
        float(fix_σ²_eps)
    elseif σ²_eps_init !== nothing
        float(σ²_eps_init)
    else
        max(0.1 * leaf_var, 1e-3)
    end
    # crude global rate: average increment variance ≈ leaf_var / mean depth.
    mean_depth = mean(_node_depths(phy, 1.0)[phy.leaf_indices])
    rate0 = max((leaf_var - σ²_eps) / max(mean_depth, eps()), 1e-3)
    μ = μ_init === nothing ? log(rate0) : float(μ_init)
    τ² = fix_τ² === nothing ? 0.5 : float(fix_τ²)
    σ²_e = fill(exp(μ), phy.n_edges)

    σ²_eps_floor = _relaxed_sigma_eps_floor(y)
    obj_trace = Float64[]
    inc_converged = false
    tau2_boundary = false
    sigma_eps_boundary = false
    iters = 0
    local μ_nodes = zeros(Float64, phy.n_nodes, n_rep)

    for iter in 1:max_iter
        iters = iter
        # ---- E-step: per-edge increment moments + reconstruction residual ----
        s_e, recon, μ_nodes_iter = estep_edge_moments(phy, σ²_e, σ²_eps, y, cm)
        μ_nodes = μ_nodes_iter

        # ---- M-step 1: per-branch log-rate shrinkage modes ----
        ρ̂, curv = shrink_logrates(s_e, df, ℓ, μ, τ²)
        σ²_e = exp.(ρ̂)

        # ---- M-step 2: hyperparameters ----
        μ = mean(ρ̂)
        if fix_τ² === nothing
            # Moment-matching / REML-style τ², from the UNSHRUNK per-edge MLEs
            # (not the shrunk modes — using the shrunk modes creates a
            # collapse fixed point: small τ² ⇒ modes pulled to μ ⇒ small
            # Var(modes) ⇒ smaller τ²). The per-edge MLE log-rate is
            #     ρ_e^MLE = log( s_e / (df · ℓ_e) ),
            # which equals the true log-rate plus sampling error of variance
            # ≈ 1/data_curv_e, where data_curv_e = (s_e/(2ℓ)) e^{−ρ̂} is the
            # per-edge data precision (the data term of −g''). Matching the
            # observed MLE dispersion to τ² + mean sampling variance:
            #     τ² = max( Var(ρ_e^MLE) − mean(1/data_curv_e), floor ).
            # Collapses to the floor ONLY when the MLE dispersion is genuinely
            # within sampling noise of a single rate (the honest weak-data
            # signal), not as an EM boundary artefact.
            ρ_mle = log.(max.(s_e ./ (df .* ℓ), 1e-300))
            data_curv = max.(curv .- 1.0 / τ², 1e-8)
            samp_var = mean(1.0 ./ data_curv)
            τ²_raw = var(ρ_mle) - samp_var
            τ² = max(τ²_raw, τ²_floor)
            tau2_boundary = τ²_raw <= τ²_floor
        end
        if fix_σ²_eps === nothing
            σ²_eps_raw = recon / (n_rep * p)
            σ²_eps = max(σ²_eps_raw, σ²_eps_floor)
            sigma_eps_boundary = σ²_eps_raw <= σ²_eps_floor
        else
            σ²_eps = float(fix_σ²_eps)
        end

        obj = _relaxed_obj(phy, σ²_e, σ²_eps, s_e, recon, μ, τ², n_rep, df)
        push!(obj_trace, obj)
        verbose && println("  iter $iter: obj=$(round(obj, digits=4)) " *
                           "μ=$(round(μ, digits=3)) τ²=$(round(τ², digits=4)) " *
                           "σ²_eps=$(round(σ²_eps, digits=4))")

        if iter > 1 && abs(obj_trace[end] - obj_trace[end-1]) < tol * (abs(obj_trace[end-1]) + tol)
            inc_converged = true
            break
        end
    end

    # A clamp at either floor is not an interior optimum: never report it as
    # converged, whatever the objective increment says.
    converged = inc_converged && !tau2_boundary && !sigma_eps_boundary
    return RelaxedClockFit(σ²_e, log.(σ²_e), μ, τ², σ²_eps, iters, converged,
                           obj_trace, μ_nodes, tau2_boundary, sigma_eps_boundary)
end

# ---------------------------------------------------------------------------
# 5. EVALUATION HELPERS (honest recovery diagnostics vs. known truth).
# ---------------------------------------------------------------------------

"""
    spearman(a, b) -> Float64

Spearman rank correlation (rank-Pearson). Self-contained; ties broken by
average ranks via `tiedrank`.
"""
function spearman(a::AbstractVector, b::AbstractVector)
    return cor(_tiedrank(a), _tiedrank(b))
end

function _tiedrank(x::AbstractVector)
    n = length(x)
    p = sortperm(x)
    r = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = i
        while j < n && x[p[j+1]] == x[p[i]]
            j += 1
        end
        avg = (i + j) / 2
        for k in i:j
            r[p[k]] = avg
        end
        i = j + 1
    end
    return r
end

"""
    shrinkage_factor(est, truth) -> Float64

Ratio of the dispersion (SD) of estimated to true per-branch (log-)rates. A
value < 1 is the signature of shrinkage: estimates pulled toward the global
mean are LESS dispersed than the truth. Computed on the log scale (the prior
scale).
"""
function shrinkage_factor(log_est::AbstractVector, log_truth::AbstractVector)
    return std(log_est) / std(log_truth)
end

"""
    clade_detection(σ²_e_est, fast_edges, slow_edges) -> NamedTuple

Did the fit separate a deliberately elevated-rate clade (`fast_edges`) from
the slow background (`slow_edges`)? Returns the mean estimated rate in each
group, their ratio, and a two-sample t-statistic on the log-rates (a crude
but honest separation signal at the prototype scale).
"""
function clade_detection(σ²_e_est::AbstractVector, fast_edges::AbstractVector{<:Integer},
                         slow_edges::AbstractVector{<:Integer})
    lf = log.(σ²_e_est[fast_edges])
    ls = log.(σ²_e_est[slow_edges])
    mean_fast = mean(σ²_e_est[fast_edges])
    mean_slow = mean(σ²_e_est[slow_edges])
    # Welch t on log-rates.
    nf, ns = length(lf), length(ls)
    sf2, ss2 = var(lf), var(ls)
    se = sqrt(sf2 / nf + ss2 / ns)
    t = se > 0 ? (mean(lf) - mean(ls)) / se : Inf
    return (; mean_fast, mean_slow, ratio = mean_fast / mean_slow, t,
            log_gap = mean(lf) - mean(ls))
end
