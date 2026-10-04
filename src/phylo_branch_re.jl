# Single-variance Brownian-motion phylogenetic model in INCIDENCE / path-
# membership form, with per-branch effects as RANDOM EFFECTS under ONE common
# variance — and per-branch "rate" estimates read off as the BLUPs (conditional
# modes) of those increments.
#
# ===========================================================================
# WHAT THIS IS (and how it differs from relaxed_clock.jl)
# ===========================================================================
# `relaxed_clock.jl` fit a SEPARATE variance per branch via a hierarchical
# prior  log σ²_e ~ N(μ, τ²)  and found τ² unidentifiable from a single trait
# (needing many i.i.d. replicates). That is a VARIABLE-RATE model: 2p−2 free
# variances tied by a hyper-prior.
#
# This file implements the SIMPLER and CORRECT object (B. Bolker, mixedmodels-
# misc notes on phylogenetics; phylogenetic ridge regression): a SINGLE-variance
# Brownian motion where every branch contributes an i.i.d. random-effect
# increment under ONE common variance σ². The per-branch "rates" are then the
# BLUPs (conditional modes) of those increments — they are PREDICTIONS, not free
# parameters, so they are available FREE and IDENTIFIABLE from a SINGLE trait.
# Only THREE parameters are estimated: {μ, σ², σ²_eps}.
#
# ===========================================================================
# THE MODEL (single trait y ∈ ℝ^p, p tips, E = 2p−2 edges)
# ===========================================================================
#       y_i = μ + (Z z)_i + ε_i,     ε ~ N(0, σ²_eps · I_p)
#       z_e ~ N(0, σ² · ℓ_e)   independent across edges
#   ⟹  DIAGONAL prior precision   W = diag( 1 / (σ² · ℓ_e) )          (E × E)
#
# Z :: p × E path-membership ("incidence") matrix:
#       Z[i, e] = 1  if edge e lies on the root → tip-i path, else 0.
# Z z is the standard BM leaf value: leaf i = Σ_{e ∈ path_i} z_e, and z_e is the
# increment on edge e with variance σ²ℓ_e. Hence the marginal leaf covariance is
#       V = Z · diag(ℓ) · Zᵀ ,   V[i,j] = Σ_{e ∈ path_i ∩ path_j} ℓ_e
# (the shared root → MRCA path length — the usual BM covariance), and
#       y ~ N(μ·1, σ²·V + σ²_eps·I).
#
# W TRICK (kept — it is the correct one): the prior precision carries the
# INVERSE branch length 1/(σ²ℓ_e) on its diagonal; Z is the raw 0/1 incidence
# (NO √ℓ_e scaling of Z). With the √ℓ_e-scaling-of-Z convention the BLUPs come
# out as z_e/√ℓ_e and need back-transforming; the inverse-W form gives z_e
# directly.
#
# ===========================================================================
# THE ESTIMATOR — EXACT ML, SPARSE PRECISION WITH A MARGINAL FALLBACK
# ===========================================================================
# The usual path uses the SPARSE E×E z-posterior precision
#       Λ = W + σ_eps⁻² · Zᵀ Z            (sparse; nnz ≈ E·depth for balanced trees)
# via the Woodbury / matrix-determinant-lemma identities for
# Σ = σ²_eps·I + Z D Zᵀ,  D = σ²·diag(ℓ),  W = D⁻¹:
#
#   Σ⁻¹ b   = σ_eps⁻² b − σ_eps⁻⁴ · Z · ( Λ⁻¹ ( Zᵀ b ) )
#   log|Σ|  = p·log σ²_eps + log|Λ| + log|D|,   log|D| = E·log σ² + Σ_e log ℓ_e
#
# Normally log|Λ| and every solve use one sparse Cholesky. Numerically unsafe
# auxiliary systems instead factor the equivalent dense marginal covariance.
# μ is profiled out by GLS: μ̂ = (1ᵀΣ⁻¹1)⁻¹ (1ᵀΣ⁻¹y). The two
# variances {σ², σ²_eps} are fit by maximising the profile marginal likelihood
# (2-D, derivative-free).
#
# BLUPs (conditional modes of the branch increments under the single variance):
#       ẑ = E[z | y] = σ_eps⁻² · Λ⁻¹ · Zᵀ (y − μ̂·1)
# Per-branch rate read-outs:
#       standardized increment  ẑ_e / √ℓ_e   (prior scale: ~N(0, σ²))
#       per-branch rate          ẑ_e² / ℓ_e   (prior mean σ²)
#
# This file is self-contained and does NOT modify any existing src file. It
# depends only on `EdgePhy` / `edge_phy` from `edge_incidence.jl`.

using LinearAlgebra
using SparseArrays
using Random
using Statistics

# ---------------------------------------------------------------------------
# 1. Sparse path-membership (incidence) matrix Z.
# ---------------------------------------------------------------------------

"""
    path_membership(phy::EdgePhy) -> SparseMatrixCSC{Float64,Int}

Build the `p × E` path-membership matrix `Z` with `Z[i, e] = 1` iff edge `e`
lies on the root → tip-i path. Each leaf's root path is collected by walking
`node_parent` up to the root and recording `node_edge`. Total non-zeros =
Σ_tips (depth in edges) = O(p·depth); for a balanced tree O(p log p).
"""
function path_membership(phy::EdgePhy)
    p = phy.n_leaves
    Iv = Int[]
    Jv = Int[]
    sizehint!(Iv, 4p)
    sizehint!(Jv, 4p)
    @inbounds for i in 1:p
        u = phy.leaf_indices[i]
        while u != phy.root_index
            e = phy.node_edge[u]
            e == 0 && break
            push!(Iv, i)
            push!(Jv, e)
            u = phy.node_parent[u]
        end
    end
    Vv = ones(Float64, length(Iv))
    return sparse(Iv, Jv, Vv, p, phy.n_edges)
end

# ---------------------------------------------------------------------------
# 2. Data-generating model (single-variance, or planted variable rate).
# ---------------------------------------------------------------------------
# We reuse `simulate_relaxed_bm` from relaxed_clock.jl when it is available
# (the tests include both files); to keep THIS file independently usable we
# provide a minimal generator here under a distinct name.

"""
    simulate_branch_re(phy, σ², σ²_eps, n_rep; rng, μ=0.0, σ²_e=nothing)
        -> (y, z_true)

Simulate `n_rep` i.i.d. single-trait BM realisations under the branch random-
effects model. Increment `z_e ~ N(0, rate_e · ℓ_e)` where `rate_e = σ²` for the
single-variance truth, or `rate_e = σ²_e[e]` if a per-edge rate vector is
supplied (planted variable-rate truth, used to test BLUP detection of a fast
clade). Leaf value = μ + Σ_{e ∈ path} z_e + ε, ε ~ N(0, σ²_eps).

Returns `y :: p × n_rep` and `z_true :: E × n_rep` (the realised increments).
"""
function simulate_branch_re(phy::EdgePhy, σ²::Real, σ²_eps::Real,
                            n_rep::Integer;
                            rng::AbstractRNG = Random.default_rng(),
                            μ::Real = 0.0,
                            σ²_e::Union{Nothing,AbstractVector} = nothing)
    n_rep ≥ 1 || throw(ArgumentError("n_rep must be ≥ 1"))
    p = phy.n_leaves
    rate = σ²_e === nothing ? fill(float(σ²), phy.n_edges) : collect(float.(σ²_e))
    length(rate) == phy.n_edges ||
        throw(ArgumentError("σ²_e length must equal n_edges $(phy.n_edges)"))
    sd_e = sqrt.(rate .* phy.branch_lengths)
    z_true = randn(rng, phy.n_edges, n_rep) .* sd_e
    Z = path_membership(phy)
    leaf_signal = Z * z_true                       # p × n_rep
    y = μ .+ leaf_signal .+ sqrt(σ²_eps) .* randn(rng, p, n_rep)
    return y, z_true
end

# ---------------------------------------------------------------------------
# 3. Sparse z-posterior system Λ = W + σ_eps⁻² ZᵀZ, and Woodbury primitives.
# ---------------------------------------------------------------------------
# All inference primitives consume a precomputed `ZtZ = Zᵀ Z` (sparse, built
# once per tree) so the per-evaluation cost is one diagonal update + one sparse
# Cholesky. Numerically unsafe auxiliary systems use a dense marginal fallback.

"""
    BranchRECache

Per-tree precomputation reused across likelihood evaluations: the sparse
incidence `Z`, its Gram matrix `ZtZ = ZᵀZ` (E×E sparse), branch lengths `ℓ`,
the sum-of-log-branch-lengths constant, and `p`.
"""
struct BranchRECache
    Z::SparseMatrixCSC{Float64,Int}
    ZtZ::SparseMatrixCSC{Float64,Int}
    ℓ::Vector{Float64}
    sum_log_ℓ::Float64
    p::Int
    E::Int
end

"""
    branch_re_cache(phy::EdgePhy) -> BranchRECache

Build the per-tree [`BranchRECache`](@ref) (incidence `Z`, Gram matrix `ZᵀZ`, branch
lengths and their log-sum) for the tree `phy` from [`edge_phy`](@ref). It is the first
argument of [`branch_re_profile_negll`](@ref) and [`branch_blups`](@ref).
"""
function branch_re_cache(phy::EdgePhy)
    Z = path_membership(phy)
    ZtZ = Z' * Z
    ℓ = copy(phy.branch_lengths)
    return BranchRECache(Z, ZtZ, ℓ, sum(log, ℓ), phy.n_leaves, phy.n_edges)
end

# Assemble the sparse posterior precision Λ = diag(1/(σ²ℓ)) + σ_eps⁻² ZᵀZ.
function _lambda(cache::BranchRECache, σ²::Real, σ²_eps::Real)
    inv_eps = 1.0 / σ²_eps
    Λ = inv_eps .* cache.ZtZ
    d = 1.0 ./ (σ² .* cache.ℓ)
    @inbounds for e in 1:cache.E
        Λ[e, e] += d[e]
    end
    return Λ
end

# Cholesky of the (SPD) sparse Λ, scaled by its largest stored magnitude.
# Λ is SPD because W ≻ 0 and σ_eps⁻²ZᵀZ ⪰ 0.  Scaling only changes the
# numerical representation supplied to CHOLMOD: Λ = λscale * Λscaled.  It
# never changes the objective with a ridge or other model perturbation.
function _lambda_chol(Λ::SparseMatrixCSC{Float64,Int})
    λscale = maximum(abs, nonzeros(Λ))
    isfinite(λscale) && λscale > 0 ||
        throw(DomainError(λscale, "Λ contains an unsafe numerical scale"))
    return cholesky(Symmetric(Λ / λscale)), λscale
end

# An auxiliary precision can be ill-conditioned while the marginal covariance
# is benign. Return nothing to choose an equivalent marginal representation,
# never to exclude such a parameter point from the statistical model.
_valid_branch_variances(a, b) = isfinite(a) && a > 0 && isfinite(b) && b > 0
function _branch_precision_factor(cache::BranchRECache, σ²::Real, σ²_eps::Real)
    inv_eps = 1.0 / σ²_eps
    isfinite(inv_eps) || return nothing
    Λ = _lambda(cache, σ², σ²_eps)
    all(isfinite, nonzeros(Λ)) || return nothing
    λscale = maximum(abs, nonzeros(Λ))
    cond_lower = cache.E > cache.p ? λscale * σ² * minimum(cache.ℓ) : 1.0
    isfinite(cond_lower) && cond_lower ≤ inv(sqrt(eps(Float64))) || return nothing
    try
        return _lambda_chol(Λ)[1], λscale, inv_eps
    catch err
        err isa PosDefException || rethrow(err)
        return nothing
    end
end

function _branch_marginal_factor(cache::BranchRECache, σ²::Real, σ²_eps::Real)
    @warn "Branch precision is numerically unsafe; using the equivalent dense marginal covariance (O(p²) memory)." maxlog=1
    scale = max(σ², σ²_eps)
    B = Matrix(cache.Z * spdiagm(0 => (σ² / scale) .* cache.ℓ) * cache.Z')
    @inbounds for i in 1:cache.p
        B[i, i] += σ²_eps / scale
    end
    all(isfinite, B) || throw(DomainError(scale, "marginal covariance is not representable"))
    return cholesky(Symmetric(B)), scale
end

function _branch_marginal_profile(cache::BranchRECache, y, σ², σ²_eps)
    C, scale = try
        _branch_marginal_factor(cache, σ², σ²_eps)
    catch err
        (err isa PosDefException || err isa DomainError) || rethrow(err)
        return Inf, NaN
    end
    ymean = mean(y)
    yc = y .- ymean
    one = ones(cache.p)
    w1 = C \ one
    wy = C \ yc
    denominator = dot(one, w1)
    isfinite(denominator) && denominator > 0 || return Inf, NaN
    μc = dot(one, wy) / denominator
    r = yc .- μc
    q = dot(r, C \ r) / scale
    nll = (cache.p * log(2π) + logdet(C) + cache.p * log(scale) + q) / 2
    μ = ymean + μc
    return isfinite(nll) && isfinite(μ) && q >= 0 ? (nll, μ) : (Inf, NaN)
end

# ---------------------------------------------------------------------------
# 4. Profile marginal log-likelihood (μ profiled out by GLS).
# ---------------------------------------------------------------------------

"""
    branch_re_profile_negll(cache, y, σ², σ²_eps) -> (negll, μ̂)

Exact profile −log-likelihood of the single-variance branch-RE model at
`(σ², σ²_eps)`, with `μ` profiled out by GLS, normally through sparse E×E
precision Λ. Numerically unsafe auxiliary systems use an equivalent dense
marginal covariance with a warning; no parameter-space restriction or ridge
is imposed. `y` is a length-p single trait.

Uses Σ⁻¹b = σ_eps⁻²b − σ_eps⁻⁴ Z Λ⁻¹ Zᵀb and
log|Σ| = p·log σ²_eps + log|Λ| + (E·log σ² + Σ log ℓ).
"""
function branch_re_profile_negll(cache::BranchRECache, y::AbstractVector,
                                 σ²::Real, σ²_eps::Real)
    length(y) == cache.p ||
        throw(ArgumentError("y length $(length(y)) must equal p $(cache.p)"))
    # Nelder–Mead is unconstrained on the log scale.  An overflowed or
    # underflowed exponent is an invalid objective trial, not a near-optimal
    # parameter point and not evidence that the SPD model needs a ridge.
    _valid_branch_variances(σ², σ²_eps) || return Inf, NaN
    all(isfinite, y) || return Inf, NaN
    precision = _branch_precision_factor(cache, σ², σ²_eps)
    isnothing(precision) && return _branch_marginal_profile(cache, y, σ², σ²_eps)
    cΛ, λscale, inv_eps = precision

    # Σ⁻¹ applied to an arbitrary RHS via Woodbury.  Solve for the posterior
    # mean first, then form the residual at b's original scale.  Expanding this
    # to inv_eps*b - inv_eps²*Z*Λ⁻¹Z'b loses the residual when σ²_eps is small.
    # Here Λ⁻¹ = Λscaled⁻¹ / λscale and λscale ≥ inv_eps.
    function Σinv(b)
        zpost = (inv_eps / λscale) .* (cΛ \ (cache.Z' * b))
        return inv_eps .* (b .- cache.Z * zpost)
    end

    # The intercept is profiled, so centre before the sparse solves. This is
    # translation-invariant and prevents a large mean from obscuring the
    # residual in the Woodbury subtraction.
    ymean = mean(y)
    yc = y .- ymean
    one_p = ones(cache.p)
    Sinv_1 = Σinv(one_p)
    Sinv_y = Σinv(yc)
    a11 = dot(one_p, Sinv_1)            # 1ᵀ Σ⁻¹ 1
    a1y = dot(one_p, Sinv_y)            # 1ᵀ Σ⁻¹ y
    isfinite(a11) && a11 > 0 && isfinite(a1y) || return Inf, NaN
    μc = a1y / a11                       # GLS intercept of the centred data
    μ̂ = ymean + μc

    r = yc .- μc                         # residual
    # At the GLS intercept rᵀΣ⁻¹y = rᵀΣ⁻¹r.  This form avoids subtracting
    # two O(μ̂²) terms in yᵀΣ⁻¹y − μ̂²·1ᵀΣ⁻¹1.
    quad = dot(r, Sinv_y)
    if !isfinite(quad)
        return Inf, NaN
    elseif quad < 0
        # This subtraction can lose its non-negative quadratic-form property
        # when the fitted intercept is large. Re-evaluate from the residual;
        # a negative result is never accepted as a healthy likelihood.
        quad = dot(r, Σinv(r))
        isfinite(quad) && quad >= 0 || return Inf, NaN
    end

    logdetΣ = cache.p * log(σ²_eps) + logdet(cΛ) + cache.E * log(λscale) +
              (cache.E * log(σ²) + cache.sum_log_ℓ)

    negll = 0.5 * (cache.p * log(2π) + logdetΣ + quad)
    return isfinite(negll) && isfinite(μ̂) ? (negll, μ̂) : (Inf, NaN)
end

# ---------------------------------------------------------------------------
# 5. BLUPs (conditional modes of the branch increments) + per-branch rates.
# ---------------------------------------------------------------------------

"""
    branch_blups(cache, y, σ², σ²_eps, μ) -> (ẑ, std_incr, rate_e)

Branch-increment BLUPs ẑ = E[z|y] = σ_eps⁻² Λ⁻¹ Zᵀ(y − μ·1), and the per-branch
rate read-outs:
  * `std_incr[e] = ẑ_e / √ℓ_e`  (prior scale; ~N(0, σ²) under the prior),
  * `rate_e[e]   = ẑ_e² / ℓ_e`  (per-branch rate; prior mean σ²).
Uses the sparse precision when safe, otherwise the same dense marginal
fallback as [`branch_re_profile_negll`](@ref). Invalid variances raise
`DomainError`; response-length mismatches raise `ArgumentError`.
"""
function branch_blups(cache::BranchRECache, y::AbstractVector,
                      σ²::Real, σ²_eps::Real, μ::Real)
    length(y) == cache.p || throw(ArgumentError("y length must equal p $(cache.p)"))
    _valid_branch_variances(σ², σ²_eps) ||
        throw(DomainError((σ², σ²_eps), "branch variances must be finite and positive"))
    all(isfinite, y) && isfinite(μ) ||
        throw(DomainError(μ, "response and mean must be finite"))
    precision = _branch_precision_factor(cache, σ², σ²_eps)
    if isnothing(precision)
        C, scale = _branch_marginal_factor(cache, σ², σ²_eps)
        ẑ = ((σ² / scale) .* cache.ℓ) .* (cache.Z' * (C \ (y .- μ)))
    else
        cΛ, λscale, inv_eps = precision
        ẑ = (inv_eps / λscale) .* (cΛ \ (cache.Z' * (y .- μ)))
    end
    sqrtℓ = sqrt.(cache.ℓ)
    std_incr = ẑ ./ sqrtℓ
    rate_e = (ẑ .^ 2) ./ cache.ℓ
    return ẑ, std_incr, rate_e
end

# ---------------------------------------------------------------------------
# 6. Public driver: ML fit of {μ, σ², σ²_eps}.
# ---------------------------------------------------------------------------

"""
    BranchREFit

Result of `fit_branch_re`. Fields:
  * `μ`, `σ²`, `σ²_eps` – the THREE ML parameters.
  * `negll`            – profile −log-likelihood at the optimum.
  * `ẑ`                – branch-increment BLUPs (length E).
  * `std_incr`         – ẑ_e/√ℓ_e (length E).
  * `rate_e`           – ẑ_e²/ℓ_e per-branch rate read-outs (length E).
  * `n_iter`, `converged` – optimiser diagnostics.
"""
struct BranchREFit
    μ::Float64
    σ²::Float64
    σ²_eps::Float64
    negll::Float64
    ẑ::Vector{Float64}
    std_incr::Vector{Float64}
    rate_e::Vector{Float64}
    n_iter::Int
    converged::Bool
end

# Post-fit: `_nparams` unlocks generic `dof`/`aic`/`bic` in src/postfit.jl.
_nparams(::BranchREFit) = 3

"""
    fit_branch_re(phy, y; σ²_init, σ²_eps_init, max_iter, tol, fix_σ²_eps)
        -> BranchREFit

ML fit of the single-variance branch random-effects model on a SINGLE trait
`y` (length p). Optimises the 2-D profile −log-likelihood over
`(log σ², log σ²_eps)` (μ profiled out analytically) with a self-contained
Nelder–Mead, then extracts the branch BLUPs and per-branch rate read-outs.

`fix_σ²_eps` pins the observation-noise variance (then a 1-D optimisation over
log σ² only). Defaults split the marginal trait variance between phylogenetic
signal and noise as a warm start.
"""
function fit_branch_re(phy::EdgePhy, y::AbstractVector;
                       σ²_init::Union{Nothing,Real} = nothing,
                       σ²_eps_init::Union{Nothing,Real} = nothing,
                       fix_σ²_eps::Union{Nothing,Real} = nothing,
                       max_iter::Integer = 500, tol::Real = 1e-9)
    cache = branch_re_cache(phy)
    yv = collect(float.(y))
    length(yv) == cache.p ||
        throw(ArgumentError("y length $(length(yv)) must equal p $(cache.p)"))

    vy = var(yv; corrected = false)
    # Mean root→tip path length (in branch-length units) for a rate warm-start.
    mean_depth = mean(cache.Z * cache.ℓ)
    σ²0 = σ²_init === nothing ? max(0.5 * vy / max(mean_depth, eps()), 1e-4) :
          float(σ²_init)
    σ²eps0 = if fix_σ²_eps !== nothing
        float(fix_σ²_eps)
    elseif σ²_eps_init !== nothing
        float(σ²_eps_init)
    else
        max(0.5 * vy, 1e-4)
    end

    if fix_σ²_eps !== nothing
        # 1-D over log σ².
        f1(lσ²) = branch_re_profile_negll(cache, yv, exp(lσ²), σ²eps0)[1]
        x̂, fbest, iters, conv = _nelder_mead_1d(f1, log(σ²0); max_iter, tol)
        σ² = exp(x̂)
        σ²_eps = σ²eps0
        negll = fbest
        n_iter = iters
        converged = conv
    else
        f2(θ) = branch_re_profile_negll(cache, yv, exp(θ[1]), exp(θ[2]))[1]
        x0 = [log(σ²0), log(σ²eps0)]
        x̂, fbest, iters, conv = _nelder_mead(f2, x0; max_iter, tol)
        σ² = exp(x̂[1])
        σ²_eps = exp(x̂[2])
        negll = fbest
        n_iter = iters
        converged = conv
    end

    isfinite(negll) && isfinite(σ²) && σ² > 0 && isfinite(σ²_eps) && σ²_eps > 0 ||
        throw(ErrorException("branch-RE optimisation found no finite likelihood point"))

    _, μ̂ = branch_re_profile_negll(cache, yv, σ², σ²_eps)
    ẑ, std_incr, rate_e = branch_blups(cache, yv, σ², σ²_eps, μ̂)
    return BranchREFit(μ̂, σ², σ²_eps, negll, ẑ, std_incr, rate_e,
                       n_iter, converged)
end

# ---------------------------------------------------------------------------
# 7. Self-contained derivative-free optimisers (no Optim dependency).
# ---------------------------------------------------------------------------
# A compact Nelder–Mead for the 2-D variance problem and a golden-section /
# bracket-Newton hybrid for the 1-D fixed-σ²_eps case. Both are deliberately
# small and assume the smooth, unimodal profile surfaces of this model.

function _nelder_mead(f, x0::AbstractVector; max_iter::Integer = 500,
                      tol::Real = 1e-9)
    n = length(x0)
    α, γ, ρ, σ = 1.0, 2.0, 0.5, 0.5
    # Initial simplex.
    simplex = [copy(float.(x0))]
    for i in 1:n
        xi = copy(float.(x0))
        xi[i] += (xi[i] == 0 ? 0.05 : 0.1 * abs(xi[i]) + 0.05)
        push!(simplex, xi)
    end
    fvals = [f(v) for v in simplex]
    iters = 0
    converged = false
    for it in 1:max_iter
        iters = it
        ord = sortperm(fvals)
        simplex = simplex[ord]; fvals = fvals[ord]
        if abs(fvals[end] - fvals[1]) ≤ tol * (abs(fvals[1]) + tol)
            converged = true
            break
        end
        xc = sum(simplex[1:end-1]) ./ n                     # centroid (drop worst)
        xr = xc .+ α .* (xc .- simplex[end])                # reflect
        fr = f(xr)
        if fr < fvals[1]
            xe = xc .+ γ .* (xr .- xc)                      # expand
            fe = f(xe)
            if fe < fr
                simplex[end] = xe; fvals[end] = fe
            else
                simplex[end] = xr; fvals[end] = fr
            end
        elseif fr < fvals[end-1]
            simplex[end] = xr; fvals[end] = fr
        else
            xk = xc .+ ρ .* (simplex[end] .- xc)            # contract
            fk = f(xk)
            if fk < fvals[end]
                simplex[end] = xk; fvals[end] = fk
            else
                for i in 2:(n + 1)                          # shrink
                    simplex[i] = simplex[1] .+ σ .* (simplex[i] .- simplex[1])
                    fvals[i] = f(simplex[i])
                end
            end
        end
    end
    ord = sortperm(fvals)
    return simplex[ord[1]], fvals[ord[1]], iters, converged
end

function _nelder_mead_1d(f, x0::Real; max_iter::Integer = 500, tol::Real = 1e-9)
    x̂, fbest, iters, conv = _nelder_mead(v -> f(v[1]), [float(x0)];
                                          max_iter, tol)
    return x̂[1], fbest, iters, conv
end

# ---------------------------------------------------------------------------
# 8. Dense-V reference fit (for the speed contrast ONLY — not the proper path).
# ---------------------------------------------------------------------------

"""
    fit_branch_re_dense(phy, y; ...) -> BranchREFit

Reference implementation that FORMS the dense p×p Σ = σ²V + σ²_eps·I and solves
it densely (O(p³) factorisation per likelihood evaluation). Used solely to
contrast scaling against the sparse `fit_branch_re`; the proper formulation
never densifies. `V = Z diag(ℓ) Zᵀ` is built once.
"""
function fit_branch_re_dense(phy::EdgePhy, y::AbstractVector;
                             σ²_init::Union{Nothing,Real} = nothing,
                             σ²_eps_init::Union{Nothing,Real} = nothing,
                             fix_σ²_eps::Union{Nothing,Real} = nothing,
                             max_iter::Integer = 500, tol::Real = 1e-9)
    p = phy.n_leaves
    yv = collect(float.(y))
    Z = path_membership(phy)
    V = Matrix(Z * spdiagm(0 => phy.branch_lengths) * Z')     # dense p×p
    one_p = ones(p)

    function negll_dense(σ², σ²_eps)
        Σ = σ² .* V
        @inbounds for i in 1:p
            Σ[i, i] += σ²_eps
        end
        cΣ = cholesky(Symmetric(Σ))
        Sinv_1 = cΣ \ one_p
        Sinv_y = cΣ \ yv
        a11 = dot(one_p, Sinv_1)
        μ̂ = dot(one_p, Sinv_y) / a11
        quad = dot(yv, Sinv_y) - μ̂^2 * a11
        return 0.5 * (p * log(2π) + logdet(cΣ) + quad), μ̂
    end

    vy = var(yv; corrected = false)
    σ²0 = σ²_init === nothing ? max(0.5 * vy, 1e-4) : float(σ²_init)
    σ²eps0 = fix_σ²_eps !== nothing ? float(fix_σ²_eps) :
             (σ²_eps_init === nothing ? max(0.5 * vy, 1e-4) : float(σ²_eps_init))

    if fix_σ²_eps !== nothing
        f1(lσ²) = negll_dense(exp(lσ²), σ²eps0)[1]
        x̂, fbest, iters, conv = _nelder_mead_1d(f1, log(σ²0); max_iter, tol)
        σ², σ²_eps = exp(x̂), σ²eps0
    else
        f2(θ) = negll_dense(exp(θ[1]), exp(θ[2]))[1]
        x̂, fbest, iters, conv = _nelder_mead(f2, [log(σ²0), log(σ²eps0)];
                                              max_iter, tol)
        σ², σ²_eps = exp(x̂[1]), exp(x̂[2])
        fbest = fbest
    end
    _, μ̂ = negll_dense(σ², σ²_eps)
    # BLUPs via the dense path: ẑ = D Zᵀ Σ⁻¹ (y − μ). (reference only)
    Σ = σ² .* V; @inbounds for i in 1:p; Σ[i, i] += σ²_eps; end
    cΣ = cholesky(Symmetric(Σ))
    D = σ² .* phy.branch_lengths
    ẑ = D .* (Z' * (cΣ \ (yv .- μ̂)))
    sqrtℓ = sqrt.(phy.branch_lengths)
    return BranchREFit(μ̂, σ², σ²_eps, negll_dense(σ², σ²_eps)[1],
                       ẑ, ẑ ./ sqrtℓ, (ẑ .^ 2) ./ phy.branch_lengths,
                       iters, conv)
end

# ---------------------------------------------------------------------------
# 9. Clade / edge plumbing and statistics for the validation gates.
# ---------------------------------------------------------------------------

"""
    clade_edges(phy, clade_root_node) -> Vector{Int}

All edge indices in the subtree rooted at `clade_root_node` (including the edge
leading INTO that node from its parent). Used to plant / detect a fast clade.
"""
function clade_edges(phy::EdgePhy, clade_root_node::Integer)
    edges = Int[]
    stack = [Int(clade_root_node)]
    while !isempty(stack)
        u = pop!(stack)
        if u != phy.root_index
            e = phy.node_edge[u]
            e != 0 && push!(edges, e)
        end
        for v in phy.node_children[u]
            push!(stack, v)
        end
    end
    return edges
end

"""
    find_clade_root(phy; target_leaves) -> Int

Find an internal node whose subtree contains close to `target_leaves` leaves
(used to pick a non-trivial clade to plant a rate shift on). Returns the node
index of the subtree root with leaf-count nearest `target_leaves`.
"""
function find_clade_root(phy::EdgePhy; target_leaves::Integer)
    # leaf count under each node via post-order
    nleaf = zeros(Int, phy.n_nodes)
    order = _postorder(phy)
    for u in order
        if isempty(phy.node_children[u])
            nleaf[u] = 1
        else
            nleaf[u] = sum(nleaf[v] for v in phy.node_children[u])
        end
    end
    best, bestnode = typemax(Int), phy.root_index
    for u in 1:phy.n_nodes
        u == phy.root_index && continue
        isempty(phy.node_children[u]) && continue       # need an internal clade
        d = abs(nleaf[u] - target_leaves)
        if d < best
            best, bestnode = d, u
        end
    end
    return bestnode
end

function _postorder(phy::EdgePhy)
    order = Int[]
    visited = falses(phy.n_nodes)
    stack = [(phy.root_index, false)]
    while !isempty(stack)
        u, processed = pop!(stack)
        if processed
            push!(order, u)
        else
            push!(stack, (u, true))
            for v in phy.node_children[u]
                push!(stack, (v, false))
            end
        end
    end
    return order
end

"""
    welch_t(a, b) -> (t, dmean, cohen_d)

Welch two-sample t-statistic for unequal variances, the raw mean difference,
and a pooled-SD Cohen's d effect size.
"""
function welch_t(a::AbstractVector, b::AbstractVector)
    na, nb = length(a), length(b)
    ma, mb = mean(a), mean(b)
    va, vb = var(a), var(b)
    se = sqrt(va / na + vb / nb)
    t = se > 0 ? (ma - mb) / se : (ma == mb ? 0.0 : Inf)
    pooled_sd = sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    d = pooled_sd > 0 ? (ma - mb) / pooled_sd : Inf
    return t, ma - mb, d
end

"""
    rank_sum_z(a, b) -> Float64

Normal-approximation z-statistic of the Mann–Whitney / Wilcoxon rank-sum test
for `a` vs `b` (a nonparametric companion to `welch_t`). Self-contained.
"""
function rank_sum_z(a::AbstractVector, b::AbstractVector)
    na, nb = length(a), length(b)
    all = vcat(collect(float.(a)), collect(float.(b)))
    r = _tiedrank_re(all)
    Ra = sum(@view r[1:na])
    U = Ra - na * (na + 1) / 2
    μU = na * nb / 2
    σU = sqrt(na * nb * (na + nb + 1) / 12)
    return σU > 0 ? (U - μU) / σU : 0.0
end

function _tiedrank_re(x::AbstractVector)
    n = length(x)
    pidx = sortperm(x)
    r = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = i
        while j < n && x[pidx[j+1]] == x[pidx[i]]
            j += 1
        end
        avg = (i + j) / 2
        for k in i:j
            r[pidx[k]] = avg
        end
        i = j + 1
    end
    return r
end

"""
    excess_kurtosis(x) -> Float64

Sample excess kurtosis (Fisher; 0 for a Gaussian). A Gaussian prior on the
standardized BLUPs predicts ≈ 0; rate variation in the truth inflates the tails
(positive excess kurtosis).
"""
function excess_kurtosis(x::AbstractVector)
    n = length(x)
    m = mean(x)
    s2 = mean((x .- m) .^ 2)
    m4 = mean((x .- m) .^ 4)
    return m4 / s2^2 - 3.0
end

"""
    qq_max_dev(x) -> Float64

Max absolute deviation between the standardized empirical quantiles of `x` and
the standard-normal quantiles (a QQ-departure statistic; ≈ 0 under normality).
"""
function qq_max_dev(x::AbstractVector)
    n = length(x)
    xs = sort((x .- mean(x)) ./ std(x))
    maxdev = 0.0
    for i in 1:n
        pq = (i - 0.5) / n
        z = sqrt(2) * _erfinv(2 * pq - 1)        # standard-normal quantile
        maxdev = max(maxdev, abs(xs[i] - z))
    end
    return maxdev
end

# Minimal inverse error function (Giles 2010 rational approximation); adequate
# for a QQ-departure diagnostic.
function _erfinv(x::Real)
    w = -log((1 - x) * (1 + x))
    if w < 5.0
        w -= 2.5
        p = 2.81022636e-08
        p = 3.43273939e-07 + p * w
        p = -3.5233877e-06 + p * w
        p = -4.39150654e-06 + p * w
        p = 0.00021858087 + p * w
        p = -0.00125372503 + p * w
        p = -0.00417768164 + p * w
        p = 0.246640727 + p * w
        p = 1.50140941 + p * w
    else
        w = sqrt(w) - 3.0
        p = -0.000200214257
        p = 0.000100950558 + p * w
        p = 0.00134934322 + p * w
        p = -0.00367342844 + p * w
        p = 0.00573950773 + p * w
        p = -0.0076224613 + p * w
        p = 0.00943887047 + p * w
        p = 1.00167406 + p * w
        p = 2.83297682 + p * w
    end
    return p * x
end
