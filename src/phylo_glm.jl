# Phylogenetic GLLVM for non-Gaussian families — joint Laplace over the augmented
# sparse phylogenetic precision (issue #61, the "make it exist" half).
#
# Model (p species = tree leaves, n sites): each species t carries a phylogenetic
# random intercept a_t, correlated across species by the tree:
#
#     u ~ N(0, (Q_cond/σ²_phy)⁻¹)   on the (2p−2) non-root tree nodes
#     a_t = u[leaf_pos[t]]          species effect (S = leaf selector)
#     η_{ts} = β_t + a_t            (shared across all sites s of species t)
#     y_{ts} ~ Family(linkinv(η_{ts}))
#
# Q_cond is `phy.Q_topology` with the root row/col dropped (PD), exactly the sparse
# precision the Gaussian phylo path uses (likelihood_sparse_phy.jl). The marginal is
# a joint Laplace over u — the same sparse-GMRF construction as the SPDE-latent model
# (spde_latent.jl), here with the leaf-selector "projector" and the per-species effect
# shared across sites:
#
#   * mode û by Fisher-scoring Newton, H = Q + Sᵀ diag(W_tot) S  (sparse),
#     W_tot[t] = Σ_s weight_{ts};  score scattered to leaf positions, Σ over sites;
#   * log p(Y) ≈ ℓ(û) − ½ ûᵀQû + ½ logdet Q − ½ logdet H.
#
# This is the WORKING fit (CHOLMOD sparse Cholesky, finite-difference outer gradient,
# like fit_spde_latent_gllvm). The O(p) Takahashi-selected-inverse analytic gradient
# (issue #61's optimisation) and ADEMP/benchmarking remain the runtime-session piece —
# CHOLMOD blocks the ForwardDiff route that made the iid-latent gradients easy.
#
# Verification anchors (no runtime): σ²_phy → 0 reduces to the independent-family
# marginal; and the augmented-state marginal equals the dense Σ_a = σ²_phy·S Q_cond⁻¹ Sᵀ
# joint Laplace (internal nodes marginalised exactly), checked in the tests.

using SparseArrays
using LinearAlgebra

# Leaf positions in the (2p−2)-vector of non-root nodes (root row/col dropped).
function _phylo_leaf_pos(phy::AugmentedPhy)
    p = phy.n_leaves
    leaf_pos = Vector{Int}(undef, p)
    @inbounds for t in 1:p
        lp = phy.leaf_indices[t]
        phy.root_index < lp && (lp -= 1)
        leaf_pos[t] = lp
    end
    return leaf_pos
end

# Q_cond (root-dropped topology precision) and the leaf positions.
function _phylo_qcond(phy::AugmentedPhy)
    keep = filter(i -> i != phy.root_index, 1:phy.n_total)
    Q_cond = phy.Q_topology[keep, keep]
    return Q_cond, _phylo_leaf_pos(phy)
end

# Fisher-scoring mode of u (length n_block) for the phylo random effect, given the
# sparse prior precision Q and leaf positions. Returns (û, cholH) or (nothing, nothing).
function _phylo_glm_mode(family, Y, N, β, link, Q, leaf_pos, n_block;
                         maxiter::Integer = 50, tol::Real = 1e-9)
    p, n = size(Y)
    u = zeros(n_block)
    local cholH
    for _ in 1:maxiter
        a = u[leaf_pos]                              # length p species effects
        η = _clamp_eta.(β .+ a)
        μ = _clamp_mu.(Ref(family), linkinv.(Ref(link), η))
        me = mu_eta.(Ref(link), η)
        s_tot = zeros(p); W_tot = zeros(p)
        @inbounds for t in 1:p, s in 1:n
            s_tot[t] += _glm_score(family, μ[t], N[t, s], me[t], Y[t, s])
            W_tot[t] += _glm_weight(family, μ[t], N[t, s], me[t])
        end
        sv = zeros(n_block); sv[leaf_pos] .= s_tot
        grad = sv .- Q * u
        H = Q + sparse(leaf_pos, leaf_pos, W_tot, n_block, n_block)
        try
            cholH = cholesky(Symmetric(H))
        catch
            return nothing, nothing
        end
        Δ = cholH \ grad
        all(isfinite, Δ) || return nothing, nothing
        u .+= Δ
        maximum(abs, Δ) < tol && break
    end
    return u, cholH
end

"""
    phylo_glm_marginal_loglik(family, Y, N, β, σ²_phy, phy; link, maxiter=50, tol=1e-9)

Laplace marginal log-likelihood of a phylogenetic GLLVM with a per-species
phylogenetic random intercept (`η_{ts} = β_t + a_t`, `a ~ N(0, σ²_phy Σ_phy)`),
evaluated over the augmented sparse precision of `phy::AugmentedPhy`. `Y`, `N` are
p×n response / trial-count matrices (p = `phy.n_leaves`), `β` the length-p intercepts,
`σ²_phy` the phylogenetic variance, `family` a `Distributions` marker, `link` a `Link`.

Returns `ℓ(û) − ½ûᵀQû + ½logdet Q − ½logdet H` with `Q = Q_cond/σ²_phy` and the joint
mode `û` over the tree nodes. Sparse CHOLMOD throughout; `-Inf` on a non-SPD step.
"""
function phylo_glm_marginal_loglik(family, Y::AbstractMatrix, N::AbstractMatrix,
        β::AbstractVector, σ²_phy::Real, phy::AugmentedPhy;
        link::Link = default_link(family),
        # Paired with `_phylo_gamma_xlv_marginal_loglik`:
        # `test/test_phylo_gamma_xlv.jl:116` pins the two equal at atol 1e-7, so
        # the two kernels must take the SAME curvature. Both now do — this one
        # via the family default, its partner via the same call. Briefly pinned
        # to `:fisher` while only this side had the contract; unpinned once the
        # partner moved, which is the "they must move together" constraint the
        # audit identified.
        hessian::Symbol = _default_hessian(family, link),
        maxiter::Integer = 50, tol::Real = 1e-9)
    (hessian === :fisher || hessian === :observed) || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    p, n = size(Y)
    p == phy.n_leaves || throw(ArgumentError("size(Y,1)=$p must equal phy.n_leaves=$(phy.n_leaves)"))
    σ²_phy > 0 || return -Inf
    Q_cond, leaf_pos = _phylo_qcond(phy)
    n_block = size(Q_cond, 1)
    Q = Q_cond ./ float(σ²_phy)
    local cholQ
    try
        cholQ = cholesky(Symmetric(Q))
    catch
        return -Inf
    end
    û, cholH = _phylo_glm_mode(family, Y, N, β, link, Q, leaf_pos, n_block;
                               maxiter = maxiter, tol = tol)
    û === nothing && return -Inf

    a = û[leaf_pos]
    η = _clamp_eta.(β .+ a)
    μ = _clamp_mu.(Ref(family), linkinv.(Ref(link), η))
    ℓ = 0.0
    @inbounds for t in 1:p, s in 1:n
        ℓ += _glm_logpdf(family, μ[t], N[t, s], Y[t, s])
    end
    quad = 0.5 * dot(û, Q * û)

    # Role separation. `_phylo_glm_mode` returns the Cholesky of the FISHER
    # Hessian from its final Newton step, and that same factorisation was reused
    # for the log-det — one object, two roles, which is the defect this
    # programme exists to fix. The Newton step wants Fisher (SPD by
    # construction); the log-det must be observed to match TMB.
    #
    # On the observed branch, rebuild H = Q + diag(W_obs) at the ALREADY-
    # CONVERGED mode and factorise separately: one extra Cholesky, on that
    # branch only, with the mode search untouched. Same treatment as
    # `coevolution_glm.jl`.
    cholL = cholH
    if hessian === :observed && !_glm_weight_matches_observed(family, link)
        me_h = mu_eta.(Ref(link), η)
        Wobs = zeros(p)
        @inbounds for t in 1:p, s in 1:n
            Wobs[t] += _glm_obs_weight(family, μ[t], N[t, s], me_h[t], Y[t, s], link, η[t])
        end
        Hobs = Q + sparse(leaf_pos, leaf_pos, Wobs, n_block, n_block)
        # The observed weight may be negative, so H is no longer SPD by
        # construction: fail to -Inf rather than return a meaningless log-det.
        cholL = try
            cholesky(Symmetric(Hobs))
        catch
            return -Inf
        end
    end
    return ℓ - quad + 0.5 * logdet(cholQ) - 0.5 * logdet(cholL)
end

# ---------------------------------------------------------------------------
# Fit driver.
# ---------------------------------------------------------------------------

"""
    PhyloGLMFit

Result of [`fit_phylo_glm`](@ref): per-species intercepts `β` (length p), the
phylogenetic variance `σ²_phy`, the estimated `dispersion` (σ² / r for the
dispersion families, `NaN` otherwise), the `link` and `family`, the maximised
Laplace `loglik`, the optimiser `converged` flag and `iterations`.
"""
struct PhyloGLMFit
    β::Vector{Float64}
    σ²_phy::Float64
    dispersion::Float64
    link::Link
    family::Any
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::PhyloGLMFit)
    print(io, "PhyloGLMFit(p=", length(f.β), ", family=", nameof(typeof(f.family)),
          ", σ²_phy=", round(f.σ²_phy; sigdigits = 4),
          isnan(f.dispersion) ? "" : ", dispersion=$(round(f.dispersion; sigdigits = 4))",
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_phylo_glm(Y, phy; family=Poisson(), link=default_link(family), N=nothing,
                  σ²_phy_init=1.0, g_tol=1e-5, iterations=300,
                  newton_maxiter=50, newton_tol=1e-9) -> PhyloGLMFit

Fit a phylogenetic GLLVM (per-species phylo random intercept) by L-BFGS on the
augmented-state Laplace marginal ([`phylo_glm_marginal_loglik`](@ref)). `Y` is p×n
(species × sites, p = `phy.n_leaves`); fits intercepts `β`, the phylogenetic variance
`σ²_phy` (on the log scale), and a dispersion parameter for the dispersion families.
Finite-difference outer gradient (the sparse-Cholesky marginal is not AD-friendly);
warm start from per-species link-scale means.
"""
function fit_phylo_glm(Y::AbstractMatrix, phy::AugmentedPhy;
        family = Poisson(), link::Link = default_link(family),
        N = nothing, σ²_phy_init::Real = 1.0,
        g_tol::Real = 1e-5, iterations::Integer = 300,
        newton_maxiter::Integer = 50, newton_tol::Real = 1e-9)
    p, n = size(Y)
    p == phy.n_leaves || throw(ArgumentError("size(Y,1)=$p must equal phy.n_leaves=$(phy.n_leaves)"))
    Ntr = N === nothing ? ones(eltype(Y), p, n) : N
    _check_family_response_support(family, Y, nothing; caller = "fit_phylo_glm", N = Ntr)
    nd = _spde_disp_len(family)

    β0 = [sum(linkfun(link, _ws_mean(family, Y[t, s], Ntr[t, s])) for s in 1:n) / n for t in 1:p]
    pbase = p + 1
    θ0 = vcat(β0, log(float(σ²_phy_init)), _spde_disp_init(family, Y))

    function negll(θ)
        β = θ[1:p]; σ²_phy = exp(θ[p + 1])
        fam = _spde_make_family(family, view(θ, (pbase + 1):(pbase + nd)))
        v = try
            -phylo_glm_marginal_loglik(fam, Y, Ntr, β, σ²_phy, phy;
                                       link = link, maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end

    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls,
                         Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    σ²_phŷ = exp(θ̂[p + 1])
    disp = _spde_disp_value(family, θ̂[(pbase + 1):(pbase + nd)])
    return PhyloGLMFit(β̂, σ²_phŷ, disp, link, family, _fit_verdict(res)...)
end

