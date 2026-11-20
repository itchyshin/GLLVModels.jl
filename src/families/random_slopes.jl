# NON-GAUSSIAN grouped random SLOPES (random regression / behavioural-syndromes
# model) for the Laplace families — stage 2 of the slopes track (Poisson first).
#
# A grouped random slope `(Z | g)` adds a per-group coefficient vector b_g ~ N(0, Σ_b)
# (q-dim) with a site design Z (q columns; col 1 = ones for the intercept, further cols
# = covariates). Its contribution to η[t,i] (shared across ALL species t) is
# Z[i,:]·b_{g(i)}. With an UNSTRUCTURED (correlated) q×q Σ_b this is the
# random-intercept-plus-slopes behavioural-syndromes model on a non-Gaussian response.
#
# KEY REPARAMETERIZATION (reuses the family-generic Laplace verbatim — no new mode
# math). The Gaussian path (src/fit_random_effects.jl) integrates b_g in closed form
# via a per-group block-Woodbury; the non-Gaussian path has no closed form, so each
# GROUP is integrated by Laplace. Whiten the slopes b_g = L_b c_g with c_g ~ N(0, I_q)
# (L_b = chol(Σ_b)). For group g with sites S_g (|S_g| = n_g) stack the responses of
# all its sites into ONE "super-site" of p̃ = n_g·p responses and the latent block
#
#     w_g = [vec(z_{S_g}); c_g] ~ N(0, I_m),  m = n_g·K + q,
#
# z_i ~ N(0, I_K) the per-site ordination scores. The super-site linear predictor for
# (site j in the group, species t) is
#
#     η = β_t + Λ[t,:]·z_{i_j} + (Z[i_j,:]·L_b)·c_g ,
#
# i.e. EXACTLY a standard GLLVM super-site at the augmented loadings Λ̃_g (p̃ × m):
#   • block j of K cols → Λ on rows of site j (0 elsewhere): the per-site z_i,
#   • last q cols       → (L_b' Z[i_j,:]) on ALL species rows of site j: the slopes.
# Its prior is the identity, so the family-generic mode-finder/marginal
# (`laplace_loglik_site`, A = Λ̃'WΛ̃ + I) absorb the extra slope columns with no
# change. Groups are iid ⇒ the total marginal is the sum over groups.
#
# Reduction (q=1, Z=ones, singleton groups) → a per-site random intercept, identical
# to `row_random_marginal_loglik_laplace` (verified to 0.0). Random-SLOPE variances at
# q≥2 are boundary-prone (σ→0, ρ→±1) and a per-site ordination LV partially confounds
# the random intercept; report convergence honestly near the boundary.

# Augmented per-group loadings Λ̃_g (p̃ × m), p̃ = n_g·p, m = n_g·K + q.
# AD-friendly element type; the slope columns carry L_b' Z[i,:] (q-vector) on every
# species row of the site, shared across species (the random slope is a community-wide
# per-site offset, exactly like a covariate's effect on η).
function _slope_augment_loadings(Λ::AbstractMatrix, Z::AbstractMatrix,
        Lb::AbstractMatrix, sites::AbstractVector{<:Integer})
    T = promote_type(eltype(Λ), eltype(Z), eltype(Lb))
    p, K = size(Λ)
    q = size(Lb, 1)
    ng = length(sites)
    m = ng * K + q
    Λ̃ = zeros(T, ng * p, m)
    @inbounds for (j, i) in enumerate(sites)
        Λ̃[((j - 1) * p + 1):(j * p), ((j - 1) * K + 1):(j * K)] .= Λ
        zb = Lb' * @view Z[i, :]                # q-vector, shared across species
        for t in 1:p, k in 1:q
            Λ̃[(j - 1) * p + t, ng * K + k] = zb[k]
        end
    end
    return Λ̃
end

"""
    random_slope_marginal_loglik_laplace(family, Y, N, Z, Λ, β, Lb, group_idx;
        link=default_link(family), maxiter=100, tol=1e-9) -> Float64

Total Laplace log-marginal over the groups of a non-Gaussian GLLVM with grouped random
slopes `(Z | g)`: per-group `b_g ~ N(0, Σ_b)` entering `η_{ts} = β_t + (Λ z_s)_t +
Z[s,:]·b_{g(s)}`. `Lb` is the lower-Cholesky factor of `Σ_b = Lb·Lbᵀ`; `Z` is the `n×q`
site design (col 1 conventionally all-ones for the random intercept); `group_idx` is a
vector of the site-index vectors per group (e.g. from `_code_grouping`).

Each group is integrated as ONE Laplace super-site of `n_g·p` responses at the augmented
loadings (the `b_g = L_b c_g` whitening), so it is a thin sum over the family-generic
[`laplace_loglik_site`](@ref). `Y`, `N` are the `p×n` response / trial-count matrices.
At q=1, `Z = ones`, singleton groups, the value reduces (to machine zero) to
[`row_random_marginal_loglik_laplace`](@ref).
"""
function random_slope_marginal_loglik_laplace(family, Y::AbstractMatrix, N::AbstractMatrix,
        Z::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector, Lb::AbstractMatrix,
        group_idx::AbstractVector{<:AbstractVector{<:Integer}};
        link::Link = default_link(family), maxiter::Integer = 100, tol::Real = 1e-9)
    p = size(Y, 1)
    acc = zero(promote_type(eltype(Λ), eltype(Lb), eltype(β), eltype(Z)))
    @inbounds for sites in group_idx
        ng = length(sites)
        Λ̃ = _slope_augment_loadings(Λ, Z, Lb, sites)
        ỹ = vec(@view Y[:, sites])             # stack sites: p̃ = ng·p
        ñ = vec(@view N[:, sites])
        β̃ = repeat(β, ng)
        acc += laplace_loglik_site(family, ỹ, ñ, Λ̃, β̃, link; maxiter = maxiter, tol = tol)
    end
    return acc
end

# ---------------------------------------------------------------------------
# Fit driver — Poisson first (stage 2). L-BFGS (finite-diff gradient; the Laplace
# inner mode-finder is not forward-AD-friendly) over
#   [β; pack_lambda(Λ); logCholesky(Σ_b)].
# ---------------------------------------------------------------------------

"""
    PoissonRandomSlopeFit

Result of [`fit_poisson_random_slope`](@ref): per-species intercepts `β` (length p),
ordination loadings `Λ` (p×K), the `q×q` random-effect covariance `Σ_b` (correlated
random slopes; `Σ_b[1,1]` is the random-intercept variance when col 1 of `Z` is the
intercept), the number of groups `nlevels`, the design width `q`, the `link`, the
maximised Laplace `loglik`, `converged`, and `iterations`.
"""
struct PoissonRandomSlopeFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    Σ_b::Matrix{Float64}
    nlevels::Int
    q::Int
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::PoissonRandomSlopeFit)
    p, K = size(f.Λ)
    print(io, "PoissonRandomSlopeFit(p=", p, ", K=", K, ", nlevels=", f.nlevels, ", q=", f.q,
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_poisson_random_slope(Y, grouping, Z; K, link=LogLink(), σ_b_init=0.4,
        g_tol=1e-5, iterations=500, newton_maxiter=100, newton_tol=1e-9)
        -> PoissonRandomSlopeFit

Fit a Poisson GLLVM with grouped random slopes `(Z | grouping)` — a per-group
coefficient vector `b_g ~ N(0, Σ_b)` with an UNSTRUCTURED (correlated) q×q `Σ_b` (the
behavioural-syndromes random-regression model: a correlated random intercept + slopes
per group). `η_{ts} = β_t + (Λ z_s)_t + Z[s,:]·b_{g(s)}`; `Y` is `p×n` (species × sites)
integer counts; `Z` the `n×q` site design (col 1 conventionally all-ones for the random
intercept, further cols covariates). Optimises `[β; pack_lambda(Λ); logCholesky(Σ_b)]`
by L-BFGS (finite-difference gradient) on the per-group Laplace super-site marginal
([`random_slope_marginal_loglik_laplace`](@ref)); groups are iid.

Supports `q = 1` (random intercept) and `q ≥ 2` (intercept + correlated slopes). At
`q = 1`, `Z = ones`, the model is a grouped Poisson random intercept. Random-SLOPE
variances at `q ≥ 2` are boundary-prone (variances → 0, ρ → ±1) and a per-site
ordination LV partially confounds the random intercept; the optimiser stays on the
log-Cholesky parameterisation so `Σ_b` is always returned SPD — but report convergence
honestly near the boundary.
"""
function fit_poisson_random_slope(Y::AbstractMatrix{<:Real}, grouping::AbstractVector,
        Z::AbstractMatrix; K::Integer, link::Link = LogLink(), σ_b_init::Real = 0.4,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    q = size(Z, 2)
    (length(grouping) == n && size(Z, 1) == n) ||
        throw(DimensionMismatch("grouping and Z rows must be n=$n"))
    K ≥ 1 || throw(ArgumentError("K must be ≥ 1; got $K"))
    σ_b_init > 0 || throw(ArgumentError("σ_b_init must be positive (Σ_b is fit on the log-Cholesky scale)"))
    codes, _ = _code_grouping(grouping)
    L = maximum(codes)
    group_idx = [findall(==(g), codes) for g in 1:L]
    rr = rr_theta_len(p, K)
    nc = _chol_cov_npar(q)
    Yc = Integer.(Y)
    N1 = ones(Int, p, n)
    Zf = Matrix{Float64}(Z)

    # warm start: per-species empirical link-scale means for β, an SVD (PPCA-style)
    # loadings init, and a small diagonal log-Cholesky Σ_b init.
    Zemp = _cov_zemp(Poisson(), Yc, N1, link)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    chol0 = zeros(nc)                          # log-Cholesky init: log-diag = log σ_b_init
    let idx = 1
        for j in 1:q
            chol0[idx] = log(float(σ_b_init)); idx += 1
            idx += (q - j)
        end
    end
    θ0 = vcat(β0, _pack_lambda_start(Λ0), chol0)

    # Guarded objective: as Σ_b → boundary the augmented Woodbury chols can fail; a
    # large finite penalty keeps the line search off the degenerate ridge (matching
    # the Gaussian slope fitter and the other RE drivers).
    negll = θ -> begin
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        v = try
            _, Lb = _unpack_chol_cov(θ[(p + rr + 1):(p + rr + nc)], q)
            -random_slope_marginal_loglik_laplace(Poisson(), Yc, N1, Zf, Λ, β, Lb, group_idx;
                                                  link = link, maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    th = Optim.minimizer(res)
    β̂ = th[1:p]
    Λ̂ = unpack_lambda(th[(p + 1):(p + rr)], p, K)
    Σ_b̂, _ = _unpack_chol_cov(th[(p + rr + 1):(p + rr + nc)], q)
    return PoissonRandomSlopeFit(β̂, Λ̂, Matrix{Float64}(Σ_b̂), L, q, link,
                                 _fit_verdict(res)...)
end
