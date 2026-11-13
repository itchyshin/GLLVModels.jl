# Random-effects fitters for the Gaussian GLLVM — grouped random SLOPES (random
# regression / behavioural-syndromes model).
#
# A grouped random slope `(Z | g)` adds a per-group coefficient vector b_g ~ N(0, Σ_b)
# (q-dim) with a site design Z (q columns; col 1 = ones for the intercept, further cols
# = covariates). Its contribution to η[t,i] (shared across ALL traits t) is
# Z[i,:] · b_{g(i)}. With an UNSTRUCTURED (correlated) q×q Σ_b this is the
# random-intercept-plus-slopes behavioural-syndromes model.
#
# Marginal over the stacked y: Σ = I_n⊗A + W Σ Wᵀ with A = ΛΛ' + σ_eps²I,
# W[(i−1)p+t,(g−1)q+k] = [g(i)==g]·Z[i,k], and Σ = I_L⊗Σ_b (iid groups). The Woodbury
# correction is BLOCK-DIAGONAL per group, so each group contributes one q×q solve on a
# single shared chol of the Woodbury core of A. Reduces to the grouped intercept at
# q=1, Z=ones, Σ_b=[σ_u²]. AD-clean (verified against a central FD gradient).

# ---------------------------------------------------------------------------
# Grouped random intercept — the q=1 anchor for the slope reduction. r_g ~ N(0, σ_u²)
# shared by all sites in group g ⇒ cross-site correlation WITHIN a group. The marginal
# over the stacked y has covariance Σ = kron(I_n, A) + kron(G, B), A = ΛΛ' + σ_eps²I,
# B = σ_u²·1ₚ1ₚᵀ, G[i,j] = 1{g(i)=g(j)}. Groups are independent, so ℓ = Σ_g ℓ_g where
# each group block is I_{n_g}⊗A + J_{n_g}⊗B — solved by the SAME rotation trick as the
# phylo path:
#   y_g'Σ_g⁻¹y_g = n_g·m_g'(A+n_g B)⁻¹m_g + tr(Y_gc'A⁻¹Y_gc),
#   logdet Σ_g   = logdet(A+n_g B) + (n_g−1)logdet(A).
# B is rank-1 ⇒ (A+n_g B)⁻¹ and logdet(A+n_g B) come from ONE chol(A) via
# Sherman-Morrison + the matrix-determinant lemma (v = 1ₚ). Cost O(p³ + L·p²).
# ---------------------------------------------------------------------------

function _grouped_intercept_loglik(y::AbstractMatrix, group_idx::Vector{Vector{Int}},
        Λ_B::AbstractMatrix, σ_eps::Real, σ_u::Real)
    p, n = size(y)
    T = promote_type(eltype(y), eltype(Λ_B), typeof(σ_eps), typeof(σ_u))
    σ² = σ_eps^2; σu² = σ_u^2
    K = size(Λ_B, 2)
    # Woodbury form of A = ΛΛ' + σ²I — factor the well-conditioned K×K core, NOT the
    # p×p A, so it stays robust as σ_eps → 0 (a direct chol(A) goes singular there):
    #   A⁻¹V = (V − Λ(σ²I+Λ'Λ)⁻¹Λ'V)/σ²,  logdet A = (p−K)logσ² + logdet(σ²I+Λ'Λ).
    Kc = Λ_B' * Λ_B
    @inbounds for k in 1:K
        Kc[k, k] += σ²
    end
    cKc = cholesky(Symmetric(Kc))
    logdetA = (p - K) * log(σ²) + logdet(cKc)
    Ainv = V -> (V .- Λ_B * (cKc \ (Λ_B' * V))) ./ σ²
    onep = ones(T, p)
    Ainv_1 = Ainv(onep)
    vAv = dot(onep, Ainv_1)
    twopi = convert(T, 2π)
    ll = zero(T)
    for idx in group_idx
        ng = length(idx)
        Yg = y[:, idx]                                   # p × ng
        mg = vec(sum(Yg, dims = 2)) ./ ng
        Ygc = Yg .- reshape(mg, p, 1)
        quad_centered = sum(Ygc .* Ainv(Ygc))            # tr(Y_gc'A⁻¹Y_gc)
        Ainv_mg = Ainv(mg)
        smcoef = (ng * σu²) / (1 + ng * σu² * vAv)
        AnB_inv_mg = Ainv_mg .- (smcoef * dot(onep, Ainv_mg)) .* Ainv_1
        quad_mean = ng * dot(mg, AnB_inv_mg)
        logdet_g = ng * logdetA + log(1 + ng * σu² * vAv)
        ll += -convert(T, 0.5) * (ng * p * log(twopi) + logdet_g + quad_mean + quad_centered)
    end
    return ll
end

"""
    gaussian_grouped_intercept_loglik(y, grouping, Λ_B, σ_eps, σ_u) -> Real

Marginal log-likelihood of a Gaussian GLLVM with a grouped random intercept: a shared
effect r_g ~ N(0, σ_u²) for every site in group g (cross-site correlation within a
group). `y` is p×n; `grouping` a length-n vector assigning each site to a group. Solved
per group by the rotation trick + a rank-1 Sherman-Morrison on one shared chol(ΛΛ'+σ²I).
Serves as the q=1 anchor for [`fit_gaussian_random_slope`](@ref).
"""
function gaussian_grouped_intercept_loglik(y::AbstractMatrix, grouping::AbstractVector,
        Λ_B::AbstractMatrix, σ_eps::Real, σ_u::Real)
    codes, _ = _code_grouping(grouping)
    L = maximum(codes)
    group_idx = [findall(==(g), codes) for g in 1:L]
    return _grouped_intercept_loglik(y, group_idx, Λ_B, σ_eps, σ_u)
end

# ---------------------------------------------------------------------------
# Grouped random SLOPES (x|g): per-group coefficient vector b_g ~ N(0, Σ_b) (q-dim) with
# a site design Z (q columns; col 1 = ones for the intercept, further cols = covariates).
# Contribution to eta[t,i] (all traits) = Z[i,:] . b_{g(i)}. Marginal = I_n⊗A + W Σ Wᵀ
# with W[(i−1)p+t,(g−1)q+k] = [g(i)==g]·Z[i,k] and Σ = I_L⊗Σ_b (iid groups). Woodbury
# stays BLOCK-DIAGONAL per group: core_g = Σ_b⁻¹ + vAv·(Z_g'Z_g),
# s_g = Σ_{i in g} Z[i,:]·(1ₚ'A⁻¹y_i). Reduces to the grouped intercept at q=1, Z=ones,
# Σ_b=[σ_u²]. AD-clean.
# ---------------------------------------------------------------------------

# log-Cholesky packing (length q(q+1)/2, column-major lower-tri, log-diagonal) -> q×q SPD.
function _unpack_chol_cov(theta::AbstractVector, q::Integer)
    T = eltype(theta)
    Lb = zeros(T, q, q)
    idx = 1
    @inbounds for j in 1:q
        Lb[j, j] = exp(theta[idx]); idx += 1
        for i in (j + 1):q
            Lb[i, j] = theta[idx]; idx += 1
        end
    end
    return Lb * Lb', Lb
end
_chol_cov_npar(q::Integer) = q * (q + 1) ÷ 2

function _grouped_slope_loglik(y::AbstractMatrix, group_idx::Vector{Vector{Int}},
        Z::AbstractMatrix, Λ_B::AbstractMatrix, σ_eps::Real, Σ_b::AbstractMatrix)
    p, n = size(y); K = size(Λ_B, 2); q = size(Z, 2)
    T = promote_type(eltype(y), eltype(Λ_B), typeof(σ_eps^2), eltype(Σ_b), eltype(Z))
    σ² = convert(T, σ_eps^2)
    Kc = Λ_B' * Λ_B
    @inbounds for k in 1:K
        Kc[k, k] += σ²
    end
    cKc = cholesky(Symmetric(Kc))
    logdetA = (p - K) * log(σ²) + logdet(cKc)
    Ainv = V -> (V .- Λ_B * (cKc \ (Λ_B' * V))) ./ σ²
    onep = ones(T, p); Ainv_1 = Ainv(onep); vAv = dot(onep, Ainv_1)
    cSb = cholesky(Symmetric(Matrix{T}(Σ_b)))
    Sb_inv = cSb \ Matrix{T}(I, q, q)
    logdetSb = logdet(cSb)
    L = length(group_idx)
    quad = zero(T); logdet_corr = zero(T)
    for idx in group_idx
        Yg = Matrix{T}(@view y[:, idx])
        quad += sum(Yg .* Ainv(Yg))
        w = vec(Ainv_1' * Yg)                 # 1ₚ'A⁻¹y_i per site in the group
        Zg = Matrix{T}(@view Z[idx, :])
        s_g = Zg' * w                         # q-vector
        Mwg = Sb_inv + vAv .* (Zg' * Zg)      # q×q Woodbury block
        cMwg = cholesky(Symmetric(Mwg))
        quad -= dot(s_g, cMwg \ s_g)
        logdet_corr += logdet(cMwg)
    end
    np = p * n
    logdet_full = n * logdetA + L * logdetSb + logdet_corr
    return -convert(T, 0.5) * (np * log(convert(T, 2π)) + logdet_full + quad)
end

"""
    GaussianRandomSlopeFit

Result of [`fit_gaussian_random_slope`](@ref): loadings `Λ` (p×K), residual SD `σ_eps`, the
q×q random-effect covariance `Σ_b` (correlated random slopes; `Σ_b[1,1]` is the random-
intercept variance when col 1 of `Z` is the intercept), the number of groups `nlevels`, the
design width `q`, the maximised `loglik`, `converged`, `iterations`.
"""
struct GaussianRandomSlopeFit
    Λ::Matrix{Float64}
    σ_eps::Float64
    Σ_b::Matrix{Float64}
    nlevels::Int
    q::Int
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::GaussianRandomSlopeFit)
    p, K = size(f.Λ)
    print(io, "GaussianRandomSlopeFit(p=", p, ", K=", K, ", nlevels=", f.nlevels, ", q=", f.q,
          ", σ_eps=", round(f.σ_eps; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7), f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_gaussian_random_slope(y, grouping, Z; K, σ_b_init=0.3, …) -> GaussianRandomSlopeFit

Fit a Gaussian GLLVM with grouped random slopes `(Z | grouping)` — a per-group coefficient
vector `b_g ~ N(0, Σ_b)` with an UNSTRUCTURED (correlated) q×q `Σ_b` (the behavioural-
syndromes random-regression model: a correlated random intercept + slopes per group). `Z`
is the `n×q` site design (col 1 conventionally all-ones for the random intercept; further
cols are covariates). Optimises `[vec(Λ); log σ_eps; logCholesky(Σ_b)]` on the per-group
block-Woodbury marginal (direct ForwardDiff; guarded BackTracking line search). Groups
are iid.

At q=1 with `Z = ones(n, 1)` the marginal reduces (rtol 1e-8) to
[`gaussian_grouped_intercept_loglik`](@ref). Random-SLOPE variances at q≥2 are
boundary-prone (they can collapse into the residual); the optimiser stays on the
log-Cholesky parameterisation so `Σ_b` is always returned SPD, but report convergence
honestly near the boundary.
"""
function fit_gaussian_random_slope(y::AbstractMatrix, grouping::AbstractVector,
        Z::AbstractMatrix; K::Integer, σ_b_init::Real = 0.3,
        g_tol::Real = 1e-6, iterations::Integer = 500)
    _refuse_na_response("fit_gaussian_random_slope", y)
    p, n = size(y); q = size(Z, 2)
    (length(grouping) == n && size(Z, 1) == n) ||
        throw(DimensionMismatch("grouping and Z rows must be n=$n"))
    K ≥ 1 || throw(ArgumentError("K must be ≥ 1; got $K"))
    codes, _ = _code_grouping(grouping)
    L = maximum(codes)
    group_idx = [findall(==(g), codes) for g in 1:L]
    rr = rr_theta_len(p, K)
    yf = Matrix{Float64}(y); Zf = Matrix{Float64}(Z)
    Λ0, σ0 = ppca_init(yf, K)
    chol0 = zeros(_chol_cov_npar(q))                 # log-Cholesky init: diag = log σ_b_init
    let idx = 1
        for j in 1:q
            chol0[idx] = log(float(σ_b_init)); idx += 1
            idx += (q - j)
        end
    end
    θ0 = vcat(pack_lambda(Λ0), log(σ0), chol0)
    nc = _chol_cov_npar(q)
    # Guarded objective: as σ_eps → 0 or Σ_b → boundary the Woodbury chols can fail or
    # the value can blow up. Returning a large finite penalty (matching main's RE fit
    # machinery, e.g. fit_row_random_gllvm) keeps the line search from wandering off into
    # a degenerate ridge. BackTracking (order 3) is main's robust default for these fits.
    nll = θ -> begin
        v = try
            Λ = unpack_lambda(θ[1:rr], p, K); σe = exp(θ[rr + 1])
            Σ_b, _ = _unpack_chol_cov(θ[(rr + 2):(rr + 1 + nc)], q)
            -_grouped_slope_loglik(yf, group_idx, Zf, Λ, σe, Σ_b)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(nll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations); autodiff = :forward)
    th = Optim.minimizer(res)
    Σ_b̂, _ = _unpack_chol_cov(th[(rr + 2):(rr + 1 + nc)], q)
    return GaussianRandomSlopeFit(unpack_lambda(th[1:rr], p, K), exp(th[rr + 1]), Matrix{Float64}(Σ_b̂),
                                  L, q, _fit_verdict(res)...)
end
