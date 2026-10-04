# Missing-predictor FIML for the Gaussian GLLVM — the mi() axis, Phase-2a slice.
#
# A single site-level continuous predictor x (one value per site, possibly
# `missing`) enters the response mean with a slope b_x broadcast across all
# traits (gllvmTMB's mi() unit-level semantic). x is given a covariate model
# x_s ~ N(μ_x + Z_s·γ, σ_x²) — an optional matrix Z of auxiliary site-level
# predictors (the design's "explicit covariate model"; Z = nothing gives the
# intercept-only model x_s ~ N(μ_x, σ_x²)). The missing x_s are integrated out
# in CLOSED FORM, because (y_s, x_s) is jointly Gaussian — full-information
# maximum likelihood, NOT impute-then-analyse.
#
# Per site s, with η_s ~ N(0, I_K), ε_s ~ N(0, σ_eps² I_p), m_x = μ_x + Z_s·γ:
#   x_s = m_x + e_x,           e_x ~ N(0, σ_x²)
#   y_s = a + b_x x_s 1_p + Λ_B η_s + ε_s
# so the joint of (y_s, x_s) is Gaussian. FIML over observed cells:
#   • x_s observed → log N(y_s | x_s) + log N(x_s)         (condition on x)
#       with Cov(y_s | x_s) = Λ_B Λ_Bᵀ + σ_eps² I
#   • x_s missing  → log N(y_s) marginal                   (integrate out x)
#       with Cov(y_s) = Λ_B Λ_Bᵀ + σ_eps² I + b_x² σ_x² 11ᵀ
#                     = Λ_aug Λ_augᵀ + σ_eps² I,  Λ_aug = [Λ_B | b_x σ_x 1_p]
# Both per-site densities are `low-rank + σ²I`, so one Woodbury kernel serves
# both (rank K observed, rank K+1 missing). No Laplace, no formula parser.
#
# Closed form is exact for Gaussian y + Gaussian x; b_x and σ_x are jointly
# identified because x is observed at the non-missing sites. Non-Gaussian
# responses or discrete/structured missing predictors need the Laplace
# augmented-latent path (out of scope for this slice).
#
# Reference: gllvmTMB mi() Phase 2a (continuous site-level predictor);
# Little & Rubin 2002 (FIML for missing data).

using LinearAlgebra

# 0.5 (p log 2π + logdet Σ + rᵀ Σ⁻¹ r) for Σ = L Lᵀ + s2 I_p, L is p×m, via the
# matrix-determinant lemma + Woodbury (only an m×m factorisation; m = K or K+1).
function _mi_lowrank_halfnll(r::AbstractVector, L::AbstractMatrix, s2::Real)
    p, m = size(L)
    M = (L' * L) ./ s2 + I            # m×m
    cholM = cholesky(Symmetric(M); check = false)
    # an extreme line-search trial point can make M numerically indefinite: report +Inf so the
    # line search backs off, instead of throwing PosDefException out of the fit
    issuccess(cholM) || return convert(promote_type(eltype(r), eltype(L), typeof(s2)), Inf)
    logdetΣ = p * log(s2) + logdet(cholM)
    Ltr = L' * r
    w = cholM \ (Ltr ./ s2)           # M⁻¹ (Lᵀ r / s2)
    Σinv_r = (r .- L * w) ./ s2
    return 0.5 * (p * log(2π) + logdetΣ + dot(r, Σinv_r))
end

# Σ⁻¹ r for Σ = L Lᵀ + s2 I (concrete post-fit use, e.g. EBLUPs).
function _mi_lowrank_solve(r::AbstractVector, L::AbstractMatrix, s2::Real)
    M = (L' * L) ./ s2 + I
    w = cholesky(Symmetric(M)) \ (L' * r ./ s2)
    return (r .- L * w) ./ s2
end

# Param layout: [a (p), b_x (1), μ_x (1), γ (q_z), log_σx (1), log_σeps (1), vec(Λ) (p*K)]
# `oidx` is `nothing` (every response observed) or, per site, the indices of the observed traits;
# a site's unobserved responses then contribute nothing (gllvmTMB's response = "include").
function _mi_fiml_nll(params, y, xobs, isobs, Z, p::Int, n::Int, K::Int, q_z::Int,
                      oidx::Union{Nothing,Vector{Vector{Int}}} = nothing)
    T = eltype(params)
    a = @view params[1:p]
    b_x = params[p + 1]
    μ_x = params[p + 2]
    γ = q_z > 0 ? (@view params[(p + 3):(p + 2 + q_z)]) : nothing
    base = p + 2 + q_z
    log_σx = params[base + 1]
    σ_x2 = exp(2 * log_σx)
    σ_eps2 = exp(2 * params[base + 2])
    Λ = reshape(@view(params[(base + 3):(base + 2 + p * K)]), p, K)
    Λ_aug = hcat(Λ, (b_x * exp(log_σx)) .* ones(T, p))

    nll = zero(T)
    @inbounds for s in 1:n
        m_x = q_z > 0 ? (μ_x + dot(@view(Z[s, :]), γ)) : μ_x
        if oidx === nothing || length(oidx[s]) == p
            ys = @view y[:, s]
            if isobs[s]
                r = ys .- (a .+ b_x * xobs[s])
                nll += _mi_lowrank_halfnll(r, Λ, σ_eps2)
            else
                r = ys .- (a .+ b_x * m_x)
                nll += _mi_lowrank_halfnll(r, Λ_aug, σ_eps2)
            end
        elseif !isempty(oidx[s])
            o = oidx[s]
            ys = @view y[o, s]
            ao = @view a[o]
            if isobs[s]
                r = ys .- (ao .+ b_x * xobs[s])
                nll += _mi_lowrank_halfnll(r, Λ[o, :], σ_eps2)
            else
                r = ys .- (ao .+ b_x * m_x)
                nll += _mi_lowrank_halfnll(r, Λ_aug[o, :], σ_eps2)
            end
        end
        if isobs[s]
            nll += 0.5 * (log(2π) + 2 * log_σx + (xobs[s] - m_x)^2 / σ_x2)
        end
    end
    return nll
end

# Backward-compatible intercept-only form (no Z); same param layout as q_z = 0.
_mi_fiml_nll(params, y, xobs, isobs, p::Int, n::Int, K::Int) =
    _mi_fiml_nll(params, y, xobs, isobs, nothing, p, n, K, 0)

"""
    fit_gaussian_mi_fiml(y, x; K, Z=nothing, g_tol=1e-8, iterations=1000)
        -> NamedTuple

Fit a Gaussian GLLVM with one site-level continuous predictor `x` whose missing
entries are integrated out by full-information maximum likelihood. `y` is `p × n`
(traits × sites); `x` is length `n` (one value per site) and may contain
`missing` or `NaN`. `y` may also contain `missing` or `NaN` cells: those responses contribute
nothing and the rest of each site enters the observed-data likelihood, as gllvmTMB's
`miss_control(response = "include", predictor = "model")` (a site with no observed response
contributes only its covariate density when `x` is observed there, and nothing otherwise). Every
trait needs at least one observed response. The predictor enters the response mean with a single slope
`b_x` broadcast across all traits, and is modelled as `x ~ N(μ_x + Z·γ, σ_x²)`,
where `Z` is an optional `n × q` matrix of auxiliary site-level predictors for
the covariate (imputation) model (`Z = nothing` ⇒ intercept-only).

Returns a NamedTuple with `b_x`, the per-trait intercepts `a`, covariate-model
params `μ_x`, `γ` (length `q`), `σ_x`, residual `σ_eps`, loadings `Λ` (`p × K`),
the conditional modes `eblup_x` (length `n`; observed values at observed sites,
`E[x_s | y_s]` at missing sites, given the site's observed responses), `logLik`, `converged`,
`n_missing` (missing `x` entries) and `n_missing_y` (missing response cells).

Faithful to gllvmTMB's `mi()` Phase 2a (continuous site-level predictor). The
closed-form FIML is exact for the all-Gaussian case — no imputation, no Laplace.
"""
function fit_gaussian_mi_fiml(y::AbstractMatrix, x::AbstractVector; K::Integer,
                              Z::Union{Nothing,AbstractMatrix} = nothing,
                              g_tol::Real = 1e-8, iterations::Integer = 1000)
    p, n = size(y)
    yobs = [!(ismissing(v) || (v isa Real && isnan(v))) for v in y]
    all(any(view(yobs, t, :)) for t in 1:p) ||
        throw(ArgumentError("every trait (row of y) needs at least one observed response."))
    n_missing_y = count(!, yobs)
    oidx = n_missing_y == 0 ? nothing : [findall(view(yobs, :, s)) for s in 1:n]
    # a complete y is used as given (its element type kept); masked cells are stored as 0
    n_missing_y == 0 || (y = [yobs[t, s] ? Float64(y[t, s]) : 0.0 for t in 1:p, s in 1:n])
    length(x) == n ||
        throw(ArgumentError("length(x) = $(length(x)) must equal n_sites = $n."))
    K ≥ 1 || throw(ArgumentError("K must be ≥ 1."))
    q_z = Z === nothing ? 0 : size(Z, 2)
    if Z !== nothing
        size(Z, 1) == n ||
            throw(ArgumentError("Z must have n_sites = $n rows; got $(size(Z, 1))."))
        any(ismissing, Z) && throw(ArgumentError("Z must be fully observed."))
    end
    Zf = Z === nothing ? nothing : Matrix{Float64}(Z)

    isobs = [!(ismissing(xi) || (xi isa Real && isnan(xi))) for xi in x]
    any(isobs) || throw(ArgumentError("x has no observed values."))
    xobs = [isobs[s] ? Float64(x[s]) : 0.0 for s in 1:n]
    n_missing = count(!, isobs)

    # warm start
    a0 = n_missing_y == 0 ? vec(Statistics.mean(y, dims = 2)) :
         vec(sum(y, dims = 2)) ./ vec(sum(yobs, dims = 2))   # observed means (masked cells are 0)
    obs = findall(isobs)
    xo = xobs[obs]
    if q_z > 0
        D = hcat(ones(length(obs)), Zf[obs, :])
        coef = D \ xo
        μ_x0 = coef[1]
        γ0 = coef[2:end]
        σ_x0 = max(Statistics.std(xo .- D * coef), 1e-3)
    else
        μ_x0 = Statistics.mean(xo)
        γ0 = Float64[]
        σ_x0 = max(Statistics.std(xo), 1e-3)
    end
    if n_missing_y == 0
        ybar = vec(Statistics.mean(y .- a0, dims = 1))
        Yc = y .- a0
    else
        Yc = (y .- a0) .* yobs                                  # masked cells at the trait mean
        ybar = vec(sum(Yc, dims = 1)) ./ max.(vec(sum(yobs, dims = 1)), 1)
    end
    b_x0 = let xc = xo .- Statistics.mean(xo)
        sum(xc .* ybar[obs]) / max(sum(abs2, xc), 1e-8)
    end
    C = Symmetric((Yc * Yc') ./ n)
    E = eigen(C)
    idx = sortperm(E.values, rev = true)[1:K]
    Λ0 = E.vectors[:, idx] .* sqrt.(max.(E.values[idx] .- 1e-2, 1e-2))'
    σ_eps0 = sqrt(max(Statistics.mean(E.values[1:max(1, p - K)]), 1e-2))

    params0 = vcat(a0, b_x0, μ_x0, γ0, log(σ_x0), log(σ_eps0), vec(Λ0))
    nll(θ) = _mi_fiml_nll(θ, y, xobs, isobs, Zf, p, n, K, q_z, oidx)

    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    # With missing responses the default HagerZhang first step can land at an extreme point
    # (log σ_eps near -100); a backtracking line search starts from the full step and only
    # shrinks it. The complete-y path keeps the default line search.
    method = n_missing_y == 0 ? Optim.LBFGS() :
             Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking())
    res = Optim.optimize(nll, params0, method, opts; autodiff = :forward)
    θ = Optim.minimizer(res)

    a = θ[1:p]
    b_x = θ[p + 1]
    μ_x = θ[p + 2]
    γ = q_z > 0 ? θ[(p + 3):(p + 2 + q_z)] : Float64[]
    base = p + 2 + q_z
    σ_x = exp(θ[base + 1])
    σ_eps = exp(θ[base + 2])
    Λ = reshape(θ[(base + 3):(base + 2 + p * K)], p, K)

    # EBLUPs: observed value, or E[x_s | y_s] at missing sites.
    Λ_aug = hcat(Λ, (b_x * σ_x) .* ones(p))
    eblup = Vector{Float64}(undef, n)
    for s in 1:n
        if isobs[s]
            eblup[s] = xobs[s]
        else
            m_x = q_z > 0 ? (μ_x + dot(view(Zf, s, :), γ)) : μ_x
            o = oidx === nothing ? (1:p) : oidx[s]
            if isempty(o)
                eblup[s] = m_x          # no observed response: the covariate-model mean
            else
                r = y[o, s] .- (a[o] .+ b_x * m_x)
                eblup[s] = m_x + b_x * σ_x^2 * sum(_mi_lowrank_solve(r, Λ_aug[o, :], σ_eps^2))
            end
        end
    end

    _ll, _cv, _it = _fit_verdict(res)
    return (b_x = b_x, a = a, μ_x = μ_x, γ = γ, σ_x = σ_x, σ_eps = σ_eps, Λ = Λ,
            eblup_x = eblup, logLik = _ll,
            converged = _cv, n_missing = n_missing, n_missing_y = n_missing_y)
end
