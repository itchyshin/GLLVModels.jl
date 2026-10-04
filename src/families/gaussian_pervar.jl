# Per-species (heteroscedastic) Gaussian GLLVM marginal log-likelihood + fit.
#
# This variant places a per-species residual SD φ_j on each trait. It does not
# reproduce R's fixed-residual plus unique-variance decomposition unless the
# fixed residual SD is supplied explicitly.
# GLLVModels.jl's shared-σ Gaussian path (src/likelihood.jl,
# src/fit.jl, src/profile.jl) uses a single σ_eps and is left UNTOUCHED. This file
# adds a parallel per-species variant.
#
# The only change relative to the scalar marginal is the per-trait diagonal of the
# marginal site covariance:
#
#   shared-σ:    M = Λ Λ' + σ_eps² · I_p          (constant diagonal)
#   per-species: M = Λ Λ' + diag(φ²_1, …, φ²_p)   (per-trait diagonal)
#
# Each site has covariance ΛΛ' + diag(φ²). Direct covariance Cholesky is
# deliberate here: subtractive Woodbury solves can lose the quadratic form near
# a zero residual variance. This costs O(p³); EM retains its own fast path.
# Fixed effects use the same complete p×n×q design convention as likelihood.jl.
#
# The shared-σ fit profiles the single σ_eps analytically (src/profile.jl). That
# closed-form profile does NOT generalise to a per-species vector, so the
# per-species fit optimises the variances numerically — expected and fine.

using LinearAlgebra

"""
    gaussian_pervar_marginal_loglik(y, Λ, φ²vec; X=nothing, β=nothing) -> Real

Gaussian GLLVM marginal log-likelihood with **per-species** (heteroscedastic)
residual variances. `y` is `p × n_sites`, `Λ` is `p × K` unit-tier loadings, and
`φ²vec` is the length-`p` vector of per-species residual **variances** `V_j = φ_j²`
(NOT SDs).

Each site `y_s ~ N(X_s β, M)` with `M = Λ Λ' + diag(φ²vec)`. The marginal is
computed by directly factoring the `p × p` covariance. This avoids subtractive
Woodbury cancellation near a zero residual variance. The factorization costs
O(p³); the intercept-only EM fitter retains its separate fast path.

Fixed effects: pass both `X::Array{<:Real,3}` of shape `(p, n_sites, q)` and
`β::Vector` of length `q`, or neither. When `β`/`X` are omitted the residual is
`y` itself (no centering) — identical to `gaussian_marginal_loglik`'s convention.

Passing a constant variance vector `fill(σ², p)` reproduces the scalar marginal
`gaussian_marginal_loglik(y, Λ, sqrt(σ²); …)` to machine precision.
"""
function gaussian_pervar_marginal_loglik(y::AbstractMatrix, Λ::AbstractMatrix,
                                         φ²vec::AbstractVector;
                                         X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                                         β::Union{Nothing, AbstractVector} = nothing)
    p, n = size(y)
    K    = size(Λ, 2)
    length(φ²vec) == p ||
        throw(ArgumentError("φ²vec length ($(length(φ²vec))) must equal p ($p)"))
    T = promote_type(eltype(y), eltype(Λ), eltype(φ²vec))

    # Residual ε = y - X β if fixed effects supplied (mirrors likelihood.jl).
    if X === nothing && β === nothing
        resid = y
        Tres = T
    else
        (X === nothing || β === nothing) &&
            throw(ArgumentError("Provide both X and β or neither"))
        q = size(X, 3)
        size(X, 1) == p ||
            throw(ArgumentError("X first dim ($(size(X,1))) must equal p ($p)"))
        size(X, 2) == n ||
            throw(ArgumentError("X second dim ($(size(X,2))) must equal n_sites ($n)"))
        length(β) == q ||
            throw(ArgumentError("β length ($(length(β))) must equal size(X, 3) ($q)"))
        Tres = promote_type(T, eltype(X), eltype(β))
        resid = Matrix{Tres}(undef, p, n)
        @inbounds for s in 1:n, t in 1:p
            μ_ts = zero(Tres)
            for k in 1:q
                μ_ts += X[t, s, k] * β[k]
            end
            resid[t, s] = y[t, s] - μ_ts
        end
    end

    # Per-species diagonal: d[t] = φ²_t. This is the ONLY substantive difference
    # from the shared-σ marginal (which uses d[t] = σ_eps² for all t).
    Td = promote_type(T, eltype(φ²vec))
    d  = Vector{Td}(undef, p)
    @inbounds for t in 1:p
        d[t] = convert(Td, φ²vec[t])
    end

    all(v -> isfinite(v) && v > 0, d) ||
        throw(ArgumentError("residual variances must be finite and positive"))
    # Direct covariance factorization remains stable when one diagonal variance
    # approaches zero but the loading contribution keeps the covariance SPD.
    F = cholesky(Symmetric(Λ * Λ' + Diagonal(d)))
    logdet_M = logdet(F)

    # Quadratic form Σ_s r_s' M⁻¹ r_s via covariance solves.
    Minv_r = F \ resid                 # p × n
    quad   = sum(resid .* Minv_r)

    Tout = promote_type(Tres, Td)
    return -convert(Tout, 0.5) * (n * p * log(convert(Tout, 2π)) + n * logdet_M + quad)
end

"""
    GaussianPerVarFit

Result of [`fit_gaussian_pervar_gllvm`](@ref): the heteroscedastic Gaussian GLLVM
fit with a per-species residual variance.

Fields:
- `β::Vector`    — GLS coefficients (length `q` when `X` is supplied), or
  per-species intercepts (length `p`) when `X=nothing`.
- `Λ::Matrix`    — fitted unit-tier loadings (`p × K`).
- `φ²::Vector`   — total per-species diagonal variances (length `p`).
- `ψ²::Vector`   — estimated unique variances; `φ² = ψ² + fixed_residual_sd²`.
- `fixed_residual_sd` — supplied independent residual SD, zero by default.
- `loglik`       — converged marginal log-likelihood.
- `converged`    — Optim convergence flag.
- `iterations`   — Optim iteration count.
- `integration` — requested/actual AGHQ fallback information, or `nothing` by default.
"""
struct GaussianPerVarFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    φ²::Vector{Float64}
    loglik::Float64
    converged::Bool
    iterations::Int
    ψ²::Vector{Float64}
    fixed_residual_sd::Float64
    integration::Union{Nothing,AbstractIntegrationInfo}
end

# Preserve construction of the original unconstrained per-variance result.
GaussianPerVarFit(β, Λ, φ², loglik, converged, iterations) =
    GaussianPerVarFit(β, Λ, φ², loglik, converged, iterations, copy(φ²), 0.0)
GaussianPerVarFit(β, Λ, φ², loglik, converged, iterations, ψ², fixed_residual_sd) =
    GaussianPerVarFit(β, Λ, φ², loglik, converged, iterations, ψ², fixed_residual_sd, nothing)

function Base.show(io::IO, fit::GaussianPerVarFit)
    p, K = size(fit.Λ)
    print(io, "GaussianPerVarFit(p=$p, K=$K, loglik=",
          round(fit.loglik; digits = 4),
          ", converged=", fit.converged,
          ", iterations=", fit.iterations)
    if fit.integration !== nothing
        info = fit.integration
        print(io, ", integration=exact Gaussian/Laplace, requested=", info.requested,
              ", reason=", info.reason)
    end
    print(io, ")")
end
Base.summary(fit::GaussianPerVarFit) = sprint(show, fit)

function _pervar_with_fallback(fit, request, controls)
    request === :off && return fit
    k = request === :auto ? 5 : request
    reason = k == 1 ? :laplace_rule : fit.fixed_residual_sd > 0 ?
        :other_random_blocks : :pervar_aghq_unimplemented
    if k != 1
        detail = reason === :other_random_blocks ?
            "Stage 1a requires a loadings-only block; this model includes trait-specific unique effects" :
            "the plain heteroscedastic Gaussian AGHQ adapter is not implemented"
        @warn "AGHQ request retained exact Gaussian/Laplace" reason=reason requested=request detail=detail
    end
    # No quadrature ran: do not fabricate modes, gradients or adaptation results.
    info = AGHQFitInfo(request, :laplace, 1, k, 1, reason, false, Inf, controls,
        (fixed_residual_sd=fit.fixed_residual_sd,), nothing, AGHQAdaptation[], nothing, "", NaN)
    return GaussianPerVarFit(fit.β, fit.Λ, fit.φ², fit.loglik, fit.converged,
        fit.iterations, fit.ψ², fit.fixed_residual_sd, info)
end

_loadings(fit::GaussianPerVarFit) = fit.Λ
_loglik(fit::GaussianPerVarFit)   = fit.loglik

# Ledermann (1940) bound for Σ = ΛΛ' + diag(ψ) with one ψ per response.
function _ledermann_pervar_k_admissible(p::Integer, K::Integer)
    (p - K)^2 >= p + K
end

# Free params: requested fixed effects + reduced loadings + residual variances.
function _nparams(fit::GaussianPerVarFit)
    p, K = size(fit.Λ)
    return length(fit.β) + rr_theta_len(p, K) + p
end

"""
    fit_gaussian_pervar_gllvm(Y; K, X=nothing, fixed_residual_sd=0.0,
                             aghq=false, aghq_control=(;), g_tol=1e-5, iterations=1000)
        -> GaussianPerVarFit

Fit a heteroscedastic (per-species variance) Gaussian GLLVM by EM or L-BFGS.

For this per-species residual model (`Σ = Λ Λ' + diag(ψ)` with one ψ per trait),
`K` must satisfy the Ledermann identifiability bound `(p - K)² ≥ p + K` (e.g.
`p = 4` → `K ≤ 1`, `p = 5` → `K ≤ 2`, `p = 10` → `K ≤ 6`). The shared-residual
fitter `fit_gaussian_gllvm` uses one scalar `σ_eps` and a weaker
computational limit (`K < p` for the PPCA warm start only). Inadmissible `K`
here raises `ArgumentError` before optimisation.

`Y` is `p × n_sites`. Optimises `θ = [vec(packed Λ); log φ²_1 … log φ²_p]` with
fixed effects profiled out analytically. With `X=nothing`, the `p` trait intercepts
are the row means of `Y`, independent of the covariance. An explicit finite,
full-column-rank design `X` of shape `(p, n_sites, q)` defines the complete mean
`X_s * β`; no intercept is added. Its `q` coefficients are profiled by GLS at
each covariance evaluation. A zero-column design specifies a zero mean.

`method=:em` (default) uses EM only for the intercept-only `K < p` case;
explicit designs use L-BFGS, as does `method=:lbfgs`. This is ML profiling,
not REML. Rank-deficient or nonfinite designs are rejected.

With `fixed_residual_sd=c > 0`, the covariance is
`Λ*Λ' + Diagonal(ψ² .+ c^2)`, where each unique variance `ψ²` is estimated
on its log scale. The fixed residual is not estimated or counted as a free
parameter. This case always uses L-BFGS, including when `method=:em` is
requested. `fit.ψ²` retains unique variances; `fit.φ²` remains the total
diagonal variance. The default `c=0` retains the original model and EM route.
There is no automatic suppression or data-derived floor: pass the fixed scale
explicitly when matching a reference model. A fixed scale must be finite and
nonnegative, and its square must be representable.

`aghq=1` retains the exact Gaussian/Laplace fit without a warning. Larger node
counts and `true`/`:auto` also retain that fit, with a warning: the fixed-residual
model includes unique effects and is outside Stage 1a's loadings-only domain.
The plain `c=0` heteroscedastic AGHQ adapter is still unimplemented and reports
that distinct reason. No requested quadrature changes the model, adds a ridge,
or runs an adaptation loop. `fit.integration` records requested/actual method,
node counts and reason; the existing `fit.converged` reports optimizer status.
By default `aghq=false` keeps `integration=nothing`. Invalid integration controls
are rejected before optimization. The formula `pervar=true` route forwards these
options; this does not establish R bridge or interval parity.

Warm start: PPCA closed form (Tipping & Bishop 1999) for `Λ` and per-species
residual variances initialised from the per-trait sample variances of `Y`.

The shared-σ `fit_gaussian_gllvm` is untouched; this is a parallel variant.
"""
function fit_gaussian_pervar_gllvm(Y::AbstractMatrix;
                                   K::Integer,
                                   X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                                   fixed_residual_sd::Real = 0.0,
                                   aghq = false,
                                   aghq_control = (;),
                                   method::Symbol = :em,
                                   g_tol::Real = 1e-5,
                                   em_tol::Real = 1e-8,
                                   iterations::Integer = 1000)
    p, n = size(Y)
    @assert K ≥ 1
    _ledermann_pervar_k_admissible(p, K) || throw(ArgumentError(
        "fit_gaussian_pervar_gllvm: per-species residual variances require the " *
        "Ledermann bound (p - K)^2 >= p + K for identifiable Λ and ψ; got p=$p, " *
        "K=$K. The shared-σ fitter fit_gaussian_gllvm uses a different condition."))
    @assert n ≥ 2 "Need n_sites ≥ 2 for per-species variances"
    method in (:em, :lbfgs) || throw(ArgumentError("method must be :em or :lbfgs"))
    c = Float64(fixed_residual_sd)
    c² = c^2
    isfinite(c) && c >= 0 && isfinite(c²) && (c == 0 || c² > 0) ||
        throw(ArgumentError("fixed_residual_sd must be finite and nonnegative with a representable square"))
    request = _aghq_request(aghq)
    integration_controls = _aghq_controls(aghq_control)

    Yf = Matrix{Float64}(Y)
    all(isfinite, Yf) || throw(ArgumentError("Y must contain only finite responses"))
    Xf = if X === nothing
        nothing
    else
        size(X, 1) == p && size(X, 2) == n ||
            throw(DimensionMismatch("X must have shape (p=$p, n=$n, q)"))
        all(isfinite, X) || throw(ArgumentError("X must contain only finite values"))
        design = Array{Float64,3}(X)
        A = reshape(design, p*n, size(design, 3))
        rank(A) == size(A, 2) || throw(ArgumentError("X must have full column rank"))
        design
    end

    # OLS residuals provide a mean-shift-equivariant covariance warm start.
    # GLS coefficients for a requested design are recomputed inside nll.
    μ0 = vec(sum(Yf, dims = 2)) ./ n           # length p
    Yc = if Xf === nothing
        Yf .- reshape(μ0, p, 1)
    elseif size(Xf, 3) == 0
        copy(Yf)
    else
        A = reshape(Xf, p*n, size(Xf, 3))
        Yf .- reshape(A * (A \ vec(Yf)), p, n)
    end

    # ----- Warm starts.
    # Λ via PPCA on the centred data (requires K < p); otherwise fall back to
    # the default lower-triangular init.
    Λ0 = if K < p
        Λp, _ = ppca_init(Yc, K)
        Λp
    else
        unpack_lambda(init_theta_rr(p, K), p, K)
    end
    θ_Λ0 = pack_lambda(Λ0)
    rrlen = rr_theta_len(p, K)

    # Per-species variance init: per-species sample variance of the centred data,
    # floored away from zero. Use a fraction so it does not absorb the loadings.
    col_var = vec(sum(abs2, Yc, dims = 2)) ./ max(n - 1, 1)   # length p
    φ²_0 = max.(0.5 .* col_var .- c², 1e-3)
    logφ²_0 = log.(φ²_0)

    params0 = vcat(θ_Λ0, logφ²_0)

    # Fast path: closed-form EM for factor analysis (Rubin & Thayer 1982, `em_fa`)
    # on the IDENTICAL Λ Λ' + diag(φ²) model. No inner AD — reaches the same ML
    # optimum 1–2 orders of magnitude faster than the L-BFGS + ForwardDiff path.
    # Requires the FA regime K < p and no fixed effects; otherwise fall through to
    # L-BFGS. The per-species intercept is the profiled row means (β = μ0), so
    # EM runs on the centred residual `Yc`.
    if method === :em && K < p && X === nothing && c == 0
        Λ_em, φ²_em, ll_em, nit_em, conv_em =
            em_fa(Yc, K; λ_init = Λ0, ψ_init = φ²_0, tol = em_tol,
                  max_iter = max(Int(iterations), 2000))
        fit = GaussianPerVarFit(
            collect(Float64, μ0),
            Matrix{Float64}(Λ_em),
            collect(Float64, φ²_em),
            Float64(ll_em),
            conv_em,
            nit_em,
        )
        return _pervar_with_fallback(fit, request, integration_controls)
    end

    # Exact ML GLS, using the same direct covariance factorization as the
    # likelihood. Do not subtract nearly equal Woodbury terms near a boundary.
    function profile_coefficients(Λ, φ²)
        q = size(Xf, 3)
        T = promote_type(eltype(Λ), eltype(φ²))
        q == 0 && return T[]
        C = cholesky(Symmetric(Λ * Λ' + Diagonal(φ²)))
        A = zeros(T, q, q)
        b = zeros(T, q)
        for site in 1:n
            Xi = @view Xf[:, site, :]
            A .+= Xi' * (C \ Xi)
            b .+= Xi' * (C \ view(Yf, :, site))
        end
        return cholesky(Symmetric(A)) \ b
    end
    rejected_covariances = Ref(0)
    function nll(params)
        θ_Λ   = @view params[1:rrlen]
        logφ² = @view params[(rrlen + 1):(rrlen + p)]
        Λ     = unpack_lambda(θ_Λ, p, K)
        ψ²    = exp.(logφ²)
        all(v -> isfinite(v) && v > 0, ψ²) || return oftype(first(params), Inf)
        φ²    = ψ² .+ c²
        all(isfinite, φ²) || return oftype(first(params), Inf)
        try
            if Xf === nothing
                return -gaussian_pervar_marginal_loglik(Yc, Λ, φ²)
            end
            β = profile_coefficients(Λ, φ²)
            return -gaussian_pervar_marginal_loglik(Yf, Λ, φ²; X = Xf, β = β)
        catch err
            err isa PosDefException || rethrow()
            rejected_covariances[] += 1
            return oftype(first(params), Inf)
        end
    end

    opts = Optim.Options(g_tol = g_tol, iterations = Int(iterations),
                         show_trace = false)
    res = Optim.optimize(nll, params0, Optim.LBFGS(), opts; autodiff = :forward)

    params_hat = Optim.minimizer(res)
    θ_Λ_hat   = params_hat[1:rrlen]
    logφ²_hat = params_hat[(rrlen + 1):(rrlen + p)]
    Λ_hat     = unpack_lambda(θ_Λ_hat, p, K)
    ψ²_hat    = exp.(logφ²_hat)
    φ²_hat    = ψ²_hat .+ c²

    β_hat = Xf === nothing ? μ0 : profile_coefficients(Λ_hat, φ²_hat)

    _, conv, iters = _fit_verdict(res)
    ll = -nll(params_hat)  # report the likelihood at the returned coordinates
    conv = conv && isfinite(ll)
    if rejected_covariances[] > 0
        @warn "Rejected numerically unfactorizable covariance evaluations; no ridge was added" count=rejected_covariances[]
    end

    fit = GaussianPerVarFit(
        collect(Float64, β_hat),
        Matrix{Float64}(Λ_hat),
        collect(Float64, φ²_hat),
        Float64(ll),
        conv,
        iters,
        collect(Float64, ψ²_hat),
        c,
    )
    return _pervar_with_fallback(fit, request, integration_controls)
end
