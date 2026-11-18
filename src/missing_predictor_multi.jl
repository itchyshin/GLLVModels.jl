# Multiple missing predictors, jointly integrated — the mi() VECTOR axis (Track T3).
#
# Generalises src/missing_predictor_poisson.jl (one site-level continuous predictor
# x_s, scalar slope b_x) to a VECTOR x_s ∈ R^q with slope vector b ∈ R^q:
#   y[t,s] ~ family(link⁻¹(η[t,s])),  η[t,s] = β_t + (b·x_s) + (Λ z_s)_t,
#   x_s ~ N(μ, Σ_x),  Σ_x a q×q covariance, parametrised by a lower-triangular
#   Cholesky factor Lx (Σ_x = Lx Lxᵀ) for positive-definiteness in the packed θ.
#
# Per site s, partition x_s into OBSERVED coordinates x_o and MISSING coordinates
# x_m. Observed coordinates are conditioned on (folded into the per-trait offset
# b_o·x_o) and contribute their marginal Gaussian density N(x_o; μ_o, Σ_oo). The
# missing coordinates are integrated JOINTLY with the latent z_s by a Laplace over
# the augmented variable u = (z, x_m), dim K + #missing. The predictor-side prior
# on x_m is the conditional N(m_cond, Σ_{mm·o}) of x_m | x_o under N(μ, Σ_x):
#   m_cond = μ_m + Σ_mo Σ_oo⁻¹ (x_o − μ_o),   P_m = (Σ_mm − Σ_mo Σ_oo⁻¹ Σ_om)⁻¹.
#
# The bordered Hessian over u = (z, x_m):
#   H = [ Λ'WΛ + I        Λ'W b_m   ;
#         (Λ'W b_m)'   b_m'(Σ_t W_t) b_m + P_m ],
# i.e. A = Λ'WΛ + I (K×K), C = Λ'W b_m' broadcast = (Λ' (W .* 1)) outer b_m, and the
# missing-block curvature B_mm + P_m where B_mm = b_m b_m' Σ_t W_t (family curvature
# of η wrt x_m via ∂η_t/∂x_{m,j} = b_{m,j}). Built with the OBSERVED weight W_obs for
# the (u)-mode + implicit step and with the Fisher weight W_F for the marginal's
# log-det — exactly the split in src/missing_predictor_poisson.jl / laplace_grad.jl.
#
# Limits to the scalar path when q = 1 (Gate 1), and to a brute-force Gaussian FIML
# when family = Normal (Gate 2; closed form, no Laplace). AD-clean via the implicit
# "one differentiable Newton step at the mode" trick on the (K + #missing) system.

using LinearAlgebra

# --- Σ_x Cholesky pack/unpack -------------------------------------------------
# Packed vector layout (column-major lower-tri, LOG-diagonal for positivity):
#   [logL11, L21, …, Lq1, logL22, L32, …, Lq2, …, logLqq].
function _mi_unpack_cholesky(c::AbstractVector, q::Integer)
    L = zeros(eltype(c), q, q)
    k = 1
    @inbounds for j in 1:q
        for i in j:q
            L[i, j] = i == j ? exp(c[k]) : c[k]
            k += 1
        end
    end
    return L
end
function _mi_pack_cholesky(L::AbstractMatrix)
    q = size(L, 1)
    out = Float64[]
    @inbounds for j in 1:q
        for i in j:q
            push!(out, i == j ? log(L[i, j]) : L[i, j])
        end
    end
    return out
end

# Cholesky that returns `nothing` instead of throwing on a non-PD/non-finite SPD
# matrix (transient L-BFGS line-search probes can drive Σ_x out of the PD cone; the
# interior optimum is unaffected — the matrices are PD there). Mirrors the NaN-on-
# failure robustness of `_safe_solve_xs` / `_safe_logdet_xs` for the covariance side.
function _safe_chol(M)
    all(isfinite, M) || return nothing
    return try
        cholesky(Symmetric(M))
    catch
        nothing
    end
end

# --- Per-site Gaussian conditioning on the observed predictor block -----------
# Given Σ_x (q×q), μ (q), the observed values x_o at index set `obs`, and the
# missing index set `mis`, return (m_cond, P_m, off_pred, lp_obs, ok):
#   m_cond  = μ_m + Σ_mo Σ_oo⁻¹ (x_o − μ_o)        conditional mean of x_m | x_o
#   P_m     = (Σ_mm − Σ_mo Σ_oo⁻¹ Σ_om)⁻¹          conditional precision
#   off_pred = b_o · x_o                            offset contribution from x_o
#   lp_obs  = log N(x_o; μ_o, Σ_oo)                 observed-block density
#   ok      = false if a covariance factorization failed (non-PD probe ⇒ NaN value)
# All differentiable in Σ_x (via Lx), μ, b. When `obs` is empty, m_cond = μ_m,
# P_m = Σ_mm⁻¹, off_pred = 0, lp_obs = 0.
function _mi_cond_block(Σx, μ, b, xvals, obs::Vector{Int}, mis::Vector{Int})
    T = promote_type(eltype(Σx), eltype(μ), eltype(b), eltype(xvals))
    Σmm = Σx[mis, mis]
    if isempty(obs)
        Cmm = _safe_chol(Matrix{T}(Σmm))
        Cmm === nothing && return (T.(μ[mis]), fill(T(NaN), length(mis), length(mis)),
                                   zero(T), zero(T), false)
        Pm = inv(Cmm)
        return (T.(μ[mis]), Matrix{T}(Pm), zero(T), zero(T), true)
    end
    Σoo = Matrix{T}(Σx[obs, obs])
    Σmo = Σx[mis, obs]
    xo = T.(xvals[obs])
    μo = μ[obs]
    Coo = _safe_chol(Σoo)
    if Coo === nothing
        return (T.(μ[mis]), fill(T(NaN), length(mis), length(mis)),
                dot(b[obs], xo), T(NaN), false)
    end
    ro = xo .- μo
    Sinv_ro = Coo \ ro                       # Σ_oo⁻¹ (x_o − μ_o)
    off_pred = dot(b[obs], xo)               # b_o · x_o (per-trait offset)
    lp_obs = -0.5 * (length(obs) * log(2π) + logdet(Coo) + dot(ro, Sinv_ro))
    if isempty(mis)
        return (T[], zeros(T, 0, 0), off_pred, lp_obs, true)
    end
    m_cond = μ[mis] .+ Σmo * Sinv_ro
    Σcond = Matrix{T}(Σmm) .- Σmo * (Coo \ Matrix{T}(Σx[obs, mis]))
    Ccond = _safe_chol(Σcond)
    Ccond === nothing && return (m_cond, fill(T(NaN), length(mis), length(mis)),
                                 off_pred, lp_obs, false)
    Pm = inv(Ccond)
    return (m_cond, Matrix{T}(Pm), off_pred, lp_obs, true)
end

# --- Primal augmented (z, x_m) Newton mode (Float64) --------------------------
function _mode_mi(family, y, n, Λ, β, link, b_m::AbstractVector, off_pred::Real,
                  m_cond::AbstractVector, Pm::AbstractMatrix;
                  maxiter::Integer = 100, tol::Real = 1e-9)
    K = size(Λ, 2)
    qm = length(b_m)
    z = zeros(K)
    x = copy(float.(m_cond))
    for _ in 1:maxiter
        η = _clamp_eta.(β .+ (off_pred + dot(b_m, x)) .+ Λ * z)
        μ, s, Wobs, _ = _xs_glm(family, link, η, n, y)
        sumW = sum(Wobs)
        gz = Λ' * s .- z
        gx = (sum(s)) .* b_m .- Pm * (x .- m_cond)
        A = Λ' * (Wobs .* Λ) + I
        C = (Λ' * Wobs) * b_m'                     # K×qm
        D = (b_m * b_m') .* sumW .+ Pm            # qm×qm
        H = [A C; C' D]
        rhs = vcat(gz, gx)
        (all(isfinite, H) && all(isfinite, rhs)) || break
        Δ = _safe_solve(H, rhs)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ[1:K]
        x = x .+ Δ[(K + 1):(K + qm)]
        maximum(abs, Δ) < tol && break
    end
    return z, x
end

# --- Differentiable per-site marginal -----------------------------------------
# `xvals` is a length-q vector (entries used only at `obs`); `obs`/`mis` partition
# 1:q. β, Λ, b, μ, Σx may carry duals; the mode solve is primal (no dual leakage).
function _site_mi_diffable(family, y, n, Λ, β, link, Σx, μ, b, xvals,
                           obs::Vector{Int}, mis::Vector{Int})
    K = size(Λ, 2)
    m_cond, Pm, off_pred, lp_obs, ok = _mi_cond_block(Σx, μ, b, xvals, obs, mis)
    b_m = b[mis]
    # Non-PD covariance probe ⇒ NaN value (right element type); Optim steps back.
    ok || return zero(promote_type(eltype(Λ), eltype(β), eltype(Σx), eltype(μ),
                                   eltype(b))) * NaN

    if isempty(mis)
        # All predictors observed: fold b·x into the offset, Laplace over z only.
        Λv = ForwardDiff.value.(Λ); βv = ForwardDiff.value.(β)
        offv = ForwardDiff.value(off_pred)
        famv = _mi_primal_family(family)
        ẑ = _laplace_mode(famv, y, n, Λv, βv, link; offset = offv)
        η = _clamp_eta.(β .+ off_pred .+ Λ * ẑ)
        μm, s, Wobs, _ = _xs_glm(family, link, η, n, y)
        A = Λ' * (Wobs .* Λ) + I
        z = ẑ .+ _safe_solve_xs(A, Λ' * s .- ẑ)
        ηz = _clamp_eta.(β .+ off_pred .+ Λ * z)
        μz, _, _, WFz = _xs_glm(family, link, ηz, n, y)
        Az = Λ' * (WFz .* Λ) + I
        ℓ = _xs_logker_sum(family, μz, n, y)
        return ℓ - 0.5 * dot(z, z) - 0.5 * _safe_logdet_xs(Az) + lp_obs
    end

    qm = length(mis)
    # Primal mode (strip duals).
    Λv = ForwardDiff.value.(Λ); βv = ForwardDiff.value.(β)
    bmv = ForwardDiff.value.(b_m); offv = ForwardDiff.value(off_pred)
    mcv = ForwardDiff.value.(m_cond); Pmv = ForwardDiff.value.(Pm)
    famv = _mi_primal_family(family)
    ẑ, x̂ = _mode_mi(famv, y, n, Λv, βv, link, bmv, offv, mcv, Pmv)

    # One differentiable Newton step at the primal mode (observed weight).
    η = _clamp_eta.(β .+ (off_pred + dot(b_m, x̂)) .+ Λ * ẑ)
    μm, s, Wobs, _ = _xs_glm(family, link, η, n, y)
    sumW = sum(Wobs)
    gz = Λ' * s .- ẑ
    gx = (sum(s)) .* b_m .- Pm * (x̂ .- m_cond)
    A = Λ' * (Wobs .* Λ) + I
    C = (Λ' * Wobs) * b_m'
    D = (b_m * b_m') .* sumW .+ Pm
    H = [A C; C' D]
    Δ = _safe_solve_xs(H, vcat(gz, gx))
    z = ẑ .+ Δ[1:K]
    x = x̂ .+ Δ[(K + 1):(K + qm)]

    # Marginal value with the Fisher weight in the log-det.
    ηz = _clamp_eta.(β .+ (off_pred + dot(b_m, x)) .+ Λ * z)
    μz, _, _, WFz = _xs_glm(family, link, ηz, n, y)
    sumWF = sum(WFz)
    Az = Λ' * (WFz .* Λ) + I
    Cz = (Λ' * WFz) * b_m'
    Dz = (b_m * b_m') .* sumWF .+ Pm
    Hz = [Az Cz; Cz' Dz]
    ℓ = _xs_logker_sum(family, μz, n, y)
    dxm = x .- m_cond
    return ℓ - 0.5 * dot(z, z) - 0.5 * dot(dxm, Pm * dxm) +
           0.5 * _safe_logdet_xs(Pm) - 0.5 * _safe_logdet_xs(Hz) + lp_obs
end

"""
    laplace_loglik_site_mi(family, y, n, Λ, β, link; x, b, μ, Lx) -> Float64

Per-site Laplace marginal for a non-Gaussian GLLVM with a VECTOR of site-level
predictors `x` (length `q`; entries may be `missing`/`NaN`), slope vector `b`,
predictor model `x ~ N(μ, Σ_x)` with `Σ_x = Lx Lxᵀ` (`Lx` a q×q lower-triangular
Cholesky factor). Observed coordinates of `x` are conditioned on; missing ones are
integrated jointly with the latent `z`. Families: `Poisson()`/`NegativeBinomial()`/
`Gamma()` + `LogLink`, `Binomial()`/`Beta()` + `LogitLink`.
"""
function laplace_loglik_site_mi(family, y::AbstractVector, n::AbstractVector,
                                Λ::AbstractMatrix, β::AbstractVector, link;
                                x::AbstractVector, b::AbstractVector,
                                μ::AbstractVector, Lx::AbstractMatrix)
    _xs_supported(family, link) || throw(ArgumentError(
        "laplace_loglik_site_mi: supports Poisson()/NegativeBinomial()/Gamma() + LogLink and " *
        "Binomial()/Beta() + LogitLink."))
    q = length(x)
    (length(b) == q && length(μ) == q && size(Lx) == (q, q)) ||
        throw(DimensionMismatch("b, μ, Lx must be consistent with length(x) = $q."))
    Σx = Lx * Lx'
    obs = Int[]; mis = Int[]
    xvals = Vector{eltype(promote_type(eltype(μ), Float64))}(undef, q)
    for j in 1:q
        xj = x[j]
        if ismissing(xj) || (xj isa Real && isnan(xj))
            push!(mis, j)
        else
            push!(obs, j)
            xvals[j] = Float64(xj)
        end
    end
    return _site_mi_diffable(family, y, n, Λ, β, link, Σx, μ, b, xvals, obs, mis)
end

"""
    marginal_loglik_laplace_mi(family, Y, N, Λ, β, link; X, b, μ, Lx) -> Float64

Total Laplace log-marginal over sites for a non-Gaussian GLLVM with a VECTOR of
site-level predictors integrated by FIML (the mi() vector axis). `X` is `n × q`
(entries may be `missing`/`NaN`); `b`, `μ` are length-`q`; `Lx` is the q×q
lower-triangular Cholesky factor of `Σ_x`. Each site contributes the joint
`(y_s, x_{o,s})` density with the missing predictor coordinates integrated out.
"""
function marginal_loglik_laplace_mi(family, Y::AbstractMatrix, N::AbstractMatrix,
                                    Λ::AbstractMatrix, β::AbstractVector, link;
                                    X::AbstractMatrix, b::AbstractVector,
                                    μ::AbstractVector, Lx::AbstractMatrix)
    _xs_supported(family, link) || throw(ArgumentError(
        "marginal_loglik_laplace_mi: supports Poisson()/NegativeBinomial()/Gamma() + LogLink and " *
        "Binomial()/Beta() + LogitLink."))
    n, q = size(X)
    (length(b) == q && length(μ) == q && size(Lx) == (q, q)) ||
        throw(DimensionMismatch("b, μ, Lx must be consistent with size(X, 2) = $q."))
    size(Y, 2) == n || throw(DimensionMismatch("size(Y, 2) must equal size(X, 1)."))
    Σx = Lx * Lx'
    acc = zero(promote_type(eltype(Λ), eltype(β), eltype(b), eltype(μ), eltype(Lx)))
    Tx = promote_type(eltype(μ), Float64)
    @inbounds for s in 1:n
        obs = Int[]; mis = Int[]
        xvals = Vector{Tx}(undef, q)
        for j in 1:q
            xj = X[s, j]
            if ismissing(xj) || (xj isa Real && isnan(xj))
                push!(mis, j)
            else
                push!(obs, j)
                xvals[j] = Tx(xj)
            end
        end
        acc += _site_mi_diffable(family, view(Y, :, s), view(N, :, s), Λ, β, link,
                                 Σx, μ, b, xvals, obs, mis)
    end
    return acc
end

# --- Gaussian multi-predictor marginal (closed form; no Laplace) --------------
# y_s = a + (b·x_s) 1_p + Λ η_s + ε_s,  η_s ~ N(0, I_K), ε_s ~ N(0, σ_eps² I_p),
# x_s ~ N(μ, Σ_x). Joint w = [y; x] ~ N(mw, V):
#   mw  = [a + b·μ ; μ],
#   Vyy = ΛΛ' + σ_eps² I + (b'Σx b) 11',  Vyx = 1_p (Σx b)',  Vxx = Σx.
# Per site, keep all of y plus the OBSERVED coords of x and evaluate that
# observed-block Gaussian density (missing predictors integrated out exactly).
# Σ_x is passed as its Cholesky factor `Lx`; σ_eps² as `σ_eps2`.
function marginal_loglik_mi_gaussian(y::AbstractMatrix, X::AbstractMatrix,
                                     a::AbstractVector, Λ::AbstractMatrix,
                                     b::AbstractVector, μ::AbstractVector,
                                     Lx::AbstractMatrix, σ_eps2::Real)
    p, n = size(y)
    q = size(X, 2)
    (length(a) == p && length(b) == q && length(μ) == q && size(Lx) == (q, q)) ||
        throw(DimensionMismatch("a/b/μ/Lx inconsistent with y (p=$p) and X (q=$q)."))
    size(X, 1) == n || throw(DimensionMismatch("size(X, 1) must equal size(y, 2)."))
    T = promote_type(eltype(a), eltype(Λ), eltype(b), eltype(μ), eltype(Lx), typeof(σ_eps2))
    Σx = Lx * Lx'
    bΣb = dot(b, Σx * b)
    Vyy = Λ * Λ' + σ_eps2 * I + bΣb .* (ones(T, p) * ones(T, p)')
    Vyx = ones(T, p) * (Σx * b)'
    V = [Vyy Vyx; Vyx' Σx]
    mw = vcat(a .+ dot(b, μ), μ)
    acc = zero(T)
    @inbounds for s in 1:n
        obsj = Int[]
        for j in 1:q
            xj = X[s, j]
            (ismissing(xj) || (xj isa Real && isnan(xj))) || push!(obsj, j)
        end
        oidx = vcat(1:p, p .+ obsj)
        mo = vcat(T.(view(y, :, s)), T[T(X[s, j]) for j in obsj])
        Vo = Symmetric(V[oidx, oidx])
        Co = cholesky(Vo)
        r = mo .- mw[oidx]
        acc += -0.5 * (length(oidx) * log(2π) + logdet(Co) + dot(r, Co \ r))
    end
    return acc
end

# --- User-facing fitter -------------------------------------------------------
function _mi_init_lambda_multi(family, Y, N, β0, K)
    return _mi_init_lambda(family, Y, N, β0, K)
end

"""
    fit_gllvm_mi_multi(family, Y, X; K, N=nothing, link=canonical, ...) -> NamedTuple

Fit a non-Gaussian GLLVM with a VECTOR of site-level continuous predictors `X`
(`n × q`, entries may be `missing`/`NaN`), where the missing predictor cells are
integrated out jointly by full-information ML (the augmented `(z, x_missing)`
Laplace of `marginal_loglik_laplace_mi`). Supported families:
`Poisson()`/`NegativeBinomial()`/`Gamma()` + `LogLink`, `Binomial()`/`Beta()` +
`LogitLink`. The predictors enter with a slope vector `b` (length `q`) and a joint
covariate model `x ~ N(μ, Σ_x)` with `Σ_x = Lx Lxᵀ` (Cholesky-parametrised).

Returns a NamedTuple with `β`, `Λ` (`p × K`), `b` (length `q`), `μ` (length `q`),
`Σx` (`q × q`), `Lx`, `dispersion` (`nothing` for canonical families), `logLik`,
`converged`, `n_missing` (count of missing predictor CELLS). Optimised with L-BFGS
over the AD-clean marginal.
"""
function fit_gllvm_mi_multi(family, Y::AbstractMatrix, X::AbstractMatrix; K::Integer,
                            N::Union{Nothing,AbstractMatrix} = nothing,
                            link = _mi_canonical_link(family),
                            g_tol::Real = 1e-8, iterations::Integer = 1000)
    p, n = size(Y)
    size(X, 1) == n || throw(ArgumentError("size(X, 1) = $(size(X, 1)) must equal n = $n."))
    K ≥ 1 || throw(ArgumentError("K must be ≥ 1."))
    q = size(X, 2)
    q ≥ 1 || throw(ArgumentError("X must have ≥ 1 column."))
    _xs_supported(family, link) || throw(ArgumentError(
        "fit_gllvm_mi_multi: supports Poisson()/NegativeBinomial()/Gamma() + LogLink and " *
        "Binomial()/Beta() + LogitLink."))
    Nm = N === nothing ? ones(Int, p, n) : N
    _check_family_response_support(family, Y, nothing; caller = "fit_gllvm_mi_multi", N = Nm)

    # Observed mask + warm starts for (μ, Σ_x) from observed predictor cells.
    isobs = [!(ismissing(X[s, j]) || (X[s, j] isa Real && isnan(X[s, j])))
             for s in 1:n, j in 1:q]
    any(isobs) || throw(ArgumentError("X has no observed values."))
    μ0 = Vector{Float64}(undef, q)
    for j in 1:q
        col = Float64[Float64(X[s, j]) for s in 1:n if isobs[s, j]]
        isempty(col) && throw(ArgumentError("Predictor column $j has no observed values."))
        μ0[j] = Statistics.mean(col)
    end
    # Covariance warm start over complete-predictor sites; fall back to a diagonal.
    complete = [all(isobs[s, :]) for s in 1:n]
    Σ0 = if count(complete) ≥ q + 1
        Xc = Float64[Float64(X[s, j]) for s in findall(complete), j in 1:q]
        Symmetric(cov(Xc) + 1e-3 * I)
    else
        d = Float64[Statistics.var(Float64[Float64(X[s, j]) for s in 1:n if isobs[s, j]]) for j in 1:q]
        Symmetric(diagm(max.(d, 1e-2)))
    end
    Lx0 = Matrix(cholesky(Σ0).L)
    cpack0 = _mi_pack_cholesky(Lx0)
    nchol = q * (q + 1) ÷ 2

    rowmean = vec(sum(Y, dims = 2)) ./ n
    β0 = if family isa Poisson || family isa NegativeBinomial
        log.(max.(rowmean, 0.5))
    elseif family isa Gamma
        log.(max.(rowmean, 1e-3))
    elseif family isa Beta
        log.(clamp.(rowmean, 1e-3, 1 - 1e-3) ./ (1 .- clamp.(rowmean, 1e-3, 1 - 1e-3)))
    else
        log.((rowmean .+ 0.5) ./ (vec(sum(Nm, dims = 2)) ./ n .- rowmean .+ 0.5))
    end
    Λ0 = _mi_init_lambda_multi(family, Y, Nm, β0, K)

    has_disp = _mi_has_dispersion(family)
    base = p + p * K
    # θ = [β (p); vec(Λ) (pK); b (q); μ (q); chol-packed (nchol)[; log disp]]
    θ0 = vcat(β0, vec(Λ0), zeros(q), μ0, cpack0)
    has_disp && (θ0 = vcat(θ0, _mi_logdisp0(family)))
    dispidx = base + 2q + nchol + 1

    function negll(θ)
        β = θ[1:p]
        Λ = reshape(θ[(p + 1):base], p, K)
        b = θ[(base + 1):(base + q)]
        μ = θ[(base + q + 1):(base + 2q)]
        L = _mi_unpack_cholesky(θ[(base + 2q + 1):(base + 2q + nchol)], q)
        fam = has_disp ? _mi_with_disp(family, exp(θ[dispidx])) : family
        return -marginal_loglik_laplace_mi(fam, Y, Nm, Λ, β, link;
                                           X = X, b = b, μ = μ, Lx = L)
    end
    ls = has_disp ? Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)) :
                    Optim.LBFGS()
    res = Optim.optimize(negll, θ0, ls,
                         Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :forward)
    θ = Optim.minimizer(res)
    β = θ[1:p]
    Λ = reshape(θ[(p + 1):base], p, K)
    b = θ[(base + 1):(base + q)]
    μ = θ[(base + q + 1):(base + 2q)]
    Lx = _mi_unpack_cholesky(θ[(base + 2q + 1):(base + 2q + nchol)], q)
    Σx = Lx * Lx'
    dispersion = has_disp ? exp(θ[dispidx]) : nothing
    return (β = β, Λ = Λ, b = b, μ = μ, Σx = Σx, Lx = Lx, dispersion = dispersion,
            logLik = _fit_verdict(res)[1], converged = _fit_verdict(res)[2],
            n_missing = count(!, isobs))
end
