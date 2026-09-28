# Two-level (between- vs within-individual) reduced-rank Gaussian GLLVM —
# the behavioural-syndromes decomposition (paper Eq 11).
#
# Model, per trait t of observation j on individual i:
#   y_ijt = μ_t + (Λ_B z_B,i)[t] + s_B,it + (Λ_W z_W,ij)[t] + s_W,ijt
# with z_B,i ~ N(0, I_{K_B}), z_W,ij ~ N(0, I_{K_W}),
#      s_B,i ~ N(0, diag(σ²_B)), s_W,ij ~ N(0, diag(σ²_W)).
#
# The between block is SHARED across all observations of an individual (the
# syndrome): cov contribution Σ_B = Λ_B Λ_Bᵀ + diag(σ²_B), p×p, common to every
# observation of i. The within block + observation residual are INDEPENDENT per
# observation (the state): Σ_W = Λ_W Λ_Wᵀ + diag(σ²_W), p×p, per observation.
#
# Stacking the n_i observations of individual i (n_i·p vector, observation-major)
# gives covariance
#   Σ_i = I_{n_i} ⊗ Σ_W + J_{n_i} ⊗ Σ_B,   J_{n_i} = 1 1ᵀ (rank 1).
# This is the SAME rotation-trick structure as the grouped-intercept / phylo
# paths: J_{n_i} has eigenvalue n_i (eigenvector 1/√n_i) and n_i−1 zeros, so in a
# basis whose first row is 1/√n_i the block is block-diagonal
# diag(Σ_W + n_i Σ_B, Σ_W, …, Σ_W). Hence per individual
#   logdet Σ_i = logdet(Σ_W + n_i Σ_B) + (n_i − 1)·logdet(Σ_W)
#   yᵢ' Σ_i⁻¹ yᵢ = n_i·m_i'(Σ_W + n_i Σ_B)⁻¹ m_i + tr(Y_ic' Σ_W⁻¹ Y_ic)
# with m_i the per-trait mean over i's observations and Y_ic the centred
# residuals. Individuals are independent ⇒ ℓ = Σ_i ℓ_i. Both p×p covariances
# (Σ_W and Σ_W + n_i Σ_B) are factored by a dense p×p Cholesky; p is small in the
# two-level regime. A Woodbury solve on the K×K core is deliberately NOT used for
# Σ_W: its subtractive form loses the quadratic form when a σ²_W[t] is tiny
# relative to Λ_W[t,:]² (overstating ℓ by up to hundreds of nats; GATE 1b in
# test/test_twolevel.jl). AD-clean (verified against a central FD gradient).
#
# μ_t (per-trait grand mean) is profiled out analytically as the GLS mean; for
# the recovery test the data are centred so μ = 0.

# Build Σ = Λ Λᵀ + diag(σ²_diag) DENSELY (for Σ_W, Σ_B and the Σ_W + n_i Σ_B sum;
# a direct dense p×p chol is simplest, stays accurate when a diagonal entry is
# tiny, and p is small in the two-level regime). Symmetrised before factoring.
function _dense_sigma(Λ::AbstractMatrix, σ²_diag::AbstractVector)
    p = size(Λ, 1)
    T = promote_type(eltype(Λ), eltype(σ²_diag))
    A = Matrix{T}(Λ * Λ')
    @inbounds for t in 1:p
        A[t, t] += σ²_diag[t]
    end
    return (A + A') ./ 2
end

# ---------------------------------------------------------------------------
# Two-level marginal log-likelihood (centred y; μ profiled to 0 by centring).
# `ind_idx` is a vector of column-index vectors, one per individual.
# ---------------------------------------------------------------------------
function _twolevel_loglik(y::AbstractMatrix, ind_idx::Vector{Vector{Int}},
        Λ_B::AbstractMatrix, σ²_B::AbstractVector,
        Λ_W::AbstractMatrix, σ²_W::AbstractVector)
    p, ntot = size(y)
    T = promote_type(eltype(y), eltype(Λ_B), eltype(σ²_B), eltype(Λ_W), eltype(σ²_W))

    # Within block Σ_W = Λ_W Λ_Wᵀ + diag(σ²_W): used for EVERY individual's
    # centred part, so factor once. A dense p×p Cholesky, not a Woodbury
    # solve: the subtractive Woodbury form D⁻¹V − D⁻¹Λ(I + ΛᵀD⁻¹Λ)⁻¹ΛᵀD⁻¹V
    # cancels terms of size Λ²/σ²_W[t] and loses the quadratic form when a
    # σ²_W[t] is tiny relative to Λ_W[t,:]² (see GATE 1b in
    # test/test_twolevel.jl; same choice as src/families/gaussian_pervar.jl).
    # A failed factorisation is reported as a failed evaluation (-Inf).
    ΣW = _dense_sigma(Λ_W, σ²_W)
    cW = cholesky(Symmetric(ΣW); check = false)
    issuccess(cW) || return convert(T, -Inf)
    logdetΣW = logdet(cW)

    # Between block Σ_B = Λ_B Λ_Bᵀ + diag(σ²_B): only ever enters as
    # Σ_W + n_i Σ_B, which we factor per distinct n_i. Cache by group size.
    ΣB = _dense_sigma(Λ_B, σ²_B)
    mean_cache = Dict{Int, Any}()        # n_i -> (cholesky(Σ_W + n_i Σ_B), logdet)

    twopi = convert(T, 2π)
    ll = zero(T)
    for idx in ind_idx
        ni = length(idx)
        Yi = Matrix{T}(@view y[:, idx])             # p × n_i
        mi = vec(sum(Yi, dims = 2)) ./ ni           # per-trait mean over obs
        Yic = Yi .- reshape(mi, p, 1)               # centred residuals

        # Centred part: tr(Y_ic' Σ_W⁻¹ Y_ic) = ‖L⁻¹ Y_ic‖², Σ_W = L Lᵀ
        quad_centered = sum(abs2, cW.L \ Yic)

        # Mean part: n_i · m_i' (Σ_W + n_i Σ_B)⁻¹ m_i
        cMean, logdetMean = get!(mean_cache, ni) do
            M = ΣW .+ ni .* ΣB
            cM = cholesky(Symmetric((M + M') ./ 2); check = false)
            (cM, issuccess(cM) ? logdet(cM) : zero(T))
        end
        issuccess(cMean) || return convert(T, -Inf)
        quad_mean = ni * dot(mi, cMean \ mi)

        # Both quadratic forms are ≥ 0 for PD Σ_W / Σ_W + n_i Σ_B. A negative
        # value would mean a solve lost all precision, and the formula would
        # then return a huge spurious POSITIVE log-likelihood that the optimiser
        # happily "converges" to. Kept as a belt-and-braces check after the
        # dense Σ_W factorisation above: report the evaluation as failed.
        (quad_centered < 0 || quad_mean < 0) && return convert(T, -Inf)

        logdet_i = logdetMean + (ni - 1) * logdetΣW
        quad_i = quad_mean + quad_centered
        ll += -convert(T, 0.5) * (ni * p * log(twopi) + logdet_i + quad_i)
    end
    return ll
end

"""
    twolevel_marginal_loglik(y, individual, Λ_B, σ²_B, Λ_W, σ²_W) -> Real

Marginal log-likelihood of the Gaussian two-level reduced-rank GLLVM
(behavioural-syndromes decomposition, paper Eq 11). `y` is `p × n_obs` (each
column one observation, observation-major); `individual` a length-`n_obs` vector
assigning each observation to an individual. The between-individual block
`Σ_B = Λ_B Λ_Bᵀ + diag(σ²_B)` is shared across all observations of an individual;
the within-individual block `Σ_W = Λ_W Λ_Wᵀ + diag(σ²_W)` is independent per
observation. `y` is assumed centred (per-trait grand mean removed). Solved per
individual by the rotation trick on `Σ_i = I_{n_i} ⊗ Σ_W + J_{n_i} ⊗ Σ_B`.
"""
function twolevel_marginal_loglik(y::AbstractMatrix, individual::AbstractVector,
        Λ_B::AbstractMatrix, σ²_B::AbstractVector,
        Λ_W::AbstractMatrix, σ²_W::AbstractVector)
    codes, _ = _code_grouping(individual)
    L = maximum(codes)
    ind_idx = [findall(==(g), codes) for g in 1:L]
    return _twolevel_loglik(y, ind_idx, Λ_B, σ²_B, Λ_W, σ²_W)
end

# ---------------------------------------------------------------------------
# Packed parameter layout (all variances on the log scale; loadings via the
# reduced-rank lower-triangular packing):
#   [ θ_rr_B (rr_theta_len(p,K_B))
#     log σ²_B  (p)
#     θ_rr_W (rr_theta_len(p,K_W))
#     log σ²_W  (p) ]
# ---------------------------------------------------------------------------
function _twolevel_unpack(θ::AbstractVector, p::Integer, K_B::Integer, K_W::Integer)
    rrB = rr_theta_len(p, K_B)
    rrW = rr_theta_len(p, K_W)
    c = 0
    Λ_B = unpack_lambda(@view(θ[(c + 1):(c + rrB)]), p, K_B); c += rrB
    σ²_B = exp.(@view θ[(c + 1):(c + p)]);                    c += p
    Λ_W = unpack_lambda(@view(θ[(c + 1):(c + rrW)]), p, K_W); c += rrW
    σ²_W = exp.(@view θ[(c + 1):(c + p)]);                    c += p
    return Λ_B, σ²_B, Λ_W, σ²_W
end
_twolevel_npar(p, K_B, K_W) = rr_theta_len(p, K_B) + p + rr_theta_len(p, K_W) + p

"""
    TwoLevelFit

Result of [`fit_twolevel_gaussian`](@ref): between-individual loadings `Λ_B`
(p×K_B) and per-trait between variances `σ²_B`; within-individual loadings `Λ_W`
(p×K_W) and per-trait within variances `σ²_W`; the assembled per-level covariances
`Σ_B = Λ_B Λ_Bᵀ + diag(σ²_B)` and `Σ_W = Λ_W Λ_Wᵀ + diag(σ²_W)`; number of
individuals `nindiv`; maximised `loglik`; `converged`; `iterations`.
"""
struct TwoLevelFit
    Λ_B::Matrix{Float64}
    σ²_B::Vector{Float64}
    Λ_W::Matrix{Float64}
    σ²_W::Vector{Float64}
    Σ_B::Matrix{Float64}
    Σ_W::Matrix{Float64}
    nindiv::Int
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::TwoLevelFit)
    p, K_B = size(f.Λ_B)
    K_W = size(f.Λ_W, 2)
    print(io, "TwoLevelFit(p=", p, ", K_B=", K_B, ", K_W=", K_W,
          ", nindiv=", f.nindiv, ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_twolevel_gaussian(y, individual; K_B, K_W, center=true, …) -> TwoLevelFit

Fit the Gaussian two-level reduced-rank GLLVM (behavioural-syndromes
decomposition). `y` is `p × n_obs` (observation-major columns); `individual` a
length-`n_obs` grouping vector. `K_B` / `K_W` are the between- / within-individual
reduced-rank dimensions. By default the per-trait grand mean is removed
(`center=true`) so μ is profiled out; pass `center=false` if `y` is already
centred. Optimises `[θ_rr_B; logσ²_B; θ_rr_W; logσ²_W]` on the per-individual
rotation-trick marginal (direct ForwardDiff; guarded BackTracking line search).

Recovers `Σ_B = Λ_B Λ_Bᵀ + diag(σ²_B)` and `Σ_W = Λ_W Λ_Wᵀ + diag(σ²_W)`; pass
the result to [`repeatability`](@ref), [`communality_B`](@ref) /
[`communality_W`](@ref), and [`correlation_B`](@ref) / [`correlation_W`](@ref).
"""
function fit_twolevel_gaussian(y::AbstractMatrix, individual::AbstractVector;
        K_B::Integer, K_W::Integer, center::Bool = true,
        σ²_B_init::Real = 0.5, σ²_W_init::Real = 0.5,
        g_tol::Real = 1e-8, iterations::Integer = 1000)
    p, ntot = size(y)
    length(individual) == ntot ||
        throw(DimensionMismatch("individual length must equal n_obs = $ntot"))
    K_B ≥ 1 && K_W ≥ 1 || throw(ArgumentError("K_B and K_W must be ≥ 1"))

    yf = Matrix{Float64}(y)
    if center
        μ = vec(sum(yf, dims = 2)) ./ ntot
        yf = yf .- reshape(μ, p, 1)
    end

    codes, _ = _code_grouping(individual)
    L = maximum(codes)
    ind_idx = [findall(==(g), codes) for g in 1:L]

    rrB = rr_theta_len(p, K_B)
    rrW = rr_theta_len(p, K_W)

    # Warm-start the loadings from PPCA on the full y (split the total variance
    # roughly between the two levels via the diag inits). The PPCA loadings give a
    # sane scale; the optimiser refines the between/within split.
    Λ0, _ = ppca_init(yf, max(K_B, K_W))
    θ0 = vcat(pack_lambda(Λ0[:, 1:K_B] ./ sqrt(2)),
              fill(log(float(σ²_B_init)), p),
              pack_lambda(Λ0[:, 1:K_W] ./ sqrt(2)),
              fill(log(float(σ²_W_init)), p))

    nll = θ -> begin
        v = try
            Λ_B, σ²_B, Λ_W, σ²_W = _twolevel_unpack(θ, p, K_B, K_W)
            -_twolevel_loglik(yf, ind_idx, Λ_B, σ²_B, Λ_W, σ²_W)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end

    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(nll, θ0, ls,
                         Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :forward)
    th = Optim.minimizer(res)
    Λ_B, σ²_B, Λ_W, σ²_W = _twolevel_unpack(th, p, K_B, K_W)
    Λ_Bf = Matrix{Float64}(Λ_B); σ²_Bf = Vector{Float64}(σ²_B)
    Λ_Wf = Matrix{Float64}(Λ_W); σ²_Wf = Vector{Float64}(σ²_W)
    Σ_B = Matrix{Float64}(_dense_sigma(Λ_Bf, σ²_Bf))
    Σ_W = Matrix{Float64}(_dense_sigma(Λ_Wf, σ²_Wf))
    return TwoLevelFit(Λ_Bf, σ²_Bf, Λ_Wf, σ²_Wf, Σ_B, Σ_W, L,
                       _fit_verdict(res)...)
end

# ---------------------------------------------------------------------------
# Two-level extractors.
# ---------------------------------------------------------------------------

"""
    repeatability(fit::TwoLevelFit) -> Vector

Per-trait repeatability `R_t = Σ_B[t,t] / (Σ_B[t,t] + Σ_W[t,t])` — the share of
the total per-trait variance attributable to stable between-individual
differences (the among-individual / total partition). Values in [0, 1].
"""
function repeatability(fit::TwoLevelFit)
    p = size(fit.Σ_B, 1)
    return [fit.Σ_B[t, t] / (fit.Σ_B[t, t] + fit.Σ_W[t, t]) for t in 1:p]
end

"""
    communality_B(fit::TwoLevelFit) -> Vector

Per-trait between-level communality `c²_B,t = (Λ_B Λ_Bᵀ)[t,t] / Σ_B[t,t]` — the
share of the between-individual trait variance carried by the shared between
loadings (vs the per-trait residual σ²_B). Values in [0, 1].
"""
function communality_B(fit::TwoLevelFit)
    ΛΛt = fit.Λ_B * fit.Λ_B'
    p = size(fit.Σ_B, 1)
    return [ΛΛt[t, t] / fit.Σ_B[t, t] for t in 1:p]
end

"""
    communality_W(fit::TwoLevelFit) -> Vector

Per-trait within-level communality `c²_W,t = (Λ_W Λ_Wᵀ)[t,t] / Σ_W[t,t]` — the
share of the within-individual (observation-level) trait variance carried by the
shared within loadings (vs the per-trait residual σ²_W). Values in [0, 1].
"""
function communality_W(fit::TwoLevelFit)
    ΛΛt = fit.Λ_W * fit.Λ_W'
    p = size(fit.Σ_W, 1)
    return [ΛΛt[t, t] / fit.Σ_W[t, t] for t in 1:p]
end

# Standardise a covariance to a correlation (diagonal exactly 1; NaN on Σ_tt ≤ 0).
function _to_correlation(Σ::AbstractMatrix)
    p = size(Σ, 1)
    R = Matrix{Float64}(undef, p, p)
    @inbounds for j in 1:p, i in 1:p
        d = Σ[i, i] * Σ[j, j]
        R[i, j] = (Σ[i, i] > 0 && Σ[j, j] > 0) ? Σ[i, j] / sqrt(d) : NaN
    end
    return R
end

"""
    correlation_B(fit::TwoLevelFit) -> Matrix

Between-individual cross-trait correlation `C_B = D_B^{-1/2} Σ_B D_B^{-1/2}` (the
behavioural-syndrome correlation matrix), `Σ_B = Λ_B Λ_Bᵀ + diag(σ²_B)`.
"""
correlation_B(fit::TwoLevelFit) = _to_correlation(fit.Σ_B)

"""
    correlation_W(fit::TwoLevelFit) -> Matrix

Within-individual cross-trait correlation `C_W = D_W^{-1/2} Σ_W D_W^{-1/2}` (the
observation-level / state correlation matrix), `Σ_W = Λ_W Λ_Wᵀ + diag(σ²_W)`.
"""
correlation_W(fit::TwoLevelFit) = _to_correlation(fit.Σ_W)

# ---------------------------------------------------------------------------
# Confidence intervals for the two-level repeatability / ICC
#   R_t = Σ_B[t,t] / (Σ_B[t,t] + Σ_W[t,t]),
# mirroring R gllvmTMB's `extract_repeatability()`
# (.unlazy/core070-aghq/oracle-source/readback/R/extract-repeatability.R).
#
# R's Wald route runs the delta method on `log_v[t] = log(vB[t]) - log(vW[t])`
# and back-transforms with `plogis`; since `plogis(log(a) - log(b)) ==
# a / (a + b)`, that is EXACTLY this repo's established logit-transformed-Wald
# convention for [0,1] ICC-type quantities (src/confint_derived_wald.jl
# header). We reuse `_tw_link(:logit)` and the same delta-method shape, built
# on the packed-θ closure `g(θ) = log(Σ_B[t,t](θ)) - log(Σ_W[t,t](θ))`.
#
# R's `method = "profile"` is a WITHDRAWN feature: it aborts with class
# `gllvmTMB_repeatability_profile_withdrawn` rather than silently falling
# back to a different estimand or method. `repeatability_ci(method =
# :profile)` mirrors that refusal with a named Julia exception instead of
# inventing a profile route.
# ---------------------------------------------------------------------------

"""
    TwoLevelRepeatabilityProfileWithdrawn

Thrown by [`repeatability_ci`](@ref) when `method = :profile` is requested.
Mirrors R gllvmTMB's `extract_repeatability(method = "profile")`, which
aborts with class `gllvmTMB_repeatability_profile_withdrawn`: a defensible
profile interval for canonical full-covariance repeatability is not
available (the former profile route estimated only a diagonal-companion
ratio and omitted the shared-latent variance). Request `method = :wald` or
`method = :bootstrap` instead.
"""
struct TwoLevelRepeatabilityProfileWithdrawn <: Exception
    msg::String
end
TwoLevelRepeatabilityProfileWithdrawn() = TwoLevelRepeatabilityProfileWithdrawn(
    "A profile interval for canonical full-covariance two-level " *
    "repeatability is not currently available. The former profile route " *
    "estimated only a diagonal-companion ratio and omitted shared latent " *
    "variance. Request method = :wald or method = :bootstrap instead.")
Base.showerror(io::IO, e::TwoLevelRepeatabilityProfileWithdrawn) = print(io, e.msg)

# Repack fit's (Λ_B, σ²_B, Λ_W, σ²_W) into the same packed-θ layout
# fit_twolevel_gaussian optimises: [θ_rr_B; log σ²_B; θ_rr_W; log σ²_W].
function _twolevel_theta_at_mle(fit::TwoLevelFit)
    return vcat(pack_lambda(fit.Λ_B), log.(fit.σ²_B),
                pack_lambda(fit.Λ_W), log.(fit.σ²_W))
end

# log(Σ_B[t,t]) - log(Σ_W[t,t]) from the packed θ. AD-friendly (unpack_lambda,
# exp, and _dense_sigma all propagate ForwardDiff Duals).
function _repeatability_log_odds_packed(θ::AbstractVector, p::Integer,
                                        K_B::Integer, K_W::Integer, t::Integer)
    Λ_B, σ²_B, Λ_W, σ²_W = _twolevel_unpack(θ, p, K_B, K_W)
    Σ_B = _dense_sigma(Λ_B, σ²_B)
    Σ_W = _dense_sigma(Λ_W, σ²_W)
    return log(Σ_B[t, t]) - log(Σ_W[t, t])
end

"""
    repeatability_wald_ci(fit::TwoLevelFit, y, individual; level=0.95,
                          center=true) -> Vector{NamedTuple}

Logit transformed-Wald CI for the per-trait two-level repeatability `R_t =
Σ_B[t,t] / (Σ_B[t,t] + Σ_W[t,t])`, mirroring R's
`extract_repeatability(method = "wald")`. Delta method on `log_v[t] =
log(Σ_B[t,t]) - log(Σ_W[t,t])`, back-transformed with the logistic — bounds
are guaranteed to lie in `(0, 1)`. The observed information is the
ForwardDiff Hessian of the packed two-level NLL reconstructed from `y` and
`individual`, evaluated at `fit`'s converged `[θ_rr_B; log σ²_B; θ_rr_W;
log σ²_W]` — the same objective `fit_twolevel_gaussian` optimised.

`y`, `individual` must be the arrays passed to `fit_twolevel_gaussian`
(this struct does not retain them), and `center` must match the `center`
flag used at fit time.

Returns a `NamedTuple` per trait with fields `estimate` (`R_t`), `lower`,
`upper`, `se_transformed` (SE on the log-odds scale), `transform = :logit`,
`pd_hessian`, `method`.
"""
function repeatability_wald_ci(fit::TwoLevelFit, y::AbstractMatrix,
                               individual::AbstractVector;
                               level::Real = 0.95, center::Bool = true)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    p, ntot = size(y)
    K_B = size(fit.Λ_B, 2)
    K_W = size(fit.Λ_W, 2)
    length(individual) == ntot ||
        throw(DimensionMismatch("individual length must equal n_obs = $ntot"))

    yf = Matrix{Float64}(y)
    if center
        μ = vec(sum(yf, dims = 2)) ./ ntot
        yf = yf .- reshape(μ, p, 1)
    end
    codes, _ = _code_grouping(individual)
    L = maximum(codes)
    ind_idx = [findall(==(g), codes) for g in 1:L]

    θ̂ = _twolevel_theta_at_mle(fit)
    nll = θ -> begin
        Λ_B, σ²_B, Λ_W, σ²_W = _twolevel_unpack(θ, p, K_B, K_W)
        -_twolevel_loglik(yf, ind_idx, Λ_B, σ²_B, Λ_W, σ²_W)
    end
    H = try
        ForwardDiff.hessian(nll, θ̂)
    catch
        nothing
    end
    Σcov, pd = if H !== nothing && all(isfinite, H)
        try
            (inv((H .+ H') ./ 2), true)
        catch
            (nothing, false)
        end
    else
        (nothing, false)
    end

    h, back, _ = _tw_link(:logit)
    z = quantile(Normal(), 0.5 + level / 2)
    R = repeatability(fit)
    out = Vector{NamedTuple}(undef, p)
    for t in 1:p
        g = θ -> _repeatability_log_odds_packed(θ, p, K_B, K_W, t)
        failed = (; estimate = R[t], lower = NaN, upper = NaN,
                  se_transformed = NaN, transform = :logit,
                  pd_hessian = false, method = :failed)
        if Σcov === nothing || !pd
            out[t] = failed
            continue
        end
        grad = try
            ForwardDiff.gradient(g, θ̂)
        catch
            nothing
        end
        if grad === nothing || !all(isfinite, grad)
            out[t] = merge(failed, (; pd_hessian = true))
            continue
        end
        var_h = dot(grad, Σcov * grad)
        if !isfinite(var_h) || var_h < 0
            out[t] = merge(failed, (; pd_hessian = true))
            continue
        end
        se_h = sqrt(var_h)
        h_hat = g(θ̂)
        out[t] = (; estimate = back(h_hat), lower = back(h_hat - z * se_h),
                  upper = back(h_hat + z * se_h), se_transformed = se_h,
                  transform = :logit, pd_hessian = true, method = :transformed_wald)
    end
    return out
end

"""
    _psd_sqrt_factor(Σ::AbstractMatrix) -> Matrix

A PSD-safe square-root factor `L` with `L*L' ≈ Σ`, for simulating
`N(0, Σ)` draws (`L * randn(size(Σ,1))`) when `Σ` is only positive
SEMI-definite (e.g. a boundary two-level fit with `σ²_B == 0` and
`K_B < p`, where `Σ_B = Λ_B Λ_Bᵀ + diag(σ²_B)` is rank-deficient and
plain `cholesky` throws `PosDefException`). Uses the eigendecomposition
of the symmetrized `Σ` with tiny/negative eigenvalues clamped to `0`
(never a jitter term added to the diagonal) — any orthonormal-basis
square root works for simulation, `L` need not be lower-triangular.
"""
function _psd_sqrt_factor(Σ::AbstractMatrix)
    Σs = Symmetric((Σ .+ Σ') ./ 2)
    ev = eigen(Σs)
    λ = clamp.(ev.values, 0.0, Inf)
    return Matrix(ev.vectors) * Diagonal(sqrt.(λ))
end

"""
    repeatability_bootstrap_ci(fit::TwoLevelFit, individual; nsim=200,
                               level=0.95, seed=nothing, center=true)
        -> Vector{NamedTuple}

Parametric-bootstrap CI for the per-trait two-level repeatability `R_t`,
mirroring R's `extract_repeatability(method = "bootstrap")` /
`bootstrap_Sigma(what = "ICC")`. Simulates `nsim` two-level datasets from
`fit`'s fitted `(Λ_B, σ²_B, Λ_W, σ²_W)` — same per-individual group sizes as
`individual` — refits [`fit_twolevel_gaussian`](@ref) at each replicate
(warm-started at `fit`'s own converged `K_B`/`K_W`), computes
[`repeatability`](@ref) on each successful refit, and takes the
`(1-level)/2`, `1-(1-level)/2` empirical percentiles (linear-interpolation,
matching `_derived_percentile` / R `type = 7`).

Returns a `NamedTuple` per trait with fields `estimate` (`R_t` on the
ORIGINAL fit), `lower`, `upper`, `n_boot` (successful replicates), `method
= :bootstrap`.

`Σ_B = Λ_B Λ_Bᵀ + diag(σ²_B)` is only positive SEMI-definite at a boundary
fit (e.g. `σ²_B == 0` with `K_B < p`); this simulates from it via the
PSD-safe [`_psd_sqrt_factor`](@ref) (eigendecomposition, tiny/negative
eigenvalues clamped to `0` — never a jitter term) rather than a plain
`cholesky`, which throws `PosDefException` and would otherwise abort the
whole call. If even that safe factorisation is not possible (non-finite
`Σ_B`/`Σ_W`), every trait's row falls back to the standard NaN-row
convention used elsewhere in this file (`lower = upper = NaN`, `n_boot =
0`) instead of raising.
"""
function repeatability_bootstrap_ci(fit::TwoLevelFit, individual::AbstractVector;
                                    nsim::Integer = 200, level::Real = 0.95,
                                    seed::Union{Nothing, Integer} = nothing,
                                    center::Bool = true)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    p = size(fit.Λ_B, 1)
    K_B = size(fit.Λ_B, 2)
    K_W = size(fit.Λ_W, 2)
    codes, _ = _code_grouping(individual)
    L = maximum(codes)
    ind_idx = [findall(==(g), codes) for g in 1:L]
    ntot = length(individual)

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    R_hat = repeatability(fit)
    LB, LW = try
        (_psd_sqrt_factor(fit.Σ_B), _psd_sqrt_factor(fit.Σ_W))
    catch
        # Σ_B / Σ_W is boundary-degenerate in a way even the PSD-safe
        # eigendecomposition cannot simulate from (e.g. non-finite entries).
        # Never abort the whole call — report the standard NaN-row convention.
        return [(; estimate = R_hat[t], lower = NaN, upper = NaN,
                 n_boot = 0, method = :bootstrap) for t in 1:p]
    end

    R_reps = [Float64[] for _ in 1:p]
    n_boot = 0
    for _ in 1:nsim
        y_sim = Matrix{Float64}(undef, p, ntot)
        for idx in ind_idx
            b_i = LB * randn(rng, p)
            for j in idx
                y_sim[:, j] = b_i .+ LW * randn(rng, p)
            end
        end
        rep = try
            fit_twolevel_gaussian(y_sim, individual; K_B = K_B, K_W = K_W,
                                  center = center)
        catch
            nothing
        end
        (rep === nothing || !rep.converged) && continue
        n_boot += 1
        Rrep = repeatability(rep)
        for t in 1:p
            push!(R_reps[t], Rrep[t])
        end
    end

    α = 1 - level
    out = Vector{NamedTuple}(undef, p)
    for t in 1:p
        if length(R_reps[t]) < 2
            out[t] = (; estimate = R_hat[t], lower = NaN, upper = NaN,
                      n_boot = length(R_reps[t]), method = :bootstrap)
        else
            out[t] = (; estimate = R_hat[t],
                      lower = _derived_percentile(R_reps[t], α / 2),
                      upper = _derived_percentile(R_reps[t], 1 - α / 2),
                      n_boot = length(R_reps[t]), method = :bootstrap)
        end
    end
    return out
end

"""
    repeatability_ci(fit::TwoLevelFit, y, individual; method=:wald,
                     level=0.95, kwargs...)

Dispatcher over the two-level repeatability CI routes, matching R's
`extract_repeatability(method = ...)` argument surface:

  - `:wald` (default) — [`repeatability_wald_ci`](@ref); forwards
    `center` via `kwargs`.
  - `:bootstrap` — [`repeatability_bootstrap_ci`](@ref); forwards `nsim`,
    `seed`, `center` via `kwargs` (`y` is accepted but unused — the
    bootstrap simulates its own data from `fit`).
  - `:profile` — throws [`TwoLevelRepeatabilityProfileWithdrawn`](@ref),
    mirroring R's withdrawn-feature abort
    (`gllvmTMB_repeatability_profile_withdrawn`); it does NOT fall back to
    a different method or estimand.
"""
function repeatability_ci(fit::TwoLevelFit, y::AbstractMatrix, individual::AbstractVector;
                          method::Symbol = :wald, level::Real = 0.95, kwargs...)
    if method === :wald
        return repeatability_wald_ci(fit, y, individual; level = level, kwargs...)
    elseif method === :bootstrap
        return repeatability_bootstrap_ci(fit, individual; level = level, kwargs...)
    elseif method === :profile
        throw(TwoLevelRepeatabilityProfileWithdrawn())
    else
        throw(ArgumentError("method must be :wald, :bootstrap, or :profile; got $(method)"))
    end
end
