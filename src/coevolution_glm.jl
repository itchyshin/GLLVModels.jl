# Cross-family (non-Gaussian) cross-lineage coevolution — Track T4.
#
# Drives the cross-lineage kernel K* (make_cross_kernel) through a non-Gaussian
# Laplace so coevolution works for the GLM families (Binomial / Poisson / NB /
# Gamma / Beta / …), not just the Gaussian matrix-normal case. This is the
# non-Gaussian companion to fit_coevolution_gaussian (the Kronecker oracle).
#
# Model (Kronecker / per-species-factor orientation):
#   * n species (host block 1:n_H, partner block n_H+1:n, the K* ordering),
#     T traits (rows of Y), optionally several site columns per species;
#   * a per-species latent factor z_j ∈ ℝ^d whose d axes are iid and each axis is
#     correlated across species by K*: the prior precision over the stacked latent
#     Z (n×d) is  P = (σ²_phy K*)⁻¹ ⊗ I_d  (column-stacked vec(Z));
#   * η[t,j] = β_t + (Λ z_j)[t],  Y[t,j] ~ Family(linkinv(η[t,j])),  Λ a T×d trait
#     loading matrix; the family supplies the per-observation dispersion.
# The coevolution estimand is the host-trait × partner-trait block
#   Γ = (Λ Λᵀ)[1:T_H, (T_H+1):T]  (rotation-invariant; mirrors extract_Gamma).
#
# Marginal via a dense joint Laplace over Z (the design's "dense path first" —
# moderate p): mode Ẑ by Fisher-scoring Newton with Hessian H = P + J (J the
# block-diagonal-in-species data Fisher information), then
#   log p(Y) ≈ ℓ(Ẑ) − ½ vec(Ẑ)ᵀ P vec(Ẑ) + ½ logdet P − ½ logdet H.
# Reuses the family dispatch (_glm_score / _glm_weight / _glm_logpdf / _clamp_mu)
# and the per-cell observation mask, so all families and block-NA come "for free".
#
# Verification anchors (test_coevolution_glm.jl):
#   * Gaussian limit (Normal/IdentityLink): the Laplace is EXACT for the linear-
#     Gaussian model, reproducing the dense closed form N(0, σ²_phy K*⊗ΛΛᵀ + σ²I)
#     to machine precision. (fit_coevolution_gaussian fits the matrix-normal
#     K*⊗Σ_T whose noise σ²(K*⊗I) is itself K*-correlated; a real family's noise is
#     iid, so this reduces to the iid-noise Gaussian model and approaches
#     fit_coevolution_gaussian only in the degenerate σ_family→0 limit.)
#   * σ²_phy → 0 ⇒ the latent is shrunk to zero ⇒ the independent per-cell family
#     marginal.
#
# References: the gllvmTMB cross-lineage coevolution kernel; the augmented-state
# phylo-GLM Laplace (phylo_glm.jl) this mirrors; Tolkoff et al. 2018 (phylogenetic
# factor analysis, the trait⊗species identifiability of Λ).

using LinearAlgebra

# Joint Laplace mode of the latent factor matrix Z (n×d), given the dense prior
# precision Pspec = (σ²_phy K*)⁻¹ (n×n, shared across the d iid axes). Returns
# (Ẑ, cholH, P) with P = I_d ⊗ Pspec (the nd×nd column-stacked precision) and
# cholH the Cholesky of H = P + J, or (nothing, …) on a non-SPD step.
function _coevolution_glm_mode(family, Y, N, β, Λ, link, Pspec, mask, T, n, d;
                               maxiter::Integer = 50, tol::Real = 1e-9)
    nd = n * d
    P = kron(Matrix{Float64}(I, d, d), Matrix(Pspec))      # I_d ⊗ Pspec, column-stacked
    Z = zeros(n, d)
    local cholH
    g = zeros(nd)
    H = zeros(nd, nd)
    for _ in 1:maxiter
        z = vec(Z)
        g .= .-(P * z)
        H .= P
        @inbounds for j in 1:n
            # per-species score (length d) and data Hessian block (d×d)
            gj = zeros(d)
            Hj = zeros(d, d)
            for t in 1:T
                (mask === nothing || mask[t, j]) || continue
                ηtj = _clamp_eta(β[t] + dot(view(Λ, t, :), view(Z, j, :)))
                μtj = _clamp_mu(family, linkinv(link, ηtj))
                metj = mu_eta(link, ηtj)
                stj = _glm_score(family, μtj, N[t, j], metj, Y[t, j])
                wtj = _glm_weight(family, μtj, N[t, j], metj)
                for a in 1:d
                    gj[a] += Λ[t, a] * stj
                    for b in 1:d
                        Hj[a, b] += Λ[t, a] * wtj * Λ[t, b]
                    end
                end
            end
            for a in 1:d
                g[(a - 1) * n + j] += gj[a]
                for b in 1:d
                    H[(a - 1) * n + j, (b - 1) * n + j] += Hj[a, b]
                end
            end
        end
        cholH = try
            cholesky(Symmetric(H))
        catch
            return nothing, nothing, P
        end
        Δ = cholH \ g
        all(isfinite, Δ) || return nothing, nothing, P
        Z .+= reshape(Δ, n, d)
        maximum(abs, Δ) < tol && break
    end
    return Z, cholH, P
end

"""
    coevolution_glm_marginal_loglik(family, Y, N, β, Λ, σ²_phy, K_star; link,
                                    mask=nothing, maxiter=50, tol=1e-9) -> Float64

Dense Laplace marginal log-likelihood of the cross-family coevolution model. `Y`,
`N` are `T × n` (trait × species) response / trial-count matrices, `β` the
length-`T` trait intercepts, `Λ` the `T × d` trait loadings, `σ²_phy` the
phylogenetic variance scaling the kernel, `K_star` the `n × n` species cross-kernel
(`make_cross_kernel`), `family` a `Distributions` marker and `link` a `Link`. `mask`
(`T × n` Bool, or `nothing`) drops missing cells (block-NA), exactly the marginal
over the observed entries.

Returns `ℓ(Ẑ) − ½ vec(Ẑ)ᵀ P vec(Ẑ) + ½ logdet P − ½ logdet H` with
`P = (σ²_phy K*)⁻¹ ⊗ I_d` and the joint latent mode `Ẑ`; `-Inf` on a non-SPD step
or `σ²_phy ≤ 0`.
"""
function coevolution_glm_marginal_loglik(family, Y::AbstractMatrix, N::AbstractMatrix,
        β::AbstractVector, Λ::AbstractMatrix, σ²_phy::Real, K_star::AbstractMatrix;
        link::Link = default_link(family), mask = nothing,
        hessian::Symbol = _default_hessian(family, link),
        maxiter::Integer = 50, tol::Real = 1e-9)
    (hessian === :fisher || hessian === :observed) || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    T, n = size(Y)
    size(Λ, 1) == T || throw(ArgumentError("size(Λ,1)=$(size(Λ,1)) must equal size(Y,1)=$T."))
    size(N) == (T, n) ||
        throw(ArgumentError("size(N)=$(size(N)) must equal size(Y)=$((T, n))."))
    length(β) == T ||
        throw(ArgumentError("length(β)=$(length(β)) must equal size(Y,1)=$T."))
    (size(K_star, 1) == n && size(K_star, 2) == n) ||
        throw(ArgumentError("K_star must be n × n = $n × $n; got $(size(K_star))."))
    σ²_phy > 0 || return -Inf
    d = size(Λ, 2)

    Kf = Symmetric(Matrix(float(σ²_phy) .* K_star))
    cholKf = try
        cholesky(Kf)
    catch
        return -Inf
    end
    Pspec = inv(cholKf)                                       # (σ²_phy K*)⁻¹ (n×n, symmetric)

    Ẑ, cholH, P = _coevolution_glm_mode(family, Y, N, β, Λ, link, Pspec, mask, T, n, d;
                                        maxiter = maxiter, tol = tol)
    Ẑ === nothing && return -Inf

    ℓ = 0.0
    @inbounds for j in 1:n, t in 1:T
        (mask === nothing || mask[t, j]) || continue
        ηtj = _clamp_eta(β[t] + dot(view(Λ, t, :), view(Ẑ, j, :)))
        μtj = _clamp_mu(family, linkinv(link, ηtj))
        ℓ += _glm_logpdf(family, μtj, N[t, j], Y[t, j])
    end
    z = vec(Ẑ)
    quad = 0.5 * dot(z, P * z)
    # logdet P = -logdet(σ²_phy K*) summed over the d iid axes = -d·logdet(Kf)
    logdetP = -d * logdet(cholKf)

    # Role separation. `_coevolution_glm_mode` returns the Cholesky of the
    # FISHER Hessian from its final Newton step, and that same factorisation was
    # previously reused for the log-det — one object serving two roles, which is
    # the architectural defect this whole programme exists to fix, one level
    # down. The Newton step legitimately wants the Fisher Hessian (SPD by
    # construction); the log-det must be the OBSERVED one to match TMB.
    #
    # So on the observed branch, rebuild H = P + J at the CONVERGED mode using
    # the observed weight and factorise it separately. That costs one extra
    # Cholesky per evaluation, paid only on that branch and only once the mode
    # is already found — the mode search itself is untouched.
    cholL = cholH
    if hessian === :observed && !_glm_weight_matches_observed(family, link)
        Hobs = copy(P)
        @inbounds for j in 1:n, t in 1:T
            (mask === nothing || mask[t, j]) || continue
            ηtj = _clamp_eta(β[t] + dot(view(Λ, t, :), view(Ẑ, j, :)))
            μtj = _clamp_mu(family, linkinv(link, ηtj))
            metj = mu_eta(link, ηtj)
            wtj = _glm_obs_weight(family, μtj, N[t, j], metj, Y[t, j], link, ηtj)
            for a in 1:d, b in 1:d
                Hobs[(a - 1) * n + j, (b - 1) * n + j] += Λ[t, a] * wtj * Λ[t, b]
            end
        end
        # PD guard: the observed weight may be negative, so H is no longer SPD by
        # construction. Fail to -Inf rather than returning a meaningless logdet.
        cholL = try
            cholesky(Symmetric(Hobs))
        catch
            return -Inf
        end
    end
    return ℓ - quad + 0.5 * logdetP - 0.5 * logdet(cholL)
end

# ---------------------------------------------------------------------------
# Fit driver (FD outer gradient, like fit_phylo_glm — the dense Laplace marginal
# is not AD-friendly through the per-step Cholesky).
# ---------------------------------------------------------------------------

"""
    CoevolutionGLMFit

Result of [`fit_coevolution_glm`](@ref): trait intercepts `β` (length T), trait
loadings `Λ` (T×d), the phylogenetic variance `σ²_phy`, the estimated `dispersion`
(σ² / r for the dispersion families, `NaN` otherwise), `n_host_traits` (the host
block size for slicing Γ; `nothing` if not supplied), the `link`, `family`, the
maximised Laplace `loglik`, the optimiser `converged` flag and `iterations`. Slice
the coevolution estimand with [`coevolution_gamma`](@ref).
"""
struct CoevolutionGLMFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    σ²_phy::Float64
    dispersion::Float64
    n_host_traits::Union{Int, Nothing}
    link::Link
    family::Any
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::CoevolutionGLMFit)
    print(io, "CoevolutionGLMFit(T=", size(f.Λ, 1), ", d=", size(f.Λ, 2),
          ", family=", nameof(typeof(f.family)),
          ", σ²_phy=", round(f.σ²_phy; sigdigits = 4),
          isnan(f.dispersion) ? "" : ", dispersion=$(round(f.dispersion; sigdigits = 4))",
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    coevolution_gamma(fit::CoevolutionGLMFit; n_host_traits) -> Matrix{Float64}

Cross-lineage coevolution estimand `Γ = (Λ Λᵀ)[1:T_H, (T_H+1):T]` from a
[`fit_coevolution_glm`](@ref) fit, where `T_H = n_host_traits` is the number of
host traits (the first rows of `Y`). `Γ` is the host-trait × partner-trait block of
the shared trait covariance `Λ Λᵀ`; it is rotation-invariant in the latent axes
(mirrors `extract_Gamma`). `n_host_traits` defaults to the value stored on the fit.
"""
function coevolution_gamma(fit::CoevolutionGLMFit;
                           n_host_traits::Union{Integer, Nothing} = fit.n_host_traits)
    n_host_traits === nothing && throw(ArgumentError(
        "n_host_traits not stored on the fit; pass `n_host_traits = T_H` explicitly."))
    T = size(fit.Λ, 1)
    T_H = Int(n_host_traits)
    (1 <= T_H < T) || throw(ArgumentError("n_host_traits must satisfy 1 ≤ T_H < T = $T."))
    Σ = fit.Λ * transpose(fit.Λ)
    return Matrix(Σ[1:T_H, (T_H + 1):T])
end

"""
    fit_coevolution_glm(Y, K_star; family=Poisson(), link=default_link(family),
                        N=nothing, d=1, n_host_traits=nothing,
                        g_tol=1e-5, iterations=300, newton_maxiter=50,
                        newton_tol=1e-9) -> CoevolutionGLMFit

Fit the cross-family (non-Gaussian) cross-lineage coevolution model by L-BFGS on
the dense Laplace marginal ([`coevolution_glm_marginal_loglik`](@ref)). `Y` is
`T × n` (trait × species; host species first, then partner, matching the
`make_cross_kernel` ordering), `K_star` the `n × n` species cross-kernel. Fits trait
intercepts `β`, the `T × d` trait loadings `Λ`, and a dispersion parameter for the
dispersion families. `Y` may contain `missing` (block-NA: each lineage measures only
its own traits) — the missing cells are dropped from the marginal. Finite-difference
outer gradient (the dense-Cholesky marginal is not AD-friendly). Slice `Γ` with
[`coevolution_gamma`](@ref).

The kernel scale `σ²_phy` is held at 1 and absorbed into `Λ`: the marginal depends
on `σ²_phy` and `Λ` only through `σ²_phy · Λ Λᵀ`, so a free `σ²_phy` would be a flat
ridge against the overall scale of `Λ` (the same scale-fold the Gaussian Kronecker
oracle `fit_coevolution_gaussian` uses, which has no `σ²_phy`). The reported
`σ²_phy` is therefore `1.0`.
"""
function fit_coevolution_glm(Y::AbstractMatrix, K_star::AbstractMatrix;
        family = Poisson(), link::Link = default_link(family),
        N = nothing, d::Integer = 1, n_host_traits = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 300,
        newton_maxiter::Integer = 50, newton_tol::Real = 1e-9)
    T, n = size(Y)
    (size(K_star, 1) == n && size(K_star, 2) == n) ||
        throw(ArgumentError("K_star must be n × n = $n × $n; got $(size(K_star))."))
    d ≥ 1 || throw(ArgumentError("d must be ≥ 1."))
    d ≤ T || throw(ArgumentError("d must be ≤ T = $T."))

    mask = _resolve_obs_mask(nothing, Y)                     # nothing if fully observed
    Yc = _sanitize_missing(Y, zero(eltype(skipmissing(Y))))  # placeholder for masked cells
    if N !== nothing
        size(N) == (T, n) ||
            throw(ArgumentError("size(N)=$(size(N)) must equal size(Y)=$((T, n))."))
    end
    Ntr = N === nothing ? ones(Float64, T, n) : Matrix{Float64}(N)
    nd = _spde_disp_len(family)

    # warm start: per-trait link-scale mean over observed cells; small random Λ.
    β0 = Vector{Float64}(undef, T)
    for t in 1:T
        cnt = 0; acc = 0.0
        for j in 1:n
            (mask === nothing || mask[t, j]) || continue
            acc += linkfun(link, _ws_mean(family, Yc[t, j], Ntr[t, j]))
            cnt += 1
        end
        β0[t] = cnt > 0 ? acc / cnt : 0.0
    end
    # σ²_phy ≡ 1 (folded into Λ — see the docstring); params = [β; vec(Λ); disp].
    Λ0 = 0.1 .* randn(T, d)
    dbase = T + T * d                                         # first dispersion-param index − 0
    θ0 = vcat(β0, vec(Λ0), _spde_disp_init(family, Yc))

    function negll(θ)
        β = θ[1:T]
        Λ = reshape(view(θ, (T + 1):(T + T * d)), T, d)
        fam = _spde_make_family(family, view(θ, (dbase + 1):(dbase + nd)))
        v = try
            -coevolution_glm_marginal_loglik(fam, Yc, Ntr, β, Λ, 1.0, K_star;
                                             link = link, mask = mask,
                                             maxiter = newton_maxiter, tol = newton_tol)
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
    β̂ = θ̂[1:T]
    Λ̂ = reshape(θ̂[(T + 1):(T + T * d)], T, d)
    disp = _spde_disp_value(family, θ̂[(dbase + 1):(dbase + nd)])
    nh = n_host_traits === nothing ? nothing : Int(n_host_traits)
    return CoevolutionGLMFit(β̂, Matrix(Λ̂), 1.0, disp, nh, link, family,
                             _fit_verdict(res)...)
end
