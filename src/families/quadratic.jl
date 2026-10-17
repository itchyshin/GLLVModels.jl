# Quadratic-response GLLVM (species optima / tolerances).
#
# In the linear Laplace GLLVM (src/families/laplace.jl) each species responds to
# the latent gradient through a *linear* predictor η_t(z) = β_t + Σ_k Λ_tk z_k.
# The quadratic-response model adds a per-species quadratic term:
#
#     η_t(z) = β_t + Σ_k Λ_tk z_k + Σ_k D_tk z_k²        (z ~ N(0, I_K))
#
# When D_tk < 0 the response is unimodal along latent axis k with an optimum
# (niche centre) at z_k* = −Λ_tk / (2 D_tk) and a tolerance (niche width) set by
# the curvature; D_tk > 0 gives a U-shaped (anti-optimum) response. This is the
# ecological "optimum/tolerance" parameterisation popularised by canonical and
# quadratic ordination (van der Aart & Smeenk-Enserink; ter Braak's Gaussian
# response model), here as a latent-variable GLLVM.
#
# Mathematically this is *the linear Laplace model with a z-dependent Jacobian*:
#     J_tk(z) = ∂η_t/∂z_k = Λ_tk + 2·D_tk·z_k.
# Everything else — the family score/weight/log-density, the μ/η clamps, the SPD
# Gauss–Newton (expected-information) Hessian and its logdet — is reused verbatim
# from the family pieces. The mode-finder differs only in that J is recomputed at
# each Fisher-scoring step (in the linear model J = Λ is constant).
#
# Reduction: at D = 0, J = Λ for all z, A = Λ'WΛ + I, and η = β + Λz, so each
# site's quadratic Laplace term is byte-for-byte the linear `laplace_loglik_site`,
# and `quadratic_marginal_loglik_laplace` reduces EXACTLY to
# `marginal_loglik_laplace` (tested at atol = 1e-8).

# --- per-site quadratic Laplace (mirrors families/laplace.jl, recomputing J) ---

# Inner Fisher-scoring mode-finder. Returns the conditional mode ẑ (length K) for
# one site under the quadratic predictor. Mirrors `_laplace_mode` but rebuilds the
# Jacobian J = Λ + 2·D·diag(z) every iteration.
function _quadratic_mode(family, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, D::AbstractMatrix, β::AbstractVector, link::Link;
        maxiter::Integer = 100, tol::Real = 1e-9)
    K = size(Λ, 2)
    z = zeros(K)
    for _ in 1:maxiter
        η  = _clamp_eta.(β .+ Λ * z .+ D * (z .^ 2))
        μ  = _clamp_mu.(Ref(family), linkinv.(Ref(link), η))
        me = mu_eta.(Ref(link), η)
        s  = _glm_score.(Ref(family), μ, n, me, y)
        W  = _glm_weight.(Ref(family), μ, n, me)
        J  = Λ .+ 2 .* D .* z'                        # p×K, J[t,k] = Λ[t,k] + 2 D[t,k] z[k]
        A  = Symmetric(J' * (W .* J) + I)             # K×K Gauss–Newton (expected-info) Hessian, SPD
        Δ  = _safe_solve(A, J' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break  # singular A ⇒ stop at current ẑ
        z  = z .+ Δ
        maximum(abs, Δ) < tol && break
    end
    return z
end

"""
    quadratic_loglik_site(family, y, n, Λ, D, β, link; maxiter=100, tol=1e-9) -> Float64

Laplace-approximated log-marginal for one site of a quadratic-response GLLVM. The
latent predictor is `η_t(z) = β_t + Σ_k Λ_tk z_k + Σ_k D_tk z_k²`. `Λ`, `D` are
both p×K (`D` are the per-species quadratic coefficients). Returns
`ℓ(ẑ) − ½ẑ'ẑ − ½logdet(J'WJ + I)`, with the Jacobian `J[t,k] = Λ_tk + 2 D_tk ẑ_k`
evaluated at the converged mode `ẑ`. At `D = 0` this equals `laplace_loglik_site`.
"""
function quadratic_loglik_site(family, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, D::AbstractMatrix, β::AbstractVector, link::Link;
        hessian::Symbol = _default_hessian(family, link),
        maxiter::Integer = 100, tol::Real = 1e-9)
    (hessian === :fisher || hessian === :observed) || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    p = size(Λ, 1)
    z  = _quadratic_mode(family, y, n, Λ, D, β, link; maxiter = maxiter, tol = tol)
    η  = _clamp_eta.(β .+ Λ * z .+ D * (z .^ 2))
    μ  = _clamp_mu.(Ref(family), linkinv.(Ref(link), η))
    me = mu_eta.(Ref(link), η)
    # This kernel builds its own J'WJ + I and logdet, so the generic core's
    # selector never reached it. The MODE SEARCH (`_quadratic_mode`) stays on
    # Fisher; only the log-det takes the selector.
    W  = if hessian === :fisher || _glm_weight_matches_observed(family, link)
        _glm_weight.(Ref(family), μ, n, me)
    else
        [_glm_obs_weight(family, μ[t], n[t], me[t], y[t], link, η[t]) for t in 1:p]
    end
    J  = Λ .+ 2 .* D .* z'
    A  = Symmetric(J' * (W .* J) + I)
    ℓ = 0.0
    @inbounds for t in 1:p
        ℓ += _glm_logpdf(family, μ[t], n[t], y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * _laplace_logdet(A)
end

"""
    quadratic_marginal_loglik_laplace(family, Y, N, Λ, D, β, link; maxiter=100, tol=1e-9) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a **quadratic-response**
GLLVM, where the latent predictor carries a per-species quadratic term:

    η_{ts}(z_s) = β_t + Σ_k Λ_tk z_{sk} + Σ_k D_tk z_{sk}²,   z_s ~ N(0, I_K).

`family` is a `Distributions` marker (e.g. `Poisson()`); `Y`, `N` are the p×n
response and trial-count matrices (`N` is ignored by families without trials);
`Λ` and `D` are both p×K; `β` length-p; `link` a `Link`. When `D < 0` species `t`
has a unimodal response with optimum at `−Λ_tk/(2 D_tk)` along latent axis `k`.

At `D = 0` (the linear model) this reduces EXACTLY to `marginal_loglik_laplace`,
because the Jacobian `J = Λ` becomes constant in `z`.
"""
function quadratic_marginal_loglik_laplace(family, Y::AbstractMatrix, N::AbstractMatrix,
        Λ::AbstractMatrix, D::AbstractMatrix, β::AbstractVector, link::Link; kwargs...)
    size(D) == size(Λ) ||
        throw(DimensionMismatch("D has size $(size(D)) but Λ has size $(size(Λ))"))
    acc = 0.0
    @inbounds for s in axes(Y, 2)
        acc += quadratic_loglik_site(family, view(Y, :, s), view(N, :, s),
                                     Λ, D, β, link; kwargs...)
    end
    return acc
end

# ---------------------------------------------------------------------------
# Fit driver.
# ---------------------------------------------------------------------------

"""
    QuadraticFit

Result of [`fit_quadratic_gllvm`](@ref): a quadratic-response GLLVM fit. Fields:
`family` (the Distributions marker), per-species intercepts `β` (length p), linear
loadings `Λ` (p×K), quadratic coefficients `D` (p×K), `dispersion` (`r`/`φ`/`α`, or
`NaN` when the family has none), `link`, the maximised Laplace `loglik`,
`converged`, and `iterations`.
"""
struct QuadraticFit
    family::Distribution
    β::Vector{Float64}
    Λ::Matrix{Float64}
    D::Matrix{Float64}
    dispersion::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
end

# ---------------------------------------------------------------------------
# Post-fit ordination: getLV / predict (parallel to the linear families in
# src/postfit.jl, but the per-site mode is the quadratic Fisher-scoring solve
# `_quadratic_mode`, and the predictor carries the per-species quadratic term).
# ---------------------------------------------------------------------------

_loadings(fit::QuadraticFit) = fit.Λ
_loglik(fit::QuadraticFit)   = fit.loglik

# Free params: β (p) + reduced linear loadings Λ + quadratic coefficients D (p×K,
# full — the quadratic term breaks the rotational symmetry, so D is not reduced)
# + a dispersion (where the family has one).
function _nparams(fit::QuadraticFit)
    p, K = size(fit.Λ)
    return p + (p * K - div(K * (K - 1), 2)) + p * K + (isnan(fit.dispersion) ? 0 : 1)
end

"""
    getLV(fit::QuadraticFit, Y; rotate=true) -> n×K matrix

Conditional latent-variable scores for a quadratic-response fit: the per-site mode
`ẑₛ` from the quadratic Fisher-scoring solve (`_quadratic_mode`) at the fitted
`(Λ, D, β)` and dispersion. `Y` is the `p×n` response matrix; `rotate=true`
applies the canonical [`rotation`](@ref).
"""
function getLV(fit::QuadraticFit, Y::AbstractMatrix{<:Real}; rotate::Bool = true)
    p, n = size(Y)
    K = size(fit.Λ, 2)
    fam = _cov_family(fit.family, fit.dispersion)
    ones_p = ones(Int, p)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _quadratic_mode(fam, view(Y, :, s), ones_p, fit.Λ, fit.D, fit.β, fit.link)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::QuadraticFit, Y; type=:response) -> p×n matrix

In-sample fitted values at the quadratic conditional mode `ẑ` (see [`getLV`](@ref)):
`type=:link` returns the linear predictor `η = β + Λ ẑ + D ẑ²`; `type=:response`
applies the inverse link to the (clamped) `η`.
"""
function predict(fit::QuadraticFit, Y::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    Z = getLV(fit, Y; rotate = false)                 # n×K
    η = fit.β .+ fit.Λ * Z' .+ fit.D * ((Z') .^ 2)    # p×n
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

function Base.show(io::IO, f::QuadraticFit)
    p, K = size(f.Λ)
    print(io, "QuadraticFit(", nameof(typeof(f.family)), ", p=", p, ", K=", K)
    isnan(f.dispersion) || print(io, ", disp=", round(f.dispersion; sigdigits = 4))
    print(io, ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_quadratic_gllvm(Y; family=Poisson(), K, link=nothing, N=nothing, …) -> QuadraticFit

Fit a **quadratic-response** GLLVM by L-BFGS over `θ = [β; pack_lambda(Λ); vec(D); (log-dispersion)]`
on the quadratic Laplace marginal (`quadratic_marginal_loglik_laplace`). The latent
predictor is `η_{ts} = β_t + Σ_k Λ_tk z_{sk} + Σ_k D_tk z_{sk}²` with `z_s ~ N(0, I_K)`;
the per-species quadratic coefficients `D` (p×K) encode species optima/tolerances
(`D_tk < 0` ⇒ unimodal, optimum at `−Λ_tk/(2 D_tk)`).

`family` is a `Distributions` marker — `Poisson()`, `NegativeBinomial()`,
`Binomial()`, `Beta()`, or `Gamma()` — and dispatches the marginal (its dispersion,
where present, is jointly estimated). `Y` is `p × n`; `N` supplies Binomial trial
counts (default all-ones). Finite-difference gradient (the Laplace inner mode-finder
is not forward-AD-friendly). Warm start: empirical link-scale intercepts `β0`
(`_cov_zemp` row means), an SVD loadings init `Λ0`, and `D0 = 0` — i.e. the fit
starts at the linear model and lets the optimiser bend in the quadratic surface.

```julia
fit = fit_quadratic_gllvm(Y; family = Poisson(), K = 2)
fit.D            # estimated quadratic (optimum/tolerance) coefficients, p×K
```
"""
function fit_quadratic_gllvm(Y::AbstractMatrix{<:Real}; family = Poisson(),
        K::Integer, link::Union{Nothing, Link} = nothing,
        N::Union{Nothing, AbstractMatrix} = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    rr = rr_theta_len(p, K)
    lk = link === nothing ? _cov_default_link(family) : link
    Nm = N === nothing ? fill(1, p, n) : N
    has_disp = _cov_has_disp(family)
    pK = p * K

    # warm start: link-scale empirical intercepts + SVD loadings, D0 = 0 (linear model)
    Zemp = _cov_zemp(family, Y, Nm, lk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    D0 = zeros(p, K)

    θ0 = has_disp ? vcat(β0, pack_lambda(Λ0), vec(D0), log(_cov_disp_init(family))) :
                    vcat(β0, pack_lambda(Λ0), vec(D0))
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        D = reshape(θ[(p + rr + 1):(p + rr + pK)], p, K)
        disp = has_disp ? exp(θ[p + rr + pK + 1]) : NaN
        fam = _cov_family(family, disp)
        v = try
            -quadratic_marginal_loglik_laplace(fam, Y, Nm, Λ, D, β, lk;
                                               maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    D̂ = reshape(θ̂[(p + rr + 1):(p + rr + pK)], p, K)
    disp̂ = has_disp ? exp(θ̂[p + rr + pK + 1]) : NaN
    return QuadraticFit(family, β̂, Λ̂, D̂, disp̂, lk, _fit_verdict(res)...)
end
