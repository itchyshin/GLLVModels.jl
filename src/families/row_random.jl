# RANDOM ROW EFFECTS (gllvm/gllvmTMB `row.eff = "random"`) for the non-Gaussian
# Laplace families.
#
# The fixed row-effect path (src/families/row_effects.jl) estimates a free per-site
# intercept ρ_s. This file specialises the *random* row effect — a per-SITE random
# intercept ρ_s, shared across all species at that site, drawn from a common normal:
#
#     η_{ts} = β_t + ρ_s + (Λ z_s)_t,   z_s ~ N(0, I_K),   ρ_s ~ N(0, σ_row²)  (iid s).
#
# KEY REPARAMETERIZATION (avoids any new Laplace math): write ρ_s = σ_row · u_s with
# u_s ~ N(0,1). Then
#
#     η_{ts} = β_t + (Λ̃ w_s)_t,   Λ̃ = [Λ | σ_row·𝟙_p]  (p×(K+1)),   w_s = [z_s; u_s].
#
# i.e. it is EXACTLY a standard (K+1)-latent GLLVM whose last loading column is the
# constant vector σ_row·𝟙_p, with w_s ~ N(0, I_{K+1}). The random-row marginal is
# therefore the existing generic marginal evaluated at Λ̃ — no new mode-finder, no new
# Hessian. The ordination loadings (Λ, p×K) and the row-effect scale (σ_row ≥ 0) are
# separated out in the returned fit; only σ_row enters that augmenting column.
#
# Reused verbatim from the family core / covariate plumbing: the generic
# `marginal_loglik_laplace` (src/families/laplace.jl), the `_laplace_mode` augmented
# mode-finder (for the BLUPs), and the `_cov_*` dispersion / link / warm-start helpers
# (src/families/covariates.jl). Only the Λ̃ augmentation and the σ_row parameter differ.

"""
    row_random_marginal_loglik_laplace(family, Y, N, Λ, β, σ_row; link=default_link(family),
                                       maxiter=100, tol=1e-9, mask=nothing, offset=nothing) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a non-Gaussian GLLVM with
a **random row effect** (per-site random intercept) of standard deviation `σ_row ≥ 0`:
`η_{ts} = β_t + ρ_s + (Λ z_s)_t` with `ρ_s ~ N(0, σ_row²)`.

By the reparameterization `ρ_s = σ_row·u_s` this is identically the generic
(K+1)-latent marginal at the augmented loadings `Λ̃ = [Λ | σ_row·𝟙_p]`, so it is a
one-liner over [`marginal_loglik_laplace`](@ref). `Y`, `N` are the `p×n` response and
trial-count matrices; `Λ` is `p×K` (the ordination loadings, WITHOUT the row column);
`β` length-p; `σ_row` the row-effect SD. `mask` / `offset` (both `p×n` or `nothing`)
pass straight through. With `σ_row == 0` the augmenting column is zero, the augmented
Hessian is block-diagonal `[[Λ'WΛ+I, 0],[0, 1]]` with a zero mode component, and the
value reduces *exactly* to the plain K-LV marginal `marginal_loglik_laplace(family, Y,
N, Λ, β, link)`.
"""
function row_random_marginal_loglik_laplace(family, Y::AbstractMatrix, N::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, σ_row::Real;
        link::Link = default_link(family), maxiter::Integer = 100, tol::Real = 1e-9,
        mask = nothing, offset = nothing)
    p = size(Λ, 1)
    Λ̃ = hcat(Λ, fill(float(σ_row), p))          # p×(K+1); σ_row == 0 ⇒ zero last column
    return marginal_loglik_laplace(family, Y, N, Λ̃, β, link;
                                   maxiter = maxiter, tol = tol, mask = mask, offset = offset)
end

# ---------------------------------------------------------------------------
# Fit driver — L-BFGS over [β; pack_lambda(Λ); log σ_row; (log-dispersion)].
# ---------------------------------------------------------------------------

"""
    RowRandomFit

Result of [`fit_row_random_gllvm`](@ref): a GLLVM fit with a **random row effect**
(per-site random intercept). Fields: `family` (the Distributions marker), per-species
intercepts `β` (length p), the ordination loadings `Λ` (p×K, WITHOUT the row column),
the row-effect SD `σ_row` (≥ 0), `dispersion` (`r`/`φ`/`α`, or `NaN` when the family
has none), `link`, the maximised Laplace `loglik`, `converged`, and `iterations`.
"""
struct RowRandomFit
    family::Distribution
    β::Vector{Float64}
    Λ::Matrix{Float64}
    σ_row::Float64
    dispersion::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::RowRandomFit)
    p, K = size(f.Λ)
    print(io, "RowRandomFit(", nameof(typeof(f.family)), ", p=", p, ", K=", K,
          ", σ_row=", round(f.σ_row; sigdigits = 4))
    isnan(f.dispersion) || print(io, ", disp=", round(f.dispersion; sigdigits = 4))
    print(io, ", loglik=", round(f.loglik; sigdigits = 7), f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::RowRandomFit) = fit.Λ
_loglik(fit::RowRandomFit)   = fit.loglik

# Free params: β (p) + reduced loadings Λ + log σ_row (1) + a dispersion (where present).
function _nparams(fit::RowRandomFit)
    p, K = size(fit.Λ)
    return p + (p * K - div(K * (K - 1), 2)) + 1 + (isnan(fit.dispersion) ? 0 : 1)
end

# Augmented loadings Λ̃ = [Λ | σ_row·𝟙_p] (p×(K+1)) — the (K+1)-LV reparameterization.
_augment_lambda(Λ::AbstractMatrix, σ_row::Real) = hcat(Λ, fill(float(σ_row), size(Λ, 1)))

"""
    fit_row_random_gllvm(Y; family=Poisson(), K, N=nothing, link=default_link(family),
                         σ_row_init=1.0, g_tol=1e-5, iterations=500,
                         newton_maxiter=100, newton_tol=1e-9) -> RowRandomFit

Fit a non-Gaussian GLLVM **with a random row effect** (per-site random intercept) by
L-BFGS over `[β; pack_lambda(Λ); log σ_row; (log-dispersion)]` on the Laplace marginal,
where `η_{ts} = β_t + ρ_s + (Λ z_s)_t` and `ρ_s ~ N(0, σ_row²)`.

Internally the marginal is the generic (K+1)-latent marginal at the augmented loadings
`Λ̃ = [Λ | σ_row·𝟙_p]` (the `ρ_s = σ_row·u_s` reparameterization), so no new Laplace
machinery is needed. `σ_row` is on the log scale during optimisation, so it stays ≥ 0.

`family` is a `Distributions` marker — `Poisson()`, `Binomial()`, `NegativeBinomial()`,
`Beta()`, or `Gamma()` — and dispatches the marginal (the dispersion, where present, is
jointly estimated). `Y` is `p × n`; `N` supplies Binomial trial counts (default
all-ones). Finite-difference gradient. Warm start: per-species empirical link-scale
means for `β`, an SVD (PPCA-style) loadings init, and `σ_row = σ_row_init`.

```julia
fit = fit_row_random_gllvm(Y; family = Poisson(), K = 2)
fit.σ_row                # estimated row-effect SD
row_effects(fit, Y)      # per-site row-effect BLUPs ρ̂_s
```
"""
function fit_row_random_gllvm(Y::AbstractMatrix{<:Real}; family = Poisson(),
        K::Integer, N::Union{Nothing, AbstractMatrix} = nothing,
        link::Union{Nothing, Link} = nothing, σ_row_init::Real = 1.0,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    n >= 1 || throw(ArgumentError("Y must have at least one site (column)"))
    σ_row_init > 0 || throw(ArgumentError("σ_row_init must be positive (it is fit on the log scale)"))
    rr = rr_theta_len(p, K)
    lk = link === nothing ? _cov_default_link(family) : link
    Nm = N === nothing ? fill(1, p, n) : N
    has_disp = _cov_has_disp(family)
    _check_family_response_support(family, Y, nothing; caller = "fit_row_random_gllvm", N = N)

    # warm start: per-species link-scale row means for β, SVD for Λ, σ_row_init for σ_row.
    Zemp = _cov_zemp(family, Y, Nm, lk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end

    θ0 = has_disp ? vcat(β0, pack_lambda(Λ0), log(σ_row_init), log(_cov_disp_init(family))) :
                    vcat(β0, pack_lambda(Λ0), log(σ_row_init))
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        σ_row = exp(θ[p + rr + 1])
        disp = has_disp ? exp(θ[p + rr + 2]) : NaN
        fam = _cov_family(family, disp)
        v = try
            -row_random_marginal_loglik_laplace(fam, Y, Nm, Λ, β, σ_row; link = lk,
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
    σ̂_row = exp(θ̂[p + rr + 1])
    disp̂ = has_disp ? exp(θ̂[p + rr + 2]) : NaN
    return RowRandomFit(family, β̂, Λ̂, σ̂_row, disp̂, lk, _fit_verdict(res)...)
end

# ---------------------------------------------------------------------------
# Post-fit: ordination scores (getLV) + row-effect BLUPs (row_effects / predict).
# ---------------------------------------------------------------------------

# Per-site augmented Laplace mode ŵ_s = [ẑ_s; û_s] (length K+1) at the fitted Λ̃.
function _row_random_modes(fit::RowRandomFit, Y::AbstractMatrix{<:Real},
                           N::Union{Nothing, AbstractMatrix})
    p, n = size(Y); K = size(fit.Λ, 2)
    Nm = N === nothing ? fill(1, p, n) : N
    fam = _cov_family(fit.family, fit.dispersion)
    Λ̃ = _augment_lambda(fit.Λ, fit.σ_row)
    W = Matrix{Float64}(undef, K + 1, n)
    @inbounds for s in 1:n
        W[:, s] = _laplace_mode(fam, view(Y, :, s), view(Nm, :, s), Λ̃, fit.β, fit.link)
    end
    return W                                       # (K+1)×n
end

"""
    getLV(fit::RowRandomFit, Y; rotate=true, N=nothing) -> n×K matrix

Conditional latent-variable (ordination) scores for a random-row fit: the first `K`
components of the per-site augmented Laplace mode `ŵ_s = [ẑ_s; û_s]` at the augmented
loadings `Λ̃ = [Λ | σ_row·𝟙_p]`. `Y` is the `p×n` response matrix; `rotate=true`
applies the canonical [`rotation`](@ref) on the ordination block.
"""
function getLV(fit::RowRandomFit, Y::AbstractMatrix{<:Real};
               rotate::Bool = true, N::Union{Nothing, AbstractMatrix} = nothing)
    K = size(fit.Λ, 2)
    W = _row_random_modes(fit, Y, N)
    Zt = permutedims(view(W, 1:K, :))              # n×K ordination scores
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    row_effects(fit::RowRandomFit, Y; N=nothing) -> n-vector

Per-site random-row-effect BLUPs `ρ̂_s = σ_row·û_s`, where `û_s` is the `(K+1)`-th
component of the per-site augmented Laplace mode `ŵ_s = [ẑ_s; û_s]` (conditional mode
of the standardised row effect). `Y` is the `p×n` response matrix.
"""
function row_effects(fit::RowRandomFit, Y::AbstractMatrix{<:Real};
                     N::Union{Nothing, AbstractMatrix} = nothing)
    K = size(fit.Λ, 2)
    W = _row_random_modes(fit, Y, N)
    return fit.σ_row .* vec(view(W, K + 1, :))     # ρ̂_s = σ_row·û_s
end

"""
    predict(fit::RowRandomFit, Y; type=:response, N=nothing) -> result

In-sample prediction at the augmented Laplace mode. `type=:roweffect` returns the
n-vector of row-effect BLUPs `ρ̂_s` (see [`row_effects`](@ref)); `type=:link` returns
the `p×n` linear predictor `η = β + ρ̂_s + Λ ẑ`; `type=:response` applies the inverse
link to the (clamped) `η`.
"""
function predict(fit::RowRandomFit, Y::AbstractMatrix{<:Real};
                 type::Symbol = :response, N::Union{Nothing, AbstractMatrix} = nothing)
    type in (:link, :response, :roweffect) ||
        throw(ArgumentError("type must be :link, :response, or :roweffect; got :$type"))
    K = size(fit.Λ, 2)
    W = _row_random_modes(fit, Y, N)               # (K+1)×n
    type === :roweffect && return fit.σ_row .* vec(view(W, K + 1, :))
    Λ̃ = _augment_lambda(fit.Λ, fit.σ_row)
    η = fit.β .+ Λ̃ * W                             # p×n, includes the row-effect column
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end
