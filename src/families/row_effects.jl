# Community ROW EFFECTS (per-site intercepts) for the non-Gaussian Laplace families.
#
# The covariate path (src/families/covariates.jl) adds an additive offset surface
# o_{ts} on the linear predictor. This file specialises that machinery to a
# COMMUNITY row effect — a per-SITE intercept ρ_s shared across every species:
#
#     η_{ts} = β_t + ρ_s + (Λ z_s)_t
#
# i.e. an offset O[t,s] = ρ_s that is constant in the species index t. This is the
# gllvmTMB "row.eff" term: a site-level baseline (total abundance / sampling effort)
# absorbed before the latent ordination.
#
# Identifiability: β (species intercepts, length p) and ρ (site effects, length n)
# are confounded by a constant shift — adding c to every β_t and subtracting c from
# every ρ_s leaves η unchanged. We pin the first site as the reference, ρ_1 ≡ 0, and
# estimate the remaining n−1 free entries ρ_2..ρ_n. The full ρ vector (length n,
# ρ[1]=0) is reconstructed as `vcat(0.0, ρfree)` inside the objective.
#
# Everything else is reused verbatim from the covariate path: the offset-aware
# per-site Laplace `_marginal_loglik_offset`, the `_cov_*` family helpers, and the
# shared links/packing. Only the offset construction and the free-parameter block
# differ.

# Build the constant-in-species offset O (p×n) from a length-n site-effect vector ρ
# (with ρ[1]=0 by construction): O[t,s] = ρ_s for every species t. Each column s is
# the constant ρ_s, so every row of O equals ρ'.
function _build_offset_row(ρ::AbstractVector, p::Integer)
    return repeat(ρ', p, 1)
end

"""
    RowEffectFit

Result of [`fit_roweffect_gllvm`](@ref): a GLLVM fit with a community row effect
(per-site intercept). Fields: `family` (the Distributions marker), per-species
intercepts `β` (length p), site/row effects `ρ` (length n, with the reference
`ρ[1] = 0`), loadings `Λ` (p×K), `dispersion` (`r`/`φ`/`α`, or `NaN` when the family
has none), `link`, the maximised Laplace `loglik`, `converged`, and `iterations`.
"""
struct RowEffectFit
    family::Distribution
    β::Vector{Float64}
    ρ::Vector{Float64}
    Λ::Matrix{Float64}
    dispersion::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
end

# ---------------------------------------------------------------------------
# Post-fit ordination: getLV / predict (parallel to GllvmCovFit in src/postfit.jl,
# but the offset is the constant-in-species row effect O[t,s] = ρ_s).
# ---------------------------------------------------------------------------

_loadings(fit::RowEffectFit) = fit.Λ
_loglik(fit::RowEffectFit)   = fit.loglik

# Free params: β (p) + free row effects ρ_2..ρ_n (length(ρ)−1; ρ_1 ≡ 0 reference)
# + reduced loadings Λ + a dispersion (where the family has one).
function _nparams(fit::RowEffectFit)
    p, K = size(fit.Λ)
    nfree = max(length(fit.ρ) - 1, 0)
    return p + nfree + (p * K - div(K * (K - 1), 2)) + (isnan(fit.dispersion) ? 0 : 1)
end

"""
    getLV(fit::RowEffectFit, Y; rotate=true, N=nothing) -> n×K matrix

Conditional latent-variable scores for a row-effect fit: the per-site offset-aware
Laplace mode `ẑₛ` (`_laplace_mode_off`) at `η = β + ρ_s + Λz`, with the row-effect
offset `O[t,s] = ρ_s`. `Y` is the `p×n` response matrix; `rotate=true` applies the
canonical [`rotation`](@ref).
"""
function getLV(fit::RowEffectFit, Y::AbstractMatrix{<:Real};
               rotate::Bool = true, N::Union{Nothing, AbstractMatrix} = nothing)
    p, n = size(Y); K = size(fit.Λ, 2)
    Nm = N === nothing ? fill(1, p, n) : N
    fam = _cov_family(fit.family, fit.dispersion)
    O = _build_offset_row(fit.ρ, p)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        η0 = fit.β .+ view(O, :, s)
        Z[:, s] = _laplace_mode_off(fam, view(Y, :, s), view(Nm, :, s), fit.Λ, η0, fit.link)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::RowEffectFit, Y; type=:response, N=nothing) -> p×n matrix

In-sample fitted values at the Laplace mode `ẑ` (see [`getLV`](@ref)): `type=:link`
returns the linear predictor `η = β + ρ_s + Λ ẑ`; `type=:response` applies the
inverse link to the (clamped) `η`.
"""
function predict(fit::RowEffectFit, Y::AbstractMatrix{<:Real};
                 type::Symbol = :response, N::Union{Nothing, AbstractMatrix} = nothing)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    Z = getLV(fit, Y; rotate = false, N = N)          # n×K
    O = _build_offset_row(fit.ρ, size(Y, 1))
    η = fit.β .+ O .+ fit.Λ * Z'                       # p×n
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

function Base.show(io::IO, f::RowEffectFit)
    p, K = size(f.Λ); n = length(f.ρ)
    print(io, "RowEffectFit(", nameof(typeof(f.family)), ", p=", p, ", n=", n, ", K=", K)
    isnan(f.dispersion) || print(io, ", disp=", round(f.dispersion; sigdigits = 4))
    print(io, ", loglik=", round(f.loglik; sigdigits = 7), f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_roweffect_gllvm(Y; family, K, link=nothing, N=nothing, …) -> RowEffectFit

Fit a non-Gaussian GLLVM **with a community row effect** (per-site intercept) by
L-BFGS over `[β; ρ_free; vec(Λ); (log-dispersion)]` on the offset-augmented Laplace
marginal, where the linear predictor is `η_{ts} = β_t + ρ_s + (Λ z_s)_t`.

`β` (species intercepts, length p) and `ρ` (site effects, length n) are confounded
by a constant shift, so the first site is pinned as the reference (`ρ_1 ≡ 0`) and
only the `n−1` free entries `ρ_2..ρ_n` are estimated; the returned `ρ` has length n
with `ρ[1] = 0`. The offset surface is `O[t,s] = ρ_s`, constant across species.

`family` is a `Distributions` marker — `Poisson()`, `NegativeBinomial()`,
`Binomial()`, `Beta()`, or `Gamma()` — and dispatches the marginal (the dispersion,
where present, is jointly estimated). `Y` is `p × n`; `N` supplies Binomial trial
counts (default all-ones). Finite-difference gradient. Warm start: per-species
empirical link-scale means for `β`, zeros for the free row effects, and an SVD
(PPCA-style) loadings init.

```julia
fit = fit_roweffect_gllvm(Y; family = Poisson(), K = 2)
fit.ρ            # estimated per-site intercepts (ρ[1] == 0 reference)
```

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or simply include
`missing` entries in `Y` — either way the masked cells are dropped from the
marginal *and* from the warm start, so the fit depends only on the observed cells.
"""
function fit_roweffect_gllvm(Y::AbstractMatrix; family,
        K::Integer, link::Union{Nothing, Link} = nothing,
        N::Union{Nothing, AbstractMatrix} = nothing, mask = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    n >= 1 || throw(ArgumentError("Y must have at least one site (column)"))
    rr = rr_theta_len(p, K)
    nfree = n - 1                       # ρ_2..ρ_n (ρ_1 ≡ 0 reference)
    lk = link === nothing ? _cov_default_link(family) : link
    Nm = N === nothing ? fill(1, p, n) : N
    has_disp = _cov_has_disp(family)

    # NA handling: derive the observation mask and a sanitized response matrix.
    msk = _resolve_obs_mask(mask, Y)
    Yc = _sanitize_missing(Y, _cov_placeholder(family))

    # warm start: per-species link-scale row means for β, zeros for ρfree, SVD for Λ
    Zemp = _cov_zemp(family, Yc, Nm, lk)
    _mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end

    θ0 = has_disp ? vcat(β0, zeros(nfree), pack_lambda(Λ0), log(_cov_disp_init(family))) :
                    vcat(β0, zeros(nfree), pack_lambda(Λ0))
    function negll(θ)
        β = θ[1:p]
        ρfree = θ[(p + 1):(p + nfree)]
        Λ = unpack_lambda(θ[(p + nfree + 1):(p + nfree + rr)], p, K)
        disp = has_disp ? exp(θ[p + nfree + rr + 1]) : NaN
        ρ = vcat(zero(eltype(ρfree)), ρfree)
        O = _build_offset_row(ρ, p)
        v = try
            # Inside the `try`: exp(log-dispersion) can underflow to 0.0 (see fit_gllvm_cov).
            fam = _cov_family(family, disp)
            -_marginal_loglik_offset(fam, Yc, Nm, Λ, β, O, lk;
                                     mask = msk, maxiter = newton_maxiter, tol = newton_tol)
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
    ρ̂ = vcat(0.0, θ̂[(p + 1):(p + nfree)])
    Λ̂ = unpack_lambda(θ̂[(p + nfree + 1):(p + nfree + rr)], p, K)
    disp̂ = has_disp ? exp(θ̂[p + nfree + rr + 1]) : NaN
    return RowEffectFit(family, β̂, ρ̂, Λ̂, disp̂, lk, _fit_verdict(res)...)
end
