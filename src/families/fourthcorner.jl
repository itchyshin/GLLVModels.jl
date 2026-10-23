# Fourth-corner / trait–environment GLLVM for the non-Gaussian Laplace families.
#
# The species-specific path (src/families/species_covariates.jl) gives each
# species t its own free coefficient row B[t, :]. The fourth-corner model instead
# *structures* the species-by-environment interaction through species traits: the
# response of species t to environmental gradient k is mediated by the species'
# traits TR[t, :] via a small fourth-corner coefficient matrix C (q×r). The
# species-by-site interaction offset is
#
#     O[t,s] = Σ_{k,l} Xenv[s,k]·TR[t,l]·C[k,l] = (TR · Cᵀ · Xenvᵀ)[t,s] ,
#
# so the linear predictor is
#
#     η_{ts} = β_t + O[t,s] + (Λ z_s)_t .
#
# This is gllvmTMB's fourth-corner term: how measured species traits modulate
# environmental responses. With q site covariates and r traits the interaction is
# captured by only q·r parameters (vs. p·q for the unstructured species-specific
# path), which is the statistical point of the fourth-corner construction.
#
# Design reuse: the offset is still constant in z, so the offset-aware Laplace core
# `_marginal_loglik_offset` (src/families/covariates.jl) applies verbatim — only the
# (p×n) offset matrix `O` is assembled differently. Every family-specific helper
# (`_cov_default_link`, `_cov_has_disp`, `_cov_disp_init`, `_cov_family`,
# `_cov_zemp`) is shared with the covariate fitters, and the L-BFGS driver mirrors
# `fit_gllvm_speciescov` exactly, differing only in the θ block that now packs the
# fourth-corner `vec(C)` (length q·r).

# Offset matrix O[t,s] = Σ_{k,l} Xenv[s,k]·TR[t,l]·C[k,l] = (TR · Cᵀ · Xenvᵀ)[t,s]
# from site covariates Xenv::(n×q), species traits TR::(p×r), and fourth-corner
# coefficients C::(q×r). Returns the (p×n) offset.
function _build_offset_fourthcorner(Xenv::AbstractMatrix, TR::AbstractMatrix, C::AbstractMatrix)
    n, q = size(Xenv)
    p, r = size(TR)
    size(C, 1) == q ||
        throw(DimensionMismatch("C has $(size(C, 1)) rows but Xenv has q = $q covariates"))
    size(C, 2) == r ||
        throw(DimensionMismatch("C has $(size(C, 2)) columns but TR has r = $r traits"))
    # (p×r)·(r×q)·(q×n) = (p×n)
    return TR * C' * Xenv'
end

"""
    FourthCornerFit

Result of [`fit_fourthcorner_gllvm`](@ref): a GLLVM fit with a **fourth-corner**
trait–environment interaction term. Fields: `family` (the Distributions marker),
per-species intercepts `β` (length p), the fourth-corner coefficient matrix `C`
(q×r, entry `C[k,l]` couples site covariate `k` to species trait `l`), loadings
`Λ` (p×K), `dispersion` (`r`/`φ`/`α`, or `NaN` when the family has none), `link`,
the maximised Laplace `loglik`, `converged`, and `iterations`.
"""
struct FourthCornerFit
    family::Distribution
    β::Vector{Float64}
    C::Matrix{Float64}
    Λ::Matrix{Float64}
    dispersion::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::FourthCornerFit)
    p, K = size(f.Λ); q, r = size(f.C)
    print(io, "FourthCornerFit(", nameof(typeof(f.family)), ", p=", p, ", q=", q, ", r=", r, ", K=", K)
    isnan(f.dispersion) || print(io, ", disp=", round(f.dispersion; sigdigits = 4))
    print(io, ", loglik=", round(f.loglik; sigdigits = 7), f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    getLV(fit::FourthCornerFit, Y, Xenv, TR; rotate=true, N=nothing) -> n×K matrix

Conditional latent scores for a fourth-corner fit: the per-site offset-aware
Laplace mode `ẑₛ` at `η = β + (TR Cᵀ Xenvᵀ) + Λz`, with the trait–environment
offset `O = _build_offset_fourthcorner(Xenv, TR, C)`. `rotate=true` applies the
canonical [`rotation`](@ref).
"""
function getLV(fit::FourthCornerFit, Y::AbstractMatrix{<:Real},
               Xenv::AbstractMatrix{<:Real}, TR::AbstractMatrix{<:Real};
               rotate::Bool = true, N::Union{Nothing, AbstractMatrix} = nothing)
    p, n = size(Y); K = size(fit.Λ, 2)
    Nm = N === nothing ? fill(1, p, n) : N
    fam = _cov_family(fit.family, fit.dispersion)
    O = _build_offset_fourthcorner(Xenv, TR, fit.C)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        η0 = fit.β .+ view(O, :, s)
        Z[:, s] = _laplace_mode_off(fam, view(Y, :, s), view(Nm, :, s), fit.Λ, η0, fit.link)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::FourthCornerFit, Y, Xenv, TR; type=:response, N=nothing) -> p×n matrix

`:link` = the linear predictor `η = β + (TR Cᵀ Xenvᵀ) + Λẑ`; `:response`
(= `:mean`) = the mean `μ = linkinv(link, η)`.
"""
function predict(fit::FourthCornerFit, Y::AbstractMatrix{<:Real},
                 Xenv::AbstractMatrix{<:Real}, TR::AbstractMatrix{<:Real};
                 type::Symbol = :response, N::Union{Nothing, AbstractMatrix} = nothing)
    type in (:response, :mean, :link) ||
        throw(ArgumentError("type must be :response, :mean, or :link; got :$type"))
    Z = getLV(fit, Y, Xenv, TR; rotate = false, N = N)
    O = _build_offset_fourthcorner(Xenv, TR, fit.C)
    η = fit.β .+ O .+ fit.Λ * Z'
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

# Post-fit accessors (mirror the covered covariate analogue GllvmCovFit in
# src/postfit.jl): `_loglik`/`_nparams` unlock the generic `aic`/`bic`, and
# `fitted` is the response-scale conditional prediction. (No `residuals`: like
# GllvmCovFit, no family-generic Dunn–Smyth residual is provided for this type.)
_loglik(fit::FourthCornerFit) = fit.loglik

# Free parameters: β (p) + vec(C) (q·r) + Λ (modulo K(K−1)/2 rotational df) + dispersion?
function _nparams(fit::FourthCornerFit)
    p, K = size(fit.Λ); q, r = size(fit.C)
    return p + q * r + (p * K - div(K * (K - 1), 2)) + (isnan(fit.dispersion) ? 0 : 1)
end

"""
    fitted(fit::FourthCornerFit, Y, Xenv, TR; N=nothing) -> p×n matrix of fitted means.
"""
fitted(fit::FourthCornerFit, Y::AbstractMatrix{<:Real},
       Xenv::AbstractMatrix{<:Real}, TR::AbstractMatrix{<:Real};
       N::Union{Nothing, AbstractMatrix} = nothing) =
    predict(fit, Y, Xenv, TR; type = :response, N = N)

"""
    fit_fourthcorner_gllvm(Y; family, Xenv, TR, K, link=nothing, N=nothing, …) -> FourthCornerFit

Fit a non-Gaussian **fourth-corner** trait–environment GLLVM by L-BFGS over
`[β; vec(C); vec(Λ); (log-dispersion)]` on the offset-augmented Laplace marginal,
where the linear predictor is
`η_{ts} = β_t + Σ_{k,l} Xenv[s,k]·TR[t,l]·C[k,l] + (Λ z_s)_t`.

The species-by-site interaction surface is structured through measured species
traits: `Xenv` is the site-by-covariate matrix (`n×q`), `TR` the species-by-trait
matrix (`p×r`), and the fourth-corner coefficients `C` (`q×r`) couple the two. This
costs only `q·r` interaction parameters, in contrast to the `p·q` free slopes of
[`fit_gllvm_speciescov`](@ref).

`family` is a `Distributions` marker — `Poisson()`, `NegativeBinomial()`,
`Binomial()`, `Beta()`, or `Gamma()` — and dispatches the marginal (the dispersion,
where present, is jointly estimated). `Y` is `p × n`; `N` supplies Binomial trial
counts (default all-ones). Finite-difference gradient.

```julia
# Poisson abundance, fourth-corner trait–environment interaction:
fit = fit_fourthcorner_gllvm(Y; family = Poisson(), Xenv = Xenv, TR = TR, K = 2)
fit.C            # q×r fourth-corner coefficient matrix
```

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or simply include
`missing` entries in `Y` — either way the masked cells are dropped from the
marginal *and* from the warm start, so the fit depends only on the observed cells.
"""
function fit_fourthcorner_gllvm(Y::AbstractMatrix; family,
        Xenv::AbstractMatrix, TR::AbstractMatrix, K::Integer,
        link::Union{Nothing, Link} = nothing,
        N::Union{Nothing, AbstractMatrix} = nothing, mask = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    size(Xenv, 1) == n ||
        throw(DimensionMismatch("Xenv must have n = $n rows (sites); got $(size(Xenv, 1))"))
    size(TR, 1) == p ||
        throw(DimensionMismatch("TR must have p = $p rows (species); got $(size(TR, 1))"))
    q = size(Xenv, 2)
    r = size(TR, 2)
    rr = rr_theta_len(p, K)
    lk = link === nothing ? _cov_default_link(family) : link
    Nm = N === nothing ? fill(1, p, n) : N
    has_disp = _cov_has_disp(family)

    # NA handling: derive the observation mask and a sanitized response matrix.
    msk = _resolve_obs_mask(mask, Y)
    Yc = _sanitize_missing(Y, _cov_placeholder(family))

    # Warm start: link-scale row means for β, zero fourth-corner coefficients, SVD
    # loadings — identical machinery to fit_gllvm_speciescov.
    Zemp = _cov_zemp(family, Yc, Nm, lk)
    _mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end

    θ0 = has_disp ? vcat(β0, zeros(q * r), pack_lambda(Λ0), log(_cov_disp_init(family))) :
                    vcat(β0, zeros(q * r), pack_lambda(Λ0))
    function negll(θ)
        β = θ[1:p]
        C = reshape(θ[(p + 1):(p + q * r)], q, r)
        Λ = unpack_lambda(θ[(p + q * r + 1):(p + q * r + rr)], p, K)
        disp = has_disp ? exp(θ[p + q * r + rr + 1]) : NaN
        O = _build_offset_fourthcorner(Xenv, TR, C)
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
    Ĉ = reshape(θ̂[(p + 1):(p + q * r)], q, r)
    Λ̂ = unpack_lambda(θ̂[(p + q * r + 1):(p + q * r + rr)], p, K)
    disp̂ = has_disp ? exp(θ̂[p + q * r + rr + 1]) : NaN
    return FourthCornerFit(family, β̂, Ĉ, Λ̂, disp̂, lk, _fit_verdict(res)...)
end
