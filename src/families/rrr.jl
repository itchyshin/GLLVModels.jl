# Reduced-rank regression (RRR) — gllvm's `num.RR` constrained ordination.
#
# This is the *fully constrained* sibling of the concurrent ordination in
# src/families/constrained_ordination.jl. There, the K latent variables carry a
# covariate-driven mean PLUS a residual random effect, z_s ~ N(B' x_s, I_K), so
# there is a latent integral to approximate (Laplace). Here the latent axes are a
# DETERMINISTIC linear combination of the environmental predictors,
#
#     z_s = B' x_s        (no residual latent variable),
#
# so the linear predictor collapses to a plain reduced-rank GLM mean,
#
#     η_{ts} = β_t + (Λ B' x_s)_t ,   μ_{ts} = linkinv(η_{ts}) ,
#
# with NO integral to approximate. The marginal log-likelihood is simply the sum
# of the per-observation conditional GLM log-densities,
#
#     loglik = Σ_{t,s} logp(y_{ts} | μ_{ts}) .
#
# Interpretation: the p×q species-by-covariate coefficient matrix is constrained
# to rank K via the factorisation Λ·B'. The q×K matrix B holds the constrained
# ordination axes (the environmental gradients the community responds to, à la
# CCA / RDA canonical axes); Λ (p×K) maps those axes onto species. The site
# ordination scores are the deterministic z_s = B' x_s — there is no per-site
# mode to solve for, in contrast to the concurrent / unconstrained ordinations
# where the scores are conditional Laplace modes of a genuine random effect.
#
# Design reuse: every family-specific helper is shared with the covariate /
# constrained fitters (`_cov_default_link`, `_cov_has_disp`, `_cov_disp_init`,
# `_cov_family`, `_cov_zemp`) and the per-observation conditional density
# `_glm_logpdf` / `_clamp_mu` / `_clamp_eta` / `linkinv` is the same family
# substrate the Laplace path uses. The only structural difference is that the
# objective is the direct GLM loglik (no `_marginal_loglik_offset` / inner Newton
# solve), so the L-BFGS driver has no newton/mode arguments.

# Reduced-rank linear predictor η (p×n): η = β .+ Λ * (B' * X'). Conventions:
# Λ is p×K, B is q×K, X is n×q (row s = x_s'), β is length p. Throws
# DimensionMismatch on any inconsistency.
function _rrr_eta(β::AbstractVector, Λ::AbstractMatrix, B::AbstractMatrix, X::AbstractMatrix)
    K = size(Λ, 2)
    size(B, 2) == K ||
        throw(DimensionMismatch("Λ has K = $K columns but B has $(size(B, 2)) columns"))
    q = size(B, 1)
    size(X, 2) == q ||
        throw(DimensionMismatch("B has q = $q rows but X has $(size(X, 2)) columns"))
    size(Λ, 1) == length(β) ||
        throw(DimensionMismatch("Λ has $(size(Λ, 1)) rows but β has length $(length(β))"))
    # η = β (p) .+ Λ (p×K) * (B' (K×q) * X' (q×n)) = (p×n)
    return β .+ Λ * (B' * X')
end

"""
    rrr_marginal_loglik(family, Y, N, Λ, B, β, X, link) -> Float64

DIRECT reduced-rank GLM log-likelihood for the constrained-ordination (`num.RR`)
model — NO integral / Laplace step. The latent axes `z_s = B' x_s` are a
deterministic linear combination of the site covariates, so the linear predictor
is `η = β .+ Λ B' X'` and the marginal is just the sum of the per-observation
conditional densities:

    Σ_{t,s} logp(y_{ts} | μ_{ts}),   μ_{ts} = linkinv(link, η_{ts}).

`family` is a fully-built `Distributions` marker carrying any dispersion (use
`_cov_family(family, disp)` to attach it). `Y` is p×n, `N` the trial-count matrix
(Binomial; all-ones otherwise), `Λ` is p×K, `B` is q×K, `β` is length p, and `X`
is the n×q site-covariate matrix. With `B = 0` this equals the intercept-only GLM
log-likelihood `Σ_{t,s} logp(y_{ts} | linkinv(β_t))`.
"""
function rrr_marginal_loglik(family, Y::AbstractMatrix, N::AbstractMatrix,
        Λ::AbstractMatrix, B::AbstractMatrix, β::AbstractVector, X::AbstractMatrix,
        link::Link)
    η = _rrr_eta(β, Λ, B, X)
    p, n = size(Y)
    acc = 0.0
    @inbounds for s in 1:n
        for t in 1:p
            μ = _clamp_mu(family, linkinv(link, _clamp_eta(η[t, s])))
            acc += _glm_logpdf(family, μ, N[t, s], Y[t, s])
        end
    end
    return acc
end

"""
    RRRFit

Result of [`fit_rrr_gllvm`](@ref): a fully-constrained reduced-rank-regression
(`num.RR`) ordination. The K latent axes are the DETERMINISTIC environmental
projection `z_s = B' x_s` (no residual latent variable), so this is a plain
reduced-rank GLM — contrast the concurrent ordination
([`ConstrainedOrdinationFit`](@ref)), whose latent variables retain a residual
random effect and therefore require a Laplace marginal.

Fields: `family` (the Distributions marker), per-species intercepts `β` (length
p), loadings `Λ` (p×K, mapping the constrained axes onto species), the RRR
coefficient matrix `B` (q×K — column `k` is the `k`-th constrained ordination
axis in covariate space), `dispersion` (`r`/`φ`/`α`, or `NaN` when the family has
none), `link`, the maximised `loglik`, `converged`, and `iterations`.
"""
struct RRRFit
    family::Distribution
    β::Vector{Float64}
    Λ::Matrix{Float64}
    B::Matrix{Float64}
    dispersion::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::RRRFit)
    p, K = size(f.Λ); q = size(f.B, 1)
    print(io, "RRRFit(", nameof(typeof(f.family)), ", p=", p, ", q=", q, ", K=", K)
    isnan(f.dispersion) || print(io, ", disp=", round(f.dispersion; sigdigits = 4))
    print(io, ", loglik=", round(f.loglik; sigdigits = 7), f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    getLV(fit::RRRFit, X; rotate=true) -> n×K matrix

Constrained ordination scores for an RRR fit: the DETERMINISTIC environmental
projection `z_s = B' x_s`, returned as the n×K matrix `(fit.B' * X')'`. `X` is the
n×q site-covariate matrix. Unlike the unconstrained / concurrent ordinations,
there is no residual latent variable and hence no per-site Laplace mode — the
scores are an exact function of the covariates. `rotate=true` applies the
canonical [`rotation`](@ref) of `Λ`.
"""
function getLV(fit::RRRFit, X::AbstractMatrix{<:Real}; rotate::Bool = true)
    Zt = (fit.B' * X')'                       # n×K constrained scores z_s = B' x_s
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::RRRFit, X; type=:response) -> p×n matrix

`:link` = the linear predictor `η = β .+ Λ B' X'`; `:response` (= `:mean`) = the
mean `μ = linkinv(link, η)`. `X` is the n×q site-covariate matrix. Because the
latent axes are deterministic in the covariates, no original `Y` is required.
"""
function predict(fit::RRRFit, X::AbstractMatrix{<:Real}; type::Symbol = :response)
    type in (:response, :mean, :link) ||
        throw(ArgumentError("type must be :response, :mean, or :link; got :$type"))
    η = _rrr_eta(fit.β, fit.Λ, fit.B, X)
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

# Post-fit accessors (mirror the covered covariate analogue GllvmCovFit in
# src/postfit.jl): `_loglik`/`_nparams` unlock the generic `aic`/`bic`. The
# response-scale `fitted(fit, X)` is already served by the generic fallback
# `fitted(fit, data; …) = predict(fit, data; type=:response, …)` (RRR's `predict`
# takes only `X`, the latent being deterministic). (No `residuals`: like
# GllvmCovFit, no family-generic Dunn–Smyth residual is provided for this type.)
_loglik(fit::RRRFit) = fit.loglik

# Free parameters: β (p) + Λ (modulo K(K−1)/2 rotational df) + vec(B) (q·K) + dispersion?
function _nparams(fit::RRRFit)
    p, K = size(fit.Λ); q = size(fit.B, 1)
    return p + (p * K - div(K * (K - 1), 2)) + q * K + (isnan(fit.dispersion) ? 0 : 1)
end

"""
    fit_rrr_gllvm(Y; family=Poisson(), X, K, link=nothing, N=nothing, …) -> RRRFit

Fit a **reduced-rank regression** (gllvm `num.RR`) constrained ordination by
L-BFGS over `[β; pack_lambda(Λ); vec(B); (log-dispersion)]` on the DIRECT GLM
log-likelihood ([`rrr_marginal_loglik`](@ref)). The K latent axes are a
deterministic linear function of the site covariates,

    z_s = B' x_s ,   η_{ts} = β_t + (Λ B' x_s)_t ,

so the p×q species-by-covariate coefficient matrix is constrained to rank K via
`Λ·B'`. There is NO residual latent variable and hence no integral to
approximate — contrast the concurrent ordination ([`fit_constrained_gllvm`](@ref) /
`fit_concurrent_gllvm`), where `z_s ~ N(B' x_s, I_K)` keeps a random effect and so
needs a Laplace marginal. The fitted q×K matrix `B` gives the canonical
constrained ordination axes (CCA / RDA-like) and `Λ` maps them onto species.

`family` is a `Distributions` marker — `Poisson()`, `NegativeBinomial()`,
`Binomial()`, `Beta()`, or `Gamma()` — and dispatches the conditional density (the
dispersion, where present, is jointly estimated). `X` is the n×q site-covariate
matrix (row `s` is `x_s'`); `Y` is p×n; `N` supplies Binomial trial counts
(default all-ones). Finite-difference gradient.

This fitter takes no `mask` and no `missing` responses: either one throws an `ArgumentError`
(every cell enters the likelihood).

```julia
# Poisson community constrained by two site covariates onto K = 2 axes:
fit = fit_rrr_gllvm(Y; family = Poisson(), X = X, K = 2)
fit.B               # q×2 constrained ordination axes (environment → latent)
fit.Λ               # p×2 species loadings on those axes
getLV(fit, X)       # n×2 deterministic site scores z_s = B' x_s
```
"""
function fit_rrr_gllvm(Y::AbstractMatrix{<:Real}; mask = nothing, family = Poisson(),
        X::AbstractMatrix{<:Real}, K::Integer, link::Union{Nothing, Link} = nothing,
        N::Union{Nothing, AbstractMatrix} = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500)
    _refuse_mask_route("fit_rrr_gllvm", mask)
    p, n = size(Y)
    size(X, 1) == n ||
        throw(DimensionMismatch("X must be (n, q) = ($n, q) with one row per site; got $(size(X))"))
    q = size(X, 2)
    rr = rr_theta_len(p, K)
    lk = link === nothing ? _cov_default_link(family) : link
    Nm = N === nothing ? fill(1, p, n) : N
    has_disp = _cov_has_disp(family)
    _check_family_response_support(family, Y, nothing; caller = "fit_rrr_gllvm", N = N)

    # Warm start: link-scale row means for β, SVD loadings, zero RRR coefficients
    # (B = 0 gives the intercept-only GLM) — the same machinery as the concurrent
    # fitter, only the marginal differs (direct GLM, no Laplace).
    Zemp = _cov_zemp(family, Y, Nm, lk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    B0 = zeros(q, K)

    θ0 = has_disp ? vcat(β0, pack_lambda(Λ0), vec(B0), log(_cov_disp_init(family))) :
                    vcat(β0, pack_lambda(Λ0), vec(B0))
    qK = q * K
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        B = reshape(θ[(p + rr + 1):(p + rr + qK)], q, K)
        disp = has_disp ? exp(θ[p + rr + qK + 1]) : NaN
        fam = _cov_family(family, disp)
        v = try
            -rrr_marginal_loglik(fam, Y, Nm, Λ, B, β, X, lk)
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
    B̂ = reshape(θ̂[(p + rr + 1):(p + rr + qK)], q, K)
    disp̂ = has_disp ? exp(θ̂[p + rr + qK + 1]) : NaN
    return RRRFit(family, β̂, Λ̂, B̂, disp̂, lk, _fit_verdict(res)...)
end
