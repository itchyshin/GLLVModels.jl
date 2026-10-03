# Zero-inflated families under gllvmTMB's names and semantics:
# `zi_poisson()`, `zi_nbinom2()`, `zi_binomial()` (gllvmTMB family ids 17, 18, 19,
# read at the P1 pin 9539352f66f2db2cc26b1c393e67212a359b60c9: R/families.R,
# src/gllvmTMB.cpp `fid == 17/18/19`, R/fit-multi.R admission).
#
# R's model, per observation (trait t, unit i):
#     P(y = 0) = zi_t + (1 - zi_t) f(0 | eta),   P(y = k > 0) = (1 - zi_t) f(k | eta)
#     zi_t = logistic(logit_zi_t)     (per trait, intercept only: no covariates,
#                                      no latent loadings on the zero part)
#     eta  = beta_t + Lambda_t' z_i,  z_i ~ N(0, I_K)
# with f the Poisson (log link), NB2 (log link; Var = mu + mu^2 / phi_t, one phi PER
# TRAIT, the same `log_phi_nbinom2` vector plain nbinom2() uses) or Binomial(N_ti, p)
# (logit link; per-row trials) count kernel. TMB integrates z_i with the Laplace
# approximation, whose log-determinant is the OBSERVED curvature of the joint
# negative log-likelihood.
#
# Julia's own ZIPoisson / ZINB / ZIB fitters (src/families/twopart.jl) share the
# mixture and the per-trait intercept-only zero part, but differ in ways that change
# the fitted numbers, so the R names do NOT route to them:
#   * all three use the expected (Fisher) count weight in the Laplace log-determinant
#     (the TWOPART_KNOWN_OPEN census gap in test/test_curvature_census.jl);
#   * ZINB uses ONE shared scalar dispersion r across traits (R: one phi per trait);
#   * ZIB uses ONE shared scalar trials count N (R: per-row trials) and admits N = 1.
# The R-named route below reuses the two-part mode search (`_twopart_mode_search`)
# and per-family pieces unchanged, through per-cell markers that carry the cell's own logit_zi / phi / N
# and supply the observed count curvature via `_tp_observed_Wc`. Julia's own
# families and fitters are untouched.

"""
    ZiPoisson()

Family marker returned by [`zi_poisson`](@ref): gllvmTMB's `zi_poisson()`
zero-inflated Poisson (log link, per-trait intercept-only structural-zero
probability). Fit with [`fit_zi_gllvm`](@ref) or `fit_gllvm(Y; family =
zi_poisson(), K)`. Distinct from Julia's own [`ZIPoisson`](@ref) route; see
[`fit_zi_gllvm`](@ref) for the difference.
"""
struct ZiPoisson end

"""
    ZiNbinom2()

Family marker returned by [`zi_nbinom2`](@ref): gllvmTMB's `zi_nbinom2()`
zero-inflated NB2 (log link, one dispersion `phi` per trait, `Var = mu +
mu^2/phi`). Distinct from Julia's own [`ZINegBin`](@ref) route, which estimates
one shared dispersion across traits.
"""
struct ZiNbinom2 end

"""
    ZiBinomial()

Family marker returned by [`zi_binomial`](@ref): gllvmTMB's `zi_binomial()`
zero-inflated binomial (logit link, per-observation trials). Distinct from
Julia's own [`ZIB`](@ref) route, which takes one shared trials count.
"""
struct ZiBinomial end

const _ZiTwinFamily = Union{ZiPoisson, ZiNbinom2, ZiBinomial}

# `fit_zi_gllvm` has no `offset` keyword: `fit_gllvm` refuses an offset on these families
# with a clear ArgumentError instead of letting it reach a MethodError.
_offset_unsupported(::_ZiTwinFamily) = true

_zi_rname(::ZiPoisson) = "zi_poisson"
_zi_rname(::ZiNbinom2) = "zi_nbinom2"
_zi_rname(::ZiBinomial) = "zi_binomial"

default_link(::ZiPoisson) = LogLink()
default_link(::ZiNbinom2) = LogLink()
default_link(::ZiBinomial) = LogitLink()

function _zi_check_link(fam::_ZiTwinFamily, link)
    want = default_link(fam)
    link isa typeof(want) && return nothing
    throw(ArgumentError(
        "$(_zi_rname(fam)): only the $(nameof(typeof(want))) link is supported " *
        "(as in gllvmTMB); got $(nameof(typeof(link)))."))
end

"""
    zi_poisson(; link = LogLink()) -> ZiPoisson

Julia twin of gllvmTMB's `zi_poisson()`: a true zero-inflation mixture
`P(y=0) = zi + (1-zi) Pois(0 | mu)`, `P(y=k>0) = (1-zi) Pois(k | mu)`, with
`mu = exp(beta_t + Lambda_t' z)` and a per-trait, intercept-only structural-zero
probability `zi_t = logistic(logit_zi_t)` (no covariates and no latent loadings
on the zero part). Only the log link is admitted, as in R.

```julia
fit = fit_gllvm(Y; family = zi_poisson(), K = 1)
fit.zi        # per-trait structural-zero probability (R: fit\$report\$zi)
```
"""
function zi_poisson(; link = LogLink())
    _zi_check_link(ZiPoisson(), link)
    return ZiPoisson()
end

"""
    zi_nbinom2(; link = LogLink()) -> ZiNbinom2

Julia twin of gllvmTMB's `zi_nbinom2()`: the [`zi_poisson`](@ref) mixture with an
NB2 count kernel, `Var = mu + mu^2/phi_t`, one dispersion `phi_t` per trait (the
convention of gllvmTMB's plain `nbinom2()`; R reuses the same `log_phi_nbinom2`
vector). Only the log link is admitted.
"""
function zi_nbinom2(; link = LogLink())
    _zi_check_link(ZiNbinom2(), link)
    return ZiNbinom2()
end

"""
    zi_binomial(; link = LogitLink()) -> ZiBinomial

Julia twin of gllvmTMB's `zi_binomial()`: the zero-inflation mixture over a
`Binomial(N_ti, p)` count, `p = logistic(beta_t + Lambda_t' z)`, with trials
`N_ti` per observation (pass `trials`, a p×n integer matrix or one integer, to
the fitter). As in R, every trait needs at least one observation with
`N_ti >= 2`: with single-trial (0/1) data `P(y=1) = (1-zi) p` does not separate
`zi` from `p`, and the fit is refused with `Binomial()` named as the
alternative. Only the logit link is admitted.
"""
function zi_binomial(; link = LogitLink())
    _zi_check_link(ZiBinomial(), link)
    return ZiBinomial()
end

# ---------------------------------------------------------------------------
# Per-cell markers for the two-part substrate. Each carries the constants the
# cell's density needs besides (eta^z, eta^c): the trait's logit_zi (needed by
# the observed-curvature hook, which receives eta^c only), and phi or N.
# `_tp_pieces` delegates to Julia's existing ZIPoisson / ZINB / ZIB pieces, whose
# log-density is the same mixture density R evaluates; only the log-det weight
# differs, supplied below.
# ---------------------------------------------------------------------------
struct _ZiPoisCell
    logit_zi::Float64
end
struct _ZiNB2Cell
    logit_zi::Float64
    phi::Float64
end
struct _ZiBinCell
    logit_zi::Float64
    N::Int
end

_tp_pieces(c::_ZiPoisCell, y, ηz, ηc) = _tp_pieces(ZIPoisson(), y, ηz, ηc)
_tp_pieces(c::_ZiNB2Cell, y, ηz, ηc) = _tp_pieces(ZINB(c.phi), y, ηz, ηc)
_tp_pieces(c::_ZiBinCell, y, ηz, ηc) = _tp_pieces(ZIB(c.N), y, ηz, ηc)

# Observed count curvature -d^2 log f / d eta^c^2 at the cell's own logit_zi, by
# nested forward-mode AD of the mixture log-density itself (the quantity TMB's AD
# differentiates), so no hand-derived second derivative is involved.
function _zi_observed_curvature(c, y, ηc)
    d1(e) = ForwardDiff.derivative(e2 -> _tp_pieces(c, y, c.logit_zi, e2)[5], e)
    return -ForwardDiff.derivative(d1, ηc)
end
_tp_observed_Wc(c::_ZiPoisCell, y, ηc, Wc) = _zi_observed_curvature(c, y, ηc)
_tp_observed_Wc(c::_ZiNB2Cell, y, ηc, Wc) = _zi_observed_curvature(c, y, ηc)
_tp_observed_Wc(c::_ZiBinCell, y, ηc, Wc) = _zi_observed_curvature(c, y, ηc)

# ---------------------------------------------------------------------------
# Laplace breakdown guard (Julia-side; not part of R's model).
#
# At y = 0 the observed curvature of the zero-inflation mixture is negative for
# moderate means (the zero can come from either process, so log f(0 | eta) is locally
# convex in eta). The site precision A = I + Λ' diag(W_obs) Λ can then fall towards 0
# while the mode search still converges, and -1/2 logdet(A) inflates the Laplace
# value. The surface is R's too: gllvmTMB's TMB objective returns the same inflated
# value at such a point, and R reaches a sensible optimum only through its start.
# Quadrature on one such dataset (K = 1, 6001-point grid) put the exact marginal at a
# near-singular Laplace maximum 363 log-likelihood units BELOW the value Laplace
# reports there, and below the sensible optimum (docs/dev-log/decisions/
# 2026-09-27-zi-laplace-breakdown-guard.md).
#
# A site whose A has an eigenvalue below the floor is treated as a failed Laplace
# evaluation (-Inf, which the fitter turns into its 1e12 sentinel). Floor = 0.1: half
# the smallest site eigenvalue measured at any sensible R optimum (0.20 on the P1
# zi_binomial fixture; >= 0.50 on the NB2 draws and the other two fixtures), so the
# guard never touches those, while it cuts off the near-singular end (1.8e-4 and 3e-4
# at the two spurious maxima measured). It does NOT remove Laplace error above the
# floor: walling at 0.5 left a point with 27 units of Laplace error on the same data.
# The start (below) and one shrunk-start retry reach the sensible optimum on most
# measured draws, not all (rates in the decisions note); the floor stops a fit that
# leaves the sensible basin from running to the singular end.
#
# The floor is safe for the optimiser only together with the "within 10% of the
# floor -> converged = false" rule in fit_zi_gllvm: Optim reports converged = true
# for a fit stalled at the wall (a line search into the 1e12 sentinel becomes a zero
# step), so that rule is load-bearing. Do not relax it as redundant.
# ---------------------------------------------------------------------------
"""
    ZI_LAPLACE_EIGMIN_FLOOR

Smallest admissible eigenvalue (0.1) of the per-site Laplace precision
`A = I + Λ' diag(W) Λ` on the [`zi_poisson`](@ref) / [`zi_nbinom2`](@ref) /
[`zi_binomial`](@ref) route. Below it the site's Laplace value is treated as a
failed evaluation: a Julia-side guard against a near-singular Laplace breakdown
that gllvmTMB's own objective shares (the repository's decision note on the
zero-inflated Laplace breakdown guard records the evidence).
"""
const ZI_LAPLACE_EIGMIN_FLOOR = 0.1

function _zi_loglik_site(fams, y, Λz, Λ, logit_zi, β; eigmin_floor, maxiter, tol)
    p, K = size(Λ)
    ẑ, ok = _twopart_mode_search(fams, y, Λz, Λ, logit_zi, β; maxiter = maxiter, tol = tol)
    ok || return -Inf
    ηz = _clamp_eta.(logit_zi .+ Λz * ẑ)
    ηc = _clamp_eta.(β .+ Λ * ẑ)
    W = Vector{Float64}(undef, p)
    ℓ = 0.0
    @inbounds for t in 1:p
        _, _, _, W_c, logf = _tp_pieces(fams[t], y[t], ηz[t], ηc[t])
        W[t] = _tp_observed_Wc(fams[t], y[t], ηc[t], W_c)
        ℓ += logf
    end
    A = Symmetric(Λ' * (W .* Λ) + I)
    eigmin(A) >= eigmin_floor || return -Inf
    return ℓ - 0.5 * dot(ẑ, ẑ) - 0.5 * logdet(A)
end

_zi_cell(::ZiPoisson, logit_zi, phi, N) = _ZiPoisCell(logit_zi)
_zi_cell(::ZiNbinom2, logit_zi, phi, N) = _ZiNB2Cell(logit_zi, phi)
_zi_cell(::ZiBinomial, logit_zi, phi, N) = _ZiBinCell(logit_zi, N)

"""
    zi_marginal_loglik_laplace(family, Y, Λ, β, logit_zi; phi = nothing,
                               trials = nothing, eigmin_floor = ZI_LAPLACE_EIGMIN_FLOOR,
                               maxiter = 100, tol = 1e-9) -> Float64

Laplace log-marginal likelihood of gllvmTMB's zero-inflated GLLVM for
`family` in `zi_poisson()`, `zi_nbinom2()`, `zi_binomial()`: count linear
predictor `eta = β_t + Λ_t' z_s`, `z_s ~ N(0, I_K)` per site (column of the p×n
`Y`), per-trait structural-zero logit `logit_zi`, per-trait NB2 dispersion `phi`
(`zi_nbinom2()` only) and trials `trials` (`zi_binomial()` only; p×n integer
matrix). The Laplace log-determinant uses the observed curvature, as TMB does.
With `Λ = 0` the value is the exact independent-mixture log-likelihood. Returns
`-Inf` if a site's mode search fails, or if a site's Laplace precision has an
eigenvalue below `eigmin_floor` (default [`ZI_LAPLACE_EIGMIN_FLOOR`](@ref); pass
`-Inf` for the unguarded value, which is what gllvmTMB's objective returns).
"""
function zi_marginal_loglik_laplace(family::_ZiTwinFamily, Y::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, logit_zi::AbstractVector;
        phi = nothing, trials = nothing, eigmin_floor::Real = ZI_LAPLACE_EIGMIN_FLOOR,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, n = size(Y)
    K = size(Λ, 2)
    size(Λ, 1) == p || throw(DimensionMismatch("Λ has $(size(Λ, 1)) rows; Y has $p traits"))
    length(β) == p && length(logit_zi) == p || throw(DimensionMismatch(
        "β and logit_zi must have length p = $p"))
    if family isa ZiNbinom2
        (phi !== nothing && length(phi) == p) || throw(ArgumentError(
            "zi_nbinom2: phi must be a length-p vector of per-trait dispersions"))
    end
    if family isa ZiBinomial
        (trials !== nothing && size(trials) == (p, n)) || throw(ArgumentError(
            "zi_binomial: trials must be a p×n integer matrix"))
    end
    Λz = zeros(p, K)
    acc = 0.0
    cells = Vector{Any}(undef, p)
    @inbounds for s in 1:n
        for t in 1:p
            cells[t] = _zi_cell(family, Float64(logit_zi[t]),
                                phi === nothing ? NaN : Float64(phi[t]),
                                trials === nothing ? 0 : Int(trials[t, s]))
        end
        fams = [c for c in cells]          # concretely typed per-site vector
        acc += _zi_loglik_site(fams, view(Y, :, s), Λz, Λ, logit_zi, β;
                               eigmin_floor = eigmin_floor, maxiter = maxiter, tol = tol)
        isfinite(acc) || return -Inf
    end
    return acc
end

"""
    ZiFit

Result of [`fit_zi_gllvm`](@ref) (gllvmTMB `zi_poisson()` / `zi_nbinom2()` /
`zi_binomial()` semantics). Fields:

- `family::Symbol`: `:zi_poisson`, `:zi_nbinom2` or `:zi_binomial`;
- `β`: per-trait count intercepts (log mean, or logit success probability);
- `Λ`: p×K count loadings (lower triangular; identified up to rotation and sign,
  so compare `Λ * Λ'` across engines);
- `logit_zi`, `zi`: per-trait structural-zero logit and probability
  (gllvmTMB's `fit\$report\$zi`);
- `phi`: per-trait NB2 dispersion (`Var = mu + mu^2/phi`) for `:zi_nbinom2`,
  empty otherwise (gllvmTMB's `fit\$report\$phi_nbinom2`);
- `trials`: the p×n trials matrix for `:zi_binomial`, `nothing` otherwise;
- `loglik`, `converged`, `iterations`;
- `min_site_eigen`: the smallest eigenvalue of the per-site Laplace precision
  `A = I + Λ' diag(W) Λ` over sites at the returned point. A fit whose optimum
  sits at the breakdown guard ([`ZI_LAPLACE_EIGMIN_FLOOR`](@ref), within 10%) is
  reported with `converged = false`: its Laplace value is not a usable
  log-likelihood (gllvmTMB's objective shares this surface).
"""
struct ZiFit
    family::Symbol
    β::Vector{Float64}
    Λ::Matrix{Float64}
    logit_zi::Vector{Float64}
    zi::Vector{Float64}
    phi::Vector{Float64}
    trials::Union{Nothing, Matrix{Int}}
    loglik::Float64
    converged::Bool
    iterations::Int
    min_site_eigen::Float64
end

function Base.show(io::IO, f::ZiFit)
    p, K = size(f.Λ)
    print(io, "ZiFit(", f.family, ", p=", p, ", K=", K,
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

# Admission, mirroring R/fit-multi.R for fid 17/18/19.
function _zi_admit(family::_ZiTwinFamily, Y::AbstractMatrix, trials)
    p, n = size(Y)
    # `missing` (in a Union{Missing, Real} matrix) and NaN get the same refusal.
    all(y -> !ismissing(y) && isfinite(y) && y >= 0 && y == round(y), Y) || throw(ArgumentError(
        "$(_zi_rname(family)): Y must hold non-negative integer counts with no missing " *
        "values (gllvmTMB masks missing responses row by row; this route does not yet)."))
    if !(family isa ZiBinomial)
        trials === nothing || throw(ArgumentError(
            "$(_zi_rname(family)): `trials` applies to zi_binomial() only."))
        return nothing
    end
    trials === nothing && throw(ArgumentError(
        "zi_binomial: supply `trials` (a p×n integer matrix, or one integer)."))
    N = trials isa Integer ? fill(Int(trials), p, n) : Matrix{Int}(round.(Int, trials))
    size(N) == (p, n) || throw(DimensionMismatch(
        "zi_binomial: trials is $(size(N)); Y is $((p, n))"))
    trials isa Integer || all(trials .== N) || throw(ArgumentError(
        "zi_binomial: trials must be integers"))
    all(N .>= 0) && all(Y .<= N) || throw(ArgumentError(
        "zi_binomial: successes must satisfy 0 <= y <= trials."))
    bad = [t for t in 1:p if !any(>=(2), view(N, t, :))]
    isempty(bad) || throw(ArgumentError(
        "zi_binomial: single-trial (0/1) responses do not identify the model. " *
        "Trait(s) $(join(bad, ", ")) have no observation with trials >= 2; with N = 1, " *
        "P(y = 1) = (1 - zi) p collapses the structural-zero probability and the " *
        "count probability into one free product. Supply multi-trial data, or use " *
        "Binomial() if the data really are single-trial."))
    return N
end

# Warm start. logit_zi follows gllvmTMB's `zi_logit_start()` (R/dispersion-trait-
# map.R): method of moments on the observed zero share against a naive count-zero
# probability, clamped to [0.02, 0.8]. β from the positive observations, Λ from an
# SVD of the working residuals (half scale), NB2 phi from a moment estimate.
function _zi_twin_warmstart(family::_ZiTwinFamily, Y::AbstractMatrix, N, K::Integer)
    p, n = size(Y)
    β0 = zeros(p); lz0 = zeros(p)
    Z = zeros(p, n)
    for t in 1:p
        yt = view(Y, t, :)
        p0obs = count(==(0), yt) / n
        if family isa ZiBinomial
            Nt = view(N, t, :)
            pbar = sum(Nt) > 0 ? sum(yt) / sum(Nt) : 0.5
            pc0 = (1 - pbar)^(sum(Nt) / n)
            pos = [j for j in 1:n if yt[j] > 0]
            ppos = isempty(pos) ? 0.5 : sum(yt[pos]) / sum(Nt[pos])
            ppos = clamp(ppos, 0.02, 0.98)
            β0[t] = log(ppos / (1 - ppos))
            for j in pos
                q = clamp(yt[j] / Nt[j], 0.02, 0.98)
                Z[t, j] = log(q / (1 - q)) - β0[t]
            end
        else
            mbar = sum(yt) / n
            pc0 = family isa ZiNbinom2 ? 1 / (1 + mbar) : exp(-mbar)
            pos = [j for j in 1:n if yt[j] > 0]
            mpos = isempty(pos) ? 1.0 : max(sum(yt[pos]) / length(pos), 1.0)
            β0[t] = log(mpos)
            for j in pos
                Z[t, j] = log(max(yt[j], 0.5)) - β0[t]
            end
        end
        pihat = clamp((p0obs - pc0) / max(1 - pc0, 1e-6), 0.02, 0.8)
        lz0[t] = log(pihat / (1 - pihat))
    end
    F = svd(Z); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    for j in 1:kk
        # Half the SVD scale: the log-count residuals of positive observations
        # overstate the latent spread (they also carry the count noise), and a start
        # with large loadings on a trait with many zeros sits near the region where
        # the zero-inflation Laplace breaks down (see ZI_LAPLACE_EIGMIN_FLOOR).
        Λ0[:, j] = F.U[:, j] .* (0.5 * F.S[j] / sqrt(n))
    end
    # NB2 dispersion: moment estimate from the positive counts, Var = m + m^2/phi,
    # clamped to [0.2, 20] (R starts every trait at phi = 1).
    logphi0 = zeros(p)
    if family isa ZiNbinom2
        for t in 1:p
            pos = [Y[t, j] for j in 1:n if Y[t, j] > 0]
            if length(pos) >= 2
                m = sum(pos) / length(pos)
                v = sum(abs2, pos .- m) / (length(pos) - 1)
                logphi0[t] = log(clamp(m^2 / max(v - m, 1e-3 * m), 0.2, 20.0))
            end
        end
    end
    return β0, lz0, Λ0, logphi0
end

"""
    fit_zi_gllvm(Y; family, K, trials = nothing, link = nothing, hessian = nothing,
                 eigmin_floor = ZI_LAPLACE_EIGMIN_FLOOR,
                 g_tol = 1e-6, iterations = 1000,
                 newton_maxiter = 100, newton_tol = 1e-9) -> ZiFit

Fit gllvmTMB's zero-inflated GLLVM (`family = zi_poisson()`, `zi_nbinom2()` or
`zi_binomial()`) by L-BFGS over `[β; logit_zi; pack(Λ); log phi]` (`log phi`
for `zi_nbinom2()` only), maximising the Laplace marginal of
[`zi_marginal_loglik_laplace`](@ref). `Y` is p×n (traits × sites) counts;
`trials` (p×n, or one integer) is required for `zi_binomial()` and refused
otherwise. A `link` other than the family's only link is refused.

The model is R's: a per-trait, intercept-only structural-zero probability (no
covariates and no latent loadings on the zero part), the count process active
at every observation, per-trait NB2 dispersion, per-observation binomial
trials, and a Laplace log-determinant from the observed curvature. It differs
from Julia's own zero-inflated fitters, which remain available unchanged:
[`fit_zip_gllvm`](@ref) / [`fit_zinb_gllvm`](@ref) / [`fit_zib_gllvm`](@ref)
use the expected (Fisher) count weight in the log-determinant, and
`fit_zinb_gllvm` estimates one shared dispersion, `fit_zib_gllvm` takes one
shared trials count. Only the no-covariate, no-row-effect model is offered on
this route; `predict`, `confint`, `simulate` and the `@formula` front door with
covariates are not wired for [`ZiFit`](@ref). Missing responses (`missing` in a
`Union{Missing, Real}` matrix, or `NaN`) are refused with an `ArgumentError`
(gllvmTMB masks them per row). `hessian` is accepted only as `:observed`.
`eigmin_floor` sets the Laplace breakdown guard ([`ZI_LAPLACE_EIGMIN_FLOOR`](@ref));
an optimum at the guard is reported with `converged = false`.
"""
function fit_zi_gllvm(Y::AbstractMatrix{<:Union{Missing, Real}}; family::_ZiTwinFamily, K::Integer,
        trials = nothing, link = nothing, hessian = nothing,
        eigmin_floor::Real = ZI_LAPLACE_EIGMIN_FLOOR,
        g_tol::Real = 1e-6, iterations::Integer = 1000,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    link === nothing || _zi_check_link(family, link)
    (hessian === nothing || hessian === :observed) || throw(ArgumentError(
        "$(_zi_rname(family)): this route always uses the observed curvature in the " *
        "Laplace log-determinant, as gllvmTMB does; `hessian = $(repr(hessian))` is not " *
        "offered. Julia's own ZIPoisson() / ZINegBin() / ZIB(N) routes take `hessian`."))
    K >= 1 || throw(ArgumentError("fit_zi_gllvm: K must be >= 1 (got $K)"))
    N = _zi_admit(family, Y, trials)
    Yf = Matrix{Float64}(Y)
    p, n = size(Yf)
    rr = rr_theta_len(p, K)
    isnb = family isa ZiNbinom2
    β0, lz0, Λ0, logphi0 = _zi_twin_warmstart(family, Yf, N, K)
    function unpack(θ)
        β = θ[1:p]; lz = θ[(p + 1):(2p)]
        Λ = unpack_lambda(θ[(2p + 1):(2p + rr)], p, K)
        phi = isnb ? exp.(θ[(2p + rr + 1):(3p + rr)]) : nothing
        return β, lz, Λ, phi
    end
    function negll(θ)
        β, lz, Λ, phi = unpack(θ)
        v = try
            -zi_marginal_loglik_laplace(family, Yf, Λ, β, lz; phi = phi, trials = N,
                                        eigmin_floor = eigmin_floor, maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    # One L-BFGS run from a start; returns the unpacked optimum, Optim's verdict and the
    # smallest site eigenvalue there.
    function run_from(Λstart)
        θ0 = isnb ? vcat(β0, lz0, pack_lambda(Λstart), logphi0) :
                    vcat(β0, lz0, pack_lambda(Λstart))
        res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                             autodiff = :finite)
        β, lz, Λ, phi = unpack(Optim.minimizer(res))
        loglik, conv, iters = _fit_verdict(res)
        mineig = _zi_min_site_eigen(family, Yf, Λ, β, lz; phi = phi, trials = N,
                                    maxiter = newton_maxiter, tol = newton_tol)
        return (β = β, lz = lz, Λ = Λ, phi = phi, loglik = loglik, converged = conv,
                iters = iters, mineig = mineig)
    end
    # An optimum within 10% of the floor sits at the guard. This rule is load-bearing,
    # not belt-and-braces: when a line search runs into the 1e12 sentinel, L-BFGS with
    # BackTracking takes a zero step and Optim reports converged = true AT the wall
    # (measured on every wall-stalled fit in the PR #557 review). Without this rule
    # those fits would be reported converged at a Laplace value that is not a usable
    # log-likelihood.
    at_guard(r) = isfinite(eigmin_floor) && r.mineig < 1.1 * eigmin_floor
    r = run_from(Λ0)
    if at_guard(r)
        # One retry from a shrunk start: loadings at a tenth of the default start
        # (a twentieth of the SVD scale), with β, logit_zi and the moment NB2 phi reset to
        # their warm-start values. A default-start fit can stall at the wall on data
        # where a sensible optimum exists (1 of 35 NB2 draws measured in the PR #557
        # review; gllvmTMB reached that optimum from its own start, and this retry
        # reaches it too; scales 0.25 and 0.5 did not). Keep the retry only if it is
        # off the guard; otherwise report the first fit as flagged. Rates: the
        # decisions note 2026-09-27-zi-laplace-breakdown-guard.md.
        r2 = run_from(0.1 .* Λ0)
        at_guard(r2) || (r = r2)
    end
    β, lz, Λ, phi = r.β, r.lz, r.Λ, r.phi
    zi = @. inv(1 + exp(-lz))
    loglik, converged, iters, mineig = r.loglik, r.converged, r.iters, r.mineig
    if converged && at_guard(r)
        converged = false
        @warn "$(_zi_rname(family)): the optimum sits at the Laplace breakdown guard " *
              "(smallest site precision eigenvalue $(round(mineig; sigdigits = 3)), floor " *
              "$(eigmin_floor)). The Laplace approximation is not reliable here and the " *
              "fit is reported as not converged; gllvmTMB's objective shares this surface."
    end
    return ZiFit(Symbol(_zi_rname(family)), β, Λ, lz, zi,
                 phi === nothing ? Float64[] : phi, N, loglik, converged, iters, mineig)
end

# Smallest per-site Laplace precision eigenvalue at given parameters (Inf if n = 0,
# -Inf if a site's mode search fails).
function _zi_min_site_eigen(family::_ZiTwinFamily, Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, logit_zi::AbstractVector; phi = nothing, trials = nothing,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, n = size(Y); K = size(Λ, 2); Λz = zeros(p, K)
    m = Inf
    for s in 1:n
        fams = [_zi_cell(family, Float64(logit_zi[t]), phi === nothing ? NaN : Float64(phi[t]),
                         trials === nothing ? 0 : Int(trials[t, s])) for t in 1:p]
        y = view(Y, :, s)
        ẑ, ok = _twopart_mode_search(fams, y, Λz, Λ, logit_zi, β; maxiter = maxiter, tol = tol)
        ok || return -Inf
        ηc = _clamp_eta.(β .+ Λ * ẑ)
        W = [_tp_observed_Wc(fams[t], y[t], ηc[t], NaN) for t in 1:p]
        m = min(m, eigmin(Symmetric(Λ' * (W .* Λ) + I)))
    end
    return m
end

_fit_gllvm(family::_ZiTwinFamily, Y::AbstractMatrix; kwargs...) =
    fit_zi_gllvm(Y; family = family, kwargs...)
