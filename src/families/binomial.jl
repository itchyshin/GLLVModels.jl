# Per-site Laplace marginal log-likelihood for the Binomial GLLVM.
#
# Model (site s, p binary/binomial responses):
#     y_{ts} ~ Binomial(n_{ts}, μ_{ts}),  μ_{ts} = linkinv(link, η_{ts}),
#     η_{ts} = β_t + (Λ z_s)_t,           z_s ~ N(0, I_K).
#
# Predictor-informed latent-score mean (C1 / Design 73):
#     z_total,s = X_lv[s, :] * alpha_lv + z_s,  z_s ~ N(0, I_K),
#     η_ts = β_t + (Λ z_total,s)_t.
#
# Conditional on the zero-mean innovation z_s this is the same Laplace problem
# with a parameter-dependent offset `Λ * alpha_lv' * X_lv[s, :]`.
#
# The marginal  ∫ p(y_s | z) N(z; 0, I) dz  is non-conjugate, so it is computed
# by a Laplace approximation: find the conditional mode ẑ_s by Fisher scoring,
# then
#     log p(y_s) ≈ ℓ(ẑ_s) − ½ ẑ_s'ẑ_s − ½ logdet(Λ' W Λ + I_K),
# where ℓ is the binomial log-likelihood and W are the Fisher working weights at
# the mode. This is the smallest correctness unit of the Binomial family (#7);
# the fit driver and gradient build on it. See the design note in the after-task
# log. Inner mode-finder uses the Fisher information (expected Hessian), so
# Λ' W Λ + I_K is always SPD.

# Binomial family pieces for the generic Laplace core (src/families/laplace.jl).
# y_t ~ Binomial(n_t, μ_t); E[y]=nμ, Var=nμ(1−μ). Score/weight wrt η below; with
# the logit link (me = μ(1−μ)) the weight reduces to the canonical nμ(1−μ).
_clamp_mu(::Binomial, μ) = clamp(μ, 1e-12, 1 - 1e-12)
_glm_score(::Binomial, μ, n, me, y) = (y - n * μ) / (μ * (one(μ) - μ)) * me
# Canonical link: −∂²ℓ/∂η² = n·μ(1−μ) is y-free, so observed ≡ Fisher pointwise.
# NOTE the link specificity — under ProbitLink/CLogLogLink the two differ, and
# those combinations deliberately do NOT get this method.
_glm_weight_matches_observed(::Binomial, ::LogitLink) = true

_glm_weight(::Binomial, μ, n, me)   = n * me^2 / (μ * (one(μ) - μ))
_glm_logpdf(::Binomial, μ, n, y)    = logpdf(Binomial(Int(n), μ), Int(y))

# Observed conditional curvature for Binomial/probit: −∂²ℓ/∂η² with μ = Φ(η),
# me = dμ/dη = φ(η) (standard normal pdf). Derivation: ℓ(η) = y·log μ +
# (n−y)·log(1−μ); with s(η) = y/μ − (n−y)/(1−μ) = (y−nμ)/(μ(1−μ)) (the score
# form already shared by `_glm_score` above, which is link-generic in `me`),
# dℓ/dη = me·s(η). Differentiating again, using dme/dη = −η·me (the standard
# normal pdf's own derivative, φ'(x) = −x·φ(x)):
#   d²ℓ/dη² = −η·me·s(η) − me²·[y/μ² + (n−y)/(1−μ)²]
#   W_obs   = η·me·(y−nμ)/(μ(1−μ)) + me²·[y/μ² + (n−y)/(1−μ)²]
# FD-verified against the FULL `_glm_logpdf` to ≤ 3e-7 relative gap across
# n ∈ {1,4,10}, η ∈ [-1.5,1.8], y ∈ 0:n (scratchpad/fd_probe_tweedie_probit.jl,
# 2026-08-28). Symmetry check holds: at y = n·μ (s(η) = 0) this collapses to
# me²·n·[1/μ + 1/(1−μ)] = n·me²/(μ(1−μ)) = `_glm_weight` exactly.
#
# NOT genuinely sign-changing, unlike Beta/Student-t: W_obs(y) is AFFINE in y
# for fixed η (both η-dependent coefficients are y-free), so its extrema over
# y ∈ [0,n] sit at y = 0 and y = n; a BigFloat evaluation at those endpoints
# across η ∈ [3,20] stayed strictly positive (converging to n as |η| grows —
# consistent with the classical GLM fact, Pratt 1981 JASA, that the probit
# binomial log-likelihood is concave in η for every y). The Float64 grid probe
# in the same scratchpad script DOES show apparent negative values right at
# η ≈ ±8, but that is a catastrophic-cancellation ARTIFACT of μ(1−μ)
# underflowing near the Float64 saturation boundary, not a genuine sign
# change — confirmed by re-evaluating the same cells in BigFloat. In the
# fitted pipeline `_clamp_mu` (above) bounds μ away from exact 0/1 before this
# function ever sees it, keeping μ(1−μ) ≥ ~1e-12 and the cancellation
# unreachable. The assembly-level PD guard in `marginal_loglik_laplace`
# remains the correct place for any residual defence — this weight is not
# clamped here, matching the repo-wide convention.
_glm_obs_weight(f::Binomial, μ, n, me, y, link::ProbitLink, η) =
    η * me * (y - n * μ) / (μ * (one(μ) - μ)) +
    me^2 * (y / μ^2 + (n - y) / (one(μ) - μ)^2)
_default_hessian(::Binomial, ::ProbitLink) = :observed

# CLogLogLink is non-canonical (like ProbitLink): observed ≠ Fisher pointwise,
# so it needs the same treatment. No hand-coded `_glm_obs_weight` override is
# supplied here — the generic nested-ForwardDiff fallback in
# `families/laplace.jl` (`_glm_obs_weight(family, μ, n, me, y, link::Link, η)`)
# already computes the correct observed curvature for cloglog; adding a
# closed-form override is optional future work, not required for correctness.
#
# CONFIRMED (2026-09-01, maintainer decisions round 1, item 2 —
# `docs/dev-log/decisions/2026-09-01-maintainer-decisions-round1.md`): at R's
# fitted coordinates on the seed-81012 retained fixture, Julia's `:fisher`
# marginal disagreed with R/`gllvmTMB`'s value by 2.0988510... nats. Exact
# quadrature of the Bernoulli-cloglog Laplace integral at the largest-gap site
# confirms `:observed` is the correct curvature (|gap to quadrature| =
# 0.0099 vs `:fisher`'s 0.1081, an ~11× improvement) and matches R to
# 7.4e-12. R/TMB differentiates its coded joint negative log-density via AD,
# which is definitionally the observed Hessian — TMB never substitutes the
# Fisher/expected information. `:fisher` was a genuine Julia-side defect, not
# an R deviation. See `docs/dev-log/core070/cloglog-leaf-notes.md` for the
# full diagnosis (ground-truth evidence, per-site gap decomposition,
# probe outputs) and its relationship to the SEPARATE 2026-08-28
# optimizer-runaway finding below (`_laplace_saturation_health` /
# `LaplaceSaturationHealth`): that pathology was measured under BOTH
# curvature selectors, so it is not a reason to keep `:fisher` as the
# marginal-likelihood default — the saturation guard, not a wrong-by-default
# likelihood value, is the correct place to police that runaway.
_default_hessian(::Binomial, ::CLogLogLink) = :observed

# Binomial-default convenience methods (back-compat: family ⇒ Binomial()), used
# by getLV(::BinomialFit) and the Binomial tests.
_laplace_mode(y::AbstractVector, n::AbstractVector, Λ::AbstractMatrix,
        β::AbstractVector, link::Link; kwargs...) =
    _laplace_mode(Binomial(), y, n, Λ, β, link; kwargs...)

laplace_loglik_site(y::AbstractVector, n::AbstractVector, Λ::AbstractMatrix,
        β::AbstractVector, link::Link; kwargs...) =
    laplace_loglik_site(Binomial(), y, n, Λ, β, link; kwargs...)

"""
    binomial_marginal_loglik_laplace(Y, N, Λ, β, link; kwargs...) -> Float64

Total Laplace log-marginal over the `n` sites of a Binomial GLLVM — a thin
wrapper over the family-generic `marginal_loglik_laplace` with `Binomial()`.
`Y`, `N` are p×n response and trial-count matrices; `Λ` p×K; `β` length-p.
"""
binomial_marginal_loglik_laplace(Y::AbstractMatrix, N::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, link::Link; kwargs...) =
    marginal_loglik_laplace(Binomial(), Y, N, Λ, β, link; kwargs...)

# ---------------------------------------------------------------------------
# Fit driver (Binomial slice 4).
# ---------------------------------------------------------------------------

"""
    BinomialFit

Result of [`fit_binomial_gllvm`](@ref): intercepts `β` (length p), loadings `Λ`
(p×K), the `link`, the maximised Laplace `loglik`, the optimiser `converged`
flag, and `iterations`. Fits using `X_lv` additionally retain `alpha_lv`, the
raw latent-axis coefficients for the predictor-informed score mean; use
[`extract_lv_effects`](@ref) for the rotation-stable trait-scale product
`Λ * alpha_lv'`. `loading_ridge` records the `loading_ridge` fit kwarg (`Inf`
when the loading ridge was off); `loglik` is always the UNPENALISED Laplace
marginal, evaluated at the (possibly ridge-penalised) optimum.
"""
# Post-fit Laplace saturation health (2026-08-28, the diagnosed cloglog
# pathology). A SATURATED cell is one whose per-site conditional mode drives
# the linear predictor to (or past) the link's μ-saturation thresholds
# (η where linkinv(η) hits the `_clamp_mu` bounds 1e-12 / 1−1e-12), or whose
# log-det weight under the fit's own curvature selector has collapsed
# (W ≤ 1e-8). At such cells the log-det penalty is effectively deleted, and
# the Laplace value can overstate the exact marginal without bound — measured
# +74.8 loglik units at a cloglog runaway with `converged = true` (check-log
# 2026-08-28). Saturation MAY be latent-mode driven (a runaway ‖Λ̂‖) or purely
# intercept-driven (extreme prevalence / separation): the diagnostic reports,
# it does not adjudicate. `nothing` means "not computed" (plateau verdicts,
# compat-constructed fits, non-dense kernels, VA fits).
struct LaplaceSaturationHealth
    n_clamp::Int         # cells whose mode-η reached a μ-saturation threshold (incl. the ±30 η-clamp)
    n_wcollapse::Int     # cells whose log-det weight ≤ 1e-8 under the fit's selector
    n_obs::Int           # observed (unmasked) cells assessed
    max_abs_eta::Float64 # largest |η̂| over observed cells at the fitted modes
    hessian_used::Symbol
end

struct BinomialFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    alpha_lv::Union{Nothing, Matrix{Float64}}
    theta_packed::Vector{Float64}
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used
    saturation::Union{Nothing, LaplaceSaturationHealth}
    integration::Union{Nothing,AGHQFitInfo}
    loading_ridge::Float64   # the fit's `loading_ridge` kwarg; `Inf` = ridge off
end
# Pre-ridge compat tier (11 positional args, the prior full field list):
# defaults the new `loading_ridge` field to `Inf` (ridge off), matching every
# pre-existing construction site's implicit behaviour.
BinomialFit(β,Λ,link,loglik,converged,iterations,alpha_lv,theta_packed,hessian,saturation,integration)=
    BinomialFit(β,Λ,link,loglik,converged,iterations,alpha_lv,theta_packed,hessian,saturation,integration,Inf)
BinomialFit(β,Λ,link,loglik,converged,iterations,alpha_lv,theta_packed,hessian,saturation)=
    BinomialFit(β,Λ,link,loglik,converged,iterations,alpha_lv,theta_packed,hessian,saturation,nothing)

# Positional compatibility constructor (2026-08-28): every pre-existing
# construction site builds a default-curvature fit; the `hessian` field
# records the objective identity so `confint`/bootstrap can rebuild THE
# SAME objective instead of guessing (the audit's confint-consistency class).
BinomialFit(β, Λ, link, loglik, converged, iterations, alpha_lv, theta_packed) =
    BinomialFit(β, Λ, link, loglik, converged, iterations, alpha_lv, theta_packed, _default_hessian(Binomial(), link))
# 9-arg tier (with hessian, pre-saturation): saturation defaults to "not computed".
BinomialFit(β, Λ, link, loglik, converged, iterations, alpha_lv, theta_packed, hessian::Symbol) =
    BinomialFit(β, Λ, link, loglik, converged, iterations, alpha_lv, theta_packed, hessian, nothing)

BinomialFit(β::Vector{Float64}, Λ::Matrix{Float64}, link::Link,
            loglik::Float64, converged::Bool, iterations::Int) =
    BinomialFit(β, Λ, link, loglik, converged, iterations, nothing, Float64[])

function Base.show(io::IO, f::BinomialFit)
    p, K = size(f.Λ)
    print(io, "BinomialFit(p=", p, ", K=", K, ", link=", nameof(typeof(f.link)),
          f.alpha_lv === nothing ? "" : ", X_lv=true",
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED",
          (f.saturation !== nothing && (f.saturation.n_clamp > 0 || f.saturation.n_wcollapse > 0)) ?
              ", SATURATED ($(max(f.saturation.n_clamp, f.saturation.n_wcollapse)) cells)" : "", ")")
end

"""
    binomial_lv_nll_packed(params, Y, N, p, K, link; X_lv, q_lv, kwargs...) -> Real

Negative Laplace log-likelihood for the predictor-informed latent-score
binomial model. Parameter layout:

- `params[1:p]` = per-trait intercepts `β`;
- next `q_lv * K` entries = `alpha_lv`, reshaped as `q_lv × K`;
- remaining entries = packed reduced-rank loadings `Λ`.

The conditional latent variable is the zero-mean innovation. The predictor mean
enters the Laplace core as the parameter-dependent offset
`Λ * alpha_lv' * X_lv[s, :]`.

`loading_ridge` (default `Inf`, i.e. off) adds `0.5 * sum(Λ.^2) / loading_ridge^2`
to the returned value — the same loading-ridge penalty as the no-`X_lv` route
(see [`fit_binomial_gllvm`](@ref)); autodiff (this path always uses
`autodiff = :finite`) differentiates it along with everything else, so no
separate analytic-gradient term is needed here.
"""
function binomial_lv_nll_packed(params::AbstractVector, Y::AbstractMatrix,
        N::AbstractMatrix, p::Integer, K::Integer, link::Link;
        X_lv::AbstractMatrix, q_lv::Integer,
        mask = nothing, offset = nothing, loading_ridge::Real = Inf,
        maxiter::Integer = 100, tol::Real = 1e-9)
    size(Y, 1) == p ||
        throw(ArgumentError("Y first dim ($(size(Y, 1))) must equal p ($p)"))
    n = size(Y, 2)
    size(N) == (p, n) || throw(DimensionMismatch("N must be $(p)×$(n)"))
    size(X_lv, 1) == n ||
        throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n_sites ($n)"))
    size(X_lv, 2) == q_lv ||
        throw(ArgumentError("X_lv second dim ($(size(X_lv, 2))) must equal q_lv ($q_lv)"))
    q_lv > 0 || throw(ArgumentError("q_lv must be positive"))

    rr = rr_theta_len(p, K)
    n_expected = p + q_lv * K + rr
    length(params) == n_expected || throw(ArgumentError(
        "params length ($(length(params))) must equal $n_expected " *
        "(p=$p + alpha_lv=$(q_lv * K) + rr=$rr)"))

    cursor = 0
    β = @view params[(cursor + 1):(cursor + p)]
    cursor += p
    alpha_vec = @view params[(cursor + 1):(cursor + q_lv * K)]
    alpha_lv = reshape(alpha_vec, q_lv, K)
    cursor += q_lv * K
    θ_rr = @view params[(cursor + 1):(cursor + rr)]
    Λ = unpack_lambda(θ_rr, p, K)

    lv_offset = _lv_mean_eta(Λ, X_lv, alpha_lv)
    off = offset === nothing ? lv_offset : offset .+ lv_offset
    v = -binomial_marginal_loglik_laplace(Y, N, Λ, β, link;
                                          mask = mask, offset = off,
                                          maxiter = maxiter, tol = tol)
    return isfinite(loading_ridge) ? v + 0.5 * sum(abs2, Λ) / loading_ridge^2 : v
end

"""
    _laplace_saturation_health(Y, N, Λ, β, link, hessian; mask = nothing)

Post-fit Laplace saturation diagnostic. Recomputes each observed site's
conditional mode at the fitted `(β, Λ)` (the same solve `getLV` performs) and
counts, over observed cells only: (`n_clamp`) cells whose mode-η reaches the
link's μ-saturation thresholds — the η at which `linkinv` hits the `_clamp_mu`
bounds `1e-12` / `1 − 1e-12` — or the ±30 η-clamp itself; and (`n_wcollapse`)
cells whose log-det weight under `hessian` (the FIT's own selector) is
`≤ 1e-8`. Where either count is positive, the Laplace log-det penalty is
locally deleted and the reported loglik can overstate the exact marginal
without bound. Assessed for the dense Binomial kernel only; the grouped /
covariate / quadratic / mixed / SPDE / phylo / AGHQ / coevolution kernels and
the VA route (`variational_binomial.jl`) are out of scope and carry
`saturation = nothing`.
"""
function _laplace_saturation_health(Y::AbstractMatrix, N::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, link::Link, hessian::Symbol;
        mask = nothing)
    p, n = size(Y)
    η_hi = linkfun(link, 1 - 1e-12)
    η_lo = linkfun(link, 1e-12)
    fam = Binomial()
    n_clamp = 0; n_wcol = 0; n_obs = 0; maxeta = 0.0
    for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        z = _laplace_mode(fam, view(Y, :, s), view(N, :, s), Λ, β, link; mask = mi)
        for t in 1:p
            (mi === nothing || mi[t]) || continue
            n_obs += 1
            η = _clamp_eta(sum(Λ[t, k] * z[k] for k in 1:size(Λ, 2); init = β[t]))
            a = abs(η); a > maxeta && (maxeta = a)
            if η >= η_hi || η <= η_lo || a >= 30.0
                n_clamp += 1
            end
            μ = _clamp_mu(fam, linkinv(link, η))
            me = mu_eta(link, η)
            W = (hessian === :fisher || _glm_weight_matches_observed(fam, link)) ?
                _glm_weight(fam, μ, N[t, s], me) :
                _glm_obs_weight(fam, μ, N[t, s], me, Y[t, s], link, η)
            abs(W) <= 1e-8 && (n_wcol += 1)
        end
    end
    return LaplaceSaturationHealth(n_clamp, n_wcol, n_obs, maxeta, hessian)
end

# Emit the (per-fit, never rate-limited) saturation warning and return the
# health record; `nothing` in ⇒ `nothing` out (plateau verdicts stay silent).
function _warn_saturation(sat::Union{Nothing, LaplaceSaturationHealth}, link::Link, Λ)
    sat === nothing && return sat
    if sat.n_clamp > 0 || sat.n_wcollapse > 0
        @warn string("Binomial/", nameof(typeof(link)), " fit reached the Laplace ",
            "saturation region: ", sat.n_clamp, " of ", sat.n_obs, " cells at a ",
            "μ-saturation threshold, ", sat.n_wcollapse, " with collapsed log-det ",
            "weight (‖Λ̂‖ = ", round(sqrt(sum(abs2, Λ)); sigdigits = 3), " for context). ",
            "This MAY indicate the Laplace approximation is unreliable at this ",
            "optimum; loadings and loglik can be strongly inflated when the ",
            "saturation is latent-mode driven, while extreme-prevalence data can ",
            "saturate benignly through the intercepts. This warning includes the ",
            "documented cloglog pathology.")
    end
    return sat
end

# Runaway-loading screen (#498 item 1), mirroring gllvmTMB's binomial check
# (R/diagnose.R, `.gllvmTMB_max_loading_by_trait` + the O-rule thresholds): per
# trait m_t = max_k |Λ[t,k]|; relative_t = m_t / max(median(m), MAD(m)) over
# traits with m > 0 (MAD with constant 1). A trait is flagged when
# relative_t >= 25 (`loading_runaway_thresh`) or m_t >= 8 (`loading_absolute_thresh`,
# a link-scale latent SD this large is a fitted probability indistinguishable from
# 0/1). Diagnostic only: estimates and `converged` are untouched.
const _BINOMIAL_LOADING_RUNAWAY_THRESH = 25.0
const _BINOMIAL_LOADING_ABSOLUTE_THRESH = 8.0

function _binomial_runaway_loadings(Λ::AbstractMatrix)
    m = vec(maximum(abs, Λ; dims = 2))
    pos = filter(x -> isfinite(x) && x > 0, m)
    typical = isempty(pos) ? NaN : median(pos)
    spread = length(pos) > 1 ? median(abs.(pos .- median(pos))) : NaN
    cands = filter(x -> isfinite(x) && x > 0, [typical, spread])
    denom = isempty(cands) ? NaN : maximum(cands)
    return [t for t in eachindex(m) if isfinite(m[t]) &&
            ((isfinite(denom) && m[t] / denom >= _BINOMIAL_LOADING_RUNAWAY_THRESH) ||
             m[t] >= _BINOMIAL_LOADING_ABSOLUTE_THRESH)]
end

function _warn_runaway_loadings(Λ::AbstractMatrix)
    idx = _binomial_runaway_loadings(Λ)
    isempty(idx) && return idx
    @warn string("Binomial fit has runaway loadings on trait(s) ", join(idx, ", "),
        " (max |Λ| = ", round(maximum(abs, Λ); sigdigits = 3), "): a loading this far above ",
        "the rest, or this large on the link scale, is an improper solution (Heywood case); ",
        "quasi-complete separation produces this at ordinary prevalence. Treat the fit as ",
        "unusable rather than interpreting it; try loading_ridge or a lower K.")
    return idx
end

"""
    fit_binomial_gllvm(Y; K, link=LogitLink(), N=nothing, X_lv=nothing, …) -> BinomialFit

Fit a Binomial GLLVM by L-BFGS on the Laplace marginal log-likelihood
(`binomial_marginal_loglik_laplace`). `Y` is a p×n integer response
matrix (responses × sites); `N` the matching trial counts (default all-ones,
i.e. Bernoulli / binary). `K` is the latent dimension. Optimises the intercepts
`β` and loadings `Λ`.

`X_lv` (n×q_lv) activates the predictor-informed latent-score mean
`z_total[s, :] = X_lv[s, :] * alpha_lv + z_s`; this point-estimate route
estimates `alpha_lv` jointly with `β` and `Λ`. `LogitLink`, `ProbitLink`, and
`CLogLogLink` are all supported through the same Laplace core. Confidence
interval engines for this expanded parameter layout remain a separate gate.

The default analytic Laplace gradient is used on the logit no-offset path, with
an internal finite-difference fallback; non-logit links and offset fits use
finite differences (this is a `link isa LogitLink` gate, not a `hessian` gate —
a probit fit never reaches the logit-specific analytic gradient regardless of
`hessian`, so its coupling to the log-det curvature is moot). `X_lv` fits also
use finite differences because the offset depends jointly on `Λ` and
`alpha_lv`. Warm start: empirical link-scale intercepts + an SVD (PPCA-style)
loadings init; `alpha_lv` starts from a least-squares regression of the
initial latent scores on `X_lv`.

`hessian` selects the Laplace log-det curvature only (`:fisher` expected /
`:observed` joint — TMB's choice); the inner mode search is always
Fisher-scored. Default: canonical logit — the two coincide (`:fisher`).
Probit defaults to `:observed` (matching TMB/`gllvmTMB` parity).
Cloglog ALSO defaults to `:observed` (changed 2026-09-01, maintainer decisions
round 1 item 2 — a confirmed Julia-side likelihood-value defect against R,
not the 2026-08-28 optimizer-runaway pathology, which was measured under
BOTH curvature selectors and is unaffected by this default). Omitting the kwarg is exactly
the default-path behaviour for every link.

`loading_ridge` (default `Inf`, i.e. off) adds an opt-in ridge penalty on the
loadings to the optimised objective: `negll(θ) + 0.5 * sum(Λ.^2) / loading_ridge^2`
(same convention as gllvmTMB's `aghq_ridge`). The reported `loglik` is always the
UNPENALISED Laplace marginal log-likelihood evaluated at the (possibly
ridge-penalised) optimum — the same `logLik` semantics `aic`/`bic` read, matching
`gllvmTMB`. Maintainer decision (vault D-293): binary data get a loading-ridge
sweep when choosing the latent dimension `K`; a finite `loading_ridge` is the
opt-in knob that sweep drives. Not yet supported together with AGHQ
(`fit_binomial_gllvm(...; aghq = ...)` raises `ArgumentError` if both are
requested).
"""
function _fit_binomial_gllvm_laplace(Y::AbstractMatrix; K::Integer,
        link::Link = LogitLink(),
        N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing, mask = nothing,
        offset = nothing, gradient::Symbol = :analytic,
        hessian::Symbol = _default_hessian(Binomial(), link),
        β_init = nothing, Λ_init = nothing,
        X_lv::Union{Nothing, AbstractMatrix} = nothing,
        alpha_lv_init = nothing,
        loading_ridge::Real = Inf,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, mask = mask,
                               caller = "_fit_binomial_gllvm_laplace")
    K >= 0 || throw(ArgumentError("K must be non-negative for fit_binomial_gllvm"))
    hessian in (:fisher, :observed) || throw(ArgumentError(
        "fit_binomial_gllvm: hessian must be :fisher or :observed; got :$hessian"))
    loading_ridge > 0 || throw(ArgumentError(
        "fit_binomial_gllvm: loading_ridge must be positive (Inf disables the ridge); got $loading_ridge"))
    if X_lv !== nothing && hessian !== _default_hessian(Binomial(), link)
        throw(ArgumentError("fit_binomial_gllvm: a non-default hessian is not yet supported " *
                            "together with X_lv"))
    end
    Nm = N === nothing ? fill(1, p, n) : N
    size(Nm) == (p, n) || throw(DimensionMismatch("N must be $(p)×$(n)"))
    rr = rr_theta_len(p, K)

    q_lv = 0
    X_lv_fit = nothing
    if X_lv !== nothing
        K > 0 || throw(ArgumentError("X_lv requires a positive latent dimension K"))
        size(X_lv, 1) == n ||
            throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n_sites ($n)"))
        q_lv = size(X_lv, 2)
        q_lv > 0 || throw(ArgumentError("X_lv must have at least one predictor column"))
        X_lv_fit = Matrix{Float64}(X_lv)
        if alpha_lv_init !== nothing
            size(alpha_lv_init, 1) == q_lv ||
                throw(ArgumentError(
                    "alpha_lv_init first dim ($(size(alpha_lv_init, 1))) must equal size(X_lv, 2) ($q_lv)"))
            size(alpha_lv_init, 2) == K ||
                throw(ArgumentError(
                    "alpha_lv_init second dim ($(size(alpha_lv_init, 2))) must equal K ($K)"))
        end
    elseif alpha_lv_init !== nothing
        throw(ArgumentError("alpha_lv_init requires X_lv"))
    end

    msk = _resolve_obs_mask(mask, Y)                  # NA handling
    Yc  = Integer.(_sanitize_missing(Y, 0))

    # warm start: empirical link-scale intercepts + SVD (PPCA-like) loadings
    Zemp = [linkfun(link, clamp((Yc[t, i] + 0.5) / (Nm[t, i] + 1), 1e-4, 1 - 1e-4))
            for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)           # offset (η = β + offset + Λz)
    _mask_warmstart!(Zemp, msk)
    β0 = β_init === nothing ? vec(sum(Zemp; dims = 2)) ./ n : collect(float.(β_init))
    Zc = Zemp .- β0
    Λ0 = if Λ_init === nothing
        F = svd(Zc)
        kk = min(K, length(F.S))
        L = zeros(p, K)
        @inbounds for j in 1:kk
            L[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
        end
        L
    else
        collect(float.(Λ_init))
    end

    alpha0 = if X_lv_fit === nothing
        nothing
    elseif alpha_lv_init === nothing
        F = svd(Zc)
        kk = min(K, length(F.S))
        scores0 = zeros(Float64, n, K)
        @inbounds for j in 1:kk
            scores0[:, j] = sqrt(n) .* F.V[:, j]
        end
        X_lv_fit \ scores0
    else
        Matrix{Float64}(alpha_lv_init)
    end

    θ0 = vcat(β0, pack_lambda(Λ0))
    ridge_on = isfinite(loading_ridge)
    ridge_τ2 = loading_ridge^2
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        v = try
            -binomial_marginal_loglik_laplace(Yc, Nm, Λ, β, link; mask = msk, offset = offset,
                                              hessian = hessian,
                                              maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        isfinite(v) || return 1e12
        ridge_on || return v
        return v + 0.5 * sum(abs2, Λ) / ridge_τ2
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = if X_lv_fit !== nothing
        θ0_lv = vcat(β0, vec(alpha0), pack_lambda(Λ0))
        negll_lv = θ -> begin
            v = try
                binomial_lv_nll_packed(θ, Yc, Nm, p, K, link;
                                       X_lv = X_lv_fit, q_lv = q_lv,
                                       mask = msk, offset = offset,
                                       loading_ridge = loading_ridge,
                                       maxiter = newton_maxiter, tol = newton_tol)
            catch
                return 1e12
            end
            return isfinite(v) ? v : 1e12
        end
        Optim.optimize(negll_lv, θ0_lv, ls, opts; autodiff = :finite)
    elseif gradient === :analytic && offset === nothing && link isa LogitLink &&
           (hessian === _default_hessian(Binomial(), link) ||
            _glm_weight_matches_observed(Binomial(), link))
        # `link isa LogitLink` here is load-bearing for the 2026-08-28 probit
        # flip too: `binomial_laplace_grad` (laplace_grad.jl) is a logit-only
        # analytic gradient (its log-det weight is the LOGIT Fisher/observed
        # identity, not link-generic), so a probit fit — under EITHER
        # `hessian` selector — never reaches this branch and always falls
        # through to the finite-difference path below. That is the "route
        # probit to FD with a measured note" resolution: no analytic-gradient
        # coupling was needed because probit was never coupled to begin with.
        ag = θ -> begin
            β = θ[1:p]; Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
            try
                g = -binomial_laplace_grad(Yc, Nm, Λ, β; mask = msk)
                if ridge_on
                    # d/dθ [0.5 * sum(Λ.^2) / τ²] restricted to the Λ block:
                    # linear in each packed entry, so it's `pack_lambda(Λ) / τ²`.
                    g[(p + 1):(p + rr)] .+= pack_lambda(Λ) ./ ridge_τ2
                end
                g
            catch
                nothing
            end
        end
        _optimize_with_analytic(negll, ag, θ0, ls, opts)
    else
        Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    end
    θ̂ = Optim.minimizer(res)
    if X_lv_fit !== nothing
        cursor = 0
        β̂ = collect(θ̂[(cursor + 1):(cursor + p)])
        cursor += p
        alpha_hat = reshape(collect(θ̂[(cursor + 1):(cursor + q_lv * K)]), q_lv, K)
        cursor += q_lv * K
        Λ̂ = unpack_lambda(@view(θ̂[(cursor + 1):(cursor + rr)]), p, K)
        ll, conv, iters = _fit_verdict(res)
        # `_fit_verdict`'s `ll` is `-Optim.minimum(res)`, the (possibly ridge-)
        # PENALISED objective at θ̂. The fit's `loglik` must be the unpenalised
        # Laplace marginal (gllvmTMB's `logLik` semantics) — recompute it at the
        # same θ̂ via the same `binomial_lv_nll_packed` route with the ridge off.
        if ridge_on && isfinite(ll)
            ll = try
                -binomial_lv_nll_packed(collect(Float64, θ̂), Yc, Nm, p, K, link;
                                        X_lv = X_lv_fit, q_lv = q_lv,
                                        mask = msk, offset = offset, loading_ridge = Inf,
                                        maxiter = newton_maxiter, tol = newton_tol)
            catch
                -Inf
            end
        end
        sat = isfinite(ll) ?
            _warn_saturation(_laplace_saturation_health(Yc, Nm, Λ̂, β̂, link, hessian;
                                                        mask = msk), link, Λ̂) : nothing
        _warn_runaway_loadings(Λ̂)
        return BinomialFit(β̂, Λ̂, link, ll, conv, iters,
                           alpha_hat, collect(Float64, θ̂), hessian, sat, nothing, loading_ridge)
    else
        β̂ = θ̂[1:p]
        Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
        ll, conv, iters = _fit_verdict(res)
        # Same unpenalised-loglik correction as above, non-X_lv route.
        if ridge_on && isfinite(ll)
            ll = try
                binomial_marginal_loglik_laplace(Yc, Nm, Λ̂, β̂, link; mask = msk, offset = offset,
                                                 hessian = hessian,
                                                 maxiter = newton_maxiter, tol = newton_tol)
            catch
                -Inf
            end
        end
        sat = isfinite(ll) ?
            _warn_saturation(_laplace_saturation_health(Yc, Nm, Λ̂, β̂, link, hessian;
                                                        mask = msk), link, Λ̂) : nothing
        _warn_runaway_loadings(Λ̂)
        return BinomialFit(β̂, Λ̂, link, ll, conv, iters, nothing, Float64[], hessian, sat, nothing, loading_ridge)
    end
end
