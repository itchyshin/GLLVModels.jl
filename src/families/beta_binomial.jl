# Beta-binomial family (gllvm family="beta.binomial", twin fid 8) for the Laplace path.
#
# Overdispersed binomial: y | N, p ~ Binomial(N, p) with p ~ Beta(a, b), so the
# trial-success probability itself is random. gllvm parameterises (see
# JenniNiku/gllvm src/gllvm.cpp:5252-5267) the Beta shapes as
#
#     a = α = μ·φ,   b = β = (1−μ)·φ,   φ = exp(lg_phi) > 0,
#
# where μ = linkinv(link, η) ∈ (0,1) is the success prob and φ = a+b is the Beta
# precision (the shape-sum). The marginal beta-binomial log-pmf is
#
#   log p(y|N,μ,φ) = lgamma(a+b) + lgamma(a+y) + lgamma(b+N−y) − lgamma(a)
#                    − lgamma(b) − lgamma(a+b+N) + lgamma(N+1) − lgamma(y+1)
#                    − lgamma(N−y+1),
#
# with E[y] = N·μ, Var[y] = N·μ(1−μ)·(1 + (N−1)·φ/(φ+1)), intraclass ρ = 1/(φ+1).
# As φ → ∞ the Beta collapses to a point mass at μ and the family → Binomial(N, μ)
# (var-inflation → 1) — the key reduction anchor.
#
# A single latent η drives μ; the family marker carries only the dispersion φ.
# This file therefore runs its OWN per-site Laplace (mirroring ordered_beta.jl):
# the per-trait score s_t = ∂log p/∂η and weight W_t = −∂²log p/∂η² are obtained
# by ForwardDiff on the scalar map η → log p (lower risk than the digamma score /
# Hessian), with W_t clamped to ≥ 1e-8 for SPD. The trial counts N are threaded
# through the marginal and the fit exactly like families/binomial.jl threads them.

"""
    BetaBinom(φ = 1.0)

Beta-binomial family marker (gllvm `family="beta.binomial"`, enum 15). `φ > 0` is
the Beta precision (the shape-sum `a+b`, i.e. the species dispersion). Named to
avoid colliding with `Distributions.BetaBinomial` — with `using GLLVModels,
Distributions` both names are in scope, and `BetaBinom` is the GLLVModels marker.
Pass it to [`fit_gllvm`](@ref) or [`gllvm`](@ref):

```julia
fit_gllvm(Y; family = BetaBinom(), K = 2, N = trials)   # per-trait φ → BetaBinomialGroupedFit
```

The trial counts travel as the `N` keyword (a p×n matrix), **not** on the marker,
and `N` is **required** on those two routes: at `N = 1` the beta-binomial collapses
to `Bernoulli(μ)` and `φ` is unidentifiable, so the named fitters' silent
all-ones default is not safe at a public boundary.

The `φ` field is a **tag payload only**: no public route reads it. `φ` is always
estimated — per trait by default, matching gllvmTMB's length-`p`
`log_phi_betabinom` — and is never used as a starting value. To seed it, pass
`φ_init` to a named fitter ([`fit_beta_binomial_gllvm`](@ref),
[`fit_beta_binomial_gllvm_grouped`](@ref)). Internally the Laplace kernels
construct their own per-iteration `BetaBinom(φ)` markers, which is what the field
is for.
"""
struct BetaBinom <: Distribution{Univariate, Discrete}
    φ::Float64
end

BetaBinom() = BetaBinom(1.0)

# logistic σ(x), numerically safe at large |x| (mirrors ordered_beta.jl).
_bb_logistic(x) = x ≥ 0 ? inv(one(x) + exp(-x)) : (e = exp(x); e / (one(x) + e))

const _BB_MU_LO = 1e-12
const _BB_MU_HI = 1 - 1e-12

# Beyond this precision the direct loggamma formula below is numerically wrong,
# not just imprecise (#515). a=μφ and b=(1−μ)φ both grow like φ, so each of the
# formula's six loggamma terms grows like φ·log(φ) while their SUM (the actual
# log-pmf) stays O(1) — a huge cancellation. Float64's ~2.2e-16 relative
# rounding in each term becomes an ABSOLUTE error in the sum of order
# φ·log(φ)·2.2e-16: negligible at φ=1e6 (≈2e-10, measured against the same
# formula in 256-bit BigFloat during the #515 diagnosis) but ~1e-3 by
# φ=1e12 and unbounded from there — a genuine beta-binomial log-pmf reads back
# as a huge, wrong, but still-finite number, silently. That is the mechanism
# behind #515's reported `loglik ≈ +7.18e54` at `φ ≈ 3.3e65`: the outer L-BFGS
# search (finite-difference gradient) reads the cancellation noise as room to
# improve and runs φ further into it. Beyond this threshold the Beta(μφ,
# (1−μ)φ) has already collapsed to a point mass at μ (file header), so the
# Binomial(N, μ) log-pmf IS the φ→∞ limit the direct formula is trying (and,
# above this threshold, numerically failing) to compute. Matches the boundary
# `_dispersion_group_boundary` (`grouped_dispersion.jl`) already uses for
# NB1/NB2 group dispersion, and is far above any φ a healthy fit reaches (this
# family's own tests fit φ_true=12), so this never perturbs a healthy fit.
const _BB_PHI_STABLE = 1e6

"""
    betabinomial_logp(y, η, N, φ; link=LogitLink()) -> Float64

Scalar beta-binomial conditional log-pmf log p(y|N,η,φ) for one trait, in the
gllvm parameterisation `a = μφ`, `b = (1−μ)φ` with `μ = linkinv(link, η)` clamped
to (1e-12, 1−1e-12). Uses `loggamma` (from SpecialFunctions, imported module-wide).
At `φ ≥ $(_BB_PHI_STABLE)` returns the Binomial(N, μ) log-pmf instead (the exact
φ→∞ limit, see `_BB_PHI_STABLE` above) — the direct formula's own cancellation
error is unbounded past that point (#515).
"""
function betabinomial_logp(y, η, N, φ; link::Link = LogitLink())
    μ = clamp(linkinv(link, η), _BB_MU_LO, _BB_MU_HI)
    φ >= _BB_PHI_STABLE &&
        return loggamma(N + 1) - loggamma(y + 1) - loggamma(N - y + 1) +
               y * log(μ) + (N - y) * log(one(μ) - μ)
    a = μ * φ
    b = (one(μ) - μ) * φ
    return loggamma(a + b) + loggamma(a + y) + loggamma(b + N - y) -
           loggamma(a) - loggamma(b) - loggamma(a + b + N) +
           loggamma(N + 1) - loggamma(y + 1) - loggamma(N - y + 1)
end

# Per-trait score s_t = ∂log p/∂η and weight W_t = −∂²log p/∂η², via ForwardDiff
# on the scalar map η → log p. W clamped to ≥ 1e-8 to keep Λ'WΛ + I SPD.
function _bb_score_weight(y, η, N, φ; link::Link = LogitLink())
    f = ηv -> betabinomial_logp(y, ηv, N, φ; link = link)
    g = ηv -> ForwardDiff.derivative(f, ηv)
    s = g(η)
    W = -ForwardDiff.derivative(g, η)
    return s, max(W, 1e-8)
end

# φ can be shared (scalar, existing API) or per-trait (length-p vector, the
# grouped/X extension); this tiny accessor keeps both call sites branch-free.
@inline _bb_phi_at(φ::Real, ::Integer) = φ
@inline _bb_phi_at(φ::AbstractVector, t::Integer) = φ[t]

# Per-site log-posterior q(z) = Σ betabinomial_logp(...) − ½z'z, masked and offset
# exactly as `_beta_binomial_mode_search` below. Used only by that search's step-
# halving accept/reject test (mirrors `_grouped_laplace_mode_logpost` in
# `grouped_dispersion.jl`).
function _bb_mode_logpost(y::AbstractVector, N::AbstractVector, Λ::AbstractMatrix,
        β::AbstractVector, φ::Union{Real, AbstractVector}, link::Link, z::AbstractVector;
        mask = nothing, offset::Union{Nothing, AbstractVector} = nothing)
    p = size(Λ, 1)
    off = offset === nothing ? false : offset
    η = β .+ off .+ Λ * z
    q = -0.5 * dot(z, z)
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        q += betabinomial_logp(y[t], η[t], N[t], _bb_phi_at(φ, t); link = link)
    end
    return q
end

# Inner Laplace mode-finder for one site (#503, the #479/#500/#507/#509 damped-
# search pattern). Returns `(z, converged)`. Mirrors `_ordered_beta_mode` in its
# score/weight but was, before this fix, an UNDAMPED Newton loop on the clamped
# observed curvature (`_bb_score_weight` floors `W` to `≥ 1e-8`, which keeps
# `Λ'WΛ + I` SPD by construction — same role as the expected-information floor
# in the Gamma/NB1/NB2/Student-t grouped kernels — but does not keep any step a
# DESCENT step) that returned whatever `z` it held at `maxiter`, converged or
# not. The class audit (Λ scaled up to 3x, warm start perturbed ±50%) measured
# 27/500 sites where the returned `z` was not stationary while a from-scratch
# restart converged cleanly, some of them finite (able to escape the fitter's
# 1e12 sentinel).
#
# Now a step that lowers the per-site log-posterior `q` is halved (the generic
# core's rule, so small steps and accepted full steps are bit-identical to the
# old loop), and `converged` requires the FULL proposed step to be below `tol`
# (the old loop's own stopping test) — a heavily halved step does not count.
# `mask` (length-p Bool, or `nothing` = all observed) drops missing responses:
# a masked entry contributes zero score and zero weight, so it neither pulls
# the mode nor enters the Hessian. `φ` is either a shared scalar or a length-p
# per-trait vector (grouped/+X extension). `offset` (length-p, or `nothing`) is
# an optional site-covariate contribution `(Xγ)[:, i]` added to the linear
# predictor before the link.
function _beta_binomial_mode_search(y::AbstractVector, N::AbstractVector, Λ::AbstractMatrix,
        β::AbstractVector, φ::Union{Real, AbstractVector}; link::Link = LogitLink(),
        mask = nothing, offset::Union{Nothing, AbstractVector} = nothing,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    z = zeros(K)
    for _ in 1:maxiter
        η = β .+ off .+ Λ * z
        s = Vector{Float64}(undef, p)
        W = Vector{Float64}(undef, p)
        @inbounds for t in 1:p
            if mask !== nothing && !mask[t]
                s[t] = 0.0; W[t] = 0.0           # masked ⇒ no contribution
                continue
            end
            st, Wt = _bb_score_weight(y[t], η[t], N[t], _bb_phi_at(φ, t); link = link)
            s[t] = st
            W[t] = Wt
        end
        A = Symmetric(Λ' * (W .* Λ) + I)
        Δ = _safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && return z, false
        maximum(abs, Δ) < tol && return z .+ Δ, true
        if norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        else
            q0 = _bb_mode_logpost(y, N, Λ, β, φ, link, z; mask = mask, offset = offset)
            if isfinite(q0)
                accepted = false
                step = 1.0
                for _half in 1:30
                    ztrial = z .+ step .* Δ
                    q1 = _bb_mode_logpost(y, N, Λ, β, φ, link, ztrial; mask = mask, offset = offset)
                    if isfinite(q1) && q1 >= q0
                        z = ztrial
                        accepted = true
                        break
                    end
                    step *= 0.5
                end
                accepted || return z, false
            else
                z = z .+ Δ
            end
        end
    end
    return z, false
end

# Public per-site mode search, retained under its original name for `getLV`/
# `predict` (which need only `z`, at the fitted parameters, not a convergence
# flag). Retries with a 20x iteration budget before giving up (mirrors #507's
# review fix for NB1 and #509's Student-t fallback: a genuinely healthy site
# can still need more than the default `maxiter = 100` damped steps under
# ill-conditioned curvature). `_beta_binomial_loglik_site` below calls
# `_beta_binomial_mode_search` directly so it can act on the `converged` flag.
function _beta_binomial_mode(y::AbstractVector, N::AbstractVector, Λ::AbstractMatrix,
        β::AbstractVector, φ::Union{Real, AbstractVector}; link::Link = LogitLink(),
        mask = nothing, offset::Union{Nothing, AbstractVector} = nothing,
        maxiter::Integer = 100, tol::Real = 1e-9)
    z, ok = _beta_binomial_mode_search(y, N, Λ, β, φ; link = link, mask = mask,
                                       offset = offset, maxiter = maxiter, tol = tol)
    ok || ((z, ok) = _beta_binomial_mode_search(y, N, Λ, β, φ; link = link, mask = mask,
                                                offset = offset, maxiter = 20 * maxiter, tol = tol))
    return z
end

# Per-site Laplace log-marginal:
#   log p(y_s) ≈ ℓ(ẑ) − ½ẑ'ẑ − ½logdet(Λ'WΛ + I).
# `mask` drops the masked entries from the score/weight (via
# `_beta_binomial_mode_search`) and from the conditional log-density sum.
# `φ`/`offset` as in `_beta_binomial_mode_search`.
#
# Calls the search directly (not the public `_beta_binomial_mode` wrapper) so
# it can act on the `converged` flag (#503): a search that cannot certify a
# stationary point — even after the wrapper's own 20x-budget retry — must not
# produce a finite value. -Inf makes the fitters' objective return its 1e12
# sentinel instead of a garbage surface (the #479 precedent).
function _beta_binomial_loglik_site(y::AbstractVector, N::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, φ::Union{Real, AbstractVector};
        link::Link = LogitLink(), mask = nothing, offset::Union{Nothing, AbstractVector} = nothing,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    off = offset === nothing ? false : offset
    z, ok = _beta_binomial_mode_search(y, N, Λ, β, φ; link = link, mask = mask, offset = offset,
                                       maxiter = maxiter, tol = tol)
    ok || ((z, ok) = _beta_binomial_mode_search(y, N, Λ, β, φ; link = link, mask = mask,
                                                offset = offset, maxiter = 20 * maxiter, tol = tol))
    ok || return -Inf
    η = β .+ off .+ Λ * z
    ℓ = 0.0
    W = Vector{Float64}(undef, p)
    @inbounds for t in 1:p
        if mask !== nothing && !mask[t]
            W[t] = 0.0                           # masked ⇒ no Hessian weight, no logpdf
            continue
        end
        φt = _bb_phi_at(φ, t)
        ℓ += betabinomial_logp(y[t], η[t], N[t], φt; link = link)
        _, Wt = _bb_score_weight(y[t], η[t], N[t], φt; link = link)
        W[t] = Wt
    end
    A = Symmetric(Λ' * (W .* Λ) + I)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

"""
    betabinomial_marginal_loglik_laplace(Y, N, Λ, β, φ; mask=nothing, link=LogitLink(),
                                         offset=nothing, maxiter=100, tol=1e-9) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a beta-binomial GLLVM.
`Y` is a p×n matrix of integer successes; `N` the matching p×n trial counts; `Λ`
p×K loadings; `β` length-p intercepts; `φ` the Beta precision (shape-sum). Runs
its own per-site Laplace (single latent η, gllvm parameterisation `a=μφ, b=(1−μ)φ`,
`μ = linkinv(link, η)`). At `Λ = 0` this reduces exactly to the sum of the
independent beta-binomial `logp`. As `φ → ∞` it approaches the Binomial marginal.

`mask` (p×n Bool, or `nothing`) marks observed cells — masked (missing) responses
are dropped per site from the score, the Hessian weight, and the log-density sum,
so the marginal is over the observed entries only (invariant to the masked-cell
placeholder). `offset` (`nothing`, or a p×n matrix such as `Xγ`) is added to the
linear predictor before the link — with a constant per-trait `φvec = fill(φ, p)`
and the same `offset`, this equals
`betabinomial_grouped_marginal_loglik_laplace` to machine precision.
"""
function betabinomial_marginal_loglik_laplace(Y::AbstractMatrix, N::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, φ::Real; mask = nothing, link::Link = LogitLink(),
        offset::Union{Nothing, AbstractMatrix} = nothing,
        maxiter::Integer = 100, tol::Real = 1e-9)
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        acc += _beta_binomial_loglik_site(view(Y, :, i), view(N, :, i), Λ, β, φ;
                                          link = link, mask = mi, offset = oi,
                                          maxiter = maxiter, tol = tol)
    end
    return acc
end

# ---------------------------------------------------------------------------
# Grouped / per-trait Beta precision φ (gllvm's `disp.group`, twin default
# under X — decision 2026-08-05). Reuses THIS file's own `_beta_binomial_mode` /
# `_beta_binomial_loglik_site` (now φ-vector- and offset-aware) rather than the
# generic grouped-dispersion machinery in `grouped_dispersion.jl` (that module's
# `fams::AbstractVector` dispatch expects the shared `_glm_score`/`_glm_weight`
# family interface, which beta-binomial does not implement — it runs its own
# ForwardDiff-based per-site Laplace instead).
# ---------------------------------------------------------------------------

"""
    betabinomial_grouped_marginal_loglik_laplace(Y, N, Λ, β, φvec; mask=nothing,
                                                 link=LogitLink(), offset=nothing,
                                                 maxiter=100, tol=1e-9) -> Float64

Total Laplace log-marginal of a beta-binomial GLLVM with **per-trait** Beta
precision `φvec` (length p; gllvm's `disp.group`, twin `log_phi_betabinom`).
`Y` is the p×n matrix of integer successes; `N` the matching p×n trial counts;
`Λ` p×K; `β` length-p. `offset` (`nothing`, or a p×n matrix such as `Xγ`) is
added to the linear predictor before the link. With a constant
`φvec = fill(φ, p)` and the same `offset` this equals
`betabinomial_marginal_loglik_laplace` to machine precision.
"""
function betabinomial_grouped_marginal_loglik_laplace(Y::AbstractMatrix, N::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, φvec::AbstractVector; mask = nothing,
        link::Link = LogitLink(), offset::Union{Nothing, AbstractMatrix} = nothing,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p = size(Λ, 1)
    length(φvec) == p || throw(ArgumentError("length(φvec)=$(length(φvec)) must equal p=$p"))
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        acc += _beta_binomial_loglik_site(view(Y, :, i), view(N, :, i), Λ, β, φvec;
                                          link = link, mask = mi, offset = oi,
                                          maxiter = maxiter, tol = tol)
    end
    return acc
end

# ---------------------------------------------------------------------------
# Fit driver.
# ---------------------------------------------------------------------------

# Same sentinel this file's own `negll` closures already return on a failed
# evaluation (`catch; return 1e12`, `isfinite(v) ? v : 1e12`); a legitimate
# evaluation never lands there.
const _BB_FAIL_PENALTY = 1e12

# The Laplace log-marginal `betabinomial_marginal_loglik_laplace` sums, per
# site, a discrete log-probability (≤0, a beta-binomial pmf value is never
# above 1) plus the Laplace correction `−½ẑ'ẑ − ½logdet(Λ'WΛ + I)`, itself
# ≤0 since `Λ'WΛ + I` is positive definite by construction (logdet ≥ 0). So a
# genuine value never rises meaningfully above 0; the small positive margin
# here is floating-point headroom, not a modelling tolerance.
const _BB_LOGLIK_MAX = 1e-6

"""
    _beta_binomial_verdict(optim_converged, nll, φ) -> (converged, loglik, reason)

Convergence contract for [`fit_beta_binomial_gllvm`](@ref) (#515, per the #502/
#505 rule: a per-family verdict rather than a change to the shared
`_fit_verdict`). `Optim`'s own flag is not enough here: it can fire at a point
where the Beta precision `φ` has run to the numerical edge (the file header's
φ→∞ reduction to Binomial) with intercepts and loadings correspondingly large
— `φ ≈ 3.3e65`, reported loglik ≈ +7.18e54, every per-site latent score
trivially at `z=0` (the #515 reproduction in `test/fixtures/`). Two
checks gate the reported flag beyond `optim_converged`; an impossible value
is reported as `loglik = -Inf`, never as a log-likelihood:

- `:objective_impossible` — the objective is non-finite, still at
  `_BB_FAIL_PENALTY` (never evaluated), or the resulting log-likelihood is
  positive beyond floating-point rounding (`> _BB_LOGLIK_MAX`; see that
  constant's own note for why a genuine value cannot be). +7e54 is not a large
  log-likelihood — it is `betabinomial_logp`'s catastrophic cancellation at
  extreme φ (see `_BB_PHI_STABLE`) read back as data.
- `:phi_at_boundary` — `φ ≥ _BB_PHI_STABLE`, the same boundary
  `_dispersion_group_boundary` (`grouped_dispersion.jl`) already uses for
  NB1/NB2 group dispersion: at or beyond it the Beta has collapsed to a point
  mass at μ and φ is not identifiable from a Binomial fit, whatever `Optim`
  reports. This never fires for a healthy fit — this family's own tests fit
  φ_true=12, orders of magnitude below the boundary.
"""
function _beta_binomial_verdict(optim_converged::Bool, nll::Real, φ::Real)
    (isfinite(nll) && nll < _BB_FAIL_PENALTY) || return (false, -Inf, :objective_impossible)
    ll = -Float64(nll)
    ll <= _BB_LOGLIK_MAX || return (false, -Inf, :objective_impossible)
    φ < _BB_PHI_STABLE || return (false, ll, :phi_at_boundary)
    return (optim_converged, ll, :ok)
end

"""
    _beta_binomial_grouped_verdict(nll, optim_converged, iterations, φg) -> (loglik, converged, iterations)

Convergence contract for [`fit_beta_binomial_gllvm_grouped`](@ref) and
[`fit_beta_binomial_gllvm_grouped_cov`](@ref) (#515 follow-up). The shared
`_fit_verdict` screen runs first, so its plateau threshold still applies; its
result then goes through [`_beta_binomial_verdict`](@ref) at the largest group
precision `maximum(φg)`. One group at `φ ≥ _BB_PHI_STABLE` is enough: that
group's Beta has collapsed to a point mass and its `φ` is not identifiable.
Measured on origin/main 52ed4281b (Julia 1.10.12): a per-species grouped fit
of genuine beta-binomial data (φ_true = 12, fixture `healthy_seed_9003`)
reported `converged = true` with one species at `φ ≈ 7.6e15`. Returns the
triple in `_fit_verdict`'s order, ready for the result constructors.
"""
function _beta_binomial_grouped_verdict(nll::Real, optim_converged::Bool,
        iterations::Integer, φg::AbstractVector{<:Real})
    ll0, conv0, iters = _fit_verdict(nll, optim_converged, iterations)
    conv, ll, _reason = _beta_binomial_verdict(conv0, -ll0, maximum(φg))
    return (ll, conv, iters)
end

"""
    BetaBinomialFit

Result of `fit_beta_binomial_gllvm`: intercepts `β` (length p), loadings
`Λ` (p×K), the `link`, the Beta precision `φ`, the maximised Laplace `loglik`, the
optimiser `converged` flag, and `iterations`.
"""
struct BetaBinomialFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    link::Link
    φ::Float64
    loglik::Float64
    converged::Bool
    iterations::Int
end

# ---------------------------------------------------------------------------
# Post-fit ordination: getLV / predict. A single latent η drives μ (the family
# marker carries φ), so the per-site mode is this file's own `_beta_binomial_mode`,
# and :mean returns the success probability μ = linkinv(link, η).
# ---------------------------------------------------------------------------

_loadings(fit::BetaBinomialFit) = fit.Λ
_loglik(fit::BetaBinomialFit)   = fit.loglik

# Free params: β (p) + reduced loadings Λ + Beta precision φ.
function _nparams(fit::BetaBinomialFit)
    p, K = size(fit.Λ)
    return p + (p * K - div(K * (K - 1), 2)) + 1       # β + Λ + φ
end

"""
    getLV(fit::BetaBinomialFit, Y; N=nothing, rotate=true) -> n×K matrix

Conditional latent-variable scores for a beta-binomial fit: the per-site Laplace
mode `ẑₛ` (`_beta_binomial_mode`) at the fitted `(Λ, β)`, link, and precision `φ`.
`Y` is the `p×n` matrix of integer successes; `N` the matching trial counts
(default all-ones); `rotate=true` applies the canonical SVD rotation of loadings.
"""
function getLV(fit::BetaBinomialFit, Y::AbstractMatrix{<:Real};
        N::Union{Nothing, AbstractMatrix{<:Real}} = nothing, rotate::Bool = true)
    p, n = size(Y)
    K = size(fit.Λ, 2)
    Nm = N === nothing ? fill(1, p, n) : N
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        Z[:, s] = _beta_binomial_mode(view(Y, :, s), view(Nm, :, s),
                                      fit.Λ, fit.β, fit.φ; link = fit.link)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::BetaBinomialFit, Y; N=nothing, type=:mean) -> p×n matrix

In-sample fitted values at the Laplace mode `ẑ` (see `getLV`): `type=:link`
returns the linear predictor `η = β + Λ ẑ`; `type=:mean` returns the success
probability `μ = linkinv(link, η)` (η clamped). Note `:mean` is the per-trial
success probability, not the count mean `E[y] = N·μ`.
"""
function predict(fit::BetaBinomialFit, Y::AbstractMatrix{<:Real};
        N::Union{Nothing, AbstractMatrix{<:Real}} = nothing, type::Symbol = :mean)
    type in (:link, :mean) ||
        throw(ArgumentError("type must be :link or :mean; got :$type"))
    Z = getLV(fit, Y; N = N, rotate = false)          # n×K
    η = fit.β .+ fit.Λ * Z'                            # p×n
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

function Base.show(io::IO, f::BetaBinomialFit)
    p, K = size(f.Λ)
    print(io, "BetaBinomialFit(p=", p, ", K=", K, ", link=", nameof(typeof(f.link)),
          ", φ=", round(f.φ; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_beta_binomial_gllvm(Y; K, N=nothing, link=LogitLink(), φ_init=nothing, …) -> BetaBinomialFit

Fit a beta-binomial GLLVM by L-BFGS on the Laplace marginal
(`betabinomial_marginal_loglik_laplace`), jointly estimating the Beta precision
`φ` (gllvm parameterisation `a=μφ, b=(1−μ)φ`). `Y` is a p×n matrix of integer
successes; `N` the matching trial counts (default all-ones, i.e. Bernoulli-
overdispersed); `K` the latent dimension. The optimiser θ = `[β(p); pack_lambda(Λ)(rr); log φ]`.
Finite-difference gradient (the Laplace inner mode-finder is not forward-AD-friendly).
Warm start = empirical link-mean intercepts (logit of `(y+0.5)/(N+1)` row means) +
an SVD (PPCA-style) loadings init + a moderate `φ₀`, mirroring `fit_binomial_gllvm`
and `fit_ordered_beta_gllvm`.

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or simply include
`missing` entries in `Y` — either way the masked cells are dropped from the
marginal *and* from the warm start, so the fit depends only on the observed cells
(it is invariant to whatever sits in the masked positions).
"""
function fit_beta_binomial_gllvm(Y::AbstractMatrix; K::Integer,
        N::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        link::Link = LogitLink(), mask = nothing,
        β_init = nothing, Λ_init = nothing, φ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    size(Nm) == (p, n) || throw(DimensionMismatch("N must be $(p)×$(n)"))
    rr = rr_theta_len(p, K)

    # NA handling: derive the observation mask (explicit `mask`, else from `missing`)
    # and a sanitized success matrix with a safe placeholder (0) in masked cells.
    msk = _resolve_obs_mask(mask, Y)
    Yc = Integer.(_sanitize_missing(Y, 0))

    # warm start: empirical link-scale intercepts + SVD (PPCA-like) loadings.
    Zemp = [linkfun(link, clamp((float(Yc[t, i]) + 0.5) / (float(Nm[t, i]) + 1),
                                1e-4, 1 - 1e-4)) for t in 1:p, i in 1:n]
    _mask_warmstart!(Zemp, msk)
    β0 = β_init === nothing ? vec(sum(Zemp; dims = 2)) ./ n : collect(float.(β_init))
    Λ0 = if Λ_init === nothing
        Zc = Zemp .- β0
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
    logφ0 = φ_init === nothing ? log(10.0) : log(float(φ_init))

    θ0 = vcat(β0, pack_lambda(Λ0), logφ0)
    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        φ = exp(θ[p + rr + 1])
        v = try
            -betabinomial_marginal_loglik_laplace(Yc, Nm, Λ, β, φ; mask = msk, link = link,
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
    φ̂ = exp(θ̂[p + rr + 1])
    conv, loglik, _reason = _beta_binomial_verdict(Optim.converged(res), Optim.minimum(res), φ̂)
    return BetaBinomialFit(β̂, Λ̂, link, φ̂, loglik, conv, Optim.iterations(res))
end

# ===========================================================================
# Grouped-dispersion fit driver — per-trait Beta precision φ, no site-X.
# Mirrors fit_nb1_gllvm_grouped's packing convention: θ = [β; pack(Λ); log φ_1
# … log φ_G]. With G=1 this matches fit_beta_binomial_gllvm exactly (same
# `_beta_binomial_loglik_site` under the hood).
# ===========================================================================

"""
    BetaBinomialGroupedFit

Result of `fit_beta_binomial_gllvm_grouped`: intercepts `β` (length p),
loadings `Λ` (p×K), the per-group Beta precision vector `φ` (length G), the
species→group map `group` (length p), the `link`, the maximised Laplace
`loglik`, `converged`, and `iterations`. The per-species precision is
`φ[group[t]]`.
"""
struct BetaBinomialGroupedFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    φ::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used —
                       # ALWAYS :fisher: this file's per-site Laplace has no
                       # analytic-Hessian variant (ForwardDiff-scored weight only;
                       # G0 lock), unlike the NB2/Beta/Gamma/NB1 grouped siblings.
end

# Positional compatibility constructor (2026-08-28): see NBGroupedFit
# (families/grouped_dispersion.jl) above. Fixed at `:fisher` — this route has
# no `hessian` kwarg to record.
BetaBinomialGroupedFit(β, Λ, φ, group, link, loglik, converged, iterations) =
    BetaBinomialGroupedFit(β, Λ, φ, group, link, loglik, converged, iterations, :fisher)

function Base.show(io::IO, f::BetaBinomialGroupedFit)
    p, K = size(f.Λ)
    print(io, "BetaBinomialGroupedFit(p=", p, ", K=", K, ", G=", length(f.φ),
          ", φ=", round.(f.φ; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::BetaBinomialGroupedFit) = fit.Λ
_loglik(fit::BetaBinomialGroupedFit)   = fit.loglik

# Free params: β (p) + reduced loadings Λ + one precision per group (G).
function _nparams(fit::BetaBinomialGroupedFit)
    p, K = size(fit.Λ)
    return p + rr_theta_len(p, K) + length(fit.φ)      # β + Λ + G precisions φ
end

"""
    getLV(fit::BetaBinomialGroupedFit, Y; N=nothing, rotate=true, mask=nothing) -> n×K matrix

Conditional latent-variable scores for a grouped-precision beta-binomial fit,
using the per-trait Beta precision `φ[group[t]]` in the same Laplace mode
equations as `betabinomial_grouped_marginal_loglik_laplace`.
"""
function getLV(fit::BetaBinomialGroupedFit, Y::AbstractMatrix{<:Real};
        N::Union{Nothing, AbstractMatrix{<:Real}} = nothing, rotate::Bool = true, mask = nothing)
    p, n = size(Y)
    K = size(fit.Λ, 2)
    Nm = N === nothing ? fill(1, p, n) : N
    φvec = [fit.φ[fit.group[t]] for t in 1:p]
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        Z[:, s] = _beta_binomial_mode(view(Y, :, s), view(Nm, :, s), fit.Λ, fit.β, φvec;
                                      link = fit.link, mask = mi)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::BetaBinomialGroupedFit, Y; N=nothing, type=:mean) -> p×n matrix

In-sample fitted values at the Laplace mode `ẑ` (see `getLV`):
`type=:link` returns `η = β + Λẑ`; `type=:mean` returns the per-trial success
probability `μ = linkinv(link, η)`.
"""
function predict(fit::BetaBinomialGroupedFit, Y::AbstractMatrix{<:Real};
        N::Union{Nothing, AbstractMatrix{<:Real}} = nothing, type::Symbol = :mean)
    type in (:link, :mean) ||
        throw(ArgumentError("type must be :link or :mean; got :$type"))
    Z = getLV(fit, Y; N = N, rotate = false)           # n×K
    η = fit.β .+ fit.Λ * Z'                              # p×n
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

"""
    fit_beta_binomial_gllvm_grouped(Y; K, N=nothing, group=1:p, link=LogitLink(),
                                    mask=nothing, φ_init=nothing, …) -> BetaBinomialGroupedFit

Fit a beta-binomial GLLVM with grouped / species-specific Beta precision
(gllvm's `disp.group`; twin `log_phi_betabinom`): species `t` shares precision
`φ[group[t]]`. `group` is a length-p vector of group ids (relabelled to `1..G`
internally; default `1:p` = per-species). `N` is the matching p×n trial-count
matrix (default all-ones). L-BFGS over `[β; vec(Λ); log φ_1 … log φ_G]`;
finite-difference gradient (the Laplace inner mode-finder is not
forward-AD-friendly); warm start from empirical link-mean intercepts + SVD
loadings + a moderate per-group `φ₀`. With one group this matches
`fit_beta_binomial_gllvm` exactly.

Missing data: pass a `mask` (p×n Bool, `false` = unobserved) or `missing`
entries in `Y`; masked cells are dropped from the marginal and the warm start.
"""
function fit_beta_binomial_gllvm_grouped(Y::AbstractMatrix; K::Integer,
        N::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        group::AbstractVector{<:Integer} = collect(1:size(Y, 1)),
        link::Link = LogitLink(), mask = nothing, φ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    size(Nm) == (p, n) || throw(DimensionMismatch("N must be $(p)×$(n)"))
    length(group) == p || throw(ArgumentError("length(group)=$(length(group)) must equal p=$p"))
    rr = rr_theta_len(p, K)
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]

    msk = _resolve_obs_mask(mask, Y)
    Yc = Integer.(_sanitize_missing(Y, 0))

    Zemp = [linkfun(link, clamp((float(Yc[t, i]) + 0.5) / (float(Nm[t, i]) + 1),
                                1e-4, 1 - 1e-4)) for t in 1:p, i in 1:n]
    _mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    logφ0 = φ_init === nothing ? log(10.0) : log(float(φ_init))
    θ0 = vcat(β0, pack_lambda(Λ0), fill(logφ0, G))

    function negll(θ)
        β = θ[1:p]
        Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        φg = exp.(θ[(p + rr + 1):(p + rr + G)])
        φvec = [φg[gidx[t]] for t in 1:p]
        v = try
            -betabinomial_grouped_marginal_loglik_laplace(Yc, Nm, Λ, β, φvec; link = link,
                                                           mask = msk, maxiter = newton_maxiter,
                                                           tol = newton_tol)
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
    φ̂g = exp.(θ̂[(p + rr + 1):(p + rr + G)])
    return BetaBinomialGroupedFit(β̂, Λ̂, φ̂g, gidx, link,
                                  _beta_binomial_grouped_verdict(Optim.minimum(res), Optim.converged(res),
                                                                 Optim.iterations(res), φ̂g)...)
end

# ===========================================================================
# Grouped-dispersion + shared site-X fit driver — per-trait Beta precision φ
# PLUS shared covariate slopes γ (twin API B under X; decision 2026-08-05).
# Mirrors fit_nb1_gllvm_grouped_cov's packing convention: θ = [β; γ_free;
# pack(Λ); log φ_1 … log φ_G]; offset O = Xγ enters the grouped marginal.
# No hessian=:observed/:fisher knob (G0 lock — FD-outer, ForwardDiff-inner
# only; this file's own per-site Laplace has no analytic-Hessian variant yet).
# ===========================================================================

"""
    BetaBinomialGroupedCovFit

Result of `fit_beta_binomial_gllvm_grouped_cov`: per-trait intercepts
`β`, shared covariate coefficients `γ` (with `γ_fixed` zero mask), loadings
`Λ`, per-group Beta precision `φ`, species→group map `group`, `link`,
maximised Laplace `loglik`, `converged`, and `iterations`. Linear predictor
`η = β + Xγ + Λz` with species precision `φ[group[t]]`.
"""
struct BetaBinomialGroupedCovFit
    β::Vector{Float64}
    γ::Vector{Float64}
    γ_fixed::Vector{Bool}
    Λ::Matrix{Float64}
    φ::Vector{Float64}
    group::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    hessian::Symbol   # the Laplace log-det curvature this fit's objective used —
                       # ALWAYS :fisher (see BetaBinomialGroupedFit above; G0 lock).
end

# Positional compatibility constructor (2026-08-28): fixed at `:fisher` — this
# route has no `hessian` kwarg to record.
BetaBinomialGroupedCovFit(β, γ, γ_fixed, Λ, φ, group, link, loglik, converged, iterations) =
    BetaBinomialGroupedCovFit(β, γ, γ_fixed, Λ, φ, group, link, loglik, converged, iterations, :fisher)

function Base.show(io::IO, f::BetaBinomialGroupedCovFit)
    p, K = size(f.Λ); q = length(f.γ)
    print(io, "BetaBinomialGroupedCovFit(p=", p, ", q=", q, ", K=", K, ", G=", length(f.φ),
          ", φ=", round.(f.φ; sigdigits = 4),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

_loadings(fit::BetaBinomialGroupedCovFit) = fit.Λ
_loglik(fit::BetaBinomialGroupedCovFit)   = fit.loglik

function _nparams(fit::BetaBinomialGroupedCovFit)
    p, K = size(fit.Λ)
    return p + count(!, fit.γ_fixed) + rr_theta_len(p, K) + length(fit.φ)
end

"""
    getLV(fit::BetaBinomialGroupedCovFit, Y, X; N=nothing, rotate=true, mask=nothing) -> n×K matrix

Conditional latent scores at `η = β + Xγ + Λz` with per-trait beta-binomial
precision `φ[group[t]]`.
"""
function getLV(fit::BetaBinomialGroupedCovFit, Y::AbstractMatrix{<:Real},
        X::AbstractArray{<:Real, 3};
        N::Union{Nothing, AbstractMatrix{<:Real}} = nothing, rotate::Bool = true, mask = nothing)
    p, n = size(Y)
    K = size(fit.Λ, 2)
    Nm = N === nothing ? fill(1, p, n) : N
    φvec = [fit.φ[fit.group[t]] for t in 1:p]
    O = _build_offset(X, fit.γ)
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        mi = mask === nothing ? nothing : view(mask, :, s)
        oi = view(O, :, s)
        Z[:, s] = _beta_binomial_mode(view(Y, :, s), view(Nm, :, s), fit.Λ, fit.β, φvec;
                                      link = fit.link, mask = mi, offset = oi)
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::BetaBinomialGroupedCovFit, Y, X; N=nothing, type=:mean) -> p×n matrix

In-sample fitted values at the Laplace mode `ẑ` (see `getLV`):
`type=:link` returns `η = β + Xγ + Λẑ`; `type=:mean` returns the per-trial
success probability `μ = linkinv(link, η)`.
"""
function predict(fit::BetaBinomialGroupedCovFit, Y::AbstractMatrix{<:Real},
        X::AbstractArray{<:Real, 3};
        N::Union{Nothing, AbstractMatrix{<:Real}} = nothing, type::Symbol = :mean)
    type in (:link, :mean) ||
        throw(ArgumentError("type must be :link or :mean; got :$type"))
    Z = getLV(fit, Y, X; N = N, rotate = false)         # n×K
    O = _build_offset(X, fit.γ)
    η = fit.β .+ O .+ fit.Λ * Z'                          # p×n
    type === :link && return η
    return linkinv.(Ref(fit.link), _clamp_eta.(η))
end

"""
    fit_beta_binomial_gllvm_grouped_cov(Y; X, K, N=nothing, group=1:p, link=LogitLink(),
                                        mask=nothing, γ_fixed=nothing, φ_init=nothing,
                                        …) -> BetaBinomialGroupedCovFit

Fit a beta-binomial GLLVM with **grouped / per-trait Beta precision** and
**shared site covariates** `X` (`p×n×q`). This parameterisation uses per-trait
`φ_t`, matching gllvmTMB's `log_phi_betabinom`, together with shared `γ`.
Working vector `[β; γ_free; pack(Λ); log φ_1 … log φ_G]`; offset `O = Xγ` is
passed into `betabinomial_grouped_marginal_loglik_laplace`.
Finite-difference outer L-BFGS gradient (G0 lock; no `hessian=:observed`
knob yet — unlike the NB1/Beta/Gamma grouped_cov siblings, this file's Laplace
core has no analytic-Hessian variant). Keep `fit_beta_binomial_gllvm`
for the shared-`φ`, no-X opt-in. Identity checks against a constant `φvec` +
offset should use `group = ones(Int, p)`.
"""
function fit_beta_binomial_gllvm_grouped_cov(Y::AbstractMatrix; X::AbstractArray{<:Real, 3},
        K::Integer, N::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        group::AbstractVector{<:Integer} = collect(1:size(Y, 1)),
        link::Link = LogitLink(), mask = nothing, γ_fixed = nothing, φ_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    size(Nm) == (p, n) || throw(DimensionMismatch("N must be $(p)×$(n)"))
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q) = ($p, $n, q); got $(size(X))"))
    length(group) == p || throw(ArgumentError("length(group)=$(length(group)) must equal p=$p"))
    q_full = size(X, 3)
    γ_fixed_mask = _fixed_zero_mask(γ_fixed, q_full, "γ_fixed")
    X_fit, _ = _slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    rr = rr_theta_len(p, K)
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]

    msk = _resolve_obs_mask(mask, Y)
    Yc = Integer.(_sanitize_missing(Y, 0))
    Zemp = [linkfun(link, clamp((float(Yc[t, i]) + 0.5) / (float(Nm[t, i]) + 1),
                                1e-4, 1 - 1e-4)) for t in 1:p, i in 1:n]
    _mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    logφ0 = φ_init === nothing ? log(10.0) : log(float(φ_init))
    θ0 = vcat(β0, zeros(q), pack_lambda(Λ0), fill(logφ0, G))

    function negll(θ)
        β = θ[1:p]
        γ = θ[(p + 1):(p + q)]
        Λ = unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        φg = exp.(θ[(p + q + rr + 1):(p + q + rr + G)])
        φvec = [φg[gidx[t]] for t in 1:p]
        O = _build_offset(X_fit, γ)
        v = try
            -betabinomial_grouped_marginal_loglik_laplace(Yc, Nm, Λ, β, φvec; link = link,
                                                           mask = msk, offset = O,
                                                           maxiter = newton_maxiter,
                                                           tol = newton_tol)
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
    γ̂_free = θ̂[(p + 1):(p + q)]
    γ̂ = collect(Float64, _expand_fixed_zero(γ̂_free, γ_fixed_mask))
    Λ̂ = unpack_lambda(θ̂[(p + q + 1):(p + q + rr)], p, K)
    φ̂g = exp.(θ̂[(p + q + rr + 1):(p + q + rr + G)])
    return BetaBinomialGroupedCovFit(β̂, γ̂, collect(Bool, γ_fixed_mask), Λ̂, φ̂g, gidx, link,
                                     _beta_binomial_grouped_verdict(Optim.minimum(res),
                                                                    Optim.converged(res),
                                                                    Optim.iterations(res), φ̂g)...)
end
