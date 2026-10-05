# Mixed-family GLLVM (A2b headline).
#
# A single shared latent block Λ (p×K) drives p traits, but EACH trait may carry
# its OWN response family/link. This is the capability neither gllvmTMB nor DRM.jl
# has natively: a Poisson count trait, a Binomial binary trait, and a Beta
# proportion trait can load on ONE latent factor and yield a true cross-family
# trait correlation on the common latent (link) scale.
#
# Model (site s):
#   y_{ts} ~ Family_t(μ_{ts}[, n_{ts}][, dispersion_t]),  μ_{ts} = linkinv(link_t, η_{ts}),
#   η_{ts} = β_t + (Λ z_s)_t,   z_s ~ N(0, I_K).
# The marginal ∫ p(y_s|z) N(z;0,I) dz is computed by the SAME dense Laplace
# machinery as the single-family fitters (families/laplace.jl): find the
# conditional mode ẑ by Fisher scoring (expected Hessian A = Λ'WΛ + I is SPD),
# then  log p(y_s) ≈ Σ_t ℓ_t(ẑ) − ½ẑ'ẑ − ½logdet(A). The only difference vs the
# single-family loops is that the per-observation pieces dispatch on families[t]
# / links[t] instead of one global (family, link).
#
# v1 design notes:
#   * The gradient is a DIRECT ForwardDiff gradient straight through the inner
#     Fisher-scoring mode-find + logdet (authorized correctness-first path;
#     analytic per-trait kernels are a later perf lane). FD-verified ≤ 1e-6.
#   * Dispersion-carrying families (Normal σ², NB2 r, Gamma α, Beta φ) pack one
#     log-scale parameter each, in increasing trait order, after [β; vec(Λ)].
#   * Ordinal is rejected in v1 (vector μ / own mode-finder / no β); see
#     _mixed_family_layout.
#
# This file is ADDITIVE: it does not edit laplace.jl, link_residual.jl,
# confint_derived.jl, packing.jl, or any existing src/families/*.jl. It REUSES
# their per-observation scalar dispatch (_glm_score/_glm_weight/_glm_logpdf/
# _clamp_mu, linkinv/mu_eta) and the family-agnostic latent-scale assemblers
# (_latent_sigma/_latent_correlation).
#
# PORT NOTE (a1 → main): a1's mixed.jl reused three helpers from a1's
# families/laplace.jl (`_positive_from_log`, `_penalty_negloglik_fg!`,
# `_penalized_negloglik_fg!`). Main's laplace.jl does not define them (the main
# single-family fitters inline `exp(θ[...])` and call `Optim.optimize(...;
# autodiff = :finite)` / `_optimize_with_analytic`). To keep this port surgical —
# additive, no edit to main's laplace.jl — those three helpers are reproduced here
# verbatim from a1 as file-local private helpers. The fitter below is otherwise
# byte-identical to the a1-verified source, so the a1-vs-main parity gate holds.

# ---------------------------------------------------------------------------
# Ported helpers (verbatim from a1 src/families/laplace.jl; see PORT NOTE).
# ---------------------------------------------------------------------------

# Positive value from a log-scale parameter (clamped to avoid Inf/0 overflow).
_positive_from_log(x) = exp(clamp(x, -30.0, 30.0))

# Barrier fallback used when the marginal value/grad is non-finite at θ: returns a
# large penalty + ‖θ‖² (with a finite gradient pushing θ back toward 0), so L-BFGS
# steps away from the bad region rather than failing the whole fit.
function _penalty_negloglik_fg!(F, G, θ)
    if G !== nothing
        any_nonzero = false
        @inbounds for i in eachindex(θ)
            gi = if isfinite(θ[i])
                2 * θ[i]
            elseif θ[i] < 0
                -one(eltype(G))
            else
                one(eltype(G))
            end
            G[i] = gi
            any_nonzero |= !iszero(gi)
        end
        !any_nonzero && !isempty(G) && (G[1] = one(eltype(G)))
    end
    if F !== nothing
        s = zero(eltype(θ))
        @inbounds for x in θ
            isfinite(x) && (s += abs2(x))
        end
        return oftype(first(θ), 1e12) + s
    end
    return nothing
end

# Optim only_fg! callback: negate the marginal value/grad (Optim minimises), with
# a robust fallback to the barrier when the objective is non-finite or throws.
function _penalized_negloglik_fg!(F, G, value_grad, θ)
    try
        value, grad = value_grad(θ)
        if !isfinite(value) || !all(isfinite, grad)
            return _penalty_negloglik_fg!(F, G, θ)
        end
        G !== nothing && (G .= .-grad)
        F !== nothing && return -value
        return nothing
    catch
        return _penalty_negloglik_fg!(F, G, θ)
    end
end

# ---------------------------------------------------------------------------
# Normal family pieces for the generic Laplace core.
#
# A Gaussian trait flows through the SAME per-observation dispatch as the GLM
# families. With the identity link (me = 1) and variance σ² (carried in
# Normal(μ, σ).σ):
#   y ~ N(η, σ²),  score wrt η = (y − η)/σ²,  Fisher weight wrt η = 1/σ².
# The Laplace approximation is EXACT for a Gaussian integrand, so an all-Normal
# mixed marginal reproduces the closed-form Gaussian marginal to machine
# precision.
#
# PORT NOTE (a1 → main): a1's mixed.jl DEFINED the four Normal kernels
# (`_clamp_mu`/`_glm_score`/`_glm_weight`/`_glm_logpdf` on `::Normal`) because a1
# had no other file that did. Main ALREADY defines exactly these in
# src/spde_latent.jl (lines 52–55, functionally identical). Re-defining them here
# would trigger "Method overwriting is not permitted during precompilation", so
# this port REUSES the existing main definitions and does NOT redeclare them.
# Normal is still treated as a fourth dispersion-carrying family below (σ on the
# packed log-dispersion tail, like NB r / Gamma α / Beta φ).
# ---------------------------------------------------------------------------

# ===========================================================================
# Per-trait dispersion layout: the single source of truth for "which traits
# carry a scalar dispersion and where it lives in the packed θ tail".
# ===========================================================================

# Does this family marker carry one scalar log-dispersion parameter?
_mixed_has_dispersion(::Normal)           = true   # σ
_mixed_has_dispersion(::NegativeBinomial) = true   # r
_mixed_has_dispersion(::Gamma)            = true   # α
_mixed_has_dispersion(::Beta)             = true   # φ
_mixed_has_dispersion(::Poisson)          = false
_mixed_has_dispersion(::Binomial)         = false
_mixed_has_dispersion(fam) = throw(ArgumentError(
    "fit_mixed_gllvm v1 supports Normal, Poisson, Binomial, NegativeBinomial, " *
    "Gamma, Beta per trait; got an unsupported family marker $(typeof(fam)). " *
    "Ordinal and other families are not yet available."))

# Default log-dispersion init per family (mirrors the single-family fitters:
# NB log r₀=log 10, Beta log φ₀=log 10, Gamma log α₀=log 2; Normal log σ₀=0).
_mixed_default_logdisp(::Normal)           = 0.0
_mixed_default_logdisp(::NegativeBinomial) = log(10.0)
_mixed_default_logdisp(::Gamma)            = log(2.0)
_mixed_default_logdisp(::Beta)             = log(10.0)

"""
    _mixed_family_layout(families) -> (disp_index, n_disp)

Single source of truth for the packed dispersion tail. `disp_index[t]` is 0 if
trait `t` carries no scalar dispersion, else its 1-based slot in the log-scale
tail (slots assigned in increasing trait order). `n_disp == count(disp_index .> 0)`.
Throws an `ArgumentError` for unsupported families (e.g. Ordinal).
"""
function _mixed_family_layout(families::AbstractVector)
    p = length(families)
    disp_index = zeros(Int, p)
    slot = 0
    @inbounds for t in 1:p
        if _mixed_has_dispersion(families[t])
            slot += 1
            disp_index[t] = slot
        end
    end
    return disp_index, slot
end

# Rebuild a dispersion-carrying family marker from its raw (positive) value,
# mirroring the single-family `family_from_aux` closures (negbin/beta/gamma).
# AD-clean: the raw value is a Dual under ForwardDiff and flows into the marker.
_with_dispersion(::Normal, d)           = Normal(zero(d), d)               # σ = d
_with_dispersion(::NegativeBinomial, d) = NegativeBinomial(d, oftype(d, 0.5))
_with_dispersion(::Gamma, d)            = Gamma(d, one(d))
_with_dispersion(::Beta, d)             = Beta(d, one(d))
_with_dispersion(f::Poisson, d)         = f
_with_dispersion(f::Binomial, d)        = f

"""
    _mixed_unpack(θ, p, K, families, disp_index) -> (β, Λ, dispersion)

Split the packed `θ = [β(1:p); pack_lambda(Λ); log-dispersion tail]` into the
per-trait intercepts `β`, the shared loadings `Λ` (p×K), and a length-p raw
dispersion vector (`dispersion[t]` the positive value for dispersion-carrying
traits; a unit sentinel where `disp_index[t] == 0`). AD-friendly: `eltype(θ)`
is preserved.
"""
function _mixed_unpack(θ::AbstractVector, p::Int, K::Int,
        families::AbstractVector, disp_index::AbstractVector{Int})
    rr = rr_theta_len(p, K)
    T = eltype(θ)
    β = θ[1:p]
    Λ = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
    dispersion = Vector{T}(undef, p)
    @inbounds for t in 1:p
        dispersion[t] = disp_index[t] > 0 ?
            _positive_from_log(θ[p + rr + disp_index[t]]) : one(T)
    end
    return β, Λ, dispersion
end

# ===========================================================================
# Mixed dense-Laplace marginal — structural twin of laplace.jl's single-family
# loops with the scalar family/link swapped to families[t]/links[t].
# ===========================================================================

# Per-site log-posterior at a given z (mixed-family twin of `_laplace_mode_logpost`
# / `_grouped_laplace_mode_logpost`; see #507/#509/#500). The step-halving line
# search inside `_mixed_laplace_mode` below accepts a trial z only when it raises
# this value. Missing cells are dropped, matching `_mixed_laplace_mode` itself.
# Linear predictor of trait `t` before clamping: β_t + offset_t + (Λz)_t. With no offset
# the expression is exactly the pre-offset `β[t] + η[t]`, so offset-free fits are unchanged.
@inline _mixed_eta(β, η, ::Nothing, t) = β[t] + η[t]
@inline _mixed_eta(β, η, offset, t) = β[t] + offset[t] + η[t]

function _mixed_logpost(families::AbstractVector, links::AbstractVector,
        y::AbstractVector, n::AbstractVector, Λ::AbstractMatrix, β::AbstractVector,
        z::AbstractVector; offset = nothing)
    p = size(Λ, 1)
    η = Λ * z
    q = -0.5 * dot(z, z)
    @inbounds for t in 1:p
        ismissing(y[t]) && continue
        ηt = _clamp_eta(_mixed_eta(β, η, offset, t))
        μt = _clamp_mu(families[t], linkinv(links[t], ηt))
        q += _glm_logpdf(families[t], μt, n[t], y[t])
    end
    return q
end

# Mixed site mode-finder: Fisher scoring, each observation on its trait's own
# family/link. `families` markers carry dispersion. Returns `(z, converged)`.
#
# Before this fix (#503, the #479/#507/#509/#500 pattern) the loop below took
# undamped Fisher-scoring steps and returned whatever `z` it held when `maxiter`
# was reached, converged or not: a step that overshoots the per-site
# log-posterior was never rejected, so a site could be left far from its
# stationary point while `_mixed_loglik_site` still reported a finite value. A
# re-measure on origin/main found this at a non-trivial rate across several
# family mixes (Poisson/Binomial/Gamma, Normal/NB2/Beta, and a four-trait
# Poisson/Gamma/Beta/Binomial mix) using this file's own gradient-of-the-score
# check (`Λ's − z`).
#
# The fix mirrors the other grouped kernels: a step that lowers the per-site
# log-posterior is halved (small steps and accepted full steps are bit-identical
# to the old loop), and `converged` requires BOTH the full proposed step and a
# scale-aware gradient check to be small. A step-size-only test is not safe
# here because `A = Λ'WΛ + I` can be ill-conditioned (e.g. a Normal trait
# whose fitted σ is driven near zero — a Heywood case the outer fitter can
# reach — makes `W = 1/σ²` enormous), and in that regime a tiny `Δ` solves
# `AΔ = g` while the raw gradient `g` is still large in the stiff direction.
# The check below is the Newton decrement `g'Δ` (both already computed to get
# `Δ`, so this is free): it scales WITH the curvature, so it stays small at a
# genuine mode even when `A`'s eigenvalues are huge, unlike an absolute bound
# on `maximum(abs, g)`. An earlier draft used the
# absolute form, reasoning by analogy to Student-t's #509 Fisher/observed
# mismatch, which does not apply here — a Normal trait under `IdentityLink`
# has a quadratic log-likelihood, so its Fisher weight `1/σ²` IS the observed
# curvature, not a defect of the #509 kind. Since every family this bridge
# supports has a nonnegative expected (Fisher) weight (file header: "expected
# Hessian ⇒ Λ'WΛ + I is always SPD"), the step-halving line search below needs
# no per-family backtrack gate and no observed-curvature fallback direction:
# Fisher scoring is always a valid ascent direction here. That says nothing
# about whether the UNDAMPED fixed-point map (the small-step shortcut just
# below) is a CONTRACTION at the mode, though: where it is not (e.g. a Gamma
# trait with shape well below 1 and `y/μ` far from 1, observed/Fisher weight
# ratio > 1), the shortcut can oscillate indefinitely rather than settle, so
# once a full-size line-search step has been rejected once, every subsequent
# step in this call goes through the line search too, unless the step is already
# at the floating-point floor, where the line search can only compare rounding
# noise.
function _mixed_laplace_mode(families::AbstractVector, links::AbstractVector,
        y::AbstractVector, n::AbstractVector, Λ::AbstractMatrix, β::AbstractVector;
        maxiter::Integer = 100, tol::Real = 1e-9, grad_tol::Real = 1e-6, nd_tol::Real = 1e-12, z_init = nothing,
        offset = nothing)
    p, K = size(Λ)
    T = promote_type(eltype(Λ), eltype(β))
    z = z_init === nothing ? zeros(T, K) : collect(T, z_init)
    s = Vector{T}(undef, p)
    W = Vector{T}(undef, p)
    linesearch_only = false   # set once a full step has been rejected
    prev_dmax = Inf           # previous max|Δ|: has the step stopped shrinking?
    for _ in 1:maxiter
        η = Λ * z
        @inbounds for t in 1:p
            if ismissing(y[t])                  # NA-aware FIML: drop the missing cell
                s[t] = zero(T); W[t] = zero(T)  # 0 score/weight ⇒ leaves A SPD, off the mode
            else
                ηt = _clamp_eta(_mixed_eta(β, η, offset, t))
                μt = _clamp_mu(families[t], linkinv(links[t], ηt))
                met = mu_eta(links[t], ηt)
                s[t] = _glm_score(families[t], μt, n[t], met, y[t])
                W[t] = _glm_weight(families[t], μt, n[t], met)
            end
        end
        g = Λ' * s .- z
        A = Symmetric(Λ' * (W .* Λ) + I)
        Δ = _safe_solve(A, g)
        (Δ === nothing || !all(isfinite, Δ)) && return z, false
        # Scale-aware convergence check: the Newton decrement `g'Δ`, not an
        # absolute bound on `g`. `Δ` solves `AΔ = g`, so this is `g'A⁻¹g`,
        # which stays small at a genuine mode even when `A` is ill-conditioned.
        decrement = abs(dot(g, Δ))
        maximum(abs, Δ) < tol && decrement < grad_tol && return z .+ Δ, true
        # The latch below forces the line search only while the step is above the
        # floating-point floor (about sqrt(eps) relative to z): at the floor the
        # log-posterior differences are rounding noise, and a line search there can
        # reject a genuine step and report a real mode as a failure.
        dmax = maximum(abs, Δ)
        at_floor = dmax <= sqrt(eps(Float64)) * (1 + norm(z))
        # At the floor, a step that has stopped shrinking with the Newton
        # decrement below `nd_tol` is the mode to rounding: the undamped map is
        # bouncing at the floor (measured: |Δ| wandering 1e-8 to 6e-8 for 2000
        # iterations, `g'Δ` about 1e-15), and no further step can be resolved.
        # A step that is still shrinking keeps going until `tol`, so a caller that
        # asks for a tighter mode (e.g. `tol = 1e-13`) still gets one.
        at_floor && dmax >= prev_dmax && decrement < nd_tol && return z .+ Δ, true
        prev_dmax = dmax
        if (!linesearch_only || at_floor) && norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        else
            q0 = _mixed_logpost(families, links, y, n, Λ, β, z; offset = offset)
            if isfinite(q0)
                zprev = z
                accepted = false
                step = 1.0
                for _half in 1:30
                    ztrial = z .+ step .* Δ
                    q1 = _mixed_logpost(families, links, y, n, Λ, β, ztrial; offset = offset)
                    if isfinite(q1) && q1 >= q0
                        z = ztrial
                        accepted = true
                        break
                    end
                    step *= 0.5
                    linesearch_only = true   # a full-size step was just rejected
                end
                # A line search that cannot move `z` (every trial rejected, or the
                # accepted trial rounds back to `z`) while the Newton decrement is
                # below `nd_tol` means the log-posterior can no longer resolve the
                # remaining step: `z` is the mode to rounding (measured: |Δ| stuck
                # near 1e-8 with `g'Δ` about 1e-14 and every trial lower by noise).
                # That is convergence, not failure.
                (!accepted || z == zprev) && decrement < nd_tol && return z, true
                accepted || return z, false
            else
                z = z .+ Δ
            end
        end
    end
    return z, false
end

# Laplace log-marginal for one mixed site: Σ_t ℓ_t(ẑ) − ½ẑ'ẑ − ½logdet(Λ'WΛ + I).
function _mixed_loglik_site(families::AbstractVector, links::AbstractVector,
        y::AbstractVector, n::AbstractVector, Λ::AbstractMatrix, β::AbstractVector;
        maxiter::Integer = 100, tol::Real = 1e-9, z_init = nothing, offset = nothing)
    p = size(Λ, 1)
    z, ok = _mixed_laplace_mode(families, links, y, n, Λ, β;
                                maxiter = maxiter, tol = tol, z_init = z_init, offset = offset)
    # Larger-budget retry (#507/#509 review pattern): a genuinely converging site
    # can still need more than the default `maxiter` Fisher-scored steps under
    # ill-conditioned cross-family curvature. Retry from the SAME `z_init` (zero
    # in every current call path) with a 20x iteration budget before declaring
    # failure; runs only where the default budget failed, so every site that
    # converged within `maxiter` keeps its original path and iteration count.
    ok || ((z, ok) = _mixed_laplace_mode(families, links, y, n, Λ, β;
                                         maxiter = 20 * maxiter, tol = tol, z_init = z_init,
                                         offset = offset))
    # A search that did not certify a stationary point must not produce a
    # finite value: -Inf makes the fitter's own 1e12 failure sentinel fire
    # instead of a silently wrong log-likelihood (#503, the #479/#507/#509/#500
    # pattern). `getLV`/`predict` call `_mixed_laplace_mode` directly (below) and
    # never see this sentinel: they use whatever z the damped search returns,
    # converged or not, exactly as the old code did.
    ok || return -Inf
    η = Λ * z
    T = promote_type(eltype(Λ), eltype(β))
    W = Vector{T}(undef, p)
    ℓ = zero(T)
    @inbounds for t in 1:p
        if ismissing(y[t])                      # NA-aware FIML: drop the missing cell
            W[t] = zero(T)                      # 0 weight (off A); skipped in the ℓ sum
        else
            ηt = _clamp_eta(_mixed_eta(β, η, offset, t))
            μt = _clamp_mu(families[t], linkinv(links[t], ηt))
            met = mu_eta(links[t], ηt)
            # Per-trait log-det curvature. Each trait takes ITS OWN family's
            # default, which is what makes the two bridge routes agree:
            # `bridge.jl:472-479` sends `family = "gamma"` to the generic core
            # but an all-same vector `["gamma", …]` here, and the comment there
            # says so explicitly. Before this, a Gamma family flipped in the core
            # but not here returned two different log-likelihoods for the SAME
            # model on the public R surface.
            #
            # Missing cells never reach this branch (guarded above), so the
            # observed weight — which reads `y` — is safe here.
            W[t] = if _default_hessian(families[t], links[t]) === :fisher ||
                      _glm_weight_matches_observed(families[t], links[t])
                _glm_weight(families[t], μt, n[t], met)
            else
                _glm_obs_weight(families[t], μt, n[t], met, y[t], links[t], ηt)
            end
            ℓ += _glm_logpdf(families[t], μt, n[t], y[t])
        end
    end
    A = Symmetric(Λ' * (W .* Λ) + I)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

"""
    mixed_marginal_loglik_laplace(families, links, Y, N, Λ, β; kwargs...) -> Real

Total Laplace log-marginal over the `n` sites (columns) of a MIXED-family GLLVM.
`families`/`links` are length-`p` per-trait recipes (dispersion baked into the
family markers); `Y`, `N` are p×n response and trial-count matrices; `Λ` p×K;
`β` length-p. Reuses the family-generic per-observation dispatch of
families/laplace.jl on a per-trait basis. `offset` (`nothing` or a p×n matrix) is added
to the linear predictor, η = β + offset + Λz.
"""
function mixed_marginal_loglik_laplace(families::AbstractVector, links::AbstractVector,
        Y::AbstractMatrix, N::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector;
        offset = nothing, kwargs...)
    acc = zero(promote_type(eltype(Λ), eltype(β)))
    @inbounds for i in axes(Y, 2)
        oi = offset === nothing ? nothing : view(offset, :, i)
        acc += _mixed_loglik_site(families, links, view(Y, :, i), view(N, :, i),
                                  Λ, β; offset = oi, kwargs...)
    end
    return acc
end

# Packed-θ value entry point: split θ, bake raw dispersions into the markers,
# then evaluate the mixed marginal. `families` here are the *bare* per-trait
# markers (dispersion is read from θ, not from these).
function _mixed_marginal_loglik_packed(θ::AbstractVector, Y::AbstractMatrix,
        N::AbstractMatrix, p::Int, K::Int, families::AbstractVector,
        links::AbstractVector, disp_index::AbstractVector{Int}; kwargs...)
    β, Λ, disp = _mixed_unpack(θ, p, K, families, disp_index)
    fams_t = [_with_dispersion(families[t], disp[t]) for t in 1:p]
    return mixed_marginal_loglik_laplace(fams_t, links, Y, N, Λ, β; kwargs...)
end

# ===========================================================================
# Fitted-model struct.
# ===========================================================================

"""
    MixedFamilyFit

Result of [`fit_mixed_gllvm`](@ref): a mixed-family GLLVM where each of the `p`
traits carries its own response family/link but all share one latent block `Λ`.

Fields:
- `β::Vector{Float64}` — per-trait intercept (length `p`), on each trait's LINK scale.
- `Λ::Matrix{Float64}` — `p×K` shared loadings (the headline: ONE latent block).
- `families::Vector{Any}` — per-trait `Distributions` family markers.
- `links::Vector{Link}` — per-trait links.
- `dispersion::Vector{Float64}` — per-trait scalar nuisance (NB2 `r` / Gamma `α`
  / Beta `φ` / Normal `σ`); `NaN` where the trait carries none.
- `disp_index::Vector{Int}` — per-trait 1-based slot in the packed log-dispersion
  tail (0 if none). Cached so consumers never re-derive the layout.
- `n_disp::Int` — number of dispersion-carrying traits.
- `link::Link` — convenience (`links[1]`); not load-bearing.
- `loglik::Float64`, `converged::Bool`, `iterations::Int` — universal fit fields.
- `offset` — the p×n training offset the fit was made with (`nothing` when it had
  none); [`predict`](@ref) and [`getLV`](@ref) use it by default.
"""
struct MixedFamilyFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    families::Vector{Any}
    links::Vector{Link}
    dispersion::Vector{Float64}
    disp_index::Vector{Int}
    n_disp::Int
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    offset::Union{Nothing, Matrix{Float64}}   # training offset (p×n); `nothing` = none
end

# Pre-offset compat tier (11 positional args): no stored training offset.
MixedFamilyFit(β, Λ, families, links, dispersion, disp_index, n_disp, link,
               loglik, converged, iterations) =
    MixedFamilyFit(β, Λ, families, links, dispersion, disp_index, n_disp, link,
                   loglik, converged, iterations, nothing)

function Base.show(io::IO, f::MixedFamilyFit)
    p, K = size(f.Λ)
    fams = "[" * join((nameof(typeof(fam)) for fam in f.families), ", ") * "]"
    print(io, "MixedFamilyFit(p=", p, ", K=", K, ", families=", fams,
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

# ===========================================================================
# Family-aware PPCA warm start: per-trait link-scale pseudodata rows, then one
# shared SVD (the cross-family-on-one-latent-block premise).
# ===========================================================================

# Per-trait link-scale pseudodata row, lifting each single-family fitter's Zemp
# line. Returns a length-n Float64 row on trait t's link scale.
function _mixed_pseudo_link_row(fam::Normal, link::Link, y::AbstractVector, n::AbstractVector)
    # NA-aware warm start (issue #27, slice 1): observed-cell identity pseudodata; a
    # missing cell is filled with the row's observed mean for the shared SVD init ONLY
    # (the FIML objective itself drops missing cells — see _mixed_loglik_site). On a
    # dense row cnt == length(y) ⇒ the fill loop is a no-op ⇒ byte-identical to the
    # old comprehension. (Only the Normal method is widened — all-Normal NA-Gaussian;
    # mixed non-Gaussian NA is a separate slice.)
    row = Vector{Float64}(undef, length(y))
    acc = 0.0; cnt = 0
    @inbounds for i in eachindex(y)
        if !ismissing(y[i])
            row[i] = linkfun(link, float(y[i]))            # identity
            acc += row[i]; cnt += 1
        end
    end
    m = cnt == 0 ? 0.0 : acc / cnt
    @inbounds for i in eachindex(y)
        ismissing(y[i]) && (row[i] = m)
    end
    return row
end
function _mixed_pseudo_link_row(fam::Union{Poisson, NegativeBinomial}, link::Link,
        y::AbstractVector, n::AbstractVector)
    return [linkfun(link, max(float(y[i]) + 0.5, 1e-4)) for i in eachindex(y)]
end
function _mixed_pseudo_link_row(fam::Binomial, link::Link, y::AbstractVector, n::AbstractVector)
    return [linkfun(link, clamp((float(y[i]) + 0.5) / (float(n[i]) + 1), 1e-4, 1 - 1e-4))
            for i in eachindex(y)]
end
function _mixed_pseudo_link_row(fam::Beta, link::Link, y::AbstractVector, n::AbstractVector)
    return [linkfun(link, clamp(float(y[i]), 1e-6, 1 - 1e-6)) for i in eachindex(y)]
end
function _mixed_pseudo_link_row(fam::Gamma, link::Link, y::AbstractVector, n::AbstractVector)
    return [linkfun(link, max(float(y[i]), 1e-6)) for i in eachindex(y)]
end

# ===========================================================================
# Public fit driver.
# ===========================================================================

"""
    fit_mixed_gllvm(Y; families, links=default, K, N=nothing, …) -> MixedFamilyFit

Fit a MIXED-family GLLVM by L-BFGS on the mixed dense-Laplace marginal
log-likelihood. Each of the `p` rows of `Y` (traits) carries its own response
`families[t]` and `links[t]`, but all share one `K`-dimensional latent block
`Λ`. This yields a true cross-family trait correlation on the common latent
(link) scale via [`correlation`](@ref).

Arguments:
- `Y::AbstractMatrix` — `p×n` response matrix (traits × sites). Mixed element
  types are allowed (counts, proportions, positive reals); each row is read by
  its trait's family.
- `families::Vector` — length-`p` `Distributions` markers; supported in v1:
  `Normal()`, `Poisson()`, `Binomial()`, `NegativeBinomial()`, `Gamma()`,
  `Beta()` (Ordinal is not yet available for mixed-family fits).
- `links` — length-`p` links; defaults to each family's canonical link.
- `K::Integer` — latent dimension.
- `N` — Binomial trial counts (`p×n`); defaults to all-ones.
- `offset` — a known addition to the linear predictor, η = β + offset + Λz: a `p×n`
  matrix (the layout of `Y`), a scalar, or a length-`p` vector (one per trait),
  normalised as in [`fit_gllvm`](@ref). A zero offset on a non-count trait next to
  nonzero offsets on the count traits is the usual mixed-family use (an exposure
  `log(e)` on the counts only). The fit keeps it in `fit.offset`.

The L-BFGS gradient is a DIRECT ForwardDiff gradient of the pure-value mixed
marginal (correctness-first v1; analytic per-trait kernels are future performance
work). FD-verified ≤ 1e-6. Warm start: family-aware link-scale pseudodata rows +
one shared SVD (PPCA-style) + per-family default dispersions.
"""
function fit_mixed_gllvm(Y::AbstractMatrix; families::AbstractVector, K::Integer,
        links::Union{Nothing, AbstractVector} = nothing,
        N::Union{Nothing, AbstractMatrix} = nothing,
        β_init = nothing, Λ_init = nothing, dispersion_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9, offset = nothing)
    p, n = size(Y)
    offset = _normalize_offset(offset, p, n; Y = Y, caller = "fit_mixed_gllvm")
    length(families) == p || throw(DimensionMismatch(
        "families has length $(length(families)); expected p = $p (one per trait/row)"))
    links_v = links === nothing ? Link[default_link(fam) for fam in families] :
              collect(Link, links)
    length(links_v) == p || throw(DimensionMismatch(
        "links has length $(length(links_v)); expected p = $p"))
    Nm = N === nothing ? ones(Int, p, n) : N
    size(Nm) == (p, n) || throw(DimensionMismatch("N must be $(p)×$(n)"))
    rr = rr_theta_len(p, K)

    # Layout (single source of truth) — also validates the families.
    disp_index, n_disp = _mixed_family_layout(families)

    # Family-aware PPCA warm start: per-trait link-scale pseudodata, one SVD.
    Zemp = Matrix{Float64}(undef, p, n)
    @inbounds for t in 1:p
        Zemp[t, :] = _mixed_pseudo_link_row(families[t], links_v[t],
                                            view(Y, t, :), view(Nm, t, :))
    end
    # Offset (η = β + offset + Λz): take it out of the link-scale pseudodata. A missing
    # cell may carry a non-finite offset; its pseudodata is a fill value, left as is.
    offset === nothing || (Zemp .-= ifelse.(isfinite.(offset), offset, 0.0))
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

    # Log-dispersion tail seeded per family (or from dispersion_init overrides).
    logdisp0 = zeros(Float64, n_disp)
    @inbounds for t in 1:p
        if disp_index[t] > 0
            logdisp0[disp_index[t]] = if dispersion_init === nothing
                _mixed_default_logdisp(families[t])
            else
                log(float(dispersion_init[t]))
            end
        end
    end

    θ0 = vcat(β0, pack_lambda(Λ0), logdisp0)
    families_bare = collect(Any, families)

    # Direct ForwardDiff gradient straight through the mixed dense-Laplace
    # marginal (authorized v1 path; FD-verified ≤ 1e-6).
    value_only(θ) = _mixed_marginal_loglik_packed(
        θ, Y, Nm, p, K, families_bare, links_v, disp_index;
        maxiter = newton_maxiter, tol = newton_tol, offset = offset)
    value_grad(θ) = (value_only(θ), ForwardDiff.gradient(value_only, θ))
    negll_fg!(F, G, θ) = _penalized_negloglik_fg!(F, G, value_grad, θ)
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(Optim.only_fg!(negll_fg!), θ0, ls,
                         Optim.Options(g_tol = g_tol, iterations = iterations))

    θ̂ = Optim.minimizer(res)
    β̂ = θ̂[1:p]
    Λ̂ = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    dispersion = fill(NaN, p)
    @inbounds for t in 1:p
        disp_index[t] > 0 && (dispersion[t] = _positive_from_log(θ̂[p + rr + disp_index[t]]))
    end
    return MixedFamilyFit(β̂, Λ̂, families_bare, links_v, dispersion, disp_index,
                          n_disp, links_v[1], _fit_verdict(res)..., _stored_offset(offset))
end

# ===========================================================================
# Post-fit: latent modes + per-trait fitted means (additive; needed by the
# σ²_d assembler below). Kept here rather than postfit.jl so the whole
# mixed-family capability lives in one file.
# ===========================================================================

"""
    getLV(fit::MixedFamilyFit, Y; N=nothing, rotate=true) -> n×K matrix

Conditional latent-variable scores (per-site Laplace mode `ẑₛ`) for a mixed fit.
`rotate=true` applies the canonical SVD rotation of `Λ`.

On a fit made with an `offset` the mode search uses it (η = β + offset + Λz): the
stored training offset (`fit.offset`) when `Y` has the training size, otherwise the
`offset` you pass (a p×n matrix, a scalar or a length-p vector). New units from an
offset fit without an `offset` are refused, as gllvmTMB refuses `newdata` that lacks
the offset variable.
"""
function getLV(fit::MixedFamilyFit, Y::AbstractMatrix;
               N::Union{Nothing, AbstractMatrix} = nothing, rotate::Bool = true,
               offset = nothing)
    p, n = size(Y)
    O = _laplace_prediction_offset(fit.offset, Y, offset, nothing, "getLV")
    Nm = N === nothing ? ones(Int, p, n) : N
    K = size(fit.Λ, 2)
    # dispersion[t] is NaN for non-dispersion traits; a unit sentinel keeps the
    # marker well-formed (Poisson/Binomial markers ignore the value anyway).
    fams_t = [_with_dispersion(fit.families[t],
                isfinite(fit.dispersion[t]) ? fit.dispersion[t] : 1.0) for t in 1:p]
    Z = Matrix{Float64}(undef, K, n)
    @inbounds for s in 1:n
        # No-sentinel contract (#503): take whichever z the damped search
        # returns, converged or not, same as the pre-#503 code, which had no
        # convergence flag to discard in the first place.
        z, _ = _mixed_laplace_mode(fams_t, fit.links, view(Y, :, s), view(Nm, :, s),
                                   fit.Λ, fit.β;
                                   offset = O === nothing ? nothing : view(O, :, s))
        Z[:, s] = z
    end
    Zt = permutedims(Z)
    return rotate ? Zt * _svd_rotation(fit.Λ) : Zt
end

"""
    predict(fit::MixedFamilyFit, Y; type=:response, N=nothing) -> p×n matrix

In-sample fitted values at the Laplace mode. `type=:link` returns
`η[t,s] = β_t + (Λ ẑ_s)_t`; `type=:response` the per-trait inverse-link
`linkinv(link_t, η[t,s])`.

On a fit made with an `offset`, `η` includes it: the stored training offset
(`fit.offset`) when `Y` has the training size, otherwise the `offset` you pass.
New units from an offset fit without an `offset` are refused (see [`getLV`](@ref)).
"""
function predict(fit::MixedFamilyFit, Y::AbstractMatrix;
                 type::Symbol = :response, N::Union{Nothing, AbstractMatrix} = nothing,
                 offset = nothing)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response; got :$type"))
    p, n = size(Y)
    O = _laplace_prediction_offset(fit.offset, Y, offset, nothing, "predict")
    Z = getLV(fit, Y; N = N, rotate = false, offset = O)
    η = fit.β .+ fit.Λ * Z'
    O === nothing || (η .+= O)
    type === :link && return η
    out = similar(η, Float64)
    @inbounds for s in 1:n, t in 1:p
        out[t, s] = linkinv(fit.links[t], η[t, s])
    end
    return out
end

# Per-trait response-scale mean fitted mean μ̂ (length p) — the mixed twin of
# _trait_mean_fitted, feeding the σ²_d assembler.
function _mixed_trait_mean_fitted(fit::MixedFamilyFit, Y::AbstractMatrix;
                                  N::Union{Nothing, AbstractMatrix} = nothing)
    μ = predict(fit, Y; type = :response, N = N)
    return vec(Statistics.mean(μ; dims = 2))
end

# ===========================================================================
# Cross-family latent-scale extractors (the headline output).
#
# These feed the EXISTING family-agnostic kernels _latent_sigma /
# _latent_correlation (link_residual.jl) a per-trait σ²_d VECTOR assembled by
# looping the EXISTING scalar 4-arg link_residual(family, link, μ̂, disp). No
# change to those kernels beyond these MixedFamilyFit dispatches.
# ===========================================================================

# Per-trait latent-scale residual for one mixed trait. For every GLM family this
# is the scalar link-implicit residual (`_link_residual_one`). For a NORMAL trait
# it is the per-trait response-scale variance σ_t² itself: a Gaussian/identity
# trait has NO link-implicit residual (`_link_residual_one(::Normal,…)==0`, since
# the single-family Gaussian path adds σ_eps² separately via
# `sigma_y_site(::GllvmFit)`), so in the mixed assembler — where nothing else
# injects it — σ_t² is the residual that belongs on diag(Σ_latent) to put the
# Normal trait on the common latent scale (Σ_latent = ΛΛᵀ + diag(σ_t²), the exact
# mixed analogue of the Gaussian fit's ΛΛᵀ + σ_eps²·I).
_mixed_trait_residual(fam, link, μ̂, disp) = Float64(_link_residual_one(fam, link, μ̂, disp))
_mixed_trait_residual(::Normal, ::IdentityLink, μ̂, σ::Real) = Float64(σ)^2

"""
    link_residual(fit::MixedFamilyFit, Y; N=nothing) -> Vector{Float64}

Per-trait link-implicit residual variance σ²_d (length `p`) for a mixed fit —
the diagonal added to `ΛΛᵀ` to put all traits on a common latent scale. Each
GLM trait calls the scalar `link_residual(family_t, link_t, μ̂_t, disp_t)`;
a Normal trait contributes its per-trait variance σ_t².
"""
function link_residual(fit::MixedFamilyFit, Y::AbstractMatrix;
                       N::Union{Nothing, AbstractMatrix} = nothing)
    p = size(fit.Λ, 1)
    μ̂ = _mixed_trait_mean_fitted(fit, Y; N = N)
    return [_mixed_trait_residual(fit.families[t], fit.links[t], μ̂[t],
                    fit.disp_index[t] > 0 ? fit.dispersion[t] : nothing) for t in 1:p]
end

"""
    sigma_y_site(fit::MixedFamilyFit, Y; N=nothing) -> Matrix

Latent-scale trait covariance `Σ_latent = ΛΛᵀ + diag(σ²_d)` for a mixed fit.
"""
sigma_y_site(fit::MixedFamilyFit, Y::AbstractMatrix;
             N::Union{Nothing, AbstractMatrix} = nothing) =
    _latent_sigma(fit.Λ, link_residual(fit, Y; N = N))

"""
    correlation(fit::MixedFamilyFit, Y; N=nothing) -> Matrix

Cross-family latent-scale trait correlation `R = D^{-1/2} Σ_latent D^{-1/2}` —
THE headline output: a true correlation between traits of different response
families on the common latent (link) scale. Diagonal exactly 1.0; off-diagonals
in [-1, 1]. Rotation-invariant (built on ΛΛᵀ).
"""
correlation(fit::MixedFamilyFit, Y::AbstractMatrix;
            N::Union{Nothing, AbstractMatrix} = nothing) =
    _latent_correlation(sigma_y_site(fit, Y; N = N))

"""
    communality(fit::MixedFamilyFit, Y; N=nothing) -> Vector

Per-trait communality `c²[t] = (ΛΛᵀ)[t,t] / Σ_latent[t,t]` on the latent scale.
"""
function communality(fit::MixedFamilyFit, Y::AbstractMatrix;
                     N::Union{Nothing, AbstractMatrix} = nothing)
    ΛΛt = fit.Λ * fit.Λ'
    Σ = sigma_y_site(fit, Y; N = N)
    return [_safe_ratio(ΛΛt[t, t], Σ[t, t]) for t in 1:size(fit.Λ, 1)]
end
