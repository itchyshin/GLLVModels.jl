# Generic Laplace-approximated marginal log-likelihood for non-Gaussian GLLVM
# families. The family-specific pieces — the Fisher-scoring score and weight, the
# μ clamp, and the conditional log-density — dispatch on the Distributions family
# type, so Binomial, Poisson, … share one mode-finder (no hardcoded family switch).
#
# Model (site s): y_{ts} ~ Family(μ_{ts}[, n_{ts}]),  μ = linkinv(link, η),
#     η = β_t + (Λ z_s)_t,  z_s ~ N(0, I_K).
# The marginal ∫ p(y_s|z) N(z;0,I) dz (non-conjugate) is computed by Laplace:
# find the conditional mode ẑ by Fisher scoring (expected Hessian ⇒ Λ'WΛ + I
# is always SPD), then  log p(y_s) ≈ ℓ(ẑ) − ½ẑ'ẑ − ½ logdet(Λ'WΛ + I).
#
# Each family provides, dispatched on its type:
#   _clamp_mu(family, μ)              domain-safe μ
#   _glm_score(family, μ, n, me, y)   ∂ℓ/∂η contribution (score)
#   _glm_weight(family, μ, n, me)     Fisher information wrt η (≥ 0)
#   _glm_logpdf(family, μ, n, y)      conditional log-density
# (see families/binomial.jl, families/poisson.jl).

# η clamp is family-agnostic; μ clamp dispatches on the family.
_clamp_eta(η) = clamp(η, -30.0, 30.0)

# Robust linear solve: returns `nothing` if the factorization is singular or
# fails, so the inner Newton can stop gracefully. A = Λ'WΛ + I is SPD by
# construction but can be numerically singular when the Fisher weights blow up
# (huge μ at the η clamp — e.g. a Poisson rate driven to exp(30)).
_safe_solve(A, b) = try
    A \ b
catch
    nothing
end

_laplace_mode_should_backtrack(family) = false
_laplace_mode_should_backtrack(family::Union{
    Poisson, Binomial, NegativeBinomial, Beta, Gamma, Exponential,
}) = true
# NB1 and TweedieED also opt in — declared in their own family files (they
# are defined AFTER this one in the include order; 2026-08-27 audit rider:
# the grouped solver's backtrack gate silently no-opped for them).

function _laplace_mode_logpost(family, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link, z::AbstractVector;
        mask = nothing, offset = nothing)
    p = size(Λ, 1)
    off = offset === nothing ? false : offset
    η = _clamp_eta.(β .+ off .+ Λ * z)
    μ = _clamp_mu.(Ref(family), linkinv.(Ref(link), η))
    q = -0.5 * dot(z, z)
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        q += _glm_logpdf(family, μ[t], n[t], y[t])
    end
    return q
end

# R3 workspace (core070 perf repair): the nine buffers `_laplace_mode` needs
# (Λz, η, μ, me, s, W, WΛ, Amat, g) were allocated fresh on EVERY call — one call
# per site per objective/gradient evaluation, measured at 543MB churn for a single
# p=50 gradient (docs/dev-log/core070/poisson-perf-diagnosis.md). `LaplaceModeWorkspace`
# lets a caller that loops over many sites at a FIXED (p, K, T) — e.g. the per-site
# loop inside `marginal_loglik_laplace`/`laplace_loglik_site`, or the R2 mode-hoist
# loop in `poisson_laplace_grad` — allocate the buffers ONCE and reuse them across
# sites. `_laplace_mode` still allocates internally when `ws === nothing` (the
# default) or when its element type/shape doesn't match, so every OTHER call site
# (postfit.jl, cv.jl, the AGHQ/mixed/grouped-dispersion families, …) is unaffected —
# this is purely opt-in. Only the CONCRETE (Float64-ish) solve is ever expected to
# reuse a workspace; a `Dual`-typed call simply falls through to fresh allocation.
"""
    LaplaceModeWorkspace{T}

Reusable buffers for [`_laplace_mode`](@ref) at a fixed `(p, K)` and element type
`T`, avoiding the per-call allocation of its nine internal arrays when the SAME
caller invokes it many times in a row (e.g. once per site in a marginal-likelihood
or gradient loop). Construct with `LaplaceModeWorkspace(T, p, K)` and pass as the
`ws` keyword; a size/type mismatch is ignored (falls back to fresh allocation), so
passing the wrong workspace is safe, just non-optimal.
"""
struct LaplaceModeWorkspace{T}
    Λz::Vector{T}
    η::Vector{T}
    μ::Vector{T}
    me::Vector{T}
    s::Vector{T}
    W::Vector{T}
    WΛ::Matrix{T}
    Amat::Matrix{T}
    g::Vector{T}
end

LaplaceModeWorkspace(::Type{T}, p::Integer, K::Integer) where {T} =
    LaplaceModeWorkspace{T}(Vector{T}(undef, p), Vector{T}(undef, p), Vector{T}(undef, p),
                            Vector{T}(undef, p), Vector{T}(undef, p), Vector{T}(undef, p),
                            Matrix{T}(undef, p, K), Matrix{T}(undef, K, K), Vector{T}(undef, K))

# Inner Laplace mode-finder (Fisher-scoring Newton). Returns the conditional mode
# ẑ (length K) for one site. Shared across families and by getLV (src/postfit.jl).
# `mask` (length-p Bool, or `nothing` = all observed) drops missing responses: a
# masked entry contributes zero score and zero Fisher weight, so it neither pulls
# the mode nor enters the Hessian — exactly the marginal over the observed cells.
# `ws` (optional `LaplaceModeWorkspace`, R3 core070): reuse pre-allocated buffers
# instead of allocating fresh ones; ignored (fresh allocation) on any type/size
# mismatch, so it is always safe to pass.
function _laplace_mode(family, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9,
        ws = nothing)
    p = size(Λ, 1)
    K = size(Λ, 2)
    off = offset === nothing ? false : offset    # additive identity ⇒ no-offset path unchanged
    T = promote_type(Base.nonmissingtype(eltype(y)), eltype(n), eltype(Λ), eltype(β))
    offset === nothing || (T = promote_type(T, Base.nonmissingtype(eltype(offset))))
    z = zeros(T, K)
    # Per-call buffers, reused across Newton iterations. Each is written in place
    # with the SAME broadcast / BLAS expression as the allocating version, so the
    # computed values and FP-operation order are bit-identical.
    Λz, η, μ, me, s, W, WΛ, Amat, g =
        if ws isa LaplaceModeWorkspace{T} && length(ws.Λz) == p && length(ws.g) == K
            ws.Λz, ws.η, ws.μ, ws.me, ws.s, ws.W, ws.WΛ, ws.Amat, ws.g
        else
            Vector{T}(undef, p), Vector{T}(undef, p), Vector{T}(undef, p),
            Vector{T}(undef, p), Vector{T}(undef, p), Vector{T}(undef, p),
            Matrix{T}(undef, p, K), Matrix{T}(undef, K, K), Vector{T}(undef, K)
        end
    restarted = false
    for _ in 1:maxiter
        mul!(Λz, Λ, z)
        η  .= _clamp_eta.(β .+ off .+ Λz)
        μ  .= _clamp_mu.(Ref(family), linkinv.(Ref(link), η))
        me .= mu_eta.(Ref(link), η)
        s  .= _glm_score.(Ref(family), μ, n, me, y)
        W  .= _glm_weight.(Ref(family), μ, n, me)
        if mask !== nothing
            s .= ifelse.(mask, s, zero(T))        # masked ⇒ no contribution (NaN safe)
            W .= ifelse.(mask, W, zero(T))
        end
        WΛ .= W .* Λ                          # = W .* Λ (p×K)
        mul!(Amat, Λ', WΛ)                     # = Λ' * (W .* Λ)
        @inbounds for d in 1:K
            Amat[d, d] += one(T)              # + I (adds 1 to each diagonal entry)
        end
        A  = Symmetric(Amat)
        mul!(g, Λ', s)                         # = Λ' * s
        g .= g .- z                           # rhs = Λ's − z
        Δ  = _safe_solve(A, g)
        if Δ === nothing || !all(isfinite, Δ)
            if !restarted
                fill!(z, zero(T))
                restarted = true
                continue
            end
            break
        end

        step_taken = 1.0
        if norm(Δ) <= 1e-3 * (1 + norm(z))
            z = z .+ Δ
        elseif !_laplace_mode_should_backtrack(family)
            z = z .+ Δ
        else
            q0 = _laplace_mode_logpost(family, y, n, Λ, β, link, z;
                                       mask = mask, offset = offset)
            if isfinite(q0)
                accepted = false
                step = 1.0
                @inbounds for _half in 1:30
                    ztrial = z .+ step .* Δ
                    q1 = _laplace_mode_logpost(family, y, n, Λ, β, link, ztrial;
                                               mask = mask, offset = offset)
                    if isfinite(q1) && q1 >= q0
                        z = ztrial
                        step_taken = step
                        accepted = true
                        break
                    end
                    step *= 0.5
                end
                accepted || break
            else
                z = z .+ Δ
            end
        end
        step_taken * maximum(abs, Δ) < tol && break
    end
    return z
end

# ===========================================================================
# Curvature role separation (2026-08-25).
#
# `_glm_weight` (documented at the top of this file) is the FISHER (expected)
# information wrt η. It plays TWO roles here, and they have different
# requirements:
#
#   (a) the Fisher-scoring MODE SEARCH (`_laplace_mode`) — expected information
#       is correct here. It solves the same score equation, so the mode is
#       unchanged, and W ≥ 0 keeps Λ'WΛ + I SPD inside Newton.
#   (b) the marginal's LOG-DET — this must be the OBSERVED curvature
#       −∂²ℓ/∂η² to match TMB, which obtains it structurally because
#       `MakeADFun(..., random=)` differentiates the coded joint nll and so
#       never faces this choice.
#
# Conflating the two is a confirmed fault class (see
# `docs/dev-log/plans/2026-08-25-laplace-structural-design.md`).
#
# SCOPE, STATED HONESTLY: this selector reaches THIS kernel only. Other kernels
# build their own Λ'WΛ + I and their own logdet — `grouped_dispersion.jl`,
# `covariates.jl`, `quadratic.jl`, `mixed.jl`, `spde_latent.jl`,
# `aghq_grid.jl`, `phylo_glm.jl`, the `phylo_*_xlv.jl` family, and
# `coevolution_glm.jl`. It is NOT an anti-recurrence guarantee, and must not be
# described as closing the class.

"""
    _glm_weight_matches_observed(family, link) -> Bool

`true` when the existing `_glm_weight` slot already yields the correct log-det
curvature at this `(family, link)` — either because observed ≡ Fisher pointwise
(canonical link, y-free curvature) or because the slot is hand-coded observed.

Trait-true families take the branch containing the UNTOUCHED original code, so
they are bit-for-bit identical under either `hessian` setting. Each family
declares its own method in its own file (this file is included first).
"""
_glm_weight_matches_observed(family, link::Link) = false

"""
    _default_hessian(family, link) -> Symbol

Which curvature the log-det uses when the caller does not choose.

Currently `:fisher` — preserving shipped behaviour exactly. Flipping this to
`:observed` is a separate, deliberate change that must land together with the
coupled analytic-gradient paths in `src/laplace_grad.jl` (NB2 `:156`, Gamma
`:221-222`, Beta `:302-303`); otherwise those gradients stop being the gradient
of the objective and degrade SILENTLY rather than erroring.
"""
_default_hessian(family, link::Link) = :fisher

"""
    _glm_obs_weight(family, μ, n, me, y, link, η) -> −∂²ℓ(y|η)/∂η²

Observed conditional curvature wrt the linear predictor at one cell — the
log-det weight TMB's Laplace uses.

**May be NEGATIVE** (Student-t for |r| > σ√ν; GP-1 where `1 + 2αy − αμ < 0`).
That is not an error: the positive-definiteness guard belongs at the
`Λ'WΛ + I` assembly, never here. Do NOT clamp this to zero — `ordinal.jl`'s
`max(·, 0)` is safe only for log-concave links and would silently diverge from
TMB elsewhere.

Default: nested `ForwardDiff` through the CODED conditional log-density, μ-clamp
included — i.e. the derivative of the function the objective actually sums,
which is TMB's semantics (it differentiates its coded nll, guards and all).

CONVENTION NOTE: where `_clamp_mu` binds, this returns the derivative of the
clamped composition (zero in the saturated region), whereas a hand-derived
analytic override evaluates the unclamped formula at the clamped μ. The two
therefore differ AT THE CLAMP BOUNDARY and agree in the interior. This is a
deliberate choice — the fallback is faithful to the coded objective — so any
override-vs-fallback gate test must be restricted to interior cells.
"""
function _glm_obs_weight(family, μ, n, me, y, link::Link, η)
    f = ηv -> _glm_logpdf(family, _clamp_mu(family, linkinv(link, ηv)), n, y)
    g = ηv -> ForwardDiff.derivative(f, ηv)
    return -ForwardDiff.derivative(g, η)
end

"""
    laplace_loglik_site(family, y, n, Λ, β, link; mask=nothing, maxiter=100, tol=1e-9) -> Float64

Laplace-approximated log-marginal for one site of a non-Gaussian GLLVM. `family`
is a `Distributions` family marker (e.g. `Binomial()`, `Poisson()`); `y`, `n` are
the response and trial counts (length p; `n` is ignored by families without
trials); `Λ` p×K; `β` length-p; `link` a `Link`. `mask` (length-p Bool, or
`nothing`) marks observed responses — masked-out (missing) entries are dropped from
the score, the Hessian weight, and the log-density sum. Returns
`ℓ(ẑ) − ½ẑ'ẑ − ½logdet(Λ'WΛ + I)`.
"""
function laplace_loglik_site(family, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, hessian::Symbol = _default_hessian(family, link),
        maxiter::Integer = 100, tol::Real = 1e-9, ws = nothing, z_precomputed = nothing)
    (hessian === :fisher || hessian === :observed) || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    p = size(Λ, 1)
    K = size(Λ, 2)
    off = offset === nothing ? false : offset
    # R8 (shared mode solve, S6 item 1): `z_precomputed` (length K), when given,
    # skips the Newton mode-finder here — the caller (e.g. `fg!` in
    # `_fit_poisson_gllvm_laplace`) already solved it at this θ for the
    # gradient path and hands the SAME concrete mode to the value path,
    # instead of each path solving its own copy of the identical fixed point.
    # A length mismatch falls back to solving fresh (never silently wrong).
    z  = if z_precomputed !== nothing && length(z_precomputed) == K
        z_precomputed
    else
        _laplace_mode(family, y, n, Λ, β, link;
                     mask = mask, offset = offset, maxiter = maxiter, tol = tol, ws = ws)
    end
    # Per-call buffers (written in place with the SAME broadcast / BLAS expressions
    # as before ⇒ bit-identical values and FP-operation order).
    Λz = Λ * z                                # Λ*z (one-shot; result reused below)
    η  = _clamp_eta.(β .+ off .+ Λz)          # clamped linear predictor
    μ  = _clamp_mu.(Ref(family), linkinv.(Ref(link), η))  # clamped mean
    me = mu_eta.(Ref(link), η)                # dμ/dη
    # Role (b): the log-det weight. The `:fisher` arm is the ORIGINAL expression,
    # verbatim, so the default path is bit-for-bit unchanged. Trait-true families
    # take it under `:observed` too, because there the two coincide pointwise.
    W  = if hessian === :fisher || _glm_weight_matches_observed(family, link)
        _glm_weight.(Ref(family), μ, n, me)  # Fisher weight wrt η
    else
        # Masked cells carry a PLACEHOLDER response. `_glm_weight` never reads
        # `y`, so placeholders were harmless; the observed weight DOES read it,
        # so it must not be evaluated there — a placeholder can make
        # `_glm_logpdf` throw outright (y = 0 for a zero-truncated family,
        # a non-integral value under `Int(y)`), which masking W afterwards
        # cannot undo. This honours the docstring promise below that the value
        # is invariant to whatever sits in the masked cells of `Y`.
        [(mask === nothing || mask[t]) ?
            _glm_obs_weight(family, μ[t], n[t], me[t], y[t], link, η[t]) : 0.0
         for t in 1:p]
    end
    if mask !== nothing
        W = ifelse.(mask, W, 0.0)
    end
    WΛ = W .* Λ                               # = W .* Λ (p×K)
    Amat = Λ' * WΛ                            # = Λ' * (W .* Λ) (K×K)
    @inbounds for d in 1:K
        Amat[d, d] += 1.0                     # + I (adds 1.0 to each diagonal entry)
    end
    A  = Symmetric(Amat)
    ℓ = 0.0
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _glm_logpdf(family, μ[t], n[t], y[t])
    end
    # PD guard, keyed on the WEIGHT'S SIGN — not on the selector and not on the
    # trait. `A = Λ'WΛ + I` is SPD by construction whenever every `W ≥ 0`, so
    # the factorisation is needed only when a negative weight is actually
    # present. Keying on the sign is both cheaper (the common path skips it
    # entirely) and strictly more correct than keying on a proxy: it fires on
    # the real condition however the weight was produced.
    #
    # Observed curvature is genuinely negative for Beta, Student-t and GP-1
    # (measured, not assumed) — so this is load-bearing, not defensive. The
    # guard lives HERE, at the assembly, never as a clamp on the weight:
    # `ordinal.jl`'s `max(·, 0)` is safe only for log-concave links and would
    # silently diverge from TMB elsewhere.
    #
    # `-Inf` is returned via `oftype(ℓ, …)` so the value carries the caller's
    # numeric type: fitters run ForwardDiff OVER this objective, and a raw
    # `Float64` returned into a `Dual` context is a type error waiting to
    # happen.
    if any(w -> w < zero(w), W)
        F = cholesky(A; check = false)
        issuccess(F) || return oftype(ℓ, -Inf)
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

"""
    marginal_loglik_laplace(family, Y, N, Λ, β, link; mask=nothing, offset=nothing, kwargs...) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a non-Gaussian GLLVM.
`Y`, `N` are p×n response and trial-count matrices. `mask` (p×n Bool, or `nothing`)
marks observed cells — missing responses (`mask` false) are dropped per site, so the
marginal is over the observed entries only (gllvm-style NA handling). The value is
invariant to whatever placeholder sits in the masked cells of `Y`.

`offset` (p×n, or `nothing`) is a known additive term in the linear predictor
`η = β + offset + Λz` (e.g. log-exposure/effort/area for counts). A constant
per-species offset is equivalent to shifting that species' intercept (the
offset-absorption identity), which serves as the exact verification anchor.

`zs` (optional `Vector` of length-K per-site modes, R8 shared mode solve): when
given, threads `zs[i]` into site `i` as `laplace_loglik_site`'s `z_precomputed`,
skipping that site's own Newton mode solve — see `laplace_loglik_site`.
"""
function marginal_loglik_laplace(family, Y::AbstractMatrix, N::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, zs = nothing, kwargs...)
    acc = 0.0
    @inbounds for i in axes(Y, 2)
        mi = mask   === nothing ? nothing : view(mask, :, i)
        oi = offset === nothing ? nothing : view(offset, :, i)
        zi = zs     === nothing ? nothing : zs[i]
        acc += laplace_loglik_site(family, view(Y, :, i), view(N, :, i), Λ, β, link;
                                   mask = mi, offset = oi, z_precomputed = zi, kwargs...)
    end
    return acc
end

"""
    observed_mask(Y) -> BitMatrix

Observation mask for a response matrix that may contain `missing`: `true` where the
entry is observed, `false` where missing. Pass the result as the `mask` keyword to
`marginal_loglik_laplace` / the family fitters for gllvm-style NA handling.
"""
observed_mask(Y::AbstractMatrix) = .!ismissing.(Y)

# Replace `missing` with a domain-safe placeholder so the family pieces never see a
# `missing` (the placeholder cells are masked out of every contribution anyway).
function _sanitize_missing(Y::AbstractMatrix, placeholder)
    any(ismissing, Y) || return Y
    return map(y -> ismissing(y) ? placeholder : y, Y)
end

# Resolve an observation mask: an explicit `mask` wins; otherwise derive it from
# `missing` entries in `Y` (or `nothing` when `Y` is fully observed).
_resolve_obs_mask(mask, Y) =
    mask === nothing ? (any(ismissing, Y) ? observed_mask(Y) : nothing) : mask

# Mask-respecting warm start: overwrite the masked cells of a link-scale empirical
# matrix `Zemp` with their row's observed mean, so the intercept and SVD loadings
# warm start ignore missing (and placeholder) values — the fit then depends only on
# the observed cells.
function _mask_warmstart!(Zemp::AbstractMatrix, msk)
    msk === nothing && return Zemp
    p, n = size(Zemp)
    @inbounds for t in 1:p
        cnt = count(view(msk, t, :))
        rowmean = cnt > 0 ? sum(Zemp[t, i] for i in 1:n if msk[t, i]) / cnt : 0.0
        for i in 1:n
            msk[t, i] || (Zemp[t, i] = rowmean)
        end
    end
    return Zemp
end
