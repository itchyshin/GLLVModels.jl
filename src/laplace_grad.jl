# Analytic (exact) gradient of the Poisson Laplace marginal — now the default
# gradient of the non-Gaussian fitters, replacing the finite-difference gradient.
#
# The per-site Laplace marginal is  L_s = ℓ(ẑ) − ½ẑ'ẑ − ½ logdet A(ẑ),  with
# A = Λ'WΛ + I and ẑ the conditional mode solving g(z) = Λ's(z) − z = 0. A naive
# `ForwardDiff` through the marginal fails because the inner Newton mode-finder is
# not AD-friendly, and a hand-derived adjoint must carry the implicit dẑ/dθ through
# the log-det term (error-prone).
#
# Instead we use the implicit-function "one Newton step at the optimum" trick: find
# the mode concretely (non-differentiated), then form
#       z(θ) = ẑ + A(ẑ,θ)⁻¹ (Λ's(ẑ;θ) − ẑ),
# which equals ẑ at θ̂ (the bracket is ≈0) but whose θ-derivative is exactly the
# implicit dẑ/dθ. Evaluating L at this differentiable `z` and applying ForwardDiff
# yields the EXACT total gradient — including the log-det and implicit terms — at the
# cost of one Newton solve plus one AD pass, versus the ~2·nθ marginal evaluations a
# finite-difference gradient needs.
#
# This is the analytic-gradient lever from issue #65 (Poisson first, then NB / Gamma
# / Beta / Binomial below). Each gradient is finite-difference-verified and is now the
# DEFAULT gradient of its production fitter (`gradient = :analytic` in `fit_*_gllvm`),
# with a central finite-difference fallback for any θ where the analytic value is
# non-finite, and an automatic fall-back to `autodiff = :finite` when a response mask
# or offset is present. The analytic-vs-finite-difference agreement of the fitted
# optimum is gated by `test/test_laplace_grad.jl`. Each family needs only an
# AD-friendly log-pmf/pdf (the score/weight are arithmetic).

# AD-friendly Poisson log-pmf (avoids Distributions' logpdf(::Poisson, ::Int) under a
# Dual mean). The lgamma(y+1) term is a constant in θ.
_pois_logpmf(μ, y) = y * log(μ) - μ - loggamma(y + 1.0)

# Shared optimiser wrapper: drive L-BFGS with an analytic gradient closure
# `analytic_grad(θ) -> Vector | nothing`, falling back to a central finite-difference
# gradient of `negll` for any θ where the analytic value is missing/non-finite. The
# fitters pass an `analytic_grad` that returns −∇(marginal) packing matched to θ.
function _optimize_with_analytic(negll, analytic_grad, θ0, ls, opts)
    function g!(G, θ)
        gg = analytic_grad(θ)
        if gg === nothing || !all(isfinite, gg)
            hh = 1e-6
            θp = copy(θ); θm = copy(θ)            # reused across indices (no per-i copy)
            @inbounds for i in eachindex(θ)
                θp[i] += hh; θm[i] -= hh
                G[i] = (negll(θp) - negll(θm)) / (2hh)
                θp[i] = θ[i]; θm[i] = θ[i]        # restore in place
            end
        else
            G .= gg
        end
        return G
    end
    return Optim.optimize(negll, g!, θ0, ls, opts)
end

# R6 (allocation-free per-site kernel, S6 item 3): scratch buffers for
# `_poisson_site_diffable` at a fixed (element type T, p, K), avoiding the
# nine-ish fresh p- or (p×K)-sized allocations the naive broadcasted version
# made on EVERY site call (η, μ, s, W, the two masked `ifelse.()` copies, WΛ,
# the `Λ'*s` product, …) — measured at 383 MB of Dual allocations for one
# p=50 gradient call (13 chunk passes × n=500 sites), see
# docs/dev-log/core070/poisson-perf-diagnosis.md. `T` is a ForwardDiff dual
# type in production (`_poisson_site_diffable` is only ever called from
# inside `poisson_laplace_grad`'s `marg` closure below, differentiated by
# `ForwardDiff.gradient`); the K×K solve/logdet stay on the small generic
# path (their cost is negligible at the K this package ever fits with).
struct PoissonSiteWorkspace{T}
    η::Vector{T}
    μ::Vector{T}
    s::Vector{T}
    W::Vector{T}
    Λz::Vector{T}
    WΛ::Matrix{T}
    Amat::Matrix{T}
end

PoissonSiteWorkspace(::Type{T}, p::Integer, K::Integer) where {T} =
    PoissonSiteWorkspace{T}(Vector{T}(undef, p), Vector{T}(undef, p), Vector{T}(undef, p),
                            Vector{T}(undef, p), Vector{T}(undef, p),
                            Matrix{T}(undef, p, K), Matrix{T}(undef, K, K))

# Differentiable per-site Poisson Laplace marginal (log link), via the implicit step.
# `β`, `Λ` may carry ForwardDiff duals; the mode is computed on their primal values.
# `mask` (length-p Bool, or `nothing` = all observed) drops unobserved responses:
# a masked cell adds zero score and zero Fisher weight (so it neither moves the mode
# nor enters the log-det Hessian A) and is skipped in the log-pmf sum — exactly the
# masking semantics of the core (src/families/laplace.jl), so the analytic gradient
# matches the masked marginal. `ws` (optional `PoissonSiteWorkspace`, R6): reuse
# pre-allocated buffers instead of allocating fresh ones; ignored (fresh allocation,
# byte-identical numerics) on any type/size mismatch, so it is always safe to pass —
# mirrors `LaplaceModeWorkspace`'s contract (src/families/laplace.jl).
function _poisson_site_diffable(y::AbstractVector, Λ::AbstractMatrix, β::AbstractVector,
                                ẑ::AbstractVector; mask = nothing, ws = nothing)
    p, K = size(Λ)
    T = promote_type(eltype(Λ), eltype(β), eltype(ẑ))
    η, μ, s, W, Λz, WΛ, Amat =
        if ws isa PoissonSiteWorkspace{T} && length(ws.η) == p && size(ws.WΛ, 2) == K
            ws.η, ws.μ, ws.s, ws.W, ws.Λz, ws.WΛ, ws.Amat
        else
            Vector{T}(undef, p), Vector{T}(undef, p), Vector{T}(undef, p), Vector{T}(undef, p),
            Vector{T}(undef, p), Matrix{T}(undef, p, K), Matrix{T}(undef, K, K)
        end
    # `ẑ` is the concrete mode at the primal (θ̂) parameters, precomputed ONCE by the
    # caller outside the ForwardDiff chunk loop (R2: mode-solve hoist, core070). Every
    # chunk pass of `ForwardDiff.gradient` below re-evaluates this function at duals
    # whose PRIMAL part is the same θ̂, so re-solving the Newton mode here on every
    # chunk (as the previous version did via `ForwardDiff.value.(Λ)`) was redundant
    # work repeated ⌈nθ/chunksize⌉ times per gradient call — see
    # docs/dev-log/core070/poisson-perf-diagnosis.md.

    # One differentiable Newton step from ẑ ⇒ z ≈ ẑ with the correct dz/dθ.
    mul!(Λz, Λ, ẑ)
    @. η = _clamp_eta(β + Λz)
    @. μ = exp(η)                     # log link
    @. s = y - μ                      # Poisson/log score wrt η
    W .= μ                            # Fisher weight wrt η
    if mask !== nothing
        @. s = ifelse(mask, s, zero(T))
        @. W = ifelse(mask, W, zero(T))
    end
    @. WΛ = W * Λ
    mul!(Amat, Λ', WΛ)
    @inbounds for d in 1:K
        Amat[d, d] += one(T)
    end
    rhs = Λ' * s                      # length K: too small to be worth workspace-caching
    rhs .-= ẑ
    z = ẑ .+ (Amat \ rhs)             # plain Matrix (AD-safe generic solve)

    # Marginal evaluated at the differentiable mode.
    mul!(Λz, Λ, z)
    @. η = _clamp_eta(β + Λz)
    @. μ = exp(η)
    if mask !== nothing
        @. W = ifelse(mask, μ, zero(T))
    else
        W .= μ
    end
    @. WΛ = W * Λ
    mul!(Amat, Λ', WΛ)
    @inbounds for d in 1:K
        Amat[d, d] += one(T)
    end
    ℓ = zero(eltype(z))
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _pois_logpmf(μ[t], y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(Amat)
end

# R7 (one GradientConfig per fit, S6 item 2): a STABLE callable-struct type for
# the summed per-site marginal, so a `ForwardDiff.GradientConfig` built against
# one instance of this type remains valid (same Tag TYPE) for every OTHER
# instance built during the SAME fit — unlike a `function marg(θ) ... end`
# closure, whose anonymous type is fresh on every `poisson_laplace_grad` call
# and so could never share a config across Optim iterations. Only `Y`, `p`,
# `rr`, `K` and `mask`'s PRESENCE (not its values) are fixed within one fit;
# `ẑs` (the per-iteration mode hoist) and the site-workspace cache are fields
# that legitimately change value call to call without changing the type.
struct PoissonMargClosure{TY <: AbstractMatrix, TM}
    Y::TY
    p::Int
    rr::Int
    K::Int
    ẑs::Vector{Vector{Float64}}
    mask::TM
    ws::Base.RefValue{Any}
end

function (c::PoissonMargClosure)(θ)
    b = θ[1:c.p]
    L = unpack_lambda(θ[(c.p + 1):(c.p + c.rr)], c.p, c.K)
    Tθ = eltype(θ)
    cur = c.ws[]
    sws = if cur isa PoissonSiteWorkspace{Tθ}
        cur
    else
        fresh = PoissonSiteWorkspace(Tθ, c.p, c.K)
        c.ws[] = fresh
        fresh
    end
    acc = zero(Tθ)
    @inbounds for s in axes(c.Y, 2)
        mi = c.mask === nothing ? nothing : view(c.mask, :, s)
        acc += _poisson_site_diffable(view(c.Y, :, s), L, b, c.ẑs[s]; mask = mi, ws = sws)
    end
    return acc
end

function _poisson_hoist_zhats(Y::AbstractMatrix, Λv::AbstractMatrix, βv::AbstractVector;
                              mask = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λv)
    # R3 (workspace reuse, core070): one Float64 workspace shared across all n site
    # mode solves in this hoist loop (concrete solve only — see LaplaceModeWorkspace).
    ws = LaplaceModeWorkspace(Float64, p, K)
    ẑs = Vector{Vector{Float64}}(undef, size(Y, 2))
    Nunit = ones(Int, p)
    @inbounds for s in axes(Y, 2)
        mi = mask === nothing ? nothing : view(mask, :, s)
        ẑs[s] = _laplace_mode(Poisson(), view(Y, :, s), Nunit, Λv, βv, LogLink();
                              mask = mi, maxiter = maxiter, tol = tol, ws = ws)
    end
    return ẑs
end

"""
    poisson_laplace_grad(Y, Λ, β; mask=nothing, gcfg=nothing) -> Vector

Exact gradient of the total Poisson Laplace marginal log-likelihood
([`poisson_marginal_loglik_laplace`](@ref)) with respect to the packed parameter
vector `θ = [β; pack_lambda(Λ)]`, computed by ForwardDiff through the
implicit-function "one Newton step at the optimum" construction (see file header).

`Y` is the p×n count matrix, `Λ` p×K loadings, `β` length-p intercepts. `mask`
(p×n Bool, or `nothing` = all observed) drops unobserved responses per site, so the
gradient is of the masked marginal over the observed cells only. The result matches a
finite-difference gradient of the (masked) marginal to ~AD precision, at a fraction
of the cost. This is the default gradient of `fit_poisson_gllvm`
(`gradient = :analytic`); a masked or offset fit falls back to `autodiff = :finite`.

`gcfg` (optional `ForwardDiff.GradientConfig`, R7): reuse a config built once per
fit (see `_fit_poisson_gllvm_laplace`) instead of `ForwardDiff.gradient` building
a fresh one — with its own seed/partial buffers — on every Optim iteration. Any
mismatch (wrong tag type, wrong length) is caught and falls back to the default,
un-cached `ForwardDiff.gradient(marg, θ̂)` path, so passing an incompatible or
stale config is always safe, just non-optimal.

`ẑs` (optional `Vector{Vector{Float64}}`, length `size(Y,2)`, R8 shared mode
solve): precomputed per-site modes at the ROUND-TRIPPED `(unpack_lambda(pack_lambda(Λ)), β)`
point (see `_poisson_hoist_zhats`) — when given (and the right length), skips
this function's own hoist loop, so a caller that already solved the SAME modes
for the value path (e.g. `fg!`'s F+G branch) does not pay for a second, identical
Newton solve per site. A length mismatch falls back to hoisting fresh.
"""
function poisson_laplace_grad(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector;
                              mask = nothing, gcfg = nothing, ẑs = nothing)
    p, K = size(Λ)
    rr = rr_theta_len(p, K)
    θ̂ = vcat(float.(β), pack_lambda(Λ))
    # R2 (mode-solve hoist, core070): solve the concrete per-site mode ONCE here, at
    # the primal (θ̂) parameters, instead of inside `_poisson_site_diffable` — which
    # `ForwardDiff.gradient` below would otherwise call once per CHUNK (chunk size 12
    # ⇒ ⌈nθ/12⌉ redundant Newton solves per gradient call, all at the identical
    # primal point). See docs/dev-log/core070/poisson-perf-diagnosis.md.
    #
    # Must use the ROUND-TRIPPED Λ/β (unpack_lambda(pack_lambda(Λ)), the θ̂ split),
    # not the raw `Λ`/`β` arguments — `unpack_lambda` enforces the lower-triangular
    # convention (packing.jl) and zeros any strict-upper entries, so a raw Λ with
    # nonzero upper-triangle differs from what `marg` below actually reconstructs
    # from θ̂ at the same point. Using the raw matrices here silently solved the
    # mode for the WRONG Λ (caught by the FD gate in test_poisson_grad_perf.jl).
    βv = θ̂[1:p]
    Λv = unpack_lambda(θ̂[(p + 1):(p + rr)], p, K)
    if ẑs === nothing || length(ẑs) != size(Y, 2)
        ẑs = _poisson_hoist_zhats(Y, Λv, βv; mask = mask)
    end
    # R6 (allocation-free per-site kernel, S6 item 3): one `PoissonSiteWorkspace`
    # per distinct element type `marg` is called with, built lazily on first use
    # and cached in a `Ref` LOCAL to this call's `marg` instance (captured as a
    # struct field, not a package-level global) — safe under `Threads.@threads`
    # bootstrap replicates (src/confint_family.jl:2853), each of which builds its
    # own `PoissonMargClosure`/cache. Every site call inside ONE
    # `ForwardDiff.gradient` invocation shares the SAME dual type (one Tag, one
    # chunk width per call), so this allocates the workspace once per
    # `poisson_laplace_grad` call rather than once per site per chunk pass
    # (n × ⌈nθ/chunksize⌉ before this change).
    marg = PoissonMargClosure(Y, p, rr, K, ẑs, mask, Ref{Any}(nothing))
    if gcfg !== nothing
        g = try
            ForwardDiff.gradient(marg, θ̂, gcfg)
        catch
            ForwardDiff.gradient(marg, θ̂)
        end
        return g
    end
    return ForwardDiff.gradient(marg, θ̂)
end

# --- Negative binomial (log link) — dispersion family: r enters θ as log r ---
# AD-friendly NB2 log-pmf kernel (mean μ, dispersion r); the loggamma(y+1) constant
# is dropped (zero gradient). r is a θ parameter (via log r), so the r-dependent
# loggamma(r+y) − loggamma(r) terms are kept and differentiated (loggamma' = digamma,
# which ForwardDiff supports).
_nb_logker(μ, r, y) = loggamma(r + y) - loggamma(r) +
                      r * log(r) - (r + y) * log(r + μ) + y * log(μ)

function _nb_site_diffable(y::AbstractVector, Λ::AbstractMatrix, β::AbstractVector, logr;
                           mask = nothing)
    p = size(Λ, 1)
    r = exp(logr)
    Λv = ForwardDiff.value.(Λ); βv = ForwardDiff.value.(β); rv = ForwardDiff.value(r)
    ẑ = _laplace_mode(NegativeBinomial(rv, 0.5), y, ones(Int, p), Λv, βv, LogLink(); mask = mask)

    η = _clamp_eta.(β .+ Λ * ẑ); μ = exp.(η)
    s = (y .- μ) ./ (1 .+ μ ./ r)             # NB2/log score (= r(y−μ)/(r+μ))
    # NB is non-canonical: the implicit dẑ/dθ uses the OBSERVED Hessian weight
    # W_obs = −∂s/∂η = μr(r+y)/(r+μ)², not the Fisher weight. (For canonical links
    # the two coincide, which is why Poisson/Binomial work with the Fisher weight.)
    Wobs = μ .* r .* (r .+ y) ./ (r .+ μ) .^ 2
    if mask !== nothing                       # masked cell: zero score + weights, skip log-pmf
        s = ifelse.(mask, s, zero(eltype(s)))
        Wobs = ifelse.(mask, Wobs, zero(eltype(Wobs)))
    end
    Aobs = Λ' * (Wobs .* Λ) + I
    z = ẑ .+ (Aobs \ (Λ' * s .- ẑ))

    ηz = _clamp_eta.(β .+ Λ * z); μz = exp.(ηz)
    # OBSERVED weight μr(r+y)/(r+μ)² per observed cell ⇒ logdet term.
    # CHANGED 2026-08-27 together with `_default_hessian(::NegativeBinomial, ::LogLink)`.
    # Previously the Fisher weight μ/(1+μ/r), deliberately matching the
    # objective's then-Fisher log-det. The objective now uses the observed
    # curvature, so this must too — an analytic gradient tuned to a different
    # log-det than the one being reported is not the gradient of the objective,
    # and it degrades optimisation SILENTLY rather than erroring.
    # `test_laplace_grad.jl` adjudicates against a finite difference of the
    # actual objective — no stored expected value.
    Wz = μz .* r .* (r .+ y) ./ (r .+ μz) .^ 2
    if mask !== nothing
        Wz = ifelse.(mask, Wz, zero(eltype(Wz)))
    end
    Az = Λ' * (Wz .* Λ) + I
    ℓ = zero(eltype(z))
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _nb_logker(μz[t], r, y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(Az)
end

"""
    nb_laplace_grad(Y, Λ, β, r) -> Vector

Exact gradient of the total negative-binomial (NB2, log link) Laplace marginal wrt
`θ = [β; pack_lambda(Λ); log r]` — including the dispersion direction — via the same
ForwardDiff + implicit-step construction as [`poisson_laplace_grad`](@ref). This is the
dispersion-family generalisation (r carried in θ as `log r`).
Finite-difference-verified; the default gradient of `fit_nb_gllvm`
(`gradient = :analytic`), including the dispersion direction.
"""
function nb_laplace_grad(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector, r::Real;
                         mask = nothing)
    p, K = size(Λ)
    rr = rr_theta_len(p, K)
    θ̂ = vcat(float.(β), pack_lambda(Λ), log(float(r)))
    function marg(θ)
        b = θ[1:p]
        L = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        logr = θ[p + rr + 1]
        acc = zero(eltype(θ))
        @inbounds for s in axes(Y, 2)
            mi = mask === nothing ? nothing : view(mask, :, s)
            acc += _nb_site_diffable(view(Y, :, s), L, b, logr; mask = mi)
        end
        return acc
    end
    return ForwardDiff.gradient(marg, θ̂)
end

# --- Gamma (log link, shape α) — dispersion family, non-canonical ----------
# AD-friendly Gamma(shape α, mean μ) log-density kernel (Var = μ²/α). Score
# s = α(y−μ)/μ; Fisher weight = α (constant); observed weight −∂s/∂η = αy/μ.
_gamma_logker(μ, α, y) = (α - 1) * log(y) - y * α / μ - α * log(μ) + α * log(α) - loggamma(α)

function _gamma_site_diffable(y::AbstractVector, Λ::AbstractMatrix, β::AbstractVector, logα;
                              mask = nothing)
    p = size(Λ, 1)
    α = exp(logα)
    Λv = ForwardDiff.value.(Λ); βv = ForwardDiff.value.(β); αv = ForwardDiff.value(α)
    ẑ = _laplace_mode(Gamma(αv, 1.0), y, ones(Int, p), Λv, βv, LogLink(); mask = mask)

    η = _clamp_eta.(β .+ Λ * ẑ); μ = exp.(η)
    s = α .* (y .- μ) ./ μ                    # Gamma/log score
    Wobs = α .* y ./ μ                        # observed weight −∂s/∂η (non-canonical)
    if mask !== nothing                       # masked cell: zero score + weights, skip log-pmf
        s = ifelse.(mask, s, zero(eltype(s)))
        Wobs = ifelse.(mask, Wobs, zero(eltype(Wobs)))
    end
    Aobs = Λ' * (Wobs .* Λ) + I
    z = ẑ .+ (Aobs \ (Λ' * s .- ẑ))

    ηz = _clamp_eta.(β .+ Λ * z); μz = exp.(ηz)
    # OBSERVED weight α·y/μ per observed cell ⇒ logdet term; masked cells drop.
    # CHANGED 2026-08-25 together with `_default_hessian(::Gamma, ::LogLink)`.
    # This previously used the Fisher weight α (constant), deliberately matching
    # the objective's Fisher log-det. The objective now uses the observed
    # curvature, so this must too: an analytic gradient tuned to a different
    # log-det than the one being reported is not the gradient of the objective,
    # and it degrades optimisation SILENTLY rather than erroring.
    # `test_laplace_grad.jl` adjudicates this by comparing against a finite
    # difference of the actual objective — it needs no stored expected value.
    Wobs_z = α .* y ./ μz
    Wf = mask === nothing ? Wobs_z : ifelse.(mask, Wobs_z, zero(eltype(Wobs_z)))
    Az = Λ' * (Wf .* Λ) + I
    ℓ = zero(eltype(z))
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _gamma_logker(μz[t], α, y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(Az)
end

"""
    gamma_laplace_grad(Y, Λ, β, α) -> Vector

Exact gradient of the total Gamma (log link, shape `α`) Laplace marginal wrt
`θ = [β; pack_lambda(Λ); log α]`, via the ForwardDiff + implicit-step construction
(observed weight `αy/μ` in the implicit step, Fisher weight `α` in the log-det).
Finite-difference-verified; the default gradient of `fit_gamma_gllvm` (`gradient = :analytic`).
"""
function gamma_laplace_grad(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector, α::Real;
                            mask = nothing)
    p, K = size(Λ)
    rr = rr_theta_len(p, K)
    θ̂ = vcat(float.(β), pack_lambda(Λ), log(float(α)))
    function marg(θ)
        b = θ[1:p]
        L = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        logα = θ[p + rr + 1]
        acc = zero(eltype(θ))
        @inbounds for s in axes(Y, 2)
            mi = mask === nothing ? nothing : view(mask, :, s)
            acc += _gamma_site_diffable(view(Y, :, s), L, b, logα; mask = mi)
        end
        return acc
    end
    return ForwardDiff.gradient(marg, θ̂)
end

# --- Beta (logit link, precision φ) — non-canonical; observed weight via AD ---
# Beta's observed Hessian weight (−∂s/∂η) involves digamma/trigamma terms, so rather
# than hand-derive it we take it from a 1-D ForwardDiff derivative of the per-species
# score at the (concrete) mode — exact and low-risk. The implicit-step matrix is then
# concrete (correct, since the bracket is 0 at the mode); the log-det uses the Fisher
# weight (Dual) to match the marginal.
_beta_score_scalar(η, φ, y) = begin
    μ = 1 / (1 + exp(-η)); me = μ * (1 - μ)
    ystar = log(y) - log1p(-y)
    μstar = digamma(μ * φ) - digamma((1 - μ) * φ)
    return φ * (ystar - μstar) * me
end
_beta_logker(μ, φ, y) = (μ * φ - 1) * log(y) + ((1 - μ) * φ - 1) * log1p(-y) -
                        (loggamma(μ * φ) + loggamma((1 - μ) * φ) - loggamma(φ))

function _beta_site_diffable(y::AbstractVector, Λ::AbstractMatrix, β::AbstractVector, logφ;
                             mask = nothing)
    p = size(Λ, 1)
    φ = exp(logφ)
    Λv = ForwardDiff.value.(Λ); βv = ForwardDiff.value.(β); φv = ForwardDiff.value(φ)
    ẑ = _laplace_mode(Beta(φv, 1.0), y, ones(Int, p), Λv, βv, LogitLink(); mask = mask)

    # Concrete observed-Hessian weights via AD-derivative of the scalar score.
    ηc = _clamp_eta.(βv .+ Λv * ẑ)
    Wobs = [-ForwardDiff.derivative(η_ -> _beta_score_scalar(η_, φv, y[t]), ηc[t]) for t in 1:p]
    if mask !== nothing                           # masked cell: zero weights + score, skip log-pmf
        Wobs = ifelse.(mask, Wobs, zero(eltype(Wobs)))
    end
    Aobs = Λv' * (Wobs .* Λv) + I                 # concrete (correct since bracket≈0 at ẑ)

    # Differentiable score at ẑ.
    η = _clamp_eta.(β .+ Λ * ẑ); μ = 1 ./ (1 .+ exp.(-η))
    me = μ .* (1 .- μ)
    ystar = log.(y) .- log1p.(-y)
    μstar = digamma.(μ .* φ) .- digamma.((1 .- μ) .* φ)
    s = φ .* (ystar .- μstar) .* me
    if mask !== nothing
        s = ifelse.(mask, s, zero(eltype(s)))
    end
    z = ẑ .+ (Aobs \ (Λ' * s .- ẑ))

    ηz = _clamp_eta.(β .+ Λ * z); μz = 1 ./ (1 .+ exp.(-ηz))
    mez = μz .* (1 .- μz)
    νz = trigamma.(μz .* φ) .+ trigamma.((1 .- μz) .* φ)
    # OBSERVED weight ⇒ logdet: φ²ν·me² − φ(y* − μ*)·me·(1 − 2μ).
    # CHANGED 2026-08-27 together with `_default_hessian(::Beta, ::LogitLink)`
    # (decision A). Previously the Fisher weight φ²ν·me², matching the
    # objective's then-Fisher log-det; the objective now uses the observed
    # curvature, so this must too. `test_laplace_grad.jl` adjudicates against
    # a finite difference of the actual objective — no stored expected value.
    ystarz = log.(y) .- log1p.(-y)
    μstarz = digamma.(μz .* φ) .- digamma.((1 .- μz) .* φ)
    WFz = φ .^ 2 .* νz .* mez .^ 2 .- φ .* (ystarz .- μstarz) .* mez .* (1 .- 2 .* μz)
    if mask !== nothing
        WFz = ifelse.(mask, WFz, zero(eltype(WFz)))
    end
    Az = Λ' * (WFz .* Λ) + I
    ℓ = zero(eltype(z))
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _beta_logker(μz[t], φ, y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(Az)
end

"""
    beta_laplace_grad(Y, Λ, β, φ) -> Vector

Exact gradient of the total Beta (logit link, precision `φ`) Laplace marginal wrt
`θ = [β; pack_lambda(Λ); log φ]`, via the ForwardDiff + implicit-step construction,
with the observed-Hessian weight obtained from an AD-derivative of the score.
Finite-difference-verified; the default gradient of `fit_beta_gllvm` (`gradient = :analytic`).
"""
function beta_laplace_grad(Y::AbstractMatrix, Λ::AbstractMatrix, β::AbstractVector, φ::Real;
                           mask = nothing)
    p, K = size(Λ)
    rr = rr_theta_len(p, K)
    θ̂ = vcat(float.(β), pack_lambda(Λ), log(float(φ)))
    function marg(θ)
        b = θ[1:p]
        L = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        logφ = θ[p + rr + 1]
        acc = zero(eltype(θ))
        @inbounds for s in axes(Y, 2)
            mi = mask === nothing ? nothing : view(mask, :, s)
            acc += _beta_site_diffable(view(Y, :, s), L, b, logφ; mask = mi)
        end
        return acc
    end
    return ForwardDiff.gradient(marg, θ̂)
end

# --- Binomial (logit link) — second no-dispersion family, same technique ---
# AD-friendly logit-Binomial log-pmf kernel (the binomial-coefficient term is a
# constant in θ ⇒ zero gradient, so it is dropped). μ = logistic(η).
_binom_logker(μ, n, y) = y * log(μ) + (n - y) * log(one(μ) - μ)

# `mask` (length-p Bool, or `nothing` = all observed) drops unobserved responses,
# matching the core's masking (zero score, zero weight, skipped log-pmf).
function _binomial_site_diffable(y::AbstractVector, nt::AbstractVector,
                                 Λ::AbstractMatrix, β::AbstractVector; mask = nothing)
    p = size(Λ, 1)
    Λv = ForwardDiff.value.(Λ); βv = ForwardDiff.value.(β)
    ẑ = _laplace_mode(Binomial(), y, nt, Λv, βv, LogitLink(); mask = mask)

    η = _clamp_eta.(β .+ Λ * ẑ)
    μ = 1 ./ (1 .+ exp.(-η))            # logistic
    s = y .- nt .* μ                    # logit-link score (y − nμ)
    W = nt .* μ .* (1 .- μ)             # logit-link weight (nμ(1−μ))
    if mask !== nothing
        s = ifelse.(mask, s, zero(eltype(s)))
        W = ifelse.(mask, W, zero(eltype(W)))
    end
    A = Λ' * (W .* Λ) + I
    z = ẑ .+ (A \ (Λ' * s .- ẑ))

    ηz = _clamp_eta.(β .+ Λ * z)
    μz = 1 ./ (1 .+ exp.(-ηz))
    Wz = nt .* μz .* (1 .- μz)
    if mask !== nothing
        Wz = ifelse.(mask, Wz, zero(eltype(Wz)))
    end
    Az = Λ' * (Wz .* Λ) + I
    ℓ = zero(eltype(z))
    @inbounds for t in 1:p
        (mask === nothing || mask[t]) || continue
        ℓ += _binom_logker(μz[t], nt[t], y[t])
    end
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(Az)
end

"""
    binomial_laplace_grad(Y, N, Λ, β; mask=nothing) -> Vector

Exact gradient of the total Binomial (logit-link) Laplace marginal wrt
`θ = [β; pack_lambda(Λ)]`, via the same ForwardDiff + implicit-step construction as
[`poisson_laplace_grad`](@ref). `N` is the p×n trial-count matrix. `mask` (p×n Bool,
or `nothing` = all observed) drops unobserved responses per site, so the gradient is
of the masked marginal over the observed cells only.
Finite-difference-verified; the default gradient of `fit_binomial_gllvm`
(`gradient = :analytic`); a masked fit falls back to `autodiff = :finite`.
"""
function binomial_laplace_grad(Y::AbstractMatrix, N::AbstractMatrix,
                               Λ::AbstractMatrix, β::AbstractVector; mask = nothing)
    p, K = size(Λ)
    rr = rr_theta_len(p, K)
    θ̂ = vcat(float.(β), pack_lambda(Λ))
    function marg(θ)
        b = θ[1:p]
        L = unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        acc = zero(eltype(θ))
        @inbounds for s in axes(Y, 2)
            mi = mask === nothing ? nothing : view(mask, :, s)
            acc += _binomial_site_diffable(view(Y, :, s), view(N, :, s), L, b; mask = mi)
        end
        return acc
    end
    return ForwardDiff.gradient(marg, θ̂)
end
