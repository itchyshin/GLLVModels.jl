# Ordinal (ordered categorical, C levels) — proportional-odds cumulative-link
# GLLVM. y ∈ {1,…,C}; latent η = (Λ z)_t with z ~ N(0, I_K); common ordered
# cutpoints τ₁<…<τ_{C-1} (shared across species) absorb the category levels, so
# there is no separate species intercept. Cumulative model (McCullagh 1980),
# with link CDF F (F = logistic for logit, F = Φ for probit):
#   P(y ≤ c | z) = F(τ_c − η),
#   P(y = c | z) = F(τ_c − η) − F(τ_{c-1} − η),   τ₀ = −∞, τ_C = +∞.
# The link is selectable (`LogitLink()` default — gllvm-parity — or `ProbitLink()`);
# only the (F, f) pair changes, the score/weight/mode machinery is link-agnostic.
#
# The "mean" here is a vector of category probabilities, so this family does NOT
# use the scalar-μ generic Laplace core (families/laplace.jl). It carries its own
# per-site Fisher-scoring mode-finder, mirroring that core's normalisation —
# log p(y_s) ≈ ℓ(ẑ) − ½ẑ'ẑ − ½ logdet(Λ'WΛ + I). Per observation, wrt η:
#   score(η) = (f(τ_{c-1}−η) − f(τ_c−η)) / P(y=c)
#   W(η)     = Σ_{k=1}^{C} (f(τ_{k-1}−η) − f(τ_k−η))² / P(y=k)    (Fisher info ≥ 0)
# with f = F' the link density (logistic·(1−logistic) for logit; φ for probit).
# `_clamp_eta`/`_safe_solve` are reused from families/laplace.jl.

"""
    Ordinal

Family marker for the ordered-categorical (proportional-odds cumulative-logit)
GLLVM. `Distributions` has no ordinal type, so GLLVModels defines its own. Categories
are coded `1:C`; the number of levels `C` is inferred from the data (`maximum(Y)`)
by the fitter, and equals `length(τ) + 1` in the marginal.
"""
struct Ordinal end

default_link(::Ordinal) = LogitLink()

function _check_ordinal_link(link::Link)
    link isa Union{LogitLink, ProbitLink} || throw(ArgumentError(
        "ordinal fitters support only LogitLink or ProbitLink; got $(typeof(link))"))
    return nothing
end

# Observed ordinal levels must be integers >= 1: the fitters index count vectors
# by level inside `@inbounds` loops. Masked cells (`obs` false) are not checked.
function _check_ordinal_levels(Y::AbstractMatrix, obs::AbstractMatrix)
    for i in eachindex(Y, obs)
        obs[i] || continue
        y = Y[i]
        y >= 1 || throw(ArgumentError(
            "ordinal response has observed level $y at index $(CartesianIndices(Y)[i]); " *
            "levels must be integers 1..C"))
    end
    return nothing
end

"""
    OrdinalLogit

Family marker returned by [`ordinal_logit`](@ref). A distinct type from
[`Ordinal`](@ref) — not merely `Ordinal()` under another name — so that the
link stays pinned to `LogitLink()` regardless of what `Ordinal()`'s own
default link happens to be, and so `fit_gllvm(Y; family = ordinal_logit(),
link = ProbitLink())` is refused rather than silently fitting a probit model
under the logit name. The underlying fit (`fit_ordinal_gllvm_pertrait` /
`fit_ordinal_gllvm_pertrait_cov`) and its result type
([`OrdinalPerTraitFit`](@ref) / `OrdinalPerTraitCovFit`) are exactly the
same as `Ordinal()`'s with `link = LogitLink()`; post-fit, `confint`,
`simulate`, `predict`, and the R bridge all dispatch on that shared result
type, not on this marker, so this type has no other dispatch surface to
maintain beyond [`fit_gllvm`](@ref) / [`gllvm`](@ref) itself.
"""
struct OrdinalLogit end

default_link(::OrdinalLogit) = LogitLink()

# Both the bare `fit_gllvm` entry point and the `@formula` front door
# (`src/formula.jl`) reach this: reject any explicitly supplied non-logit
# `link` up front rather than silently overriding it, since the whole point
# of a distinct marker is that `ordinal_logit()` never quietly becomes a
# probit fit.
function _check_ordinal_logit_link(kwargs)
    haskey(kwargs, :link) && !(kwargs[:link] isa LogitLink) && throw(ArgumentError(
        "ordinal_logit() only supports LogitLink(); got $(typeof(kwargs[:link])). " *
        "Use Ordinal() with link = ProbitLink() for the probit link."))
    return nothing
end

"""
    ordinal_logit(; link::Link = LogitLink())

The Julia twin of `gllvmTMB`'s `ordinal_logit()` response family (`family_id
20`, gllvmTMB ≥ 0.7.1): a cumulative-**logit** threshold model for K ≥ 3
ordered categories, with a per-trait intercept and per-trait cutpoints
(``\\tau_1 = 0`` fixed, ``K_t - 2`` free log-spaced cutpoints). It fits
exactly the same model as [`Ordinal`](@ref) with `link = LogitLink()` — via
[`fit_ordinal_gllvm_pertrait`](@ref) / [`fit_ordinal_gllvm_pertrait_cov`](@ref)
— but returns the distinct [`OrdinalLogit`](@ref) marker, which pins the
link: `fit_gllvm(Y; family = ordinal_logit(), link = ProbitLink())` throws
`ArgumentError` rather than silently fitting a probit model under the logit
name. As in `gllvmTMB`, `link` supports only the logit link and exists for
API symmetry with the family constructor shape; pass anything else and this
throws, naming [`Ordinal`](@ref) with `ProbitLink()` as the alternative —
exactly as R's `ordinal_logit(link = "probit")` names `ordinal_probit()`.

The shared-cutpoint, no-intercept [`fit_ordinal_gllvm`](@ref) route (the
`Ordinal()` marker's other, non-default fitter) has no `gllvmTMB` twin.

# Examples
```julia
fit = fit_gllvm(Y; family = ordinal_logit(), K = 1)
```
"""
function ordinal_logit(; link::Link = LogitLink())
    link isa LogitLink || throw(ArgumentError(
        "ordinal_logit supports only LogitLink(); got $(typeof(link)). " *
        "Use Ordinal() with link = ProbitLink() for the probit link."))
    return OrdinalLogit()
end

# Link CDF F and density f = F'. The cumulative model and the analytic
# score/Fisher-weight are written generically in (F, f), so a new link only
# swaps these two. Logit (default) keeps its exact prior numerics; probit uses
# the standard-Normal CDF/pdf. η is clamped identically for both (parity of the
# mode-finder); probit does not overflow but the clamp is harmless on [−c, c].
_ord_F(x, ::LogitLink) = inv(one(x) + exp(-_clamp_eta(x)))            # logistic CDF (η-clamped)
_ord_f(x, ::LogitLink) = (Fx = _ord_F(x, LogitLink()); Fx * (one(Fx) - Fx))  # logistic density
_ord_F(x, ::ProbitLink) = cdf(Normal(), _clamp_eta(x))               # Φ (η-clamped for parity)
_ord_f(x, ::ProbitLink) = pdf(Normal(), _clamp_eta(x))               # φ
# Logit default: preserve byte-for-byte the original argument-less call sites.
_ord_F(x) = _ord_F(x, LogitLink())
_ord_f(x) = _ord_f(x, LogitLink())

# P(y = c) at linear predictor η with ordered cutpoints τ (length C−1).
@inline function _ord_prob(c::Integer, η, τ::AbstractVector, link::Link = LogitLink())
    C = length(τ) + 1
    Fhi = c == C ? one(η) : _ord_F(τ[c] - η, link)
    Flo = c == 1 ? zero(η) : _ord_F(τ[c - 1] - η, link)
    return Fhi - Flo
end

# Derivative of the link density f = F' with respect to its argument.
_ord_fp(x, ::LogitLink) = begin
    Fx = _ord_F(x, LogitLink())
    fx = Fx * (one(Fx) - Fx)
    fx * (one(Fx) - 2Fx)
end
_ord_fp(x, ::ProbitLink) = -_clamp_eta(x) * _ord_f(x, ProbitLink())

# Score ∂logP(y=c)/∂η and observed curvature −∂²logP(y=c)/∂η² at η.
function _ord_score_weight(c::Integer, η, τ::AbstractVector, link::Link = LogitLink())
    C = length(τ) + 1
    fhi = c == C ? zero(η) : _ord_f(τ[c] - η, link)
    flo = c == 1 ? zero(η) : _ord_f(τ[c - 1] - η, link)
    fp_hi = c == C ? zero(η) : _ord_fp(τ[c] - η, link)
    fp_lo = c == 1 ? zero(η) : _ord_fp(τ[c - 1] - η, link)
    P = max(_ord_prob(c, η, τ, link), 1e-12)
    dP = flo - fhi
    ddP = fp_hi - fp_lo
    score = dP / P
    W = max(dP^2 / P^2 - ddP / P, zero(η))
    return score, W
end

# Per-site Laplace mode ẑ (Fisher-scoring Newton); η = Λ z (no intercept).
function _ordinal_laplace_mode(y::AbstractVector, Λ::AbstractMatrix, τ::AbstractVector,
        link::Link = LogitLink(); mask = nothing, offset = nothing,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    z = zeros(K)
    s = Vector{Float64}(undef, p)
    W = Vector{Float64}(undef, p)
    for _ in 1:maxiter
        η = offset === nothing ? _clamp_eta.(Λ * z) : _clamp_eta.(Λ * z .+ offset)
        @inbounds for t in 1:p
            if mask !== nothing && !mask[t]
                s[t] = 0.0; W[t] = 0.0          # masked (missing) ⇒ no contribution
            else
                st, wt = _ord_score_weight(Int(y[t]), η[t], τ, link)
                s[t] = st; W[t] = wt
            end
        end
        A = Symmetric(Λ' * (W .* Λ) + I)
        Δ = _safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < tol && break
    end
    return z
end

"""
    ordinal_loglik_site(y, Λ, τ; maxiter=100, tol=1e-9) -> Float64

Laplace log-marginal for one site of a cumulative-logit ordinal GLLVM:
`ℓ(ẑ) − ½ẑ'ẑ − ½logdet(Λ'WΛ + I)`. `y` length-p ordinal responses (`1:C`),
`Λ` p×K, `τ` the `C−1` ordered cutpoints.
"""
function ordinal_loglik_site(y::AbstractVector, Λ::AbstractMatrix, τ::AbstractVector,
        link::Link = LogitLink(); mask = nothing, offset = nothing,
        maxiter::Integer = 100, tol::Real = 1e-9)
    p = size(Λ, 1)
    z = _ordinal_laplace_mode(y, Λ, τ, link;
                              mask = mask, offset = offset,
                              maxiter = maxiter, tol = tol)
    η = offset === nothing ? _clamp_eta.(Λ * z) : _clamp_eta.(Λ * z .+ offset)
    W = Vector{Float64}(undef, p)
    ℓ = 0.0
    @inbounds for t in 1:p
        if mask !== nothing && !mask[t]
            W[t] = 0.0                          # masked ⇒ dropped from logdet, no logpdf
        else
            ℓ += log(max(_ord_prob(Int(y[t]), η[t], τ, link), 1e-12))
            _, wt = _ord_score_weight(Int(y[t]), η[t], τ, link)
            W[t] = wt
        end
    end
    A = Symmetric(Λ' * (W .* Λ) + I)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

"""
    ordinal_marginal_loglik_laplace(Y, Λ, τ; link=LogitLink(), kwargs...) -> Float64

Total Laplace log-marginal over the `n` sites (columns) of a proportional-odds
cumulative ordinal GLLVM. `Y` is the p×n matrix of ordinal responses coded
`1:C`; `Λ` p×K; `τ` the `C−1` ordered cutpoints (shared across species). `link`
selects the cumulative-link CDF `F` (`LogitLink()` default, `ProbitLink()`). With
`Λ = 0` (η ≡ 0) the latent variable drops out and this reduces to the exact
independent cumulative-link log-likelihood `Σ log(F(τ_c) − F(τ_{c−1}))`.
"""
function ordinal_marginal_loglik_laplace(Y::AbstractMatrix, Λ::AbstractMatrix,
        τ::AbstractVector; link::Link = LogitLink(), mask = nothing,
        offset = nothing, kwargs...)
    acc = 0.0
    @inbounds for s in axes(Y, 2)
        mcol = mask === nothing ? nothing : view(mask, :, s)
        ocol = offset === nothing ? nothing : view(offset, :, s)
        acc += ordinal_loglik_site(view(Y, :, s), Λ, τ, link;
                                   mask = mcol, offset = ocol, kwargs...)
    end
    return acc
end

# Per-trait cutpoint variant. Native gllvmTMB estimates cutpoints separately for
# each ordinal response; the shared-cutpoint fitter above is kept for Julia-side
# experiments and backward compatibility.
@inline function _trait_cutpoints(τ::AbstractMatrix, C::AbstractVector{<:Integer}, t::Integer)
    return view(τ, t, 1:(C[t] - 1))
end

function _ordinal_laplace_mode_pertrait(y::AbstractVector, Λ::AbstractMatrix,
        β::AbstractVector, τ::AbstractMatrix, C::AbstractVector{<:Integer},
        link::Link = LogitLink();
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p, K = size(Λ)
    length(β) == p || throw(ArgumentError("ordinal intercept length must equal p"))
    z = zeros(K)
    s = Vector{Float64}(undef, p)
    W = Vector{Float64}(undef, p)
    for _ in 1:maxiter
        η = offset === nothing ? _clamp_eta.(β .+ Λ * z) :
            _clamp_eta.(β .+ offset .+ Λ * z)
        @inbounds for t in 1:p
            if mask !== nothing && !mask[t]
                s[t] = 0.0
                W[t] = 0.0
            else
                τt = _trait_cutpoints(τ, C, t)
                st, wt = _ord_score_weight(Int(y[t]), η[t], τt, link)
                s[t] = st
                W[t] = wt
            end
        end
        A = Symmetric(Λ' * (W .* Λ) + I)
        Δ = _safe_solve(A, Λ' * s .- z)
        (Δ === nothing || !all(isfinite, Δ)) && break
        z = z .+ Δ
        maximum(abs, Δ) < tol && break
    end
    return z
end

function ordinal_loglik_site_pertrait(y::AbstractVector, Λ::AbstractMatrix,
        β::AbstractVector, τ::AbstractMatrix, C::AbstractVector{<:Integer},
        link::Link = LogitLink();
        mask = nothing, offset = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    p = size(Λ, 1)
    z = _ordinal_laplace_mode_pertrait(y, Λ, β, τ, C, link;
                                       mask = mask, offset = offset,
                                       maxiter = maxiter, tol = tol)
    η = offset === nothing ? _clamp_eta.(β .+ Λ * z) :
        _clamp_eta.(β .+ offset .+ Λ * z)
    W = Vector{Float64}(undef, p)
    ℓ = 0.0
    @inbounds for t in 1:p
        if mask !== nothing && !mask[t]
            W[t] = 0.0
        else
            τt = _trait_cutpoints(τ, C, t)
            c = Int(y[t])
            1 <= c <= C[t] || throw(ArgumentError(
                "ordinal response category $c is outside 1:$(C[t]) for trait $t"))
            ℓ += log(max(_ord_prob(c, η[t], τt, link), 1e-12))
            _, wt = _ord_score_weight(c, η[t], τt, link)
            W[t] = wt
        end
    end
    A = Symmetric(Λ' * (W .* Λ) + I)
    return ℓ - 0.5 * dot(z, z) - 0.5 * logdet(A)
end

function ordinal_marginal_loglik_laplace_pertrait(Y::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, τ::AbstractMatrix,
        C::AbstractVector{<:Integer};
        link::Link = LogitLink(), mask = nothing, offset = nothing, kwargs...)
    acc = 0.0
    @inbounds for s in axes(Y, 2)
        mcol = mask === nothing ? nothing : view(mask, :, s)
        ocol = offset === nothing ? nothing : view(offset, :, s)
        acc += ordinal_loglik_site_pertrait(view(Y, :, s), Λ, β, τ, C, link;
                                            mask = mcol, offset = ocol, kwargs...)
    end
    return acc
end

ordinal_marginal_loglik_laplace_pertrait(Y::AbstractMatrix, Λ::AbstractMatrix,
        τ::AbstractMatrix, C::AbstractVector{<:Integer}; kwargs...) =
    ordinal_marginal_loglik_laplace_pertrait(Y, Λ, zeros(eltype(Λ), size(Λ, 1)),
                                             τ, C; kwargs...)

# ---------------------------------------------------------------------------
# Fit driver (Ordinal family slice 2).
# ---------------------------------------------------------------------------

"""
    OrdinalFit

Result of [`fit_ordinal_gllvm`](@ref): loadings `Λ` (p×K), the `C−1` ordered
cutpoints `τ`, the number of categories `C`, the `link`, the maximised Laplace
`loglik`, the optimiser `converged` flag, and `iterations`. Fits using `X_lv`
also retain `alpha_lv`, the raw latent-axis coefficients for the
predictor-informed score mean. There is no species intercept; the common
cutpoints carry the category levels.
"""
struct OrdinalFit
    Λ::Matrix{Float64}
    τ::Vector{Float64}
    C::Int
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    alpha_lv::Union{Nothing, Matrix{Float64}}
    theta_packed::Vector{Float64}
end

OrdinalFit(Λ::Matrix{Float64}, τ::Vector{Float64}, C::Int, link::Link,
           loglik::Float64, converged::Bool, iterations::Int) =
    OrdinalFit(Λ, τ, C, link, loglik, converged, iterations, nothing, Float64[])

"""
    OrdinalPerTraitFit

Ordinal GLLVM fit with trait-specific ordered cutpoints. `τ` is a
`p × max(C_t - 1)` matrix padded with `NaN` after each trait's last cutpoint, and
`C` is the per-trait category count. This is the native `gllvmTMB` parity shape
used by the R bridge; [`OrdinalFit`](@ref) remains the shared-cutpoint shape.
"""
struct OrdinalPerTraitFit
    Λ::Matrix{Float64}
    β::Vector{Float64}
    τ::Matrix{Float64}
    C::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::OrdinalFit)
    p, K = size(f.Λ)
    print(io, "OrdinalFit(p=", p, ", K=", K, ", C=", f.C,
          ", link=", nameof(typeof(f.link)),
          f.alpha_lv === nothing ? "" : ", X_lv=true",
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

# Ordered cutpoints from unconstrained ψ: τ₁ = ψ₁, τ_c = τ_{c-1} + exp(ψ_c).
function _unpack_cutpoints(ψ::AbstractVector)
    m = length(ψ)
    τ = Vector{float(eltype(ψ))}(undef, m)
    τ[1] = ψ[1]
    @inbounds for c in 2:m
        τ[c] = τ[c - 1] + exp(ψ[c])
    end
    return τ
end

function _unpack_cutpoints_pertrait(ψ::AbstractVector, C::AbstractVector{<:Integer})
    p = length(C)
    Cmax = maximum(C)
    τ = fill(NaN, p, Cmax - 1)
    pos = 1
    @inbounds for t in 1:p
        m = C[t] - 1
        τ[t, 1] = 0.0
        for c in 2:m
            τ[t, c] = τ[t, c - 1] + exp(ψ[pos])
            pos += 1
        end
    end
    return τ
end

_ord_quantile(p, ::LogitLink) = log(p / (1 - p))
_ord_quantile(p, ::ProbitLink) = quantile(Normal(), p)

function _pack_initial_ordinal_pertrait(Y::AbstractMatrix, obs::AbstractMatrix,
                                        C::AbstractVector{<:Integer}, link::Link)
    p = size(Y, 1)
    β0 = zeros(Float64, p)
    pieces = Vector{Float64}[]
    @inbounds for t in 1:p
        counts = zeros(Int, C[t])
        for i in axes(Y, 2)
            obs[t, i] && (counts[Int(Y[t, i])] += 1)
        end
        total = sum(counts)
        total > 0 || throw(ArgumentError("ordinal response trait $t has no observed cells"))
        cum = cumsum(counts ./ total)
        τ0 = [_ord_quantile(clamp(cum[c], 1e-3, 1 - 1e-3), link)
              for c in 1:(C[t] - 1)]
        β0[t] = -τ0[1]
        ψ0 = Float64[]
        for c in 2:(C[t] - 1)
            push!(ψ0, log(max(τ0[c] - τ0[c - 1], 1e-3)))
        end
        push!(pieces, ψ0)
    end
    return β0, reduce(vcat, pieces; init = Float64[])
end

"""
    ordinal_lv_nll_packed(params, Y, p, K, link, C; X_lv, q_lv, kwargs...) -> Real

Negative Laplace log-likelihood for the predictor-informed latent-score ordinal
model. Parameter layout:

- first `q_lv * K` entries = `alpha_lv`, reshaped as `q_lv × K`;
- next entries = packed reduced-rank loadings `Λ`;
- final `C - 1` entries = unconstrained shared cutpoint increments.

The conditional latent variable is the zero-mean innovation. The predictor mean
enters the ordinal Laplace core as the parameter-dependent link-scale offset
`Λ * alpha_lv' * X_lv[s, :]`.
"""
function ordinal_lv_nll_packed(params::AbstractVector, Y::AbstractMatrix,
        p::Integer, K::Integer, link::Link, C::Integer;
        X_lv::AbstractMatrix, q_lv::Integer,
        mask = nothing, maxiter::Integer = 100, tol::Real = 1e-9)
    size(Y, 1) == p ||
        throw(ArgumentError("Y first dim ($(size(Y, 1))) must equal p ($p)"))
    n = size(Y, 2)
    size(X_lv, 1) == n ||
        throw(ArgumentError("X_lv first dim ($(size(X_lv, 1))) must equal n_sites ($n)"))
    size(X_lv, 2) == q_lv ||
        throw(ArgumentError("X_lv second dim ($(size(X_lv, 2))) must equal q_lv ($q_lv)"))
    q_lv > 0 || throw(ArgumentError("q_lv must be positive"))

    rr = rr_theta_len(p, K)
    n_expected = q_lv * K + rr + C - 1
    length(params) == n_expected || throw(ArgumentError(
        "params length ($(length(params))) must equal $n_expected " *
        "(alpha_lv=$(q_lv * K) + rr=$rr + cutpoints=$(C - 1))"))

    cursor = 0
    alpha_vec = @view params[(cursor + 1):(cursor + q_lv * K)]
    alpha_lv = reshape(alpha_vec, q_lv, K)
    cursor += q_lv * K
    θ_rr = @view params[(cursor + 1):(cursor + rr)]
    Λ = unpack_lambda(θ_rr, p, K)
    cursor += rr
    τ = _unpack_cutpoints(@view params[(cursor + 1):(cursor + C - 1)])

    lv_offset = _lv_mean_eta(Λ, X_lv, alpha_lv)
    return -ordinal_marginal_loglik_laplace(Y, Λ, τ;
                                            link = link, mask = mask,
                                            offset = lv_offset,
                                            maxiter = maxiter, tol = tol)
end

"""
    fit_ordinal_gllvm(Y; K, link=LogitLink(), X_lv=nothing, alpha_lv_init=nothing, …) -> OrdinalFit

Fit a proportional-odds cumulative ordinal GLLVM by L-BFGS over
`[vec(Λ); ψ]`, where the `C−1` ordered cutpoints are the unconstrained increments
`τ₁ = ψ₁, τ_c = τ_{c-1} + exp(ψ_c)` (so ordering holds for free) and the marginal
is [`ordinal_marginal_loglik_laplace`](@ref). `link` selects the cumulative-link
CDF (`LogitLink()` default, `ProbitLink()`). `Y` is a p×n matrix of ordinal
responses coded `1:C` (`C = maximum(Y)`). With `X_lv`, the fitted latent-score
mean is `X_lv * alpha_lv`, `alpha_lv_init` can seed that q×K coefficient matrix,
and [`extract_lv_effects`](@ref) / [`confint_lv_effects`](@ref) target the
rotation-stable `B_lv = Λ * alpha_lv'` product. Finite-difference gradient; warm
start = empirical cumulative-proportion cutpoints + a normal-scores SVD loadings
init. Unsupported links throw `ArgumentError` before response access.
"""
function fit_ordinal_gllvm(Y::AbstractMatrix{<:Integer}; K::Integer,
        link::Link = LogitLink(), Λ_init = nothing, mask = nothing,
        X_lv::Union{Nothing, AbstractMatrix} = nothing,
        alpha_lv_init = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    _check_ordinal_link(link)
    p, n = size(Y)
    obs = mask === nothing ? trues(p, n) : mask
    _check_ordinal_levels(Y, obs)
    # Category count and warm starts use OBSERVED cells only, so a masked cell's
    # (arbitrary) value never leaks into the fit.
    C = 0
    @inbounds for i in eachindex(Y)
        obs[i] && (C = max(C, Int(Y[i])))
    end
    C ≥ 2 || throw(ArgumentError("ordinal response needs ≥ 2 observed categories; got $C"))
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
    # Sanitise masked cells to a valid category for the warm starts only.
    Ys = mask === nothing ? Y : [obs[t, i] ? Int(Y[t, i]) : 1 for t in 1:p, i in 1:n]

    # Loadings warm start: SVD of a row-centred normal-scores latent proxy.
    Zproxy = [quantile(Normal(), clamp((Ys[t, i] - 0.5) / C, 1e-3, 1 - 1e-3))
              for t in 1:p, i in 1:n]
    Λ0 = if Λ_init === nothing
        Zc = Zproxy .- (sum(Zproxy; dims = 2) ./ n)
        F = svd(Zc); kk = min(K, length(F.S))
        L = zeros(p, K)
        @inbounds for j in 1:kk
            L[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
        end
        L
    else
        collect(float.(Λ_init))
    end
    Zc = Zproxy .- (sum(Zproxy; dims = 2) ./ n)
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

    # Cutpoint warm start: τ_c = logit(empirical P(y ≤ c)); to ψ increments.
    counts = zeros(Int, C)
    @inbounds for i in eachindex(Ys)
        obs[i] && (counts[Int(Ys[i])] += 1)
    end
    cum = cumsum(counts ./ sum(counts))
    τ0 = [log(clamp(cum[c], 1e-3, 1 - 1e-3) / (1 - clamp(cum[c], 1e-3, 1 - 1e-3)))
          for c in 1:(C - 1)]
    ψ0 = similar(τ0)
    ψ0[1] = τ0[1]
    @inbounds for c in 2:(C - 1)
        ψ0[c] = log(max(τ0[c] - τ0[c - 1], 1e-3))
    end

    θ0 = vcat(pack_lambda(Λ0), ψ0)
    function negll(θ)
        Λ = unpack_lambda(θ[1:rr], p, K)
        τ = _unpack_cutpoints(θ[(rr + 1):(rr + C - 1)])
        v = try
            -ordinal_marginal_loglik_laplace(Y, Λ, τ; link = link, mask = mask,
                                             maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    res = if X_lv_fit !== nothing
        θ0_lv = vcat(vec(alpha0), pack_lambda(Λ0), ψ0)
        negll_lv = θ -> begin
            v = try
                ordinal_lv_nll_packed(θ, Y, p, K, link, C;
                                      X_lv = X_lv_fit, q_lv = q_lv,
                                      mask = mask, maxiter = newton_maxiter,
                                      tol = newton_tol)
            catch
                return 1e12
            end
            return isfinite(v) ? v : 1e12
        end
        Optim.optimize(negll_lv, θ0_lv, ls, opts; autodiff = :finite)
    else
        Optim.optimize(negll, θ0, ls, opts; autodiff = :finite)
    end
    θ̂ = Optim.minimizer(res)
    if X_lv_fit !== nothing
        cursor = 0
        alpha_hat = reshape(collect(θ̂[(cursor + 1):(cursor + q_lv * K)]), q_lv, K)
        cursor += q_lv * K
        Λ̂ = unpack_lambda(@view(θ̂[(cursor + 1):(cursor + rr)]), p, K)
        cursor += rr
        τ̂ = _unpack_cutpoints(@view(θ̂[(cursor + 1):(cursor + C - 1)]))
        return OrdinalFit(Λ̂, τ̂, C, link, _fit_verdict(res)...,
                          alpha_hat, collect(Float64, θ̂))
    else
        Λ̂ = unpack_lambda(θ̂[1:rr], p, K)
        τ̂ = _unpack_cutpoints(θ̂[(rr + 1):(rr + C - 1)])
        return OrdinalFit(Λ̂, τ̂, C, link, _fit_verdict(res)...)
    end
end

"""
    fit_ordinal_gllvm_pertrait(Y; K, link=LogitLink(), …) -> OrdinalPerTraitFit

Fit a cumulative ordinal GLLVM with one ordered cutpoint vector per trait. The
cutpoint contribution to the degrees of freedom is `sum(C_t - 1)`, matching the
native `gllvmTMB` ordinal bridge target. Only `LogitLink()` and `ProbitLink()`
are supported; other links throw `ArgumentError` before response access. The shared-cutpoint
[`fit_ordinal_gllvm`](@ref) is preserved for experiments and old tests.
"""
function fit_ordinal_gllvm_pertrait(Y::AbstractMatrix{<:Integer}; K::Integer,
        link::Link = LogitLink(), Λ_init = nothing, mask = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    _check_ordinal_link(link)
    p, n = size(Y)
    obs = mask === nothing ? trues(p, n) : mask
    _check_ordinal_levels(Y, obs)
    C = zeros(Int, p)
    @inbounds for t in 1:p, i in 1:n
        obs[t, i] && (C[t] = max(C[t], Int(Y[t, i])))
    end
    all(>=(2), C) || throw(ArgumentError(
        "ordinal response needs >= 2 observed categories for every trait; got $C"))
    rr = rr_theta_len(p, K)
    Ys = mask === nothing ? Y : [obs[t, i] ? Int(Y[t, i]) : 1 for t in 1:p, i in 1:n]

    Zproxy = [quantile(Normal(), clamp((Ys[t, i] - 0.5) / C[t], 1e-3, 1 - 1e-3))
              for t in 1:p, i in 1:n]
    Λ0 = if Λ_init === nothing
        Zc = Zproxy .- (sum(Zproxy; dims = 2) ./ n)
        F = svd(Zc); kk = min(K, length(F.S))
        L = zeros(p, K)
        @inbounds for j in 1:kk
            L[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
        end
        L
    else
        collect(float.(Λ_init))
    end

    β0, ψ0 = _pack_initial_ordinal_pertrait(Ys, obs, C, link)
    θ0 = vcat(β0, pack_lambda(Λ0), ψ0)
    ncut = sum(C .- 2)
    function negll(θ)
        β = @view θ[1:p]
        Λ = unpack_lambda(@view(θ[(p + 1):(p + rr)]), p, K)
        τ = _unpack_cutpoints_pertrait(@view(θ[(p + rr + 1):(p + rr + ncut)]), C)
        v = try
            -ordinal_marginal_loglik_laplace_pertrait(Y, Λ, β, τ, C;
                link = link, mask = mask, maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(negll, θ0, ls, Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    θ̂ = Optim.minimizer(res)
    β̂ = collect(@view θ̂[1:p])
    Λ̂ = unpack_lambda(@view(θ̂[(p + 1):(p + rr)]), p, K)
    τ̂ = _unpack_cutpoints_pertrait(@view(θ̂[(p + rr + 1):(p + rr + ncut)]), C)
    return OrdinalPerTraitFit(Λ̂, β̂, τ̂, C, link, _fit_verdict(res)...)
end

"""
    OrdinalPerTraitCovFit

Result of [`fit_ordinal_gllvm_pertrait_cov`](@ref): per-trait intercepts `β`,
shared covariate coefficients `γ` (with `γ_fixed` zero mask), loadings `Λ`,
per-trait ordered cutpoints `τ` (`p × max(C_t−1)`, τ₁=0 fixed), per-trait
category counts `C`, `link`, maximised Laplace `loglik`, `converged`, and
`iterations`. Linear predictor `η = β + Xγ + Λz` with twin cutpoint packing
(τ₁=0 / K−2 free log-spacings). Shared-cutpoint [`OrdinalFit`](@ref) remains
the Julia-side comparator and is **not** the public X default.
"""
struct OrdinalPerTraitCovFit
    β::Vector{Float64}
    γ::Vector{Float64}
    γ_fixed::Vector{Bool}
    Λ::Matrix{Float64}
    τ::Matrix{Float64}
    C::Vector{Int}
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
end

function Base.show(io::IO, f::OrdinalPerTraitCovFit)
    p, K = size(f.Λ); q = length(f.γ)
    print(io, "OrdinalPerTraitCovFit(p=", p, ", q=", q, ", K=", K,
          ", C=", f.C, ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    fit_ordinal_gllvm_pertrait_cov(Y; X, K, link=LogitLink(), mask=nothing,
                                   γ_fixed=nothing, …) -> OrdinalPerTraitCovFit

Fit a cumulative ordinal GLLVM with **per-trait cutpoints** (τ₁=0 fixed; K−2
free log-spacings per trait) and **shared site covariates** `X` (`p×n×q`).
Working vector `[β; γ_free; pack(Λ); ψ]` with ψ the unconstrained per-trait
log-spacings; offset `O = Xγ` enters the per-trait Laplace marginal as
`η = β + O + Λz`. Public / bridge / `@formula` default under X for Ordinal
(twin API B; decision 2026-08-03). Keep shared-cutpoint
[`fit_ordinal_gllvm`](@ref) as an explicit comparator — do not route public X
through shared cutpoints. Only `LogitLink()` and `ProbitLink()` are supported;
other links throw `ArgumentError` before response access.
"""
function fit_ordinal_gllvm_pertrait_cov(Y::AbstractMatrix{<:Integer};
        X::AbstractArray{<:Real, 3}, K::Integer,
        link::Link = LogitLink(), Λ_init = nothing, mask = nothing,
        γ_fixed = nothing,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    _check_ordinal_link(link)
    p, n = size(Y)
    size(X, 1) == p && size(X, 2) == n ||
        throw(DimensionMismatch("X must be (p, n, q) = ($p, $n, q); got $(size(X))"))
    q_full = size(X, 3)
    γ_fixed_mask = _fixed_zero_mask(γ_fixed, q_full, "γ_fixed")
    X_fit, _ = _slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    obs = mask === nothing ? trues(p, n) : mask
    _check_ordinal_levels(Y, obs)
    C = zeros(Int, p)
    @inbounds for t in 1:p, i in 1:n
        obs[t, i] && (C[t] = max(C[t], Int(Y[t, i])))
    end
    all(>=(2), C) || throw(ArgumentError(
        "ordinal response needs >= 2 observed categories for every trait; got $C"))
    rr = rr_theta_len(p, K)
    ncut = sum(C .- 2)
    Ys = mask === nothing ? Y : [obs[t, i] ? Int(Y[t, i]) : 1 for t in 1:p, i in 1:n]

    Zproxy = [quantile(Normal(), clamp((Ys[t, i] - 0.5) / C[t], 1e-3, 1 - 1e-3))
              for t in 1:p, i in 1:n]
    Λ0 = if Λ_init === nothing
        Zc = Zproxy .- (sum(Zproxy; dims = 2) ./ n)
        F = svd(Zc); kk = min(K, length(F.S))
        L = zeros(p, K)
        @inbounds for j in 1:kk
            L[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
        end
        L
    else
        collect(float.(Λ_init))
    end

    β0, ψ0 = _pack_initial_ordinal_pertrait(Ys, obs, C, link)
    θ0 = vcat(β0, zeros(q), pack_lambda(Λ0), ψ0)
    function negll(θ)
        β = @view θ[1:p]
        γ = @view θ[(p + 1):(p + q)]
        Λ = unpack_lambda(@view(θ[(p + q + 1):(p + q + rr)]), p, K)
        τ = _unpack_cutpoints_pertrait(@view(θ[(p + q + rr + 1):(p + q + rr + ncut)]), C)
        O = _build_offset(X_fit, γ)
        v = try
            -ordinal_marginal_loglik_laplace_pertrait(Y, Λ, β, τ, C;
                link = link, mask = mask, offset = O,
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
    β̂ = collect(@view θ̂[1:p])
    γ̂_free = @view θ̂[(p + 1):(p + q)]
    γ̂ = collect(Float64, _expand_fixed_zero(γ̂_free, γ_fixed_mask))
    Λ̂ = unpack_lambda(@view(θ̂[(p + q + 1):(p + q + rr)]), p, K)
    τ̂ = _unpack_cutpoints_pertrait(@view(θ̂[(p + q + rr + 1):(p + q + rr + ncut)]), C)
    return OrdinalPerTraitCovFit(β̂, γ̂, collect(Bool, γ_fixed_mask), Λ̂, τ̂, C, link,
                                 _fit_verdict(res)...)
end
