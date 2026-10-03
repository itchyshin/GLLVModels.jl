# One-part lognormal family (twin gllvmTMB family_id 3).
#
# Model (site s): y_{ts} > 0 with  log(y_{ts}) ~ Normal(η_{ts}, σ²),
#     η_{ts} = β_t + (Λ z_s)_t,  z_s ~ N(0, I_K).
# Equivalently y = exp(η + σ·ε), ε ~ N(0,1). On the LOG (link) scale the model is
# exactly the Gaussian GLLVM, so the closed-form Gaussian machinery
# (likelihood.jl, fit.jl, profile.jl) is REUSED on Z = log(Y); the only extra
# term is the change-of-variables Jacobian −Σ log(y) that converts the log-scale
# Gaussian density to the y-scale lognormal density (twin cpp fid==3).
#
# Distinct from DeltaLogNormal (two-part hurdle, fid 12) in families/twopart.jl.
# Identity: docs/dev-log/decisions/2026-08-15-lognormal-identity.md
#
# AD-clean: `lognormal_marginal_loglik` is `gaussian_marginal_loglik` (already
# ForwardDiff-clean) plus a parameter-free data constant, so Duals flow through.

"""
    Lognormal()

Marker for one-part lognormal (twin `lognormal()`, family_id 3; support `(0,∞)`;
log link on `E[log y] = η`). Distinct from `Distributions.LogNormal` and from
the two-part `DeltaLogNormal` hurdle (fid 12).
"""
struct Lognormal end

default_link(::Lognormal) = LogLink()

"""
    lognormal_marginal_loglik(Y, Λ, β, σ; kwargs...) -> Float64

Marginal log-likelihood (y-scale) of a one-part lognormal GLLVM at explicit
parameters: intercepts `β` (length `p`), loadings `Λ` (`p×K`), and log-scale
residual SD `σ` (`Var(log y) = σ²`). `Y` must be strictly positive.

Computed as the Gaussian marginal of the centred log-responses `log(Y) .- β`
under `ΛΛᵀ + σ²I`, minus the change-of-variables Jacobian `Σ log(y)`. The
Jacobian is part of the reported likelihood (twin parity / light RCall Δ).
"""
function lognormal_marginal_loglik(Y::AbstractMatrix, Λ::AbstractMatrix,
        β::AbstractVector, σ::Real; kwargs...)
    p, n = size(Y)
    length(β) == p || throw(DimensionMismatch(
        "β length ($(length(β))) must equal p ($p)"))
    all(>(0), Y) || throw(ArgumentError(
        "lognormal requires y > 0; found non-positive response"))
    Z = log.(Y)
    R = Z .- β
    gauss = gaussian_marginal_loglik(R, Λ, float(σ); kwargs...)
    return gauss - sum(Z)
end

# Alias for Identity / Laplace-family naming consistency (closed-form, not Laplace).
lognormal_marginal_loglik_laplace(Y, Λ, β, σ; kwargs...) =
    lognormal_marginal_loglik(Y, Λ, β, σ; kwargs...)

"""
    LognormalFit

Result of [`fit_lognormal_gllvm`](@ref): intercepts `β` (length `p`; mean of
`log y`), loadings `Λ` (`p×K`) on the log scale, residual SD `σ`
(`Var(log y) = σ²`), `link` (always `LogLink()`), maximised y-scale `loglik`
(includes Jacobian), `converged`, `iterations`, and free-σ reference packing
`theta_packed = [β; pack(Λ); log σ]` (Identity layout; fit path may profile σ).
"""
struct LognormalFit
    β::Vector{Float64}
    Λ::Matrix{Float64}
    σ::Float64
    link::Link
    loglik::Float64
    converged::Bool
    iterations::Int
    theta_packed::Vector{Float64}
end

function Base.show(io::IO, f::LognormalFit)
    p, K = size(f.Λ)
    print(io, "LognormalFit(p=", p, ", K=", K, ", σ=", round(f.σ; sigdigits = 4),
          ", link=", nameof(typeof(f.link)),
          ", loglik=", round(f.loglik; sigdigits = 7),
          f.converged ? "" : ", NOT CONVERGED", ")")
end

"""
    lognormal_response_mean(η, σ) -> Float64

Conditional response mean `E[y | η] = exp(η + σ²/2)` (twin bias correction).
`exp(η)` is the median, not the mean.
"""
lognormal_response_mean(η::Real, σ::Real) = exp(η + σ^2 / 2)

"""
    fit_lognormal_gllvm(Y; K, link=LogLink(), …) -> LognormalFit

Fit a one-part lognormal GLLVM (`log(y) ~ Normal(η, σ²)`, log link only) by
reusing the closed-form Gaussian fitter on centred log-responses. `Y` is a
`p×n` matrix of strictly positive responses; `K` the latent dimension.

Per-trait intercepts `β_t = mean_s (log(Y[t,s]) − offset[t,s])` are removed before
[`fit_gaussian_gllvm`](@ref) estimates `(Λ, σ)` on the centred log scale
(profile-admissible Identity path). Reported `loglik` is the y-scale marginal
at fitted `(β, Λ, σ)` including `−Σ log y`. Remaining keywords pass through to
`fit_gaussian_gllvm`.

`offset` is the known additive term on the log scale, `log(y) = β + offset + Λz + ε`
(for example log-exposure), as for the other families. It is subtracted from `log(Y)` before
the trait means are taken, so a constant offset is absorbed by the intercepts (same `loglik`,
`β` shifted by `−offset`), and `β` is the offset-free intercept. It takes the shapes
[`fit_gllvm`](@ref) documents (a scalar, a `p×n`, `1×n` or `p×1` matrix, a length-`p` vector;
anything else throws an `ArgumentError`), and a non-finite offset at any cell is refused.
This fitter does not support `mask` (passing one throws an `ArgumentError`): the trait means
and the y-scale `loglik` run over every cell.
"""
function fit_lognormal_gllvm(Y::AbstractMatrix{<:Real}; K::Integer,
        link::Link = LogLink(), offset = nothing, kwargs...)
    link isa LogLink || throw(ArgumentError(
        "fit_lognormal_gllvm: only LogLink is supported (twin lognormal)"))
    # `mask` would reach the Gaussian fit of the centred log-responses but not the trait means
    # or the `-Σ log y` term, which run over every cell, so a masked fit came back with the
    # wrong intercepts and logLik. Refuse it rather than return that.
    get(kwargs, :mask, nothing) === nothing || throw(ArgumentError(
        "fit_lognormal_gllvm: mask is not supported (the trait means and the y-scale " *
        "logLik run over every cell, so a masked fit would be silently wrong); drop the mask"))
    p, n = size(Y)
    # `offset` is made a p×n matrix here (see `_normalize_offset`); with no mask every cell is
    # observed, so a non-finite offset anywhere is refused.
    offset = _normalize_offset(offset, p, n; Y = Y, mask = nothing,
                               caller = "fit_lognormal_gllvm", maskable = false)
    all(>(0), Y) || throw(ArgumentError(
        "lognormal requires y > 0; found non-positive response"))
    Z = log.(Y)
    # η = β + offset + Λz is the mean of log(y), so it is log(y) − offset that is centred
    # and fitted; the intercepts then absorb a constant offset exactly.
    Zo = offset === nothing ? Z : Z .- offset
    β̂ = vec(sum(Zo; dims = 2)) ./ n
    R = Zo .- β̂
    gfit = fit_gaussian_gllvm(R; K = K, kwargs...)
    Λ̂ = Matrix{Float64}(gfit.pars.Λ)
    σ̂ = Float64(gfit.pars.σ_eps)
    ll = Float64(gfit.logLik - sum(Z))
    θ = vcat(collect(float.(β̂)), pack_lambda(Λ̂), log(σ̂))
    return LognormalFit(collect(float.(β̂)), Λ̂, σ̂, LogLink(), ll,
                        gfit.converged, gfit.n_iter, θ)
end
