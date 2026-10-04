# Per-trait link-implicit residual variance σ²_d on the latent (link) scale, plus
# the cross-family latent-scale extractors (`sigma_y_site`, `communality`,
# `correlation`) for the non-Gaussian one-part fits.
#
# For non-Gaussian responses each trait carries an implicit observation-level
# residual on the latent (link) scale. Putting σ²_d on the diagonal of the
# per-trait Σ_latent = ΛΛᵀ + diag(σ²_d) is what makes the loadings ΛΛᵀ — which
# live on the LINK scale for non-Gaussian families — comparable across families
# on a common latent scale. Without it, binomial/ordinal diagonals are too small
# relative to a Gaussian latent of comparable variance and the implied cross-
# trait correlations are inflated.
#
# This is the GLLVModels.jl twin of gllvmTMB's `link_residual_per_trait()`
# (gllvmTMB/R/extract-sigma.R) — the `link_residual = "auto"` convention. Each
# per-family formula below is confirmed against that R source.
#
# Per-family σ²_d (one entry per trait t; μ̂_t the per-trait mean fitted mean on
# the RESPONSE scale):
#   * Gaussian (identity)            : 0
#   * Binomial (logit)               : π²/3   (≈ 3.2899)
#   * Binomial (probit)              : 1
#   * Binomial (cloglog)             : π²/6   (≈ 1.6449)
#   * Poisson (log)                  : log(1 + 1/μ̂_t)         (lognormal-Poisson)
#   * NegativeBinomial (NB2, log)    : trigamma(r)            (r the Var=μ+μ²/r dispersion)
#   * Gamma (log)                    : trigamma(α)            (α the shape, Var=μ²/α)
#   * Beta (logit)                   : trigamma(μ̂_t φ) + trigamma((1−μ̂_t) φ)
#   * Ordinal (cumulative logit)     : π²/3
#
# References:
#   * Nakagawa & Schielzeth (2010) Biol. Rev. 85, 935–956 (the GLMM latent-scale
#     residual table; binomial-logit π²/3, NB/Gamma trigamma forms).
#   * Smithson & Verkuilen (2006) Psychol. Methods 11, 54–71 (the Beta-on-logit
#     delta-method residual trigamma(μφ) + trigamma((1−μ)φ)).
#   * McCullagh (1980) JRSS-B 42, 109–142 (the ordered cumulative-logit threshold
#     model: a standard logistic latent residual has variance π²/3).
#
# NOTE ON ORDINAL: gllvmTMB ships an ordinal_*probit* family (latent residual = 1
# by the standard-normal threshold construction). GLLVModels.jl's `OrdinalFit` supports
# BOTH links: the cumulative-*logit* model (`LogitLink()`, the default), whose
# latent residual is standard-logistic with variance π²/3, and the cumulative-
# *probit* model (`ProbitLink()`, matching gllvmTMB), whose standard-normal latent
# residual is exactly 1. Both are exact (no delta-method approximation); they
# differ only by the link's latent scale.

# ---------------------------------------------------------------------------
# Binomial link → constant σ²_d (μ̂-free). Confirmed: extract-sigma.R lines
# 156–162 (logit π²/3, probit 1, cloglog π²/6).
# ---------------------------------------------------------------------------
_binomial_link_residual(::LogitLink)   = π^2 / 3
_binomial_link_residual(::ProbitLink)  = 1.0
_binomial_link_residual(::CLogLogLink) = π^2 / 6
_binomial_link_residual(link::Link) = throw(ArgumentError(
    "Binomial link-residual is only defined for logit/probit/cloglog links; got $(typeof(link))."))

# Clamp a per-trait mean proportion away from {0, 1} before forming the Beta
# (a, b) = (μφ, (1−μ)φ) shape parameters. Mirrors the gllvmTMB Beta branch
# (extract-sigma.R line 230): a saturated fit with μ̂ → 0 or 1 would otherwise
# send one trigamma argument to ~0 and blow σ²_d up to ~1e24.
_clamp_mu_prop(μ) = clamp(μ, 1e-6, 1 - 1e-6)

# ---------------------------------------------------------------------------
# Family-marker core: σ²_d from (family marker, link, μ̂_t, dispersion).
#
# `μ̂` is the per-trait mean fitted mean on the response scale; `dispersion` is
# the family's scalar nuisance (NB2 r, Gamma shape α, Beta precision φ) and is
# ignored by families that do not use one. Returns a scalar σ²_d for one trait.
# AD-clean: only arithmetic + trigamma/log1p, so ForwardDiff Duals flow through.
# ---------------------------------------------------------------------------

# Gaussian (identity): σ²_d = 0 — the Gaussian σ_eps² already models the
# response-scale residual directly, so there is no LINK-implicit residual to add.
# extract-sigma.R fid 0 (line 140). Present so a mixed/Gaussian trait reduces
# cleanly and the Gaussian-reduction convention check has a method to call.
_link_residual_one(::Normal, ::IdentityLink, μ̂::Real, dispersion) = 0.0

# Poisson-log: lognormal-Poisson approximation log(1 + 1/μ̂). extract-sigma.R 171.
function _link_residual_one(::Poisson, ::LogLink, μ̂::Real, dispersion)
    return (isfinite(μ̂) && μ̂ > 0) ? log1p(1 / μ̂) : 0.0
end

# Binomial: constant per link (μ̂ / dispersion unused).
_link_residual_one(::Binomial, link::Link, μ̂::Real, dispersion) =
    _binomial_link_residual(link)

# NB2-log: trigamma(r), r the Var = μ + μ²/r dispersion. extract-sigma.R 188–191.
_link_residual_one(::NegativeBinomial, ::LogLink, μ̂::Real, dispersion::Real) =
    trigamma(max(dispersion, 1e-12))

# Gamma-log: trigamma(shape). extract-sigma.R 182–183 (nu_hat = 1/σ², the shape).
# GLLVModels.jl carries the shape α directly (Var = μ²/α), so dispersion == α.
_link_residual_one(::Gamma, ::LogLink, μ̂::Real, dispersion::Real) =
    trigamma(max(dispersion, 1e-12))

# Student-t (identity): σ²_d = σ²·ν/(ν−2) for ν > 2, the EXACT marginal variance of
# the scaled-t residual y − η = σ·t_ν (Var(t_ν) = ν/(ν−2); Johnson, Kotz &
# Balakrishnan 1995, Continuous Univariate Distributions vol. 2, ch. 28). The
# identity link puts this directly on the latent scale (μ̂ unused). For ν ≤ 2 the
# t has no finite variance, so there is no finite latent-scale residual variance;
# we return `Inf` to flag that (callers should treat a heavy-tailed ν ≤ 2 fit's
# latent-scale Σ as undefined rather than silently zero/clamped). `dispersion` is σ.
function _link_residual_one(f::StudentTFamily, ::IdentityLink, μ̂::Real, dispersion::Real)
    ν = f.ν
    (ν === nothing || ν <= 2) && return Inf
    return dispersion^2 * ν / (ν - 2)
end

# Beta-logit: trigamma(μ̂φ) + trigamma((1−μ̂)φ). extract-sigma.R 216–233.
function _link_residual_one(::Beta, ::LogitLink, μ̂::Real, dispersion::Real)
    μc = _clamp_mu_prop(μ̂)
    φ = dispersion
    a = max(μc * φ, 1e-12)
    b = max((1 - μc) * φ, 1e-12)
    return trigamma(a) + trigamma(b)
end

# Ordinal cumulative-logit: standard-logistic latent residual variance π²/3.
_link_residual_one(::Ordinal, ::LogitLink, μ̂::Real, dispersion) = π^2 / 3

# Ordinal cumulative-probit: standard-NORMAL threshold latent residual ⇒ σ²_d = 1
# (the variance of the standard-normal latent in the threshold construction). This
# matches gllvmTMB's ordinal-probit `fid`, whose latent residual is exactly 1.
_link_residual_one(::Ordinal, ::ProbitLink, μ̂::Real, dispersion) = 1.0

# ---------------------------------------------------------------------------
# Per-trait mean fitted mean μ̂ (response scale), one entry per trait.
#
# Reuses the fit's own `predict(...; type = :response)` so this is exactly the
# fitted mean each family reports — matching gllvmTMB's `mean(invlink(eta))` per
# trait. Ordinal predict returns the modal CLASS, not a response-scale mean, so
# its μ̂ is unused by `_link_residual_one(::Ordinal, …)`.
# ---------------------------------------------------------------------------
function _masked_trait_mean(μ::AbstractMatrix, mask)
    mask === nothing && return vec(Statistics.mean(μ; dims = 2))
    p, n = size(μ)
    out = zeros(Float64, p)
    @inbounds for t in 1:p
        cnt = 0
        acc = 0.0
        for s in 1:n
            mask[t, s] || continue
            cnt += 1
            acc += μ[t, s]
        end
        out[t] = cnt == 0 ? 0.0 : acc / cnt
    end
    return out
end

function _trait_mean_fitted(fit::PoissonFit, Y::AbstractMatrix; mask = nothing)
    if _has_lv_predictor(fit)
        # An X_lv fit cannot reconstruct per-site latent scores without X_lv (which
        # this latent-scale Σ extractor does not carry). The marginal per-trait mean
        # count is a consistent estimate of the Poisson rate for the link-implicit
        # residual scaling; the X_lv-route Σ is a point-estimate report only.
        return _masked_trait_mean(Float64.(Y), mask)
    end
    Z = getLV(fit, Y; rotate = false, mask = mask)
    η = fit.β .+ fit.Λ * Z'
    O = _laplace_prediction_offset(fit.offset, Y, nothing, mask, "link_residual")
    O === nothing || (η .+= O)        # the fitted means at the data include its offset
    μ = linkinv.(Ref(fit.link), η)
    return _masked_trait_mean(μ, mask)
end
function _trait_mean_fitted(fit::NBFit, Y::AbstractMatrix; mask = nothing)
    if _has_lv_predictor(fit)
        # X_lv fit: per-site scores need X_lv (absent here); the marginal per-trait
        # mean count is a consistent NB2 rate estimate for the link residual.
        return _masked_trait_mean(Float64.(Y), mask)
    end
    Z = getLV(fit, Y; rotate = false, mask = mask)
    η = fit.β .+ fit.Λ * Z'
    O = _laplace_prediction_offset(fit.offset, Y, nothing, mask, "link_residual")
    O === nothing || (η .+= O)        # the fitted means at the data include its offset
    μ = linkinv.(Ref(fit.link), η)
    return _masked_trait_mean(μ, mask)
end
function _trait_mean_fitted(fit::GammaFit, Y::AbstractMatrix; mask = nothing)
    if _has_lv_predictor(fit)
        # X_lv fit: per-site scores need X_lv (absent here); the marginal per-trait
        # mean response is a consistent Gamma rate estimate for the link residual.
        return _masked_trait_mean(Float64.(Y), mask)
    end
    Z = getLV(fit, Y; rotate = false, mask = mask)
    η = fit.β .+ fit.Λ * Z'
    μ = linkinv.(Ref(fit.link), η)
    return _masked_trait_mean(μ, mask)
end
function _trait_mean_fitted(fit::BetaFit, Y::AbstractMatrix; mask = nothing)
    if _has_lv_predictor(fit)
        # X_lv fit: per-site scores need X_lv (absent here); the marginal per-trait
        # mean proportion is a consistent Beta mean estimate for the link residual.
        return _masked_trait_mean(Float64.(Y), mask)
    end
    Z = getLV(fit, Y; rotate = false, mask = mask)
    η = fit.β .+ fit.Λ * Z'
    μ = linkinv.(Ref(fit.link), η)
    return _masked_trait_mean(μ, mask)
end
function _trait_mean_fitted(fit::BinomialFit, Y::AbstractMatrix; N = nothing, mask = nothing)
    Nm = N === nothing ? fill(1, size(Y)...) : N
    Z = getLV(fit, Y; N = Nm, rotate = false, mask = mask)
    η = fit.β .+ fit.Λ * Z'
    O = _laplace_prediction_offset(fit.offset, Y, nothing, mask, "link_residual")
    O === nothing || (η .+= O)        # the fitted means at the data include its offset
    μ = linkinv.(Ref(fit.link), η)
    return _masked_trait_mean(μ, mask)
end

# Scalar dispersion accessor per fit type (the family nuisance parameter).
_fit_dispersion(::PoissonFit)  = nothing
_fit_dispersion(::BinomialFit) = nothing
_fit_dispersion(fit::NBFit)    = fit.r
_fit_dispersion(fit::GammaFit) = fit.α
_fit_dispersion(fit::BetaFit)  = fit.φ
_fit_dispersion(fit::StudentTFit) = fit.σ
_fit_dispersion(::OrdinalFit)  = nothing
_fit_dispersion(::OrdinalPerTraitFit) = nothing

# Family marker per fit type (for dispatching `_link_residual_one`).
_fit_family(::PoissonFit)  = Poisson()
_fit_family(::BinomialFit) = Binomial()
_fit_family(fit::NBFit)    = NegativeBinomial(fit.r, 0.5)
_fit_family(fit::GammaFit) = Gamma(fit.α, 1.0)
_fit_family(fit::BetaFit)  = Beta(fit.φ, 1.0)
_fit_family(fit::StudentTFit) = StudentTFamily(fit.ν isa Vector ? fit.ν[1] : fit.ν, fit.σ isa Vector ? fit.σ[1] : fit.σ)
_fit_family(::OrdinalFit)  = Ordinal()
_fit_family(::OrdinalPerTraitFit) = Ordinal()

# ---------------------------------------------------------------------------
# Public API: link_residual.
# ---------------------------------------------------------------------------

"""
    link_residual(family, link, μ̂, dispersion) -> Float64

Per-trait link-implicit residual variance σ²_d on the latent (link) scale for a
single trait, given the response `family` marker, the `link`, the trait's mean
fitted mean `μ̂` (response scale), and the family's scalar `dispersion` (the NB2
dispersion `r`, the Gamma shape `α`, or the Beta precision `φ`; pass `nothing`
for families without one). This is the GLLVModels.jl twin of gllvmTMB's
`link_residual_per_trait` (`link_residual = "auto"`).

Per-family forms (each confirmed against gllvmTMB/R/extract-sigma.R):

| family / link                     | σ²_d                                     |
|:----------------------------------|:-----------------------------------------|
| `Normal`, `IdentityLink`          | 0                                        |
| `Binomial`, `LogitLink`           | π²/3                                      |
| `Binomial`, `ProbitLink`          | 1                                        |
| `Binomial`, `CLogLogLink`         | π²/6                                      |
| `Poisson`, `LogLink`              | log(1 + 1/μ̂)                             |
| `NegativeBinomial`, `LogLink`     | trigamma(r)                              |
| `Gamma`, `LogLink`                | trigamma(α)                             |
| `Beta`, `LogitLink`               | trigamma(μ̂ φ) + trigamma((1−μ̂) φ)       |
| `Ordinal`, `LogitLink`            | π²/3                                      |

The Beta branch clamps μ̂ to `[1e-6, 1−1e-6]` before forming `(μ̂φ, (1−μ̂)φ)`
(mirrors the R source) to keep a saturated fit's σ²_d finite.
"""
link_residual(family, link::Link, μ̂::Real, dispersion) =
    _link_residual_one(family, link, μ̂, dispersion)

"""
    link_residual(fit, Y; N=nothing) -> Vector{Float64}

Per-trait link-implicit residual variance σ²_d (length `p`) for a fitted
non-Gaussian GLLVM (`PoissonFit`, `BinomialFit`, `NBFit`, `BetaFit`, `GammaFit`,
or `OrdinalFit`). `Y` is the response matrix the fit was computed on (the fits do
not store the data); `N` (Binomial only) the trial counts.

Each entry uses the fit's own link, scalar dispersion, and per-trait mean fitted
mean `μ̂_t = mean_s predict(fit, Y; type = :response)[t, s]` (response scale).
Returns the vector added to `diag(ΛΛᵀ)` to form the latent-scale
`Σ_latent = ΛΛᵀ + diag(σ²_d)` (see [`sigma_y_site`](@ref)). Rotation-invariant
and family-agnostic on the latent scale (matches gllvmTMB `link_residual="auto"`).
"""
function link_residual(fit::Union{PoissonFit, NBFit, BetaFit, GammaFit}, Y::AbstractMatrix;
                       mask = nothing)
    link = fit.link
    fam  = _fit_family(fit)
    disp = _fit_dispersion(fit)
    μ̂    = _trait_mean_fitted(fit, Y; mask = mask)
    return [Float64(_link_residual_one(fam, link, μ̂[t], disp)) for t in eachindex(μ̂)]
end

function link_residual(fit::BinomialFit, Y::AbstractMatrix;
                       N::Union{Nothing, AbstractMatrix} = nothing, mask = nothing)
    link = fit.link
    p = size(fit.Λ, 1)
    # Binomial σ²_d is μ̂-free, so we don't need the fitted mean; one value per trait.
    v = _binomial_link_residual(link)
    return fill(Float64(v), p)
end

function link_residual(fit::OrdinalFit, Y::AbstractMatrix; mask = nothing)
    p = size(fit.Λ, 1)
    # Cumulative threshold residual, μ̂-free (no species intercept, latent η has
    # zero mean by construction): π²/3 for the logit link (standard-logistic
    # latent), 1 for the probit link (standard-normal latent).
    return fill(Float64(_link_residual_one(Ordinal(), fit.link, 0.0, nothing)), p)
end
function link_residual(fit::OrdinalPerTraitFit, Y::AbstractMatrix; mask = nothing)
    p = size(fit.Λ, 1)
    return fill(Float64(_link_residual_one(Ordinal(), fit.link, 0.0, nothing)), p)
end

# ===========================================================================
# Latent-scale cross-family extractors for the non-Gaussian one-part fits.
#
# These are ADDITIVE methods of `sigma_y_site`, `communality`, and
# `correlation` for `PoissonFit`, `BinomialFit`, `NBFit`, `BetaFit`,
# `GammaFit`, and `OrdinalFit`. The Gaussian `correlation(::GllvmFit)` in
# confint_derived.jl is left UNCHANGED.
#
# For a non-Gaussian family the loadings ΛΛᵀ live on the LINK (latent) scale, so
# we put each trait on a common latent scale by adding a per-family link-implicit
# residual variance σ²_d (above) to the diagonal:
#
#     Σ_latent = Λ Λᵀ + diag(σ²_d)
#     correlation = D^{-1/2} Σ_latent D^{-1/2},   D = diag(Σ_latent)
#     communality = diag(Λ Λᵀ) / diag(Σ_latent)
#
# This mirrors gllvmTMB's `extract_Sigma(..., link_residual = "auto")` (there is
# no `unique()` Ψ component in these single-tier non-Gaussian fits, so Ψ = 0 and
# Σ_latent = ΛΛᵀ + diag(σ²_d) exactly). The construction is ROTATION-INVARIANT
# (ΛΛᵀ is) and family-agnostic on the latent scale. The fits do not store the
# data, so the response matrix `Y` (and trial counts `N` for Binomial) must be
# passed — exactly the matrix the fit was computed on.
# ===========================================================================

# Union of the one-part non-Gaussian fit types that share the ΛΛᵀ + diag(σ²_d)
# latent-scale construction. (Ordinal and Binomial are listed in their own method
# signatures below because they take/forward different keyword args.)
const _NonGaussianLatentFit = Union{PoissonFit, NBFit, BetaFit, GammaFit}

# Assemble the symmetric latent-scale Σ = ΛΛᵀ + diag(σ²_d) from a loadings matrix
# and a per-trait residual vector.
function _latent_sigma(Λ::AbstractMatrix, σ²_d::AbstractVector)
    A = Λ * Λ'
    @inbounds for t in eachindex(σ²_d)
        A[t, t] += σ²_d[t]
    end
    return (A + A') ./ 2
end

# Safe ratio with an explicit-NaN denominator floor: returns `num/den` for
# `den > 0`, else `NaN`. Used by the latent-scale correlation and communality so a
# degenerate Σ_tt ≤ 0 (non-PD assembled covariance) yields an explicit NaN rather
# than a silent Inf/NaN from a division. Behaviour-preserving for all valid PSD
# inputs (Σ_tt > 0 returns the exact same value).
_safe_ratio(num::Real, den::Real) = den > 0 ? num / den : NaN

# Standardise a covariance to a correlation: R[i,j] = Σ[i,j]/√(Σ[i,i]Σ[j,j]),
# with the denominator floor — any Σ_tt ≤ 0 makes the affected row/column NaN.
function _latent_correlation(Σ::AbstractMatrix)
    p = size(Σ, 1)
    R = similar(Σ, Float64)
    @inbounds for j in 1:p, i in 1:p
        d = Σ[i, i] * Σ[j, j]
        R[i, j] = (Σ[i, i] > 0 && Σ[j, j] > 0) ? Σ[i, j] / sqrt(d) : NaN
    end
    return R
end

"""
    sigma_y_site(fit, Y; N=nothing) -> Matrix

Latent-scale trait covariance `Σ_latent = Λ Λᵀ + diag(σ²_d)` for a fitted
non-Gaussian GLLVM (`PoissonFit`, `BinomialFit`, `NBFit`, `BetaFit`, `GammaFit`,
`OrdinalFit`). The loadings `Λ Λᵀ` are on the LINK scale; the per-trait
link-implicit residual `σ²_d` (see [`link_residual`](@ref)) puts all traits on a
common latent scale. `Y` is the response matrix the fit was computed on; `N`
(Binomial only) the trial counts. The construction is rotation-invariant and
matches gllvmTMB `extract_Sigma(..., link_residual = "auto")` with no `unique()`
component (Ψ = 0).
"""
function sigma_y_site(fit::_NonGaussianLatentFit, Y::AbstractMatrix; mask = nothing)
    return _latent_sigma(fit.Λ, link_residual(fit, Y; mask = mask))
end
function sigma_y_site(fit::BinomialFit, Y::AbstractMatrix;
                      N::Union{Nothing, AbstractMatrix} = nothing, mask = nothing)
    return _latent_sigma(fit.Λ, link_residual(fit, Y; N = N, mask = mask))
end
function sigma_y_site(fit::OrdinalFit, Y::AbstractMatrix; mask = nothing)
    return _latent_sigma(fit.Λ, link_residual(fit, Y; mask = mask))
end
function sigma_y_site(fit::OrdinalPerTraitFit, Y::AbstractMatrix; mask = nothing)
    return _latent_sigma(fit.Λ, link_residual(fit, Y; mask = mask))
end

"""
    communality(fit, Y; N=nothing) -> Vector

Per-trait communality `c²[t] = (Λ Λᵀ)[t,t] / Σ_latent[t,t]` on the latent scale
for a fitted non-Gaussian GLLVM — the share of the latent-scale trait variance
carried by the shared loadings, with `Σ_latent = Λ Λᵀ + diag(σ²_d)` (see
[`sigma_y_site`](@ref)). Values are in [0, 1]. `Y` is the response matrix the fit
was computed on; `N` (Binomial only) the trial counts.
"""
function communality(fit::_NonGaussianLatentFit, Y::AbstractMatrix; mask = nothing)
    Λ = fit.Λ
    ΛΛt = Λ * Λ'
    Σ = sigma_y_site(fit, Y; mask = mask)
    return [_safe_ratio(ΛΛt[t, t], Σ[t, t]) for t in 1:size(Λ, 1)]
end
function communality(fit::BinomialFit, Y::AbstractMatrix;
                     N::Union{Nothing, AbstractMatrix} = nothing, mask = nothing)
    Λ = fit.Λ
    ΛΛt = Λ * Λ'
    Σ = sigma_y_site(fit, Y; N = N, mask = mask)
    return [_safe_ratio(ΛΛt[t, t], Σ[t, t]) for t in 1:size(Λ, 1)]
end
function communality(fit::OrdinalFit, Y::AbstractMatrix; mask = nothing)
    Λ = fit.Λ
    ΛΛt = Λ * Λ'
    Σ = sigma_y_site(fit, Y; mask = mask)
    return [_safe_ratio(ΛΛt[t, t], Σ[t, t]) for t in 1:size(Λ, 1)]
end
function communality(fit::OrdinalPerTraitFit, Y::AbstractMatrix; mask = nothing)
    Λ = fit.Λ
    ΛΛt = Λ * Λ'
    Σ = sigma_y_site(fit, Y; mask = mask)
    return [_safe_ratio(ΛΛt[t, t], Σ[t, t]) for t in 1:size(Λ, 1)]
end

"""
    correlation(fit, Y; N=nothing) -> Matrix

Latent-scale cross-trait correlation `R = D^{-1/2} Σ_latent D^{-1/2}` for a
fitted non-Gaussian GLLVM, with `Σ_latent = Λ Λᵀ + diag(σ²_d)` (see
[`sigma_y_site`](@ref)). Diagonal entries are exactly 1.0; off-diagonals are in
[-1, 1] and driven by the shared loadings on the common latent (link) scale. The
construction is rotation-invariant and family-agnostic (matches gllvmTMB
`link_residual = "auto"`). `Y` is the response matrix the fit was computed on;
`N` (Binomial only) the trial counts.

This is the non-Gaussian twin of [`correlation(::GllvmFit)`](@ref); for the
Gaussian family the response and latent scales coincide (σ²_d = 0, the residual
is the Gaussian σ²_eps), so no `Y` argument is needed there.
"""
function correlation(fit::_NonGaussianLatentFit, Y::AbstractMatrix; mask = nothing)
    return _latent_correlation(sigma_y_site(fit, Y; mask = mask))
end
function correlation(fit::BinomialFit, Y::AbstractMatrix;
                     N::Union{Nothing, AbstractMatrix} = nothing, mask = nothing)
    return _latent_correlation(sigma_y_site(fit, Y; N = N, mask = mask))
end
function correlation(fit::OrdinalFit, Y::AbstractMatrix; mask = nothing)
    return _latent_correlation(sigma_y_site(fit, Y; mask = mask))
end
function correlation(fit::OrdinalPerTraitFit, Y::AbstractMatrix; mask = nothing)
    return _latent_correlation(sigma_y_site(fit, Y; mask = mask))
end

# Student-t: σ²_d = σ²·ν/(ν−2) is μ̂-free (identity link), so no per-site mode solve
# is needed. Documented at `_link_residual_one(::StudentTFamily, …)` above (Inf when
# ν ≤ 2, where the t has no finite variance).
function link_residual(fit::StudentTFit, Y::AbstractMatrix)
    p = size(fit.Λ, 1)
    σ_vec = fit.σ isa Real ? fill(fit.σ, p) : fit.σ
    ν_vec = fit.ν isa Real ? fill(fit.ν, p) : fit.ν
    return [_link_residual_one(StudentTFamily(ν_vec[t], σ_vec[t]), fit.link, 0.0, σ_vec[t]) for t in 1:p]
end
