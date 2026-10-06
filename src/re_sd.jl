# Posterior standard deviations for the latent random effects (latent_score_sd,
# renamed from getREsd — see the R-SURFACE SHADOWING NOTE below).
#
# STATISTICAL SCOPE, STATED EXPLICITLY: every quantity this file returns is a
# CONDITIONAL-ON-θ̂ standard deviation of the latent factor scores z at the
# fitted mode — sqrt(diag(Cov(z | y, θ̂))). It does NOT propagate uncertainty
# in θ̂ itself (the fixed effects, loadings, dispersion, ...).
#
# HONEST NAME-COLLISION NOTE: this is NOT "exactly TMB's sdreport() convention"
# for random effects. TMB's DEFAULT `sdreport()` on a `random`-declared block
# propagates the fixed-effect (θ̂) uncertainty into the reported random-effect
# SD (a marginal-variance quantity that can diverge by an order of magnitude
# from the conditional one near boundaries) — see the R readback
# `.unlazy/core070-aghq/oracle-source/readback/R/re-uncertainty.R:62-80`.
# What this file computes is TMB's `ignore.parm.uncertainty = TRUE` variant:
# Cov(z | y, θ̂) treating θ̂ as fixed, exactly the Laplace curvature at the
# mode (Kristensen, Nielsen, Berg, Skaug & Bell 2016, J Stat Soft 70(5), §2.3)
# with NO fixed-effect propagation term added on top. Coverage of an interval
# built from these SDs has not been separately measured here.
#
# R-SURFACE SHADOWING NOTE: R's `getREsd(fit, block = ...)` (also `.unlazy/
# core070-aghq/oracle-source/readback/R/re-uncertainty.R`) covers AUXILIARY
# random-effect blocks (random slopes, grouping-factor intercepts, ...) and
# explicitly routes the latent FACTOR SCORES to `getLV(se = TRUE)` instead.
# Julia's `latent_score_sd(fit, y)` (renamed from `getREsd` — maintainer
# decision docs/dev-log/decisions/2026-09-01-maintainer-decisions-round2-3.md
# #5 — freeing the `getREsd` name for a future true mirror of R's auxiliary-
# RE-block surface) here returns exactly the latent-score SDs — the quantity
# R's `getREsd` does NOT cover and R's `getLV(se=TRUE)` does.
#
# TWO EXACT CASES, per family class:
#
# 1. Gaussian closed-form (GllvmFit). The conditional distribution of the
#    latent score z_s given y_s and θ̂ is EXACTLY Gaussian (see the marginal
#    covariance identity in src/likelihood.jl and the posterior-mean formula
#    already used by `getLV` in src/postfit.jl):
#        z_s | y_s, θ̂ ~ N(m_s, M⁻¹),   M = I_K + Λ' Ψ⁻¹ Λ,   Ψ = Σ_y − ΛΛ'
#    `M` does not depend on the site `s` (only `Σ_y`, `Λ` do, both fixed at
#    θ̂), so every site shares the SAME K×K conditional covariance. This is
#    an EXACT (not Laplace-approximate) identity — cross-checked in
#    test/test_se_machinery.jl against a direct dense computation of the
#    same M⁻¹ to machine precision.
#
# 2. Dense-Laplace non-Gaussian families (Binomial, Poisson, NegativeBinomial,
#    Gamma, Beta — the families whose mode-finder lives in
#    src/families/laplace.jl). The Laplace approximation already treats
#        z_s | y_s, θ̂ ≈ N(ẑ_s, A_s⁻¹),   A_s = Λ' W_s Λ + I_K
#    where `W_s` is the per-cell curvature wrt η that
#    `laplace_loglik_site` (src/families/laplace.jl) already assembles for
#    its log-det term — Fisher (`_glm_weight`) or observed
#    (`_glm_obs_weight`), matching whichever curvature that fit's objective
#    actually used (`fit.hessian` when the fit type records it, else the
#    family's `_default_hessian`). `A_s` VARIES by site (unlike the Gaussian
#    case) because `W_s` depends on the fitted mean μ_s = linkinv(η_s).
#
# Refs: Kristensen et al. 2016 (TMB); Louis 1982 (observed-information
# decomposition, for context — not used directly here since these are
# random-effect, not fixed-effect, SDs).

using LinearAlgebra

# ---------------------------------------------------------------------------
# Gaussian closed-form path.
# ---------------------------------------------------------------------------

"""
    latent_score_sd(fit::GllvmFit, y; X=nothing, rotate=true) -> Matrix{Float64}

Posterior (conditional-on-θ̂) standard deviations of the latent factor scores
for a Gaussian GLLVM fit. Returns an `n_sites × K` matrix. The Gaussian
conditional covariance `M = I_K + Λ' Ψ⁻¹ Λ` (`Ψ = Σ_y − ΛΛ'`) does not
depend on the site, so every row is identical.

`rotate` (default `true`) MATCHES `getLV`'s own default: the SDs are
reported in the same canonical SVD-rotated basis (`sqrt.(diag(R' M⁻¹ R))`,
`R = `[`rotation`](@ref)`(fit)`) that `getLV(fit, y; rotate=true)` (the
default) uses for its conditional MEAN. Pass `rotate=false` for the
unrotated `sqrt.(diag(M⁻¹))`, paired with `getLV(fit, y; rotate=false)`.
Mixing `rotate=true` scores from one call with `rotate=false` SDs from the
other (or vice versa) puts the point estimate and its SD in different
bases — always request the same `rotate` value from both.

`y` (and `X`, if the fit used fixed effects) is required to reconstruct
`Σ_y`'s scale in the same way `getLV` does; only its size is used here (the
covariance does not depend on the response values), but the argument is kept
for interface parity with `getLV` and to guard against a site-count
mismatch with a caller's other post-fit calls.

These SDs are EXACT (not Laplace-approximate) given θ̂ — see the module
docstring above for the derivation and the accompanying finite/closed-form
test.

STRUCTURAL SCOPE: refuses (with an honest `ArgumentError`) for fits this
closed-form identity does not cover — a phylogenetic block (`K_phy > 0` or
`has_phy_unique`; the phylo effect is shared across sites, breaking the
per-site-independent conditioning this formula assumes), a `K_W` (within/
unit_obs) tier, or a masked/offset/AGHQ Gaussian-record fit (`Σ_y_site` from
[`sigma_y_site`](@ref) does not reconstruct the record objective's actual
per-cell covariance for those fits).
"""
function latent_score_sd(fit::GllvmFit, y::AbstractMatrix;
                 X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                 rotate::Bool = true)
    _has_gaussian_record(fit) && throw(ArgumentError(
        "latent_score_sd is not implemented for masked/offset/AGHQ Gaussian-record " *
        "fits; sigma_y_site does not reconstruct the record objective's " *
        "actual per-cell covariance for these fits"))
    (fit.model.K_phy > 0 || fit.model.has_phy_unique) && throw(ArgumentError(
        "latent_score_sd is not implemented for fits with a phylogenetic block " *
        "(K_phy > 0 or has_phy_unique = true); the phylo effect is shared " *
        "across sites, which breaks the per-site-independent conditioning " *
        "this closed-form identity assumes"))
    fit.model.K_W > 0 && throw(ArgumentError(
        "latent_score_sd is not implemented for fits with a K_W (within/unit_obs) " *
        "latent tier"))
    n = size(y, 2)
    Λ = fit.pars.Λ
    K = size(Λ, 2)
    Σ = sigma_y_site(fit)
    Ψ = Σ - Λ * Λ'
    ΨiΛ = Ψ \ Λ
    M = Symmetric(I + Λ' * ΨiΛ)
    Minv = inv(M)
    if rotate
        R = _svd_rotation(Λ)
        Minv = Symmetric(R' * Minv * R)
    end
    sd = sqrt.(max.(diag(Minv), 0.0))          # length K, shared by every site
    return repeat(reshape(sd, 1, K), n, 1)     # n × K
end

# ---------------------------------------------------------------------------
# Dense-Laplace non-Gaussian path (shared kernel).
# ---------------------------------------------------------------------------

# Per-site conditional precision A_s = Λ' W_s Λ + I_K, reusing the SAME
# W-selection logic `laplace_loglik_site` (src/families/laplace.jl) uses for
# its log-det term — the "A matrix the Laplace code already factors".
function _laplace_re_precision_site(family, y::AbstractVector, n::AbstractVector,
        Λ::AbstractMatrix, β::AbstractVector, link::Link, z::AbstractVector;
        mask = nothing, offset = nothing, hessian::Symbol = _default_hessian(family, link))
    (hessian === :fisher || hessian === :observed) || throw(ArgumentError(
        "hessian must be :fisher or :observed; got :$hessian"))
    p = size(Λ, 1)
    K = size(Λ, 2)
    off = offset === nothing ? false : offset
    Λz = Λ * z
    η  = _clamp_eta.(β .+ off .+ Λz)
    μ  = _clamp_mu.(Ref(family), linkinv.(Ref(link), η))
    me = mu_eta.(Ref(link), η)
    W  = if hessian === :fisher || _glm_weight_matches_observed(family, link)
        _glm_weight.(Ref(family), μ, n, me)
    else
        [(mask === nothing || mask[t]) ?
            _glm_obs_weight(family, μ[t], n[t], me[t], y[t], link, η[t]) : 0.0
         for t in 1:p]
    end
    if mask !== nothing
        W = ifelse.(mask, W, 0.0)
    end
    WΛ = W .* Λ
    Amat = Λ' * WΛ
    @inbounds for d in 1:K
        Amat[d, d] += 1.0
    end
    return Symmetric(Amat)
end

# n_sites × K matrix of per-site conditional SDs sqrt(diag(A_s⁻¹)); Amat is
# singular only if a family/link's curvature ever goes badly negative at the
# mode (documented as a NaN row rather than throwing, matching the
# `bootstrap_ci` convention of surfacing non-finite results rather than
# aborting the whole call).
function _laplace_re_sd(family, Y::AbstractMatrix, N::AbstractMatrix,
        Λ::AbstractMatrix, β::AbstractVector, link::Link;
        mask = nothing, offset = nothing, hessian::Symbol = _default_hessian(family, link))
    p, n = size(Y)
    K = size(Λ, 2)
    out = Matrix{Float64}(undef, n, K)
    @inbounds for s in 1:n
        yi = view(Y, :, s)
        ni = view(N, :, s)
        mi = mask === nothing ? nothing : view(mask, :, s)
        oi = offset === nothing ? nothing : view(offset, :, s)
        z = _laplace_mode(family, yi, ni, Λ, β, link; mask = mi, offset = oi)
        A = _laplace_re_precision_site(family, yi, ni, Λ, β, link, z;
                                       mask = mi, offset = oi, hessian = hessian)
        Ainv = try
            inv(A)
        catch
            fill(NaN, K, K)
        end
        d = diag(Ainv)
        for k in 1:K
            out[s, k] = (isfinite(d[k]) && d[k] > 0) ? sqrt(d[k]) : NaN
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Per-family public methods.
#
# Scope, stated honestly: covers the five dense-Laplace families whose fit
# structs carry a common (β, Λ, link[, dispersion]) shape and a plain
# `_laplace_mode` mode-finder (Binomial, Poisson, NegativeBinomial, Gamma,
# Beta). AGHQ-integrated fits, Ordinal (cutpoint-based, no plain per-site
# `_laplace_mode` signature), and every other family in src/families/ are
# NOT covered — calling `latent_score_sd` on them is a `MethodError`, which is the
# honest failure mode rather than a silent wrong answer. Predictor-informed
# (`X_lv`/`alpha_lv`) fits within the five covered families ALSO refuse, with
# an honest `ArgumentError` (this method does not thread the fitted-mean
# offset `Λ·(X_lv·α_lv)'` through `_laplace_mode`/`_laplace_re_precision_site`
# the way `getLV` does for those fits — see src/postfit.jl).
# ---------------------------------------------------------------------------

"""
    latent_score_sd(fit::BinomialFit, Y; N=nothing, mask=nothing, offset=nothing) -> Matrix{Float64}

Laplace-approximate conditional SDs of the per-site latent scores at the
fitted mode `ẑ_s`: `sqrt(diag((Λ' W_s Λ + I)⁻¹))`, with `W_s` the SAME
per-cell curvature `laplace_loglik_site` (src/families/laplace.jl) uses for
its log-det term at this fit's own `fit.hessian` (Fisher or observed).
Returns an `n_sites × K` matrix. Conditional on θ̂ — see the module
docstring in src/re_sd.jl.
"""
function latent_score_sd(fit::BinomialFit, Y::AbstractMatrix;
                 N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                 mask = nothing, offset = nothing)
    _is_binomial_aghq(fit) && throw(ArgumentError(
        "latent_score_sd is not implemented for AGHQ-integrated binomial fits"))
    _has_lv_predictor(fit) && throw(ArgumentError(
        "latent_score_sd is not implemented for predictor-informed (X_lv) fits: " *
        "the mode/curvature at each site needs the latent-mean offset " *
        "Λ·(X_lv·α_lv)' threaded through, which this method does not do"))
    eltype(Y) <: Integer || throw(ArgumentError("latent_score_sd requires integer responses"))
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    return _laplace_re_sd(Binomial(), Y, Nm, fit.Λ, fit.β, fit.link;
                          mask = mask, offset = offset, hessian = fit.hessian)
end

"""
    latent_score_sd(fit::PoissonFit, Y; N=nothing, mask=nothing, offset=nothing) -> Matrix{Float64}

Laplace-approximate conditional SDs of the per-site latent scores; see
[`latent_score_sd(::BinomialFit, ::AbstractMatrix)`](@ref).
"""
function latent_score_sd(fit::PoissonFit, Y::AbstractMatrix;
                 N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                 mask = nothing, offset = nothing)
    _is_poisson_aghq(fit) && throw(ArgumentError(
        "latent_score_sd is not implemented for AGHQ-integrated Poisson fits"))
    fit.weights === nothing || throw(ArgumentError(
        "latent_score_sd is not implemented for weighted Poisson fits"))
    _has_lv_predictor(fit) && throw(ArgumentError(
        "latent_score_sd is not implemented for predictor-informed (X_lv) fits: " *
        "the mode/curvature at each site needs the latent-mean offset " *
        "Λ·(X_lv·α_lv)' threaded through, which this method does not do"))
    eltype(Y) <: Integer || throw(ArgumentError("latent_score_sd requires integer responses"))
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    return _laplace_re_sd(Poisson(), Y, Nm, fit.Λ, fit.β, fit.link;
                          mask = mask, offset = offset, hessian = fit.hessian)
end

"""
    latent_score_sd(fit::NBFit, Y; N=nothing, mask=nothing, offset=nothing) -> Matrix{Float64}

Laplace-approximate conditional SDs of the per-site latent scores at the
fitted dispersion `r`; see [`latent_score_sd(::BinomialFit, ::AbstractMatrix)`](@ref).
"""
function latent_score_sd(fit::NBFit, Y::AbstractMatrix{<:Integer};
                 N::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
                 mask = nothing, offset = nothing)
    _has_lv_predictor(fit) && throw(ArgumentError(
        "latent_score_sd is not implemented for predictor-informed (X_lv) fits: " *
        "the mode/curvature at each site needs the latent-mean offset " *
        "Λ·(X_lv·α_lv)' threaded through, which this method does not do"))
    p, n = size(Y)
    Nm = N === nothing ? fill(1, p, n) : N
    return _laplace_re_sd(NegativeBinomial(fit.r, 0.5), Y, Nm, fit.Λ, fit.β, fit.link;
                          mask = mask, offset = offset, hessian = fit.hessian)
end

"""
    latent_score_sd(fit::GammaFit, Y; mask=nothing, offset=nothing) -> Matrix{Float64}

Laplace-approximate conditional SDs of the per-site latent scores at the
fitted shape `α`; see [`latent_score_sd(::BinomialFit, ::AbstractMatrix)`](@ref).
Gamma defaults to the `:observed` log-det curvature (see
src/families/laplace.jl `_default_hessian(::Gamma, ::LogLink)`), so this
uses the observed curvature by default too — the SD matches whatever
curvature `fit.hessian` actually recorded.
"""
function latent_score_sd(fit::GammaFit, Y::AbstractMatrix{<:Real};
                 mask = nothing, offset = nothing)
    _has_lv_predictor(fit) && throw(ArgumentError(
        "latent_score_sd is not implemented for predictor-informed (X_lv) fits: " *
        "the mode/curvature at each site needs the latent-mean offset " *
        "Λ·(X_lv·α_lv)' threaded through, which this method does not do"))
    p, n = size(Y)
    N1 = ones(Int, p, n)
    return _laplace_re_sd(Gamma(fit.α, 1.0), Y, N1, fit.Λ, fit.β, fit.link;
                          mask = mask, offset = offset, hessian = fit.hessian)
end

"""
    latent_score_sd(fit::BetaFit, Y; mask=nothing, offset=nothing) -> Matrix{Float64}

Laplace-approximate conditional SDs of the per-site latent scores at the
fitted precision `φ`; see [`latent_score_sd(::BinomialFit, ::AbstractMatrix)`](@ref).
"""
function latent_score_sd(fit::BetaFit, Y::AbstractMatrix{<:Real};
                 mask = nothing, offset = nothing)
    _has_lv_predictor(fit) && throw(ArgumentError(
        "latent_score_sd is not implemented for predictor-informed (X_lv) fits: " *
        "the mode/curvature at each site needs the latent-mean offset " *
        "Λ·(X_lv·α_lv)' threaded through, which this method does not do"))
    p, n = size(Y)
    N1 = ones(Int, p, n)
    return _laplace_re_sd(Beta(fit.φ, 1.0), Y, N1, fit.Λ, fit.β, fit.link;
                          mask = mask, offset = offset, hessian = fit.hessian)
end

# ---------------------------------------------------------------------------
# Deprecated forwarding shim. `getREsd` is renamed to `latent_score_sd`
# (maintainer decision docs/dev-log/decisions/2026-09-01-maintainer-decisions-
# round2-3.md #5) so the `getREsd` name is free for a future true mirror of
# R's `getREsd(fit, block=)` auxiliary-random-effect-block surface, which
# this function has never covered — see the module docstring above.
# ---------------------------------------------------------------------------

function getREsd(args...; kwargs...)
    Base.depwarn(
        "getREsd is deprecated: renamed to latent_score_sd; the name getREsd is reserved " *
        "for a future R-mirroring surface", :getREsd)
    return latent_score_sd(args...; kwargs...)
end
