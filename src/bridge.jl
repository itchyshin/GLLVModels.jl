# R -> Julia bridge entry point for GLLVModels.jl, exposing the fitted one-part families
# to R via JuliaCall (the canonical transport across the drmTMB<->DRM.jl ecosystem).
#
# `bridge_fit` accepts plain matrices + family strings (the R side, gllvmTMB, owns
# formula parsing; GLLVModels.jl owns the numerical fit) and returns a FLAT NamedTuple of
# JuliaCall-convertible primitives only (Float64 scalars/arrays, Ints, Strings,
# Bools) — no Julia struct ever crosses the language boundary.
#
# Contract keys:
#   family       :: String            — requested family
#   families     :: Vector{String}    — per-trait family (length p; repeated)
#   model        :: String            — short model tag ("poisson_rr", ...)
#   d            :: Int               — latent dimension K
#   n_traits     :: Int               — p (rows of y)
#   n_units      :: Int               — n (columns of y)
#   trait_names  :: Vector{String}    — length p
#   unit_names   :: Vector{String}    — length n
#   loadings     :: Matrix{Float64}   — p x d rotated loadings
#   alpha        :: Vector{Float64}   — per-trait intercept (link scale; NaN for Ordinal)
#   dispersion   :: Vector{Float64}   — per-trait nuisance (r/phi/alpha; NaN if none)
#   dispersion_group        :: Vector{Float64} — optional grouped-dispersion values
#   dispersion_group_id     :: Vector{Int}     — optional per-trait group id
#   dispersion_parameter    :: String          — optional engine parameter name
#   dispersion_engine_scale :: String          — optional engine variance rule
#   dispersion_public_scale :: String          — optional R/gllvmTMB public map
#   sigma_eps    :: Float64           — Gaussian residual SD (NaN otherwise)
#   Sigma        :: Matrix{Float64}   — p x p latent-scale trait covariance
#   correlation  :: Matrix{Float64}   — p x p latent-scale trait correlation
#   communality  :: Vector{Float64}   — per-trait communality c^2 (length p)
#   scores       :: Matrix{Float64}   — n x d latent scores (0x0 if unavailable)
#   loglik       :: Float64
#   aic, bic     :: Float64
#   df           :: Int               — free-parameter count for AIC
#   nobs         :: Int               — observed cells (p*n for complete data)
#   converged    :: Bool
#   gradient_max :: Float64           — max-abs gradient of the fit's packed NLL at
#                                        the fitted parameters (ACC-BRIDGE-GRADIENT);
#                                        NaN when no packed objective can be rebuilt
#                                        for this fit (documented per family below)
#   iterations   :: Int
#   message      :: String
#   link         :: Vector{String}    — per-trait link name
#   note         :: String            — caveats for the R side
#
# Optional coefficient keys:
#   mean_coef    :: Vector{Float64}   — Gaussian-X full mean coefficient vector
#   mean_coef_status :: Vector{String} — "estimated"/"fixed" for mean_coef
#   beta_cov     :: Vector{Float64}   — non-Gaussian-X per-trait intercepts
#   gamma        :: Vector{Float64}   — non-Gaussian-X shared covariate slopes
#   gamma_status :: Vector{String}    — "estimated"/"fixed" for gamma
#   beta_zero    :: Vector{Float64}   — zero-inflated structural-zero logits βz
#   trials       :: Int               — ZIB shared scalar trials count N
#
# Optional predictor-informed latent-score keys:
#   lv_effects        :: Matrix{Float64} — Gaussian-X_lv trait effects Λ*alpha_lv'
#   alpha_lv          :: Matrix{Float64} — raw q_lv x d latent-axis coefficients
#   scores_mean       :: Matrix{Float64} — n x d rotated score mean X_lv*alpha_lv
#   scores_innovation :: Matrix{Float64} — n x d rotated posterior innovation scores
#
# Optional CI keys (present ONLY when `options["ci_method"]` ∈ {"wald","profile",
# "bootstrap"}; absent for the default "none", so the no-CI contract above is
# byte-identical to before):
#   ci_method      :: String          — the method actually run
#   ci_level       :: Float64         — nominal coverage (default 0.95)
#   ci_param_names :: Vector{String}  — term names (engine-native ordering)
#   ci_estimate    :: Vector{Float64} — point estimates (raw scale for dispersions)
#   ci_lower       :: Vector{Float64} — lower CI bounds
#   ci_upper       :: Vector{Float64} — upper CI bounds
#   ci_note        :: String          — caveats (empty unless CIs were skipped)
#
# v1 scope: the one-part families main provides a fitter for (gaussian,
# poisson, binomial, negbinomial/nb2, nb1, beta, gamma, ordinal,
# ordinal_probit). For the Gaussian fit the
# latent-scale Sigma/correlation/communality use the package extractors; for the
# non-Gaussian fits they use the self-contained shared-block (Lambda*Lambda') form,
# pending the salvage of the link-residual table + non-Gaussian extractors (then the
# cross-family correlation gains its distribution-specific residual). A `family`
# VECTOR routes to the MIXED-family path (fit_mixed_gllvm): one shared latent block
# across distinct response families, with the cross-distribution latent-scale
# `correlation` as the headline. Lognormal is admitted no-X (`fit_lognormal_gllvm`,
# twin fid 3); CI / X / X_lv / masks remain follow-ups. Truncated Poisson is
# admitted no-X (`fit_truncated_poisson_gllvm`, twin fid 10; y ≥ 1); CI / X /
# X_lv / masks remain follow-ups. Fixed-effect
# X is wired (Gaussian plus the admitted one-part covariate kernels);
# predictor-informed latent-score X_lv is wired for complete-response one-part
# Gaussian, Poisson, NB2, Beta, Gamma, and binomial logit/probit/cloglog bridge
# rows. Admitted X_lv rows route Wald B_lv CIs only; profile/bootstrap, masks,
# mixed-family X_lv, and source-specific X_lv remain separate gates.
# Confidence intervals (Wald / profile / bootstrap) route
# through `options["ci_method"]` for scalar-CI one-part families (Gaussian,
# Poisson, Binomial) and grouped-dispersion NB2/NB1/Beta/Gamma rows. NB2, NB1,
# and Beta default to per-trait grouped
# dispersion for R-twin parity. Gamma uses the same grouped engine with one
# shared group, matching current native gllvmTMB's scalar-CV Gamma oracle until
# a native per-trait Gamma expansion lands. Ordinal/ordinal_probit default to
# per-trait cutpoints (no-X and complete-response fixed-effect-X); these
# cutpoint routes currently reject CI routing loudly until matching CI engines
# land. Mixed-family and REML paths skip-with-note since their fits have no
# native confint engine yet.
#
# ADDITIVE: this file + an include/export line in GLLVModels.jl. It edits no fitter or
# extractor; it is included LAST so every dispatch target already exists.

# --- plain-data helpers ----------------------------------------------------

function _bridge_names(x, n::Integer, prefix::AbstractString)
    x === nothing && return ["$(prefix)$i" for i in 1:n]
    names = String.(collect(x))
    length(names) == n ||
        throw(ArgumentError("$(prefix)_names length ($(length(names))) must equal $n"))
    return names
end

function _bridge_get(options, key::AbstractString, default)
    options === nothing && return default
    if options isa AbstractDict
        for k in (key, Symbol(key))
            haskey(options, k) && return options[k]
        end
        return default
    end
    return default
end

# Truthy coercion for bridge option flags coming from R/JuliaCall, where a logical
# may arrive as Bool `true`, integer `1`, or the string "true"/"TRUE".
_bridge_truthy(v::Bool) = v
_bridge_truthy(v::Real) = v != 0
_bridge_truthy(v::AbstractString) = lowercase(strip(v)) in ("true", "t", "1", "yes")
_bridge_truthy(::Nothing) = false
_bridge_truthy(v) = false

function _bridge_coef_fixed(options, q::Integer, label::AbstractString)
    raw = _bridge_get(options, "coef_fixed", nothing)
    raw === nothing && (raw = _bridge_get(options, "xcoef_fixed", nothing))
    raw === nothing && (raw = _bridge_get(options, "beta_fixed", nothing))
    raw === nothing && (raw = _bridge_get(options, "gamma_fixed", nothing))
    return _fixed_zero_mask(raw, q, label)
end

function _bridge_family_key(family::AbstractString)
    key = lowercase(strip(family))
    key in ("gaussian", "normal")                                   && return "gaussian"
    key in ("poisson",)                                             && return "poisson"
    key in ("lognormal",)                                           && return "lognormal"
    key in ("binomial", "bernoulli", "binomial_logit", "bernoulli_logit") && return "binomial"
    key in ("binomial_probit", "bernoulli_probit")                  && return "binomial_probit"
    key in ("binomial_cloglog", "bernoulli_cloglog")                && return "binomial_cloglog"
    key in ("negbinomial", "negative_binomial", "nbinom2", "nb2", "negbin") && return "negbinomial"
    key in ("nb1", "nbinom1")                                       && return "nb1"
    key in ("beta",)                                                && return "beta"
    key in ("gamma",)                                               && return "gamma"
    key in ("betabinomial", "beta_binomial", "beta.binomial")       && return "betabinomial"
    key in ("ordinal", "ordered")                                   && return "ordinal"
    key in ("ordinal_probit", "ordered_probit")                     && return "ordinal_probit"
    key in ("zip", "zipoisson", "zero_inflated_poisson", "zi_poisson") && return "zip"
    key in ("zinb", "zinegbin", "zero_inflated_negbin", "zi_negbin",
            "zinegativebinomial", "zero_inflated_nbinom2") && return "zinb"
    key in ("zib", "zibinomial", "zero_inflated_binomial", "zi_binomial") && return "zib"
    key in ("truncated_poisson", "truncpois", "truncatedpoisson") && return "truncated_poisson"
    throw(ArgumentError(
        "bridge_fit: unsupported family \"$family\"; this engine build supports " *
        "gaussian, poisson, lognormal, binomial, binomial_probit, binomial_cloglog, " *
        "negbinomial (nbinom2), nb1, beta, gamma, betabinomial, ordinal, ordinal_probit, " *
        "zip, zinb, zib, truncated_poisson"))
end

const _BRIDGE_ONEPART_FAMILIES = (
    "gaussian",
    "poisson",
    "lognormal",
    "binomial",
    "binomial_probit",
    "binomial_cloglog",
    "negbinomial",
    "nb1",
    "beta",
    "gamma",
    "betabinomial",
    "ordinal",
    "ordinal_probit",
    "zip",
    "zinb",
    "zib",
    "truncated_poisson",
)

const _BRIDGE_BINOMIAL_FAMILIES = ("binomial", "binomial_probit", "binomial_cloglog")
const _BRIDGE_BINOMIAL_XLV_FAMILIES = _BRIDGE_BINOMIAL_FAMILIES
# Trials-based families that read a binomial-style N (trial count) and map to the
# R-side `cbind(success, failure)` response syntax: binomial (any link) plus
# beta-binomial (overdispersed binomial; N threads through the same way).
const _BRIDGE_TRIALS_FAMILIES = (_BRIDGE_BINOMIAL_FAMILIES..., "betabinomial")
# Families with a point-estimate predictor-informed latent-score (X_lv) bridge
# route: ordinary Gaussian, Poisson (log link), and binomial logit/probit/cloglog.
const _BRIDGE_XLV_FAMILIES = ("gaussian", "poisson", "negbinomial", "gamma", "beta", _BRIDGE_BINOMIAL_FAMILIES...)

# One-part NON-Gaussian families with a fixed-effect-X bridge route. NB2/Beta/Gamma
# use per-trait grouped_cov; ordinal/ordinal_probit use per-trait cutpoint cov;
# poisson/binomial use shared-dispersion `fit_gllvm_cov`. NB1/beta-binomial use
# grouped_cov (beta-binomial threads trial counts N; see fit_beta_binomial_gllvm_grouped_cov).
const _BRIDGE_X_FAMILIES = ("poisson", "binomial", "negbinomial", "nb1", "beta", "gamma",
                            "betabinomial", "ordinal", "ordinal_probit", "zip", "zinb")
# Families with no-X CI but no CI-under-X yet. Keep `ci_no_x_*` honest while
# fencing `ci_x_*` in bridge_capabilities. Empty: ZIP+X and ZINB+X now route CI.
const _BRIDGE_NO_CI_X_FAMILIES = ()

# Map a bridge family key to the `Distributions` marker `fit_gllvm_cov` dispatches
# on (the dispersion field is re-estimated, so the init values here are irrelevant).
# Ordinal keys are admitted in `_BRIDGE_X_FAMILIES` but route via
# `fit_ordinal_gllvm_pertrait_cov` (not this marker).
function _bridge_cov_marker(key::AbstractString)
    key == "poisson"     && return Poisson()
    key == "binomial"    && return Binomial()
    key == "negbinomial" && return NegativeBinomial(10.0, 0.5)
    key == "beta"        && return Beta(10.0, 1.0)
    key == "gamma"       && return Gamma(2.0, 1.0)
    throw(ArgumentError(
        "bridge_fit: family key \"$key\" has no shared-dispersion covariate " *
        "(`fit_gllvm_cov`) marker; X is supported for " *
        join(_BRIDGE_X_FAMILIES, ", ") *
        " (ordinal routes via fit_ordinal_gllvm_pertrait_cov)"))
end

_bridge_rr_df(p::Integer, K::Integer) = p * K - div(K * (K - 1), 2)
_bridge_link_name(link::Link) = String(nameof(typeof(link)))
_bridge_binomial_link(key::AbstractString) =
    key == "binomial_probit" ? ProbitLink() :
    key == "binomial_cloglog" ? CLogLogLink() :
    LogitLink()

function _bridge_corr_from_sigma(Σ::AbstractMatrix)
    p = size(Σ, 1)
    R = Matrix{Float64}(undef, p, p)
    @inbounds for j in 1:p, i in 1:p
        denom = sqrt(Σ[i, i] * Σ[j, j])
        R[i, j] = denom > 0 ? Σ[i, j] / denom : (i == j ? 1.0 : 0.0)
    end
    return R
end

# Rotated p x d loadings via the public extractor (works for every fit type).
_bridge_loadings(fit) = Matrix{Float64}(getLoadings(fit; rotate = true))

# Defensive latent-score extraction: getLV signatures vary per family; if a call
# does not apply (no matching method), scores degrade to empty rather than
# failing the whole fit. Only `MethodError` is treated that way (with a one-time
# warning); any other exception (non-PD residual covariance, dimension mismatch,
# a real bug) propagates so it is not reported as a successful fit (#138).
function _bridge_scores(f)
    try
        return Matrix{Float64}(f())
    catch e
        e isa MethodError || rethrow()
        @warn "latent scores unavailable for this fit (getLV has no applicable method); returning empty scores" maxlog = 1
        return zeros(Float64, 0, 0)
    end
end

# --- gradient diagnostics (ACC-BRIDGE-GRADIENT) -----------------------------
#
# real-workflow-acceptance-lessons.md class 6: the R side had no way to read
# the Julia gradient / convergence health through the bridge. Every payload
# now carries `gradient_max`, the max-abs gradient of the fit's OWN packed
# negative log-likelihood at the fitted parameters, on the same basis a native
# `confint`/`profile_ci` call would see it. Rebuilt via the SAME objective
# reconstruction the confint machinery already uses (never re-derived here)
# and differentiated with ForwardDiff; NaN whenever no packed objective can be
# rebuilt for this fit or ForwardDiff fails on the rebuilt objective — never
# silently absent.

# Gaussian path (GllvmFit / REML's synthetic GllvmFit). Mirrors confint()'s own
# guards: a REML fit carries no packed vector (`fit.pars.θ_packed` is empty),
# and an X_lv (C1 predictor-informed latent-score) fit's legacy θ_packed layout
# excludes alpha_lv, so the reconstructed nll would not be the fit's actual
# objective — both cases return NaN rather than a misleading gradient.
function _bridge_gradient_max_gaussian(fit::GllvmFit, Y, X, Σ_phy)
    (isempty(fit.pars.θ_packed) || _has_lv_predictor(fit)) && return NaN
    try
        nll = _confint_reconstruct_nll(fit, Y, X, Σ_phy)
        g = ForwardDiff.gradient(nll, fit.pars.θ_packed)
        return all(isfinite, g) ? maximum(abs, g) : NaN
    catch e
        e isa InterruptException && rethrow()
        return NaN
    end
end

# Non-Gaussian (Laplace) family path: reuse the `_family_ci` per-family adapter
# (src/confint_family.jl) for the fit's own (θ, nll). Families with no adapter
# there (lognormal, truncated_poisson, ordinal per-trait +/- X, mixed-family)
# raise MethodError, caught below -> NaN. X_lv fits are rejected inside
# `_family_ci` itself (same reason confint() rejects them) -> also NaN.
function _bridge_gradient_max_family(fit, Y; kwargs...)
    try
        ci = _family_ci(fit, Y; kwargs...)
        # An adapter's exact gradient (#552) when it has one; else ForwardDiff.
        g = ci.grad === nothing ? nothing : ci.grad(ci.θ)
        g === nothing && (g = ForwardDiff.gradient(ci.nll, ci.θ))
        return all(isfinite, g) ? maximum(abs, g) : NaN
    catch e
        e isa InterruptException && rethrow()
        return NaN
    end
end

# --- confidence-interval routing -------------------------------------------
#
# Optionally route Wald / profile-likelihood / parametric-bootstrap CIs through
# the bridge by reusing the NATIVE CI engines on the fit object the bridge
# already produced — no CI math is re-implemented here. Controlled by the
# `ci_method` option (default "none", so existing callers are unchanged):
#
#   "none"      — no CIs (the default; the flat contract is byte-identical).
#   "wald"      — observed-information Wald intervals.
#   "profile"   — profile-likelihood (LRT inversion) intervals.
#   "bootstrap" — parametric bootstrap (fixed seed → reproducible).
#
# The returned CI fields are FLAT, JuliaCall-convertible arrays merged onto the
# base contract: ci_method::String, ci_level::Float64, ci_param_names,
# ci_estimate, ci_lower, ci_upper (Vector{Float64}), and ci_note::String.

const _BRIDGE_CI_METHODS = ("none", "wald", "profile", "bootstrap")

function _bridge_ci_method(options)
    raw = _bridge_get(options, "ci_method", "none")
    m = lowercase(strip(String(raw)))
    m in _BRIDGE_CI_METHODS || throw(ArgumentError(
        "bridge_fit: unsupported ci_method \"$raw\"; use one of " *
        join(_BRIDGE_CI_METHODS, ", ")))
    return m
end

_bridge_ci_level(options) = Float64(_bridge_get(options, "ci_level", 0.95))
_bridge_ci_nboot(options) = Int(_bridge_get(options, "ci_nboot", 200))
_bridge_ci_seed(options)  = Int(_bridge_get(options, "ci_seed", 0))

# Empty CI payload (the "none" default and the skip-with-note cases).
function _bridge_ci_payload(method::AbstractString, level::Real, note::AbstractString)
    return (
        ci_method      = String(method),
        ci_level       = Float64(level),
        ci_param_names = String[],
        ci_estimate    = Float64[],
        ci_lower       = Float64[],
        ci_upper       = Float64[],
        ci_note        = String(note),
    )
end

# Build the CI payload from a native CI NamedTuple (term/estimate/lower/upper).
function _bridge_ci_from_native(method::AbstractString, level::Real, ci;
                                note::AbstractString = "")
    return (
        ci_method      = String(method),
        ci_level       = Float64(level),
        ci_param_names = Vector{String}(ci.term),
        ci_estimate    = Vector{Float64}(ci.estimate),
        ci_lower       = Vector{Float64}(ci.lower),
        ci_upper       = Vector{Float64}(ci.upper),
        ci_note        = String(note),
    )
end

# Non-Gaussian one-part families (PoissonFit/BinomialFit/...): the
# unified confint(fit, Y; method, level, N, mask, n_boot, seed) covers all three
# methods, so route every method through it directly.
function _bridge_compute_ci_ng(fit, Ydata, N, method::AbstractString,
                               level::Real, nboot::Integer, seed::Integer;
                               mask = nothing)
    method == "none" && return _bridge_ci_payload("none", level, "")
    msym = method == "wald" ? :wald : (method == "profile" ? :profile : :bootstrap)
    ci = confint(fit, Ydata; method = msym, level = level, N = N,
                 mask = mask, n_boot = nboot, seed = seed)
    return _bridge_ci_from_native(method, level, ci)
end

function _bridge_compute_ci_cov(fit::Union{GllvmCovFit, NBGroupedCovFit, NB1GroupedCovFit,
                                           BetaGroupedCovFit, GammaGroupedCovFit,
                                           BetaBinomialGroupedCovFit, ZIPCovFit, ZINBCovFit},
                                Ydata, N, X,
                                method::AbstractString, level::Real,
                                nboot::Integer, seed::Integer)
    method == "none" && return _bridge_ci_payload("none", level, "")
    msym = method == "wald" ? :wald : (method == "profile" ? :profile : :bootstrap)
    ci = confint(fit, Ydata; method = msym, level = level, N = N, X = X,
                 n_boot = nboot, seed = seed)
    return _bridge_ci_from_native(method, level, ci)
end

# Gaussian fit (GllvmFit): the three native engines have distinct signatures, so
# normalise each to a (term, estimate, lower, upper) table.
#   - wald      → confint(fit; y, level)
#   - profile   → loop profile_ci(fit, i; y, level) over the packed params,
#                 borrowing the Wald estimate vector for ci_estimate (raw scale,
#                 term-aligned by construction).
#   - bootstrap → bootstrap_ci(fit; y, n_boot, level, seed)
# A GllvmFit with an empty packed vector (e.g. the REML synthetic fit) has no
# observed-information / profile substrate — skip with a note rather than fake.
function _bridge_compute_ci_gaussian(fit::GllvmFit, ydata, method::AbstractString,
                                     level::Real, nboot::Integer, seed::Integer;
                                     X = nothing)
    method == "none" && return _bridge_ci_payload("none", level, "")
    if isempty(fit.pars.θ_packed)
        return _bridge_ci_payload(method, level,
            "CIs are not routed for this Gaussian fit (no packed parameter " *
            "vector, e.g. a REML fit); a documented follow-up.")
    end
    if method == "wald"
        ci = confint(fit; y = ydata, level = level, X = X)
        return _bridge_ci_from_native("wald", level, ci)
    elseif method == "bootstrap"
        ci = bootstrap_ci(fit; y = ydata, n_boot = nboot, level = level, seed = seed, X = X)
        return _bridge_ci_from_native("bootstrap", level, ci)
    else  # profile
        wald = confint(fit; y = ydata, level = level, X = X)  # term names + estimates
        lo = Float64[]; hi = Float64[]
        for i in 1:length(fit.pars.θ_packed)
            pc = profile_ci(fit, i; y = ydata, level = level, X = X)
            push!(lo, pc.lower); push!(hi, pc.upper)
        end
        return (
            ci_method      = "profile",
            ci_level       = Float64(level),
            ci_param_names = Vector{String}(wald.term),
            ci_estimate    = Vector{Float64}(wald.estimate),
            ci_lower       = lo,
            ci_upper       = hi,
            ci_note        = "",
        )
    end
end

# Delta-method Wald CI fields for predictor-informed latent-score effects,
# reshaped from confint_lv_effects' vec(B_lv) order to the p×q_lv layout that
# matches `lv_effects`. Merged into the X_lv bridge rows when ci_method=="wald".
function _bridge_lv_ci_fields(ci, q_lv::Integer)
    p = length(ci.estimate) ÷ q_lv
    return (lv_effects_lower = reshape(collect(Float64, ci.lower), p, q_lv),
            lv_effects_upper = reshape(collect(Float64, ci.upper), p, q_lv),
            lv_effects_se = reshape(collect(Float64, ci.se), p, q_lv),
            lv_effects_ci_level = float(ci.level),
            lv_effects_ci_method = "wald",
            lv_effects_ci_pd = ci.pd_hessian)
end

# --- public entry point ----------------------------------------------------

"""
    bridge_fit(; y, family, d=1, N=nothing, X=nothing, X_lv=nothing,
               trait_names=nothing, unit_names=nothing, phylo=nothing,
               options=Dict())

Plain-data R->Julia bridge (JuliaCall transport). Fits a one-part GLLVM for the
requested `family` and returns a flat, JuliaCall-convertible NamedTuple (see the
file header for the key->type contract). `y` is a `p x n` response matrix
(traits x units); `d` is the latent dimension `K`; `N` (Binomial trials, `p x n`
or a scalar) is forwarded to the Binomial fitter.

`phylo` is an optional diagnostic consume hook: an admitted [`PrecisionPhy`](@ref)
or a flat precision payload. When set, `y` is a length-`p` trait vector and the
return is the univariate [`fit_phylo_gaussian`](@ref) result (`diagnostic_only =
true`). This does not lift the R `phylo_rr` gate.

Set `options["phylo_model"]="multivariate"` explicitly to select the Gaussian
multivariate precision candidate instead. Supply a traits-by-observations `y`,
rank `d`, and required one-based observation-to-tip `options["species_id"]`.
`options["mode"]` is `"barelowrank"` or `"explicitunique"`; intervals support
`"none"` or `"wald"`. It returns a distinct flat candidate contract, not the
legacy univariate diagnostic result. R `phylo_rr` admission remains separately
gated. Missing responses, `N`, `X_lv`, and named-label overrides are unsupported.

`family = "zib"` reads `N` differently: the zero-inflated binomial carries ONE
shared scalar trials count, so `N` is **required** (there is no safe default —
`N = 1` is the zero-inflated Bernoulli, whose two intercepts are aliased) and a
`p x n` `N` is accepted only when every entry is equal, then collapsed to that
scalar. The value actually used is returned as `trials`.

For `family = "truncated_poisson"`, responses must be finite positive integers
exactly representable by the bridge as `Int`; fractional counts are rejected,
never rounded.

Confidence intervals are routed through `options` (all optional):
  - `"ci_method"` ∈ {`"none"` (default), `"wald"`, `"profile"`, `"bootstrap"`}.
    When not `"none"`, the returned tuple gains the `ci_*` keys documented in the
    file header. These reuse the NATIVE confint / profile_ci / bootstrap_ci
    engines on the fit object the bridge already produced (no CI math is
    re-implemented), so bounds are identical to the equivalent native call.
  - `"ci_level"` — nominal coverage (default `0.95`).
  - `"ci_nboot"` — bootstrap replicates (default `200`).
  - `"ci_seed"`  — bootstrap RNG seed (default `0`; fixed → reproducible).
"""
function bridge_fit(; y,
                    family = nothing,
                    d::Real = 1,
                    N = nothing,
                    X = nothing,
                    X_lv = nothing,
                    mask = nothing,
                    trait_names = nothing,
                    unit_names = nothing,
                    sources = nothing,
                    phylo = nothing,
                    options = Dict{String,Any}())
    # R numeric literals reach Julia as Float64: accept any integral Real.
    (isfinite(d) && d == round(d)) || throw(ArgumentError(
        "bridge_fit: d must be an integer (got $d)"))
    d = Int(d)
    phylo_model = String(_bridge_get(options, "phylo_model", "diagnostic"))
    phylo_model in ("diagnostic", "multivariate") || throw(ArgumentError(
        "bridge_fit: phylo_model must be diagnostic or multivariate"))
    phylo === nothing && phylo_model != "diagnostic" && throw(ArgumentError(
        "bridge_fit: multivariate phylo_model requires an explicit precision payload"))
    if phylo !== nothing
        sources === nothing || throw(ArgumentError(
            "bridge_fit: phylo precision consume hook cannot combine with sources"))
        if phylo_model == "multivariate"
            all(isnothing, (N, X_lv, mask, trait_names, unit_names)) || throw(ArgumentError(
                "bridge_fit: multivariate precision does not support N, X_lv, mask or label overrides"))
            return _bridge_fit_precision_multivariate(y, phylo;
                family=family, d=d, X=X, options=options)
        end
        return _bridge_fit_phylo_precision(y, phylo; family = family, options = options)
    end
    family === nothing && throw(ArgumentError("bridge_fit: family is required"))
    K = Int(d)
    K >= 0 || throw(ArgumentError("d must be a non-negative integer"))
    if sources !== nothing
        return _bridge_fit_sources(y, family, K, sources;
            X = X, X_lv = X_lv, mask = mask, N = N,
            trait_names = trait_names, unit_names = unit_names,
            options = options)
    end
    # Fixed-effect covariates X (a p×n×q array) are wired for the Gaussian family
    # and the one-part NON-Gaussian `_BRIDGE_X_FAMILIES` (incl. nb1 grouped_cov and
    # ordinal/ordinal_probit per-trait cutpoint cov). Mixed-family X remains a
    # documented follow-up — reject loudly rather than silently dropping X.
    if X !== nothing
        if family isa AbstractVector
            throw(ArgumentError(
                "bridge_fit: fixed-effect covariates X are not yet wired for the " *
                "mixed-family path; a documented follow-up"))
        end
        key = _bridge_family_key(String(family))
        (key == "gaussian" || key in _BRIDGE_X_FAMILIES) || throw(ArgumentError(
            "bridge_fit: fixed-effect covariates X are wired for family ∈ {gaussian, " *
            join(_BRIDGE_X_FAMILIES, ", ") * "}; family=\"$(family)\" has no covariate fitter"))
    end
    # Predictor-informed latent-score covariates are narrower than ordinary
    # fixed-effect X in this bridge slice: complete Gaussian and binomial
    # logit/probit/cloglog point estimates only. The native fitters can be
    # widened later, but the bridge contract should not silently imply broader
    # non-Gaussian or mixed-family parity.
    if X_lv !== nothing
        if family isa AbstractVector
            throw(ArgumentError(
                "bridge_fit: predictor-informed latent-score covariates X_lv " *
                "are not yet wired for the mixed-family path; use a one-part " *
                "gaussian or binomial family."))
        end
        key = _bridge_family_key(String(family))
        key in _BRIDGE_XLV_FAMILIES || throw(ArgumentError(
            "bridge_fit: predictor-informed latent-score covariates X_lv are " *
            "currently wired only for family=\"gaussian\", \"poisson\", " *
            "\"binomial\", \"binomial_probit\", or \"binomial_cloglog\"; " *
            "family=\"$(family)\" is a separate validation gate."))
        X === nothing || throw(ArgumentError(
            "bridge_fit: simultaneous fixed-effect X and latent-score X_lv is " *
            "not admitted in the bridge yet; fit one mean route at a time."))
    end
    # Mixed-family: a vector of per-trait family strings ⇒ one shared latent block,
    # a TRUE cross-distribution VCV (the headline). A length-1 vector or an all-same
    # vector still routes here (the mixed fitter handles the degenerate one-family
    # case); the cross-family `correlation` is the contract's headline field.
    if family isa AbstractVector
        mask === nothing || throw(ArgumentError(
            "bridge_fit: missing-response masks are not yet wired for the " *
            "mixed-family path; use a one-part non-Gaussian family or engine='tmb'."))
        X_lv === nothing || throw(ArgumentError(
            "bridge_fit: X_lv is not wired for the mixed-family path."))
        return _bridge_fit_mixed(y, collect(String, String.(family)), K, N,
                                 trait_names, unit_names, options)
    end
    return _bridge_fit_onepart(y, _bridge_family_key(String(family)), K, N,
                               trait_names, unit_names, options;
                               X = X, X_lv = X_lv, mask = mask)
end

# --- one-part dispatch -----------------------------------------------------

const _BRIDGE_MASK_FAMILIES = (
    "poisson", "binomial", "binomial_probit", "binomial_cloglog",
    "negbinomial", "nb1", "beta", "gamma", "betabinomial", "ordinal", "ordinal_probit",
)

const _BRIDGE_GROUPED_DISPERSION_FAMILIES = ("negbinomial", "nb1", "beta", "gamma",
                                            "betabinomial")
const _BRIDGE_PERTRAIT_ORDINAL_FAMILIES = ("ordinal", "ordinal_probit")
const _BRIDGE_MASK_CI_FAMILIES = (
    "poisson", "binomial", "binomial_probit", "binomial_cloglog",
    "negbinomial", "nb1", "beta", "gamma", "betabinomial",
)
# One-part families whose default fit route has NO native confint engine yet:
# the per-trait ordinal-cutpoint routes, plus lognormal (`LognormalFit` is not
# in `_CIFit`) and truncated Poisson (`TruncatedPoissonFit` is not in `_CIFit`).
# Beta-binomial grouped(_cov) now routes Wald/profile/bootstrap via `_family_ci`
# (finite-difference Hessian; the BB Laplace still has no analytic OH knob).
const _BRIDGE_NO_CI_FAMILIES = (_BRIDGE_PERTRAIT_ORDINAL_FAMILIES..., "lognormal",
                                "truncated_poisson")
# One-part families with no scalar-mean postfit extractor (residuals = y - mu,
# parametric simulate): the ordinal families (no scalar response mean), plus
# beta-binomial (no `residuals`/`simulate` method for BetaBinomialFit or its
# grouped/grouped_cov siblings yet), plus lognormal (no extractor on
# `LognormalFit` yet) and truncated Poisson (no extractor on
# `TruncatedPoissonFit` yet).
const _BRIDGE_NO_SCALAR_POSTFIT_FAMILIES = (_BRIDGE_PERTRAIT_ORDINAL_FAMILIES...,
    "betabinomial", "lognormal", "truncated_poisson")
# One-part families with `residuals` but NO `simulate` method on this engine: the
# three zero-inflated fit types. They share the scalar-mean `residuals` extractor,
# so the broader set above would advertise `postfit_simulate` for a route that
# does not exist; this narrows that one column without changing any behaviour.
const _BRIDGE_NO_SIMULATE_FAMILIES = ("zip", "zinb", "zib")

# ZIB carries ONE shared scalar trials count `N::Int` (`struct ZIB`), not the
# per-observation `cbind(success, failure)` counts the binomial / beta-binomial
# routes use. Normalise the bridge's `N` argument to that scalar:
#   * a number            → rounded to Int;
#   * a p×n array         → admitted ONLY if every entry is equal (R's
#                           `cbind(success, failure)` naturally builds a matrix),
#                           then collapsed to that shared value;
#   * unequal entries     → error naming the shared-scalar contract. Taking
#                           `N[1, 1]` would silently fit a different model.
#   * `nothing`           → error. The binomial default `N = 1` is NOT safe here:
#                           at N = 1 ZIB is the zero-inflated Bernoulli, where the
#                           structural-zero and success intercepts are aliased, so
#                           the optimiser reports an arbitrary point on a flat
#                           ridge with no warning.
function _bridge_zib_trials(N, p::Integer, n::Integer)
    N === nothing && throw(ArgumentError(
        "bridge_fit: family=\"zib\" requires an explicit trials count N. ZIB uses " *
        "ONE shared scalar N for every observation, and there is no safe default: " *
        "N = 1 is the zero-inflated Bernoulli, whose structural-zero and success " *
        "intercepts are not separately identified. Pass N as a scalar, or as a " *
        "$(p)×$(n) array whose entries are all equal."))
    if N isa Number
        _bridge_exact_integer(N, 1) || throw(ArgumentError(
            "bridge_fit: family=\"zib\" trials N must be a finite integer " *
            "≥ 1 exactly representable as Int; values are not rounded (got $(N))"))
        Ni = Int(N)
        return Ni
    end
    A = Matrix(N)
    size(A) == (p, n) || throw(ArgumentError(
        "bridge_fit: family=\"zib\" trials N must be a scalar or a $(p)×$(n) array; " *
        "got $(size(A))"))
    Ai = _bridge_require_integer_trials(N, p, n, "zib")
    Ni = first(Ai)
    all(==(Ni), Ai) || throw(ArgumentError(
        "bridge_fit: family=\"zib\" requires ONE shared scalar trials count N, but " *
        "the supplied $(p)×$(n) N has unequal entries (min $(minimum(Ai)), " *
        "max $(maximum(Ai))). Per-observation cbind(success, failure) trials are " *
        "not the ZIB contract; pass a uniform N."))
    Ni >= 1 || throw(ArgumentError(
        "bridge_fit: family=\"zib\" needs trials N >= 1; got $(Ni)"))
    return Ni
end

# Count / trial inputs must be exact integers (truncated_poisson already rejects
# rounding). Other families must match that contract so fractional R-side data
# cannot silently change the fitted dataset.
function _bridge_exact_integer(v::Real, lo::Real)
    isfinite(v) && v >= lo && isinteger(v) && v <= Float64(typemax(Int))
end

function _bridge_require_integer_counts(Yf::AbstractMatrix, family::AbstractString; lo::Real = 0)
    for i in eachindex(Yf)
        v = Yf[i]
        _bridge_exact_integer(v, lo) && continue
        r, c = Tuple(CartesianIndices(Yf)[i])
        throw(ArgumentError(
            "bridge_fit: family=\"$(family)\" requires finite integer counts " *
            "exactly representable as Int (minimum $(lo)); values are not rounded " *
            "(bad cell row=$(r), col=$(c), value=$(v))"))
    end
    return Int.(Yf)
end

function _bridge_require_integer_trials(N, p::Integer, n::Integer, family::AbstractString)
    if N === nothing
        return fill(1, p, n)
    end
    if N isa Number
        _bridge_exact_integer(N, 1) || throw(ArgumentError(
            "bridge_fit: family=\"$(family)\" trials N must be a finite integer " *
            "≥ 1 exactly representable as Int; values are not rounded (got $(N))"))
        Ni = Int(N)
        return fill(Ni, p, n)
    end
    A = Matrix(N)
    size(A) == (p, n) || throw(ArgumentError(
        "bridge_fit: family=\"$(family)\" trials N must be a scalar or a " *
        "$(p)×$(n) array; got $(size(A))"))
    for i in eachindex(A)
        v = A[i]
        _bridge_exact_integer(v, 1) && continue
        r, c = Tuple(CartesianIndices(A)[i])
        throw(ArgumentError(
            "bridge_fit: family=\"$(family)\" requires finite integer trial counts " *
            "exactly representable as Int (≥ 1); values are not rounded " *
            "(bad cell row=$(r), col=$(c), value=$(v))"))
    end
    return Int.(A)
end

function _bridge_ci_guard_pertrait_ordinal(key::AbstractString, ci_method::AbstractString)
    ci_method == "none" && return nothing
    throw(ArgumentError(
        "bridge_fit: confidence intervals for per-trait ordinal-cutpoint " *
        "$key fits are not routed yet; use ci_method=\"none\" or the shared-" *
        "cutpoint OrdinalFit directly as a Julia-side comparator."))
end

function _bridge_ci_guard_lognormal(ci_method::AbstractString)
    ci_method == "none" && return nothing
    throw(ArgumentError(
        "bridge_fit: confidence intervals for family=\"lognormal\" are not " *
        "routed yet; LognormalFit is not in the native confint union. " *
        "Use ci_method=\"none\"."))
end

function _bridge_ci_guard_truncated_poisson(ci_method::AbstractString)
    ci_method == "none" && return nothing
    throw(ArgumentError(
        "bridge_fit: confidence intervals for family=\"truncated_poisson\" are not " *
        "routed yet; TruncatedPoissonFit is not in the native confint union. " *
        "Use ci_method=\"none\"."))
end

function _bridge_dispersion_payload(group_values::AbstractVector,
                                    group_id::AbstractVector{<:Integer},
                                    parameter::AbstractString,
                                    engine_scale::AbstractString,
                                    public_scale::AbstractString)
    g = collect(Float64, group_values)
    gid = collect(Int, group_id)
    return (
        dispersion_group = g,
        dispersion_group_id = gid,
        dispersion_parameter = String(parameter),
        dispersion_engine_scale = String(engine_scale),
        dispersion_public_scale = String(public_scale),
    )
end

_bridge_expand_dispersion(payload) =
    [payload.dispersion_group[g] for g in payload.dispersion_group_id]

_bridge_group_df(p::Integer, K::Integer, payload) =
    p + _bridge_rr_df(p, K) + length(payload.dispersion_group)

"""
    bridge_capabilities()

Return the flat capability surface currently exposed by `bridge_fit`.

The result is a JuliaCall-friendly `NamedTuple` of vectors. It reports the
Julia bridge surface only; R-side admission gates may be narrower until
metadata, labels, parity rows, and confidence-interval status rows are
validated in `gllvmTMB`.

The `ci_no_x_*` columns report that a native route exists for complete one-part
no-covariate fits. The `ci_mask_*` columns are narrower: no-covariate one-part
response-mask fits whose masked likelihood can also drive Wald/profile/bootstrap
intervals. The `ci_x_*` columns are complete-response one-part fixed-effect-X
fits. `predictor_informed_lv` marks complete-response one-part X_lv bridge
routes for Gaussian, Poisson, NB2, Beta, Gamma, and binomial
logit/probit/cloglog. It does not imply source-specific X_lv, mixed-family X_lv,
response-mask X_lv, or profile/bootstrap X_lv intervals; Wald B_lv CI payloads
for admitted X_lv fits are routed by `bridge_fit` but intentionally have no
separate capability column. None of the CI groups imply mixed-family or
R-bridge parity coverage. Use `status` and `notes` for public claim wording.
"""
function bridge_capabilities()
    onepart = collect(_BRIDGE_ONEPART_FAMILIES)
    family = vcat(onepart, ["mixed-family vector"])
    x_families = Set(vcat(["gaussian"], collect(_BRIDGE_X_FAMILIES)))
    xlv_families = Set(collect(_BRIDGE_XLV_FAMILIES))
    mask_families = Set(_BRIDGE_MASK_FAMILIES)
    mask_ci_families = Set(_BRIDGE_MASK_CI_FAMILIES)
    no_ci_families = Set(_BRIDGE_NO_CI_FAMILIES)
    # Scalar-mean post-fit (residuals = y − μ, parametric simulate) excludes the
    # ordinal families (no scalar response mean) and beta-binomial (no
    # residuals/simulate extractor on this build yet). predict() IS wired for
    # ordinal via the cutpoints payload (type "prob"/"class") AND for
    # beta-binomial (the success probability μ), so it uses every one-part family.
    postfit_families = Set(filter(f -> !(f in _BRIDGE_NO_SCALAR_POSTFIT_FAMILIES), onepart))
    # `simulate` is narrower than `residuals`: the zero-inflated fit types have a
    # residuals method but no simulate method on this engine.
    simulate_families = Set(filter(f -> !(f in _BRIDGE_NO_SIMULATE_FAMILIES),
                                   collect(postfit_families)))
    predict_families = Set(onepart)
    # Ordinal+X point fits are wired; CI under X remains a follow-up for the
    # per-trait ordinal fence. ZIP+X, ZINB+X, and beta-binomial grouped(_cov) route CI.
    no_ci_x = Set(_BRIDGE_NO_CI_X_FAMILIES)

    return (
        family = family,
        fit_no_x = vcat(fill(true, length(onepart)), [true]),
        fixed_effect_X = vcat([f in x_families for f in onepart], [false]),
        predictor_informed_lv = vcat([f in xlv_families for f in onepart], [false]),
        missing_response = vcat([f in mask_families for f in onepart], [false]),
        cbind_binomial = [f in _BRIDGE_TRIALS_FAMILIES for f in family],
        ci_no_x_wald = vcat([!(f in no_ci_families) for f in onepart], [false]),
        ci_no_x_profile = vcat([!(f in no_ci_families) for f in onepart], [false]),
        ci_no_x_bootstrap = vcat([!(f in no_ci_families) for f in onepart], [false]),
        ci_mask_wald = vcat([f in mask_ci_families for f in onepart], [false]),
        ci_mask_profile = vcat([f in mask_ci_families for f in onepart], [false]),
        ci_mask_bootstrap = vcat([f in mask_ci_families for f in onepart], [false]),
        ci_x_wald = vcat([f in x_families && !(f in no_ci_families) && !(f in no_ci_x)
                          for f in onepart], [false]),
        ci_x_profile = vcat([f in x_families && !(f in no_ci_families) && !(f in no_ci_x)
                             for f in onepart], [false]),
        ci_x_bootstrap = vcat([f in x_families && !(f in no_ci_families) && !(f in no_ci_x)
                               for f in onepart], [false]),
        postfit_coef = vcat(fill(true, length(onepart)), [true]),
        postfit_fit_stats = vcat(fill(true, length(onepart)), [true]),
        postfit_summary = vcat(fill(true, length(onepart)), [true]),
        postfit_predict = vcat([f in predict_families for f in onepart], [true]),
        postfit_residuals = vcat([f in postfit_families for f in onepart], [true]),
        postfit_simulate = vcat([f in simulate_families for f in onepart], [true]),
        postfit_ordination = vcat(fill(true, length(onepart)), [true]),
        status = vcat(fill("partial", length(onepart)), ["partial"]),
        notes = vcat(
            [
                f == "negbinomial" ?
                    "one-part reduced-rank bridge family; default no-X route uses per-trait grouped dispersion; no-X, masked no-X, and complete-response fixed-effect-X Wald/profile/bootstrap CI payloads are routed; predictor-informed latent-score X_lv via the shared-dispersion fitter is wired for complete-response point fits; X_lv Wald B_lv CI payloads are routed; profile/bootstrap X_lv CIs remain follow-ups" :
                f == "beta" ?
                    "one-part reduced-rank bridge family; default no-X route uses per-trait grouped dispersion; no-X, masked no-X, and complete-response fixed-effect-X Wald/profile/bootstrap CI payloads are routed; predictor-informed latent-score X_lv via the shared-precision fitter is wired for complete-response point fits; X_lv Wald B_lv CI payloads are routed; profile/bootstrap X_lv CIs remain follow-ups" :
                f == "nb1" ?
                    "one-part reduced-rank bridge family; default no-X route uses per-trait grouped dispersion; no-X and masked no-X Wald/profile/bootstrap CI payloads are routed; fixed-effect-X remains a follow-up" :
                f == "gamma" ?
                    "one-part reduced-rank bridge family; default no-X route uses shared Gamma grouped dispersion to match current native scalar-CV Gamma; no-X, masked no-X, and complete-response fixed-effect-X Wald/profile/bootstrap CI payloads are routed; predictor-informed latent-score X_lv via the shared-shape fitter is wired for complete-response point fits; X_lv Wald B_lv CI payloads are routed; profile/bootstrap X_lv CIs remain follow-ups; per-trait Gamma is a native-expansion follow-up" :
                f == "betabinomial" ?
                    "one-part reduced-rank bridge family; default no-X and complete-response fixed-effect-X routes use per-trait Beta-binomial precision (disp.group) with binomial-style cbind(success, failure) trial counts N; missing-response masks are wired for the no-X route; no-X, masked no-X, and complete-response fixed-effect-X Wald/profile/bootstrap CI payloads are routed (finite-difference Hessian; no analytic OH); residuals/simulate are not wired (no scalar-mean postfit extractor yet); route support is narrower than full R-user parity" :
                f in _BRIDGE_PERTRAIT_ORDINAL_FAMILIES ?
                    "one-part reduced-rank bridge family; default no-X and complete-response fixed-effect-X routes use per-trait ordinal cutpoints (τ₁=0 / K−2); CI routing is a follow-up" :
                f == "zip" ?
                    "two-part ZIP bridge family (Julia-forward / twin-asymmetric); no-X routes fit_zip_gllvm with Wald/profile/bootstrap CI; complete-response fixed-effect-X routes fit_zip_gllvm_cov (separate γz/γc, Λz=0) with Wald/profile/bootstrap CI under X (finite-difference Hessian); no twin light RCall Δ (twin ZIP cut); route support is narrower than full R-user parity" :
                f == "zinb" ?
                    "two-part ZINB bridge family (Julia-forward / twin-asymmetric); no-X routes fit_zinb_gllvm with Wald/profile/bootstrap CI and one shared scalar r (log r); complete-response fixed-effect-X routes fit_zinb_gllvm_cov (separate γz/γc, Λz=0, shared scalar r) with Wald/profile/bootstrap CI under X (finite-difference Hessian); no twin light RCall Δ (twin ZINB cut); route support is narrower than full R-user parity" :
                f == "zib" ?
                    "two-part ZIB bridge family (Julia-forward / twin-asymmetric); no-X routes fit_zib_gllvm with Wald/profile/bootstrap CI and one shared scalar trials count N (not per-observation cbind); fixed-effect-X, missing-response masks, and CI under X remain follow-ups; no twin light RCall Δ — the twin gllvmTMB has no ZIB, so a Δ would be invented (contrast ZIP/ZINB, which the twin cut); route support is narrower than full R-user parity" :
                f == "poisson" ?
                    "one-part reduced-rank bridge family; no-X, masked no-X, and complete-response fixed-effect-X Wald/profile/bootstrap CI payloads are routed; predictor-informed latent-score X_lv is wired for complete-response point fits; X_lv Wald B_lv CI payloads are routed; profile/bootstrap X_lv CIs remain follow-ups; route support is narrower than full R-user parity" :
                f == "lognormal" ?
                    "one-part reduced-rank bridge family (twin fid 3); no-X routes fit_lognormal_gllvm (shared scalar σ on log y; y-scale loglik includes Jacobian); CI, fixed-effect-X, X_lv, and missing-response masks remain follow-ups; light no-X RCall logLik Δ PAID (test/parity/test_lognormal_parity.jl @ GLLVM_PARITY_TESTS=1; receipt abs≈2.24e-8 @ rtol 1e-6, seed=52, p=5, K=2, n=60); route support is narrower than full R-user parity" :
                f == "truncated_poisson" ?
                    "one-part reduced-rank bridge family (twin fid 10); no-X routes fit_truncated_poisson_gllvm (log link on untruncated μ; support y ≥ 1); CI, fixed-effect-X, X_lv, and missing-response masks remain follow-ups; light no-X RCall logLik Δ PAID (test/parity/test_truncated_poisson_parity.jl @ GLLVM_PARITY_TESTS=1; receipt abs≈2.71e-9 @ rtol 1e-6, seed=53, p=5, K=2, n=60); route support is narrower than full R-user parity" :
                f == "binomial" ?
                    "one-part reduced-rank bridge family; no-X, masked no-X, and complete-response fixed-effect-X Wald/profile/bootstrap CI payloads are routed; predictor-informed latent-score X_lv is wired for complete-response point fits; X_lv Wald B_lv CI payloads are routed; profile/bootstrap X_lv CIs remain follow-ups; route support is narrower than full R-user parity" :
                f in _BRIDGE_BINOMIAL_XLV_FAMILIES ?
                    "one-part reduced-rank bridge family; no-X and masked no-X Wald/profile/bootstrap CI payloads are routed; predictor-informed latent-score X_lv is wired for complete-response point fits; X_lv Wald B_lv CI payloads are routed; profile/bootstrap X_lv CIs remain follow-ups; route support is narrower than full R-user parity" :
                f in _BRIDGE_MASK_CI_FAMILIES ?
                    "one-part reduced-rank bridge family; no-X, masked no-X, and complete-response fixed-effect-X Wald/profile/bootstrap CI payloads are routed; route support is narrower than full R-user parity" :
                f == "gaussian" ?
                    "one-part reduced-rank bridge family; fixed-effect-X and predictor-informed latent-score X_lv routes are wired for complete-response point fits; X_lv Wald B_lv CI payloads are routed; profile/bootstrap X_lv CIs, mixed-family X_lv, and source-specific X_lv remain follow-ups; route support is narrower than full R-user parity" :
                    "one-part reduced-rank bridge family; route support is narrower than full R-user parity"
                for f in onepart
            ],
            ["mixed-family vector route; no X, mask, or CI routing"],
        ),
    )
end

function _bridge_mask(mask, p::Integer, n::Integer)
    mask === nothing && return nothing
    M = Matrix{Bool}(mask)
    size(M) == (p, n) || throw(ArgumentError(
        "bridge_fit: mask must be p×n ($(p)×$(n)); got $(size(M))"))
    all(M) && return nothing
    any(M) || throw(ArgumentError(
        "bridge_fit: mask has no observed cells; at least one response must be observed"))
    return M
end

function _bridge_fit_onepart(y, key::AbstractString, K::Integer, N,
                             trait_names, unit_names, options;
                             X = nothing, X_lv = nothing, mask = nothing)
    Yf = Matrix{Float64}(y)
    p, n = size(Yf)
    traits = _bridge_names(trait_names, p, "trait")
    units = _bridge_names(unit_names, n, "unit")
    M = _bridge_mask(mask, p, n)

    # CI routing options (validated up-front so a bad ci_method errors before the
    # — potentially expensive — fit runs). ci_method="none" ⇒ ci stays nothing ⇒
    # the assembled contract is byte-identical to the no-CI path.
    ci_method = _bridge_ci_method(options)
    ci_level  = _bridge_ci_level(options)
    ci_nboot  = _bridge_ci_nboot(options)
    ci_seed   = _bridge_ci_seed(options)

    if M !== nothing
        key in _BRIDGE_MASK_FAMILIES || throw(ArgumentError(
            "bridge_fit: missing-response masks are wired for " *
            join(_BRIDGE_MASK_FAMILIES, ", ") *
            "; family=\"$key\" is not yet supported"))
        X === nothing || throw(ArgumentError(
            "bridge_fit: missing-response masks with fixed-effect covariates X " *
            "are not wired yet; use a complete response table or engine='tmb'."))
    end

    if X_lv !== nothing
        key in _BRIDGE_XLV_FAMILIES || throw(ArgumentError(
            "bridge_fit: X_lv is wired only for family in " *
            "$(join(_BRIDGE_XLV_FAMILIES, ", "))."))
        X === nothing || throw(ArgumentError(
            "bridge_fit: simultaneous fixed-effect X and latent-score X_lv is " *
            "not admitted in the bridge yet; fit one mean route at a time."))
        M === nothing || throw(ArgumentError(
            "bridge_fit: missing-response masks with X_lv are not wired yet; " *
            "use a complete response table or engine='tmb'."))
        K > 0 || throw(ArgumentError(
            "bridge_fit: X_lv requires a positive latent dimension d."))
        ci_method in ("none", "wald") || throw(ArgumentError(
            "bridge_fit: only ci_method=\"wald\" (delta-method Wald on B_lv) is " *
            "admitted for X_lv fits; profile/bootstrap remain gated."))
    end

    # X (a p×n×q covariate array) routes to the covariate fitters: the Gaussian
    # branch below handles key=="gaussian"; NB2/Beta/Gamma use per-trait
    # grouped_cov; ordinal/ordinal_probit use per-trait cutpoint cov; other
    # one-part _BRIDGE_X_FAMILIES use fit_gllvm_cov. Defend the invariant here
    # too so a future DIRECT caller can't slip X past a family with no covariate
    # fitter (nb1) and have it silently dropped.
    if X !== nothing && key != "gaussian"
        key in _BRIDGE_X_FAMILIES ||
            throw(ArgumentError("bridge_fit: X is not wired for family=\"$key\"; " *
                "supported families with covariates are gaussian, " *
                join(_BRIDGE_X_FAMILIES, ", ")))
        return _bridge_fit_onepart_cov(Yf, key, K, N, traits, units, X,
                                       ci_method, ci_level, ci_nboot, ci_seed,
                                       options)
    end

    if key == "gaussian"
        if X_lv !== nothing
            Xlv = Matrix{Float64}(X_lv)
            size(Xlv, 1) == n || throw(ArgumentError(
                "bridge_fit: X_lv must be n×q_lv ($(n)×q_lv); got $(size(Xlv))"))
            size(Xlv, 2) > 0 || throw(ArgumentError(
                "bridge_fit: X_lv must have at least one predictor column"))

            # Preserve the no-X Gaussian bridge convention: trait means live in
            # alpha, while the latent-score predictor fit sees centred responses.
            alpha = vec(Statistics.mean(Yf; dims = 2))
            Yc = Yf .- alpha
            fit = fit_gaussian_gllvm(Yc; K = K, X_lv = Xlv)
            ci_extra = ci_method == "wald" ?
                _bridge_lv_ci_fields(confint_lv_effects(fit, Yc, Xlv; level = ci_level),
                                     size(Xlv, 2)) : NamedTuple()
            Sigma = Matrix{Float64}(sigma_y_site(fit))
            corr  = Matrix{Float64}(correlation(fit))
            comm  = Vector{Float64}(communality(fit))
            scores_total = Matrix{Float64}(
                getLV(fit, Yc; X_lv = Xlv, component = :total, rotate = true))
            scores_mean = Matrix{Float64}(
                getLV(fit, Yc; X_lv = Xlv, component = :mean, rotate = true))
            scores_innovation = Matrix{Float64}(
                getLV(fit, Yc; X_lv = Xlv, component = :innovation, rotate = true))
            df = p + _nparams(fit)
            base = _bridge_assemble(fit, "gaussian", "gaussian_xlv_rr", traits, units;
                alpha = alpha, dispersion = fill(NaN, p), sigma_eps = fit.pars.σ_eps,
                link = fill("IdentityLink", p), Sigma = Sigma, corr = corr, comm = comm,
                scores = scores_total, df = df, loglik = fit.logLik,
                converged = fit.converged, iterations = fit.n_iter,
                note = "predictor-informed latent-score fit (Gaussian C1): alpha " *
                       "are pre-fit trait means, scores are total latent scores, " *
                       "scores_mean = X_lv*alpha_lv, scores_innovation are the " *
                       "posterior zero-mean score deviations, and lv_effects = " *
                       "Lambda*alpha_lv' is the rotation-stable trait-effect matrix. " *
                       "Confidence intervals on B_lv are available via " *
                       "ci_method=\"wald\" (delta method); profile/bootstrap, " *
                       "response masks, mixed-family X_lv, and source-specific " *
                       "X_lv remain separate gates.",
                ci = nothing, gradient_max = _bridge_gradient_max_gaussian(fit, Yc, nothing, nothing))
            return merge(base, (lv_effects = Matrix{Float64}(extract_lv_effects(fit)),
                                alpha_lv = Matrix{Float64}(fit.pars.alpha_lv),
                                scores_mean = scores_mean,
                                scores_innovation = scores_innovation), ci_extra)
        end
        if X !== nothing
            # Fixed-effect covariate path. The caller's X (p×n×q) already carries the
            # FULL mean structure — per-trait intercept dummies AND covariates — so we
            # do NOT pre-centre Y; fit_gaussian_gllvm estimates β jointly with Λ̂/σ̂.
            # `alpha` is the per-trait fitted mean (mean over sites of Xₜₛ·β̂), the
            # natural per-trait intercept summary when the mean is covariate-driven.
            Xarr = Array{Float64,3}(X)
            size(Xarr, 1) == p && size(Xarr, 2) == n || throw(ArgumentError(
                "bridge_fit: X must be p×n×q ($(p)×$(n)×q); got $(size(Xarr))"))
            q = size(Xarr, 3)
            coef_fixed = _bridge_coef_fixed(options, q, "coef_fixed")
            fit = fit_gaussian_gllvm(Yf; K = K, X = Xarr, β_fixed = coef_fixed)
            β = collect(Float64, fit.pars.β)
            alpha = zeros(Float64, p)
            @inbounds for t in 1:p
                acc = 0.0
                for s in 1:n, k in 1:q
                    acc += Xarr[t, s, k] * β[k]
                end
                alpha[t] = acc / n
            end
            Sigma = Matrix{Float64}(sigma_y_site(fit))
            corr  = Matrix{Float64}(correlation(fit))
            comm  = Vector{Float64}(communality(fit))
            scores = _bridge_scores(() -> getLV(fit, Yf; X = Xarr, rotate = true))
            df = count(!, coef_fixed) + _bridge_rr_df(p, K) + 1
            ci = ci_method == "none" ? nothing :
                 _bridge_compute_ci_gaussian(fit, Yf, ci_method, ci_level, ci_nboot,
                                             ci_seed; X = Xarr)
            base = _bridge_assemble(fit, "gaussian", "gaussian_x_rr", traits, units;
                alpha = alpha, dispersion = fill(NaN, p), sigma_eps = fit.pars.σ_eps,
                link = fill("IdentityLink", p), Sigma = Sigma, corr = corr, comm = comm,
                scores = scores, df = df, loglik = fit.logLik,
                converged = fit.converged, iterations = fit.n_iter,
                note = "fixed-effect covariate fit: X carries the full mean structure " *
                       "(per-trait intercepts + covariates); alpha is the per-trait " *
                       "fitted mean. coef_fixed entries, if any, are fixed at zero.", ci = ci,
                gradient_max = _bridge_gradient_max_gaussian(fit, Yf, Xarr, nothing))
            return merge(base, (mean_coef = β,
                                mean_coef_status = _fixed_status(coef_fixed)))
        end
        reml = _bridge_truthy(_bridge_get(options, "reml", false))
        if reml
            # Restricted ML: per-trait intercepts enter as the GLS fixed effects X
            # (q = p, Xₜₛₜ = 1), so the trait means are REML-adjusted rather than
            # pre-centred. The fitted Λ̂/σ̂ are wrapped in a GllvmFit so the SAME
            # Gaussian extractors build the flat contract; the GLS β̂ is the alpha.
            Xrt = zeros(Float64, p, n, p)
            @inbounds for t in 1:p, s in 1:n
                Xrt[t, s, t] = 1.0
            end
            rfit = fit_gaussian_reml(Yf, Xrt; K = K)
            alpha = collect(Float64, rfit.β)
            fit = GllvmFit(GllvmModel(p, K),
                (σ_eps = rfit.σ_eps, Λ = rfit.Λ, β = nothing,
                 Λ_W = nothing, σ²_B = nothing, σ²_W = nothing,
                 Λ_phy = nothing, σ_phy = nothing, θ_packed = Float64[]),
                rfit.reml_loglik, rfit.iterations, rfit.converged, nothing, NaN)
            Yc = Yf .- alpha
            Sigma = Matrix{Float64}(sigma_y_site(fit))
            corr  = Matrix{Float64}(correlation(fit))
            comm  = Vector{Float64}(communality(fit))
            scores = _bridge_scores(() -> getLV(fit, Yc; rotate = true))
            df = p + _bridge_rr_df(p, K) + 1
            # REML's synthetic GllvmFit carries no packed vector ⇒ no observed-
            # information / profile substrate; _bridge_compute_ci_gaussian returns a
            # skip-with-note payload rather than fabricating bounds.
            ci = ci_method == "none" ? nothing :
                 _bridge_compute_ci_gaussian(fit, Yc, ci_method, ci_level, ci_nboot, ci_seed)
            return _bridge_assemble(fit, "gaussian", "gaussian_reml_rr", traits, units;
                alpha = alpha, dispersion = fill(NaN, p), sigma_eps = rfit.σ_eps,
                link = fill("IdentityLink", p), Sigma = Sigma, corr = corr, comm = comm,
                scores = scores, df = df, loglik = rfit.reml_loglik,
                converged = rfit.converged, iterations = rfit.iterations,
                note = "REML fit (restricted ML): loglik is the REML criterion, " *
                       "not directly comparable to ML loglik; alpha are GLS trait means.",
                ci = ci,
                # Same empty-θ_packed guard as the CI skip above: NaN, not fabricated.
                gradient_max = _bridge_gradient_max_gaussian(fit, Yc, nothing, nothing))
        end
        alpha = vec(Statistics.mean(Yf; dims = 2))
        Yc = Yf .- alpha
        fit = fit_gaussian_gllvm(Yc; K = K)
        Sigma = Matrix{Float64}(sigma_y_site(fit))
        corr  = Matrix{Float64}(correlation(fit))
        comm  = Vector{Float64}(communality(fit))
        scores = _bridge_scores(() -> getLV(fit, Yc; rotate = true))
        df = p + _bridge_rr_df(p, K) + 1
        ci = ci_method == "none" ? nothing :
             _bridge_compute_ci_gaussian(fit, Yc, ci_method, ci_level, ci_nboot, ci_seed)
        return _bridge_assemble(fit, "gaussian", "gaussian_rr", traits, units;
            alpha = alpha, dispersion = fill(NaN, p), sigma_eps = fit.pars.σ_eps,
            link = fill("IdentityLink", p), Sigma = Sigma, corr = corr, comm = comm,
            scores = scores, df = df, loglik = fit.logLik,
            converged = fit.converged, iterations = fit.n_iter, note = "", ci = ci,
            gradient_max = _bridge_gradient_max_gaussian(fit, Yc, nothing, nothing))
    end

    # Non-Gaussian: fit, then build the latent-scale derived quantities. The six
    # families with link-residual extractors on main (poisson/binomial/negbinomial/
    # beta/gamma/ordinal) get the real cross-family Sigma/correlation/communality;
    # NB1 (no extractor yet) falls back to the shared block via _bridge_assemble_ng.
    # Non-Gaussian CI: the unified confint(fit, Y; method, …) covers wald/profile/
    # bootstrap for all six families below (the bridge fit objects are exactly the
    # _CIFit types it dispatches on). Pass Float64 data so the parity oracle (which
    # uses Float64.(Y)) matches to machine precision; nb1 routes too (its FamilyFit
    # is in _CIFit even though its latent-scale extractor is not yet present).
    if key == "poisson"
        Yi = _bridge_require_integer_counts(Yf, key)
        if X_lv !== nothing
            Xlv = Matrix{Float64}(X_lv)
            size(Xlv, 1) == n || throw(ArgumentError(
                "bridge_fit: X_lv must be n×q_lv ($(n)×q_lv); got $(size(Xlv))"))
            size(Xlv, 2) > 0 || throw(ArgumentError(
                "bridge_fit: X_lv must have at least one predictor column"))
            fit = fit_poisson_gllvm(Yi; K = K, X_lv = Xlv)
            ci_extra = ci_method == "wald" ?
                _bridge_lv_ci_fields(confint_lv_effects(fit, Yi, Xlv; level = ci_level),
                                     size(Xlv, 2)) : NamedTuple()
            scores_total = Matrix{Float64}(
                getLV(fit, Yi; X_lv = Xlv, component = :total, rotate = true))
            scores_mean = Matrix{Float64}(
                getLV(fit, Yi; X_lv = Xlv, component = :mean, rotate = true))
            scores_innovation = Matrix{Float64}(
                getLV(fit, Yi; X_lv = Xlv, component = :innovation, rotate = true))
            base = _bridge_assemble_ng(fit, "poisson", "poisson_xlv_rr", traits, units,
                p, K, Yi, nothing;
                alpha = fit.β, dispersion = fill(NaN, p),
                df = _nparams(fit), scores = scores_total, ci = nothing,
                mask = nothing)
            xlv_note = "predictor-informed latent-score fit (Poisson C1): " *
                       "scores are total latent scores, scores_mean = " *
                       "X_lv*alpha_lv, scores_innovation are the posterior " *
                       "zero-mean Laplace score modes, and lv_effects = " *
                       "Lambda*alpha_lv' is the rotation-stable trait-effect " *
                       "matrix. Wald confidence intervals on B_lv are available via ci_method=\"wald\"; profile/bootstrap, response masks, mixed-family X_lv, and source-specific X_lv remain separate validation gates."
            return merge(base, (note = isempty(base.note) ? xlv_note :
                                      string(base.note, " ", xlv_note),
                                lv_effects = Matrix{Float64}(extract_lv_effects(fit)),
                                alpha_lv = Matrix{Float64}(fit.alpha_lv),
                                scores_mean = scores_mean,
                                scores_innovation = scores_innovation), ci_extra)
        end
        fit = fit_poisson_gllvm(Yi; K = K, mask = M)
        scores = _bridge_scores(() -> getLV(fit, Yi; rotate = true, mask = M))
        ci = ci_method == "none" ? nothing :
             _bridge_compute_ci_ng(fit, Float64.(Yi), nothing, ci_method, ci_level, ci_nboot, ci_seed; mask = M)
        return _bridge_assemble_ng(fit, "poisson", "poisson_rr", traits, units, p, K, Yi, nothing;
            alpha = fit.β, dispersion = fill(NaN, p), df = p + _bridge_rr_df(p, K),
            scores = scores, ci = ci, mask = M)
    elseif key == "lognormal"
        # No-X only. LognormalFit has no getLV / link-residual / confint adapter
        # on this engine, so assemble the shared block ΛΛᵀ directly (the same
        # honest fallback `_bridge_assemble_ng` uses) and fence CI / X / X_lv /
        # masks via the existing list membership plus a loud CI guard.
        all(>(0), Yf) || throw(ArgumentError(
            "bridge_fit: family=\"lognormal\" requires y > 0; found non-positive response"))
        _bridge_ci_guard_lognormal(ci_method)
        fit = fit_lognormal_gllvm(Yf; K = K)
        L = Matrix{Float64}(fit.Λ * _svd_rotation(fit.Λ))
        Σ = L * L'; Σ = (Σ + Σ') ./ 2
        corr = _bridge_corr_from_sigma(Σ)
        comm = ones(Float64, p)
        return _bridge_assemble(fit, "lognormal", "lognormal_rr", traits, units;
            alpha = collect(Float64, fit.β),
            dispersion = fill(Float64(fit.σ), p),
            sigma_eps = NaN,
            link = fill(_bridge_link_name(fit.link), p),
            Sigma = Σ, corr = corr, comm = comm,
            scores = zeros(Float64, 0, 0),
            df = p + _bridge_rr_df(p, K) + 1,
            loglik = fit.loglik, converged = fit.converged,
            iterations = fit.iterations, loadings = L,
            note = "one-part lognormal (twin fid 3): log(y) ~ Normal(η, σ²), " *
                   "shared scalar σ; y-scale loglik includes −Σ log y; " *
                   "no-X only; Sigma/correlation use the shared block " *
                   "Lambda*Lambda' only (communality 1); scores empty " *
                   "(no getLV(::LognormalFit)); CI, X, X_lv, and masks " *
                   "remain follow-ups; light no-X RCall logLik Δ PAID " *
                   "(test/parity/test_lognormal_parity.jl @ GLLVM_PARITY_TESTS=1; " *
                   "receipt abs≈2.24e-8 @ rtol 1e-6, seed=52, p=5, K=2, n=60)",
            ci = nothing,
            # gradient_max: NaN — LognormalFit has no _family_ci adapter (no packed
            # objective to rebuild on this engine yet).
            gradient_max = NaN)
    elseif key == "truncated_poisson"
        # No-X only. TruncatedPoissonFit has no getLV / link-residual / confint
        # adapter on this engine, so assemble the shared block ΛΛᵀ directly
        # (same honest fallback as lognormal) and fence CI / X / X_lv / masks
        # via list membership plus a loud CI guard. Frozen twin fid 10 admits
        # positive integers only. Validate before conversion or CI dispatch:
        # rounding here would fit a response different from the supplied one.
        all(eachindex(Yf)) do i
            v = Yf[i]
            isfinite(v) && v >= 1 && isinteger(v) &&
                v < Float64(typemax(Int)) && v == y[i]
        end || throw(ArgumentError(
            "bridge_fit: family=\"truncated_poisson\" requires positive integer " *
            "responses y ≥ 1 exactly representable by the bridge as Int; values are not rounded"))
        _bridge_ci_guard_truncated_poisson(ci_method)
        Yi = Int.(Yf)
        fit = fit_truncated_poisson_gllvm(Yi; K = K)
        L = Matrix{Float64}(fit.Λ * _svd_rotation(fit.Λ))
        Σ = L * L'; Σ = (Σ + Σ') ./ 2
        corr = _bridge_corr_from_sigma(Σ)
        comm = ones(Float64, p)
        return _bridge_assemble(fit, "truncated_poisson", "truncated_poisson_rr",
            traits, units;
            alpha = collect(Float64, fit.β),
            dispersion = fill(NaN, p),
            sigma_eps = NaN,
            link = fill(_bridge_link_name(fit.link), p),
            Sigma = Σ, corr = corr, comm = comm,
            scores = zeros(Float64, 0, 0),
            df = p + _bridge_rr_df(p, K),
            loglik = fit.loglik, converged = fit.converged,
            iterations = fit.iterations, loadings = L,
            note = "one-part truncated Poisson (twin fid 10): " *
                   "y ~ TruncPois(μ = exp(η)), log link on the untruncated " *
                   "mean; support y ≥ 1; no-X only; Sigma/correlation use " *
                   "the shared block Lambda*Lambda' only (communality 1); " *
                   "scores empty (no getLV(::TruncatedPoissonFit)); CI, X, " *
                   "X_lv, and masks remain follow-ups; light no-X RCall logLik Δ PAID " *
                   "(test/parity/test_truncated_poisson_parity.jl @ GLLVM_PARITY_TESTS=1; " *
                   "receipt abs≈2.71e-9 @ rtol 1e-6, seed=53, p=5, K=2, n=60)",
            ci = nothing,
            # gradient_max: NaN — TruncatedPoissonFit has no _family_ci adapter (no
            # packed objective to rebuild on this engine yet).
            gradient_max = NaN)
    elseif key in _BRIDGE_BINOMIAL_FAMILIES
        Yi = _bridge_require_integer_counts(Yf, key)
        Ni = _bridge_require_integer_trials(N, p, n, key)
        link = _bridge_binomial_link(key)
        if X_lv !== nothing
            Xlv = Matrix{Float64}(X_lv)
            size(Xlv, 1) == n || throw(ArgumentError(
                "bridge_fit: X_lv must be n×q_lv ($(n)×q_lv); got $(size(Xlv))"))
            size(Xlv, 2) > 0 || throw(ArgumentError(
                "bridge_fit: X_lv must have at least one predictor column"))
            fit = fit_binomial_gllvm(Yi; K = K, N = Ni, link = link, X_lv = Xlv)
            ci_extra = ci_method == "wald" ?
                _bridge_lv_ci_fields(confint_lv_effects(fit, Yi, Xlv; N = Ni, level = ci_level),
                                     size(Xlv, 2)) : NamedTuple()
            scores_total = Matrix{Float64}(
                getLV(fit, Yi; N = Ni, X_lv = Xlv, component = :total,
                      rotate = true))
            scores_mean = Matrix{Float64}(
                getLV(fit, Yi; N = Ni, X_lv = Xlv, component = :mean,
                      rotate = true))
            scores_innovation = Matrix{Float64}(
                getLV(fit, Yi; N = Ni, X_lv = Xlv, component = :innovation,
                      rotate = true))
            base = _bridge_assemble_ng(fit, key, "$(key)_xlv_rr", traits, units,
                p, K, Yi, Ni;
                alpha = fit.β, dispersion = fill(NaN, p),
                df = _nparams(fit), scores = scores_total, ci = nothing,
                mask = nothing)
            xlv_note = "predictor-informed latent-score fit (binomial C1): " *
                       "scores are total latent scores, scores_mean = " *
                       "X_lv*alpha_lv, scores_innovation are the posterior " *
                       "zero-mean Laplace score modes, and lv_effects = " *
                       "Lambda*alpha_lv' is the rotation-stable trait-effect " *
                       "matrix. Wald confidence intervals on B_lv are available via ci_method=\"wald\"; profile/bootstrap, response masks, mixed-family X_lv, and source-specific X_lv remain separate validation gates."
            return merge(base, (note = isempty(base.note) ? xlv_note :
                                      string(base.note, " ", xlv_note),
                                lv_effects = Matrix{Float64}(extract_lv_effects(fit)),
                                alpha_lv = Matrix{Float64}(fit.alpha_lv),
                                scores_mean = scores_mean,
                                scores_innovation = scores_innovation), ci_extra)
        end
        fit = fit_binomial_gllvm(Yi; K = K, N = Ni, link = link, mask = M)
        scores = _bridge_scores(() -> getLV(fit, Yi; N = Ni, rotate = true, mask = M))
        ci = ci_method == "none" ? nothing :
             _bridge_compute_ci_ng(fit, Float64.(Yi), Ni, ci_method, ci_level, ci_nboot, ci_seed; mask = M)
        return _bridge_assemble_ng(fit, key, "$(key)_rr", traits, units, p, K, Yi, Ni;
            alpha = fit.β, dispersion = fill(NaN, p), df = p + _bridge_rr_df(p, K),
            scores = scores, ci = ci, mask = M)
    elseif key == "negbinomial"
        Yi = _bridge_require_integer_counts(Yf, key)
        if X_lv !== nothing
            Xlv = Matrix{Float64}(X_lv)
            size(Xlv, 1) == n || throw(ArgumentError(
                "bridge_fit: X_lv must be n×q_lv ($(n)×q_lv); got $(size(Xlv))"))
            size(Xlv, 2) > 0 || throw(ArgumentError(
                "bridge_fit: X_lv must have at least one predictor column"))
            # X_lv route uses the shared-dispersion NB2 fitter (not the per-trait
            # grouped route) — a narrow point-estimate predictor-informed slice.
            fit = fit_nb_gllvm(Yi; K = K, X_lv = Xlv)
            ci_extra = ci_method == "wald" ?
                _bridge_lv_ci_fields(confint_lv_effects(fit, Yi, Xlv; level = ci_level),
                                     size(Xlv, 2)) : NamedTuple()
            scores_total = Matrix{Float64}(
                getLV(fit, Yi; X_lv = Xlv, component = :total, rotate = true))
            scores_mean = Matrix{Float64}(
                getLV(fit, Yi; X_lv = Xlv, component = :mean, rotate = true))
            scores_innovation = Matrix{Float64}(
                getLV(fit, Yi; X_lv = Xlv, component = :innovation, rotate = true))
            base = _bridge_assemble_ng(fit, "negbinomial", "negbinomial_xlv_rr",
                traits, units, p, K, Yi, nothing;
                alpha = fit.β, dispersion = fill(fit.r, p),
                df = _nparams(fit), scores = scores_total, ci = nothing,
                mask = nothing)
            xlv_note = "predictor-informed latent-score fit (NB2 C1): scores are " *
                       "total latent scores, scores_mean = X_lv*alpha_lv, " *
                       "scores_innovation are the posterior zero-mean Laplace score " *
                       "modes, and lv_effects = Lambda*alpha_lv' is the " *
                       "rotation-stable trait-effect matrix; the shared NB2 " *
                       "dispersion r is jointly estimated. Wald confidence intervals on B_lv are available via ci_method=\"wald\"; profile/bootstrap, response masks, grouped dispersion, mixed-family X_lv, and source-specific X_lv remain separate validation gates."
            return merge(base, (note = isempty(base.note) ? xlv_note :
                                      string(base.note, " ", xlv_note),
                                lv_effects = Matrix{Float64}(extract_lv_effects(fit)),
                                alpha_lv = Matrix{Float64}(fit.alpha_lv),
                                scores_mean = scores_mean,
                                scores_innovation = scores_innovation), ci_extra)
        end
        fit = fit_nb_gllvm_grouped(Yi; K = K, group = collect(1:p), mask = M)
        disp = _bridge_dispersion_payload(fit.r_group, fit.group, "r",
            "Var = mu + mu^2 / r",
            "gllvm phi = 1 / r; gllvmTMB sigma = 1 / sqrt(r)")
        scores = _bridge_scores(() -> getLV(fit, Yi; rotate = true, mask = M))
        ci = ci_method == "none" ? nothing :
             _bridge_compute_ci_ng(fit, Float64.(Yi), nothing, ci_method, ci_level, ci_nboot, ci_seed; mask = M)
        base = _bridge_assemble_ng(fit, "negbinomial", "negbinomial_rr", traits, units, p, K, Yi, nothing;
            alpha = fit.β, dispersion = _bridge_expand_dispersion(disp), df = _bridge_group_df(p, K, disp),
            scores = scores, ci = ci, mask = M)
        return merge(base, disp)
    elseif key == "nb1"
        Yi = _bridge_require_integer_counts(Yf, key)
        fit = fit_nb1_gllvm_grouped(Yi; K = K, group = collect(1:p), mask = M)
        disp = _bridge_dispersion_payload(fit.φ, fit.group, "phi",
            "Var = mu * (1 + phi)",
            "identity on the NB1 overdispersion scale")
        scores = _bridge_scores(() -> getLV(fit, Yi; rotate = true, mask = M))
        ci = ci_method == "none" ? nothing :
             _bridge_compute_ci_ng(fit, Float64.(Yi), nothing, ci_method, ci_level, ci_nboot, ci_seed; mask = M)
        base = _bridge_assemble_ng(fit, "nb1", "nb1_rr", traits, units, p, K, Yi, nothing;
            alpha = fit.β, dispersion = _bridge_expand_dispersion(disp), df = _bridge_group_df(p, K, disp),
            scores = scores, ci = ci, mask = M)
        return merge(base, disp)
    elseif key == "beta"
        if X_lv !== nothing
            Xlv = Matrix{Float64}(X_lv)
            size(Xlv, 1) == n || throw(ArgumentError(
                "bridge_fit: X_lv must be n×q_lv ($(n)×q_lv); got $(size(Xlv))"))
            size(Xlv, 2) > 0 || throw(ArgumentError(
                "bridge_fit: X_lv must have at least one predictor column"))
            # X_lv route uses the shared-precision Beta fitter (not the per-trait
            # grouped route) — a narrow point-estimate slice.
            fit = fit_beta_gllvm(Yf; K = K, X_lv = Xlv)
            ci_extra = ci_method == "wald" ?
                _bridge_lv_ci_fields(confint_lv_effects(fit, Yf, Xlv; level = ci_level),
                                     size(Xlv, 2)) : NamedTuple()
            scores_total = Matrix{Float64}(
                getLV(fit, Yf; X_lv = Xlv, component = :total, rotate = true))
            scores_mean = Matrix{Float64}(
                getLV(fit, Yf; X_lv = Xlv, component = :mean, rotate = true))
            scores_innovation = Matrix{Float64}(
                getLV(fit, Yf; X_lv = Xlv, component = :innovation, rotate = true))
            base = _bridge_assemble_ng(fit, "beta", "beta_xlv_rr",
                traits, units, p, K, Yf, nothing;
                alpha = fit.β, dispersion = fill(fit.φ, p),
                df = _nparams(fit), scores = scores_total, ci = nothing,
                mask = nothing)
            xlv_note = "predictor-informed latent-score fit (Beta C1): scores are " *
                       "total latent scores, scores_mean = X_lv*alpha_lv, " *
                       "scores_innovation are the posterior zero-mean Laplace score " *
                       "modes, and lv_effects = Lambda*alpha_lv' is the " *
                       "rotation-stable trait-effect matrix; the shared precision " *
                       "phi is jointly estimated. Wald confidence intervals on B_lv are available via ci_method=\"wald\"; profile/bootstrap, response masks, per-trait precision, mixed-family X_lv, and source-specific X_lv remain separate validation gates."
            return merge(base, (note = isempty(base.note) ? xlv_note :
                                      string(base.note, " ", xlv_note),
                                lv_effects = Matrix{Float64}(extract_lv_effects(fit)),
                                alpha_lv = Matrix{Float64}(fit.alpha_lv),
                                scores_mean = scores_mean,
                                scores_innovation = scores_innovation), ci_extra)
        end
        fit = fit_beta_gllvm_grouped(Yf; K = K, group = collect(1:p), mask = M)
        disp = _bridge_dispersion_payload(fit.φ, fit.group, "phi",
            "Var = mu * (1 - mu) / (1 + phi)",
            "gllvmTMB sigma = 1 / sqrt(phi)")
        scores = _bridge_scores(() -> getLV(fit, Yf; rotate = true, mask = M))
        ci = ci_method == "none" ? nothing :
             _bridge_compute_ci_ng(fit, Yf, nothing, ci_method, ci_level, ci_nboot, ci_seed; mask = M)
        base = _bridge_assemble_ng(fit, "beta", "beta_rr", traits, units, p, K, Yf, nothing;
            alpha = fit.β, dispersion = _bridge_expand_dispersion(disp), df = _bridge_group_df(p, K, disp),
            scores = scores, ci = ci, mask = M)
        return merge(base, disp)
    elseif key == "gamma"
        if X_lv !== nothing
            Xlv = Matrix{Float64}(X_lv)
            size(Xlv, 1) == n || throw(ArgumentError(
                "bridge_fit: X_lv must be n×q_lv ($(n)×q_lv); got $(size(Xlv))"))
            size(Xlv, 2) > 0 || throw(ArgumentError(
                "bridge_fit: X_lv must have at least one predictor column"))
            # X_lv route uses the shared-shape Gamma fitter (consistent with the
            # no-X shared-shape bridge route) — a narrow point-estimate slice.
            fit = fit_gamma_gllvm(Yf; K = K, X_lv = Xlv)
            ci_extra = ci_method == "wald" ?
                _bridge_lv_ci_fields(confint_lv_effects(fit, Yf, Xlv; level = ci_level),
                                     size(Xlv, 2)) : NamedTuple()
            scores_total = Matrix{Float64}(
                getLV(fit, Yf; X_lv = Xlv, component = :total, rotate = true))
            scores_mean = Matrix{Float64}(
                getLV(fit, Yf; X_lv = Xlv, component = :mean, rotate = true))
            scores_innovation = Matrix{Float64}(
                getLV(fit, Yf; X_lv = Xlv, component = :innovation, rotate = true))
            base = _bridge_assemble_ng(fit, "gamma", "gamma_xlv_rr",
                traits, units, p, K, Yf, nothing;
                alpha = fit.β, dispersion = fill(fit.α, p),
                df = _nparams(fit), scores = scores_total, ci = nothing,
                mask = nothing)
            xlv_note = "predictor-informed latent-score fit (Gamma C1): scores are " *
                       "total latent scores, scores_mean = X_lv*alpha_lv, " *
                       "scores_innovation are the posterior zero-mean Laplace score " *
                       "modes, and lv_effects = Lambda*alpha_lv' is the " *
                       "rotation-stable trait-effect matrix; the shared shape alpha " *
                       "is jointly estimated. Wald confidence intervals on B_lv are available via ci_method=\"wald\"; profile/bootstrap, response masks, mixed-family X_lv, and source-specific X_lv remain separate validation gates."
            return merge(base, (note = isempty(base.note) ? xlv_note :
                                      string(base.note, " ", xlv_note),
                                lv_effects = Matrix{Float64}(extract_lv_effects(fit)),
                                alpha_lv = Matrix{Float64}(fit.alpha_lv),
                                scores_mean = scores_mean,
                                scores_innovation = scores_innovation), ci_extra)
        end
        # Native gllvmTMB ordinary Gamma currently has one scalar sigma_eps/CV for
        # all Gamma traits. Use a single grouped-Gamma shape here for R-oracle
        # parity; the per-trait grouped Gamma engine remains available for a later
        # native per-trait Gamma expansion.
        fit = fit_gamma_gllvm_grouped(Yf; K = K, group = fill(1, p), mask = M)
        disp = _bridge_dispersion_payload(fit.α, fit.group, "alpha",
            "Var = mu^2 / alpha",
            "gllvmTMB sigma = 1 / sqrt(alpha)")
        scores = _bridge_scores(() -> getLV(fit, Yf; rotate = true, mask = M))
        ci = ci_method == "none" ? nothing :
             _bridge_compute_ci_ng(fit, Yf, nothing, ci_method, ci_level, ci_nboot, ci_seed; mask = M)
        base = _bridge_assemble_ng(fit, "gamma", "gamma_rr", traits, units, p, K, Yf, nothing;
            alpha = fit.β, dispersion = _bridge_expand_dispersion(disp), df = _bridge_group_df(p, K, disp),
            scores = scores, ci = ci, mask = M)
        return merge(base, disp)
    elseif key == "betabinomial"
        Yi = _bridge_require_integer_counts(Yf, key)
        Ni = _bridge_require_integer_trials(N, p, n, key)
        fit = fit_beta_binomial_gllvm_grouped(Yi; K = K, N = Ni, group = collect(1:p), mask = M)
        disp = _bridge_dispersion_payload(fit.φ, fit.group, "phi",
            "Var = N * mu * (1 - mu) * (1 + (N - 1) * phi / (phi + 1))",
            "gllvm Beta precision phi = a + b (twin log_phi_betabinom); direct passthrough")
        scores = _bridge_scores(() -> getLV(fit, Yi; N = Ni, rotate = true, mask = M))
        ci = ci_method == "none" ? nothing :
             _bridge_compute_ci_ng(fit, Yi, Ni, ci_method, ci_level, ci_nboot, ci_seed; mask = M)
        base = _bridge_assemble_ng(fit, "betabinomial", "betabinomial_rr", traits, units, p, K, Yi, Ni;
            alpha = fit.β, dispersion = _bridge_expand_dispersion(disp), df = _bridge_group_df(p, K, disp),
            scores = scores, ci = ci, mask = M)
        return merge(base, disp)
    elseif key in ("ordinal", "ordinal_probit")
        Yi = _bridge_require_integer_counts(Yf, key)
        link = key == "ordinal_probit" ? ProbitLink() : LogitLink()
        _bridge_ci_guard_pertrait_ordinal(key, ci_method)
        fit = fit_ordinal_gllvm_pertrait(Yi; K = K, link = link, mask = M)
        scores = _bridge_scores(() -> getLV(fit, Yi; rotate = true, mask = M))
        family_out = key == "ordinal_probit" ? "ordinal_probit" : "ordinal"
        model_out = key == "ordinal_probit" ? "ordinal_probit_rr" : "ordinal_rr"
        # gradient_max resolves to NaN here: OrdinalPerTraitFit has no _family_ci
        # adapter (no packed objective to rebuild on this engine yet).
        base = _bridge_assemble_ng(fit, family_out, model_out, traits, units, p, K, Yi, nothing;
            alpha = fit.β, dispersion = fill(NaN, p),
            df = _nparams(fit), scores = scores, ci = nothing, mask = M)
        # Ordinal-only FLAT extras (ASCII keys, primitive arrays): per-trait
        # ordered cutpoints (NaN-padded after each trait's final threshold) and
        # per-trait category counts. This is the native gllvmTMB parity shape.
        return merge(base, (cutpoints = Matrix{Float64}(fit.τ),
                            n_categories = Vector{Int}(fit.C),
                            cutpoint_mode = "per_trait",
                            cutpoint_link = _bridge_link_name(fit.link)))
    elseif key == "zip"
        Yi = _bridge_require_integer_counts(Yf, key)
        M === nothing || throw(ArgumentError(
            "bridge_fit: missing-response masks are not wired for family=\"zip\" yet"))
        fit = fit_zip_gllvm(Yi; K = K)
        scores = _bridge_scores(() -> getLV(fit, Yi; rotate = true))
        ci = ci_method == "none" ? nothing :
             _bridge_compute_ci_ng(fit, Float64.(Yi), nothing, ci_method, ci_level,
                                   ci_nboot, ci_seed; mask = nothing)
        # ZIPFit carries no `link` field and no link-residual extractor, so assemble
        # directly (as ZINB/ZIB do) rather than through _bridge_assemble_ng, whose
        # `fit.link` read does not apply here: build Sigma/correlation/communality
        # from the shared block ΛcΛcᵀ and name the count-part link explicitly.
        L = Matrix{Float64}(getLoadings(fit; rotate = true))
        Σ = L * L'; Σ = (Σ + Σ') ./ 2
        corr = _bridge_corr_from_sigma(Σ)
        comm = ones(Float64, p)
        base = _bridge_assemble(fit, "zip", "zip_rr", traits, units;
            alpha = collect(Float64, fit.βc),
            dispersion = fill(NaN, p),
            sigma_eps = NaN,
            link = fill("log", p), Sigma = Σ, corr = corr,
            comm = comm, scores = scores, df = _nparams(fit),
            loglik = fit.loglik, converged = fit.converged,
            iterations = fit.iterations, loadings = L,
            note = "ZIP no-X (Julia-forward / twin-asymmetric): structural-zero " *
                   "logits beta_zero, count intercepts alpha=beta_c, Λz=0; " *
                   "Sigma/correlation use the shared block Lambda*Lambda' only " *
                   "(communality 1); no twin light Δ.",
            ci = ci, gradient_max = _bridge_gradient_max_family(fit, Float64.(Yi)))
        return merge(base, (beta_zero = collect(Float64, fit.βz),))
    elseif key == "zinb"
        Yi = _bridge_require_integer_counts(Yf, key)
        M === nothing || throw(ArgumentError(
            "bridge_fit: missing-response masks are not wired for family=\"zinb\" yet"))
        fit = fit_zinb_gllvm(Yi; K = K)
        scores = _bridge_scores(() -> getLV(fit, Yi; rotate = true))
        ci = ci_method == "none" ? nothing :
             _bridge_compute_ci_ng(fit, Float64.(Yi), nothing, ci_method, ci_level,
                                   ci_nboot, ci_seed; mask = nothing)
        L = Matrix{Float64}(getLoadings(fit; rotate = true))
        Σ = L * L'; Σ = (Σ + Σ') ./ 2
        corr = _bridge_corr_from_sigma(Σ)
        comm = ones(Float64, p)
        base = _bridge_assemble(fit, "zinb", "zinb_rr", traits, units;
            alpha = collect(Float64, fit.βc),
            dispersion = fill(Float64(fit.r), p),
            sigma_eps = NaN,
            link = fill("log", p), Sigma = Σ, corr = corr,
            comm = comm, scores = scores, df = _nparams(fit),
            loglik = fit.loglik, converged = fit.converged,
            iterations = fit.iterations, loadings = L,
            note = "ZINB no-X (Julia-forward / twin-asymmetric): structural-zero " *
                   "logits beta_zero, count intercepts alpha=beta_c, Λz=0, " *
                   "shared scalar r; no twin light Δ.",
            ci = ci, gradient_max = _bridge_gradient_max_family(fit, Float64.(Yi)))
        return merge(base, (beta_zero = collect(Float64, fit.βz),))
    elseif key == "zib"
        Yi = _bridge_require_integer_counts(Yf, key)
        M === nothing || throw(ArgumentError(
            "bridge_fit: missing-response masks are not wired for family=\"zib\" yet"))
        Ni = _bridge_zib_trials(N, p, n)
        all(v -> 0 <= v <= Ni, Yi) || throw(ArgumentError(
            "bridge_fit: family=\"zib\" responses must lie in 0:N (N = $(Ni)); got " *
            "min $(minimum(Yi)), max $(maximum(Yi))"))
        fit = fit_zib_gllvm(Yi; K = K, N = Ni)
        scores = _bridge_scores(() -> getLV(fit, Yi; rotate = true))
        ci = ci_method == "none" ? nothing :
             _bridge_compute_ci_ng(fit, Float64.(Yi), nothing, ci_method, ci_level,
                                   ci_nboot, ci_seed; mask = nothing)
        # ZIBFit carries no `link` field and no link-residual extractor, so build
        # Sigma/correlation/communality from the shared block ΛcΛcᵀ directly (the
        # same honest fallback _bridge_assemble_ng applies elsewhere) and name the
        # count-part link explicitly.
        L = Matrix{Float64}(getLoadings(fit; rotate = true))
        Σ = L * L'; Σ = (Σ + Σ') ./ 2
        corr = _bridge_corr_from_sigma(Σ)
        comm = ones(Float64, p)
        base = _bridge_assemble(fit, "zib", "zib_rr", traits, units;
            alpha = collect(Float64, fit.βc),
            dispersion = fill(NaN, p),
            sigma_eps = NaN,
            link = fill(_bridge_link_name(LogitLink()), p), Sigma = Σ, corr = corr,
            comm = comm, scores = scores, df = _nparams(fit),
            loglik = fit.loglik, converged = fit.converged,
            iterations = fit.iterations, loadings = L,
            note = "ZIB no-X (Julia-forward / twin-asymmetric): structural-zero " *
                   "logits beta_zero, success-logit intercepts alpha=beta_c, Λz=0, " *
                   "one shared scalar trials count N (required at the boundary; " *
                   "not per-observation cbind); Sigma/correlation use the shared " *
                   "block Lambda*Lambda' only (communality 1); no twin light Δ — " *
                   "the twin gllvmTMB has no ZIB.",
            ci = ci, gradient_max = _bridge_gradient_max_family(fit, Float64.(Yi)))
        return merge(base, (beta_zero = collect(Float64, fit.βz),
                            trials = Ni))
    end
    throw(ArgumentError("bridge_fit: unhandled family key \"$key\""))  # unreachable
end

# --- one-part NON-Gaussian covariate dispatch (fit_gllvm_cov) ---------------
#
# Route the one-part non-Gaussian families that carry a covariate kernel
# (_BRIDGE_X_FAMILIES) through `fit_gllvm_cov`, whose linear predictor is
# η_{ts} = β_t + Σ_k X[t,s,k]·γ_k + (Λ z_s)_t. The flat contract MIRRORS the
# Gaussian-X return (loadings, alpha, dispersion, Sigma/correlation/communality,
# scores, df, …) and ADDS two coefficient arrays the R side reads to fill the
# covariate coefficient table:
#
#   beta_cov :: Vector{Float64}  — per-trait intercepts β (length p)
#   gamma    :: Vector{Float64}  — shared covariate coefficients γ (length q)
#
# `alpha` mirrors β (the per-trait intercept on the link scale) so the existing
# intercept field stays meaningful. Σ_y/correlation/communality use the shared
# block ΛΛᵀ (GllvmCovFit has no link-residual extractor yet — same honest fallback
# as NB1 in _bridge_assemble_ng). CI routing uses native
# confint(fit, Y; X=…, N=…) and returns the same flat bridge CI payload contract
# as no-X fits.
function _bridge_fit_onepart_cov(Yf::AbstractMatrix{Float64}, key::AbstractString,
                                 K::Integer, N, traits, units, X,
                                 ci_method::AbstractString, ci_level::Real,
                                 ci_nboot::Integer, ci_seed::Integer, options)
    p, n = size(Yf)
    Xarr = Array{Float64,3}(X)
    size(Xarr, 1) == p && size(Xarr, 2) == n || throw(ArgumentError(
        "bridge_fit: X must be p×n×q ($(p)×$(n)×q); got $(size(Xarr))"))
    q = size(Xarr, 3)
    coef_fixed = _bridge_coef_fixed(options, q, "coef_fixed")

    # Per-family response coercion + Binomial trial counts (mirror the no-X path):
    # count / ordinal families require exact integer inputs; continuous pass through.
    is_count = key in ("poisson", "binomial", "negbinomial", "betabinomial")
    is_ordinal = key in ("ordinal", "ordinal_probit")
    Ydata = if is_count || is_ordinal
        Float64.(_bridge_require_integer_counts(Yf, key))
    else
        Yf
    end
    Nm = key in ("binomial", "betabinomial") ?
         _bridge_require_integer_trials(N, p, n, key) : nothing

    # Twin API B under X: NB2/NB1/Beta/Gamma default to per-trait φ/α + shared site-X;
    # ordinal/ordinal_probit default to per-trait cutpoints (τ₁=0 / K−2) + shared γ.
    # Shared-dispersion + X remains available via direct `fit_gllvm_cov` where that
    # path exists; shared-cutpoint ordinal stays an explicit Julia comparator.
    if key == "negbinomial"
        Yi = Int.(Ydata)
        fit = fit_nb_gllvm_grouped_cov(Yi; X = Xarr, K = K, group = collect(1:p),
                                       γ_fixed = coef_fixed)
        return _bridge_assemble_grouped_cov(fit, key, traits, units, Yi, Xarr,
                                            coef_fixed, ci_method, ci_level,
                                            ci_nboot, ci_seed; N = nothing)
    elseif key == "nb1"
        Yi = Int.(Ydata)
        fit = fit_nb1_gllvm_grouped_cov(Yi; X = Xarr, K = K, group = collect(1:p),
                                        γ_fixed = coef_fixed)
        return _bridge_assemble_grouped_cov(fit, key, traits, units, Yi, Xarr,
                                            coef_fixed, ci_method, ci_level,
                                            ci_nboot, ci_seed; N = nothing)
    elseif key == "beta"
        fit = fit_beta_gllvm_grouped_cov(Ydata; X = Xarr, K = K, group = collect(1:p),
                                         γ_fixed = coef_fixed)
        return _bridge_assemble_grouped_cov(fit, key, traits, units, Ydata, Xarr,
                                            coef_fixed, ci_method, ci_level,
                                            ci_nboot, ci_seed; N = nothing)
    elseif key == "gamma"
        fit = fit_gamma_gllvm_grouped_cov(Ydata; X = Xarr, K = K, group = collect(1:p),
                                          γ_fixed = coef_fixed)
        return _bridge_assemble_grouped_cov(fit, key, traits, units, Ydata, Xarr,
                                            coef_fixed, ci_method, ci_level,
                                            ci_nboot, ci_seed; N = nothing)
    elseif key == "betabinomial"
        Yi = Int.(Ydata)
        fit = fit_beta_binomial_gllvm_grouped_cov(Yi; X = Xarr, K = K, N = Nm,
                                                  group = collect(1:p), γ_fixed = coef_fixed)
        return _bridge_assemble_grouped_cov(fit, key, traits, units, Yi, Xarr,
                                            coef_fixed, ci_method, ci_level,
                                            ci_nboot, ci_seed; N = Nm)
    elseif key in ("ordinal", "ordinal_probit")
        Yi = Int.(Ydata)
        link = key == "ordinal_probit" ? ProbitLink() : LogitLink()
        _bridge_ci_guard_pertrait_ordinal(key, ci_method)
        fit = fit_ordinal_gllvm_pertrait_cov(Yi; X = Xarr, K = K, link = link,
                                             γ_fixed = coef_fixed)
        return _bridge_assemble_ordinal_cov(fit, key, traits, units, Yi, Xarr,
                                            coef_fixed)
    elseif key == "zip"
        Yi = Int.(Ydata)
        fit = fit_zip_gllvm_cov(Yi; X = Xarr, K = K, γ_fixed = coef_fixed)
        return _bridge_assemble_zip_cov(fit, traits, units, Yi, Xarr, coef_fixed,
                                        ci_method, ci_level, ci_nboot, ci_seed)
    elseif key == "zinb"
        Yi = Int.(Ydata)
        fit = fit_zinb_gllvm_cov(Yi; X = Xarr, K = K, γ_fixed = coef_fixed)
        return _bridge_assemble_zinb_cov(fit, traits, units, Yi, Xarr, coef_fixed,
                                         ci_method, ci_level, ci_nboot, ci_seed)
    end

    marker = _bridge_cov_marker(key)
    fit = Nm === nothing ?
          fit_gllvm_cov(Ydata; family = marker, X = Xarr, K = K, γ_fixed = coef_fixed) :
          fit_gllvm_cov(Ydata; family = marker, X = Xarr, K = K, N = Nm,
                        γ_fixed = coef_fixed)

    β   = collect(Float64, fit.β)
    γ   = collect(Float64, fit.γ)
    L   = Matrix{Float64}(getLoadings(fit; rotate = true))
    disp = fill(Float64(fit.dispersion), p)   # NaN where the family has none

    scores = _bridge_scores(() -> getLV(fit, Ydata, Xarr; rotate = true,
                                        N = (Nm === nothing ? nothing : Nm)))
    ci = ci_method == "none" ? nothing :
         _bridge_compute_ci_cov(fit, Ydata, Nm, Xarr, ci_method, ci_level,
                                ci_nboot, ci_seed)

    # Shared-block latent-scale derived quantities (no link-residual extractor for
    # GllvmCovFit yet): Σ = ΛΛᵀ, correlation from Σ, communality = 1.
    Λr = L
    Σ  = Λr * Λr'; Σ = (Σ + Σ') ./ 2
    corr = _bridge_corr_from_sigma(Σ)
    comm = ones(Float64, p)

    df = p + count(!, coef_fixed) + _bridge_rr_df(p, K) + (isnan(fit.dispersion) ? 0 : 1)
    base = _bridge_assemble(fit, key, "$(key)_x_rr", traits, units;
        alpha = β, dispersion = disp, sigma_eps = NaN,
        link = fill(_bridge_link_name(fit.link), p), Sigma = Σ, corr = corr,
        comm = comm, scores = scores, df = df, loglik = fit.loglik,
        converged = fit.converged, iterations = fit.iterations,
        loadings = L, note =
            "fixed-effect covariate fit (non-Gaussian): eta = beta + X*gamma + " *
            "Lambda*z. beta_cov = per-trait intercepts, gamma = shared covariate " *
            "coefficients. coef_fixed entries, if any, are fixed at zero. " *
            "Sigma/correlation use the shared block Lambda*Lambda' " *
            "(communality 1).",
        ci = ci, gradient_max = _bridge_gradient_max_family(fit, Ydata; X = Xarr, N = Nm))
    return merge(base, (beta_cov = β, gamma = γ, gamma_status = _fixed_status(coef_fixed)))
end

# Assemble the flat bridge contract for per-trait-dispersion + shared-X fits
# (NB2/Beta/Gamma API B under X). Dispersion is length-p (per trait); df counts G
# free dispersion parameters.
function _bridge_assemble_grouped_cov(fit::Union{NBGroupedCovFit, NB1GroupedCovFit, BetaGroupedCovFit, GammaGroupedCovFit, BetaBinomialGroupedCovFit},
                                      key::AbstractString, traits, units,
                                      Ydata, Xarr, coef_fixed,
                                      ci_method::AbstractString, ci_level::Real,
                                      ci_nboot::Integer, ci_seed::Integer; N = nothing)
    p = size(fit.Λ, 1)
    β = collect(Float64, fit.β)
    γ = collect(Float64, fit.γ)
    L = Matrix{Float64}(getLoadings(fit; rotate = true))
    if fit isa NBGroupedCovFit
        disp = Float64[fit.r_group[fit.group[t]] for t in 1:p]
        G = length(fit.r_group)
        disp_note = "per-trait NB2 size r (disp.group); shared site-X gamma."
    elseif fit isa NB1GroupedCovFit
        disp = Float64[fit.φ[fit.group[t]] for t in 1:p]
        G = length(fit.φ)
        disp_note = "per-trait NB1 linear-variance phi (disp.group); shared site-X gamma."
    elseif fit isa BetaGroupedCovFit
        disp = Float64[fit.φ[fit.group[t]] for t in 1:p]
        G = length(fit.φ)
        disp_note = "per-trait Beta precision phi (disp.group); shared site-X gamma."
    elseif fit isa BetaBinomialGroupedCovFit
        disp = Float64[fit.φ[fit.group[t]] for t in 1:p]
        G = length(fit.φ)
        disp_note = "per-trait Beta-binomial precision phi (disp.group); shared site-X " *
                    "gamma; binomial-style trial counts N. Wald/profile/bootstrap CI " *
                    "payloads are routed (finite-difference Hessian)."
    else
        disp = Float64[fit.α[fit.group[t]] for t in 1:p]
        G = length(fit.α)
        disp_note = "per-trait Gamma shape alpha (disp.group); shared site-X gamma."
    end
    # BetaBinomialGroupedCovFit's getLV needs the trial-count matrix N (it
    # defaults to all-ones = Bernoulli otherwise); the other grouped_cov fit
    # types here have no trials concept and ignore N entirely.
    scores = fit isa BetaBinomialGroupedCovFit ?
        _bridge_scores(() -> getLV(fit, Ydata, Xarr; N = N, rotate = true)) :
        _bridge_scores(() -> getLV(fit, Ydata, Xarr; rotate = true))
    ci = ci_method == "none" ? nothing :
         _bridge_compute_ci_cov(fit, Ydata, N, Xarr, ci_method, ci_level,
                                ci_nboot, ci_seed)
    Λr = L
    Σ = Λr * Λr'; Σ = (Σ + Σ') ./ 2
    corr = _bridge_corr_from_sigma(Σ)
    comm = ones(Float64, p)
    df = p + count(!, coef_fixed) + _bridge_rr_df(p, size(fit.Λ, 2)) + G
    base = _bridge_assemble(fit, key, "$(key)_x_rr", traits, units;
        alpha = β, dispersion = disp, sigma_eps = NaN,
        link = fill(_bridge_link_name(fit.link), p), Sigma = Σ, corr = corr,
        comm = comm, scores = scores, df = df, loglik = fit.loglik,
        converged = fit.converged, iterations = fit.iterations,
        loadings = L, note =
            "fixed-effect covariate fit (non-Gaussian, twin API B): eta = beta + " *
            "X*gamma + Lambda*z. $disp_note Shared-dispersion + X remains via " *
            "fit_gllvm_cov. Sigma/correlation use Lambda*Lambda' (communality 1).",
        ci = ci, gradient_max = _bridge_gradient_max_family(fit, Ydata; X = Xarr, N = N))
    return merge(base, (beta_cov = β, gamma = γ, gamma_status = _fixed_status(coef_fixed)))
end

# Assemble ZIP+X under Identity: separate γz/γc, Λz=0; CI via `_family_ci`.
function _bridge_assemble_zip_cov(fit::ZIPCovFit, traits, units, Ydata, Xarr, coef_fixed,
                                  ci_method::AbstractString, ci_level::Real,
                                  ci_nboot::Integer, ci_seed::Integer)
    p = size(fit.Λc, 1)
    βc = collect(Float64, fit.βc)
    βz = collect(Float64, fit.βz)
    γz = collect(Float64, fit.γz)
    γc = collect(Float64, fit.γc)
    L = Matrix{Float64}(getLoadings(fit; rotate = true))
    scores = _bridge_scores(() -> getLV(fit, Ydata, Xarr; rotate = true))
    ci = ci_method == "none" ? nothing :
         _bridge_compute_ci_cov(fit, Ydata, nothing, Xarr, ci_method, ci_level,
                                ci_nboot, ci_seed)
    Λr = L
    Σ = Λr * Λr'; Σ = (Σ + Σ') ./ 2
    corr = _bridge_corr_from_sigma(Σ)
    comm = ones(Float64, p)
    df = _nparams(fit)
    base = _bridge_assemble(fit, "zip", "zip_x_rr", traits, units;
        alpha = βc, dispersion = fill(NaN, p), sigma_eps = NaN,
        link = fill("log", p), Sigma = Σ, corr = corr,
        comm = comm, scores = scores, df = df, loglik = fit.loglik,
        converged = fit.converged, iterations = fit.iterations,
        loadings = L, note =
            "fixed-effect covariate ZIP fit (Julia-forward / twin-asymmetric): " *
            "ηz = βz + X*γz (Λz=0), ηc = βc + X*γc + Λc*z; separate γz/γc. " *
            "Wald/profile/bootstrap CI under X are routed (finite-difference Hessian). " *
            "No twin light RCall Δ (twin ZIP cut). " *
            "Sigma/correlation use Lambda*Lambda' (communality 1).",
        ci = ci, gradient_max = _bridge_gradient_max_family(fit, Ydata; X = Xarr))
    return merge(base, (beta_cov = βc, beta_zero = βz, gamma = γc, gamma_z = γz,
                        gamma_c = γc, gamma_status = _fixed_status(coef_fixed)))
end

# Assemble ZINB+X under Identity: separate γz/γc, Λz=0, shared scalar r.
# CI via `_family_ci` (ZIP clone + log-r tail).
function _bridge_assemble_zinb_cov(fit::ZINBCovFit, traits, units, Ydata, Xarr, coef_fixed,
                                   ci_method::AbstractString, ci_level::Real,
                                   ci_nboot::Integer, ci_seed::Integer)
    p = size(fit.Λc, 1)
    βc = collect(Float64, fit.βc)
    βz = collect(Float64, fit.βz)
    γz = collect(Float64, fit.γz)
    γc = collect(Float64, fit.γc)
    L = Matrix{Float64}(getLoadings(fit; rotate = true))
    scores = _bridge_scores(() -> getLV(fit, Ydata, Xarr; rotate = true))
    ci = ci_method == "none" ? nothing :
         _bridge_compute_ci_cov(fit, Ydata, nothing, Xarr, ci_method, ci_level,
                                ci_nboot, ci_seed)
    Λr = L
    Σ = Λr * Λr'; Σ = (Σ + Σ') ./ 2
    corr = _bridge_corr_from_sigma(Σ)
    comm = ones(Float64, p)
    df = _nparams(fit)
    base = _bridge_assemble(fit, "zinb", "zinb_x_rr", traits, units;
        alpha = βc, dispersion = fill(Float64(fit.r), p), sigma_eps = NaN,
        link = fill("log", p), Sigma = Σ, corr = corr,
        comm = comm, scores = scores, df = df, loglik = fit.loglik,
        converged = fit.converged, iterations = fit.iterations,
        loadings = L, note =
            "fixed-effect covariate ZINB fit (Julia-forward / twin-asymmetric): " *
            "ηz = βz + X*γz (Λz=0), ηc = βc + X*γc + Λc*z; separate γz/γc; " *
            "one shared scalar r (log r). " *
            "Wald/profile/bootstrap CI under X are routed (finite-difference Hessian). " *
            "No twin light RCall Δ (twin ZINB cut). " *
            "Sigma/correlation use Lambda*Lambda' (communality 1).",
        ci = ci, gradient_max = _bridge_gradient_max_family(fit, Ydata; X = Xarr))
    return merge(base, (beta_cov = βc, beta_zero = βz, gamma = γc, gamma_z = γz,
                        gamma_c = γc, gamma_status = _fixed_status(coef_fixed)))
end

# Assemble the flat bridge contract for per-trait ordinal cutpoints + shared-X
# (twin API B under X). Dispersion is NaN (no φ); cutpoint extras mirror no-X.
function _bridge_assemble_ordinal_cov(fit::OrdinalPerTraitCovFit,
                                      key::AbstractString, traits, units,
                                      Ydata, Xarr, coef_fixed)
    p = size(fit.Λ, 1)
    β = collect(Float64, fit.β)
    γ = collect(Float64, fit.γ)
    L = Matrix{Float64}(getLoadings(fit; rotate = true))
    scores = _bridge_scores(() -> getLV(fit, Ydata, Xarr; rotate = true))
    Λr = L
    Σ = Λr * Λr'; Σ = (Σ + Σ') ./ 2
    corr = _bridge_corr_from_sigma(Σ)
    comm = ones(Float64, p)
    df = _nparams(fit)
    family_out = key == "ordinal_probit" ? "ordinal_probit" : "ordinal"
    model_out = key == "ordinal_probit" ? "ordinal_probit_x_rr" : "ordinal_x_rr"
    base = _bridge_assemble(fit, family_out, model_out, traits, units;
        alpha = β, dispersion = fill(NaN, p), sigma_eps = NaN,
        link = fill(_bridge_link_name(fit.link), p), Sigma = Σ, corr = corr,
        comm = comm, scores = scores, df = df, loglik = fit.loglik,
        converged = fit.converged, iterations = fit.iterations,
        loadings = L, note =
            "fixed-effect covariate fit (ordinal, twin API B): eta = beta + " *
            "X*gamma + Lambda*z with per-trait cutpoints (τ₁=0, K−2 log-spacings). " *
            "CI routing remains a follow-up. Sigma/correlation use Lambda*Lambda' " *
            "(communality 1).",
        ci = nothing,
        # gradient_max: NaN — OrdinalPerTraitCovFit has no _family_ci adapter (no
        # packed objective to rebuild on this engine yet).
        gradient_max = NaN)
    return merge(base, (beta_cov = β, gamma = γ, gamma_status = _fixed_status(coef_fixed),
                        cutpoints = Matrix{Float64}(fit.τ),
                        n_categories = Vector{Int}(fit.C),
                        cutpoint_mode = "per_trait",
                        cutpoint_link = _bridge_link_name(fit.link)))
end

# --- mixed-family dispatch (the cross-distribution VCV headline) -----------

# Map a bridge family string to the `Distributions` marker `fit_mixed_gllvm`
# dispatches on. v1 supports the SIX families the mixed fitter supports
# (gaussian/poisson/binomial/negbinomial/gamma/beta); ordinal and nb1 (no mixed
# kernels yet) are documented follow-ups, rejected here with a clear message.
function _bridge_mixed_family_marker(family::AbstractString)
    key = _bridge_family_key(family)
    key == "gaussian"    && return Normal()
    key == "poisson"     && return Poisson()
    key == "binomial"    && return Binomial()
    key == "negbinomial" && return NegativeBinomial(10.0, 0.5)
    key == "gamma"       && return Gamma(2.0, 1.0)
    key == "beta"        && return Beta(10.0, 1.0)
    throw(ArgumentError(
        "bridge_fit (mixed): family \"$family\" is not yet supported per-trait in a " *
        "mixed-family fit; v1 supports gaussian, poisson, binomial, negbinomial, " *
        "gamma, beta. Ordinal and nb1 are documented follow-ups."))
end

# Mixed-family bridge: per-trait families share one latent block Λ; the flat
# contract's `correlation` is the TRUE cross-distribution latent-scale correlation.
# Per-trait response coercion: count families (poisson/binomial/negbinomial) round
# to integer-valued Float64 (the family logpdf takes Int(y)); continuous families
# (gaussian/gamma/beta) pass through as Float64.
function _bridge_fit_mixed(y, family_strs::AbstractVector, K::Integer, N,
                           trait_names, unit_names, options)
    Yf = Matrix{Float64}(y)
    p, n = size(Yf)
    length(family_strs) == p || throw(ArgumentError(
        "bridge_fit (mixed): family vector length $(length(family_strs)) must equal " *
        "the number of traits (rows of y) = $p"))
    traits = _bridge_names(trait_names, p, "trait")
    units = _bridge_names(unit_names, n, "unit")

    # Validate the CI option up-front (a bad ci_method must error loudly even on
    # the mixed path). The mixed fit is a MixedFamilyFit, which the native confint
    # engines do not dispatch on, so any actual CI request is skipped-with-note
    # rather than faked — a documented follow-up.
    ci_method = _bridge_ci_method(options)
    ci_level  = _bridge_ci_level(options)

    keys_norm = [_bridge_family_key(f) for f in family_strs]
    families = [_bridge_mixed_family_marker(f) for f in family_strs]
    links = Link[default_link(fam) for fam in families]

    # Per-trait response matrix: count rows must be exact integers (in Float64);
    # continuous rows pass through unchanged.
    Ymix = copy(Yf)
    is_count = (k -> k in ("poisson", "binomial", "negbinomial"))
    @inbounds for t in 1:p
        is_count(keys_norm[t]) || continue
        for s in 1:n
            v = Yf[t, s]
            _bridge_exact_integer(v, 0) || throw(ArgumentError(
                "bridge_fit (mixed): trait $(t) (family=\"$(keys_norm[t])\") requires " *
                "finite integer counts exactly representable as Int; values are not " *
                "rounded (bad cell row=$(t), col=$(s), value=$(v))"))
            Ymix[t, s] = float(Int(v))
        end
    end

    # Binomial trial counts (p×n; defaults to 1). Only the Binomial rows read N.
    Nm = _bridge_require_integer_trials(N, p, n, "mixed")

    fit = fit_mixed_gllvm(Ymix; families = families, links = links, K = K, N = Nm)

    Sigma = Matrix{Float64}(sigma_y_site(fit, Ymix; N = Nm))
    corr  = Matrix{Float64}(correlation(fit, Ymix; N = Nm))
    comm  = Vector{Float64}(communality(fit, Ymix; N = Nm))
    scores = _bridge_scores(() -> getLV(fit, Ymix; N = Nm, rotate = true))

    # alpha is the per-trait link-scale intercept; dispersion is per-trait (NaN
    # where the family carries none — already the MixedFamilyFit convention).
    alpha = collect(Float64, fit.β)
    dispersion = collect(Float64, fit.dispersion)
    link_names = [_bridge_link_name(links[t]) for t in 1:p]

    # Free-parameter count: p intercepts + reduced-rank loadings + n_disp dispersions.
    df = p + _bridge_rr_df(p, K) + fit.n_disp
    fams_tag = join(keys_norm, "+")

    ci = ci_method == "none" ? nothing :
         _bridge_ci_payload(ci_method, ci_level,
             "CIs are not routed for the mixed-family path yet (the cross-family " *
             "MixedFamilyFit has no native confint engine); a documented follow-up.")

    return _bridge_assemble(fit, fams_tag, "mixed_rr", traits, units;
        alpha = alpha, dispersion = dispersion, sigma_eps = NaN,
        link = link_names, Sigma = Sigma, corr = corr, comm = comm,
        scores = scores, df = df, loglik = fit.loglik,
        converged = fit.converged, iterations = fit.iterations,
        loadings = Matrix{Float64}(fit.Λ * _svd_rotation(fit.Λ)),  # canonical SVD-rotated p×K loadings
        families = keys_norm,
        note = "mixed-family GLLVM: one shared latent block across distinct response " *
               "families; `correlation` is the cross-distribution latent-scale " *
               "correlation. `families` is the per-trait family vector.", ci = ci,
        # gradient_max: NaN — MixedFamilyFit has no _family_ci adapter (no packed
        # objective to rebuild on this engine yet, same as its no native confint
        # engine above).
        gradient_max = NaN)
end

# Non-Gaussian assembler: REAL latent-scale derived quantities via the salvaged
# link-residual extractors (sigma_y_site/correlation/communality, ΛΛᵀ + diag(d_t)).
# Falls back to the shared block ΛΛᵀ ONLY when a family has no extractor on this
# engine (narrow MethodError catch — e.g. NB1); other errors propagate.
function _bridge_assemble_ng(fit, family, model, traits, units, p, K, Ydata, N;
                             alpha, dispersion, df, scores, ci = nothing, mask = nothing,
                             gradient_max = _bridge_gradient_max_family(fit, Ydata; N = N, mask = mask))
    Sigma, corr, comm, note = try
        S  = N === nothing ? sigma_y_site(fit, Ydata; mask = mask)  : sigma_y_site(fit, Ydata; N = N, mask = mask)
        C  = N === nothing ? correlation(fit, Ydata; mask = mask)   : correlation(fit, Ydata; N = N, mask = mask)
        cm = N === nothing ? communality(fit, Ydata; mask = mask)   : communality(fit, Ydata; N = N, mask = mask)
        (Matrix{Float64}(S), Matrix{Float64}(C), Vector{Float64}(cm), "")
    catch e
        e isa MethodError || rethrow()
        Λ = _bridge_loadings(fit)
        Σ = Λ * Λ'; Σ = (Σ + Σ') ./ 2
        (Σ, _bridge_corr_from_sigma(Σ), ones(Float64, p),
         "$(family) has no link-residual extractor on this engine yet; " *
         "Sigma/correlation use the shared block Lambda*Lambda' only (communality 1).")
    end
    return _bridge_assemble(fit, family, model, traits, units;
        alpha = alpha, dispersion = dispersion, sigma_eps = NaN,
        link = fill(_bridge_link_name(fit.link), p), Sigma = Sigma, corr = corr,
        comm = comm, scores = scores, df = df, loglik = fit.loglik,
        converged = fit.converged, iterations = fit.iterations, note = note, ci = ci,
        nobs = mask === nothing ? nothing : count(mask), gradient_max = gradient_max)
end

# Shared flat-NamedTuple builder. `ci` (when non-nothing) is a flat CI payload
# NamedTuple (from _bridge_compute_ci_*) MERGED onto the base contract; passing
# `ci = nothing` (the ci_method="none" default) returns the base tuple unchanged,
# so the existing contract is byte-identical for callers that request no CIs.
function _bridge_assemble(fit, family::AbstractString, model::AbstractString,
                          traits, units;
                          alpha, dispersion, sigma_eps, link, Sigma, corr, comm,
                          scores, df, loglik, converged, iterations, note,
                          loadings = nothing, families = nothing, ci = nothing,
                          nobs = nothing, gradient_max::Float64 = NaN)
    p = length(traits)
    n = length(units)
    L = loadings === nothing ? _bridge_loadings(fit) : loadings
    K = size(L, 2)
    ll = Float64(loglik)
    nobs_val = nobs === nothing ? p * n : Int(nobs)
    family_vec = families === nothing ? fill(family, p) : Vector{String}(families)
    length(family_vec) == p || throw(ArgumentError(
        "bridge_fit: per-trait families length $(length(family_vec)) must equal " *
        "the number of traits $p"))
    base = (
        family       = family,
        families     = family_vec,
        model        = model,
        d            = K,
        n_traits     = p,
        n_units      = n,
        trait_names  = traits,
        unit_names   = units,
        loadings     = Matrix{Float64}(L),
        alpha        = Vector{Float64}(alpha),
        dispersion   = Vector{Float64}(dispersion),
        sigma_eps    = Float64(sigma_eps),
        Sigma        = Matrix{Float64}(Sigma),
        correlation  = Matrix{Float64}(corr),
        communality  = Vector{Float64}(comm),
        scores       = Matrix{Float64}(scores),
        loglik       = ll,
        aic          = 2 * df - 2 * ll,
        # nobs_val (not `n`, the site count) — R's p·n cell-count convention
        # (docs/dev-log/decisions/2026-09-01-maintainer-decisions-round1.md
        # #1): stats::BIC.default(object) uses nobs(object), which counts
        # observed cells, not sites. Was `df * log(n) - 2 * ll` (disagreed
        # with the `nobs` field two lines below, which already used the
        # cell count).
        bic          = df * log(nobs_val) - 2 * ll,
        df           = df,
        nobs         = nobs_val,
        converged    = converged,
        iterations   = iterations,
        message      = converged ? "converged" : "not converged",
        link         = Vector{String}(link),
        note         = note,
        gradient_max = gradient_max,
    )
    return ci === nothing ? base : merge(base, ci)
end

# --- structured-covariance sources route (Gaussian, v1) ----------------------
# Design: docs/dev-log/julia-bridge-structured-design-julia-side.md. Single
# source, default trait-intercept mean, no CI/mask/X composition — everything
# unsupported rejects loudly. The native target is `fit_gaussian_sources`.

function _bridge_source_from_dict(spec, n::Integer, index::Integer)
    spec isa AbstractDict || throw(ArgumentError(
        "bridge_fit: sources[$index] must be a Dict-like source spec"))
    name = Symbol(String(_bridge_get(spec, "name", "source$index")))
    C = _bridge_get(spec, "covariance", nothing)
    C isa AbstractMatrix || throw(ArgumentError(
        "bridge_fit: sources[$index] needs a square covariance matrix"))
    Cm = Matrix{Float64}(C)
    mode_raw = lowercase(strip(String(_bridge_get(spec, "mode", "latent"))))
    mode_raw in ("latent", "indep", "dep") || throw(ArgumentError(
        "bridge_fit: sources[$index] mode must be latent, indep, or dep; " *
        "got \"$mode_raw\""))
    mode = Symbol(mode_raw)
    rank_raw = _bridge_get(spec, "rank", nothing)
    rank = rank_raw === nothing ? nothing : Int(rank_raw)
    mode !== :latent && rank !== nothing && throw(ArgumentError(
        "bridge_fit: sources[$index] rank is only valid for mode = latent"))
    uniq = Bool(_bridge_get(spec, "unique", false))
    comm = Bool(_bridge_get(spec, "common", false))
    proj = _bridge_get(spec, "projection", nothing)
    groups = _bridge_get(spec, "groups", nothing)
    if proj !== nothing
        P = Matrix{Float64}(proj)
        size(P, 1) == n || throw(ArgumentError(
            "bridge_fit: sources[$index] projection needs $n rows (one per unit)"))
        return mode === :latent ?
            SourceCovariance(Cm, P; name = name, mode = mode,
                rank = rank === nothing ? 1 : rank, unique = uniq, common = comm) :
            SourceCovariance(Cm, P; name = name, mode = mode,
                unique = uniq, common = comm)
    end
    groups === nothing && throw(ArgumentError(
        "bridge_fit: sources[$index] needs groups or projection"))
    g = [Int(x) for x in collect(groups)]
    length(g) == n || throw(ArgumentError(
        "bridge_fit: sources[$index] groups length $(length(g)) must equal " *
        "the unit count $n"))
    return mode === :latent ?
        SourceCovariance(Cm; groups = g, name = name, mode = mode,
            rank = rank === nothing ? 1 : rank, unique = uniq, common = comm) :
        SourceCovariance(Cm; groups = g, name = name, mode = mode,
            unique = uniq, common = comm)
end

function _bridge_fit_sources(y, family, K::Integer, sources;
        X, X_lv, mask, N, trait_names, unit_names, options)
    fam = _bridge_family_key(String(family))
    fam == "gaussian" || throw(ArgumentError(
        "bridge_fit: sources are supported for the gaussian family only"))
    X === nothing || throw(ArgumentError(
        "bridge_fit: sources and X are mutually exclusive in this slice"))
    X_lv === nothing || throw(ArgumentError(
        "bridge_fit: sources and X_lv are mutually exclusive"))
    mask === nothing || throw(ArgumentError(
        "bridge_fit: sources do not support response masks"))
    N === nothing || throw(ArgumentError(
        "bridge_fit: sources do not take a trials matrix"))
    ci_method = lowercase(String(_bridge_get(options, "ci_method", "none")))
    ci_method == "none" || throw(ArgumentError(
        "bridge_fit: sources support ci_method = \"none\" only " *
        "(fit_gaussian_sources has no bridge CI engine yet)"))
    specs = collect(sources)
    isempty(specs) && throw(ArgumentError("bridge_fit: sources is empty"))
    length(specs) == 1 || throw(ArgumentError(
        "bridge_fit: exactly one source is supported in this slice; " *
        "multi-source transport is a documented follow-up"))
    Yf = Matrix{Float64}(y)
    p, n = size(Yf)
    native = [_bridge_source_from_dict(specs[1], n, 1)]
    g_tol = Float64(_bridge_get(options, "g_tol", 1e-6))
    iterations = Int(_bridge_get(options, "iterations", 500))
    fit = fit_gaussian_sources(Yf; sources = native, g_tol = g_tol,
        iterations = iterations)
    src = only(fit.sources)
    B = Matrix{Float64}(only(fit.trait_covariances))
    Sigma = B + fit.sigma_eps^2 * Matrix{Float64}(I, p, p)
    dstd = sqrt.(max.(diag(Sigma), 0.0))
    corr = Sigma ./ (dstd * dstd')
    comm = [Sigma[i, i] > 0 ? B[i, i] / Sigma[i, i] : NaN for i in 1:p]
    q = p   # default trait-intercept mean design
    L = if src.mode === :latent
        unpack_lambda(fit.parameters[(q + 1):(q + rr_theta_len(p, src.rank))],
            p, src.rank)
    elseif src.mode === :dep
        unpack_lambda(fit.parameters[(q + 1):(q + rr_theta_len(p, p))], p, p)
    else
        zeros(p, 0)
    end
    traits = _bridge_names(trait_names, p, "trait")
    units = _bridge_names(unit_names, n, "unit")
    base = _bridge_assemble(fit, "gaussian", "gaussian_sources", traits, units;
        alpha = fit.beta, dispersion = fill(NaN, p), sigma_eps = fit.sigma_eps,
        link = fill("identity", p), Sigma = Sigma, corr = corr, comm = comm,
        scores = zeros(0, n), df = length(fit.parameters),
        loglik = fit.loglik, converged = fit.converged,
        iterations = fit.iterations, note = "", loadings = L,
        gradient_max = Float64(fit.gradient_norm))
    return merge(base, (
        source_names = [String(src.name)],
        source_modes = [String(src.mode)],
        source_common = [src.common],
        source_unique = [src.unique],
        source_rank = [src.mode === :latent ? src.rank : 0],
        source_covariance = B,
    ))
end

# ---------------------------------------------------------------------------
# S3a — Julia-side phylogenetic precision payload.
# Flat primitives only (JuliaCall). `species_aug_id` is 0-indexed on the
# wire (`fit-multi.R:4638`); `i,j` are 1-based Matrix/findnz triplets.
# This does not lift the R `phylo_rr` gate. S3-FIT adds a thin
# `bridge_fit(; phylo=payload)` consume hook (diagnostic only).
# ---------------------------------------------------------------------------
const PHYLO_PRECISION_PAYLOAD_KEYS = (
    :i, :j, :x, :n_aug, :n_leaves, :species_aug_id,
    :node_labels, :scale, :log_det,
)
const PHYLO_PRECISION_LOGDET_TOL = 1e-8

"""
    phylo_precision_payload(pp::PrecisionPhy)

Pack a `PrecisionPhy` into a JuliaCall-flat NamedTuple for the phylo
transport wire. Field meanings match frozen gllvmTMB 0.7.0:
`Ainv_phy_rr` triplets (`i`, `j`, `x`), `n_aug_phy`, tip count,
0-indexed unique tip-to-node `species_aug_id`, node labels, applied `scale`,
and `log_det = logdet(Q)`. The R adapter must negate R's covariance
`log_det_A_phy_rr` to obtain this precision checksum. R's observation-level
`species_aug_id` may repeat; this payload instead carries one entry per tip,
with observation-to-tip indexing transported separately by the model adapter.
"""
function phylo_precision_payload(pp::PrecisionPhy)
    I, J, V = findnz(pp.Q)
    return (
        i = collect(Int, I),
        j = collect(Int, J),
        x = collect(Float64, V),
        n_aug = Int(pp.n_aug),
        n_leaves = Int(pp.n_leaves),
        species_aug_id = collect(Int, pp.species_aug_id) .- 1,
        node_labels = collect(String, pp.node_labels),
        scale = Float64(pp.scale),
        log_det = Float64(pp.log_det),
    )
end

function _phylo_payload_as_nt(payload)
    payload isa NamedTuple && return payload
    if payload isa AbstractDict
        kwargs = Dict{Symbol,Any}()
        for key in PHYLO_PRECISION_PAYLOAD_KEYS
            raw = haskey(payload, key) ? payload[key] :
                  haskey(payload, String(key)) ? payload[String(key)] :
                  throw(ArgumentError("GJL-GATE-PHYLO-PAYLOAD-DIM: missing field $(key)"))
            kwargs[key] = raw
        end
        return (; kwargs...)
    end
    throw(ArgumentError("GJL-GATE-PHYLO-PAYLOAD-DIM: payload must be a NamedTuple or Dict"))
end

function _phylo_payload_gate(tag::AbstractString, msg::AbstractString)
    throw(ArgumentError("$(tag): $(msg)"))
end

"""
    admit_phylo_precision_payload(payload) :: PrecisionPhy

Validate a Julia-side precision payload and reconstruct `PrecisionPhy`.
Rejects malformed dimensions (`n_aug < n_leaves`), indices, tip maps,
labels, non-finite values, and a shipped log-determinant that disagrees
with an independent checksum by more than `1e-8`. Tree payloads keep
`n_aug = 2p − 2`; animal / sparse `Ainv` payloads may be tip-only
(`n_aug = n_leaves`) or ancestor-augmented (`n_aug > n_leaves`).
"""
admit_phylo_precision_payload(; kwargs...) = admit_phylo_precision_payload((; kwargs...))

function admit_phylo_precision_payload(payload)
    nt = _phylo_payload_as_nt(payload)
    for key in PHYLO_PRECISION_PAYLOAD_KEYS
        haskey(nt, key) ||
            _phylo_payload_gate("GJL-GATE-PHYLO-PAYLOAD-DIM", "missing field $(key)")
    end

    n_aug = Int(nt.n_aug)
    n_leaves = Int(nt.n_leaves)
    # Tree payloads use n_aug = 2p − 2 (root dropped). Animal / sparse Ainv
    # payloads are tip-only (n_aug = n_leaves) or keep unphenotyped
    # ancestors (n_aug > n_leaves). Subsetting a precision would condition
    # on the dropped nodes, not marginalise them.
    n_leaves > 0 ||
        _phylo_payload_gate("GJL-GATE-PHYLO-PAYLOAD-DIM", "n_leaves must be positive")
    n_aug >= n_leaves ||
        _phylo_payload_gate("GJL-GATE-PHYLO-PAYLOAD-DIM",
            "n_aug ($(n_aug)) must be >= n_leaves ($(n_leaves))")

    I = collect(Int, nt.i)
    J = collect(Int, nt.j)
    V = collect(Float64, nt.x)
    length(I) == length(J) == length(V) ||
        _phylo_payload_gate("GJL-GATE-PHYLO-PAYLOAD-DIM",
            "sparse triplets i, j, x must have equal length")

    (all(>=(1), I) && all(<=(n_aug), I) && all(>=(1), J) && all(<=(n_aug), J)) ||
        _phylo_payload_gate("GJL-GATE-PHYLO-PAYLOAD-INDEX",
            "sparse triplet indices must lie in 1:n_aug")

    tip0 = collect(Int, nt.species_aug_id)
    length(tip0) == n_leaves ||
        _phylo_payload_gate("GJL-GATE-PHYLO-PAYLOAD-TIPMAP",
            "species_aug_id length ($(length(tip0))) must equal n_leaves ($(n_leaves))")
    (all(>=(0), tip0) && all(<(n_aug), tip0) && length(unique(tip0)) == n_leaves) ||
        _phylo_payload_gate("GJL-GATE-PHYLO-PAYLOAD-TIPMAP",
            "species_aug_id must be a unique 0-based map into 0:n_aug-1")

    labels = collect(String, nt.node_labels)
    length(labels) == n_aug ||
        _phylo_payload_gate("GJL-GATE-PHYLO-PAYLOAD-LABEL",
            "node_labels length ($(length(labels))) must equal n_aug ($(n_aug))")
    all(!isempty, labels) ||
        _phylo_payload_gate("GJL-GATE-PHYLO-PAYLOAD-LABEL",
            "node_labels must be non-empty strings")

    scale = Float64(nt.scale)
    log_det = Float64(nt.log_det)
    (all(isfinite, V) && isfinite(scale) && isfinite(log_det)) ||
        _phylo_payload_gate("GJL-GATE-PHYLO-PAYLOAD-NONFINITE",
            "precision values, scale, and log_det must be finite")

    tip1 = tip0 .+ 1
    pp = PrecisionPhy(I, J, V, n_aug, n_leaves, labels, log_det, scale, tip1)
    recomputed, shipped, abs_diff = precision_logdet_check(pp)
    abs_diff <= PHYLO_PRECISION_LOGDET_TOL ||
        _phylo_payload_gate("GJL-GATE-PHYLO-PAYLOAD-LOGDET",
            "shipped log_det ($(shipped)) disagrees with recomputed ($(recomputed)) by $(abs_diff)")
    return pp
end

# S3-FIT — thin diagnostic consume hook. Admits a PrecisionPhy payload (or
# accepts an already-admitted PrecisionPhy) and fits the univariate
# phylogenetic Gaussian model. Does not lift the R phylo_rr gate.
function _bridge_fit_phylo_precision(y, phylo; family = nothing, options = nothing)
    if family !== nothing && _bridge_family_key(String(family)) != "gaussian"
        throw(ArgumentError(
            "bridge_fit: phylo precision consume hook is Gaussian-only (diagnostic)"))
    end
    pp = phylo isa PrecisionPhy ? phylo : admit_phylo_precision_payload(phylo)
    yv = if y isa AbstractMatrix
        size(y, 2) == 1 || throw(ArgumentError(
            "bridge_fit: phylo precision consume hook takes a length-p trait vector " *
            "(or a p×1 matrix); got size $(size(y))"))
        vec(y)
    else
        collect(float.(y))
    end
    fit = fit_phylo_gaussian(pp, yv)
    return (
        μ = fit.μ,
        sigma2_phy = fit.σ²_phy,
        sigma2_eps = fit.σ²_eps,
        negll = fit.negll,
        loglik = -fit.negll,
        converged = fit.converged,
        iterations = fit.iterations,
        diagnostic_only = true,
    )
end
