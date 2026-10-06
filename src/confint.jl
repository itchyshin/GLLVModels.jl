# Wald confidence intervals via the observed information matrix.
#
# Strategy:
#   - Reconstruct the negative log-likelihood used during fitting by
#     calling gaussian_nll_packed on the legacy-layout θ_packed vector
#     stored on the GllvmFit (fit.pars.θ_packed).
#   - Observed information matrix H = ForwardDiff.hessian(nll, θ̂).
#   - Asymptotic covariance Σ = inv(H); SEs = sqrt.(diag(Σ)).
#   - Wald CI: θ̂ ± z * SE on the working scale of each parameter, where
#     z = quantile(Normal(), 0.5 + level/2).
#
# Working-scale CIs: σ_eps, σ_B, σ_W are stored as logs in the packed
# vector, so their CI bounds are on the *raw* (positive) scale via
# exp(log_θ ± z * SE_log). σ_phy uses an identity (signed) link — its
# CI is the plain Wald θ̂ ± z * SE. β and Λ entries are linear in the
# packed vector and reported as-is. This matches glmmTMB/gllvmTMB's
# default behaviour for SD-style parameters (reported on the raw scale,
# Wald-on-log-then-exponentiated) while treating σ_phy as the signed
# loading-like quantity it actually is in the marginal likelihood.
#
# Non-PD Hessian handling:
#   - If ForwardDiff.hessian errors, return NaN bounds with pd_hessian=false.
#   - If the Hessian is finite but inversion fails or any diagonal is
#     non-positive, mark pd_hessian=false and return NaN bounds for those
#     entries.

using Distributions: Normal, quantile

# Build the lambda part of the term-name list in pack_lambda order.
# Mirrors `pack_lambda` / `unpack_lambda` in src/packing.jl: diagonals
# (k = 1..K) first, then strict-lower entries column-by-column.
function _confint_lambda_term_names(prefix::String, p::Integer, K::Integer)
    out = String[]
    for k in 1:K
        push!(out, "$(prefix)[$k,$k]")
    end
    for k in 1:K
        for i in (k + 1):p
            push!(out, "$(prefix)[$i,$k]")
        end
    end
    return out
end

# Build the canonical term-name vector matching the legacy θ_packed layout:
#
#     [β[1..q]; sigma_eps;
#      sigma_B[1..p]; sigma_W[1..p]      (if has_diag)
#      Lambda_B[i,k]                      (pack_lambda order)
#      Lambda_W[i,k]                      (if K_W > 0)
#      sigma_phy[1..p]                    (if has_phy_unique)
#      Lambda_phy[i,k]                    (if K_phy > 0)]
#
# Returns (terms, kinds) where each kind is :linear (β, Λ) or :log_sd
# (the SD-style parameters). Kind drives the raw-vs-working CI transform.
function _confint_all_term_names(fit::GllvmFit)
    model = fit.model
    p     = model.p
    K_B   = model.K
    K_W   = model.K_W
    has_diag = model.has_diag
    K_phy    = model.K_phy
    has_phy_unique = model.has_phy_unique
    q_full = fit.pars.β === nothing ? 0 : length(fit.pars.β)
    β_fixed = _pars_fixed_mask(fit.pars, :β_fixed, q_full)
    β_free = _free_coeff_indices(β_fixed)

    terms = String[]
    kinds = Symbol[]

    for j in β_free
        push!(terms, "beta[$j]")
        push!(kinds, :linear)
    end

    push!(terms, "sigma_eps")
    push!(kinds, :log_sd)

    if has_diag
        for t in 1:p
            push!(terms, "sigma_B[$t]")
            push!(kinds, :log_sd)
        end
        for t in 1:p
            push!(terms, "sigma_W[$t]")
            push!(kinds, :log_sd)
        end
    end

    for nm in _confint_lambda_term_names("Lambda_B", p, K_B)
        push!(terms, nm)
        push!(kinds, :linear)
    end

    if K_W > 0
        for nm in _confint_lambda_term_names("Lambda_W", p, K_W)
            push!(terms, nm)
            push!(kinds, :linear)
        end
    end

    if has_phy_unique
        # σ_phy uses an identity (signed) link — Wald CI is plain linear.
        for t in 1:p
            push!(terms, "sigma_phy[$t]")
            push!(kinds, :linear)
        end
    end

    if K_phy > 0
        for nm in _confint_lambda_term_names("Lambda_phy", p, K_phy)
            push!(terms, nm)
            push!(kinds, :linear)
        end
    end

    return terms, kinds
end

# Resolve a single selector string against the term list.
#   "sigma_eps"           -> matches the single sigma_eps entry
#   "Lambda"              -> matches all Lambda_* entries (B, W, phy)
#   "Lambda_B"            -> matches all Lambda_B entries
#   "Lambda:1,1"          -> matches Lambda_B[1,1] (B is the default tier)
#   "Lambda_W:2,1"        -> matches Lambda_W[2,1]
#   "Lambda_B[3,1]"       -> exact match
#   "sigma_B"             -> all sigma_B[*] entries
#   "sigma_B[2]"          -> exact match
function _confint_select_indices_one(selector::String, terms::Vector{String})
    idx = findfirst(==(selector), terms)
    if !isnothing(idx)
        return [idx]
    end

    if startswith(selector, "Lambda:")
        return _confint_select_indices_one(
            "Lambda_B[" * selector[length("Lambda:") + 1:end] * "]", terms)
    end
    for prefix in ("Lambda_B:", "Lambda_W:", "Lambda_phy:")
        if startswith(selector, prefix)
            base = prefix[1:end-1]
            return _confint_select_indices_one(
                "$(base)[" * selector[length(prefix) + 1:end] * "]", terms)
        end
    end

    if selector == "Lambda"
        return findall(t -> startswith(t, "Lambda_"), terms)
    end
    if selector in ("Lambda_B", "Lambda_W", "Lambda_phy")
        return findall(t -> startswith(t, "$(selector)["), terms)
    end
    if selector in ("sigma_B", "sigma_W", "sigma_phy")
        return findall(t -> startswith(t, "$(selector)["), terms)
    end
    if selector == "beta"
        return findall(t -> startswith(t, "beta["), terms)
    end

    throw(ArgumentError(
        "Could not resolve parm selector \"$selector\" against term names. " *
        "Use one of the names returned by confint(fit).term."))
end

function _confint_select_indices(parm, terms::Vector{String})
    parm === nothing && return collect(1:length(terms))
    if parm isa Symbol
        parm = String(parm)
    end
    if parm isa AbstractString
        return _confint_select_indices_one(String(parm), terms)
    elseif parm isa AbstractVector
        idxs = Int[]
        for p in parm
            append!(idxs, _confint_select_indices_one(String(p), terms))
        end
        return idxs
    else
        throw(ArgumentError(
            "`parm` must be nothing, String, Symbol, or Vector{String}; got $(typeof(parm))"))
    end
end

# Reconstruct the NLL closure that gaussian_nll_packed (NamedTuple spec
# signature) requires to evaluate at an arbitrary θ. The θ stored on
# fit.pars.θ_packed is in the legacy layout
#   [β; log_σ_eps; (log_σ_B; log_σ_W if has_diag);
#    θ_rr_B; θ_rr_W; (σ_phy if has_phy_unique, identity link); θ_rr_phy]
# which is exactly what gaussian_nll_packed expects.
function _confint_reconstruct_nll(fit::GllvmFit, y::AbstractMatrix,
                                  X::Union{Nothing, AbstractArray{<:Real, 3}},
                                  Σ_phy::Union{Nothing, AbstractMatrix})
    _has_gaussian_record(fit) && return _gaussian_record_ci(fit,y;X=X,Σ_phy=Σ_phy).nll
    X = _mean_X(fit, X, size(y, 2))
    model = fit.model
    q_full = fit.pars.β === nothing ? 0 : length(fit.pars.β)
    β_fixed = _pars_fixed_mask(fit.pars, :β_fixed, q_full)
    β_free = _free_coeff_indices(β_fixed)
    X_free = X === nothing || isempty(β_free) ? nothing : Array{Float64,3}(X[:, :, β_free])
    spec = (q = length(β_free), p = model.p, K_B = model.K, K_W = model.K_W,
            has_diag = model.has_diag, K_phy = model.K_phy,
            has_phy_unique = model.has_phy_unique)
    return θ -> gaussian_nll_packed(θ, y; spec = spec, X = X_free, Σ_phy = Σ_phy)
end

# Observed-information covariance for the legacy (no retained integration
# record) Gaussian GllvmFit path, shared by `confint(fit::GllvmFit; ...)` and
# `vcov(fit::GllvmFit, Y)`. Returns (θ̂, terms, kinds, Σ, se_all, pd) over the
# full θ_packed layout. Σ = inv((H + Hᵀ)/2) with H the ForwardDiff Hessian of
# the marginal NLL at θ̂ — no regularisation. On a fit with `lambda_constraint`
# pins the inverse is taken over the free parameters only, and the pinned
# loadings get SE 0 and zero rows and columns in Σ (R's `cov.fixed`, #794).
# Non-PD convention (unchanged from the SE code): if the Hessian errors, is non-finite, or cannot be inverted,
# every SE is NaN and Σ is all-NaN; if it inverts but a diagonal entry is
# non-finite or non-positive, that SE is NaN and pd = false. For Σ, the rows
# and columns of such entries are set to NaN (a covariance with an invalid
# variance is not reported), off-diagonals are symmetrised (inv() is
# symmetric only up to rounding), and the diagonal is set to se_all.^2 so that
# diag(vcov) is bit-identical to the squared Wald SEs that `confint` reports.
function _confint_gaussian_wald_covariance(fit::GllvmFit, y,
                                           X::Union{Nothing, AbstractArray{<:Real, 3}},
                                           Σ_phy::Union{Nothing, AbstractMatrix})
    _has_lv_predictor(fit) && throw(ArgumentError(
        "confint for fit_gaussian_gllvm(...; X_lv=...) is not admitted in the C1 predictor-informed latent-score path; use extract_lv_effects for point estimates"))
    y === nothing && throw(ArgumentError(
        "confint requires the data matrix `y` (the same matrix passed to fit_gaussian_gllvm)"))
    X = _mean_X(fit, X, size(y, 2))

    θ̂ = fit.pars.θ_packed
    n_par = length(θ̂)

    terms, kinds = _confint_all_term_names(fit)
    length(terms) == n_par || error(
        "Internal: term-name vector length ($(length(terms))) does not match θ_packed length ($n_par). " *
        "This is a packing layout bug.")

    nll = _confint_reconstruct_nll(fit, y, X, Σ_phy)

    H = nothing
    pd = true
    try
        H = ForwardDiff.hessian(nll, θ̂)
    catch
        H = nothing
        pd = false
    end

    # Loadings pinned by `lambda_constraint` are not parameters (#794): invert the
    # information of the free parameters only, as R's `sd_report$cov.fixed`, and
    # report each pinned entry with variance and covariances 0.
    pins = _lambda_constraint_pinned_theta_indices(fit)

    se_all = fill(NaN, n_par)
    V = fill(NaN, n_par, n_par)
    if H !== nothing && all(isfinite, H)
        Σ = nothing
        try
            Hsym = (H .+ H') ./ 2
            Σ = isempty(pins) ? inv(Hsym) : _inv_free_block(Hsym, pins)
        catch
            Σ = nothing
            pd = false
        end

        if Σ !== nothing
            diagΣ = diag(Σ)
            for i in 1:n_par
                if !isempty(pins) && insorted(i, pins)
                    se_all[i] = 0.0
                    continue
                end
                v = diagΣ[i]
                if isfinite(v) && v > 0
                    se_all[i] = sqrt(v)
                else
                    pd = false
                end
            end
            ok = findall(isfinite, se_all)
            V[ok, ok] .= (Σ[ok, ok] .+ Σ[ok, ok]') ./ 2   # inv() is symmetric only to rounding
            for i in ok
                V[i, i] = se_all[i]^2
            end
        end
    else
        pd = false
    end
    return θ̂, terms, kinds, V, se_all, pd
end

"""
    confint(fit::GllvmFit; level=0.95, parm=nothing, method=:wald,
            y=nothing, X=nothing, Σ_phy=nothing, kwargs...) -> NamedTuple
    confint(fit::GllvmFit, y; level=0.95, parm=nothing, method=:wald,
            X=nothing, Σ_phy=nothing, kwargs...) -> NamedTuple

Confidence intervals for the parameters of a fitted Gaussian GLLVM (Wald by
default; profile-likelihood and parametric-bootstrap intervals with `method`).
With `method = :wald` (the default) the call returns a NamedTuple with fields:

  - `term::Vector{String}`     — parameter names
  - `estimate::Vector{Float64}` — point estimates (raw scale for SDs)
  - `lower::Vector{Float64}`    — lower CI bound at `level`
  - `upper::Vector{Float64}`    — upper CI bound at `level`
  - `se::Vector{Float64}`       — standard errors (working scale)
  - `pd_hessian::Bool`          — whether the observed information matrix
                                  was positive definite at the MLE

`level` is the nominal coverage (default 0.95 → two-sided 95% CI).

`parm` selects a subset of parameters by name. Acceptable forms:
  - `nothing` (default) — all parameters
  - `"sigma_eps"` — single name
  - `"Lambda"` — all Λ entries across all tiers (B, W, phy)
  - `"Lambda:1,1"` — shorthand for `"Lambda_B[1,1]"`
  - `["sigma_eps", "Lambda:1,1"]` — mixed list

Working-scale convention: σ_eps, σ_B, σ_W are parameterised on the log
scale internally. The CI bounds returned for those entries are on the
*raw* (positive) scale via `exp(log_θ ± z * SE_log)`. σ_phy uses an
identity (signed) link — its Wald CI is the plain `θ̂ ± z * SE`. β and
Λ entries are reported on their native (linear) scale.

`σ_phy` is a signed parameter (#136), so this interval can cross zero. When
`K_phy = 0`, the sign of each group of rows that `Σ_phy` links can flip on its
own (for a tree-derived `Σ_phy`, each daughter clade of the root); when `K_phy ≥ 1`, it is not identified separately from
`Λ_phy` and a per-entry interval is not interpretable. It is not gllvmTMB's `phylo_unique` scale; see the gllvmTMB
parity page.

The Hessian is computed via ForwardDiff at the fitted parameter vector
stored on `fit.pars.θ_packed`. The function needs the original data
matrix `y` (and optionally `X`, `Σ_phy`) to reconstruct the NLL closure.
If the Hessian is not positive definite, this function returns NaN
bounds for the affected entries with `pd_hessian = false` (matching the
R glmmTMB / gllvmTMB convention).

On a fit made with `fit_gaussian_gllvm(y; K, lambda_constraint = M)`, the
pinned loadings are not parameters: their rows and columns are removed from
the observed information before it is inverted (R's `sd_report\$cov.fixed`),
each pinned `Lambda_B[i,k]` is reported with `se = 0` and `lower = upper =
estimate`, and `pd_hessian` refers to the free parameters. `vcov` gives the
pinned entries zero rows and columns.

When PERF lands with reverse-mode AD, the integration agent can swap the
ForwardDiff.hessian call for the faster path; the public API stays
stable.

# Choosing the method

`method` is `:wald` (the default), `:profile` or `:bootstrap`, and the call
either runs that method or throws an `ArgumentError` that names the methods
available for the fit and `parm`. It never returns a Wald interval for a profile
or bootstrap request, and a keyword the route does not use is refused by name
rather than ignored.

| `method`     | route                                           | extra fields |
|--------------|-------------------------------------------------|--------------|
| `:wald`      | observed-information Wald (above)               | `se`, `pd_hessian` |
| `:profile`   | [`profile_ci`](@ref) for each selected term     | `status` (`:profile`, `:partial` or `:failed`), `method` |
| `:bootstrap` | [`bootstrap_ci`](@ref) for the selected terms   | `n_converged`, `replicates`, `method` |

`:profile` and `:bootstrap` need `y`. Their controls are `n_boot` (default 200)
and `seed` (default 0) for the bootstrap, and `profile_iterations`,
`profile_g_tol`, `profile_max_expand` and `profile_max_bisect` for the profile.
A fit that retains its data record (`aghq` or a masked or offset fit) also takes
`mask`, `offset`, `objective` and `parallel`. A control for another method is
accepted and ignored, as in the non-Gaussian `confint`. `mask`, `offset`, `N`,
`objective` (only `:fit` or `:laplace`) and `parallel` (only `false`) are
accepted on a fit without a record only at their "not used" value. A fit made
with `lambda_constraint` pins takes `:wald` only, because the profile and
bootstrap refits would drop the pins.

# Derived quantities

`parm` also names derived quantities, reached through the same `method` argument
(`n_boot`, `seed`, `profile_max_expand`, `profile_max_bisect` and
`penalty_weight` are the controls):

| `parm`                        | quantity                                      | `:wald` (transform)                    | `:profile`               | `:bootstrap` |
|-------------------------------|-----------------------------------------------|----------------------------------------|--------------------------|--------------|
| `"communality[t]"`            | `communality(fit)[t]`                         | [`communality_wald_ci`](@ref) (logit)  | withdrawn (refused)      | `bootstrap_ci_derived` |
| `"icc[t]"` or `"repeatability[t]"` | [`extract_ICC_site`](@ref)`(fit)[t]`  | [`icc_wald_ci`](@ref) (logit)          | `profile_ci_derived`     | same |
| `"rho[i,j]"` or `"correlation[i,j]"` | `correlation(fit)[i,j]`                | [`correlation_wald_ci`](@ref) (Fisher-z) | withdrawn (refused) | same |
| `"proportion:<component>[t]"` | `proportions(fit; component)[t]`              | [`icc_wald_ci`](@ref) (logit)          | withdrawn (refused)      | same |
| `"phylo_signal[t]"`           | `phylo_signal(fit)[t]`                        | [`phylo_signal_wald_ci`](@ref) (logit) | `profile_ci_phylo_signal` | same |

Leave out `[t]` for every trait (`[i,j]` for every pair); separate several traits
or pairs with `;`, as in `"rho[1,2;1,3]"`; or pass a vector of these names. The
components are `shared`, `unique_B`, `unique_W`, `unique_Wd` and `residual` (see
[`GLLVModels.proportions`](@ref)). gllvmTMB's tier spellings (`communality:unit:t`,
`rho:unit:i,j`, `proportion:shared_unit:t`) are refused: they name gllvmTMB's aligned
estimands, which differ from these quantities.

The result carries `term` (for example `"communality[2]"`), `estimate`, `lower`
and `upper` per quantity, and `method`. `:wald` adds `se_transformed`,
`transform`, `pd_hessian` and `status` (`:transformed_wald` or `:failed`; a
quantity on its boundary has no interval and returns `NaN` bounds with
`:failed`); `:profile` adds `status` and `boundary` (the profile is clamped to
the quantity's natural range); `:bootstrap` adds `n_converged`, `n_valid` and
`replicates` (one column per quantity, one bootstrap per quantity). The default
method is `:wald`.

`method = :profile` is refused for communality, rho and proportion, as in
gllvmTMB 9539352f6, which withdrew those profile intervals (condition class
`gllvmTMB_nonlinear_profile_withdrawn`) pending an exact constraint solver: the
ArgumentError says the profile is withdrawn and points to `:wald` or `:bootstrap`.
The penalty-based `profile_ci_derived` route that served them stays available as
an exploratory internal function, and it still serves `icc` here.

```julia
fit = fit_gaussian_gllvm(y; K = 2)
confint(fit, y)                                             # Wald, every term
confint(fit, y; parm = "sigma_eps", method = :profile)      # profile-likelihood
confint(fit, y; parm = "Lambda", method = :bootstrap, n_boot = 500)
confint(fit, y; parm = "communality")                       # transformed Wald, every trait
confint(fit, y; parm = "rho[1,2]", method = :bootstrap, n_boot = 500)
```
"""
function confint(fit::GllvmFit;
                 level::Real = 0.95,
                 parm::Union{Nothing, AbstractString, Symbol, AbstractVector} = nothing,
                 y::Union{Nothing, AbstractMatrix} = nothing,
                 X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                 Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                 method = :wald,
                 kwargs...)
    return _confint_gaussian(fit, y, level, parm, X, Σ_phy, method, kwargs)
end

function confint(fit::GllvmFit, y::AbstractMatrix;
                 level::Real = 0.95,
                 parm::Union{Nothing, AbstractString, Symbol, AbstractVector} = nothing,
                 X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                 Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                 method = :wald,
                 kwargs...)
    return _confint_gaussian(fit, y, level, parm, X, Σ_phy, method, kwargs)
end

# ---------------------------------------------------------------------------
# Method dispatch for Gaussian fits.
#
# A request for `method = m` either runs method `m` or throws an ArgumentError
# that names the methods the fit and parameter can use. A keyword the route does
# not understand is refused by name too. Nothing is dropped, and no Wald interval
# is returned for a profile or bootstrap request.
# ---------------------------------------------------------------------------

const _CONFINT_METHODS = (:wald, :profile, :bootstrap)

# Keywords each route understands (beyond level, parm, y, X, Σ_phy, method).
const _CONFINT_RECORD_KWARGS = (:mask, :offset, :objective, :n_boot, :seed, :parallel,
                                :profile_iterations, :profile_g_tol,
                                :profile_max_expand, :profile_max_bisect)
const _CONFINT_PLAIN_KWARGS = (:n_boot, :seed, :profile_iterations, :profile_g_tol,
                               :profile_max_expand, :profile_max_bisect)
const _CONFINT_DERIVED_KWARGS = (:n_boot, :seed, :profile_max_expand, :profile_max_bisect,
                                 :penalty_weight)
# Keywords that generic callers (`select_lv`, `coef_table`, ...) pass with their
# "not used" value. They are accepted at that value and refused at any other.
const _CONFINT_INERT_KWARGS = (:mask, :offset, :objective, :parallel, :N)

_confint_inert_ok(k::Symbol, v) = k === :objective ? v in (:fit, :laplace) :
                                  k === :parallel ? v === false : v === nothing

_confint_fit_label(fit::GllvmFit) =
    _has_gaussian_record(fit) ? "Gaussian GllvmFit (retained data record)" : "Gaussian GllvmFit"

_confint_parm_label(parm) = parm === nothing ? "all terms" :
                            parm isa AbstractVector ? join(string.(parm), ", ") : string(parm)

_confint_list(syms) = join((":" * string(s) for s in syms), ", ")

function _confint_check_method(method::Symbol, label::AbstractString, parm;
                               available = _CONFINT_METHODS)
    method in available && return nothing
    throw(ArgumentError(
        "confint: method = :$method is not available for $label / parm " *
        "$(_confint_parm_label(parm)); available: $(_confint_list(available))"))
end

function _confint_check_kwargs(kw::NamedTuple, accepted::Tuple, inert::Tuple,
                               label::AbstractString, method::Symbol)
    for (k, v) in pairs(kw)
        k in accepted && continue
        if k in inert
            _confint_inert_ok(k, v) && continue
            shown = v isa AbstractArray ? summary(v) : repr(v)
            throw(ArgumentError(
                "confint: keyword $k = $shown is not supported for $label " *
                "(method = :$method); accepted keywords: " *
                join(string.(accepted), ", ")))
        end
        throw(ArgumentError(
            "confint: unknown keyword $k for $label (method = :$method); " *
            "accepted keywords: level, parm, y, X, Σ_phy, " * join(string.(accepted), ", ")))
    end
    return nothing
end

# `bootstrap_ci` and `profile_ci` refit without the loading pins, so on a fit made
# with `lambda_constraint` they would answer for a different model. A constraint
# whose only numeric entries are structural zeros pins nothing in `θ_packed` and
# leaves the ordinary model, so it is not refused.
function _confint_check_unpinned(fit::GllvmFit, method::Symbol, label::AbstractString, parm)
    isempty(_lambda_constraint_pinned_theta_indices(fit)) && return nothing
    throw(ArgumentError(
        "confint: method = :$method is not available for $label with lambda_constraint " *
        "pins / parm $(_confint_parm_label(parm)); available: :wald " *
        "(profile and bootstrap refits would drop the pins and fit a different model)"))
end

# The same refusal for the direct entry points (`profile_ci`, `tmbprofile_wrapper`,
# `bootstrap_ci`, `bootstrap_ci_derived`, `profile_ci_derived`), which are public and
# would otherwise refit a pinned fit without its pins (refs #794).
function _check_unpinned_refit(fit::GllvmFit, fname::AbstractString)
    isempty(_lambda_constraint_pinned_theta_indices(fit)) && return nothing
    throw(ArgumentError(
        "$fname is not available for a fit with lambda_constraint pins: its refits " *
        "would drop the pins and fit a different model. Use loading_profile(fit; y) " *
        "to profile the free loadings of a pinned fit, or confint(fit, y) for Wald " *
        "intervals"))
end

function _confint_gaussian(fit::GllvmFit, y, level::Real, parm, X, Σ_phy,
                           method, kwargs)
    method isa Symbol || throw(ArgumentError(
        "confint: method must be a Symbol, one of $(_confint_list(_CONFINT_METHODS)); " *
        "got $(repr(method))"))
    kw = (; kwargs...)
    targets = _confint_derived_targets(fit, parm)
    targets === nothing || return _confint_derived(fit, targets, y, level, X, Σ_phy,
                                                   method, kw, parm)
    label = _confint_fit_label(fit)
    _confint_check_method(method, label, parm)
    if _has_gaussian_record(fit)
        _confint_check_kwargs(kw, _CONFINT_RECORD_KWARGS, (), label, method)
        data = y === nothing ? fit.integration.data.responses : y
        return _gaussian_record_confint(fit, data; level = level, parm = parm, X = X,
                                        Σ_phy = Σ_phy, method = method, kwargs...)
    end
    return _confint_plain(fit, y, level, parm, X, Σ_phy, method, kw, label)
end

# Wald on the packed θ: the observed-information interval, unchanged.
function _confint_plain_wald(fit::GllvmFit, y, level::Real, parm, X, Σ_phy)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    θ̂, terms, kinds, _, se_all, pd = _confint_gaussian_wald_covariance(fit, y, X, Σ_phy)

    sel = _confint_select_indices(parm, terms)
    isempty(sel) && throw(ArgumentError("parm selector matched no parameters"))

    z = quantile(Normal(), 0.5 + level / 2)

    term_out     = String[]
    estimate_out = Float64[]
    lower_out    = Float64[]
    upper_out    = Float64[]
    se_out       = Float64[]

    for i in sel
        push!(term_out, terms[i])
        push!(se_out, se_all[i])

        θi = θ̂[i]
        sei = se_all[i]
        kind = kinds[i]

        if kind === :log_sd
            est_raw = exp(θi)
            push!(estimate_out, est_raw)
            if isfinite(sei)
                push!(lower_out, exp(θi - z * sei))
                push!(upper_out, exp(θi + z * sei))
            else
                push!(lower_out, NaN)
                push!(upper_out, NaN)
            end
        else
            push!(estimate_out, θi)
            if isfinite(sei)
                push!(lower_out, θi - z * sei)
                push!(upper_out, θi + z * sei)
            else
                push!(lower_out, NaN)
                push!(upper_out, NaN)
            end
        end
    end

    return (term = term_out,
            estimate = estimate_out,
            lower = lower_out,
            upper = upper_out,
            se = se_out,
            pd_hessian = pd)
end

# Profile and bootstrap on the packed θ of a fit with no retained data record:
# the existing `profile_ci` / `bootstrap_ci`, returned in the shape the family
# and record routes use (`term`, `estimate`, `lower`, `upper`, plus `status` or
# `n_converged`, and `method`).
function _confint_plain(fit::GllvmFit, y, level::Real, parm, X, Σ_phy,
                        method::Symbol, kw::NamedTuple, label::AbstractString)
    _confint_check_kwargs(kw, _CONFINT_PLAIN_KWARGS, _CONFINT_INERT_KWARGS, label, method)
    method === :wald && return _confint_plain_wald(fit, y, level, parm, X, Σ_phy)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    _confint_check_unpinned(fit, method, label, parm)
    y === nothing && throw(ArgumentError(
        "confint(...; method = :$method) requires the data matrix `y` " *
        "(the same matrix passed to fit_gaussian_gllvm)"))

    terms, kinds = _confint_all_term_names(fit)
    sel = _confint_select_indices(parm, terms)
    isempty(sel) && throw(ArgumentError("parm selector matched no parameters"))

    if method === :profile
        θ̂ = fit.pars.θ_packed
        lo = Float64[]; hi = Float64[]; status = Symbol[]
        for i in sel
            r = profile_ci(fit, i; level = level, y = y, X = X, Σ_phy = Σ_phy,
                           profile_iterations = get(kw, :profile_iterations, 200),
                           profile_g_tol = get(kw, :profile_g_tol, 1e-4),
                           profile_max_expand = get(kw, :profile_max_expand, 20),
                           profile_max_bisect = get(kw, :profile_max_bisect, 30))
            push!(lo, r.lower); push!(hi, r.upper); push!(status, r.method)
        end
        est = [kinds[i] === :log_sd ? exp(θ̂[i]) : θ̂[i] for i in sel]
        return (term = terms[sel], estimate = est, lower = lo, upper = hi,
                status = status, method = :profile)
    end

    r = bootstrap_ci(fit; n_boot = get(kw, :n_boot, 200), level = level,
                     seed = get(kw, :seed, 0), y = y, X = X, Σ_phy = Σ_phy,
                     parms = terms[sel])
    # This route reports SD terms on the raw scale, like the Wald and profile routes.
    # `bootstrap_ci` does too since its #156 fix; before that its estimate and bounds
    # for `sigma_eps`, `sigma_B[t]` and `sigma_W[t]` were log(σ). The estimate tells
    # which: it equals the packed value only on the log scale (exp(x) == x has no
    # solution), so only then are the three columns converted.
    θ̂ = fit.pars.θ_packed
    est = copy(r.estimate); lo = copy(r.lower); hi = copy(r.upper)
    for (k, i) in enumerate(sel)
        if kinds[i] === :log_sd && est[k] == θ̂[i]
            est[k] = exp(est[k]); lo[k] = exp(lo[k]); hi[k] = exp(hi[k])
        end
    end
    return merge(r, (estimate = est, lower = lo, upper = hi, method = :bootstrap))
end

# ---------------------------------------------------------------------------
# Derived quantities by `parm` name (communality, icc, rho, proportion,
# phylo_signal), routed to the transformed-Wald, derived-profile and
# derived-bootstrap functions of confint_derived_wald.jl / confint_derived.jl.
# ---------------------------------------------------------------------------

const _CONFINT_DERIVED_KINDS = Dict(
    "communality" => :communality,
    "icc" => :icc, "repeatability" => :icc,
    "rho" => :rho, "correlation" => :rho,
    "proportion" => :proportion, "proportions" => :proportion,
    "phylo_signal" => :phylo_signal)

# Methods each derived quantity can use. To refuse one of them, remove it here.
# gllvmTMB 9539352f6 withdrew the profile interval for communality, rho and
# proportion (condition class gllvmTMB_nonlinear_profile_withdrawn), so those
# kinds refuse :profile with the withdrawn message below.
const _CONFINT_DERIVED_METHODS = (communality = (:wald, :bootstrap),
                                  icc = (:wald, :profile, :bootstrap),
                                  rho = (:wald, :bootstrap),
                                  proportion = (:wald, :bootstrap),
                                  phylo_signal = (:wald, :profile, :bootstrap))

# The quantity name R's withdrawn-profile message uses, for each kind whose profile
# is withdrawn (R/z-confint-gllvmTMB.R:964-968, :1055-1060, :1202-1206 at 9539352f6).
const _CONFINT_PROFILE_WITHDRAWN = (communality = "communality", rho = "correlations",
                                    proportion = "variance proportions")

function _confint_profile_withdrawn(kind::Symbol, parm)
    what = getproperty(_CONFINT_PROFILE_WITHDRAWN, kind)
    alt = kind === :rho ? "method = :wald (the Fisher-z interval)" : "method = :wald"
    throw(ArgumentError(
        "confint: nonlinear profile intervals for $what are withdrawn (parm " *
        "$(_confint_parm_label(parm))). gllvmTMB withdrew its penalty-based " *
        "constrained-refit profile pending an exact constraint solver and diagnostic " *
        "contract, and this route follows it. Request $alt or method = :bootstrap, and " *
        "report the method's limitations."))
end

# `proportions(fit; component)` names. gllvmTMB's `shared_unit` / `unique_unit` are not
# accepted: its proportions are the aligned `extract_proportions` estimands, which differ
# from `proportions(fit)` (review of #709), so the same word would name a different number.
const _CONFINT_PROPORTION_COMPONENTS = Dict(
    "shared" => :shared,
    "unique_B" => :unique_B,
    "unique_W" => :unique_W, "unique_Wd" => :unique_Wd, "residual" => :residual)

const _CONFINT_DERIVED_FORMS =
    "communality[t], icc[t] (or repeatability), rho[i,j] (or correlation), " *
    "proportion:<component>[t], phylo_signal[t]; leave out [..] for every trait " *
    "(every pair for rho); separate several traits or pairs with ';'"

# `nothing` when `s` does not start with a derived-quantity name (it is then an
# ordinary packed-term selector).
function _confint_parse_derived(s::String)
    str = replace(strip(s), r"\s+" => "")
    m0 = match(r"^[A-Za-z_]+", str)
    m0 === nothing && return nothing
    kind = get(_CONFINT_DERIVED_KINDS, m0.match, nothing)
    kind === nothing && return nothing
    m = match(r"^([A-Za-z_]+)(?::([A-Za-z_]+))?(?:\[([0-9,;]+)\]|:([0-9,;]+))?$", str)
    m === nothing && throw(ArgumentError(
        "confint: could not parse parm \"$s\" as a derived quantity; accepted forms: " *
        _CONFINT_DERIVED_FORMS))
    spec = m.captures[3] !== nothing ? m.captures[3] : m.captures[4]
    return (kind = kind, word = m.captures[2], spec = spec, raw = s)
end

function _confint_parse_index(tok::AbstractString, p::Integer, raw::AbstractString)
    v = tryparse(Int, tok)
    v === nothing && throw(ArgumentError(
        "confint: \"$tok\" in parm \"$raw\" is not an integer trait index"))
    1 <= v <= p || throw(ArgumentError(
        "confint: trait index $v in parm \"$raw\" is out of range 1:$p"))
    return v
end

function _confint_expand_derived(fit::GllvmFit, pr)
    p = fit.model.p
    kind, word, raw = pr.kind, pr.word, pr.raw
    # gllvmTMB's tier spellings (communality:unit:t, rho:unit:i,j) select its aligned
    # estimands (extract_communality, extract_correlations), which differ from the
    # communality(fit) / correlation(fit) quantities this route computes; refuse them
    # rather than let one name mean two numbers.
    tier_msg(q, form) = "confint: gllvmTMB's tier spelling \"$raw\" names its aligned $q estimand, " *
        "which differs from the $q this route computes; write $form instead (see the " *
        "confint docstring for which quantity that is)"
    tier_ok = word === nothing
    if kind === :rho
        tier_ok || throw(ArgumentError(tier_msg("correlation", "rho[i,j]")))
        p >= 2 || throw(ArgumentError("confint: rho needs at least two traits"))
        pairs = if pr.spec === nothing
            [(i, j) for i in 1:p for j in (i + 1):p]
        else
            map(split(pr.spec, ';')) do item
                toks = split(item, ',')
                length(toks) == 2 || throw(ArgumentError(
                    "confint: \"$item\" in parm \"$raw\" is not a pair; write rho[i,j]"))
                i, j = (_confint_parse_index(t, p, raw) for t in toks)
                i == j && throw(ArgumentError(
                    "confint: rho[$i,$j] is the constant 1; give two different traits"))
                (i, j)
            end
        end
        return [(kind = :rho, label = "rho[$i,$j]", t = 0, i = i, j = j, comp = :none)
                for (i, j) in pairs]
    end
    comp = :none
    if kind === :communality
        tier_ok || throw(ArgumentError(tier_msg("communality", "communality[t]")))
    elseif kind === :proportion
        comp = word === nothing ? :shared : get(_CONFINT_PROPORTION_COMPONENTS, word, nothing)
        comp === nothing && throw(ArgumentError(
            "confint: unknown proportion component \"$word\" in parm \"$raw\"; available: " *
            join(sort(collect(keys(_CONFINT_PROPORTION_COMPONENTS))), ", ")))
        model = fit.model
        (comp === :unique_W && model.K_W == 0) && throw(ArgumentError(
            "confint: proportion:unique_W is identically zero for this fit (K_W = 0)"))
        (comp in (:unique_B, :unique_Wd) && !model.has_diag) && throw(ArgumentError(
            "confint: proportion:$comp is identically zero for this fit (has_diag = false)"))
    else  # :icc, :phylo_signal take no tier
        word === nothing || throw(ArgumentError(
            "confint: $kind takes no tier or component; got \"$word\" in parm \"$raw\""))
        if kind === :phylo_signal
            model = fit.model
            (model.K_phy > 0 || model.has_phy_unique) || throw(ArgumentError(
                "confint: phylo_signal is undefined for this fit (no phylogenetic block)"))
        end
    end
    traits = if pr.spec === nothing
        collect(1:p)
    else
        occursin(',', pr.spec) && throw(ArgumentError(
            "confint: parm \"$raw\" has a comma; a trait is one integer, separate several with ';'"))
        [_confint_parse_index(t, p, raw) for t in split(pr.spec, ';')]
    end
    return [(kind = kind, label = kind === :proportion ? "proportion:$comp[$t]" : "$kind[$t]",
             t = t, i = 0, j = 0, comp = comp) for t in traits]
end

# Targets for a derived `parm`, or `nothing` when `parm` is an ordinary packed
# selector. A vector must be all derived names or all packed names.
function _confint_derived_targets(fit::GllvmFit, parm)
    items = parm isa Union{AbstractString, Symbol} ? [parm] :
            parm isa AbstractVector ? parm : return nothing
    all(x -> x isa Union{AbstractString, Symbol}, items) || return nothing
    parsed = [_confint_parse_derived(String(x)) for x in items]
    n = count(!isnothing, parsed)
    n == 0 && return nothing
    n == length(parsed) || throw(ArgumentError(
        "confint: parm mixes derived-quantity names with parameter names " *
        "($(join(string.(items), ", "))); request them in separate calls"))
    targets = NamedTuple[]
    for pr in parsed
        append!(targets, _confint_expand_derived(fit, pr))
    end
    return targets
end

function _confint_derived_closure(tg, spec::NamedTuple, Σ_phy)
    tg.kind === :communality && return _make_communality_closure(spec, tg.t)
    tg.kind === :icc && return _make_icc_closure(spec, tg.t)
    tg.kind === :rho && return _make_correlation_closure(spec, tg.i, tg.j)
    tg.kind === :proportion && return _make_proportion_closure(spec, tg.t, tg.comp)
    return _make_phylo_signal_closure(spec, tg.t;
        diag_Σphy = Σ_phy === nothing ? nothing : diag(Σ_phy))
end

# The public accessor of the quantity, as a function of a fit: what the bootstrap
# evaluates on every refit (so the interval is for exactly that accessor).
function _confint_derived_accessor(tg, Σ_phy)
    tg.kind === :communality && return f -> communality(f)[tg.t]
    tg.kind === :icc && return f -> extract_ICC_site(f)[tg.t]
    tg.kind === :rho && return f -> correlation(f)[tg.i, tg.j]
    tg.kind === :proportion && return f -> proportions(f; component = tg.comp)[tg.t]
    return f -> phylo_signal(f; Σ_phy = Σ_phy)[tg.t]
end

function _confint_derived(fit::GllvmFit, targets::Vector, y, level::Real, X, Σ_phy,
                          method::Symbol, kw::NamedTuple, parm)
    label = _confint_fit_label(fit)
    for kind in unique(tg.kind for tg in targets)
        method === :profile && haskey(_CONFINT_PROFILE_WITHDRAWN, kind) &&
            _confint_profile_withdrawn(kind, parm)
        _confint_check_method(method, label, parm; available = getproperty(_CONFINT_DERIVED_METHODS, kind))
    end
    _confint_check_kwargs(kw, _CONFINT_DERIVED_KWARGS, _CONFINT_INERT_KWARGS, label, method)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    _has_lv_predictor(fit) && throw(ArgumentError(
        "confint for derived quantities is not admitted for fit_gaussian_gllvm(...; X_lv=...) " *
        "(the C1 predictor-informed latent-score path)"))
    method === :wald || _confint_check_unpinned(fit, method, label, parm)
    data = y === nothing && _has_gaussian_record(fit) ? fit.integration.data.responses : y
    data === nothing && throw(ArgumentError(
        "confint(...; parm = $(_confint_parm_label(parm))) requires the data matrix `y` " *
        "(the same matrix passed to fit_gaussian_gllvm)"))

    spec = _derived_spec(fit)
    fns = [_confint_derived_closure(tg, spec, Σ_phy) for tg in targets]
    terms = String[tg.label for tg in targets]
    θ̂ = fit.pars.θ_packed

    if method === :wald
        transforms = [tg.kind === :rho ? :fisher_z : :logit for tg in targets]
        # One observed information serves every target (the single-target wrappers
        # form the same matrix); it is skipped when no estimate is interior.
        need = any(zip(fns, transforms)) do (f, tr)
            g = Float64(f(θ̂))
            lo, hi = _tw_link(tr)[3]
            isfinite(g) && lo < g < hi
        end
        Σ, pd = need ? _tw_sigma_from_hessian(fit, data, X, Σ_phy) : (nothing, false)
        rows = [_transformed_wald_ci_with_sigma(θ̂, f, Σ, pd; transform = tr, level = level)
                for (f, tr) in zip(fns, transforms)]
        return (term = terms, estimate = Float64[r.estimate for r in rows],
                lower = Float64[r.lower for r in rows], upper = Float64[r.upper for r in rows],
                se_transformed = Float64[r.se_transformed for r in rows],
                transform = Symbol[r.transform for r in rows],
                pd_hessian = Bool[r.pd_hessian for r in rows],
                status = Symbol[r.method for r in rows], method = :wald)
    elseif method === :profile
        pkw = (max_expand = get(kw, :profile_max_expand, 20),
               max_bisect = get(kw, :profile_max_bisect, 30),
               penalty_weight = get(kw, :penalty_weight, 1e6))
        rows = map(zip(targets, fns)) do (tg, f)
            if tg.kind === :phylo_signal
                profile_ci_phylo_signal(fit, tg.t; level = level, y = data, X = X,
                                        Σ_phy = Σ_phy, pkw...)
            else
                r = profile_ci_derived(fit, f; level = level, y = data, X = X,
                                       Σ_phy = Σ_phy, pkw...)
                lo, hi = tg.kind === :rho ? (-1.0, 1.0) : (0.0, 1.0)
                _profile_ci_bounded(fit, f, r; level = level, y = data, X = X,
                                    Σ_phy = Σ_phy, lo_bound = lo, hi_bound = hi)
            end
        end
        return (term = terms, estimate = Float64[r.estimate for r in rows],
                lower = Float64[r.lower for r in rows], upper = Float64[r.upper for r in rows],
                status = Symbol[r.method for r in rows],
                boundary = Bool[r.boundary for r in rows], method = :profile)
    end

    # One bootstrap per quantity, as in gllvmTMB; the same seed gives every
    # quantity the same simulated data sets.
    rows = [bootstrap_ci_derived(fit, _confint_derived_accessor(tg, Σ_phy);
                                 n_boot = get(kw, :n_boot, 200), level = level,
                                 seed = get(kw, :seed, 0), y = data, X = X, Σ_phy = Σ_phy)
            for tg in targets]
    return (term = terms, estimate = Float64[r.estimate for r in rows],
            lower = Float64[r.lower for r in rows], upper = Float64[r.upper for r in rows],
            n_converged = rows[1].n_converged, n_valid = Int[r.n_valid for r in rows],
            replicates = reduce(hcat, [r.replicates for r in rows]), method = :bootstrap)
end
