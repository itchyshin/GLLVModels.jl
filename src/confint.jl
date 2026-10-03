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
# the marginal NLL at θ̂ — no regularisation. Non-PD convention (unchanged from
# the SE code): if the Hessian errors, is non-finite, or cannot be inverted,
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

    se_all = fill(NaN, n_par)
    V = fill(NaN, n_par, n_par)
    if H !== nothing && all(isfinite, H)
        Σ = nothing
        try
            Hsym = (H .+ H') ./ 2
            Σ = inv(Hsym)
        catch
            Σ = nothing
            pd = false
        end

        if Σ !== nothing
            diagΣ = diag(Σ)
            for i in 1:n_par
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
    confint(fit::GllvmFit; level=0.95, parm=nothing,
            y=nothing, X=nothing, Σ_phy=nothing) -> NamedTuple

Wald confidence intervals for the parameters of a fitted Gaussian
GLLVM. Returns a NamedTuple with fields:

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
`K_phy = 0`, its sign can flip independently within each group of rows that
`Σ_phy` links (for example each daughter clade of the root when the root edge
has length zero); when `K_phy ≥ 1`, it is not identified separately from
`Λ_phy` and a per-entry interval is not interpretable. It is not gllvmTMB's `phylo_unique` scale; see the gllvmTMB
parity page.

The Hessian is computed via ForwardDiff at the fitted parameter vector
stored on `fit.pars.θ_packed`. The function needs the original data
matrix `y` (and optionally `X`, `Σ_phy`) to reconstruct the NLL closure.
If the Hessian is not positive definite, this function returns NaN
bounds for the affected entries with `pd_hessian = false` (matching the
R glmmTMB / gllvmTMB convention).

When PERF lands with reverse-mode AD, the integration agent can swap the
ForwardDiff.hessian call for the faster path; the public API stays
stable.
"""
function confint(fit::GllvmFit;
                 level::Real = 0.95,
                 parm::Union{Nothing, AbstractString, Symbol, AbstractVector} = nothing,
                 y::Union{Nothing, AbstractMatrix} = nothing,
                 X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                 Σ_phy::Union{Nothing, AbstractMatrix} = nothing,kwargs...)

    if _has_gaussian_record(fit)
        data=y===nothing ? fit.integration.data.responses : y
        return _gaussian_record_confint(fit,data;level=level,parm=parm,X=X,Σ_phy=Σ_phy,kwargs...)
    end
    isempty(kwargs) || throw(ArgumentError("extra inference controls require retained Gaussian integration data"))

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

function confint(fit::GllvmFit, y::AbstractMatrix;
                 level::Real = 0.95,
                 parm::Union{Nothing, AbstractString, Symbol, AbstractVector} = nothing,
                 X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                 Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
                 method::Symbol = :wald,
                 kwargs...)
    if _has_gaussian_record(fit)
        return _gaussian_record_confint(fit,y;level=level,parm=parm,X=X,Σ_phy=Σ_phy,method=method,kwargs...)
    end
    return confint(fit; level = level, parm = parm, y = y, X = X, Σ_phy = Σ_phy)
end
