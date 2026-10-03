# Profile-likelihood confidence intervals for the Gaussian GLLVM.
#
# For parameter θ_i with MLE θ̂_i and full log-likelihood ℓ̂,
# the profile log-lik at candidate value c is
#     ℓ_p(c) = max_{θ_{-i}} ℓ(c, θ_{-i}),
# i.e., re-optimise over the remaining parameters with θ_i fixed.
# The deviance D(c) = 2(ℓ̂ − ℓ_p(c)) is ~ χ²_1 under the null θ_i = c.
# The 100(1−α)% profile CI is {c : D(c) ≤ qchisq(1−α, df=1)}.
#
# Why profile CIs:
#   For the ADEMP coverage simulation, Wald CIs have poor coverage when
#   σ_eps or σ_phy is near zero (boundary), the likelihood is asymmetric
#   (common for ratios like ICC, H²), or the fit is near-singular.
#   Profile CIs invert the LRT directly — wider tail support, better
#   coverage at boundaries.
#
# Algorithm:
#   1. Build the full negative log-likelihood closure as a function of the
#      legacy-layout θ_packed vector (same NLL used by confint.jl).
#   2. Compute Wald SEs at the MLE via ForwardDiff.hessian. Used only to
#      seed the initial bracket — the SE doesn't enter the final answer.
#   3. For each side (lower and upper) of θ̂_i, expand c outward in steps
#      of grid_extent·SE/n_steps until the deviance D(c) crosses the
#      chisq cutoff. At each candidate c we re-optimise over θ_{-i} via
#      LBFGS (warm-started from the previous solution) and compute
#      D(c) = 2(ℓ̂ − ℓ_p(c)).
#   4. Root-find inside the bracket via false position on √D (≈ linear in c
#      near the MLE, since D ≈ (c−θ̂)²/SE²), with a bisection safeguard — a few
#      refits instead of ~log2(width/tol) bisections, at the same crossing.
#
# The result on log-SD-style parameters (σ_eps, σ_B, σ_W) is converted to
# the raw scale via exp(.) to match the convention of src/confint.jl: bounds
# are reported on the natural (positive) scale. σ_phy is the exception: it is
# packed on an identity (signed) scale, so its bounds stay on that raw scale
# and can be negative (#136).
#
# Constrained-refit mechanics: Optim.jl has no first-class
# "hold parameter k fixed" interface, so we instead define a closure
# over a reduced parameter vector θ_red ∈ R^{N-1} that inserts the
# fixed value c at index i to form θ_full ∈ R^N before calling
# gaussian_nll_packed. This is the standard Julia idiom and keeps
# ForwardDiff happy (the closure is differentiable in θ_red).

using Distributions: Chisq, quantile

# Build the lambda part of the term-name list in pack_lambda order.
# Diagonals (k = 1..K) first, then strict-lower entries column-by-column.
# Mirrors `pack_lambda` / `unpack_lambda` in src/packing.jl.
function _profile_lambda_term_names(prefix::String, p::Integer, K::Integer)
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

# Build the canonical term-name + kind vectors matching the legacy
# θ_packed layout, identical to the sister Wald confint.jl.
#
#     [β[1..q]; sigma_eps;
#      sigma_B[1..p]; sigma_W[1..p]      (if has_diag)
#      Lambda_B[i,k]                      (pack_lambda order)
#      Lambda_W[i,k]                      (if K_W > 0)
#      sigma_phy[1..p]                    (if has_phy_unique)
#      Lambda_phy[i,k]                    (if K_phy > 0)]
#
# kinds[i] ∈ {:linear, :log_sd} drives the raw-vs-working transform.
function _profile_all_term_names(fit::GllvmFit)
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

    for nm in _profile_lambda_term_names("Lambda_B", p, K_B)
        push!(terms, nm)
        push!(kinds, :linear)
    end

    if K_W > 0
        for nm in _profile_lambda_term_names("Lambda_W", p, K_W)
            push!(terms, nm)
            push!(kinds, :linear)
        end
    end

    if has_phy_unique
        # σ_phy uses an identity (signed) link — profile bounds stay on the raw scale.
        for t in 1:p
            push!(terms, "sigma_phy[$t]")
            push!(kinds, :linear)
        end
    end

    if K_phy > 0
        for nm in _profile_lambda_term_names("Lambda_phy", p, K_phy)
            push!(terms, nm)
            push!(kinds, :linear)
        end
    end

    return terms, kinds
end

# Resolve a parm name (matching the confint() naming convention) to
# the integer index into θ_packed.
function _profile_parm_index(fit::GllvmFit, parm::AbstractString)
    terms, _ = _profile_all_term_names(fit)
    selector = String(parm)
    idx = findfirst(==(selector), terms)
    if !isnothing(idx)
        return idx
    end
    if startswith(selector, "Lambda:")
        return _profile_parm_index(fit, "Lambda_B[" * selector[length("Lambda:") + 1:end] * "]")
    end
    for prefix in ("Lambda_B:", "Lambda_W:", "Lambda_phy:")
        if startswith(selector, prefix)
            base = prefix[1:end-1]
            return _profile_parm_index(fit, "$(base)[" * selector[length(prefix) + 1:end] * "]")
        end
    end
    throw(ArgumentError(
        "Could not resolve parm selector \"$selector\" to a single θ_packed index. " *
        "Use one of the names returned by confint(fit).term."))
end

# Build the spec NamedTuple used by gaussian_nll_packed.
function _profile_spec(fit::GllvmFit)
    model = fit.model
    q_full = fit.pars.β === nothing ? 0 : length(fit.pars.β)
    β_fixed = _pars_fixed_mask(fit.pars, :β_fixed, q_full)
    return (q = count(!, β_fixed), p = model.p, K_B = model.K, K_W = model.K_W,
            has_diag = model.has_diag, K_phy = model.K_phy,
            has_phy_unique = model.has_phy_unique)
end

function _profile_free_X(fit::GllvmFit, X::Union{Nothing, AbstractArray{<:Real, 3}})
    X === nothing && return nothing
    q_full = fit.pars.β === nothing ? 0 : length(fit.pars.β)
    β_fixed = _pars_fixed_mask(fit.pars, :β_fixed, q_full)
    β_free = _free_coeff_indices(β_fixed)
    isempty(β_free) && return nothing
    return Array{Float64,3}(X[:, :, β_free])
end

# Wald SE at θ̂_i via the observed information matrix. Returns NaN if
# the Hessian is non-finite or the i-th diagonal of inv(H) is ≤ 0.
function _profile_wald_se(fit::GllvmFit, i::Integer,
                          y::AbstractMatrix,
                          X::Union{Nothing, AbstractArray{<:Real, 3}},
                          Σ_phy::Union{Nothing, AbstractMatrix})
    spec = _profile_spec(fit)
    θ̂ = fit.pars.θ_packed
    X_free = _profile_free_X(fit, X)
    nll = θ -> gaussian_nll_packed(θ, y; spec = spec, X = X_free, Σ_phy = Σ_phy)
    H = try
        ForwardDiff.hessian(nll, θ̂)
    catch
        return NaN
    end
    if !all(isfinite, H)
        return NaN
    end
    Σ_inv = try
        inv((H .+ H') ./ 2)
    catch
        return NaN
    end
    v = diag(Σ_inv)[i]
    return (isfinite(v) && v > 0) ? sqrt(v) : NaN
end

# Constrained refit: re-optimise the NLL over θ_{-i} with θ_i fixed at c.
# Returns (ll_profile::Float64, success::Bool). On optimisation failure
# returns (NaN, false).
#
# Mechanics — how we "fix" a parameter for Optim:
#   Optim has no first-class held-constant interface, so we wrap the NLL
#   in a closure that maps a reduced (N-1)-vector θ_red to the full
#   N-vector by inserting c at index i, then evaluates gaussian_nll_packed
#   on the full vector. ForwardDiff differentiates through the insertion
#   because it's just an indexing/concat operation.
function _profile_refit_with_fixed(fit::GllvmFit, i::Integer, c::Real,
                                   y::AbstractMatrix,
                                   X::Union{Nothing, AbstractArray{<:Real, 3}},
                                   Σ_phy::Union{Nothing, AbstractMatrix};
                                   θ_red_warm::Union{Nothing, AbstractVector} = nothing,
                                   x_tol::Real = 1e-6,
                                   f_tol::Real = 1e-8,
                                   g_tol::Real = 1e-4,
                                   iterations::Integer = 200)
    spec = _profile_spec(fit)
    X_free = _profile_free_X(fit, X)
    θ̂ = fit.pars.θ_packed
    N = length(θ̂)
    1 ≤ i ≤ N || throw(ArgumentError("param_index $i out of range 1:$N"))

    # Reduced warm-start: drop index i.
    θ_red0 = if θ_red_warm === nothing
        vcat(θ̂[1:(i - 1)], θ̂[(i + 1):N])
    else
        collect(Float64, θ_red_warm)
    end

    function _full_from_red(θ_red, c_val)
        # Insert c_val at position i. Preserve eltype for AD.
        T = promote_type(eltype(θ_red), typeof(c_val))
        θ_full = Vector{T}(undef, N)
        @inbounds for j in 1:(i - 1)
            θ_full[j] = θ_red[j]
        end
        θ_full[i] = c_val
        @inbounds for j in (i + 1):N
            θ_full[j] = θ_red[j - 1]
        end
        return θ_full
    end

    c_float = float(c)
    nll_red = θ_red -> gaussian_nll_packed(_full_from_red(θ_red, c_float), y;
                                           spec = spec, X = X_free, Σ_phy = Σ_phy)

    opts = Optim.Options(
        x_abstol = x_tol,
        f_reltol = f_tol,
        g_tol    = g_tol,
        iterations = iterations,
        show_trace = false,
    )

    res = try
        Optim.optimize(nll_red, θ_red0, Optim.LBFGS(), opts; autodiff = :forward)
    catch
        return (NaN, false, θ_red0)
    end

    nll_min = Optim.minimum(res)
    if !isfinite(nll_min)
        return (NaN, false, θ_red0)
    end
    return (-nll_min, true, Optim.minimizer(res))
end

# Bracket-then-root-find on one side (lower if Δ_init < 0, upper if > 0).
#
# Phase 1 (bracket): walk outward from x0 = θ̂_i in geometrically doubling steps
# until the deviance D(c) crosses cutoff or we exhaust max_expand expansions.
#
# Phase 2 (root): locate D(c) = cutoff inside the bracket via false position on
# s(c) = √(max(D(c),0)). Because D(c) ≈ (c−θ̂)²/SE² near the MLE, √D is nearly
# LINEAR in c, so false position converges in a few refits where plain bisection
# needs ~log2(width/tol) (~15). A bisection safeguard — taken whenever a proposal
# lands on a bracket endpoint (one-sided stagnation) or the outer deviance is
# non-finite (singular refit) — keeps it a strict bracketing method with worst-case
# bisection rate. Each D call is one constrained refit (the dominant cost), so this
# minimises refits at identical accuracy.
function _profile_root_falsepos(D::Function, a::Real, b::Real,
                                Da::Real, Db::Real, cutoff::Real;
                                max_iter::Integer = 40, tol_x::Real = 1e-4,
                                tol_D::Real = 1e-3)
    S  = sqrt(cutoff)
    fa = sqrt(max(float(Da), 0.0)) - S          # < 0  (D(a) < cutoff)
    fb = isfinite(Db) ? sqrt(max(float(Db), 0.0)) - S : NaN   # ≥ 0, or NaN if singular
    a  = float(a); b = float(b)
    for _ in 1:max_iter
        w = abs(b - a)
        w < tol_x && break
        lo, hi = minmax(a, b)
        usefp = isfinite(fa) && isfinite(fb) && (fb - fa) != 0
        c = usefp ? b - fb * (b - a) / (fb - fa) : (a + b) / 2
        margin = 1e-3 * w
        if !isfinite(c) || c ≤ lo + margin || c ≥ hi - margin
            c = (a + b) / 2                       # safeguard ⇒ bisection step
        end
        Dc = D(c)
        if !isfinite(Dc)
            b = c; fb = NaN                       # singular outer region: contract in
            continue
        end
        abs(Dc - cutoff) < tol_D && return c      # crossing located ⇒ stop (saves refits)
        fc = sqrt(max(Dc, 0.0)) - S
        if fc ≥ 0
            b = c; fb = fc
        else
            a = c; fa = fc
        end
    end
    # A singular/failed refit is not evidence of a likelihood crossing.
    # Require a finite outer endpoint; otherwise expose the missing bound.
    return isfinite(fb) ? (a + b) / 2 : NaN
end

function _profile_bisect_side(D::Function, x0::Real, step_init::Real,
                              cutoff::Real;
                              max_expand::Integer = 20,
                              max_bisect::Integer = 30,
                              tol_x::Real = 1e-4,
                              tol_D::Real = 1e-3)
    sign_step = sign(step_init)
    sign_step == 0 && return NaN
    abs_step = abs(step_init)

    # Phase 1 — bracket: find x_out with D(x_out) ≥ cutoff while D(x_in) < cutoff.
    x_in = float(x0)
    D_in = 0.0  # D(θ̂_i) = 0 by construction
    x_out = x_in + sign_step * abs_step
    D_out = NaN
    found_bracket = false
    for k in 1:max_expand
        D_val = D(x_out)
        if !isfinite(D_val)
            # Refit failed: bracket the singular region (root-finder contracts in).
            D_out = Inf
            found_bracket = true
            break
        end
        if D_val ≥ cutoff
            D_out = D_val
            found_bracket = true
            break
        end
        x_in = x_out
        D_in = D_val
        abs_step *= 2                              # geometric expansion ⇒ O(log) refits
        x_out = x_in + sign_step * abs_step
    end
    found_bracket || return NaN

    # Phase 2 — root: false position on √D within [x_in, x_out].
    return _profile_root_falsepos(D, x_in, x_out, D_in, D_out, cutoff;
                                  max_iter = max_bisect, tol_x = tol_x, tol_D = tol_D)
end

"""
    profile_ci(fit::GllvmFit, param_index::Integer;
               level = 0.95, grid_extent = 5, max_expand = 20,
               max_bisect = 30, profile_iterations = 200,
               profile_g_tol = 1e-4, profile_max_expand = nothing,
               profile_max_bisect = nothing, y = nothing, X = nothing,
               Σ_phy = nothing)
        -> NamedTuple{(:lower, :upper, :method)}

Profile-likelihood CI for the parameter at packed position `param_index`
in `fit.pars.θ_packed`.

`grid_extent` controls how far (in Wald SEs from θ̂_i, geometrically
expanding) the initial bracket walks before bisection. Larger values
help for asymmetric likelihoods; the geometric expansion keeps the
total number of refits at O(log) even at large `grid_extent`.

`level` is the nominal coverage (default 0.95 → χ²_1 cutoff ≈ 3.841).

`profile_iterations` and `profile_g_tol` control the constrained LBFGS refits.
`profile_max_expand` and `profile_max_bisect` are aliases for the bracketing and
root-finding caps; when omitted they use the legacy `max_expand` and
`max_bisect` values.

The data matrix `y` (the same `y` passed to `fit_gaussian_gllvm`) must
be supplied so this function can reconstruct the NLL closure. `X` and
`Σ_phy` are required iff the fit used them.

Returns a NamedTuple with fields:
  - `lower::Float64` — lower CI bound on the raw scale for SD-style
    parameters (σ_eps, σ_B, σ_W, σ_phy), native scale for β / Λ.
    `σ_phy` is a signed parameter (#136), so this interval can cross zero.
    When `K_phy = 0`, the sign of each group of rows that `Σ_phy` links can
    flip on its own (for a tree-derived `Σ_phy`, each daughter clade of the
    root); when `K_phy ≥ 1`, it is not
    identified separately from `Λ_phy` and a per-entry interval is not
    interpretable. It is not gllvmTMB's
    `phylo_unique` scale; see the gllvmTMB parity page.
  - `upper::Float64` — upper CI bound, same scale convention.
  - `method::Symbol` — `:profile` if both bounds were bracketed,
    `:partial` if only one side was found (the other is NaN), or
    `:failed` if neither side could be bracketed (both NaN).

Failure modes (each side independently):
  - The bracket never crosses the chisq cutoff within
    `max_expand` geometric expansions → that bound is `NaN`.
  - A constrained refit at a candidate value fails → bracket contracts
    inward on that side, still typically yielding a finite bound.
"""
function profile_ci(fit::GllvmFit, param_index::Integer;
                    level::Real = 0.95,
                    grid_extent::Real = 5,
                    max_expand::Integer = 20,
                    max_bisect::Integer = 30,
                    profile_iterations::Integer = 200,
                    profile_g_tol::Real = 1e-4,
                    profile_max_expand::Union{Nothing, Integer} = nothing,
                    profile_max_bisect::Union{Nothing, Integer} = nothing,
                    y::Union{Nothing, AbstractMatrix} = nothing,
                    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                    Σ_phy::Union{Nothing, AbstractMatrix} = nothing)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))
    profile_iterations > 0 ||
        throw(ArgumentError("profile_iterations must be positive; got $profile_iterations"))
    isfinite(profile_g_tol) && profile_g_tol > 0 ||
        throw(ArgumentError("profile_g_tol must be positive and finite; got $profile_g_tol"))
    max_expand_eff = profile_max_expand === nothing ? max_expand : profile_max_expand
    max_bisect_eff = profile_max_bisect === nothing ? max_bisect : profile_max_bisect
    max_expand_eff > 0 ||
        throw(ArgumentError("profile_max_expand/max_expand must be positive; got $max_expand_eff"))
    max_bisect_eff > 0 ||
        throw(ArgumentError("profile_max_bisect/max_bisect must be positive; got $max_bisect_eff"))
    _has_lv_predictor(fit) && throw(ArgumentError(
        "profile_ci for fit_gaussian_gllvm(...; X_lv=...) is not admitted in the C1 predictor-informed latent-score path; use extract_lv_effects for point estimates"))
    y === nothing && throw(ArgumentError(
        "profile_ci requires the data matrix `y` (the same matrix passed to fit_gaussian_gllvm)"))
    X = _mean_X(fit, X, size(y, 2))

    if _has_gaussian_record(fit)
        terms,_=_confint_all_term_names(fit)
        1<=param_index<=length(terms) || throw(ArgumentError("parameter index out of bounds"))
        r=_gaussian_record_confint(fit,y;X=X,Σ_phy=Σ_phy,method=:profile,parm=terms[param_index],level=level,
            profile_iterations=profile_iterations,profile_g_tol=profile_g_tol,
            profile_max_expand=max_expand_eff,profile_max_bisect=max_bisect_eff)
        lo,hi=only(r.lower),only(r.upper)
        return (lower=lo,upper=hi,method=isnan(lo) && isnan(hi) ? :failed : isnan(lo) || isnan(hi) ? :partial : :profile)
    end

    θ̂ = fit.pars.θ_packed
    N = length(θ̂)
    1 ≤ param_index ≤ N ||
        throw(ArgumentError("param_index $param_index out of range 1:$N"))

    cutoff = quantile(Chisq(1), level)
    θ̂_i = float(θ̂[param_index])
    ll_full = fit.logLik

    # Wald SE on the working scale; fall back to a heuristic if non-PD.
    se_i = _profile_wald_se(fit, param_index, y, X, Σ_phy)
    if isnan(se_i) || se_i ≤ 0
        se_i = max(abs(θ̂_i) / 2, 0.1)
    end

    # Warm-start cache for the constrained refits. Re-using the previous
    # θ_red on each step typically cuts LBFGS iterations to ~5–15 because
    # the constrained MLE is a smooth function of c.
    θ_red_warm_lower = vcat(θ̂[1:(param_index - 1)], θ̂[(param_index + 1):N])
    θ_red_warm_upper = copy(θ_red_warm_lower)

    function deviance_lower(c)
        ll_c, ok, θ_red_new = _profile_refit_with_fixed(
            fit, param_index, c, y, X, Σ_phy;
            θ_red_warm = θ_red_warm_lower,
            g_tol = profile_g_tol,
            iterations = profile_iterations)
        if ok
            # Update warm-start in the enclosing scope for the next call.
            θ_red_warm_lower = θ_red_new
            return 2.0 * (ll_full - ll_c)
        else
            return NaN
        end
    end
    function deviance_upper(c)
        ll_c, ok, θ_red_new = _profile_refit_with_fixed(
            fit, param_index, c, y, X, Σ_phy;
            θ_red_warm = θ_red_warm_upper,
            g_tol = profile_g_tol,
            iterations = profile_iterations)
        if ok
            θ_red_warm_upper = θ_red_new
            return 2.0 * (ll_full - ll_c)
        else
            return NaN
        end
    end

    # Initial step: aim the first candidate near the Wald bound (θ̂ ± √cutoff·SE),
    # where D ≈ cutoff, so the bracket is found in ~1 refit; geometric expansion
    # inside _profile_bisect_side handles asymmetric / non-quadratic deviances.
    step_init = max(min(sqrt(cutoff), grid_extent) * se_i, 1e-3)

    lower = _profile_bisect_side(deviance_lower, θ̂_i, -step_init, cutoff;
                                 max_expand = max_expand_eff,
                                 max_bisect = max_bisect_eff)
    upper = _profile_bisect_side(deviance_upper, θ̂_i,  step_init, cutoff;
                                 max_expand = max_expand_eff,
                                 max_bisect = max_bisect_eff)

    # Raw-scale conversion for log-SD parameters (σ_eps etc.).
    _, kinds = _profile_all_term_names(fit)
    if kinds[param_index] === :log_sd
        lower = isnan(lower) ? NaN : exp(lower)
        upper = isnan(upper) ? NaN : exp(upper)
    end

    method = if isnan(lower) && isnan(upper)
        :failed
    elseif isnan(lower) || isnan(upper)
        :partial
    else
        :profile
    end
    return (lower = lower, upper = upper, method = method)
end

"""
    profile_ci(fit::GllvmFit, parm::AbstractString; kwargs...)
        -> NamedTuple{(:lower, :upper, :method)}

Convenience method that looks up `parm` by name (e.g., `"sigma_eps"`,
`"Lambda_B[1,1]"`, `"Lambda:1,1"`) and calls the integer-index method.

Naming convention matches `confint(fit)` from src/confint.jl: SDs are
reported on the raw (positive) scale, β and Λ on their native scale.
"""
function profile_ci(fit::GllvmFit, parm::AbstractString; kwargs...)
    idx = _profile_parm_index(fit, parm)
    return profile_ci(fit, idx; kwargs...)
end

function profile_ci(fit::GllvmFit, parm::Symbol; kwargs...)
    return profile_ci(fit, String(parm); kwargs...)
end

# ---------------------------------------------------------------------------
# APPEND (core070 E-cluster, PERF+SE-machinery): bare profile-curve wrappers.
#
# `profile_ci` above reduces the bracket-then-bisect walk to a single
# (lower, upper) bound — the Julia analogue of R's `tmbprofile_wrapper()`
# RETURN VALUE. R's `tmbprofile_wrapper()` gets there by first calling
# `TMB::tmbprofile()`, which returns the raw (parameter value, deviance)
# TRACE, and only then reducing that trace to bounds via `.profile_bounds()`.
# GLLVModels.jl's `profile_ci` never materialises that trace — every constrained
# refit's deviance is used and discarded inside `_profile_bisect_side`.
#
# `tmbprofile_wrapper` here fills that namespace/surface gap: it re-walks
# the SAME bracket-then-expand loop `_profile_bisect_side` uses (reusing its
# helpers `_profile_wald_se` / `_profile_refit_with_fixed` verbatim), but
# records every evaluated `(θ_i value, nll)` pair instead of throwing them
# away, and returns the sorted trace alongside the reduced bound (obtained
# by simply calling `profile_ci` — no need to duplicate the root-finding
# root-finding logic, since only the CURVE was the missing surface, not the
# bound). `profile_curve_targets` batches this over several parameters;
# `profile_phylo_signal` is the phylo_unique-scoped convenience form — see
# its docstring for the honest scope note (packed `sigma_phy[t]`, not the
# composite H²-like `phylo_signal(fit)[t]` derived quantity).
# ---------------------------------------------------------------------------

# Internal: walk outward from θ̂_i on both sides, recording every evaluated
# (θ_i, nll) pair from the constrained refits. Stops each side once the
# deviance crosses `cutoff` (one point past the crossing) or `max_expand`
# geometric expansions are exhausted — the same stopping rule
# `_profile_bisect_side` uses for its OWN (discarded) trace.
function _tmbprofile_curve(fit::GllvmFit, param_index::Integer;
                           level::Real = 0.95,
                           grid_extent::Real = 5,
                           max_expand::Integer = 20,
                           max_bisect::Integer = 30,
                           profile_iterations::Integer = 200,
                           profile_g_tol::Real = 1e-4,
                           y::AbstractMatrix,
                           X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                           Σ_phy::Union{Nothing, AbstractMatrix} = nothing)
    _has_gaussian_record(fit) && throw(ArgumentError(
        "tmbprofile_wrapper/_tmbprofile_curve is not implemented for masked/" *
        "offset/AGHQ Gaussian-record fits: the TRACE walk here always " *
        "evaluates gaussian_nll_packed (the closed-form dense surface), " *
        "while profile_ci routes these fits to the record objective " *
        "(_gaussian_record_confint) for both the cutoff and the bounds — " *
        "mixing the two surfaces silently would put the trace and the " *
        "cutoff on different objectives. Use profile_ci for the bound; a " *
        "record-objective trace is not yet implemented."))
    θ̂ = fit.pars.θ_packed
    N = length(θ̂)
    1 ≤ param_index ≤ N ||
        throw(ArgumentError("param_index $param_index out of range 1:$N"))

    θ̂_i = float(θ̂[param_index])
    ll_full = fit.logLik
    cutoff = quantile(Chisq(1), level)

    se_i = _profile_wald_se(fit, param_index, y, X, Σ_phy)
    if isnan(se_i) || se_i ≤ 0
        se_i = max(abs(θ̂_i) / 2, 0.1)
    end
    step_init = max(min(sqrt(cutoff), grid_extent) * se_i, 1e-3)

    theta = Float64[θ̂_i]
    nll   = Float64[-ll_full]
    θ_red0 = vcat(θ̂[1:(param_index - 1)], θ̂[(param_index + 1):N])

    for sgn in (-1.0, 1.0)
        x_in = θ̂_i
        abs_step = step_init
        warm = θ_red0
        for _ in 1:max_expand
            x_out = x_in + sgn * abs_step
            ll_c, ok, warm_new = _profile_refit_with_fixed(
                fit, param_index, x_out, y, X, Σ_phy;
                θ_red_warm = warm, g_tol = profile_g_tol,
                iterations = profile_iterations)
            ok || break
            push!(theta, x_out)
            push!(nll, -ll_c)
            warm = warm_new
            D = 2.0 * (ll_full - ll_c)
            x_in = x_out
            (isfinite(D) && D ≥ cutoff) && break
            abs_step *= 2
        end
    end

    perm = sortperm(theta)
    theta_sorted = theta[perm]
    nll_sorted   = nll[perm]

    bound = profile_ci(fit, param_index;
                       level = level, grid_extent = grid_extent,
                       max_expand = max_expand, max_bisect = max_bisect,
                       profile_iterations = profile_iterations,
                       profile_g_tol = profile_g_tol,
                       y = y, X = X, Σ_phy = Σ_phy)

    return (theta = theta_sorted, nll = nll_sorted, estimate = θ̂_i,
            lower = bound.lower, upper = bound.upper, method = bound.method)
end

"""
    tmbprofile_wrapper(fit::GllvmFit, param_index::Integer; level=0.95,
                       grid_extent=5, max_expand=20, max_bisect=30,
                       profile_iterations=200, profile_g_tol=1e-4,
                       y, X=nothing, Σ_phy=nothing)
        -> NamedTuple

Bare profile-CURVE variant of [`profile_ci`](@ref) — the Julia analogue of
R's `tmbprofile_wrapper()`/`TMB::tmbprofile()` pair. `profile_ci` already
reproduces the REDUCED bound R's function returns; this exposes the
underlying `(θ_i value, nll)` TRACE that bound is reduced from (see the
comment block above this method for the derivation).

Returns a NamedTuple:
  - `theta::Vector{Float64}` — evaluated `θ_i` values, sorted ascending
    (including `θ̂_i` itself, whose `nll` equals `-fit.logLik`)
  - `nll::Vector{Float64}` — negative log-likelihood at each constrained
    refit, in the same order as `theta`
  - `estimate::Float64` — `θ̂_i` (the packed value at `param_index`, on the
    WORKING scale — no `exp` back-transform, unlike `profile_ci`'s `lower`/
    `upper`, since a raw NLL curve is inherently on the working scale)
  - `lower`, `upper`, `method` — identical to `profile_ci(fit, param_index;
    kwargs...)`, i.e. already back-transformed for `:log_sd` terms

`y` is required (the same data matrix passed to `fit_gaussian_gllvm`); `X`
and `Σ_phy` are required iff the fit used them. See [`profile_ci`](@ref)
for the full keyword contract these are forwarded to.
"""
function tmbprofile_wrapper(fit::GllvmFit, param_index::Integer;
                            level::Real = 0.95,
                            grid_extent::Real = 5,
                            max_expand::Integer = 20,
                            max_bisect::Integer = 30,
                            profile_iterations::Integer = 200,
                            profile_g_tol::Real = 1e-4,
                            y::Union{Nothing, AbstractMatrix} = nothing,
                            X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
                            Σ_phy::Union{Nothing, AbstractMatrix} = nothing)
    _has_lv_predictor(fit) && throw(ArgumentError(
        "tmbprofile_wrapper for fit_gaussian_gllvm(...; X_lv=...) is not admitted in the C1 predictor-informed latent-score path"))
    y === nothing && throw(ArgumentError(
        "tmbprofile_wrapper requires the data matrix `y` (the same matrix passed to fit_gaussian_gllvm)"))
    return _tmbprofile_curve(fit, param_index;
                             level = level, grid_extent = grid_extent,
                             max_expand = max_expand, max_bisect = max_bisect,
                             profile_iterations = profile_iterations,
                             profile_g_tol = profile_g_tol,
                             y = y, X = X, Σ_phy = Σ_phy)
end

"""
    tmbprofile_wrapper(fit::GllvmFit, parm::AbstractString; kwargs...) -> NamedTuple

Convenience method resolving `parm` by name (same convention as
[`profile_ci`](@ref), e.g. `"sigma_eps"`, `"Lambda_B[1,1]"`).
"""
function tmbprofile_wrapper(fit::GllvmFit, parm::AbstractString; kwargs...)
    idx = _profile_parm_index(fit, parm)
    return tmbprofile_wrapper(fit, idx; kwargs...)
end

tmbprofile_wrapper(fit::GllvmFit, parm::Symbol; kwargs...) =
    tmbprofile_wrapper(fit, String(parm); kwargs...)

"""
    profile_curve_targets(fit::GllvmFit, targets=nothing; kwargs...)
        -> Dict{String, NamedTuple}

Batch curve wrapper: calls [`tmbprofile_wrapper`](@ref) for every entry of
`targets` (a vector of packed-parameter names or integer indices; default
`nothing` profiles EVERY term in `confint(fit).term` order), returning a
`Dict` keyed by term name.

R's `profile_targets()` is a READINESS REGISTRY (which parameters/targets a
fit COULD be profiled on, without running anything, because
`TMB::tmbprofile()` needs the fit's live `tmb_obj` checkpoint machinery to
run cheaply and reversibly). GLLVModels.jl's `profile_ci` has no comparable
checkpoint/restore step — running it IS the cheap operation — so this
function runs every target's curve directly rather than reporting a
separate readiness flag; a fit for which a given `parm` cannot be resolved
(see [`profile_ci`](@ref)'s parm-name resolution) raises the same
`ArgumentError` that resolution already raises elsewhere in this file,
which is the honest failure mode.

`kwargs` (in particular `y`, and `X`/`Σ_phy` if the fit used them) are
forwarded verbatim to every `tmbprofile_wrapper` call.
"""
function profile_curve_targets(fit::GllvmFit, targets::Union{Nothing, AbstractVector} = nothing;
                         kwargs...)
    terms, _ = _profile_all_term_names(fit)
    targs = targets === nothing ? collect(1:length(terms)) : targets
    out = Dict{String, NamedTuple}()
    for t in targs
        name = t isa Integer ? terms[Int(t)] : String(t)
        out[name] = tmbprofile_wrapper(fit, t; kwargs...)
    end
    return out
end

"""
    profile_phylo_signal(fit::GllvmFit, t::Integer; kwargs...) -> NamedTuple

Bare profile-CURVE variant scoped to the per-trait phylogenetic-unique scale
`sigma_phy[t]` (a raw packed parameter on an identity, signed link — no `exp`
back-transform; present iff the fit used `has_phy_unique = true`).

`σ_phy` is a signed parameter (#136), so this curve can extend below zero.
When `K_phy = 0`, the sign of each group of rows that `Σ_phy` links can flip
on its own (for a tree-derived `Σ_phy`, each daughter clade of the root); when `K_phy ≥ 1`, it is not identified separately from
`Λ_phy` and a per-entry interval is not interpretable. It is not gllvmTMB's `phylo_unique` scale; see the gllvmTMB
parity page.

Honest scope note: this is NOT the composite phylogenetic-SIGNAL summary
`phylo_signal(fit)[t]` (an H²-like ratio of variance components) that
`profile_ci_phylo_signal` (`src/confint_derived.jl`) already profiles via
its own constrained-refit-WITH-PENALTY machinery on that nonlinear derived
quantity. Building a curve variant for THAT quantity needs the identical
penalty-profile plumbing `confint_derived.jl` already owns; duplicating it
here would create two divergent implementations of the same profile, so
this function is deliberately scoped to the packed `sigma_phy[t]` term —
the raw SCALE parameter, not the derived SIGNAL ratio.
"""
function profile_phylo_signal(fit::GllvmFit, t::Integer; kwargs...)
    fit.model.has_phy_unique || throw(ArgumentError(
        "profile_phylo_signal requires a fit with has_phy_unique = true"))
    1 ≤ t ≤ fit.model.p ||
        throw(ArgumentError("t = $t out of range 1:$(fit.model.p)"))
    return tmbprofile_wrapper(fit, "sigma_phy[$t]"; kwargs...)
end

"""
    profile_targets(args...; kwargs...)

Deprecated forwarding shim: renamed to [`profile_curve_targets`](@ref). The
name `profile_targets` is reserved for a future R-style readiness registry,
which would be a different function.
"""
function profile_targets(args...; kwargs...)
    Base.depwarn("profile_targets is deprecated: renamed to profile_curve_targets; the name profile_targets is reserved for a future R-mirroring readiness-registry surface", :profile_targets)
    return profile_curve_targets(args...; kwargs...)
end
