# D3 confirmatory `loading_profile` — internal Stage 1 plumbing (not exported).
#
# Stage 0 pin semantics live in test fixtures; this file moves the substrate
# into `src/` for fitter wiring without exporting `loading_profile`.
# Public export + ledger rebind still require maintainer paste `G0 Stage 1`.

const _D3_LOADING_PROFILE_STAGE1_PASTE_EXACT = "G0 Stage 1"

# True only when `ENV["GLLVM_STAGE1_PASTE"]` equals the maintainer paste exactly.
function _d3_loading_profile_stage1_paste_authorized()
    return get(ENV, "GLLVM_STAGE1_PASTE", "") == _D3_LOADING_PROFILE_STAGE1_PASTE_EXACT
end

function _d3_loading_profile_stage1_require_paste!()
    _d3_loading_profile_stage1_paste_authorized() ||
        throw(ArgumentError(
            "D3 confirmatory loading_profile engine slice requires maintainer paste " *
            repr(_D3_LOADING_PROFILE_STAGE1_PASTE_EXACT) *
            " in ENV[\"GLLVM_STAGE1_PASTE\"]"))
    return nothing
end

# Match `gllvmTMB` `loading-profile.R`: user `NaN`/`missing` = free, numeric = pinned,
# plus engine strict-upper-triangle pins for `i <= min(n_traits, K)` and `k > i`.
function _lambda_constraint_is_pinned(
    M_user::Union{Nothing, AbstractMatrix{<:Real}},
    n_traits::Int,
    K::Int,
)
    n_traits > 0 && K > 0 ||
        throw(ArgumentError("n_traits and K must be positive"))
    is_pinned = falses(n_traits, K)
    if M_user !== nothing
        size(M_user) == (n_traits, K) ||
            throw(ArgumentError("pin matrix must be $(n_traits)×$(K)"))
        for i in 1:n_traits, k in 1:K
            v = M_user[i, k]
            is_pinned[i, k] = !(v isa Real && isnan(v))
        end
    end
    for i in 1:min(n_traits, K), k in 1:K
        k > i && (is_pinned[i, k] = true)
    end
    return is_pinned
end

# Drop above-diagonal user entries in the first `min(p,K)` rows (R ignores them).
# Free slots stay `NaN`; fixed slots keep their numeric value.
function _normalize_lambda_constraint_pin_matrix(M::AbstractMatrix{<:Real})
    p, K = size(M)
    out = Matrix{Float64}(undef, p, K)
    for i in 1:p, k in 1:K
        if i <= min(p, K) && k > i
            out[i, k] = NaN
        else
            v = M[i, k]
            out[i, k] = v isa Real && isnan(v) ? NaN : Float64(v)
        end
    end
    return out
end

# Return `Vector{Tuple{Int,Int}}` of `(trait, axis)` pairs R would profile.
function _enumerate_free_lambda_entries(
    M_user::Union{Nothing, AbstractMatrix{<:Real}},
    n_traits::Int,
    K::Int;
    entries::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
)
    is_pinned = _lambda_constraint_is_pinned(M_user, n_traits, K)
    if entries === nothing
        free = Tuple{Int, Int}[]
        for i in 1:n_traits, k in 1:K
            !is_pinned[i, k] && push!(free, (i, k))
        end
        return free
    end
    size(entries, 2) >= 2 ||
        throw(ArgumentError("entries must have at least two columns (i, k)"))
    out = Tuple{Int, Int}[]
    for row in 1:size(entries, 1)
        i = Int(entries[row, 1])
        k = Int(entries[row, 2])
        (1 <= i <= n_traits && 1 <= k <= K) ||
            throw(ArgumentError("entry ($i, $k) out of range for $(n_traits)×$(K) Lambda"))
        is_pinned[i, k] &&
            throw(ArgumentError("entry ($i, $k) is pinned or structurally zero"))
        push!(out, (i, k))
    end
    return out
end

# Pin matrix for one profile grid point: preserve other user pins, set `(i,k)` to `c`.
function _profile_refit_lambda_constraint(
    M_user::Union{Nothing, AbstractMatrix{<:Real}},
    n_traits::Int,
    K::Int,
    i::Integer,
    k::Integer,
    c::Real,
)
    if M_user !== nothing
        size(M_user) == (n_traits, K) ||
            throw(ArgumentError(
                "pin matrix is $(size(M_user, 1))×$(size(M_user, 2)); expected " *
                "$(n_traits)×$(K) to match the fit"))
    end
    M = if M_user === nothing
        fill(NaN, n_traits, K)
    else
        _normalize_lambda_constraint_pin_matrix(M_user)
    end
    M[i, k] = Float64(c)
    return M
end

# Stage 1 bounded slice: ordinary unit-tier Gaussian J1 only. Refuses (Gauss,
# 2026-09-24): W/diag/phylo blocks; predictor-informed `alpha_lv` (X_lv) fits,
# whose packed-θ layout `_profile_spec`/`_derived_unpack` does not cover —
# without this check the pin-index mapping targets the wrong θ_packed slot and
# the refit silently returns NaN; and AGHQ/masked/offset fits, which attach a
# non-`nothing` `fit.integration` and whose closed-form `gaussian_nll_packed`
# evaluation (what the refit actually re-optimises) does not reproduce the
# integrated/masked/offset objective the fit itself was estimated under.
# `X_lv` fits also always attach a non-`nothing` `fit.integration`, so the
# single `integration === nothing` check below covers AGHQ/masked/offset/X_lv
# together; the `alpha_lv === nothing` check is kept as an explicit,
# self-documenting second layer for the specific case Gauss named.
function _confirmatory_j1_fit_admitted(fit::GllvmFit)
    m = fit.model
    return m.K_W == 0 && !m.has_diag && m.K_phy == 0 && !m.has_phy_unique &&
        fit.integration === nothing && fit.pars.alpha_lv === nothing
end

# Index into `fit.pars.θ_packed` for raw-scale `Lambda_B[i,k]` naming (confint layout).
function _lambda_b_theta_index(fit::GllvmFit, i::Integer, k::Integer)
    return _profile_parm_index(fit, "Lambda_B[$i,$k]")
end

# Map a user pin matrix (raw Lambda scale, R convention) to fixed `(theta_index,
# working_value)` pairs for a single confirmatory profile grid point.
#
# `θ_packed`'s Lambda_B block is raw-scale: `gaussian_nll_packed` (src/likelihood.jl)
# unpacks it directly via `unpack_lambda` and passes it unchanged into
# `gaussian_marginal_loglik`, with no σ_eps rescaling anywhere on that path. The
# fixed working value is therefore the raw pin value unchanged (Gauss verdict,
# 2026-09-24: confirmed by an independent constrained-MLE oracle to 8 dp).
function _confirmatory_lambda_pin_theta_fixes(
    fit::GllvmFit,
    M_user::Union{Nothing, AbstractMatrix{<:Real}},
    profile_i::Integer,
    profile_k::Integer,
    profile_c::Real;
    component::Symbol = :B,
)
    component === :B ||
        throw(ArgumentError("Stage 1 internal pin path supports component = :B only"))
    _confirmatory_j1_fit_admitted(fit) ||
        throw(ArgumentError(
            "confirmatory lambda pin refit is J1-only in Stage 1 plumbing"))
    p = fit.model.p
    K = fit.model.K
    M = _profile_refit_lambda_constraint(M_user, p, K, profile_i, profile_k, profile_c)
    fixes = Tuple{Int, Float64}[]
    for i in 1:p, k in 1:K
        v = M[i, k]
        v isa Real && isnan(v) && continue
        idx = _lambda_b_theta_index(fit, i, k)
        push!(fixes, (idx, Float64(v)))
    end
    return fixes
end

# Fit-time counterpart of `_confirmatory_lambda_pin_theta_fixes`: map ALL of a
# user pin matrix's numeric (non-NaN) entries to fixed `(theta_index,
# working_value)` pairs, with no extra single-entry override. Used to build the
# confirmatory (pinned) fit itself, before any profiling grid runs. Raw-scale,
# no σ_eps rescaling — see the note above `_confirmatory_lambda_pin_theta_fixes`.
function _confirmatory_lambda_constraint_theta_fixes(
    fit::GllvmFit,
    M_user::AbstractMatrix{<:Real};
    component::Symbol = :B,
)
    component === :B ||
        throw(ArgumentError("Stage 1 internal pin path supports component = :B only"))
    _confirmatory_j1_fit_admitted(fit) ||
        throw(ArgumentError(
            "lambda_constraint fit-time pinning is J1-only in Stage 1 plumbing"))
    p = fit.model.p
    K = fit.model.K
    size(M_user) == (p, K) ||
        throw(ArgumentError(
            "pin matrix is $(size(M_user, 1))×$(size(M_user, 2)); expected " *
            "$(p)×$(K) to match the fit"))
    M = _normalize_lambda_constraint_pin_matrix(M_user)
    fixes = Tuple{Int, Float64}[]
    for i in 1:p, k in 1:K
        v = M[i, k]
        v isa Real && isnan(v) && continue
        idx = _lambda_b_theta_index(fit, i, k)
        push!(fixes, (idx, Float64(v)))
    end
    return fixes
end

# Re-optimise the Gaussian NLL over all parameters except those listed in `fixed`
# (each `(index, value)` holds one `theta_packed` entry at `value`).
function _profile_refit_with_multi_fixed(
    fit::GllvmFit,
    fixed::AbstractVector{Tuple{Int, Float64}},
    y::AbstractMatrix;
    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
    Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
    x_tol::Real = 1e-6,
    f_tol::Real = 1e-8,
    g_tol::Real = 1e-4,
    iterations::Integer = 200,
    θ_red_warm::Union{Nothing, AbstractVector} = nothing,
)
    spec = _profile_spec(fit)
    X_free = _profile_free_X(fit, X)
    θ̂ = fit.pars.θ_packed
    N = length(θ̂)
    fixed_dict = Dict{Int, Float64}(fixed)
    isempty(fixed_dict) &&
        throw(ArgumentError("fixed must contain at least one (index, value) pair"))
    for (idx, _) in fixed
        1 <= idx <= N ||
            throw(ArgumentError("fixed index $idx out of range 1:$N"))
    end
    fixed_idx = sort(collect(keys(fixed_dict)))
    free_idx = [j for j in 1:N if !haskey(fixed_dict, j)]

    θ_red0 = if θ_red_warm === nothing
        θ̂[free_idx]
    else
        collect(Float64, θ_red_warm)
    end
    length(θ_red0) == length(free_idx) ||
        throw(ArgumentError("θ_red_warm length must match free parameter count"))

    function _full_from_red(θ_red)
        T = eltype(θ_red)
        θ_full = Vector{T}(undef, N)
        @inbounds for j in 1:N
            if haskey(fixed_dict, j)
                θ_full[j] = convert(T, fixed_dict[j])
            end
        end
        @inbounds for (r, j) in enumerate(free_idx)
            θ_full[j] = θ_red[r]
        end
        return θ_full
    end

    nll_red = θ_red -> gaussian_nll_packed(_full_from_red(θ_red), y;
        spec = spec, X = X_free, Σ_phy = Σ_phy)

    opts = Optim.Options(
        x_abstol = x_tol,
        f_reltol = f_tol,
        g_tol = g_tol,
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

# One confirmatory grid-point refit (J1 Gaussian). When `require_paste` (default),
# calls `_d3_loading_profile_stage1_require_paste!` before running the optimiser.
function _confirmatory_profile_refit_lambda_pin(
    fit::GllvmFit,
    y::AbstractMatrix,
    M_user::Union{Nothing, AbstractMatrix{<:Real}},
    profile_i::Integer,
    profile_k::Integer,
    profile_c::Real;
    require_paste::Bool = true,
    X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
    Σ_phy::Union{Nothing, AbstractMatrix} = nothing,
    kwargs...,
)
    require_paste && _d3_loading_profile_stage1_require_paste!()
    fixes = _confirmatory_lambda_pin_theta_fixes(
        fit, M_user, profile_i, profile_k, profile_c)
    ll, ok, _ = _profile_refit_with_multi_fixed(
        fit, fixes, y; X = X, Σ_phy = Σ_phy, kwargs...)
    return (ll, ok)
end

# Stage 1 fit-time `lambda_constraint`: given an already-fitted ordinary J1
# Gaussian `base` (X = nothing, no W/diag/phylo blocks), hold the numeric
# (non-NaN) entries of `M_user` fixed at their raw-scale value and re-optimise
# every other free parameter. Returns a new `GllvmFit` whose `pars` carries
# `lambda_constraint = M_user` (normalised) so `loading_profile` can read back
# which entries are free vs pinned. Structural upper-triangle zeros (`k > i`
# for `i <= min(p, K)`) are already enforced by the lower-triangular packing
# convention on every ordinary fit and need no extra fixing.
#
# X = nothing only: the packed-θ layout this reuses (`_profile_spec` /
# `_derived_unpack`) does not carry `alpha_lv`, and Stage 1 scope (Stage 0
# fixtures) has no `X` design — see the runbook's bounded-slice fence.
function _fit_confirmatory_lambda_constraint(
    base::GllvmFit,
    Y::AbstractMatrix,
    M_user::AbstractMatrix{<:Real};
    kwargs...,
)
    _confirmatory_j1_fit_admitted(base) ||
        throw(ArgumentError(
            "lambda_constraint is supported for the ordinary J1 Gaussian fitter " *
            "only (K_W = 0, has_diag = false, K_phy = 0, has_phy_unique = false)"))
    (base.pars.β === nothing || isempty(base.pars.β)) ||
        throw(ArgumentError(
            "lambda_constraint currently supports X = nothing (zero-mean) fits only"))
    p = base.model.p
    K = base.model.K
    size(M_user) == (p, K) ||
        throw(ArgumentError(
            "lambda_constraint must be $(p)×$(K) (p traits × K axes); got $(size(M_user))"))
    M_norm = _normalize_lambda_constraint_pin_matrix(M_user)
    fixes = _confirmatory_lambda_constraint_theta_fixes(base, M_user)
    if isempty(fixes)
        # No additional user pins beyond the engine's own structural zeros:
        # numerically identical to `base`, only the pin metadata is new.
        pars = merge(base.pars, (lambda_constraint = M_norm,))
        return GllvmFit(base.model, pars, base.logLik, base.n_iter, base.converged,
                         base.optim_result, base.cputime, base.integration)
    end
    ll, ok, θ_red = _profile_refit_with_multi_fixed(base, fixes, Y; kwargs...)
    ok || throw(ArgumentError(
        "confirmatory lambda_constraint refit did not converge"))
    N = length(base.pars.θ_packed)
    fixed_dict = Dict{Int, Float64}(fixes)
    free_idx = [j for j in 1:N if !haskey(fixed_dict, j)]
    θ_full = Vector{Float64}(undef, N)
    for (idx, v) in fixes
        θ_full[idx] = v
    end
    for (r, j) in enumerate(free_idx)
        θ_full[j] = θ_red[r]
    end
    spec = _profile_spec(base)
    u = _derived_unpack(θ_full, spec)
    pars = merge(base.pars, (σ_eps = u.σ_eps, Λ = u.Λ_B, θ_packed = θ_full,
                              lambda_constraint = M_norm))
    return GllvmFit(base.model, pars, ll, base.n_iter, ok, base.optim_result,
                     base.cputime, base.integration)
end
