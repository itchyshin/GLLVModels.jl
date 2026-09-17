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
    M = if M_user === nothing
        fill(NaN, n_traits, K)
    else
        _normalize_lambda_constraint_pin_matrix(M_user)
    end
    M[i, k] = Float64(c)
    return M
end

# Stage 1 bounded slice: ordinary unit-tier Gaussian J1 only (no W/diag/phylo blocks).
function _confirmatory_j1_fit_admitted(fit::GllvmFit)
    m = fit.model
    return m.K_W == 0 && !m.has_diag && m.K_phy == 0 && !m.has_phy_unique
end

# Index into `fit.pars.θ_packed` for raw-scale `Lambda_B[i,k]` naming (confint layout).
function _lambda_b_theta_index(fit::GllvmFit, i::Integer, k::Integer)
    return _profile_parm_index(fit, "Lambda_B[$i,$k]")
end

# Map a user pin matrix (raw Lambda scale, R convention) to fixed `(theta_index,
# working_value)` pairs for a single confirmatory profile grid point.
#
# Working values use the J1 packed scale `L = Lambda / sigma_eps` at the reference
# fit. Full R parity still requires fit-time `lambda_constraint` after the paste.
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
    σ_eps = fit.pars.σ_eps
    σ_eps > 0 && isfinite(σ_eps) ||
        throw(ArgumentError("fit.pars.σ_eps must be finite and positive"))
    fixes = Tuple{Int, Float64}[]
    for i in 1:p, k in 1:K
        v = M[i, k]
        v isa Real && isnan(v) && continue
        idx = _lambda_b_theta_index(fit, i, k)
        push!(fixes, (idx, Float64(v) / σ_eps))
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
