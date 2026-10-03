# Final missing-surface cluster (core070 §1) — small post-fit tables and
# helpers that did not have a natural home in postfit.jl / extractors.jl.
# Each function documents the R contract it mirrors (file:line in the frozen
# oracle readback) and any deliberate scope reduction.
#
# Item 1.1 — deviance (mirrors methods-gllvmTMB.R:2862-2864).

"""
    deviance(fit) -> Float64

`-2 * loglikelihood(fit)`. Mirrors `gllvmTMB`'s `deviance.gllvmTMB_multi`
(`methods-gllvmTMB.R:2862-2864`), which is exactly `-2 * logLik(object)` — no
null/saturated-model comparison, delegating through the same maximised
marginal log-likelihood that a ridged fit reports (R's "unpenalised logLik at
penalised MAP" caveat is inherited unchanged here).

This calculation does not use [`nobs`](@ref). When comparing [`bic`](@ref)
across R and Julia, check the observation-count convention: site counts and
counts of likelihood-contributing cells can differ.
"""
StatsAPI.deviance(fit::AnyGllvmFit) = -2 * StatsAPI.loglikelihood(fit)

# ---------------------------------------------------------------------------
# Item 1.2 — profile_cross_rho_ci (mirrors kernel-helpers.R:294-359; the
# post-verification correction narrows the citation range — :361-372 are
# unrelated cross-kernel helpers).
# ---------------------------------------------------------------------------

# Linear-interpolation crossing of the line through (x1,y1)-(x2,y2) at y=threshold.
function _cross_rho_interp(x1::Real, y1::Real, x2::Real, y2::Real, threshold::Real)
    return x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
end

# Walk the (sorted-by-rho) grid away from `best_i` in direction `dir` (:down
# decreasing index / :up increasing index) looking for the first point whose
# delta_deviance crosses `threshold`. Returns (bound, bounded::Bool).
function _cross_rho_bracket(r::AbstractVector, dd::AbstractVector, best_i::Integer,
                            threshold::Real, dir::Symbol)
    n = length(r)
    if dir === :down
        j = best_i - 1
        while j >= 1
            if dd[j] >= threshold
                return _cross_rho_interp(r[j], dd[j], r[j + 1], dd[j + 1], threshold), true
            end
            j -= 1
        end
        return r[1], false
    else
        j = best_i + 1
        while j <= n
            if dd[j] >= threshold
                return _cross_rho_interp(r[j - 1], dd[j - 1], r[j], dd[j], threshold), true
            end
            j += 1
        end
        return r[n], false
    end
end

"""
    profile_cross_rho_ci(rho, delta_deviance; level = 0.95) -> NamedTuple

Confidence interval for the cross-lineage coevolution kernel parameter `rho`
by inverting a `delta_deviance` profile TABLE (as produced by
[`profile_cross_rho`](@ref)) — pure post-processing, no refitting. Mirrors
`gllvmTMB::profile_cross_rho_ci` (`kernel-helpers.R:294-359`).

`rho` and `delta_deviance` are equal-length vectors; non-finite pairs are
dropped, and at least 2 finite points must remain. `level ∈ (0, 1)`. The
threshold is `quantile(Chisq(1), level)` (the standard 1-df LRT cutoff). The
best (`estimate`) point is the grid value with the smallest `delta_deviance`.
Bounds are found by walking outward from the best point and LINEARLY
INTERPOLATING between the two grid points that bracket the threshold
crossing (not a quadratic/bisection refit — this is grid interpolation on an
already-computed table). A side that never crosses the threshold within the
grid returns that side's grid edge with the corresponding `*_bounded = false`
flag. Bounds are clamped to `[-1, 1]` (rho's admissible range).

Returns `(estimate, lower, upper, level, lower_bounded, upper_bounded,
threshold)`.
"""
function profile_cross_rho_ci(rho::AbstractVector{<:Real}, delta_deviance::AbstractVector{<:Real};
                              level::Real = 0.95)
    length(rho) == length(delta_deviance) ||
        throw(ArgumentError("rho and delta_deviance must have equal length."))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1); got $level"))

    keep = findall(i -> isfinite(rho[i]) && isfinite(delta_deviance[i]), eachindex(rho))
    length(keep) >= 2 ||
        throw(ArgumentError("profile_cross_rho_ci needs >= 2 finite (rho, delta_deviance) points; got $(length(keep))."))

    perm = sortperm(rho[keep])
    r = Float64.(rho[keep][perm])
    dd = Float64.(delta_deviance[keep][perm])

    threshold = quantile(Chisq(1), level)
    best_i = argmin(dd)
    estimate = r[best_i]

    lower, lower_bounded = _cross_rho_bracket(r, dd, best_i, threshold, :down)
    upper, upper_bounded = _cross_rho_bracket(r, dd, best_i, threshold, :up)
    lower = clamp(lower, -1.0, 1.0)
    upper = clamp(upper, -1.0, 1.0)

    return (estimate = estimate, lower = lower, upper = upper, level = level,
            lower_bounded = lower_bounded, upper_bounded = upper_bounded,
            threshold = threshold)
end

# ---------------------------------------------------------------------------
# Item 1.3 — predict_cross_covariance, positional form (mirrors
# extract-sigma.R:1837-1957; kernel from `.kernel_level_matrix`, :1961-2005).
# ---------------------------------------------------------------------------

"""
    predict_cross_covariance(fit, K; row_levels, col_levels, row_traits, col_traits)
        -> NamedTuple of vectors

Cross-lineage predicted covariance table `covariance = kernel_value * gamma_shape`
for every combination of `(row_level, col_level, row_trait, col_trait)`, one row
per combination (nested loop order: `row_levels` outer, `col_levels`, `row_traits`,
`col_traits` inner). Mirrors `gllvmTMB::predict_cross_covariance`
(`extract-sigma.R:1837-1957`), point estimates only.

`K` is the `n × n` cross-lineage kernel (e.g. from [`make_cross_kernel`](@ref));
`row_levels`/`col_levels` are 1-based integer indices into `K`. `gamma_shape`
is `Γ = (Λ_phy Λ_phyᵀ)[row_traits, col_traits]` via [`extract_Gamma`](@ref)
(already on R's "shape" scale) — `row_traits`/`col_traits` are 1-based integer
indices into the stacked two-lineage entity set that `fit` was fitted on.

Returns a `NamedTuple` of equal-length vectors with fields
`row_level, col_level, row_trait, col_trait, kernel_value, gamma_shape,
covariance`, corresponding to the rows of R's table. R's `rho` and
`kernel_includes_rho` metadata columns are not returned: `K` is an unnamed
matrix with no stored `rho`.

Throws `ArgumentError` if any level index falls outside `axes(K)` (mirroring
R's abort at `extract-sigma.R:1885-1908`).
"""
function predict_cross_covariance(fit::GllvmFit, K::AbstractMatrix;
                                  row_levels::AbstractVector{<:Integer},
                                  col_levels::AbstractVector{<:Integer},
                                  row_traits::AbstractVector{<:Integer},
                                  col_traits::AbstractVector{<:Integer})
    nK1, nK2 = size(K)
    all(l -> 1 <= l <= nK1, row_levels) ||
        throw(ArgumentError("row_levels must index 1:$nK1 (rows of K)."))
    all(l -> 1 <= l <= nK2, col_levels) ||
        throw(ArgumentError("col_levels must index 1:$nK2 (columns of K)."))

    Γ = extract_Gamma(fit; row_traits = row_traits, col_traits = col_traits)

    n = length(row_levels) * length(col_levels) * length(row_traits) * length(col_traits)
    row_level = Vector{Int}(undef, n)
    col_level = Vector{Int}(undef, n)
    row_trait = Vector{Int}(undef, n)
    col_trait = Vector{Int}(undef, n)
    kernel_value = Vector{Float64}(undef, n)
    gamma_shape = Vector{Float64}(undef, n)
    covariance = Vector{Float64}(undef, n)

    idx = 0
    for rl in row_levels, cl in col_levels, (ti, rt) in enumerate(row_traits), (tj, ct) in enumerate(col_traits)
        idx += 1
        kv = Float64(K[rl, cl])
        gs = Γ[ti, tj]
        row_level[idx] = rl
        col_level[idx] = cl
        row_trait[idx] = rt
        col_trait[idx] = ct
        kernel_value[idx] = kv
        gamma_shape[idx] = gs
        covariance[idx] = kv * gs
    end

    return (row_level = row_level, col_level = col_level,
            row_trait = row_trait, col_trait = col_trait,
            kernel_value = kernel_value, gamma_shape = gamma_shape,
            covariance = covariance)
end

# ---------------------------------------------------------------------------
# Item 1.4 — predict_missing, explicit-args form (mirrors
# methods-gllvmTMB.R:3948-4085; predict(object, type) restricted to masked
# rows).
# ---------------------------------------------------------------------------

"""
    predict_missing(fit, Y; mask = nothing, type = :link) -> NamedTuple

Predicted values at the MASKED (unobserved) cells only. Mirrors
`gllvmTMB::predict.gllvmTMB_multi` restricted to `which(is_y_observed == 0)`
(`methods-gllvmTMB.R:3948-4085`).

`mask` is a `p × n` `Bool` matrix (`true` = observed, `false` = masked —
`GLLVModels.jl`'s existing mask convention, `src/families/laplace.jl:96-133`);
`mask = nothing` (default) means every cell is observed, giving a zero-row
result (R's complete-data behaviour). `type` is forwarded to `predict`
(`:link` or `:response`; `src/postfit.jl:174-242`). `fit` (and hence
`predict`) must support the `mask` keyword for the masked call to succeed —
currently the AGHQ Gaussian route and the dense-Laplace non-Gaussian
families (e.g. Binomial). GLLVModels.jl's `GllvmFit`/`Y` do not store their own
mask, so the caller must supply it here. The single-argument form
`predict_missing(fit)` is not supported.

Returns `(row, col, est)`: `row`/`col` are the 1-based `(trait, site)`
indices of each masked cell (findall order — column-major, i.e. `row` varies
fastest, matching Julia's native `p × n` layout rather than R's long-format
`original_row`/`model_row` columns), and `est` is `predict(fit, Y; type,
mask)` at that cell.

Deviation from R: the `ordinal_probit` `type = "response"` expected-category
replacement (`methods-gllvmTMB.R:4076-4083`) and the experimental `se=`
routes are not currently available.
"""
function predict_missing(fit, Y::AbstractMatrix;
                         mask::Union{Nothing, AbstractMatrix{Bool}} = nothing,
                         type::Symbol = :link)
    p, n = size(Y)
    m = mask === nothing ? trues(p, n) : mask
    size(m) == (p, n) ||
        throw(ArgumentError("mask must be p×n = $(p)×$(n); got $(size(m))."))

    est_full = predict(fit, Y; type = type, mask = mask)

    idx = findall(!, m)
    row = [I[1] for I in idx]
    col = [I[2] for I in idx]
    est = [est_full[I] for I in idx]
    return (row = row, col = col, est = est)
end

# ---------------------------------------------------------------------------
# Item 1.5 — simulate_unit_trait (mirrors simulate-unit-trait.R:78-164).
#
# Ownership note: the spec's landing spot is `src/simulate.jl` (the
# designated placeholder for Julia-side simulators); this slice is confined
# to `src/postfit_tables.jl` by lane ownership (see
# docs/dev-log/core070/final-surface-slice-notes.md), so it lands here
# instead — a location deviation, not a scope reduction.
# ---------------------------------------------------------------------------

"""
    simulate_unit_trait([rng]; n_units=50, n_obs_per_unit=3, n_traits=5,
                        K_B=1, K_W=1, alpha=zeros(n_traits),
                        Lambda_B=nothing, Lambda_W=nothing,
                        psi_B=fill(0.5, n_traits), psi_W=fill(0.5, n_traits),
                        sigma2_eps=0.5)
        -> (Y, individual, truth)

Balanced two-level Gaussian DGP, mirroring `gllvmTMB::simulate_unit_trait`
(`simulate-unit-trait.R:78-164`):
`y_uot = α_t + b_ut(Λ_B z + ψ_B) + w_uot(Λ_W z + ψ_W) + N(0, σ²_eps)`, i.e.
`b_u ~ N(0, Σ_B)` (one draw per unit `u`, shared across that unit's
`n_obs_per_unit` observations) and `w_uo ~ N(0, Σ_W)` (one draw per
observation), with `Σ_B = Λ_B Λ_Bᵀ + diag(ψ_B)` and
`Σ_W = Λ_W Λ_Wᵀ + diag(ψ_W .+ σ²_eps)` — this is exactly what
[`fit_twolevel_gaussian`](@ref) fits (its `Σ_W` has no separate
measurement-error term, so `σ²_eps` is folded into `ψ_W`, matching R's own
"psi recycling reflected in truth" note).

`Lambda_B`/`Lambda_W` default to `0.7 .* randn(rng, n_traits, K_B)` /
`0.5 .* randn(rng, n_traits, K_W)` when not supplied.

Output uses a matrix and `NamedTuple` rather than R's long-format
`(data, truth)` data frame:
returns `(Y, individual, truth)` — `Y` is `n_traits × (n_units*n_obs_per_unit)`,
`individual` is the length-matching grouping vector `fit_twolevel_gaussian`
expects, and `truth` is a `NamedTuple` with fields `alpha, Lambda_B, Lambda_W,
psi_B, psi_W, sigma2_eps, Sigma_B, Sigma_W`.
"""
function simulate_unit_trait(rng::Random.AbstractRNG = Random.default_rng();
                             n_units::Integer = 50, n_obs_per_unit::Integer = 3,
                             n_traits::Integer = 5, K_B::Integer = 1, K_W::Integer = 1,
                             alpha::AbstractVector{<:Real} = zeros(n_traits),
                             Lambda_B::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                             Lambda_W::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                             psi_B::AbstractVector{<:Real} = fill(0.5, n_traits),
                             psi_W::AbstractVector{<:Real} = fill(0.5, n_traits),
                             sigma2_eps::Real = 0.5)
    n_units >= 1 && n_obs_per_unit >= 1 ||
        throw(ArgumentError("n_units and n_obs_per_unit must be ≥ 1."))
    length(alpha) == n_traits ||
        throw(ArgumentError("alpha must have length n_traits = $n_traits."))
    length(psi_B) == n_traits && length(psi_W) == n_traits ||
        throw(ArgumentError("psi_B and psi_W must have length n_traits = $n_traits."))
    sigma2_eps >= 0 || throw(ArgumentError("sigma2_eps must be ≥ 0."))

    ΛB = Lambda_B === nothing ? 0.7 .* randn(rng, n_traits, K_B) : Matrix{Float64}(Lambda_B)
    ΛW = Lambda_W === nothing ? 0.5 .* randn(rng, n_traits, K_W) : Matrix{Float64}(Lambda_W)
    size(ΛB) == (n_traits, K_B) || throw(ArgumentError("Lambda_B must be n_traits × K_B."))
    size(ΛW) == (n_traits, K_W) || throw(ArgumentError("Lambda_W must be n_traits × K_W."))

    Σ_B = ΛB * ΛB' + Diagonal(collect(Float64, psi_B))
    ψ_W_total = collect(Float64, psi_W) .+ sigma2_eps
    Σ_W = ΛW * ΛW' + Diagonal(ψ_W_total)

    LB = _psd_sqrt_factor(Σ_B)
    LW = _psd_sqrt_factor(Σ_W)

    n_obs = n_units * n_obs_per_unit
    Y = Matrix{Float64}(undef, n_traits, n_obs)
    individual = Vector{Int}(undef, n_obs)
    col = 0
    for u in 1:n_units
        b_u = LB * randn(rng, n_traits)
        for _ in 1:n_obs_per_unit
            col += 1
            individual[col] = u
            Y[:, col] = alpha .+ b_u .+ LW * randn(rng, n_traits)
        end
    end

    truth = (alpha = collect(Float64, alpha), Lambda_B = ΛB, Lambda_W = ΛW,
            psi_B = collect(Float64, psi_B), psi_W = ψ_W_total,
            sigma2_eps = Float64(sigma2_eps), Sigma_B = Matrix(Σ_B), Sigma_W = Matrix(Σ_W))
    return (Y = Y, individual = individual, truth = truth)
end

# ---------------------------------------------------------------------------
# Item 1.6 — profile_cross_rho (mirrors kernel-helpers.R:166-262): a
# fixed-kernel sensitivity driver over the cross-lineage rho, NOT a TMB
# parameter profile. Unblocks §1.2's integration test case.
# ---------------------------------------------------------------------------

# Best-effort duck-typed field lookup on the caller's refit return value.
function _cross_rho_field(f, names::Tuple{Vararg{Symbol}}, default)
    for nm in names
        hasproperty(f, nm) && return getproperty(f, nm)
    end
    return default
end

"""
    profile_cross_rho(A_H, A_P, W, refit; rho_grid = range(-0.9, 0.9; length=19),
                      eps = 1e-8, metrics = nothing, keep_fits = false)
        -> NamedTuple

Fixed-kernel sensitivity driver over the cross-LINEAGE coevolution kernel
parameter `rho` — mirrors `gllvmTMB::profile_cross_rho`
(`kernel-helpers.R:166-262`). This is NOT a TMB profile likelihood of a
fitted parameter; it re-derives `K = make_cross_kernel(A_H, A_P, W; rho, eps)`
at each grid value and calls the caller-supplied `refit(K, rho)` in a
`try`/`catch` (so one failing grid point does not abort the sweep). `refit`
targets e.g. [`fit_coevolution_gaussian`](@ref) / `fit_coevolution_blockna`,
which take `K_star` directly.

`rho_grid` entries must be finite and in `[-1, 1]`. `metrics`, when given, is
a function `f -> NamedTuple` called on each successful refit result and its
fields are appended as extra table columns (`missing` on rows that errored
or were never reached). `keep_fits::Bool` additionally returns the raw
refit results.

The refit result's fields are read by best-effort duck typing:
log-likelihood from `:logLik` or `:loglik`; convergence from `:converged`
(defaults to `true` when absent); `pd_hessian` from `:pd_hessian` (defaults
to `true` when absent — most refit targets here have no Hessian check yet).

Returns `(table, best_rho, fits)`, `fits === nothing` unless `keep_fits`.
`table` is a `NamedTuple` of equal-length vectors with columns `rho, logLik,
relative_logLik, delta_deviance, is_best, convergence, pd_hessian, status,
error` (+ any `metrics` columns) — `relative_logLik = logLik - max(logLik)`,
`delta_deviance = 2*(max(logLik) - logLik)`, `is_best` flags the maximiser,
`status` is `:ok`/`:error`, `error` the caught exception message (empty
string on success).
"""
function profile_cross_rho(A_H, A_P, W, refit;
                           rho_grid::AbstractVector{<:Real} = range(-0.9, 0.9; length = 19),
                           eps::Real = 1e-8,
                           metrics::Union{Nothing, Function} = nothing,
                           keep_fits::Bool = false)
    applicable(refit, A_H, 0.0) || throw(ArgumentError("refit must be callable as refit(K, rho)."))
    all(r -> isfinite(r) && abs(r) <= 1, rho_grid) ||
        throw(ArgumentError("every rho_grid entry must be finite and in [-1, 1]."))

    n = length(rho_grid)
    rho = Float64.(collect(rho_grid))
    logLik = fill(NaN, n)
    convergence = falses(n)
    pd_hessian = falses(n)
    status = Vector{Symbol}(undef, n)
    errmsg = fill("", n)
    fits = Vector{Any}(undef, n)
    metric_cols = Dict{Symbol, Vector{Any}}()

    for i in 1:n
        try
            K = make_cross_kernel(A_H, A_P, W; rho = rho[i], eps = eps)
            f = refit(K, rho[i])
            fits[i] = f
            logLik[i] = Float64(_cross_rho_field(f, (:logLik, :loglik), NaN))
            convergence[i] = Bool(_cross_rho_field(f, (:converged, :convergence), true))
            pd_hessian[i] = Bool(_cross_rho_field(f, (:pd_hessian,), true))
            status[i] = :ok
            if metrics !== nothing
                m = metrics(f)
                for k in propertynames(m)
                    col = get!(() -> fill(missing, n), metric_cols, k)
                    col[i] = getproperty(m, k)
                end
            end
        catch e
            fits[i] = nothing
            status[i] = :error
            errmsg[i] = sprint(showerror, e)
        end
    end

    finite_ll = findall(isfinite, logLik)
    isempty(finite_ll) &&
        throw(ArgumentError("every refit failed or returned a non-finite logLik; see the `error` column."))
    maxll = maximum(view(logLik, finite_ll))
    relative_logLik = logLik .- maxll
    delta_deviance = 2 .* (maxll .- logLik)
    is_best = falses(n)
    is_best[finite_ll[argmax(view(logLik, finite_ll))]] = true
    best_rho = rho[findfirst(is_best)]

    table = (rho = rho, logLik = logLik, relative_logLik = relative_logLik,
            delta_deviance = delta_deviance, is_best = is_best,
            convergence = convergence, pd_hessian = pd_hessian,
            status = status, error = errmsg)
    for k in keys(metric_cols)
        table = merge(table, NamedTuple{(k,)}((metric_cols[k],)))
    end

    return (table = table, best_rho = best_rho, fits = keep_fits ? fits : nothing)
end

# ---------------------------------------------------------------------------
# Item 1.8 — rotate_loadings (mirrors rotate-loadings.R:90-217).
#
# Scope reduction: `GllvmFit` / `level = :unit` only (the R contract's
# `TwoLevelFit` :unit/:unit_obs level mapping is not built in this slice —
# an honest omission, not a stub). Leaves the existing fixed-canonical
# `_svd_rotation`/`rotation`/`getLoadings(rotate=true)` convention untouched.
# ---------------------------------------------------------------------------

# Kaiser-normalized orthogonal varimax (standard SVD-iteration algorithm).
# Returns (Λ_rotated, T) with T orthogonal, T'T ≈ I.
function _varimax_rotation(Λ::AbstractMatrix; normalize::Bool = true,
                           maxit::Integer = 1000, tol::Real = 1e-6)
    p, k = size(Λ)
    if k <= 1
        return Matrix{Float64}(Λ), Matrix{Float64}(I, k, k)
    end
    h = normalize ? vec(sqrt.(sum(abs2, Λ; dims = 2))) : ones(p)
    h = max.(h, 1e-12)
    Λn = Λ ./ h
    T = Matrix{Float64}(I, k, k)
    d_prev = 0.0
    for _ in 1:maxit
        Λ2 = Λn * T
        colsq = vec(sum(abs2, Λ2; dims = 1)) ./ p
        u = Λ2 .^ 3 .- Λ2 * Diagonal(colsq)
        M = Λn' * u
        F = svd(M)
        T = F.U * F.Vt
        d_new = sum(F.S)
        (d_prev != 0 && d_new < d_prev * (1 + tol)) && break
        d_prev = d_new
    end
    Λrot = normalize ? (Λn * T) .* h : Λ * T
    return Λrot, T
end

# Oblique promax (Hendrickson & White 1964) on top of a varimax start.
# Returns (Λ_rotated, T) with T generally NOT orthogonal.
function _promax_rotation(Λ::AbstractMatrix; m::Real = 4)
    Λv, Tv = _varimax_rotation(Λ)
    k = size(Λ, 2)
    k <= 1 && return Λv, Tv
    Q = Λv .* abs.(Λv) .^ (m - 1)
    U = (Λv' * Λv) \ (Λv' * Q)
    dscale = 1.0 ./ diag(U' * U)
    U = U * Diagonal(sqrt.(dscale))
    T = Tv * U
    return Λ * T, T
end

"""
    rotate_loadings(fit::GllvmFit, Y; level = :unit, method = :varimax,
                    order_axes = true, sign_anchor = :auto,
                    anchor_traits = nothing) -> NamedTuple

Rotate a fitted GLLVM's loadings, mirroring `gllvmTMB::rotate_loadings`
(`rotate-loadings.R:90-217`). `method`: `:varimax` (orthogonal, Kaiser-
normalized), `:promax` (oblique, `m = 4`), or `:none` (identity — also the
automatic short-circuit at `d = 1`). `order_axes = true` reorders axes by
decreasing loading sum-of-squares (`colSums(Λ²)`, computed on the raw
rotated `Λ` before any standardization); `sign_anchor = :auto` flips each
axis's sign so its anchor trait (the largest-`|loading|` trait, or
`anchor_traits[k]` when supplied) loads positive; `sign_anchor = :none`
skips sign-fixing. Permutation and sign are folded into the returned `T` so
`Λ_rotated ≈ Λ * T` and, for orthogonal `T`, `scores ≈ getLV(fit, Y) * T`
(`scores ≈ getLV(fit, Y) * inv(T)'` for the oblique `:promax` case).

Current scope: `fit::GllvmFit` at `level = :unit` only — the
R contract's `TwoLevelFit`-level mapping (`:unit`/`:unit_obs`) is not built.
`Y` must match what was passed to the fit (as everywhere else in
`GLLVModels.jl` — the fit does not store its data).

Returns `(Lambda, scores, T, method, axis_variance, axis_order, axis_sign,
anchor_traits)`: `axis_variance` is `colSums(Λ_rotated²)` in the RETURNED
axis order; `axis_order` is the permutation applied (`1:d` when
`order_axes = false`); `axis_sign` is the `±1` applied per axis;
`anchor_traits` is the resolved per-axis anchor trait index (`missing` when
`sign_anchor = :none`).
"""
function rotate_loadings(fit::GllvmFit, Y::AbstractMatrix;
                         level::Symbol = :unit, method::Symbol = :varimax,
                         order_axes::Bool = true, sign_anchor::Symbol = :auto,
                         anchor_traits::Union{Nothing, AbstractVector{<:Integer}} = nothing)
    level === :unit || throw(ArgumentError(
        "rotate_loadings currently supports level = :unit only for GllvmFit " *
        "(TwoLevelFit level mapping is not built — core070 spec §1.8 scope note)."))
    method in (:varimax, :promax, :none) ||
        throw(ArgumentError("method must be :varimax, :promax, or :none; got :$method"))
    sign_anchor in (:auto, :none) ||
        throw(ArgumentError("sign_anchor must be :auto or :none; got :$sign_anchor"))

    Λ = Matrix{Float64}(fit.pars.Λ)
    p, d = size(Λ)
    S = getLV(fit, Y; rotate = false)   # n×d

    if anchor_traits !== nothing
        length(anchor_traits) == d ||
            throw(ArgumentError("anchor_traits must have length d = $d."))
        all(t -> 1 <= t <= p, anchor_traits) ||
            throw(ArgumentError("anchor_traits must index 1:$p."))
    end

    if method === :none || d <= 1
        Λrot = copy(Λ)
        T = Matrix{Float64}(I, d, d)
        oblique = false
    elseif method === :varimax
        Λrot, T = _varimax_rotation(Λ)
        oblique = false
    else # :promax
        Λrot, T = _promax_rotation(Λ)
        oblique = true
    end
    scores = oblique ? S * inv(T)' : S * T

    axis_order = collect(1:d)
    if order_axes && d > 1
        ss = vec(sum(abs2, Λrot; dims = 1))
        axis_order = sortperm(ss; rev = true)
        Λrot = Λrot[:, axis_order]
        scores = scores[:, axis_order]
        T = T[:, axis_order]
    end

    axis_sign = ones(Int, d)
    resolved_anchor_traits = sign_anchor === :none ? nothing : Vector{Int}(undef, d)
    if sign_anchor === :auto
        for k in 1:d
            # anchor_traits[k], when supplied, names the anchor for the k-th
            # RETURNED axis (i.e. after any order_axes permutation).
            at = anchor_traits === nothing ? argmax(abs.(view(Λrot, :, k))) : anchor_traits[k]
            resolved_anchor_traits[k] = at
            if Λrot[at, k] < 0
                axis_sign[k] = -1
                Λrot[:, k] .*= -1
                scores[:, k] .*= -1
                T[:, k] .*= -1
            end
        end
    end

    axis_variance = vec(sum(abs2, Λrot; dims = 1))
    return (Lambda = Λrot, scores = scores, T = T, method = method,
            axis_variance = axis_variance, axis_order = axis_order,
            axis_sign = axis_sign, anchor_traits = resolved_anchor_traits)
end

# ---------------------------------------------------------------------------
# Item 1.9 — extract_rotated_loadings_table (thin wrapper over §1.8; mirrors
# rotate-loadings.R:282-378 / .standardize_loadings_by_total_variance:381-405).
# ---------------------------------------------------------------------------

"""
    extract_rotated_loadings_table(fit::GllvmFit, Y; level = :unit,
        method = :varimax, order_axes = true, sign_anchor = :auto,
        anchor_traits = nothing, loading_scale = :raw) -> NamedTuple

Long (tidy) table of [`rotate_loadings`](@ref)'s output, one row per
`(trait, axis)` (axis-major order: all traits for axis 1, then axis 2, …).
Mirrors `gllvmTMB::extract_rotated_loadings_table`
(`rotate-loadings.R:282-378`).

`loading_scale = :raw` (default) reports the rotated loadings as returned by
`rotate_loadings`. `loading_scale = :standardized` divides each trait's row
by `sqrt(diag(extract_Sigma(fit; level, part = :total).Sigma))` — the
`.standardize_loadings_by_total_variance` step (`:381-405`) — and throws
`ArgumentError` if any trait's total variance is non-positive.
`axis_variance`/`axis_share` are always computed from the RAW rotated `Λ`
(pre-standardization), matching R.

Returns a `NamedTuple` of equal-length vectors: `level, trait, axis, loading,
abs_loading, axis_variance, axis_share, rotation, order_axes, sign_anchor,
anchor_trait, loading_scale`.
"""
function extract_rotated_loadings_table(fit::GllvmFit, Y::AbstractMatrix;
                                        level::Symbol = :unit, method::Symbol = :varimax,
                                        order_axes::Bool = true, sign_anchor::Symbol = :auto,
                                        anchor_traits::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                                        loading_scale::Symbol = :raw)
    loading_scale in (:raw, :standardized) ||
        throw(ArgumentError("loading_scale must be :raw or :standardized; got :$loading_scale"))

    rot = rotate_loadings(fit, Y; level = level, method = method, order_axes = order_axes,
                          sign_anchor = sign_anchor, anchor_traits = anchor_traits)
    p, d = size(rot.Lambda)

    Λout = rot.Lambda
    if loading_scale === :standardized
        total_var = diag(extract_Sigma(fit; level = level, part = :total).Sigma)
        all(v -> v > 0, total_var) ||
            throw(ArgumentError("extract_rotated_loadings_table(loading_scale=:standardized) " *
                "requires strictly positive per-trait total variance."))
        Λout = rot.Lambda ./ sqrt.(total_var)
    end

    total_axis_var = sum(rot.axis_variance)
    n_rows = p * d
    level_c = fill(level, n_rows)
    trait = Vector{Int}(undef, n_rows)
    axis = Vector{Int}(undef, n_rows)
    loading = Vector{Float64}(undef, n_rows)
    abs_loading = Vector{Float64}(undef, n_rows)
    axis_variance = Vector{Float64}(undef, n_rows)
    axis_share = Vector{Float64}(undef, n_rows)
    rotation_c = fill(method, n_rows)
    order_axes_c = fill(order_axes, n_rows)
    sign_anchor_c = fill(sign_anchor, n_rows)
    anchor_trait = Vector{Union{Int, Missing}}(undef, n_rows)
    loading_scale_c = fill(loading_scale, n_rows)

    idx = 0
    for k in 1:d, t in 1:p
        idx += 1
        trait[idx] = t
        axis[idx] = k
        loading[idx] = Λout[t, k]
        abs_loading[idx] = abs(Λout[t, k])
        axis_variance[idx] = rot.axis_variance[k]
        axis_share[idx] = rot.axis_variance[k] / total_axis_var
        anchor_trait[idx] = rot.anchor_traits === nothing ? missing : rot.anchor_traits[k]
    end

    return (level = level_c, trait = trait, axis = axis, loading = loading,
            abs_loading = abs_loading, axis_variance = axis_variance,
            axis_share = axis_share, rotation = rotation_c, order_axes = order_axes_c,
            sign_anchor = sign_anchor_c, anchor_trait = anchor_trait,
            loading_scale = loading_scale_c)
end

# ---------------------------------------------------------------------------
# Item 1.11 — extract_coevolution_modules, core math (mirrors
# extract-sigma.R:2203-2352 / .matrix_inv_sqrt:2355-2377).
#
# Scope now (core070 spec §1.11): `scale = :shape` only (operates directly
# on a shared covariance matrix Λ Λᵀ, already shape-scale — e.g. from
# `fit_coevolution_gaussian`'s `Λ * Λ'` or `extract_Sigma(part = :shared)`),
# positional trait indices. Named kernel-tier levels and `scale = :effect`
# (Γ·ρ, needing a stored ρ) are §2.7, not built.
# ---------------------------------------------------------------------------

# Symmetric pseudo-inverse-square-root: keeps only eigenvalues > eps (a
# Moore-Penrose-style pseudo-inverse for the discarded near-zero subspace),
# throws on a non-PSD matrix or a numerically-zero block.
function _matrix_inv_sqrt(A::AbstractMatrix; eps::Real = 1e-8)
    Asym = Symmetric((A .+ A') ./ 2)
    ev = eigen(Asym)
    minimum(ev.values) >= -eps ||
        throw(ArgumentError("block is not positive semidefinite (min eigenvalue $(minimum(ev.values)))."))
    keep = findall(v -> v > eps, ev.values)
    isempty(keep) &&
        throw(ArgumentError("block is numerically zero (max eigenvalue $(maximum(ev.values)) <= eps=$eps)."))
    V = ev.vectors[:, keep]
    return V * Diagonal(1.0 ./ sqrt.(ev.values[keep])) * V'
end

"""
    extract_coevolution_modules(Sigma_shared; row_traits, col_traits, eps = 1e-8)
        -> NamedTuple

Cross-lineage coevolution "modules" — the SVD decomposition of the
correlation-scaled cross block `R = Σ_row^(-1/2) Γ Σ_col^(-1/2)`. Mirrors
`gllvmTMB::extract_coevolution_modules` (`extract-sigma.R:2203-2352`).

`Sigma_shared` is a shared (`ΛΛᵀ`-type) covariance matrix indexed by the
stacked entity set — e.g. `fit_coevolution_gaussian(...).Λ * Λ'`, or
`extract_Sigma(fit; part = :shared).Sigma` for the phylo-loadings
coevolution fit. `row_traits`/`col_traits` are 1-based positional index
vectors into `Sigma_shared` selecting the host/partner sub-blocks:
`Σ_row = Sigma_shared[row_traits, row_traits]`,
`Σ_col = Sigma_shared[col_traits, col_traits]`,
`Γ = Sigma_shared[row_traits, col_traits]`.

`Σ_row^(-1/2)`/`Σ_col^(-1/2)` use a symmetric pseudo-inverse-square-root
(`eps` floors near-zero eigenvalues out of the pseudo-inverse); throws
`ArgumentError` if a block is not positive semidefinite or is numerically
zero.

Returns `(R, modules, row_axes, col_axes)`:
- `R` — the `length(row_traits) × length(col_traits)` correlation-scaled
  cross block.
- `modules` — `NamedTuple` of vectors `component, singular_value,
  squared_share` (`squared_share = d_k² / Σd²`), one row per singular value.
- `row_axes`/`col_axes` — long tables `(trait, component, loading)` from the
  left/right singular vectors `U`/`V` (`R = U * Diagonal(d) * V'`).
"""
function extract_coevolution_modules(Sigma_shared::AbstractMatrix;
                                     row_traits::AbstractVector{<:Integer},
                                     col_traits::AbstractVector{<:Integer},
                                     eps::Real = 1e-8)
    Σ_row = Sigma_shared[row_traits, row_traits]
    Σ_col = Sigma_shared[col_traits, col_traits]
    Γ = Sigma_shared[row_traits, col_traits]

    invsqrt_row = _matrix_inv_sqrt(Σ_row; eps = eps)
    invsqrt_col = _matrix_inv_sqrt(Σ_col; eps = eps)
    R = invsqrt_row * Γ * invsqrt_col

    F = svd(R)
    nc = length(F.S)
    total_sq = sum(abs2, F.S)
    component = collect(1:nc)
    singular_value = collect(F.S)
    squared_share = total_sq > 0 ? (F.S .^ 2) ./ total_sq : fill(NaN, nc)
    modules = (component = component, singular_value = singular_value,
              squared_share = squared_share)

    nrow = length(row_traits)
    row_axes_trait = Vector{Int}(undef, nrow * nc)
    row_axes_component = Vector{Int}(undef, nrow * nc)
    row_axes_loading = Vector{Float64}(undef, nrow * nc)
    idx = 0
    for k in 1:nc, i in 1:nrow
        idx += 1
        row_axes_trait[idx] = row_traits[i]
        row_axes_component[idx] = k
        row_axes_loading[idx] = F.U[i, k]
    end
    row_axes = (trait = row_axes_trait, component = row_axes_component, loading = row_axes_loading)

    ncol = length(col_traits)
    col_axes_trait = Vector{Int}(undef, ncol * nc)
    col_axes_component = Vector{Int}(undef, ncol * nc)
    col_axes_loading = Vector{Float64}(undef, ncol * nc)
    idx = 0
    for k in 1:nc, j in 1:ncol
        idx += 1
        col_axes_trait[idx] = col_traits[j]
        col_axes_component[idx] = k
        col_axes_loading[idx] = F.V[j, k]
    end
    col_axes = (trait = col_axes_trait, component = col_axes_component, loading = col_axes_loading)

    return (R = Matrix(R), modules = modules, row_axes = row_axes, col_axes = col_axes)
end

# ---------------------------------------------------------------------------
# Item 1.12 — imputed, reduced Gaussian-FIML form (mirrors
# missing-predictor.R:2597-2725 / gll_imputed_missing_predictor_se:2731-2755).
# ---------------------------------------------------------------------------

"""
    imputed(fitmi, x::AbstractVector) -> NamedTuple

Table of imputed/observed values for the missing site-level predictor `x` of
a [`fit_gaussian_mi_fiml`](@ref) result, mirroring `gllvmTMB::imputed`
(`missing-predictor.R:2597-2725`) for the Gaussian-FIML route.

`fitmi` is the `NamedTuple` returned by `fit_gaussian_mi_fiml` (must carry an
`eblup_x` field — a `GllvmFit` or any other fit type throws `ArgumentError`).
`x` is the SAME predictor vector passed to `fit_gaussian_mi_fiml` (the fit
does not store it): `estimate` is `fitmi.eblup_x` (the observed value where
`x` is observed, the Gaussian conditional mode `E[x_s | y_s]` where it is
`missing`/`NaN` — free from the fit, `fit_gaussian_mi_fiml` already computes
it), and `observed` flags exactly the non-missing entries of `x`.

Current scope: conditional standard
errors (`gll_imputed_missing_predictor_se`, `:2731-2755`, an extra
per-site Hessian-block computation over the augmented latent) are NOT
computed — every row reports `std_error = NaN`, `status = :se_not_computed`,
honestly, rather than a stub SE. `fit_gllvm_mi` (non-Gaussian response,
`missing_predictor_poisson.jl:380-386`) returns no imputed values in R
either — out of scope here.

Returns `(variable, level, estimate, observed, std_error, status)` —
`variable` is `:x` on every row (one predictor is currently supported);
`level` is the 1-based site index.
"""
function imputed(fitmi, x::AbstractVector)
    hasproperty(fitmi, :eblup_x) || throw(ArgumentError(
        "imputed() requires a fit_gaussian_mi_fiml() result (a NamedTuple with an " *
        "`eblup_x` field); got $(typeof(fitmi))."))
    n = length(fitmi.eblup_x)
    length(x) == n || throw(ArgumentError("length(x) = $(length(x)) must equal the fit's n_sites = $n."))

    isobs = [!(ismissing(xi) || (xi isa Real && isnan(xi))) for xi in x]
    return (variable = fill(:x, n), level = collect(1:n),
            estimate = collect(Float64, fitmi.eblup_x), observed = isobs,
            std_error = fill(NaN, n), status = fill(:se_not_computed, n))
end

# ---------------------------------------------------------------------------
# Item 1.13 — tidy, core tiers (mirrors methods-gllvmTMB.R:1177-1369).
#
# Scope reduction (this slice, tighter than the spec's already-reduced
# "Scope now"): `fit::GllvmFit` (plain Gaussian) ONLY. Non-Gaussian family
# fits, `spde_fit.jl` κ/τ, and rr-implied sds are NOT covered by this
# method — an honest dispatch restriction, not a stub: only the tiers a
# `GllvmFit` fit; `:cutpoint` is always empty-but-typed for `GllvmFit`
# (ordinal cutpoints need `OrdinalFit`/`OrdinalPerTraitFit`, out of scope
# here). Rows are unified across tiers to one schema (`effect, term,
# estimate, std_error, link, conf_low, conf_high`) rather than R's per-tier
# column sets — `link` is `:identity` for `:fixed` rows (the Gaussian
# family) and `missing` for `:ran_pars` rows.
# ---------------------------------------------------------------------------

function _tidy_rows_from_coef_table(ct::GllvmCoefTable, effect::Symbol, link;
                                    conf_int::Bool)
    n = length(ct.term)
    return [(effect = effect, term = ct.term[i], estimate = ct.estimate[i],
            std_error = ct.std_error[i], link = link,
            conf_low = conf_int ? ct.lower[i] : NaN,
            conf_high = conf_int ? ct.upper[i] : NaN) for i in 1:n]
end

function _tidy_fixed_rows(fit::GllvmFit, Y::AbstractMatrix;
                          conf_int::Bool, conf_level::Real,
                          X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing)
    try
        ct = coef_table(fit, Y; parm = "beta", level = conf_level, X = X)
        return _tidy_rows_from_coef_table(ct, :fixed, :identity; conf_int = conf_int)
    catch e
        e isa ArgumentError || rethrow()
        return NamedTuple[]   # no fixed-effect coefficients on this fit
    end
end

function _tidy_ran_pars_rows(fit::GllvmFit, Y::AbstractMatrix;
                             conf_int::Bool, conf_level::Real,
                             X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing)
    fit.model.has_diag || return NamedTuple[]
    rows = NamedTuple[]
    for parm in ("sigma_B", "sigma_W")
        try
            ct = coef_table(fit, Y; parm = parm, level = conf_level, X = X)
            append!(rows, _tidy_rows_from_coef_table(ct, :ran_pars, missing; conf_int = conf_int))
        catch e
            e isa ArgumentError || rethrow()
        end
    end
    return rows
end

"""
    tidy(fit::GllvmFit, Y; effects = (:fixed,), conf_int = false,
        conf_level = 0.95) -> Vector{NamedTuple}

Tidy inference table over one or more parameter TIERS, mirroring
`gllvmTMB::tidy.gllvmTMB_multi` (`methods-gllvmTMB.R:1177-1369`).
`effects` is a `Symbol` or collection of `Symbol`s from `(:fixed,
:ran_pars, :cutpoint)`.

- `:fixed` — the fixed-effect coefficients (`coef_table(fit, Y; parm =
  "beta")`); empty when the fit has no `X`/no free coefficients.
- `:ran_pars` — per-trait diagonal random-effect SDs (`sigma_B[t]`,
  `sigma_W[t]`, `coef_table(fit, Y; parm = "sigma_B"/"sigma_W")`); empty
  when the fit has no diagonal-RE tier (`has_diag = false`).
- `:cutpoint` — always empty (this `GllvmFit` method has no ordinal
  cutpoints to report; see the scope note above).

`conf_int = true` adds Wald `conf_low`/`conf_high` at `conf_level`
(`estimate ± quantile(Normal(), (1+conf_level)/2) * std_error` for the
LINEAR-scale `:fixed` tier; the log-scale-transformed bound for `:ran_pars`,
via [`confint`](@ref)'s existing back-transform convention). `conf_int =
false` (default) reports `NaN` for both.

`X` is forwarded to `coef_table`/`confint` — REQUIRED (matching what was
passed to `fit_gaussian_gllvm`) whenever the fit has fixed effects, since
`GllvmFit` does not store its own design (as everywhere else in
`GLLVModels.jl`); omitting it silently mismatches the Hessian reconstruction and
returns `NaN` standard errors for the `:fixed` tier.

Returns a `Vector{NamedTuple}`, ROW-UNIFIED across tiers: `(effect, term,
estimate, std_error, link, conf_low, conf_high)`.
"""
function tidy(fit::GllvmFit, Y::AbstractMatrix;
             effects::Union{Symbol, AbstractVector{Symbol}, NTuple{N, Symbol} where N} = (:fixed,),
             conf_int::Bool = false, conf_level::Real = 0.95,
             X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing)
    effs = effects isa Symbol ? (effects,) : effects
    all(e -> e in (:fixed, :ran_pars, :cutpoint), effs) ||
        throw(ArgumentError("effects entries must be :fixed, :ran_pars, or :cutpoint."))

    rows = NamedTuple[]
    for e in effs
        if e === :fixed
            append!(rows, _tidy_fixed_rows(fit, Y; conf_int = conf_int, conf_level = conf_level, X = X))
        elseif e === :ran_pars
            append!(rows, _tidy_ran_pars_rows(fit, Y; conf_int = conf_int, conf_level = conf_level, X = X))
        end
        # :cutpoint contributes no rows for GllvmFit.
    end
    return rows
end

# ---------------------------------------------------------------------------
# Item 1.14 — summary, core (mirrors methods-gllvmTMB.R:744-863, print :866).
#
# Scope reduction (this slice): `fit::GllvmFit` only (plain Gaussian).
# mspl / likelihood-weights / AGHQ-penalised-MAP annotations have no Julia
# engine counterpart — omitted per core070 spec §3.10 (no stub fields
# without a maintainer yes). The `$missing` block (masked/dropped counts)
# waits on §2.5 (fit-stored mask). Communality is reported as GLLVModels.jl's
# single combined `extract_communality(fit)` vector, NOT R's separate
# `communality_B`/`communality_W` split — `communality_B`/`communality_W`
# exist in this checkout only for `TwoLevelFit`, and `GllvmFit`'s single-tier
# `Σ_y_site`-denominator communality (see `extract_communality`'s own
# docstring deviation note) does not have a natural B/W split to report.
#
# Deviation from the spec's building-block note: the spec's §1.14 "Reuse"
# claims `extract_communality` is `TwoLevelFit`-only and needs extending to
# `GllvmFit` "in this slice (small, implementable-now)". That claim is
# WRONG against this checkout: `extract_communality(fit::GllvmFit)`
# (`src/extractors.jl:262`) already exists and is used directly here — no
# extension needed, and none was added (extending an already-correct method
# would be an unrelated/duplicate change under the surgical-changes rule).
# ---------------------------------------------------------------------------

"""
    GllvmSummary

Classed post-fit summary of a `GllvmFit`, as produced by
[`summary(fit::GllvmFit, Y)`](@ref). Mirrors `gllvmTMB`'s
`summary.gllvmTMB_multi` (`methods-gllvmTMB.R:744-863`) — a plain data
holder, formatting happens only in `Base.show`.

Fields: `p, K_B, K_W, has_diag, K_phy, has_phy_unique` (dims, from
`fit.model`), `estimator` (always `:ML` — `GLLVModels.jl`'s Gaussian path has no
REML variance-component-only estimator distinct from `fit_gaussian_reml`),
`logLik`, `converged`, `n_iter`, `fixef` (`Vector{NamedTuple}`, the `:fixed`
tier of [`tidy`](@ref)), `Sigma_B`/`Sigma_W` (`extract_Sigma` at
`level=:unit`/`:unit_obs`), `ICC` ([`extract_ICC_site`](@ref)),
`communality` ([`extract_communality`](@ref)), `se_status`
(`:ok`/`:sdreport_nonfinite`/`:sdreport_error`, see `Base.summary` docstring
below).
"""
struct GllvmSummary
    p::Int
    K_B::Int
    K_W::Int
    has_diag::Bool
    K_phy::Int
    has_phy_unique::Bool
    estimator::Symbol
    logLik::Float64
    converged::Bool
    n_iter::Int
    fixef::Vector{NamedTuple}
    Sigma_B::Matrix{Float64}
    Sigma_W::Matrix{Float64}
    ICC::Vector{Float64}
    communality::Vector{Float64}
    se_status::Symbol
end

# Three-way se_status classification off a Wald-CI's `se` vector: :ok unless
# EVERY se is non-finite (matching R's "a single NA does NOT trip it").
_gllvm_se_status_from_se(se::AbstractVector) =
    (isempty(se) || any(isfinite, se)) ? :ok : :sdreport_nonfinite

function _gllvm_se_status(fit::GllvmFit, Y::AbstractMatrix;
                          X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing)
    try
        ci = confint(fit, Y; method = :wald, X = X)
        return _gllvm_se_status_from_se(collect(Float64, ci.se))
    catch
        return :sdreport_error   # analogue of R's sd_report being NULL
    end
end

"""
    summary(fit::GllvmFit, Y::AbstractMatrix; X = nothing) -> GllvmSummary

Post-fit summary, mirroring `gllvmTMB::summary.gllvmTMB_multi`
(`methods-gllvmTMB.R:744-863`) — see the scope-reduction note above
`GllvmSummary` for the supported output. `Y` must
match what was passed to `fit_gaussian_gllvm` (the fit does not store its
data — as everywhere else in `GLLVModels.jl`). `X` is REQUIRED (matching what
was passed to `fit_gaussian_gllvm`) whenever the fit has fixed effects —
see [`tidy`](@ref)'s equivalent note; omitting it silently degrades
`fixef`'s standard errors and `se_status` to all-`NaN`.

`se_status` is the three-way classification R's `.gllvmTMB_se_status`
computes: `:sdreport_error` when the Wald machinery itself fails (the
analogue of R's `sd_report` being `NULL`); `:sdreport_nonfinite` when EVERY
standard error is non-finite (R's non-PD-Hessian signature — one `NaN`
among otherwise-finite SEs does NOT trip this); `:ok` otherwise.
"""
function Base.summary(fit::GllvmFit, Y::AbstractMatrix;
                      X::Union{Nothing, AbstractArray{<:Real, 3}} = nothing)
    m = fit.model
    fixef = _tidy_fixed_rows(fit, Y; conf_int = true, conf_level = 0.95, X = X)
    Sigma_B = extract_Sigma(fit; level = :unit, part = :total).Sigma
    Sigma_W = extract_Sigma(fit; level = :unit_obs, part = :total).Sigma
    ICC = extract_ICC_site(fit)
    comm = extract_communality(fit)
    status = _gllvm_se_status(fit, Y; X = X)
    return GllvmSummary(m.p, m.K, m.K_W, m.has_diag, m.K_phy, m.has_phy_unique,
                        :ML, fit.logLik, fit.converged, fit.n_iter, fixef,
                        Matrix(Sigma_B), Matrix(Sigma_W), collect(Float64, ICC),
                        collect(Float64, comm), status)
end

function Base.show(io::IO, s::GllvmSummary)
    println(io, "GllvmSummary — Gaussian GLLVM (p=$(s.p), K_B=$(s.K_B), K_W=$(s.K_W))")
    println(io, "  estimator: $(s.estimator)   logLik: $(round(s.logLik; sigdigits = 6))" *
                "   converged: $(s.converged) ($(s.n_iter) iters)   se_status: $(s.se_status)")
    println(io, "  fixef: $(length(s.fixef)) coefficient(s)")
    println(io, "  ICC (per trait): ", round.(s.ICC; sigdigits = 4))
    println(io, "  communality (per trait): ", round.(s.communality; sigdigits = 4))
end
