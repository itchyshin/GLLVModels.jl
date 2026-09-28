# Extractors and helper routes of the temporal source (slice 1).

_temporal_require_fit(fit, what) = fit isa TemporalGaussianFit ||
    _temporal_abort(what)

"""
    extract_temporal(fit::TemporalGaussianFit)

Fitted temporal-source details, twin of gllvmTMB's `extract_temporal()`.
Returns a `NamedTuple` with

- `parameters`: `(mode, structure, workflow, n_series, n_pairs)`;
- `time`: `(parameter, value)`, where `parameter` is `"phi"` (AR1
  persistence `(1 - 1e-6) tanh(theta)`) or `"ou_rate"` (OU rate `exp(theta)`);
- `pair_index`: the state table `(pair_id, series, time)`, ordered by series
  then time;
- `loadings`: `Lambda_temporal` (`p × rank`, rows in trait order, raw with no
  sign flip) for `:dep` and `:latent`, else `nothing`;
- `variance`: `(trait, value, component)` with component
  `"temporal_indep_variance"` (`:indep`) or `"temporal_Psi_variance"`
  (`:latent` with `unique = true`), else empty vectors.

Each table is a `NamedTuple` of vectors (Tables.jl-compatible).
"""
function extract_temporal(fit)
    _temporal_require_fit(fit, "`extract_temporal()` requires a fit made with a temporal covariance term.")
    spec = fit.spec
    variance = if spec.mode === :indep
        (trait=copy(spec.traits), value=copy(fit.psi),
            component=fill("temporal_indep_variance", length(spec.traits)))
    elseif spec.unique
        (trait=copy(spec.traits), value=copy(fit.psi),
            component=fill("temporal_Psi_variance", length(spec.traits)))
    else
        (trait=String[], value=Float64[], component=String[])
    end
    return (
        parameters=(mode=String(spec.mode), structure=String(spec.structure),
            workflow=String(spec.workflow), n_series=length(unique(spec.pair_table.series)),
            n_pairs=length(spec.pair_table.pair_id)),
        time=(parameter=fit.time_parameter === :phi ? "phi" : "ou_rate", value=fit.time_value),
        pair_index=(pair_id=copy(spec.pair_table.pair_id), series=copy(spec.pair_table.series),
            time=copy(spec.pair_table.time)),
        loadings=fit.loadings === nothing ? nothing : copy(fit.loadings),
        variance=variance,
    )
end

# Stable public orientation for rank-one loadings (R/temporal.R:466-483): the
# first loading's sign unless it is below 1e-8 of the largest, then the first
# largest in trait order. The fitted coordinates are never changed.
function _temporal_report_sign(loadings::AbstractVecOrMat; traits=nothing)
    loading = vec(loadings isa AbstractMatrix ? loadings[:, 1] : loadings)
    names = traits === nothing ? string.(eachindex(loading)) : traits
    max_abs = maximum(abs, loading)
    fallback = isfinite(max_abs) && max_abs > 0 && abs(loading[1]) < 1e-8 * max_abs
    i = fallback ? argmax(abs.(loading)) : 1
    return (multiplier=loading[i] < 0 ? -1 : 1, anchor_trait=names[i],
        first_loading_negligible=fallback, anchor_loading=loading[i])
end

# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

# Convergence code recorded where gllvmTMB records `opt$convergence`: 0 when
# the fit verdict is `:converged`, otherwise a positive code for the stopping
# reason (1 iteration limit, 2 gradient not converged, 3 invalid final value).
_temporal_convergence_code(f::TemporalGaussianFit) =
    f.stopping_reason === :converged ? 0 :
    f.stopping_reason === :iteration_limit ? 1 :
    f.stopping_reason === :gradient_not_converged ? 2 : 3

# `.gllvmTMB_predict_unhandled_re_tiers(object, handled = "temporal")`: the
# ordinary and structured tiers active beside the temporal source. Slice 1 fits
# the temporal source alone, so this is always empty; the refusals below keep
# R's check order for when composition arrives.
_temporal_other_tiers(f::TemporalGaussianFit) = String[]

_temporal_nll_closure(f::TemporalGaussianFit) =
    let rows = _temporal_series_rows(f.spec)
        t -> temporal_marginal_nll(t, f.y, f.X, f.spec; series_rows=rows)
    end

# Covariance between two row sets given series, times and trait indices.
function _temporal_cross_covariance(f::TemporalGaussianFit, ls, lt, lj, rs, rt, rj)
    a = f.time_value
    C = zeros(length(ls), length(rs))
    for j in eachindex(rs), i in eachindex(ls)
        ls[i] == rs[j] || continue
        C[i, j] = _temporal_corr(f.spec.structure, a, lt[i], rt[j]) * f.Sigma_T[lj[i], rj[j]]
    end
    return C
end

# Refit the saved call on new data (gllvmTMB's `update(object, data = )`).
_temporal_refit(f::TemporalGaussianFit, data) =
    fit_temporal_gllvm(data; formula=f.formula, temporal=f.term, trait=f.trait,
        g_tol=f.g_tol, iterations=f.max_iterations)

# ---------------------------------------------------------------------------
# compare_temporal (R/temporal-selection.R)
# ---------------------------------------------------------------------------

"""
    compare_temporal(; fits...)

AIC table for two or more named temporal fits, twin of gllvmTMB's
`compare_temporal()`. Candidates are keyword arguments (`compare_temporal(ar1
= f1, ou = f2)`); every mode, structure and workflow is admitted, provided each
fit has the temporal source by itself and all use identical response rows in
the same order. Returns a `NamedTuple` with `model`, `logLik`, `df` (the free
outer coordinates, `length(opt\$par)` in gllvmTMB), `AIC = -2 logLik + 2 df`
and `convergence` (0 when the fit verdict is `:converged`). No likelihood-ratio
test is computed.
"""
function compare_temporal(args...; fits...)
    isempty(args) || _temporal_abort("Temporal candidates must be named.")
    length(fits) >= 2 || _temporal_abort("Supply at least two named temporal fits.")
    names = collect(keys(fits)); models = collect(values(fits))
    all(m -> m isa TemporalGaussianFit, models) ||
        _temporal_abort("Every candidate must be a native temporal fit.")
    composed = [String(n) for (n, m) in zip(names, models) if !isempty(_temporal_other_tiers(m))]
    isempty(composed) || _temporal_abort("`compare_temporal()` currently requires the temporal source by itself.";
        kind=:gllvmTMB_temporal_selection_composed,
        info="Candidate(s) with another covariance tier: " * join(composed, ", ") * ".",
        action="AIC comparison for temporal source pairs needs its own selection contract and evidence.")
    all(m -> m.y == models[1].y, models) ||
        _temporal_abort("Temporal candidates must use identical response rows and ordering.")
    return (model=String.(names), logLik=[loglikelihood(m) for m in models],
        df=[dof(m) for m in models], AIC=[aic(m) for m in models],
        convergence=[_temporal_convergence_code(m) for m in models])
end

# ---------------------------------------------------------------------------
# forecast_temporal (R/temporal-forecast.R)
# ---------------------------------------------------------------------------

_temporal_bound_formula(f::TemporalGaussianFit) =
    StatsModels.apply_schema(f.formula, StatsModels.schema(f.formula, f.data),
        StatsModels.StatisticalModel)

"""
    forecast_temporal(fit, newdata; se_fit = false)

Gaussian forecast at future occasions of already fitted series, twin of
gllvmTMB's `forecast_temporal()`. `newdata` is a Tables.jl table with the
series, time and trait columns, holding a complete trait panel at each
requested series-occasion pair; each time must be strictly after that series'
last fitted time. The forecast conditions on the whole observed response at
the fitted parameters: `est = X_new beta + C_of' C_oo^{-1} (y - X beta)`.
`se_fit` is the conditional predictive standard deviation
`sqrt(diag(C_ff - C_of' C_oo^{-1} C_of))`; it ignores parameter uncertainty and
is not a calibrated prediction interval.

Supported: an unreplicated `temporal_indep` fit with the temporal source by
itself (the gllvmTMB contract). Julia fits carry no offset, so the forecast
has none. Returns the `newdata` columns in input order plus `est` and, when
requested, `se_fit`.
"""
function forecast_temporal(fit, newdata; se_fit=false)
    fit isa TemporalGaussianFit ||
        _temporal_abort("`forecast_temporal()` requires a native temporal `gllvmTMB()` fit.")
    Tables.istable(newdata) || _temporal_abort("`newdata` must be a data frame.")
    se_fit isa Bool || _temporal_abort("`se.fit` must be TRUE or FALSE.")
    spec = fit.spec
    spec.replicate_col === nothing ||
        _temporal_abort("`forecast_temporal()` does not yet support replicated temporal panels.";
            kind=:gllvmTMB_temporal_forecast_replicated,
            action="Use a temporal-only unreplicated Gaussian fit for this fitted-parameter forecast route.")
    spec.mode === :indep ||
        _temporal_abort("`forecast_temporal()` currently supports `temporal_indep()` only.";
            kind=:gllvmTMB_temporal_forecast_mode,
            action="Forecasts for temporal dependent and latent trait covariance need mode-specific oracle evidence.")
    other = _temporal_other_tiers(fit)
    isempty(other) || _temporal_abort("`forecast_temporal()` currently supports the temporal source by itself.";
        kind=:gllvmTMB_temporal_forecast_composed,
        info="Active additional tiers: " * join(other, ", ") * ".",
        action="Forecasts with ordinary or structured source effects need a joint conditioning contract.")
    nd = Tables.columntable(newdata)
    required = [spec.series_col, spec.time_col, spec.trait_col]
    missing_cols = filter(c -> !haskey(nd, c), required)
    isempty(missing_cols) || _temporal_abort("`newdata` is missing required temporal column(s): " *
        join(String.(missing_cols), ", ") * ".")
    raw_series = nd[spec.series_col]; raw_time = nd[spec.time_col]; raw_trait = nd[spec.trait_col]
    if length(raw_time) == 0 || any(_temporal_isna, raw_series) || any(_temporal_isna, raw_time) ||
            any(_temporal_isna, raw_trait)
        _temporal_abort("`newdata` needs non-missing series, time, and trait values.")
    end
    all(x -> x isa Real && !(x isa Bool), raw_time) ||
        _temporal_abort("The temporal forecast time column must be numeric.")
    times = Float64.(raw_time)
    all(isfinite, times) || _temporal_abort("The temporal forecast time column must contain finite values.")
    if spec.structure === :ar1 && any(t -> abs(t - round(t)) > sqrt(eps(Float64)), times)
        _temporal_abort("AR1 temporal forecasts require integer occasions.")
    end
    series = string.(raw_series); traits = string.(raw_trait)
    unknown = setdiff(unique(series), unique(spec.row_series))
    isempty(unknown) || _temporal_abort("`forecast_temporal()` supports existing series only.";
        kind=:gllvmTMB_temporal_forecast_new_series,
        info="Unknown series: " * join(unknown, ", ") * ".",
        action="New-series forecasts need a separately validated marginal prediction contract.")
    unknown_trait = setdiff(unique(traits), spec.traits)
    isempty(unknown_trait) || _temporal_abort("`newdata` names unknown trait(s): " * join(unknown_trait, ", ") * ".")
    keys3 = collect(zip(series, times, traits))
    length(unique(keys3)) == length(keys3) ||
        _temporal_abort("`newdata` has duplicate series--time--trait rows.")
    panels = Dict{Tuple{String,Float64},Vector{String}}()
    for (s, t, j) in keys3; push!(get!(panels, (s, t), String[]), j); end
    all(v -> length(v) == length(spec.traits) && Set(v) == Set(spec.traits), values(panels)) ||
        _temporal_abort("`newdata` must contain a complete trait panel at every future series--occasion pair.";
            kind=:gllvmTMB_temporal_forecast_panel,
            action="Supply one row for each fitted trait at each requested time.")
    latest = Dict(s => maximum(spec.row_time[spec.row_series .== s]) for s in unique(spec.row_series))
    all(i -> times[i] > latest[series[i]], eachindex(times)) ||
        _temporal_abort("Temporal forecast occasions must be strictly after each series' fitted occasions.";
            kind=:gllvmTMB_temporal_forecast_not_future,
            action="Use `predict()` for fitted rows; this route forecasts future occasions only.")

    X_new = StatsModels.modelcols(_temporal_bound_formula(fit).rhs, nd)
    X_new = X_new isa AbstractVector ? reshape(X_new, :, 1) : X_new
    eta_new = X_new * fit.beta
    residual = fit.y .- fit.X * fit.beta
    trait_index = Dict(t => j for (j, t) in enumerate(spec.traits))
    new_j = [trait_index[t] for t in traits]
    s2 = fit.sigma_eps^2
    C_oo = _temporal_cross_covariance(fit, spec.row_series, spec.row_time, spec.trait_id,
        spec.row_series, spec.row_time, spec.trait_id) + s2 * I
    C_ff = _temporal_cross_covariance(fit, series, times, new_j, series, times, new_j) + s2 * I
    C_of = _temporal_cross_covariance(fit, spec.row_series, spec.row_time, spec.trait_id,
        series, times, new_j)
    F = cholesky(Symmetric(C_oo); check=false)
    issuccess(F) || _temporal_abort("The fitted temporal response covariance was not positive definite for forecasting.")
    solved = F \ hcat(residual, C_of)
    est = eta_new .+ C_of' * solved[:, 1]
    se_fit || return merge(nd, (est=est,))
    v = diag(C_ff .- C_of' * solved[:, 2:end])
    any(<(-1e-8), v) && _temporal_abort("The temporal forecast produced a negative conditional variance.")
    return merge(nd, (est=est, se_fit=sqrt.(max.(v, 0.0))))
end

# ---------------------------------------------------------------------------
# simulate (R/methods-gllvmTMB.R:1535-1560, 1653-1709)
# ---------------------------------------------------------------------------

# Conditional (posterior-mean) temporal effect on each row: (V - s2 I) V^-1 r,
# the value gllvmTMB's Laplace mode reports for this all-Gaussian model.
function _temporal_fitted_effect(f::TemporalGaussianFit)
    V = _temporal_covariance(f.parameters, f.spec, size(f.X, 2))
    r = f.y .- f.X * f.beta
    return r .- f.sigma_eps^2 .* (cholesky(Symmetric(V)) \ r)
end

# Stationary recursive draw of the states (the unconditional redraw of
# `.simulate_temporal_effect`): innovations for the rank-`r` scores, then for
# the `p` unique components, state by state in pair-table order.
function _temporal_redraw_effect(f::TemporalGaussianFit, rng::AbstractRNG)
    spec = f.spec; p = length(spec.traits); S = length(spec.pair_table.pair_id)
    L = TemporalLayout(size(f.X, 2), spec)
    theta = f.parameters[L.time]
    rank = spec.rank
    z = zeros(max(rank, 1), S); q = zeros(p, S)
    sd_q = spec.unique ? exp.(f.parameters[L.diag]) : zeros(p)
    for s in 1:S
        prev = spec.predecessor[s]
        a = prev == 0 ? 0.0 : spec.structure === :ar1 ? _temporal_phi(theta)^spec.gap[s] :
            exp(-exp(theta) * spec.elapsed[s])
        innov = spec.structure === :ou ? sqrt(-expm1(-2 * exp(theta) * spec.elapsed[s])) : sqrt(1 - a^2)
        if rank > 0
            draw = randn(rng, rank)
            z[1:rank, s] .= prev == 0 ? draw : a .* z[1:rank, prev] .+ innov .* draw
        end
        if spec.unique
            draw = sd_q .* randn(rng, p)
            q[:, s] .= prev == 0 ? draw : a .* q[:, prev] .+ innov .* draw
        end
    end
    out = zeros(length(f.y))
    for o in eachindex(out)
        s = spec.state_id[o]; j = spec.trait_id[o]
        rank > 0 && (out[o] += dot(view(f.loadings, j, :), view(z, 1:rank, s)))
        spec.unique && (out[o] += q[j, s])
    end
    return out
end

"""
    simulate(fit::TemporalGaussianFit; nsim = 1, condition_on_RE = false,
             rng = Random.default_rng())

Draw `nsim` Gaussian response vectors (an `n × nsim` matrix, rows in data
order) from a temporal fit. With `condition_on_RE = true` the draw is centred
on the fitted predictor, which includes the conditional temporal states; with
`false` the temporal states are redrawn from their stationary AR1/OU recursion
before the residual noise is added, as in gllvmTMB.
"""
function simulate(fit::TemporalGaussianFit; nsim::Integer=1, condition_on_RE::Bool=false,
        rng::AbstractRNG=default_rng())
    nsim >= 1 || throw(ArgumentError("nsim must be a positive integer"))
    fixed = fit.X * fit.beta
    out = zeros(length(fit.y), nsim)
    fitted = condition_on_RE ? _temporal_fitted_effect(fit) : nothing
    for k in 1:nsim
        effect = condition_on_RE ? fitted : _temporal_redraw_effect(fit, rng)
        out[:, k] .= fixed .+ effect .+ fit.sigma_eps .* randn(rng, length(fit.y))
    end
    return out
end

"""
    predict(fit::TemporalGaussianFit)

In-sample fitted predictor of a temporal fit (training rows only), returned
with the series, time, replicate (when present) and trait columns plus `est`.
`est` includes the conditional temporal states. New-data prediction is not
available for temporal fits; use [`forecast_temporal`](@ref) for future
occasions.
"""
function predict(fit::TemporalGaussianFit; newdata=nothing)
    newdata === nothing || _temporal_abort("`predict()` with `newdata` is not yet available for `temporal_latent()` fits.";
        kind=:gllvmTMB_temporal_predict_newdata,
        info="New rows need an explicitly reconstructed series--occasion score index.",
        action="Use `predict(fit)` for the training rows or `re_form = ~0` for fixed-effects-only training predictions.")
    spec = fit.spec
    cols = unique(filter(!isnothing, [spec.series_col, spec.time_col, spec.replicate_col, spec.trait_col]))
    est = fit.X * fit.beta .+ _temporal_fitted_effect(fit)
    return merge(NamedTuple{Tuple(cols)}(Tuple(fit.data[c] for c in cols)), (est=est,))
end

predict(fit::TemporalGaussianFit, newdata) = predict(fit; newdata=newdata)

# ---------------------------------------------------------------------------
# bootstrap_temporal (R/temporal-bootstrap.R)
# ---------------------------------------------------------------------------

"""
    bootstrap_temporal(fit; n_boot = 100, seed = nothing)

Parametric bootstrap of the temporal persistence or rate, twin of gllvmTMB's
`bootstrap_temporal()`. Each replicate draws an unconditional response
(`simulate(fit; condition_on_RE = false)`) and refits the saved model; failed
and non-converged refits are kept. Returns a `NamedTuple` with `replicate`,
`seed` (the draw seed of that replicate), `convergence` (0 when the refit
verdict is `:converged`, `missing` after an error), `objective` (the refit's
negative log-likelihood), `time_estimate` (`phi` for AR1, `kappa` for OU) and
`error` (empty when converged). No interval is computed.

Supported: an unreplicated `temporal_indep` fit with the temporal source by
itself. Draw seeds come from `Random.Xoshiro(seed)`; the same `seed`
reproduces a run within a Julia version, but the draws differ from R's. The
caller's global random stream is never advanced.
"""
function bootstrap_temporal(fit; n_boot=100, seed=nothing)
    fit isa TemporalGaussianFit || _temporal_abort("`bootstrap_temporal()` requires a native temporal fit.")
    other = _temporal_other_tiers(fit)
    isempty(other) || _temporal_abort("`bootstrap_temporal()` currently requires the temporal source by itself.";
        kind=:gllvmTMB_temporal_bootstrap_composed,
        info="The fit also uses covariance tier(s): " * join(other, ", ") * ".",
        action="A parametric bootstrap for temporal source pairs needs its own contract and evidence.")
    (fit.spec.mode === :indep && fit.spec.replicate_col === nothing) ||
        _temporal_abort("`bootstrap_temporal()` currently supports unreplicated Gaussian `temporal_indep()` fits only.")
    n_boot isa Real && !(n_boot isa Bool) && isfinite(n_boot) && n_boot >= 1 && isinteger(n_boot) ||
        _temporal_abort("`n_boot` must be a positive integer.")
    seed === nothing || (seed isa Real && !(seed isa Bool) && isfinite(seed) && seed >= 0 &&
        seed <= typemax(Int32) && isinteger(seed)) ||
        _temporal_abort("`seed` must be one non-negative whole number or `NULL`.")
    n = Int(n_boot)
    seed_rng = seed === nothing ? copy(default_rng()) : Random.Xoshiro(Int(seed))
    draw_seeds = rand(seed_rng, 1:Int(typemax(Int32)), n)
    resp = fit.formula.lhs.sym
    replicate = collect(1:n)
    convergence = Vector{Union{Missing,Int}}(undef, n)
    objective = Vector{Union{Missing,Float64}}(undef, n)
    time_estimate = Vector{Union{Missing,Float64}}(undef, n)
    errors = Vector{String}(undef, n)
    for i in 1:n
        draw = simulate(fit; nsim=1, condition_on_RE=false, rng=Random.Xoshiro(draw_seeds[i]))[:, 1]
        data = merge(fit.data, NamedTuple{(resp,)}((draw,)))
        refit = try
            _temporal_refit(fit, data)
        catch e
            e
        end
        if refit isa TemporalGaussianFit
            code = _temporal_convergence_code(refit)
            convergence[i] = code
            objective[i] = -refit.loglik
            time_estimate[i] = refit.time_value
            errors[i] = code == 0 ? "" : "non-converged optimizer code $code"
        else
            convergence[i] = missing; objective[i] = missing; time_estimate[i] = missing
            errors[i] = sprint(showerror, refit)
        end
    end
    return (replicate=replicate, seed=draw_seeds, convergence=convergence,
        objective=objective, time_estimate=time_estimate, error=errors)
end

# ---------------------------------------------------------------------------
# profile_temporal: TMB::tmbprofile (TMB 1.9.21) as tmbprofile_wrapper calls
# it, then .profile_bounds / .profile_terminus_status (R/profile-ci.R:157-296)
# ---------------------------------------------------------------------------

"""
    _temporal_tmbprofile(fit; ystep, ytol, parm_range = (-Inf, Inf), h = 1e-4)

Port of `TMB::tmbprofile(obj, name = theta_temporal_time, ...)` with
`adaptive = TRUE`, `slice = FALSE`: starting at the fitted coordinates, walk
up then down from displacement 0 with initial step `h`; each value is the
minimum of the NLL over the other coordinates (warm-started from the previous
displacement, reset to the fitted values at the start of each direction). A
direction stops when the parameter leaves `parm_range`, the value is missing,
`|y - y0| > ytol`, or after `ceil(5 ytol / ystep)` steps; the step halves when
one step changes the value by more than `ystep` and doubles below `ystep / 4`
(climbing) or `ystep / 8`. Returns `(theta, value)` sorted by `theta`, with
`missing` for a failed inner solve, as R returns `NA`.
"""
function _temporal_tmbprofile(fit::TemporalGaussianFit; ystep::Real, ytol::Real,
        parm_range=(-Inf, Inf), h::Real=1e-4)
    nll = _temporal_nll_closure(fit)
    par = copy(fit.parameters)
    idx = TemporalLayout(size(fit.X, 2), fit.spec).time
    others = setdiff(eachindex(par), idx)
    that = par[idx]
    maxit = ceil(Int, 5 * ytol / ystep)
    start = zeros(length(others))
    function f(x)
        base = copy(par); base[idx] += x
        inner(p0) = begin
            t = similar(p0, length(par))
            t .= base
            t[others] .+= p0
            nll(t)
        end
        try
            res = Optim.optimize(inner, start, Optim.LBFGS(),
                Optim.Options(g_tol=min(fit.g_tol, 1e-8), iterations=1000); autodiff=:forward)
            v = Optim.minimum(res)
            isfinite(v) || return missing
            start .= Optim.minimizer(res)
            return v
        catch e
            e isa InterruptException && rethrow()
            return missing
        end
    end
    function along(step)
        fill!(start, 0.0)
        xs = [0.0]; ys = Union{Missing,Float64}[f(0.0)]
        for _ in 1:maxit
            yinit = ys[1]; xcur = xs[end]; ycur = ys[end]
            xnext = xcur + step
            xnext + that < parm_range[1] && break
            parm_range[2] < xnext + that && break
            ynext = f(xnext)
            push!(xs, xnext); push!(ys, ynext)
            (ismissing(ynext) || ismissing(yinit)) && break
            abs(ynext - yinit) > ytol && break
            speed_min = ynext >= yinit ? ystep / 4 : ystep / 8
            abs(ynext - ycur) > ystep && (step /= 2)
            abs(ynext - ycur) < speed_min && (step *= 2)
        end
        return xs, ys
    end
    x1, y1 = along(float(h))
    x2, y2 = along(-float(h))
    theta = vcat(x1, x2) .+ that
    value = vcat(y1, y2)
    ord = sortperm(theta)            # stable, as R's order()
    return (theta=theta[ord], value=value[ord])
end

# .profile_terminus_status (R/profile-ci.R:157-184)
function _temporal_terminus_status(p_sub, v_sub, side; slope_ratio_tol=0.1)
    ord = side === :lower ? sortperm(p_sub; rev=true) : sortperm(p_sub)
    p = p_sub[ord]; v = v_sub[ord]
    keep = [!ismissing(a) && !ismissing(b) && isfinite(a) && isfinite(b) for (a, b) in zip(p, v)]
    p = Float64.(p[keep]); v = Float64.(v[keep])
    length(p) < 3 && return :undecidable
    dp = abs.(diff(p)); dv = diff(v)
    ok = dp .> 0
    any(ok) || return :undecidable
    slope = dv[ok] ./ dp[ok]
    max_slope = maximum(slope)
    (!isfinite(max_slope) || max_slope <= 0) && return :asymptotic
    tail = slope[end]
    isfinite(tail) || return :undecidable
    return tail <= slope_ratio_tol * max_slope ? :asymptotic : :truncated
end

# .profile_bounds (R/profile-ci.R:188-296), on the theta scale.
function _temporal_profile_bounds(theta, value, mle_val, mle_par, crit)
    thresh = mle_val + crit
    function cross(idx, side)
        length(idx) < 2 && return (missing, :failed)
        p_sub = theta[idx]; v_sub = value[idx]
        e_sub = [ismissing(v) ? missing : v - thresh for v in v_sub]
        pos = findall(e -> !ismissing(e) && e > 0, e_sub)
        neg = findall(e -> !ismissing(e) && e <= 0, e_sub)
        if isempty(pos)
            status = _temporal_terminus_status(p_sub, v_sub, side)
            status === :asymptotic && return (side === :lower ? -Inf : Inf, :asymptotic)
            return (missing, :truncated)
        end
        isempty(neg) && return (missing, :failed)
        sg = [ismissing(e) ? missing : sign(e) for e in e_sub]
        transitions = findall(k -> !ismissing(sg[k+1]) && !ismissing(sg[k]) && sg[k+1] - sg[k] != 0,
            1:length(sg)-1)
        isempty(transitions) && return (missing, :failed)
        i = side === :lower ? maximum(transitions) : minimum(transitions)
        p1, p2 = p_sub[i], p_sub[i+1]; e1, e2 = e_sub[i], e_sub[i+1]
        zeta(p, v) = (d = 2 * (v - mle_val); d = (!isfinite(d) || d < 0) ? 0.0 : d; sign(p - mle_par) * sqrt(d))
        z1 = zeta(p1, v_sub[i]); z2 = zeta(p2, v_sub[i+1])
        z_star = side === :lower ? -sqrt(2crit) : sqrt(2crit)
        if isfinite(z1) && isfinite(z2) && z2 != z1
            return (p1 + (z_star - z1) * (p2 - p1) / (z2 - z1), :crossed)
        end
        e2 == e1 && return (missing, :failed)
        return (p1 + (0 - e1) * (p2 - p1) / (e2 - e1), :crossed)
    end
    lo = cross(findall(<(mle_par), theta), :lower)
    hi = cross(findall(>(mle_par), theta), :upper)
    return (lower=lo[1], upper=hi[1], lower_status=lo[2], upper_status=hi[2])
end

"""
    profile_temporal(fit; level = 0.95, ystep = 0.5, ytol = nothing,
                     parm_range = (-Inf, Inf), h = 1e-4)

Likelihood profile of the temporal time parameter with every other coordinate
re-optimised, twin of gllvmTMB's `profile_temporal()`. The walk is a port of
`TMB::tmbprofile` as gllvmTMB calls it; `ytol = nothing` uses
`qchisq(level, 1) / 2 + 1`. Bounds are the crossings of `nll_hat +
qchisq(level, 1) / 2`, interpolated on the signed-root (zeta) scale, and are
reported on the persistence (`phi`, AR1) or rate (`kappa`, OU) scale. Returns
`(estimate, lower, upper)`: a bound is `missing` when the profile did not
establish an endpoint (truncated or failed search), and the boundary value
(`±(1 - 1e-6)` for `phi`, `0` or `Inf` for `kappa`) when the profile flattens.
This is a fitted-parameter profile, not a calibrated interval.

Supported: an unreplicated `temporal_indep` fit with the temporal source by
itself.
"""
function profile_temporal(fit; level::Real=0.95, ystep::Real=0.5, ytol=nothing,
        parm_range=(-Inf, Inf), h::Real=1e-4)
    fit isa TemporalGaussianFit || _temporal_abort("`profile_temporal()` requires a native temporal fit.")
    (fit.spec.mode === :indep && fit.spec.replicate_col === nothing) ||
        _temporal_abort("`profile_temporal()` currently supports unreplicated Gaussian `temporal_indep()` fits only.")
    isempty(_temporal_other_tiers(fit)) ||
        _temporal_abort("`profile_temporal()` currently requires the temporal source by itself.")
    0 < level < 1 || throw(ArgumentError("`level` must be a single value in (0, 1); got $level."))
    crit = quantile(Chisq(1), level) / 2
    tol = ytol === nothing ? crit + 1 : Float64(ytol)
    transform(x) = fit.spec.structure === :ar1 ? _temporal_phi(x) : exp(x)
    idx = TemporalLayout(size(fit.X, 2), fit.spec).time
    mle_par = fit.parameters[idx]
    tr = _temporal_tmbprofile(fit; ystep=ystep, ytol=tol, parm_range=parm_range, h=h)
    length(tr.theta) < 3 && return (estimate=transform(mle_par), lower=missing, upper=missing)
    b = _temporal_profile_bounds(tr.theta, tr.value, -fit.loglik, mle_par, crit)
    tb(x) = ismissing(x) ? missing : transform(x)
    return (estimate=transform(mle_par), lower=tb(b.lower), upper=tb(b.upper))
end

# ---------------------------------------------------------------------------
# Generic iid-score inference routes refuse temporal fits
# (`.temporal_assert_no_iid_inference`, R/temporal.R:455-464)
# ---------------------------------------------------------------------------

function _temporal_no_iid_inference(method::AbstractString)
    _temporal_abort("`$(method)()` is not available for `temporal_latent()` fits.";
        kind=:gllvmTMB_temporal_inference_unsupported,
        info="Its existing algorithm assumes iid latent scores or an iid refit path.",
        action="Use `extract_temporal()` for fitted parameters. The bounded `profile_temporal()` and `bootstrap_temporal()` helpers have their own contracts; this generic iid route remains unavailable.")
end

confint(::TemporalGaussianFit, args...; kwargs...) = _temporal_no_iid_inference("confint")
bootstrap_ci(::TemporalGaussianFit, args...; kwargs...) = _temporal_no_iid_inference("bootstrap_ci")
ordination_uncertainty(::TemporalGaussianFit, args...; kwargs...) =
    _temporal_no_iid_inference("ordination_uncertainty")

"""
    getLV(fit::TemporalGaussianFit)

Conditional (posterior-mean) temporal latent scores of a rank-one
`temporal_latent` fit, one per state in `pair_index` order, in the stable public
orientation of gllvmTMB (`.temporal_report_sign`: the sign of the first
loading, or of the first largest one when the first is negligible). Returns a
`NamedTuple` `(scores, loadings, pair_index, sign)`, where `scores` is an
`S × 1` matrix and `loadings` the correspondingly oriented `p × 1` loadings;
returns `nothing` for `temporal_indep` and `temporal_dep` fits, whose
covariance is available from [`extract_temporal`](@ref).
"""
function getLV(fit::TemporalGaussianFit)
    spec = fit.spec
    spec.mode === :latent || return nothing
    V = _temporal_covariance(fit.parameters, spec, size(fit.X, 2))
    w = cholesky(Symmetric(V)) \ (fit.y .- fit.X * fit.beta)
    S = length(spec.pair_table.pair_id)
    lambda = fit.loadings[:, 1]
    scores = zeros(S, 1)
    for s in 1:S, o in eachindex(w)
        spec.row_series[o] == spec.pair_table.series[s] || continue
        scores[s, 1] += _temporal_corr(spec.structure, fit.time_value, spec.pair_table.time[s],
            spec.row_time[o]) * lambda[spec.trait_id[o]] * w[o]
    end
    sgn = _temporal_report_sign(fit.loadings; traits=spec.traits)
    pair_index = (pair_id=copy(spec.pair_table.pair_id), series=copy(spec.pair_table.series),
        time=copy(spec.pair_table.time))
    return (scores=sgn.multiplier .* scores, loadings=sgn.multiplier .* fit.loadings,
        pair_index=pair_index, sign=sgn)
end
