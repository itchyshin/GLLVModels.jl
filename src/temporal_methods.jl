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
