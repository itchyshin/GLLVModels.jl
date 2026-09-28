# predict / fitted for integrated species-distribution fits
# (docs/design/isdm-port-spec.md sections 1.7 and 3.2), the twin of
# predict.gllvmTMB_multi on an isdm_sources() fit at gllvmTMB P1
# (R/methods-gllvmTMB.R:2786-3220).

# re_form: `:all` is R's `~.` (full random effects); `:zero`, `nothing`, `0`
# and `missing` are R's `~0`, `NULL`-free zero forms (`~0`, `NA`, `0`).
function _isdm_re_form_zero(re_form)
    re_form === :all && return false
    (re_form === :zero || re_form === nothing || re_form === missing ||
     (re_form isa Real && re_form == 0)) && return true
    throw(ArgumentError(
        "re_form = $(repr(re_form)) is not supported: use :all (R's ~.) for the full " *
        "conditional prediction, or :zero / nothing / 0 / missing (R's ~0, NA, 0) for fixed " *
        "effects plus offset."))
end

_isdm_expr_symbols(ex) = ex isa Symbol ? Symbol[ex] :
    ex isa Expr && ex.head === :call ? reduce(vcat, (_isdm_expr_symbols(a) for a in ex.args[2:end]); init = Symbol[]) :
    Symbol[]

# Rebuild the design, offset and index vectors of `newdata` from the fit's
# frozen basis (by column name; fit-time QR alias removal is not repeated).
function _isdm_newdata_design(fit::IsdmFit, newdata)
    tab = fit.table
    cols = Tables.columntable(newdata)
    haskey(cols, :isdm_source) || throw(ArgumentError(
        "Integrated-source prediction needs source column `isdm_source` in `newdata`. Use a " *
        "declared source name on every prediction row."))
    for key in (tab.trait, tab.unit)
        haskey(cols, key) || throw(ArgumentError("column `$key` not found in `newdata`."))
    end
    src = _isdm_labels(getproperty(cols, :isdm_source))
    any(ismissing, src) && throw(ArgumentError(
        "Integrated-source prediction does not allow missing source labels in `newdata`."))
    nms = String.(tab.sources.names)
    unknown = setdiff(unique(src), nms)
    isempty(unknown) || throw(ArgumentError(
        "New data names undeclared integrated source(s): $(join(("\"$u\"" for u in unknown), ", ")). " *
        "Use the source names supplied to `isdm_sources()` when fitting."))
    n = length(src)
    srcid = [findfirst(==(s), nms) for s in src]

    X = zeros(n, length(tab.X_names))
    fsyms = _isdm_rhs_symbols(tab)
    for s in fsyms
        haskey(cols, s) || throw(ArgumentError("column `$s` not found in `newdata`."))
    end
    fcols = NamedTuple{Tuple(fsyms)}(Tuple(_isdm_narrow(getproperty(cols, s)) for s in fsyms))
    Xf = StatsModels.modelcols(tab.fixed_rhs, fcols)
    Xf = Xf isa AbstractVector ? reshape(Xf, :, 1) : Xf
    for (j, nm) in enumerate(tab.fixed_names)
        X[:, findfirst(==(nm), tab.X_names)] = Xf[:, j]
    end
    for (s, b) in tab.obs_basis
        rows = findall(==(String(s)), src)
        isempty(rows) && continue
        miss = [v for v in b.vars if !haskey(cols, v)]
        isempty(miss) || throw(ArgumentError(
            "Observation formula for source \"$s\" uses variable(s) absent from `newdata`. " *
            "Missing: " * join(("\"$v\"" for v in miss), ", ") * "."))
        for v in b.vars
            any(ismissing, view(getproperty(cols, v), rows)) && throw(ArgumentError(
                "Observation formula for source \"$s\" has missing values in `newdata`."))
        end
        sub = NamedTuple{Tuple(b.vars)}(Tuple(collect(skipmissing(getproperty(cols, v)[rows])) for v in b.vars))
        mm = StatsModels.modelcols(b.rhs, sub)
        mm = mm isa AbstractVector ? reshape(mm, :, 1) : mm
        for (j, nm) in enumerate(b.columns)
            k = findfirst(==(nm), tab.X_names)
            k === nothing && continue          # a column the fit dropped as aliased
            X[rows, k] = mm[:, j]
        end
    end

    off = if tab.offset_expr === nothing
        zeros(n)
    else
        absent = [v for v in _isdm_expr_symbols(tab.offset_expr) if !haskey(cols, v)]
        isempty(absent) || throw(ArgumentError(
            "`newdata` is missing the offset variable(s) $(join(("\"$v\"" for v in absent), ", ")). " *
            "The model was fitted with offset($(tab.offset_expr)), so predicting new rows needs " *
            "the same offset supplied for them."))
        _isdm_eval_expr(tab.offset_expr, cols, n)
    end

    traits = string.(getproperty(cols, tab.trait)); units = string.(getproperty(cols, tab.unit))
    tid = [something(findfirst(==(t), tab.trait_levels), 0) for t in traits]
    any(==(0), tid) && throw(ArgumentError(
        "`newdata` names trait level(s) not seen at fit time: $(join(unique(traits[tid .== 0]), ", "))."))
    uid = [something(findfirst(==(u), tab.unit_levels), 0) for u in units]   # 0 = unseen unit
    return X, off, uid, tid, srcid, units, traits, src
end

_isdm_rhs_symbols(tab::IsdmTable) = unique(_isdm_parse_formula(tab.formula; trait = tab.trait,
                                                               unit = tab.unit).fixed_symbols)

"""
    predict(fit::IsdmFit; newdata = nothing, type = :link, re_form = :all,
            se_fit = false) -> NamedTuple

Predictions from an integrated species-distribution fit, the twin of gllvmTMB's
`predict()` on an `isdm_sources()` fit at P1. Returns a `NamedTuple` of columns
`(unit, trait, isdm_source, est)`, named by the fit's `unit` and `trait` columns,
with `est` last.

- In-sample (`newdata = nothing`), `re_form = :all`: `est` is the per-row linear
  predictor at the latent modes (including the unique effects of a
  `unique = TRUE` fit), exactly `fit.eta`.
- `re_form = :zero` (also `nothing`, `0`, `missing`, R's `~0` and `NA`): fixed
  effects plus offset, `X * b_fix + offset`.
- `newdata`: the fixed design is rebuilt from the fitted basis by column name
  (the response column is not needed), source-observation columns are filled
  only on their source's rows, the offset is re-evaluated against `newdata`, and
  the latent contribution `Λ[t, :] . zhat[:, s]` (and, on a `unique = TRUE` fit,
  the unique effect `s_B[t, s]`) is re-added for units seen at fit time; an
  unseen unit falls back to the fixed-only prediction, as in R. The source
  column must be present, non-missing, and name declared sources.
- `type = :response` applies each row's own inverse link: `exp(eta)` on count
  rows, `1 - exp(-exp(eta))` on detection rows. It includes the offset, so a
  count row is an expected count at that support; set the offset column to zero
  in `newdata` for the effort-free scale.

`se_fit = true` is refused: with `newdata` as in R, and in-sample because the
fixed-effect-only delta-method standard error is fenced in P1.
"""
function predict(fit::IsdmFit; newdata = nothing, type::Symbol = :link, re_form = :all,
        se_fit::Bool = false)
    type in (:link, :response) || throw(ArgumentError("type must be :link or :response; got :$type"))
    zero_re = _isdm_re_form_zero(re_form)
    if se_fit
        newdata === nothing || throw(ArgumentError(
            "`se.fit = TRUE` is not yet supported together with `newdata`. Standard errors are " *
            "only available for the training rows."))
        throw(ArgumentError(
            "`se_fit = true` is not available on the Julia integrated door in P1: R's in-sample " *
            "value is a fixed-effect-only delta-method standard error, not a map interval, and it " *
            "is fenced here."))
    end
    tab = fit.table
    if newdata === nothing
        est = zero_re ? tab.X * fit.b_fix .+ tab.offset : copy(fit.eta)
        units = tab.unit_levels[tab.unit_id]
        traits = tab.trait_levels[tab.trait_id]
        srcs = String.(tab.sources.names[tab.source_id])
        fid = tab.fid
    else
        X, off, uid, tid, srcid, units, traits, srcs = _isdm_newdata_design(fit, newdata)
        est = X * fit.b_fix .+ off
        if !zero_re && size(fit.Λ, 2) > 0
            for i in eachindex(est)
                uid[i] == 0 && continue
                est[i] += dot(view(fit.Λ, tid[i], :), view(fit.zhat, :, uid[i]))
                fit.unique && (est[i] += fit.s_B[tid[i], uid[i]])
            end
        end
        fid = tab.sources.fid[srcid]
    end
    if type === :response
        est = [fid[i] == 2 ? exp(est[i]) : -expm1(-exp(est[i])) for i in eachindex(est)]
    end
    return NamedTuple{(tab.unit, tab.trait, :isdm_source, :est)}((units, traits, srcs, est))
end

"""
    fitted(fit::IsdmFit; type = :response, kwargs...) -> NamedTuple

In-sample predictions, `predict(fit; newdata = nothing, type, kwargs...)`; the
default scale is the response, as in gllvmTMB.
"""
fitted(fit::IsdmFit; type::Symbol = :response, kwargs...) =
    predict(fit; newdata = nothing, type = type, kwargs...)
