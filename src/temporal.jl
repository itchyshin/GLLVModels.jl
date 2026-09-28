# Temporal covariance source (gllvmTMB P1 port, slice 1).
#
# Constructors, the temporal pre-pass (R/temporal.R:100-453 at gllvmTMB
# 9539352f6) and the error type. The likelihood lives in
# `temporal_likelihood.jl`, the fitter in `temporal_fit.jl`, and the helpers in
# `temporal_methods.jl`. Design: docs/design/temporal-port-spec.md.

import StatsModels
import Tables

const _TEMPORAL_ACTION = "See `temporal_latent()` for the admitted temporal workflow."

"""
    TemporalContractError(message, kind)

Error thrown by every temporal refusal. `message` is gllvmTMB's message with
the cli markup removed. `kind` is the R condition class: the specific class R
attaches (for example `:gllvmTMB_temporal_forecast_mode`) or `:rlang_error`
when R attaches none, so tests can match on class as R's tests do.
"""
struct TemporalContractError <: Exception
    message::String
    kind::Symbol
end

Base.showerror(io::IO, e::TemporalContractError) =
    print(io, "TemporalContractError (", e.kind, "): ", e.message)

# Mirror of `.temporal_abort`: the standard action line is appended only when
# the message carries no action line of its own.
function _temporal_abort(message::AbstractString; kind::Symbol=:rlang_error,
        info=nothing, action=_TEMPORAL_ACTION)
    parts = String[message]
    info === nothing || push!(parts, String(info))
    action === nothing || push!(parts, String(action))
    throw(TemporalContractError(join(parts, "\n"), kind))
end

"""
    TemporalTerm

Marker for one temporal covariance source, returned by [`temporal_indep`](@ref),
[`temporal_dep`](@ref) and [`temporal_latent`](@ref). Fields: `formula` (the
bar expression, for example `:(0 + trait | series)`), `time` (the time column),
`mode` (`:indep`, `:dep` or `:latent`), `d` (`1` for `:latent`, else
`nothing`), `unique` (always `true` for `:indep`, the user flag for
`:latent`, `false` for `:dep`), `structure` (`:ar1` or `:ou`) and `replicate`
(a column name or `nothing`). It mirrors gllvmTMB's `gllvmTMB_temporal`
marker list.
"""
struct TemporalTerm
    formula::Expr
    time::Symbol
    mode::Symbol
    d::Union{Nothing,Int}
    unique::Bool
    structure::Symbol
    replicate::Union{Nothing,Symbol}
end

function Base.show(io::IO, t::TemporalTerm)
    print(io, "temporal_", t.mode, "(", t.formula, ", time = ", t.time,
        ", structure = :", t.structure)
    t.replicate === nothing || print(io, ", replicate = ", t.replicate)
    t.mode === :latent && print(io, ", d = ", t.d, ", unique = ", t.unique)
    print(io, ")")
end

function _temporal_marker(formula, time, mode::Symbol; d=nothing, unique=false,
        structure=:ar1, replicate=nothing)
    if !(formula isa Expr && formula.head === :call && length(formula.args) == 3 &&
         formula.args[1] === :|)
        _temporal_abort("A temporal covariance term requires a formula of the form `0 + trait | series`.")
    end
    time isa Symbol || _temporal_abort("`time` must be a bare column name.")
    replicate === nothing || replicate isa Symbol ||
        _temporal_abort("`replicate` must be NULL or a bare column name.")
    if mode === :latent && !(d isa Real && !(d isa Bool) && isfinite(d) && d == 1)
        _temporal_abort("`temporal_latent()` currently supports rank one only (`d = 1`).")
    end
    s = structure isa AbstractString ? Symbol(structure) : structure
    s isa Symbol && s in (:ar1, :ou) ||
        _temporal_abort("`structure` must be either \"ar1\" or \"ou\".")
    unique isa Bool || _temporal_abort("`unique` must be TRUE or FALSE.")
    return TemporalTerm(formula, time, mode, mode === :latent ? 1 : nothing,
        mode === :latent ? unique : mode === :indep, s, replicate)
end

"""
    temporal_indep(formula, time; structure = :ar1, replicate = nothing)

Temporal source with a separate AR1 or OU process for each trait:
`Sigma_T = diag(psi_1, ..., psi_p)`. `formula` is the quoted bar
`:(0 + trait | series)`; `time` names the occasion column (integer-valued
for AR1, which keeps its gaps as integer powers; elapsed numeric time for
OU). `replicate` optionally names the column that distinguishes repeated
measurements at a series-occasion-trait cell. Twin of gllvmTMB's
`temporal_indep()`; see [`fit_temporal_gllvm`](@ref).
"""
temporal_indep(formula, time; structure=:ar1, replicate=nothing) =
    _temporal_marker(formula, time, :indep; structure=structure, replicate=replicate)

"""
    temporal_dep(formula, time; structure = :ar1, replicate = nothing)

Temporal source whose AR1 or OU process has an unstructured trait covariance
`Sigma_T = L L'` with a free lower-triangular `p × p` factor `L`. Arguments
as in [`temporal_indep`](@ref). Twin of gllvmTMB's `temporal_dep()`.
"""
temporal_dep(formula, time; structure=:ar1, replicate=nothing) =
    _temporal_marker(formula, time, :dep; structure=structure, replicate=replicate)

"""
    temporal_latent(formula, time; d = 1, structure = :ar1, replicate = nothing,
                    unique = false)

Temporal source with rank-one trait loadings: `Sigma_T = lambda lambda'`,
plus `diag(psi)` when `unique = true`. The diagonal `psi` part follows the
same AR1 or OU process across occasions; it is not independent occasion noise.
Only `d = 1` is admitted, as in gllvmTMB P1. Twin of gllvmTMB's
`temporal_latent()`.
"""
temporal_latent(formula, time; d=1, structure=:ar1, replicate=nothing, unique=false) =
    _temporal_marker(formula, time, :latent; d=d, unique=unique, structure=structure,
        replicate=replicate)

# ---------------------------------------------------------------------------
# Pre-pass (R/temporal.R:100-453)
# ---------------------------------------------------------------------------

"""
    TemporalSpec

Internal state index built by the temporal pre-pass. States are the ordered
`(series, time)` pairs (`pair_table`, sorted by series label then time, as R
orders them). `state_id` and `trait_id` are one-based per data row;
`predecessor` is one-based per state with `0` at a series start; `gap` holds
integer AR1 gaps and `elapsed` numeric OU gaps (both `0` at a series start).
"""
struct TemporalSpec
    mode::Symbol
    structure::Symbol
    rank::Int
    unique::Bool
    workflow::Symbol
    series_col::Symbol
    time_col::Symbol
    trait_col::Symbol
    replicate_col::Union{Nothing,Symbol}
    traits::Vector{String}
    pair_table::NamedTuple{(:pair_id, :series, :time),Tuple{Vector{String},Vector{String},Vector{Float64}}}
    state_id::Vector{Int}
    trait_id::Vector{Int}
    predecessor::Vector{Int}
    gap::Vector{Int}
    elapsed::Vector{Float64}
    row_series::Vector{String}
    row_time::Vector{Float64}
end

_temporal_isna(x) = x === missing || x === nothing || (x isa AbstractFloat && isnan(x))

# R's as.character() of a numeric occasion in a pair id.
_temporal_time_label(t::Real) = isinteger(t) && abs(t) < 1e15 ? string(Int(t)) : string(Float64(t))

const _TEMPORAL_SOURCE_HEADS = r"^(phylo|animal|spatial|kernel|meta_|propto$|equalto$|spde$)"
const _TEMPORAL_PROVIDER_HEADS = r"^(latent|indep|dep|unique|scalar|animal_|phylo_|spatial_|kernel_|meta_|rr$|diag$|propto$|equalto$|spde$)"

function _temporal_structure_heads(structure)
    heads = String[]
    walk(x) = if x isa Expr
        if x.head === :call && x.args[1] isa Symbol
            push!(heads, String(x.args[1]))
        end
        foreach(walk, x.args)
    end
    foreach(walk, structure)
    return heads
end

function _parse_temporal_term(term::TemporalTerm, cols::NamedTuple; trait::Symbol,
        response::Symbol, structure=Expr[])
    heads = _temporal_structure_heads(structure)
    n_marker = 1 + count(h -> h in ("temporal_indep", "temporal_dep", "temporal_latent"), heads)
    n_marker == 1 || _temporal_abort("Only one temporal covariance term is supported in a model.")
    providers = unique(filter(h -> occursin(_TEMPORAL_PROVIDER_HEADS, h), heads))
    source_terms = filter(h -> occursin(_TEMPORAL_SOURCE_HEADS, h), providers)
    allowed_pair = term.mode === :indep && length(source_terms) == 1 &&
        source_terms[1] in ("kernel_indep", "phylo_indep", "animal_indep", "spatial_indep")
    if !allowed_pair && !isempty(source_terms)
        _temporal_abort("A temporal covariance term cannot be combined with another covariance source in this version.";
            info="Found source provider(s): " * join(string.(source_terms, "()"), ", ") * ".",
            action="Use ordinary unit/unit_obs terms, or one of the admitted replicated AR1 `temporal_indep() + kernel_indep()`, `temporal_indep() + phylo_indep()`, `temporal_indep() + animal_indep()`, or `temporal_indep() + spatial_indep()` cells. Other temporal source pairs remain deferred.")
    end
    if !haskey(cols, response) || any(_temporal_isna, cols[response])
        _temporal_abort("`temporal_latent()` requires complete Gaussian response values.";
            info="Temporal panels are validated before ordinary missing-response handling.",
            action="Remove or impute missing responses before fitting this version.")
    end
    bar = term.formula
    bar.args[3] isa Symbol || _temporal_abort("`temporal_latent()` requires `0 + trait | series`.")
    lhs = replace(string(bar.args[2]), r"\s+" => "")
    if lhs != "0+" * String(trait)
        _temporal_abort("`temporal_latent()` requires one trait-intercept block `0 + $(trait) | series`.";
            info="The wide `traits(...)` interface is expanded to this form before temporal parsing.",
            action="For long data, use `temporal_latent(0 + trait | series, time = occasion)`.")
    end
    series = bar.args[3]::Symbol
    time = term.time
    needed = [series, time, trait]
    missing_cols = filter(c -> !haskey(cols, c), needed)
    isempty(missing_cols) ||
        _temporal_abort("Temporal data are missing column(s): " * join(String.(missing_cols), ", ") * ".")
    tcol = cols[time]
    if !all(x -> x isa Real && !(x isa Bool) && isfinite(x), tcol)
        _temporal_abort("`time` must contain finite numeric occasions.")
    end
    times = Float64.(tcol)
    if term.structure === :ar1 && any(t -> t != floor(t), times)
        _temporal_abort("AR1 `time` must contain finite integer-valued occasions.")
    end
    if any(_temporal_isna, cols[series]) || any(_temporal_isna, cols[trait])
        _temporal_abort("Temporal series and trait identifiers must be complete.")
    end
    row_series = string.(cols[series])
    row_trait = string.(cols[trait])
    traits = sort(unique(row_trait))
    length(traits) >= 3 || _temporal_abort("`temporal_latent()` requires at least three traits.")
    series_levels = sort(unique(row_series))
    for s in series_levels
        occ = sort(unique(times[row_series .== s]))
        if length(occ) < 3 || any(diff(occ) .<= 0)
            _temporal_abort("Each temporal series needs at least three strictly ordered occasions.")
        end
    end
    pairs = sort(unique(collect(zip(row_series, times))); by=x -> (x[1], x[2]))
    pair_series = String[p[1] for p in pairs]
    pair_time = Float64[p[2] for p in pairs]
    pair_id = [string(s, ".", _temporal_time_label(t)) for (s, t) in pairs]
    pair_index = Dict(p => i for (i, p) in enumerate(pairs))
    state_id = [pair_index[(row_series[o], times[o])] for o in eachindex(times)]
    trait_index = Dict(t => j for (j, t) in enumerate(traits))
    trait_id = [trait_index[t] for t in row_trait]
    n = length(times)

    workflow = :unreplicated
    rep = term.replicate
    if rep !== nothing
        haskey(cols, rep) || _temporal_abort("`replicate` must name a column in `data`.")
        any(_temporal_isna, cols[rep]) && _temporal_abort("`replicate` must be complete.")
        workflow = :replicated
        rlab = string.(cols[rep])
        keys = collect(zip(state_id, rlab, trait_id))
        length(unique(keys)) == n ||
            _temporal_abort("Temporal data contain duplicate series--occasion--replicate--trait rows.")
        panel = Dict{Tuple{Int,String},Int}()
        for o in 1:n
            k = (state_id[o], rlab[o]); panel[k] = get(panel, k, 0) + 1
        end
        all(==(length(traits)), values(panel)) ||
            _temporal_abort("Each temporal replicate must contain one complete trait panel.")
        per_state = zeros(Int, length(pairs))
        for o in 1:n; per_state[state_id[o]] += 1; end
        all(>=(2 * length(traits)), per_state) ||
            _temporal_abort("Replicated temporal data require at least two measurements at every occasion.")
    else
        keys = collect(zip(state_id, trait_id))
        length(unique(keys)) == n ||
            _temporal_abort("Repeated temporal observations require `replicate =` to distinguish measurements.")
        per_state = zeros(Int, length(pairs))
        for o in 1:n; per_state[state_id[o]] += 1; end
        all(==(length(traits)), per_state) ||
            _temporal_abort("Unreplicated temporal data require one complete trait panel at every occasion.")
    end
    if allowed_pair
        if workflow !== :replicated || term.structure !== :ar1
            _temporal_abort("The temporal cross-source cell requires replicated AR1 observations.";
                info="At zero persistence, unreplicated temporal diagonal variation cannot be separated from observation-level noise.",
                action="Supply `replicate = measurement` with at least two complete measurements at every series--occasion, and use `structure = :ar1`.")
        end
        throw(ArgumentError("temporal cross-source cells ($(source_terms[1])) are not implemented in GLLVModels.jl yet; see docs/design/temporal-port-spec.md section 7, Q1"))
    end
    isempty(providers) || throw(ArgumentError(
        "ordinary unit/unit_obs terms beside a temporal source are not implemented in GLLVModels.jl yet (temporal port slice 2)"))

    S = length(pairs)
    predecessor = zeros(Int, S); gap = zeros(Int, S); elapsed = zeros(Float64, S)
    for s in 2:S
        if pair_series[s] == pair_series[s-1]
            predecessor[s] = s - 1
            d = pair_time[s] - pair_time[s-1]
            term.structure === :ar1 ? (gap[s] = Int(d)) : (elapsed[s] = d)
        end
    end
    p = length(traits)
    rank = term.mode === :dep ? p : term.mode === :latent ? 1 : 0
    return TemporalSpec(term.mode, term.structure, rank, term.unique, workflow, series,
        time, trait, rep, traits, (pair_id=pair_id, series=pair_series, time=pair_time),
        state_id, trait_id, predecessor, gap, elapsed, row_series, times)
end
