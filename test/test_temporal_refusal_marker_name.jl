# Parser refusals in `_parse_temporal_term` name the marker the user called
# (`temporal_indep()`, `temporal_dep()` or `temporal_latent()`), refs #1286.
# Message-only: the same inputs are refused for every marker, and the
# `temporal_latent()` text is pinned byte for byte. No fitting; the parser is
# called directly.
using Test, GLLVModels
const GMR = GLLVModels

function _trm_cols(; traits=["t1", "t2", "t3"], value_missing=false)
    rows = [(s, o, t) for t in traits for o in (1.0, 3.0, 7.0) for s in ("a", "b")]
    v = Union{Missing,Float64}[Float64(i) for i in eachindex(rows)]
    value_missing && (v[1] = missing)
    return (series=[r[1] for r in rows], occasion=[r[2] for r in rows],
        trait=[r[3] for r in rows], value=v)
end

const _TRM_MARKERS = (:latent, :indep, :dep)

function _trm_message(mode, formula, cols)
    mk = mode === :latent ? temporal_latent : mode === :indep ? temporal_indep : temporal_dep
    term = mk(formula, :occasion)
    err = try
        GMR._parse_temporal_term(term, cols; trait=:trait, response=:value)
        nothing
    catch e
        e
    end
    err isa TemporalContractError || return nothing
    return err.message
end

# The exact `temporal_latent()` strings on main 1b8de5c9.
const _TRM_LATENT = Dict(
    :complete => "`temporal_latent()` requires complete Gaussian response values.\n" *
        "Temporal panels are validated before ordinary missing-response handling.\n" *
        "Remove or impute missing responses before fitting this version.",
    :bar => "`temporal_latent()` requires `0 + trait | series`.\n" *
        "See `temporal_latent()` for the admitted temporal workflow.",
    :block => "`temporal_latent()` requires one trait-intercept block `0 + trait | series`.\n" *
        "The wide `traits(...)` interface is expanded to this form before temporal parsing.\n" *
        "For long data, use `temporal_latent(0 + trait | series, time = occasion)`.",
    :three => "`temporal_latent()` requires at least three traits.\n" *
        "See `temporal_latent()` for the admitted temporal workflow.",
)

const _TRM_ACT = "\nSee `temporal_latent()` for the admitted temporal workflow."

# Long-format panel; `reps` empty means no replicate column. `drop` removes row indices.
function _trm_long(; occ=(1.0, 3.0, 7.0), reps=(), drop=(), traits=("t1", "t2", "t3"))
    rr = isempty(reps) ? (nothing,) : reps
    rows = [(s, o, t, r) for r in rr for t in traits for o in occ for s in ("a", "b")]
    rows = [rows[i] for i in eachindex(rows) if !(i in drop)]
    nt = (series=Any[r[1] for r in rows], occasion=Float64[r[2] for r in rows],
        trait=Any[r[3] for r in rows], value=Float64[i for i in eachindex(rows)])
    isempty(reps) || (nt = merge(nt, (rep=Any[r[4] for r in rows],)))
    return nt
end

_trm_with(nt, k, f) = (v = copy(getproperty(nt, k)); f(v); merge(nt, NamedTuple{(k,)}((v,))))

# Remaining parser refusals and constructor refusals. Each entry is
# (key, thunk(mode) -> message or nothing, exact latent text on main 1b8de5c9).
_trm_parse(mode, cols; formula=:(0 + trait | series), kw=(;), structure=Expr[]) = begin
    mk = mode === :latent ? temporal_latent : mode === :indep ? temporal_indep : temporal_dep
    term = mk(formula, :occasion; kw...)
    try
        GMR._parse_temporal_term(term, cols; trait=:trait, response=:value, structure=structure)
        nothing
    catch e
        e isa TemporalContractError ? e.message : rethrow()
    end
end
_trm_ctor(f) = try; f(); nothing; catch e; e isa TemporalContractError ? e.message : rethrow(); end
_trm_mk(mode) = mode === :latent ? temporal_latent : mode === :indep ? temporal_indep : temporal_dep

const _TRM_MORE = (
    (:one_term, m -> _trm_parse(m, _trm_long(); structure=Expr[:(temporal_dep(0 + trait | series))]),
        "Only one temporal covariance term is supported in a model." * _TRM_ACT),
    (:missing_col, m -> _trm_parse(m, Base.structdiff(_trm_long(), (series=nothing,))),
        "Temporal data are missing column(s): series." * _TRM_ACT),
    (:time_finite, m -> _trm_parse(m, _trm_with(_trm_long(), :occasion, v -> v[1] = Inf)),
        "`time` must contain finite numeric occasions." * _TRM_ACT),
    (:ar1_int, m -> _trm_parse(m, _trm_with(_trm_long(), :occasion, v -> v[1] = 2.5)),
        "AR1 `time` must contain finite integer-valued occasions." * _TRM_ACT),
    (:ids, m -> _trm_parse(m, _trm_with(_trm_long(), :series, v -> v[1] = missing)),
        "Temporal series and trait identifiers must be complete." * _TRM_ACT),
    (:three_occ, m -> _trm_parse(m, _trm_long(occ=(1.0, 3.0))),
        "Each temporal series needs at least three strictly ordered occasions." * _TRM_ACT),
    (:rep_nocol, m -> _trm_parse(m, _trm_long(); kw=(replicate=:rep,)),
        "`replicate` must name a column in `data`." * _TRM_ACT),
    (:rep_na, m -> _trm_parse(m, _trm_with(_trm_long(reps=("r1", "r2")), :rep, v -> v[1] = missing); kw=(replicate=:rep,)),
        "`replicate` must be complete." * _TRM_ACT),
    (:dup, m -> _trm_parse(m, _trm_with(_trm_long(reps=("r1", "r2")), :rep, v -> fill!(v, "r1")); kw=(replicate=:rep,)),
        "Temporal data contain duplicate series--occasion--replicate--trait rows." * _TRM_ACT),
    (:panel, m -> _trm_parse(m, _trm_long(reps=("r1", "r2"), drop=(1,)); kw=(replicate=:rep,)),
        "Each temporal replicate must contain one complete trait panel." * _TRM_ACT),
    (:two_meas, m -> _trm_parse(m, _trm_long(reps=("r1",)); kw=(replicate=:rep,)),
        "Replicated temporal data require at least two measurements at every occasion." * _TRM_ACT),
    (:need_rep, m -> _trm_parse(m, _trm_long(reps=("r1", "r2"))),
        "Repeated temporal observations require `replicate =` to distinguish measurements." * _TRM_ACT),
    (:unrep_panel, m -> _trm_parse(m, _trm_long(drop=(1,))),
        "Unreplicated temporal data require one complete trait panel at every occasion." * _TRM_ACT),
    (:ctor_formula, m -> _trm_ctor(() -> _trm_mk(m)(:(a + b), :occasion)),
        "A temporal covariance term requires a formula of the form `0 + trait | series`." * _TRM_ACT),
    (:ctor_time, m -> _trm_ctor(() -> _trm_mk(m)(:(0 + trait | series), "occasion")),
        "`time` must be a bare column name." * _TRM_ACT),
    (:ctor_rep, m -> _trm_ctor(() -> _trm_mk(m)(:(0 + trait | series), :occasion; replicate="r")),
        "`replicate` must be NULL or a bare column name." * _TRM_ACT),
    (:ctor_structure, m -> _trm_ctor(() -> _trm_mk(m)(:(0 + trait | series), :occasion; structure=:foo)),
        "`structure` must be either \"ar1\" or \"ou\"." * _TRM_ACT),
)

const _TRM_CASES = (
    (:complete, :(0 + trait | series), () -> _trm_cols(value_missing=true)),
    (:bar, :(0 + trait | f(series)), () -> _trm_cols()),
    (:block, :(1 | series), () -> _trm_cols()),
    (:three, :(0 + trait | series), () -> _trm_cols(traits=["t1", "t2"])),
)

@testset "temporal parser refusals name the marker used (#1286)" begin
    for (key, formula, mkcols) in _TRM_CASES
        @testset "$key" begin
            for mode in _TRM_MARKERS
                name = "temporal_" * String(mode)
                msg = _trm_message(mode, formula, mkcols())
                @test msg !== nothing                       # still refused
                expected = replace(_TRM_LATENT[key], "temporal_latent" => name)
                @test msg == expected
                if mode !== :latent
                    @test !occursin("temporal_latent", msg)
                end
            end
        end
    end
    @testset "admitted input is still admitted for every marker" begin
        for mode in _TRM_MARKERS
            mk = mode === :latent ? temporal_latent : mode === :indep ? temporal_indep : temporal_dep
            term = mk(:(0 + trait | series), :occasion)
            cols = merge(_trm_cols(), (value=Float64[1:18;],))
            spec = GMR._parse_temporal_term(term, cols; trait=:trait, response=:value)
            @test spec.mode === mode
        end
    end
    @testset "every other shared-parser and constructor refusal" begin
        for (key, thunk, latent_text) in _TRM_MORE
            @testset "$key" begin
                for mode in _TRM_MARKERS
                    msg = thunk(mode)
                    @test msg !== nothing                   # still refused
                    if mode === :latent
                        @test msg == latent_text            # main's text, byte for byte
                    else
                        name = "temporal_" * String(mode)
                        @test msg == replace(latent_text, "temporal_latent" => name)
                        @test !occursin("temporal_latent", msg)
                    end
                end
            end
        end
    end
    @testset "refusals that keep their own wording" begin
        # Rank one is reachable only through temporal_latent(), unchanged on main.
        @test _trm_ctor(() -> temporal_latent(:(0 + trait | series), :occasion; d=2)) ==
            "`temporal_latent()` currently supports rank one only (`d = 1`)." * _TRM_ACT
        # The cross-source refusal recommends temporal_indep() cells for every marker.
        cross = "A temporal covariance term cannot be combined with another covariance source in this version.\n" *
            "Found source provider(s): kernel_indep().\n" *
            "Use ordinary unit/unit_obs terms, or one of the admitted replicated AR1 `temporal_indep() + kernel_indep()`, `temporal_indep() + phylo_indep()`, `temporal_indep() + animal_indep()`, or `temporal_indep() + spatial_indep()` cells. Other temporal source pairs remain deferred."
        for mode in (:dep, :latent)
            @test _trm_parse(mode, _trm_long(); structure=Expr[:(kernel_indep(0 + trait | site))]) == cross
        end
    end
end
