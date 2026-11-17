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
end
