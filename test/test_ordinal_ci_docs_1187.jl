using GLLVModels, Test

# The docs said per-trait ordinal confidence intervals were unavailable.
# Native `confint` already returns Wald and bootstrap intervals; only
# `bridge_fit` refuses them (#1187).

const _ROOT1187 = dirname(dirname(pathof(GLLVModels)))

function _doc1187(rel, needle)
    return occursin(needle, read(joinpath(_ROOT1187, rel), String))
end

@testset "per-trait ordinal CI docs match native confint (#1187)" begin
    @test !_doc1187("docs/src/gllvmtmb-parity.md",
                    "per-trait ordinal-cutpoint CI endpoints are follow-ups")
    @test _doc1187("docs/src/gllvmtmb-parity.md", "native `confint`")
    @test _doc1187("docs/src/gllvmtmb-parity.md",
                   "`bridge_fit` still refuses per-trait ordinal")
    @test !_doc1187("docs/src/gllvmtmb-parity.md",
                    "grouped Tweedie and per-trait ordinal-cutpoint CI endpoints")
    @test !_doc1187("docs/src/roadmap.md",
                    "per-trait ordinal endpoints remain explicit unavailable")
    @test _doc1187("docs/src/confidence-intervals.md", "`tau[t,c]`")
    @test !_doc1187("src/bridge.jl", "OrdinalPerTraitFit has no _family_ci")
end
