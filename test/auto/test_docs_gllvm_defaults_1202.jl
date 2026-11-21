using Test

# Docs check for #1202: the gllvmtmb-parity convention table must name the gllvm 2.0.15
# defaults that change the model (binomial/beta probit link, scale.X, fourth-corner
# trait standardisation, fixed Tweedie power, no Laplace for ordinal).
@testset "docs: gllvm default-change rows in gllvmtmb-parity.md (#1202)" begin
    page = read(joinpath(@__DIR__, "..", "..", "docs", "src", "gllvmtmb-parity.md"), String)
    i = findfirst("**convention differences**", page)
    @test i !== nothing
    tail = page[first(i):end]
    for needle in ("`link = \"probit\"`", "`scale.X = TRUE`", "`fit\$TR`", "`Power = 1.1`", "cannot yet handle ordinal")
        @test occursin(needle, tail)
    end
    @test occursin("`LogitLink()`", tail)
end
