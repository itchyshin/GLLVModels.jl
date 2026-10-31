using Test

# #1020: the parity page and the CI page must say that the confint default
# differs from gllvmTMB (R "profile", Julia :wald).
@testset "docs: confint default differs from gllvmTMB (#1020)" begin
    docs = joinpath(dirname(@__DIR__), "docs", "src")
    parity = read(joinpath(docs, "gllvmtmb-parity.md"), String)
    row = filter(l -> startswith(l, "| `confint` default method"), split(parity, '\n'))
    @test length(row) == 1
    @test occursin("method = \"profile\"", only(row))
    @test occursin("method = :wald", only(row))
    ci = read(joinpath(docs, "confidence-intervals.md"), String)
    @test occursin("defaults to `method = \"profile\"`", ci)
end
