using Test

@testset "issue 725 response-families Extractors snippet takes (fit, Y)" begin
    path = joinpath(@__DIR__, "..", "docs", "src", "response-families.md")
    @test isfile(path)
    text = read(path, String)
    start = findfirst("\n## Extractors\n", text)
    @test start !== nothing
    section = text[first(start):end]
    nxt = findnext("\n## ", section, 5)
    nxt === nothing || (section = section[1:first(nxt)])

    # Non-Gaussian fits have no one-argument communality/correlation/getLV
    # (src/link_residual.jl, src/postfit.jl): the snippet must pass Y.
    for f in ("communality", "correlation", "getLV")
        @test !occursin(Regex("^$(f)\\(fit\\)", "m"), section)
        @test occursin(Regex("^$(f)\\(fit, Y\\)", "m"), section)
    end
    @test occursin("Gaussian", section)
end
