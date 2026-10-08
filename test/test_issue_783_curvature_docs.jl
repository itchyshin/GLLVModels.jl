# Pin the #783 docs fix: Laplace curvature defaults in
# docs/src/response-families.md must match the code
# (`_default_hessian(::Binomial, ::CLogLogLink) === :observed`).
using Test

@testset "issue #783 Laplace curvature docs" begin
    path = joinpath(@__DIR__, "..", "docs", "src", "response-families.md")
    @test isfile(path)
    text = replace(read(path, String), r"\s+" => " ")
    @test occursin(
        "Binomial/**cloglog** now defaults to **observed**, matching TMB / `gllvmTMB` (confirmed 2026-09-01); `:fisher` was a Julia-side defect.",
        text,
    )
end
