# Issue #721: pin the Logit / Log default-link wording in src/families/links.jl.
# Logit is canonical for Binomial and a conventional default for Beta. Log is
# canonical for Poisson and a default (not canonical) for Gamma and
# NegativeBinomial. This file reads the source and checks those phrases stay
# in the docstrings.

using Test

const LINKS_SRC = read(joinpath(@__DIR__, "..", "src", "families", "links.jl"), String)

@testset "issue 721 link documentation" begin
    @test occursin(
        "Canonical link for `Binomial`. Conventional default (not canonical) for `Beta`.",
        LINKS_SRC,
    )
    @test occursin(
        "Canonical link for `Poisson`. Default (not canonical) for `Gamma` and `NegativeBinomial`.",
        LINKS_SRC,
    )
    @test occursin("`Binomial` → `LogitLink` (canonical)", LINKS_SRC)
    @test occursin("`Poisson` → `LogLink` (canonical)", LINKS_SRC)
    @test occursin("`NegativeBinomial` → `LogLink` (default; not canonical)", LINKS_SRC)
    @test occursin("`Beta` → `LogitLink` (conventional default; not a canonical-link GLM)", LINKS_SRC)
    @test occursin("`Gamma` → `LogLink` (default; not canonical)", LINKS_SRC)
end
