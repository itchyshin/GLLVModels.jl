using Test

# Structure of CHANGELOG.md and docs/src/changelog.md (#756).
const _ROOT = dirname(@__DIR__)
const _CHANGELOG_LINES = readlines(joinpath(_ROOT, "CHANGELOG.md"))
const _DOCS_CL = readlines(joinpath(_ROOT, "docs", "src", "changelog.md"))

@testset "changelog structure (#756)" begin
    v030 = findfirst(l -> startswith(l, "## v0.3.0"), _CHANGELOG_LINES)
    @test v030 !== nothing
    above = _CHANGELOG_LINES[1:(v030 - 1)]
    unreleased = filter(l -> occursin(r"^## (Unreleased|Development)\b", l), above)
    @test length(unreleased) == 1
    @test any(l -> startswith(l, "All notable changes"), _CHANGELOG_LINES[1:5])
    @test any(l -> occursin(r"^## .*v0\.3\.0", l), _DOCS_CL)
end
