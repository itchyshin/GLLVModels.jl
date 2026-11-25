using Test

# Docs check for #1226: the mixed-family section of gllvmtmb-parity.md and the
# fit_mixed_gllvm docstring must say that each Gaussian trait gets its own SD
# (gllvmTMB: one shared sigma_eps) and that logLik / aic are not comparable.
@testset "docs: mixed-family per-trait Gaussian sigma (#1226)" begin
    root = joinpath(@__DIR__, "..", "..")
    page = read(joinpath(root, "docs", "src", "gllvmtmb-parity.md"), String)
    i = findfirst("### Different response families in one model", page)
    @test i !== nothing
    sec = page[first(i):end]
    @test occursin("sigma_eps", sec)
    @test occursin("one SD per Gaussian trait", sec)
    @test occursin("not comparable", sec)
    src = read(joinpath(root, "src", "families", "mixed.jl"), String)
    @test occursin("shared `sigma_eps`", src)
end
