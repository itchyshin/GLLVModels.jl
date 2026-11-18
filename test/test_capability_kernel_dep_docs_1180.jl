using Test

@testset "capability-status documents kernel dep R twin (refs #1180)" begin
    path = joinpath(@__DIR__, "..", "docs", "design", "capability-status.md")
    text = read(path, String)
    @test occursin("fit_kernel_dep_gllvm", text)
    @test occursin("R twin", text)
    @test occursin("fit_phylo_dep_gllvm", text)
    @test occursin("Hadamard", text)
end