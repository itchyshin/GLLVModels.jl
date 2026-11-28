# Capability matrix maps R phylo_dep/animal_dep to fit_kernel_dep (#1180).

using GLLVModels, Test

@testset "capability-status R dep twins (#1180)" begin
    text = read(joinpath(@__DIR__, "..", "docs", "design", "capability-status.md"), String)
    @test occursin("fit_kernel_dep_gllvm", text)
    @test occursin("fit_phylo_dep_gllvm", text)
    @test occursin("Hadamard", text)
    @test isdefined(GLLVModels, :fit_kernel_dep_gllvm)
    @test isdefined(GLLVModels, :fit_phylo_dep_gllvm)
end