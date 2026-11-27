using Test, Random, StatsModels, Distributions, GLLVModels

@testset "formula: empty tuple site data (refs #1414)" begin
    Random.seed!(1126)
    Y = randn(3, 20)
    @test gllvm(@formula(y ~ 1), Y, NamedTuple(); family = Normal(), K = 1) isa GllvmFit
    @test gllvm(@formula(y ~ 1), Y, (); family = Normal(), K = 1) isa GllvmFit
end