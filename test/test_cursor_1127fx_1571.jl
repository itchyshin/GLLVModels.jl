# Issue #1571: all-zero unit weights must be refused at fit time (not converged=true + broken post-fit).
using Test
using GLLVModels
using Distributions
using Random

@testset "issue 1571: all-zero Poisson weights refused at fit" begin
    Random.seed!(1127)
    Y = abs.(round.(Int, rand(5, 12) .+ 1))
    @test_throws ArgumentError fit_gllvm(Y; family = Poisson(), K = 1, weights = zeros(12), iterations = 40)
    @test_throws ArgumentError GLLVModels._normalize_weights(zeros(12), 5, 12; caller = "fit_gllvm")
end