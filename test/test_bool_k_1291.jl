# Issue #1291: Bool must not pass as K (Julia Integer includes Bool).
using GLLVModels, Test, Distributions, Random, StatsModels

@testset "Bool K refused (#1291)" begin
    rng = MersenneTwister(1)
    Y = rand(rng, 0:5, 4, 30)
    msg = "fit_gllvm: K must be a non-negative integer (got K = true of type Bool)"
    @test_throws ArgumentError(msg) fit_gllvm(Y; family = Poisson(), K = true)
    @test_throws ArgumentError("fit_gllvm: K must be a non-negative integer (got K = false of type Bool)") fit_gllvm(Y; family = Poisson(), K = false)
    @test_throws ArgumentError("fit_gllvm: K must be a non-negative integer (got K = true of type Bool)") fit_gllvm(Y; family = Poisson(), num_lv = true)
    fit = fit_gllvm(Y; family = Poisson(), K = 1, iterations = 2)
    @test fit isa PoissonFit
    @test gllvm(@formula(y ~ 1), Y, (;); family = Poisson(), K = 1, iterations = 2) isa PoissonFit
    @test_throws ArgumentError("gllvm: K must be a non-negative integer (got K = true of type Bool)") gllvm(@formula(y ~ 1), Y, (;); family = Poisson(), K = true)
end