# refs #1493 — simulate(fit, n) for ZIPFit, ZINBFit, HurdlePoissonFit
using GLLVModels, Test, Random, Distributions

@testset "simulate two-part fits (ZIP / ZINB / Hurdle-Poisson)" begin
    @testset "ZIPFit" begin
        Random.seed!(214)
        Y = [rand(Poisson(s <= 20 ? 0 : 3)) for t in 1:4, s in 1:60]
        fit = fit_zip_gllvm(Y; K = 1, iterations = 100)
        Ys = simulate(fit, 25; rng = MersenneTwister(214))
        @test size(Ys) == (4, 25)
        @test eltype(Ys) <: Integer && all(Ys .>= 0)
        @test simulate(fit, 30; rng = MersenneTwister(7)) ==
              simulate(fit, 30; rng = MersenneTwister(7))
    end

    @testset "ZINBFit" begin
        Random.seed!(311)
        Y2 = rand(Poisson(2), 4, 60)
        fit = fit_zinb_gllvm(Y2; K = 1, iterations = 100)
        Ys = simulate(fit, 25; rng = MersenneTwister(311))
        @test size(Ys) == (4, 25)
        @test eltype(Ys) <: Integer && all(Ys .>= 0)
        @test simulate(fit, 20; rng = MersenneTwister(99)) ==
              simulate(fit, 20; rng = MersenneTwister(99))
    end

    @testset "HurdlePoissonFit" begin
        Random.seed!(310)
        Y3 = rand(Poisson(1), 4, 70)
        fit = fit_hurdle_poisson_gllvm(Y3; K = 1, iterations = 80)
        Ys = simulate(fit, 20; rng = MersenneTwister(310))
        @test size(Ys) == (4, 20)
        @test eltype(Ys) <: Integer && all(Ys .>= 0)
        @test simulate(fit, 15; rng = MersenneTwister(42)) ==
              simulate(fit, 15; rng = MersenneTwister(42))
    end
end