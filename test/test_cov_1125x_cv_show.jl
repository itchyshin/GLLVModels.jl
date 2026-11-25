# Lane 1125x round 3: cv.jl argument guards and CVResult extractors/show.
using Test, GLLVModels, Random, Distributions
using GLLVModels: StatsAPI, loglikelihood, residuals, predict, fitted

const G = GLLVModels

@testset "1125x cv_gllvm validation and CVResult surfaces" begin
    rng = MersenneTwister(112505)
    p, n = 3, 18
    Y = randn(rng, p, n)

    @testset "cv_gllvm refuses bad k_folds and split before fitting" begin
        @test_throws ArgumentError cv_gllvm(Y; k_folds = 1, K = 1, rng = MersenneTwister(1))
        @test_throws ArgumentError cv_gllvm(Y; k_folds = 2, split = :traits, K = 1, rng = MersenneTwister(1))
        Ynone = fill(missing, p, n)
        @test_throws ArgumentError cv_gllvm(Ynone; k_folds = 2, K = 1, rng = MersenneTwister(1))
    end

    @testset "k_folds cannot exceed observed cells / sites / species" begin
        @test_throws ArgumentError cv_gllvm(Y; k_folds = 200, split = :random, K = 1, rng = MersenneTwister(2))
        @test_throws ArgumentError cv_gllvm(Y; k_folds = 50, split = :site, K = 1, rng = MersenneTwister(2))
        @test_throws ArgumentError cv_gllvm(Y; k_folds = 10, split = :species, K = 1, rng = MersenneTwister(2))
    end

    @testset "Gaussian X shape guard" begin
        @test_throws ArgumentError cv_gllvm(Y; k_folds = 2, family = Normal(), K = 1,
                                            X = randn(rng, p + 1, n, 1), rng = MersenneTwister(3))
    end

    @testset "_cv_extract_params refusal" begin
        @test_throws ArgumentError G._cv_extract_params((x = 1,))
    end

    @testset "CVResult StatsAPI + show" begin
        res = CVResult(
            3, :random, -12.5, 0.02, 0.01, 0.98,
            [-4.0, -4.1, -4.4], [0.01, 0.02, 0.03],
            [0.0, 0.02, 0.01], [1.0, 0.95, 1.05],
            zeros(p, n), zeros(p, n), Any[],
        )
        @test loglikelihood(res) == -12.5
        @test residuals(res) === res.residuals
        @test predict(res) === res.predictions
        @test fitted(res) === res.predictions
        plain = sprint(show, res)
        @test occursin("CVResult", plain)
        mime = sprint(show, MIME"text/plain"(), res)
        @test occursin("Fold", mime)
        @test occursin("logLik", mime)
    end
end