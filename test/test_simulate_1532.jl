# refs #1532 — simulate(fit, n) for HurdleNBFit
using GLLVModels, Test, Random, Distributions, Statistics

@testset "simulate HurdleNBFit (#1532)" begin
    Random.seed!(2711)
    p, K, n = 3, 1, 11
    Y = rand(Poisson(1.2), p, n)
    fit = fit_gllvm(Y; family = HurdleNB(), K = K, iterations = 500)
    @test fit isa HurdleNBFit

    rng = MersenneTwister(2711)
    n_sim = 2500
    Ys = simulate(fit, n_sim; rng = rng)
    @test size(Ys) == (p, n_sim)
    @test eltype(Ys) <: Integer && all(Ys .>= 0)
    @test simulate(fit, 30; rng = MersenneTwister(7)) ==
          simulate(fit, 30; rng = MersenneTwister(7))

    μ = predict(fit, Ys; type = :response)
    @test isapprox(vec(mean(Ys; dims = 2)), vec(mean(μ; dims = 2)); rtol = 0.3)
end