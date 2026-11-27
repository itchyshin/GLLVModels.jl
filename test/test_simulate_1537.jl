# refs #1537 — simulate(fit, n) for ZIBFit
using GLLVModels, Test, Random, Distributions, Statistics

@testset "simulate ZIBFit (#1537)" begin
    Random.seed!(1)
    p, K, n = 3, 1, 10
    Y = rand(Binomial(1, 0.25), p, n)
    fit = fit_gllvm(Y; family = ZIB(1), K = K, iterations = 500)
    @test fit isa ZIBFit

    rng = MersenneTwister(1537)
    n_sim = 3000
    Ys = simulate(fit, n_sim; rng = rng)
    @test size(Ys) == (p, n_sim)
    @test eltype(Ys) <: Integer && all(0 .<= Ys .<= 1)

    μ = predict(fit, Ys; type = :response)
    @test isapprox(vec(mean(Ys; dims = 2)), vec(mean(μ; dims = 2)); rtol = 0.35)
end