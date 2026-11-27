# refs #1536 — simulate(fit, n) for BetaBinomialGroupedFit
using GLLVModels, Test, Random, Distributions, Statistics

@testset "simulate BetaBinomialGroupedFit (#1536)" begin
    Random.seed!(1)
    p, K, n = 3, 1, 10
    N = fill(10, p, n)
    Y = [rand(Binomial(N[t, s], 0.2)) for t in 1:p, s in 1:n]
    fit = fit_gllvm(Y; family = BetaBinom(), K = K, N = N, iterations = 500)
    @test fit isa BetaBinomialGroupedFit

    Nsim = fill(10, p, 3000)
    rng = MersenneTwister(1536)
    Ys = simulate(fit, 3000; N = Nsim, rng = rng)
    @test size(Ys) == (p, 3000)
    @test eltype(Ys) <: Integer && all(0 .<= Ys .<= 10)

    μ = predict(fit, Ys; N = Nsim, type = :mean)
    @test isapprox(vec(mean(Ys ./ Nsim; dims = 2)), vec(mean(μ; dims = 2)); rtol = 0.3)
end