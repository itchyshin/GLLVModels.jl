# Issue #1420: predict(getLV) on Float64 0/1 responses when counts are integral.
using Test, GLLVModels, Random, Distributions

@testset "Binomial predict accepts integral Float64 responses (#1420)" begin
    rng = MersenneTwister(3)
    p, n = 6, 60
    Y = Int.(rand(rng, Bernoulli(0.3), p, n))
    fit = fit_gllvm(Y; family = Binomial(), K = 1, iterations = 100)
    Yf = Float64.(Y)
    @test predict(fit, Y) ≈ predict(fit, Yf)
    @test size(getLV(fit, Yf)) == (n, 1)
end

@testset "Binomial predict refuses non-integral Float64 (#1420)" begin
    rng = MersenneTwister(11)
    p, n = 4, 20
    Y = Int.(rand(rng, Bernoulli(0.4), p, n))
    fit = fit_gllvm(Y; family = Binomial(), K = 1, iterations = 50)
    Ybad = Float64.(Y)
    Ybad[1, 1] = 0.5
    @test_throws ArgumentError predict(fit, Ybad)
end