using GLLVModels, Test, Random, Distributions

@testset "CensoredPoissonFit post-fit (#1534)" begin
    Random.seed!(1)
    Y = rand(Poisson(2), 3, 10) .+ 1
    fit = fit_gllvm(Y; family = CensoredPoisson(), K = 1, iterations = 400)
    @test fit.converged
    μ = predict(fit, Y)
    @test size(μ) == size(Y)
    @test all(isfinite, μ)
    @test fitted(fit, Y) ≈ μ
    R = residuals(fit, Y; type = :pearson)
    @test size(R) == size(Y)
    Ysim = simulate(fit, 5; rng = MersenneTwister(1534))
    @test size(Ysim) == (3, 5)
    @test eltype(Ysim) <: Integer
    @test all(>=(0), Ysim)
end