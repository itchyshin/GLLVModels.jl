# Issue #1473: OrderedBeta fitted/residuals/simulate + COMPoisson fitted (:response routing).

using Test
using GLLVModels, Distributions, Random

@testset "OrderedBeta post-fit (issue #1473)" begin
    Random.seed!(1)
    Y = clamp.(rand(Beta(2, 2), 3, 40), 0.01, 0.99)
    fit = fit_gllvm(Y; family = OrderedBeta(), K = 1, iterations = 120)
    μ_pred = predict(fit, Y)
    μ_fit = fitted(fit, Y)
    @test μ_pred ≈ μ_fit
    @test all(isfinite.(μ_fit))
    R = residuals(fit, Y)
    @test size(R) == size(Y)
    @test all(isfinite.(R))
    Ysim = simulate(fit, 40)
    @test size(Ysim) == (3, 40)
    @test all(0 .<= Ysim .<= 1)
end

@testset "COMPoisson fitted via :response (issue #1473)" begin
    Random.seed!(2)
    Y = max.(0, rand(Poisson(3), 3, 40))
    fit = fit_compoisson_gllvm(Y; K = 1, iterations = 80)
    μ_rate = predict(fit, Y; type = :rate)
    μ_fit = fitted(fit, Y)
    @test μ_rate ≈ μ_fit
end