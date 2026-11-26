# Issue #1413: deviance(fit, Y) requires training response shape.
using Test, GLLVModels, Random, Distributions

@testset "deviance(fit, Y) matches training shape (#1413)" begin
    Random.seed!(2)
    p, K, n = 3, 1, 5
    y = randn(p, n)
    fit = fit_gaussian_gllvm(y; K = K, iterations = 80)
    @test deviance(fit, y) ≈ deviance(fit) atol = 0.0
    @test_throws ArgumentError deviance(fit, randn(p + 1, n))

    p2, n2 = 4, 8
    Y = rand(0:3, p2, n2)
    O = zeros(p2, n2)
    fit_p = fit_gllvm(Y; family = Poisson(), K = 1, offset = O, iterations = 60)
    @test deviance(fit_p, Y) ≈ deviance(fit_p) atol = 0.0
    @test_throws ArgumentError deviance(fit_p, Y[:, 1:(n2 - 1)])
end