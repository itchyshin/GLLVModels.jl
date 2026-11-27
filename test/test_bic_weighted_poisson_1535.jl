# refs #1535 — weighted PoissonFit: bic(fit) should match aic ArgumentError
using GLLVModels, Test, Random, Distributions

@testset "weighted PoissonFit bic vs aic (#1535)" begin
    Random.seed!(1)
    Y = rand(Poisson(2), 3, 10)
    w = ones(3, 10)
    w[:, 1] .= 0
    fit = fit_gllvm(Y; family = Poisson(), K = 1, weights = w, iterations = 200)

    err_aic = @test_throws ArgumentError aic(fit)
    err_bic = @test_throws ArgumentError bic(fit)
    @test err_aic.value == err_bic.value
end