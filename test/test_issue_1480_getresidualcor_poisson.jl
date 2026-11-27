using Test, Random, Distributions, GLLVModels

@testset "getResidualCor on Laplace count fits (refs #1480)" begin
    Random.seed!(77)
    Y = rand(Poisson(2), 4, 30)
    fit = fit_poisson_gllvm(Y; K = 1, iterations = 100)
    R1 = correlation(fit, Y)
    R2 = getResidualCor(fit, Y)
    @test R1 ≈ R2
    Yn = rand(NegativeBinomial(2, 0.4), 4, 28)
    fitn = fit_nb_gllvm(Yn; K = 1, iterations = 120)
    @test getResidualCor(fitn, Yn) ≈ correlation(fitn, Yn)
end