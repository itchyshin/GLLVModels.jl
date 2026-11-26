# Issues #1442 (Poisson) and #1443 (NB2): post-fit accepts integral Float64 counts.
using Test, GLLVModels, Distributions, Random

@testset "Poisson post-fit accepts integral Float64 counts (#1442)" begin
    Y_int = rand(0:4, 8, 10)
    Y = float.(Y_int)
    fit = fit_gllvm(Y; family = Poisson(), K = 2, iterations = 80)
    @test predict(fit, Y) ≈ predict(fit, Y_int)
    @test fitted(fit, Y) ≈ fitted(fit, Y_int)
    @test size(residuals(fit, Y)) == size(Y)
    Ybad = copy(Y)
    Ybad[2, 3] = 1.5
    @test_throws ArgumentError predict(fit, Ybad)
end

@testset "NB2 post-fit accepts integral Float64 counts (#1443)" begin
    Random.seed!(17)
    Y_int = rand(0:8, 8, 10)
    Y = float.(Y_int)
    fit = fit_gllvm(Y; family = NegativeBinomial(), K = 2, iterations = 80)
    @test predict(fit, Y) ≈ predict(fit, Int.(Y))
    @test size(getLV(fit, Y)) == (size(Y, 2), 2)
    Ybad = copy(Y)
    Ybad[1, 1] = 2.2
    @test_throws ArgumentError predict(fit, Ybad)
end