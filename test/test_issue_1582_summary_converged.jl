using Test
using GLLVModels, Distributions, Random

@testset "issue 1582: Base.summary reports NOT CONVERGED (refs #1582)" begin
    Random.seed!(1)
    Y = rand(Poisson(1), 3, 20)
    fit = fit_poisson_gllvm(Y; K = 2, iterations = 1)
    @test !fit.converged
    @test occursin("NOT CONVERGED", summary(fit))
    @test occursin("NOT CONVERGED", sprint(show, fit))
    fit2 = fit_poisson_gllvm(Y; K = 1, iterations = 80)
    if fit2.converged
        s = summary(fit2)
        @test !occursin("NOT CONVERGED", s)
    end
end