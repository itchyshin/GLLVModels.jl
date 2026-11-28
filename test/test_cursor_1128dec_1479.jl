# refs #1479 — bic(fit) must direct callers to bic(fit, Y), not MethodError
using GLLVModels, Test, Random, Distributions
using GLLVModels: StatsAPI, aic, bic

@testset "bic(fit) on Laplace fits (#1479)" begin
    Random.seed!(42)
    Y = rand(Poisson(2), 4, 30)
    fit = fit_poisson_gllvm(Y; K = 1, iterations = 100)

    @test isfinite(aic(fit))

    err = @test_throws ArgumentError bic(fit)
    msg = sprint(showerror, err.value)
    @test occursin("bic(fit, Y)", msg)
    @test isfinite(bic(fit, Y))
    @test isfinite(bic(fit, size(Y, 2)))
end