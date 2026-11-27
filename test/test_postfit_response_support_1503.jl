# #1503: predict/fitted/residuals refuse out-of-support Y like the Laplace fitters (#730).
using Test, GLLVModels, Random, Distributions
using GLLVModels: predict, fitted, residuals

@testset "#1503 post-fit response support" begin
    rng = MersenneTwister(27001)

    @testset "Binomial" begin
        Y = rand(rng, 0:1, 4, 20)
        fit = fit_binomial_gllvm(Y; K = 1, iterations = 120)
        Ybad = Y .+ 1
        @test_throws ArgumentError fit_binomial_gllvm(Ybad; K = 1, iterations = 80)
        for f in (Y -> predict(fit, Y), Y -> fitted(fit, Y), Y -> residuals(fit, Y))
            err = try
                f(Ybad)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("predict:", err.msg)
        end
    end

    @testset "Poisson" begin
        Yp = rand.(rng, Poisson(2), 4, 18)
        fitp = fit_poisson_gllvm(Yp; K = 1, iterations = 100)
        Ypbad = Yp .- 3
        @test_throws ArgumentError fit_poisson_gllvm(Ypbad; K = 1, iterations = 80)
        @test_throws ArgumentError predict(fitp, Ypbad)
    end

    @testset "Gamma" begin
        Yg = exp.(randn(rng, 3, 15))
        fitg = fit_gamma_gllvm(Yg; K = 1, iterations = 100)
        Ygbad = copy(Yg)
        Ygbad[1, 1] = -0.5
        @test_throws ArgumentError fit_gamma_gllvm(Ygbad; K = 1, iterations = 80)
        @test_throws ArgumentError predict(fitg, Ygbad)
    end
end