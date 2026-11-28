# refs #1548 — DeltaLogNormal and HurdlePoisson post-fit refuse out-of-support Y.
using Test, GLLVModels, Random, Distributions
using GLLVModels: predict, fitted, residuals

@testset "#1548 post-fit response support" begin
    @testset "DeltaLogNormal" begin
        Random.seed!(11270402)
        Y = zeros(Float64, 3, 24)
        for t in 1:3, s in 1:24
            Y[t, s] = rand() < 0.2 ? 0.0 : exp(0.3 * randn())
        end
        fit = fit_gllvm(Y; family = DeltaLogNormal(), K = 1, iterations = 80)
        Ybad = copy(Y)
        Ybad[2, 3] = -0.25
        @test_throws ArgumentError fit_gllvm(Ybad; family = DeltaLogNormal(), K = 1, iterations = 20)
        @test_throws ArgumentError predict(fit, Ybad)
        @test_throws ArgumentError fitted(fit, Ybad)
    end

    @testset "HurdlePoisson" begin
        Random.seed!(11270403)
        Yh = rand(0:4, 3, 28)
        fit_h = fit_gllvm(Yh; family = HurdlePoisson(), K = 1, iterations = 70)
        Yhbad = copy(Yh)
        Yhbad[1, 1] = -1
        @test_throws ArgumentError fit_gllvm(Yhbad; family = HurdlePoisson(), K = 1, iterations = 20)
        @test_throws ArgumentError predict(fit_h, Yhbad)
        @test_throws ArgumentError fitted(fit_h, Yhbad)
        @test_throws ArgumentError residuals(fit_h, Yhbad)
    end
end