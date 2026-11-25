# Coverage round 4: re_sd.jl refusal branches and Laplace precision guard.
using Test
using GLLVModels
using Distributions

const G = GLLVModels

@testset "1125y re_sd guards" begin
    Y = randn(4, 12)
    fit = fit_gaussian_gllvm(Y; K = 1, iterations = 10)
    sd = latent_score_sd(fit, Y)
    @test size(sd) == (12, 1)
    @test all(x -> x > 0, sd[1, :])

    @testset "K_W tier refusal" begin
        y_w = randn(3, 10)
        fit_w = fit_gaussian_gllvm(y_w; K = 1, K_W = 1, iterations = 8)
        @test_throws ArgumentError latent_score_sd(fit_w, y_w)
    end

    Yp = [rand(Poisson(2)) for _ in 1:3, _ in 1:8]
    fitp = fit_poisson_gllvm(Yp; K = 1, iterations = 10)

    @testset "_laplace_re_precision_site hessian guard" begin
        z = zeros(1)
        @test_throws ArgumentError G._laplace_re_precision_site(
            Poisson(), view(Yp, :, 1), ones(3), fitp.Λ, fitp.β, fitp.link, z;
            hessian = :invalid)
    end

    @testset "Poisson laplace SD matrix shape" begin
        sd2 = latent_score_sd(fitp, Yp)
        @test size(sd2) == (8, 1)
        @test all(isfinite, sd2)
    end
end