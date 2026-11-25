# Lane cursor-cloud-1125y14 — src/families/com_poisson.jl
using GLLVModels, Test, Random, ForwardDiff

const G = GLLVModels

@testset "cov 1125y14 com poisson" begin
    rng = MersenneTwister(11251402)

    @testset "compoisson_logz identities" begin
        @test G.compoisson_logz(log(2.0), 1.0) ≈ 2.0 rtol = 1e-10
        @test G.compoisson_logz(-20.0, 1.0) ≈ exp(-20.0) rtol = 1e-6
        z1 = G.compoisson_logz(2.0, 1.2)
        z2 = G.compoisson_logz(2.0f0, 1.2f0)
        @test z1 ≈ Float64(z2) rtol = 1e-6
        z_big = G.compoisson_logz(12.0, 2.0)
        @test isfinite(z_big)
        @test z_big > 0
    end

    @testset "compoisson_logpdf and score/weight" begin
        for (η, ν, y) in ((0.5, 1.0, 3), (1.2, 1.5, 0), (2.0, 2.0, 5))
            lp = G.compoisson_logpdf(y, η, ν)
            @test isfinite(lp) || y > 0 && η < -5
            s, w = G._cmp_score_weight(y, η, ν)
            @test isfinite(s)
            @test w >= 1e-8
        end
        fd = η -> G.compoisson_logpdf(4, η, 1.3)
        s_fd = ForwardDiff.derivative(fd, 0.8)
        s, _ = G._cmp_score_weight(4, 0.8, 1.3)
        @test s ≈ s_fd rtol = 1e-4
    end

    @testset "tiny fit_compoisson_gllvm" begin
        p, n = 4, 16
        Y = rand(rng, 0:6, p, n)
        fit = fit_compoisson_gllvm(Y; K = 1, iterations = 60, g_tol = 1e-3)
        @test fit isa COMPoissonFit
        @test isfinite(fit.loglik)
        shown = sprint(show, fit)
        @test occursin("COMPoissonFit", shown)
        pred = predict(fit, Y; type = :rate)
        @test size(pred) == size(Y)
        @test all(pred .>= 0)
    end

    @testset "fit guards" begin
        Y = rand(rng, 0:3, 3, 5)
        @test_throws ArgumentError fit_compoisson_gllvm(Y; K = -1)
        Ybad = fill(-1, 2, 3)
        @test_throws ArgumentError fit_compoisson_gllvm(Ybad; K = 1)
    end
end