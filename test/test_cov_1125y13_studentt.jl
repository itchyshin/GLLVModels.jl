# Lane cursor-cloud-1125y13 — src/families/studentt.jl
using GLLVModels, Test, Random, Distributions, ForwardDiff

const G = GLLVModels

@testset "cov 1125y13 studentt" begin
    rng = MersenneTwister(1125135)
    ν, σ = 5.0, 0.8
    f = StudentTFamily(ν, σ)
    μ = 0.3
    y = 0.5
    n = 1
    me = 1.0

    @testset "GLM hooks" begin
        s = G._glm_score(f, μ, n, me, y)
        w = G._glm_weight(f, μ, n, me)
        @test isfinite(s)
        @test w > 0
        r = y - μ
        @test s ≈ (ν + 1) * r / (ν * σ^2 + r^2)
        @test w ≈ (ν + 1) / ((ν + 3) * σ^2)
        @test G._laplace_mode_should_backtrack(f)
        @test G._laplace_mode_robust(f)
        @test G._clamp_mu(f, μ) == μ
    end

    @testset "logpdf finite and AD-clean" begin
        ll = G._glm_logpdf(f, μ, n, y)
        @test isfinite(ll)
        g = ForwardDiff.derivative(μ_ -> G._glm_logpdf(f, μ_, n, y), μ)
        @test isfinite(g)
    end

    @testset "StudentT aliases" begin
        @test StudentT() isa StudentTFamily
        @test G.default_link(StudentT(4.0)) isa IdentityLink
    end

    @testset "tiny fit" begin
        p, n = 3, 25
        Y = μ .+ σ .* randn(rng, p, n)
        fit = fit_studentt_gllvm(Y; K = 1, nu = 7.0, iterations = 100, g_tol = 1e-4)
        @test isfinite(fit.loglik)
        @test fit.converged || fit.gradient_norm < 0.1
        fit_est = fit_studentt_gllvm(Y; K = 1, iterations = 80, g_tol = 1e-4)
        @test isfinite(fit_est.loglik)
    end

    @testset "refusals" begin
        Y = randn(2, 5)
        @test_throws ArgumentError fit_studentt_gllvm(Y; nu = 0.0, K = 1)
    end
end