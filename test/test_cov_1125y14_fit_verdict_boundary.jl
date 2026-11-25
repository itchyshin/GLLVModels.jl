# Lane cursor-cloud-1125y14 — src/fit_verdict.jl, src/boundary_inference.jl, src/flag_unreliable_loadings.jl
using GLLVModels, Test, Random, Distributions

const G = GLLVModels

@testset "cov 1125y14 fit_verdict and boundary" begin
    @testset "_nll_failed and _fit_verdict" begin
        @test G._nll_failed(1e12)
        @test G._nll_failed(1e11)
        @test !G._nll_failed(100.0)
        @test !G._nll_failed(-50.0)
        ll, conv, it = G._fit_verdict(1.0e12, true, 0)
        @test ll == -Inf
        @test conv == false
        @test it == 0
        ll2, conv2, it2 = G._fit_verdict(3.5, true, 12)
        @test ll2 ≈ -3.5
        @test conv2
        @test it2 == 12
        ll3, conv3, it3 = G._fit_verdict(4.0, false, 7)
        @test ll3 ≈ -4.0
        @test !conv3
        @test it3 == 7
    end

    @testset "chibar2_pvalue and variance_lrt" begin
        @test G.chibar2_pvalue(0.0, 1) == 1.0
        @test G.chibar2_pvalue(-1.0, 2) == 1.0
        @test_throws ArgumentError G.chibar2_pvalue(NaN, 1)
        @test_throws ArgumentError G.chibar2_pvalue(3.0, 0)
        p1 = G.chibar2_pvalue(2.71, 1)
        @test 0 < p1 < 1
        p2 = G.chibar2_pvalue(5.0, 2)
        @test p2 < p1
        out = G.variance_lrt(10.0, 8.0; n_boundary = 1)
        @test out.LRT ≈ 4.0
        @test 0 < out.pvalue < 1
        @test out.n_boundary == 1
        @test_throws ArgumentError G.variance_lrt(NaN, 1.0)
    end

    @testset "_bisect_root and profile_ci_variance" begin
        g = x -> 1 - x^2
        root = G._bisect_root(g, 0.0, 2.0, 1e-10, 80)
        @test abs(root - 1.0) < 1e-4
        refit = v -> 10.0 - (v - 0.5)^2
        ci = G.profile_ci_variance(refit, 0.5, 10.0; level = 0.95, lower = 0.0)
        @test ci.lower >= 0.0
        @test ci.upper > ci.lower
        @test ci.level == 0.95
    end

    @testset "flag_unreliable_loadings vector API" begin
        rows = [(
            estimate = 0.05, lower = -0.02, upper = 0.12, pinned = false,
        )]
        out = G.flag_unreliable_loadings(rows; null_region = (-0.1, 0.1))
        @test out[1].unreliable == true
        rows2 = [(
            estimate = 0.5, lower = 0.2, upper = 0.8, pinned = false,
        )]
        out2 = G.flag_unreliable_loadings(rows2)
        @test out2[1].unreliable == false
        @test_throws ArgumentError G.flag_unreliable_loadings([(estimate = 1.0,)])
        @test_throws ArgumentError G._flag_null_region((0.2, 0.1))
        @test_throws ArgumentError G._flag_null_region([1.0])
    end

    @testset "flag_unreliable_loadings fit guards" begin
        rng = MersenneTwister(11251401)
        p, n = 5, 12
        Y = randn(rng, p, n)
        M = fill(NaN, p, 1)
        M[1, 1] = 0.0
        fit = fit_gaussian_gllvm(Y; K = 1, lambda_constraint = M, iterations = 40, g_tol = 1e-3)
        rows = G.flag_unreliable_loadings(fit, Y)
        @test length(rows) == p
        @test_throws ArgumentError G.flag_unreliable_loadings(fit, Y; level = :species)
        @test_throws ArgumentError G.flag_unreliable_loadings(fit, Y; conf_level = 1.5)
        fit_plain = fit_gaussian_gllvm(Y; K = 1, iterations = 20, g_tol = 1e-3)
        @test_throws ArgumentError G.flag_unreliable_loadings(fit_plain, Y)
        Ybad = randn(rng, p + 1, n)
        @test_throws DimensionMismatch G.flag_unreliable_loadings(fit, Ybad)
    end
end