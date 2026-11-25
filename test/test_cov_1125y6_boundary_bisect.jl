# Lane cursor-cloud-1125y6: boundary_inference.jl bisection and profile_ci_variance edges.
using Test, GLLVModels, Distributions

const G = GLLVModels

@testset "1125y6 boundary inference helpers" begin
    @testset "variance_lrt at boundary" begin
        r = variance_lrt(-10.0, -12.5; n_boundary = 2)
        @test r.LRT ≈ 5.0
        @test 0 < r.pvalue < 1
        @test r.n_boundary == 2
        r0 = variance_lrt(-5.0, -5.0)
        @test r0.LRT == 0.0
        @test r0.pvalue == 1.0
    end

    @testset "chibar2 mixture weights" begin
        p1 = chibar2_pvalue(3.84, 1)
        @test p1 ≈ 0.5 * ccdf(Chisq(1), 3.84) rtol = 1e-12
        p2 = chibar2_pvalue(5.0, 2)
        w = [binomial(2, j) * 2.0^(-2) * ccdf(Chisq(j), 5.0) for j in 1:2]
        @test p2 ≈ sum(w) rtol = 1e-12
        @test chibar2_pvalue(-1.0, 1) == 1.0
        @test chibar2_pvalue(0.0, 2) == 1.0
    end

    @testset "profile_ci_variance synthetic" begin
        ℓ_max = -10.0
        refit(v) = -10.0 - (v - 2.0)^2
        ci = profile_ci_variance(refit, 2.0, ℓ_max; level = 0.95, tol = 1e-6, maxit = 80)
        @test ci.lower >= 0.0
        @test ci.upper > ci.lower
        @test ci.level ≈ 0.95
        refit_flat(v) = -10.0
        ci_b = profile_ci_variance(refit_flat, 1.0, ℓ_max; level = 0.90, tol = 1e-5)
        @test ci_b.at_boundary
        @test ci_b.lower == 0.0
    end
end