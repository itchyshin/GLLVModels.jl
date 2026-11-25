# Coverage round 4: model_comparison.jl error paths and show (not covered elsewhere).
using Test
using GLLVModels
using Distributions

const G = GLLVModels

grab(f) = try
    f()
    nothing
catch e
    e
end

@testset "1125y anova guards and show" begin
    @testset "_anova_core argument checks" begin
        @test_throws ArgumentError G._anova_core([1, 2], [-1.0, -2.0], [1, 2]; test = :invalid)
        @test_throws ArgumentError G._anova_core([1, 2, 3], [-1.0, -2.0], [1, 2])
        @test_throws ArgumentError G._anova_core([1, 1], [-1.0, -2.0], [1, 1])
        tab = G._anova_core([10, 13], [-120.0, -100.0], [1, 2], 4; test = :chibar)
        @test tab.test[2] == "chibar (q=3)"
        @test tab.pvalue[2] !== missing
        tab2 = G._anova_core([10, 13], [-120.0, -100.0], [1, 2], 4; test = :none)
        @test tab2.test[2] == "none"
        @test tab2.pvalue[2] === missing
        tab_fix = G._anova_core([10, 12], [-120.0, -115.0], [1, 1], 4; test = :chibar)
        @test tab_fix.test[2] == "chisq"
        @test tab_fix.pvalue[2] !== missing
    end

    @testset "_anova_latent_rank and _anova_p" begin
        p, n, K = 3, 20, 1
        Y = randn(p, n)
        f = fit_gaussian_gllvm(Y; K = K, iterations = 8)
        @test G._anova_latent_rank(f) == K
        @test G._anova_p(f) == p
        @test G._anova_family_key(f) === nothing
        g = fit_poisson_gllvm(rand(0:3, p, n); K = 1, iterations = 8)
        @test G._anova_family_key(g) === nothing
        @test G._anova_family_key(f) === nothing
    end

    @testset "gllvm_anova global refusals" begin
        p, n = 4, 25
        Y = randn(p, n)
        f1 = fit_gaussian_gllvm(Y; K = 1, iterations = 8)
        f2 = fit_gaussian_gllvm(Y; K = 2, iterations = 8)
        @test_throws ArgumentError gllvm_anova(f1)
        Yp = [rand(Poisson(2)) for _ in 1:p, _ in 1:n]
        fp = fit_poisson_gllvm(Yp; K = 1, iterations = 8)
        @test_throws ArgumentError gllvm_anova(f1, fp)
        Xrt = zeros(p, n, p)
        for t in 1:p, s in 1:n
            Xrt[t, s, t] = 1.0
        end
        reml = fit_gaussian_reml(Y, Xrt; K = 1, iterations = 30)
        @test_throws ArgumentError gllvm_anova(f1, reml)
        Y2 = randn(p + 1, n)
        fbig = fit_gaussian_gllvm(Y2; K = 1, iterations = 8)
        @test_throws ArgumentError gllvm_anova(f1, fbig)
    end

    @testset "GllvmAnovaTable text/plain show" begin
        tab = G._anova_core([10, 13], [-120.0, -100.0], [1, 2], 4; test = :chibar)
        buf = sprint(show, "text/plain", tab)
        @test occursin("Likelihood-ratio comparison", buf)
        @test occursin("Model 1", buf)
        @test occursin("chibar", buf)
        @test occursin("Notes:", buf)
    end
end