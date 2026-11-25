# Lane cursor-cloud-1125y11 — fit_random_effects.jl and random_effects.jl
using GLLVModels, Test, LinearAlgebra, Random

@testset "cov 1125y11 random effects engine" begin
    rng = MersenneTwister(112511)

    @testset "_code_grouping stable order and round-trip" begin
        g = [:a, :b, :a, :c, :b, :a]
        codes, levels = GLLVModels._code_grouping(g)
        @test levels == [:a, :b, :c]
        @test codes == [1, 2, 1, 3, 2, 1]
        @test levels[codes] == g
        codes2, levels2 = GLLVModels._code_grouping(string.(g))
        @test length(levels2) == 3
        @test levels2[codes2] == string.(g)
    end

    @testset "_unpack_chol_cov round-trip" begin
        q = 2
        npar = GLLVModels._chol_cov_npar(q)
        θ = [log(0.5), 0.1, log(0.8)]
        Σ, Lb = GLLVModels._unpack_chol_cov(θ, q)
        @test size(Σ) == (2, 2)
        @test Σ ≈ Σ' rtol = 0 atol = 0
        @test all(eigvals(Symmetric(Σ)) .> 0)
        @test Lb[1, 1] ≈ exp(θ[1])
        @test Lb[2, 1] == θ[2]
    end

    @testset "gaussian_grouped_intercept_loglik is finite" begin
        p, n = 4, 12
        Y = randn(rng, p, n)
        grouping = repeat(1:3; inner = 4)
        Λ = 0.3 .* randn(rng, p, 1)
        ll = gaussian_grouped_intercept_loglik(Y, grouping, Λ, 0.6, 0.25)
        @test isfinite(ll)
        ll2 = GLLVModels._grouped_intercept_loglik(Y,
            [findall(==(g), GLLVModels._code_grouping(grouping)[1]) for g in 1:3],
            Λ, 0.6, 0.25)
        @test ll ≈ ll2 rtol = 1e-12
    end

    @testset "intercept q=1 matches grouped slope at q=1" begin
        p, n = 3, 9
        Y = randn(rng, p, n)
        grouping = repeat(1:3; inner = 3)
        Z = ones(n, 1)
        Λ = 0.2 .* randn(rng, p, 1)
        σe = 0.55
        σu = 0.3
        ll_int = gaussian_grouped_intercept_loglik(Y, grouping, Λ, σe, σu)
        codes, _ = GLLVModels._code_grouping(grouping)
        idx = [findall(==(g), codes) for g in 1:maximum(codes)]
        Σb = fill(σu^2, 1, 1)
        ll_slp = GLLVModels._grouped_slope_loglik(Y, idx, Z, Λ, σe, Σb)
        @test ll_int ≈ ll_slp rtol = 1e-8 atol = 1e-8
    end

    @testset "fit_gaussian_random_slope tiny fit" begin
        p, n = 3, 15
        Y = 0.5 .* randn(rng, p, n)
        grouping = repeat(1:5; inner = 3)
        Z = hcat(ones(n), randn(rng, n))
        fit = fit_gaussian_random_slope(Y, grouping, Z; K = 1, iterations = 80, g_tol = 1e-4)
        @test fit isa GaussianRandomSlopeFit
        @test size(fit.Λ) == (p, 1)
        @test fit.q == 2
        @test fit.nlevels == 5
        @test isfinite(fit.loglik)
        @test fit.σ_eps > 0
        @test all(eigvals(Symmetric(fit.Σ_b)) .> 0)
        shown = sprint(show, fit)
        @test occursin("GaussianRandomSlopeFit", shown)
    end

    @testset "refusals" begin
        p, n = 2, 4
        Y = randn(p, n)
        @test_throws DimensionMismatch fit_gaussian_random_slope(Y, 1:3, ones(4, 1); K = 1)
        @test_throws ArgumentError fit_gaussian_random_slope(Y, 1:n, ones(n, 1); K = 0)
    end
end