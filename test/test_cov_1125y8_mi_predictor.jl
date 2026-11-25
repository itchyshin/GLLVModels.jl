# Lane cursor-cloud-1125y8: missing_predictor_multi.jl Cholesky pack and cond block.
using Test, GLLVModels, LinearAlgebra, Random, Distributions

const G = GLLVModels

@testset "cov 1125y8 missing predictor multi" begin
    rng = MersenneTwister(1125085)

    @testset "Cholesky pack round trip" begin
        for q in 1:3
            L = cholesky(Symmetric(randn(rng, q, q) * randn(rng, q, q)' + 3.0I)).L
            c = G._mi_pack_cholesky(L)
            L2 = G._mi_unpack_cholesky(c, q)
            @test L2 ≈ L atol = 1e-14
            @test length(c) == q * (q + 1) ÷ 2
        end
    end

    @testset "_safe_chol" begin
        M = [1.0 0.2; 0.2 1.0]
        @test G._safe_chol(M) !== nothing
        @test G._safe_chol([1.0 2.0; 2.0 1.0]) === nothing
        @test G._safe_chol(fill(NaN, 2, 2)) === nothing
    end

    @testset "_mi_cond_block" begin
        q = 3
        Σ = [1.0 0.3 0.1; 0.3 1.2 0.0; 0.1 0.0 0.9]
        μ = [0.1, -0.2, 0.0]
        b = [0.5, -0.3, 0.8]
        xvals = [0.2, -0.4, 0.6]
        m, P, off, lp, ok = G._mi_cond_block(Σ, μ, b, xvals, Int[], collect(1:q))
        @test ok
        @test off == 0.0 && lp == 0.0
        @test length(m) == q
        obs, mis = [1, 3], [2]
        m2, P2, off2, lp2, ok2 = G._mi_cond_block(Σ, μ, b, xvals, obs, mis)
        @test ok2 && isfinite(lp2)
        @test off2 ≈ dot(b[obs], xvals[obs]) atol = 1e-14
        ref = logpdf(MvNormal(μ[obs], Σ[obs, obs]), xvals[obs])
        @test lp2 ≈ ref atol = 1e-10
        m3, _, _, _, ok3 = G._mi_cond_block(Σ, μ, b, xvals, collect(1:q), Int[])
        @test ok3 && isempty(m3)
    end
end