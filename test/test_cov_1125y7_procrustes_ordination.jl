using GLLVModels, Test, LinearAlgebra, Statistics, Random

@testset "cov 1125y7 ordination uncertainty alignment" begin
    @testset "_procrustes_align rotation invariance" begin
        rng = MersenneTwister(11257)
        n, K = 30, 3
        B = randn(rng, n, K)
        R = qr(randn(rng, K, K)).Q
        A = B * R' .+ 0.01 .* randn(rng, n, K)
        Aligned = GLLVModels._procrustes_align(A, B)
        @test size(Aligned) == size(B)
        @test all(isfinite, Aligned)
        @test mean(Aligned; dims = 1) ≈ mean(B; dims = 1) atol = 1e-12
        @test cor(vec(Aligned), vec(B)) > 0.99
    end

    @testset "_procrustes_align translation" begin
        B = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        A = B .+ 100.0
        Aligned = GLLVModels._procrustes_align(A, B)
        @test Aligned ≈ B atol = 1e-10
    end

    @testset "_procrustes_align reflection" begin
        B = [1.0 0.0; 0.0 1.0; -1.0 0.0]
        A = [-1.0 0.0; 0.0 -1.0; 1.0 0.0]
        Aligned = GLLVModels._procrustes_align(A, B)
        dist_before = norm(A - B)
        dist_after = norm(Aligned - B)
        @test dist_after < dist_before
        @test dist_after < 1e-10
    end
end