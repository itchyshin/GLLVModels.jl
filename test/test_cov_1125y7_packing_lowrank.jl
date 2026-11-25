using GLLVModels, Test, LinearAlgebra, ForwardDiff, Random

@testset "cov 1125y7 packing and lowrank cholesky" begin
    @testset "rr_theta_len and init_theta_rr" begin
        @test GLLVModels.rr_theta_len(0, 0) == 0
        @test GLLVModels.rr_theta_len(5, 0) == 0
        @test GLLVModels.rr_theta_len(5, 1) == 5
        @test GLLVModels.rr_theta_len(5, 2) == 9
        @test GLLVModels.rr_theta_len(10, 3) == 27
        θ = GLLVModels.init_theta_rr(4, 2)
        @test length(θ) == 7
        @test θ[1] == θ[2] == 0.5
        @test all(==(0.0), θ[3:end])
        @test_throws ArgumentError GLLVModels.init_theta_rr(2, 3)
        @test_throws ArgumentError GLLVModels.init_theta_rr(5, -1)
    end

    @testset "pack_lambda round-trip and errors" begin
        p, K = 6, 3
        Λ = zeros(p, K)
        for k in 1:K
            Λ[k, k] = 0.5 + 0.1k
            for i in (k + 1):p
                Λ[i, k] = 0.07 * (i - k)
            end
        end
        θ = GLLVModels.pack_lambda(Λ)
        Λ2 = GLLVModels.unpack_lambda(θ, p, K)
        @test Λ2 ≈ Λ
        @test size(Λ2, 2) == K && size(Λ2, 1) == p
        for k in 1:K - 1
            for i in 1:k
                @test Λ2[i, k + 1] == 0.0
            end
        end
        @test_throws ArgumentError GLLVModels.unpack_lambda(θ[1:end-1], p, K)
        @test_throws ArgumentError GLLVModels.unpack_lambda(θ, p, K + 1)
        @test_throws ArgumentError GLLVModels.unpack_lambda(θ, p, p + 1)
    end

    @testset "_lower_triangular_loadings preserves ΛΛ'" begin
        rng = MersenneTwister(1125)
        Λraw = randn(rng, 8, 3)
        L = GLLVModels._lower_triangular_loadings(Λraw)
        @test size(L) == (8, 3)
        @test L ≈ tril(L)
        for k in 1:3
            @test L[k, k] >= 0
        end
        @test Λraw * Λraw' ≈ L * L' atol = 1e-10
        @test GLLVModels._lower_triangular_loadings(Λraw[:, 1:1]) ≈ Λraw[:, 1:1]
    end

    @testset "ForwardDiff through pack/unpack" begin
        p, K = 4, 2
        θ0 = GLLVModels.init_theta_rr(p, K)
        f(θ) = sum(GLLVModels.unpack_lambda(θ, p, K))
        g = ForwardDiff.gradient(f, θ0)
        @test all(isfinite, g)
        @test length(g) == length(θ0)
    end

    @testset "low_rank_chol vs dense" begin
        rng = MersenneTwister(77)
        p, K = 12, 3
        Λ = 0.3 .* randn(rng, p, K)
        d = 0.5 .+ rand(rng, p)
        F = GLLVModels.low_rank_chol(Λ, d)
        M = Λ * Λ' + Diagonal(d)
        x = randn(rng, p)
        @test F \ x ≈ M \ x atol = 1e-10 rtol = 1e-10
        B = randn(rng, p, 4)
        @test F \ B ≈ M \ B atol = 1e-10 rtol = 1e-10
        @test logdet(F) ≈ logdet(Symmetric(M)) atol = 1e-9 rtol = 1e-9
        y = similar(x)
        mul!(y, F, x)
        @test y ≈ M * x atol = 1e-10
        @test size(F) == (p, p)
        @test eltype(F) == Float64
    end

    @testset "low_rank_chol validation" begin
        Λ = ones(4, 2)
        @test_throws DimensionMismatch GLLVModels.low_rank_chol(Λ, ones(3))
        @test_throws ArgumentError GLLVModels.low_rank_chol(Λ, fill(0.0, 4))
        @test_throws ArgumentError GLLVModels.low_rank_chol(Λ, fill(-1.0, 4))
        out = similar(ones(4))
        F = GLLVModels.low_rank_chol(Λ, ones(4))
        @test_throws DimensionMismatch GLLVModels.ldiv!(out, F, ones(3))
        @test_throws DimensionMismatch GLLVModels.ldiv!(ones(3), F, ones(4))
    end
end