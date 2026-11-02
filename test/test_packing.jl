using GLLVModels
using Test
using Random

# Bring the packing helpers and ForwardDiff into scope explicitly. The
# helpers are defined in src/packing.jl but the module's export list
# lives in src/GLLVModels.jl which is owned by a sibling agent — so we
# reach in by name here. ForwardDiff is a dep of GLLVModels.jl so we get
# it via the parent module.
using GLLVModels: rr_theta_len, pack_lambda, unpack_lambda, init_theta_rr
using GLLVModels: _lower_triangular_loadings
using LinearAlgebra: svd
const ForwardDiff = GLLVModels.ForwardDiff

@testset "packing" begin
    @testset "rr_theta_len matches the engine formula" begin
        @test rr_theta_len(5, 1) == 5
        @test rr_theta_len(5, 2) == 5*2 - 2*1÷2          # 9
        @test rr_theta_len(10, 3) == 10*3 - 3*2÷2        # 27
    end

    @testset "pack/unpack round-trip" begin
        Random.seed!(1)
        for p in (3, 5, 10), K in (1, 2, min(3, p))
            n = rr_theta_len(p, K)
            θ = randn(n)
            Λ = unpack_lambda(θ, p, K)
            @test size(Λ) == (p, K)
            # strict upper triangle is zero
            for i in 1:K, k in (i+1):K
                @test Λ[i, k] == 0
            end
            # round-trip
            θ2 = pack_lambda(Λ)
            @test θ ≈ θ2 atol=1e-12
        end
    end

    @testset "init defaults match gllvmTMB::init_rr_theta" begin
        for p in (5, 10), K in (1, 2, 3)
            θ₀ = init_theta_rr(p, K)
            @test length(θ₀) == rr_theta_len(p, K)
            @test all(θ₀[1:K] .== 0.5)            # diagonals = 0.5
            @test all(θ₀[(K+1):end] .== 0.0)      # strict lower = 0
        end
    end

    @testset "AD-friendly: works with ForwardDiff.Dual" begin
        θ = collect(1.0:9.0)  # p=5, K=2 → length 9
        f = θ -> sum(unpack_lambda(θ, 5, 2))
        g = ForwardDiff.gradient(f, θ)
        @test all(isfinite, g)
    end

    @testset "SVD loadings start is QR-rotated, not truncated (#865)" begin
        Random.seed!(865)
        for K in (2, 3)
            p, n = 8, 40
            Zc = randn(p, K) * randn(K, n) .+ 0.1 .* randn(p, n)
            F = svd(Zc)
            L = F.U[:, 1:K] .* (F.S[1:K] ./ sqrt(n))'   # SVD loadings, as in fit_gllvm_cov
            target = L * L'
            # The old start (pack straight away) drops the SVD's upper triangle.
            old = unpack_lambda(pack_lambda(L), p, K)
            @test !isapprox(old * old', target; rtol = 1e-3)
            # Rotated start keeps LL' and is lower-triangular with positive diagonal.
            R = _lower_triangular_loadings(L)
            @test R * R' ≈ target
            @test all(R[k, j] == 0 for j in 1:K for k in 1:(j - 1))
            @test all(R[k, k] > 0 for k in 1:K)
            new = unpack_lambda(pack_lambda(R), p, K)
            @test new * new' ≈ target
        end
        # K = 0 and K = 1 pass through unchanged.
        @test size(_lower_triangular_loadings(zeros(5, 0))) == (5, 0)
        L1 = randn(5, 1)
        @test _lower_triangular_loadings(L1) == L1
        # Rank-deficient input (fewer informative columns than K) stays finite.
        L0 = hcat(randn(6), zeros(6))
        R0 = _lower_triangular_loadings(L0)
        @test R0 * R0' ≈ L0 * L0'
        @test all(isfinite, R0)
    end
end
