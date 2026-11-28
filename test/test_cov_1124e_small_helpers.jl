# Coverage: cheap argument checks in packing, PPCA rotation, the Woodbury
# factor, and kernel × indep admission. No successful kernel fit.
using Test
using GLLVModels
using LinearAlgebra
using Random

grab(f) = try
    f()
    nothing
catch e
    e
end

@testset "packing, PPCA, Woodbury, kernel admission (1124e)" begin
    @testset "loading pack length and K = 0" begin
        e = grab(() -> GLLVModels.unpack_lambda([1.0, 2.0], 4, 1))
        @test e isa ArgumentError
        @test occursin("unpack_lambda: θ has length 2; expected 4 for p=4, K=1", e.msg)
        Λ0 = GLLVModels.unpack_lambda(Float64[], 3, 0)
        @test size(Λ0) == (3, 0)
        @test GLLVModels.pack_lambda(Λ0) == Float64[]
        @test GLLVModels.init_theta_rr(3, 0) == Float64[]
        @test GLLVModels.rr_theta_len(3, 0) == 0
    end

    @testset "PPCA rank and lower-triangular rotation" begin
        y = randn(MersenneTwister(1124), 4, 6)
        e = grab(() -> ppca_init(y, 0))
        @test e isa ArgumentError
        @test e.msg == "K must be ≥ 1"
        e = grab(() -> ppca_init(y, 4))
        @test e isa ArgumentError
        @test e.msg == "PPCA requires K < p; got K=4, p=4"
        e = grab(() -> ppca_init(y[:, 1:2], 5))
        @test e isa ArgumentError
        @test occursin("got K=5, p=4", e.msg)
        Λa, sa = ppca_init(y, 2)
        Λb, sb = ppca_init(y, 2; lower_tri = false)
        @test sa == sb
        @test sa > 0
        @test size(Λb) == (4, 2)
        @test Λa * Λa' ≈ Λb * Λb' atol = 1e-10
        @test all(abs(Λa[i, k]) < 1e-8 for k in 1:2 for i in 1:(k - 1))
        @test all(Λa[k, k] > 0 for k in 1:2)
        e = grab(() -> GLLVModels.rotate_to_lower_triangular([1.0 0.0 0.2; 0.0 1.0 -0.3]))
        @test e isa ArgumentError
        @test e.msg == "rotate_to_lower_triangular requires p ≥ K; got p=2, K=3"
        Λ = [1.0 -2.0; 0.5 0.3; -0.2 0.4]
        R = GLLVModels.rotate_to_lower_triangular(Λ)
        @test size(R) == (3, 2)
        @test abs(R[1, 2]) < 1e-8
        @test R[1, 1] > 0
        @test R[2, 2] > 0
        @test R * R' ≈ Λ * Λ' atol = 1e-10
    end

    @testset "low_rank_chol shape" begin
        Λ = [1.0 0.0; 0.5 0.2]
        e = grab(() -> GLLVModels.low_rank_chol(Λ, [1.0]))
        @test e isa DimensionMismatch
        @test occursin("length(d) = 1 must equal size(Λ, 1) = 2", e.msg)
    end

    @testset "kernel × indep admission" begin
        Y = randn(MersenneTwister(7), 2, 3)
        rectangular = [1.0 0.2; 0.2 1.0; 0.0 0.1]
        e = grab(() -> fit_kernel_indep_gllvm(Y, rectangular, [1, 1, 1]))
        @test e isa ArgumentError
        @test occursin("K must be square; got (3, 2)", e.msg)
        K = [1.0 0.2; 0.2 1.0]
        e = grab(() -> fit_kernel_indep_gllvm(Y, K, fill(true, 3)))
        @test e isa ArgumentError
        @test occursin("groups must be one-based indices in 1:2", e.msg)
        e = grab(() -> fit_kernel_indep_gllvm(Y, K, fill(1.0, 3)))
        @test e isa ArgumentError
        @test occursin("groups must be one-based indices in 1:2", e.msg)
    end
end
