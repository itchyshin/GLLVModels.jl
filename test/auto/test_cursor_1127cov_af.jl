# Coverage slice for src/[a-f]*.jl (Cursor 27 Nov 2026). Auto-discovered via test/auto/.
using GLLVModels
using Test
using LinearAlgebra
using Random
using Optim

const _ANIMAL_DEP_A_1127 = Float64[
    1.0 0.0 0.0 0.0
    0.0 1.0 0.0 0.0
    0.0 0.0 1.0 0.5
    0.0 0.0 0.5 1.0
]

@testset "Cursor 1127cov src/a-f helpers" begin
    GM = GLLVModels

    @testset "fixed_coefficients mask helpers" begin
        q = 4
        @test GM._fixed_zero_mask(nothing, q, "β_fixed") == falses(4)
        @test GM._fixed_zero_mask(Bool[1, 0, 0, 1], q, "β_fixed") == Bool[1, 0, 0, 1]
        @test GM._fixed_zero_mask([2, 4], q, "β_fixed") == Bool[0, 1, 0, 1]
        @test GM._fixed_zero_mask(Float64[1, 0, 0, 1], q, "β_fixed") == Bool[1, 0, 0, 1]
        @test_throws ArgumentError GM._fixed_zero_mask([1, 0, 0, 1], q, "β_fixed")
        @test GM._fixed_zero_mask(Dict("2" => 0, 3 => 0.0), q, "β_fixed") == Bool[0, 1, 1, 0]

        @test_throws ArgumentError GM._fixed_zero_mask(-1, q, "β_fixed")
        @test_throws ArgumentError GM._fixed_zero_mask([1, 5], q, "β_fixed")
        @test_throws ArgumentError GM._fixed_zero_mask(Bool[1, 0], q, "β_fixed")
        @test_throws ArgumentError GM._fixed_zero_mask([0.5, 1.5], 2, "β_fixed")
        @test_throws ArgumentError GM._fixed_zero_mask(Dict(2 => 1), q, "β_fixed")
        @test_throws ArgumentError GM._fixed_zero_mask(-1, -1, "β_fixed")

        X = zeros(2, 3, 4)
        mask = Bool[1, 0, 0, 0]
        Xf, free = GM._slice_fixed_X(X, mask)
        @test size(Xf, 3) == 3 && free == [2, 3, 4]
        @test_throws ArgumentError GM._slice_fixed_X(X, Bool[1, 0])

        @test GM._expand_fixed_zero([2.0, 3.0], Bool[0, 1, 0]) ≈ [2.0, 0.0, 3.0]
        @test_throws ArgumentError GM._expand_fixed_zero([1.0], Bool[0, 1, 0])

        @test GM._fixed_status(Bool[1, 0, 1]) == ["fixed", "estimated", "fixed"]

        pars = (β_fixed = Bool[1, 0],)
        @test GM._pars_fixed_mask(pars, :β_fixed, 2) == Bool[1, 0]
        @test GM._pars_fixed_mask((;), :β_fixed, 2) == falses(2)
        @test_throws ArgumentError GM._pars_fixed_mask((β_fixed = Bool[1],), :β_fixed, 2)

        @test GM._fixed_init_free(nothing, falses(2), "init") === nothing
        @test GM._fixed_init_free([1.0, 2.0], Bool[0, 1], "init") ≈ [1.0]
        @test_throws ArgumentError GM._fixed_init_free([1.0], falses(2), "init")
    end

    @testset "boundary_inference refuses NaN inputs (gllvmTMB chibar parity)" begin
        @test_throws ArgumentError chibar2_pvalue(NaN, 1)
        @test_throws ArgumentError variance_lrt(NaN, -100.0)
        @test_throws ArgumentError variance_lrt(-100.0, NaN)
    end

    @testset "fit_coevolution_blockna validation" begin
        T_H, T_P = 2, 2
        n_H, n_P = 3, 2
        A_H = Matrix{Float64}(I, n_H, n_H)
        A_P = Matrix{Float64}(I, n_P, n_P)
        K_HP = zeros(n_H, n_P)
        Y_HH = randn(T_H, n_H)
        Y_PP = randn(T_P, n_P)
        @test_throws ArgumentError fit_coevolution_blockna(Y_HH, Y_PP, A_H, A_P, K_HP; d = 0)
        @test_throws ArgumentError fit_coevolution_blockna(Y_HH, Y_PP, A_H[1:2, 1:2], A_P, K_HP; d = 1)
        @test_throws ArgumentError fit_coevolution_blockna(Y_HH, Y_PP, A_H, A_P, K_HP[1:2, :]; d = 1)
    end

    @testset "fit_animal_dep_gllvm custom Σ_animal must match p" begin
        p = size(_ANIMAL_DEP_A_1127, 1)
        y = randn(p, 10)
        Σ_bad = Matrix{Float64}(I, p + 1, p + 1)
        @test_throws ArgumentError fit_animal_dep_gllvm(y, _ANIMAL_DEP_A_1127; Σ_animal = Σ_bad)
        Σ_ok = relatedness_cov(_ANIMAL_DEP_A_1127; jitter = 1e-8)
        fit = fit_animal_dep_gllvm(y, _ANIMAL_DEP_A_1127; Σ_animal = Σ_ok)
        @test fit isa GllvmFit
    end

    @testset "make_cross_kernel extra validation branches" begin
        A_H = [1.0 0.2; 0.2 1.0]
        A_P = [1.0 0.1; 0.1 1.0]
        W = [1.0 0.0; 0.5 0.0]
        @test_throws ArgumentError make_cross_kernel(A_H, A_P, W; rho = NaN)
        @test_throws ArgumentError make_cross_kernel(A_H, A_P, W; eps = 0.0)
        @test_throws ArgumentError make_cross_kernel(A_H, A_P, W; eps = -1.0)
        A_nf = [1.0 Inf; 0.2 1.0]
        @test_throws ArgumentError make_cross_kernel(A_nf, A_P, W)
        @test_throws ArgumentError make_cross_kernel([1.0 0.2 0.1], A_P, W)
    end

    @testset "_fit_verdict screens Optim plateau at iteration zero" begin
        f(θ) = 1.0e12
        res = optimize(f, [0.0], LBFGS(); autodiff = :forward)
        ll, conv, it = GM._fit_verdict(res)
        @test ll == -Inf && conv == false && it == Optim.iterations(res)
    end

    @testset "coevolution Kronecker precompute and fit guards" begin
        K_bad = [1.0 0.0; 0.0 0.0]
        @test_throws ArgumentError GLLVModels._coevolution_kron_precompute(K_bad)
        Y = randn(4, 5)
        K = Matrix{Float64}(I, 4, 4)
        @test_throws ArgumentError fit_coevolution_gaussian(Y, K[1:3, 1:3]; d = 1)
        @test_throws ArgumentError fit_coevolution_gaussian(Y, K; d = 0)
        @test_throws ArgumentError fit_coevolution_gaussian(Y, K; d = 5)
    end

    @testset "em_fa accepts user-supplied λ_init and ψ_init" begin
        Random.seed!(1127)
        p, n, K = 6, 40, 2
        y = randn(p, n)
        Λ0 = 0.2 .* randn(p, K)
        ψ0 = fill(0.8, p)
        Λ1, ψ1, ll, nit, conv = em_fa(y, K; λ_init = Λ0, ψ_init = ψ0, max_iter = 15, tol = 1e-6)
        @test size(Λ1) == (p, K)
        @test length(ψ1) == p
        @test all(>(0), ψ1)
        @test isfinite(ll)
        @test nit >= 1
    end
end