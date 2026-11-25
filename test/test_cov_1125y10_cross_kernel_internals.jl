# Lane cursor-cloud-1125y10: cross_kernel.jl private validators and sqrt helper.
using Test, GLLVModels, LinearAlgebra

const G = GLLVModels

@testset "1125y10 cross_kernel internals" begin
    A_H = [1.0 0.4 0.1; 0.4 1.0 0.2; 0.1 0.2 1.0]
    A_P = [1.0 0.25; 0.25 1.0]
    W = [1.0 0.5; 0.3 0.8; 0.1 0.2]

    @testset "_cross_kernel_as_matrix" begin
        m = G._cross_kernel_as_matrix(A_H, "A_H")
        @test m == A_H
        @test_throws ArgumentError G._cross_kernel_as_matrix([1.0 Inf], "A_H")
        @test_throws ArgumentError G._cross_kernel_as_matrix([1 2 3], "A_H")
        rect = G._cross_kernel_as_matrix(W, "W"; square = false)
        @test size(rect) == (3, 2)
    end

    @testset "_cross_kernel_check_correlation" begin
        G._cross_kernel_check_correlation(A_H, "A_H", 1e-8)
        nonsym = [1.0 0.9; 0.1 1.0]
        @test_throws ArgumentError G._cross_kernel_check_correlation(nonsym, "A_H", 1e-8)
        nonunit = [2.0 0.0; 0.0 2.0]
        @test_throws ArgumentError G._cross_kernel_check_correlation(nonunit, "A_H", 1e-8)
        nd = [1.0 0.0; 0.0 -0.5]
        @test_throws ArgumentError G._cross_kernel_check_correlation(nd, "A_P", 1e-8)
    end

    @testset "_cross_kernel_symmetric_sqrt" begin
        L = G._cross_kernel_symmetric_sqrt(A_H, 1e-10)
        @test L * L' ≈ A_H atol = 1e-8
        Lp = G._cross_kernel_symmetric_sqrt(A_P, 1e-10)
        @test tr(Lp' * Lp) ≈ tr(A_P) rtol = 1e-10
    end

    @testset "make_cross_kernel eps and rho" begin
        K = make_cross_kernel(A_H, A_P, W; rho = 0.0)
        @test size(K) == (5, 5)
        @test all(≈(1.0), diag(K))
        @test_throws ArgumentError make_cross_kernel(A_H, A_P, W; rho = NaN)
        @test_throws ArgumentError make_cross_kernel(A_H, A_P, W; eps = 0.0)
        @test_throws ArgumentError make_cross_kernel(A_H, A_P, W; eps = -1.0)
        @test_throws ArgumentError make_cross_kernel(A_H, A_P, W; eps = Inf)
    end

    @testset "spectral scaling keeps PSD at |rho|=1" begin
        K1 = Matrix(make_cross_kernel(A_H, A_P, W; rho = 1.0))
        Km = Matrix(make_cross_kernel(A_H, A_P, W; rho = -1.0))
        @test minimum(eigvals(Symmetric(K1))) >= -1e-5
        @test minimum(eigvals(Symmetric(Km))) >= -1e-5
        @test K1[1, 4] ≈ -Km[1, 4] rtol = 1e-10
    end

    @testset "off-diagonal blocks vanish at rho=0" begin
        K0 = Matrix(make_cross_kernel(A_H, A_P, W; rho = 0.0))
        @test K0[1:3, 4:5] ≈ zeros(3, 2)
        @test K0[4:5, 1:3] ≈ zeros(2, 3)
        @test K0[1:3, 1:3] ≈ A_H
        @test K0[4:5, 4:5] ≈ A_P
    end
end