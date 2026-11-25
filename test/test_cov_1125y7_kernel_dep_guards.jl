using GLLVModels, Test, LinearAlgebra, Random

@testset "cov 1125y7 kernel dep/indep guard rails" begin
    rng = MersenneTwister(7)
    p, n = 3, 9
    Y = randn(rng, p, n)
    groups = repeat(1:3; inner = 3)
    K = [1.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.0]

    @testset "fit_kernel_dep_gllvm validation" begin
        @test_throws ArgumentError fit_kernel_dep_gllvm(Y, K)
        @test_throws ArgumentError fit_kernel_dep_gllvm(Y, K, groups[1:8])
        @test_throws ArgumentError fit_kernel_dep_gllvm(Y, K[1:2, :], groups)
        @test_throws ArgumentError fit_kernel_dep_gllvm(Y, K, fill(0, n))
        @test_throws ArgumentError fit_kernel_dep_gllvm(Y, K, fill(4, n))
        @test_throws ArgumentError fit_kernel_dep_gllvm(Y, K, groups; rho = 0.2)
        @test_throws ArgumentError fit_kernel_dep_gllvm(Y, K, groups; K_phy = 1)
        @test_throws ArgumentError fit_kernel_dep_gllvm(Y, K, groups; family = GLLVModels.Poisson())
    end

    @testset "dep versus indep knobs differ" begin
        @test_throws ArgumentError fit_kernel_dep_gllvm(Y, K, groups; common = true)
        dep = fit_kernel_dep_gllvm(Y, K, groups; name = :dep, iterations = 0, g_tol = 1e-4)
        indep = fit_kernel_indep_gllvm(Y, K, groups; name = :indep, iterations = 0, g_tol = 1e-4)
        @test dep isa GaussianSourcesFit
        @test indep isa GaussianSourcesFit
        @test dep.sources[1].mode == :dep
        @test indep.sources[1].mode == :indep
        @test isfinite(dep.loglik)
        @test isfinite(indep.loglik)
    end
end