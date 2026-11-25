# Lane cursor-cloud-1125y12 — extract_gamma.jl, animal_dep.jl
using GLLVModels, Test, LinearAlgebra, Random, Distributions

@testset "cov 1125y12 extract_Gamma and animal_dep" begin
    rng = MersenneTwister(11251204)
    p, n = 6, 8
    Y = randn(rng, p, n)

    @testset "extract_Gamma refusals on ordinary GllvmFit" begin
        fit = fit_gaussian_gllvm(Y; K = 1, iterations = 60, g_tol = 1e-4)
        @test_throws ArgumentError extract_Gamma(fit; row_traits = 1:3, col_traits = 4:6)
    end

    Araw = Float64[
        1.0 0.2 0.1 0.0 0.0 0.0
        0.2 1.0 0.15 0.0 0.0 0.0
        0.1 0.15 1.0 0.4 0.0 0.0
        0.0 0.0 0.4 1.0 0.25 0.0
        0.0 0.0 0.0 0.25 1.0 0.2
        0.0 0.0 0.0 0.0 0.2 1.0
    ]

    @testset "extract_Gamma index guards" begin
        fit = fit_animal_dep_gllvm(Y, Araw; iterations = 60, g_tol = 1e-4)
        Γ = extract_Gamma(fit; row_traits = 1:3, col_traits = 4:6)
        @test size(Γ) == (3, 3)
        @test all(isfinite, Γ)
        @test_throws ArgumentError extract_Gamma(fit; row_traits = [0], col_traits = [1])
        @test_throws ArgumentError extract_Gamma(fit; row_traits = [1], col_traits = [p + 1])
    end

    @testset "fit_animal_dep_gllvm guards" begin
        @test_throws ArgumentError fit_animal_dep_gllvm(Y, randn(p, p + 1))
        @test_throws ArgumentError fit_animal_dep_gllvm(Y, I(p); family = Poisson())
        @test_throws ArgumentError fit_animal_dep_gllvm(Y, I(p); K = 2)
        @test_throws ArgumentError fit_animal_dep_gllvm(Y, I(p); K_phy = 2)
        phy = augmented_phy("((A:0.1,B:0.2):0.3,C:0.1);")
        @test_throws ArgumentError fit_animal_dep_gllvm(Y, phy)
    end

    @testset "fit_animal_dep tiny smoke" begin
        fit = fit_animal_dep_gllvm(Y, Araw; jitter = 1e-5, iterations = 50, g_tol = 1e-4)
        @test isfinite(fit.logLik)
        @test fit.pars.Λ_phy !== nothing
        @test size(fit.pars.Λ_phy, 1) == p
    end
end