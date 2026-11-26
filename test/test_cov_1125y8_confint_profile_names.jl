# Lane cursor-cloud-1125y8: confint_profile.jl term-name plumbing.
using Test, GLLVModels, Random, LinearAlgebra

const G = GLLVModels

@testset "cov 1125y8 confint profile term names" begin
    @testset "_profile_lambda_term_names pack order" begin
        names = G._profile_lambda_term_names("Λ_B", 4, 2)
        @test names == ["Λ_B[1,1]", "Λ_B[2,2]", "Λ_B[2,1]", "Λ_B[3,1]", "Λ_B[4,1]", "Λ_B[3,2]", "Λ_B[4,2]"]
        @test length(names) == G.rr_theta_len(4, 2)
        @test G._profile_lambda_term_names("Λ_W", 3, 1) == ["Λ_W[1,1]", "Λ_W[2,1]", "Λ_W[3,1]"]
        @test isempty(G._profile_lambda_term_names("Λ_phy", 5, 0))
    end

    rng = MersenneTwister(1125082)
    p, n, K = 5, 30, 1
    y = randn(rng, p, n)
    fit = fit_gaussian_gllvm(y; K = K, iterations = 8)

    @testset "_profile_all_term_names kinds" begin
        names, kinds = G._profile_all_term_names(fit)
        @test length(names) == length(kinds)
        @test "sigma_eps" in names
        @test any(n -> occursin("Lambda_B", n) || occursin("Λ_B", n), names)
        @test all(k -> k in (:linear, :log_sd), kinds)
        fit2 = fit_gaussian_gllvm(y; K = K, has_diag = true, iterations = 8)
        n2, k2 = G._profile_all_term_names(fit2)
        @test sum(==( :log_sd), k2) >= p + 1 # σ_eps plus diag scales
        @test length(n2) > length(names)
    end

    @testset "profile names align with unpack length" begin
        Λ = fit.pars.Λ
        p, K = size(Λ)
        θ = G.pack_lambda(Λ)
        labeled = G._profile_lambda_term_names("Λ_B", p, K)
        @test length(labeled) == length(θ)
    end
end