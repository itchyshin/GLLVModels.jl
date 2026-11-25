# Lane cursor-cloud-1125y14 — src/likelihood_contrasts.jl, src/likelihood_edge_incidence.jl, src/phylo_contrasts.jl
using GLLVModels, Test, Random, LinearAlgebra

const G = GLLVModels

@testset "cov 1125y14 likelihood contrasts and edge" begin
    rng = MersenneTwister(11251406)
    newick = "((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);"
    tree_aug = augmented_phy(newick)
    tree_edge = edge_phy(newick)
    p, n = 4, 6
    Y = randn(rng, p, n)
    Λ = 0.1 .* randn(rng, p, 1)
    σ = 0.5

    @testset "gaussian_marginal_loglik_contrasts guards" begin
        Ybad = randn(rng, p + 1, n)
        @test_throws ArgumentError G.gaussian_marginal_loglik_contrasts(
            Ybad, Λ, σ; tree = tree_aug, σ²_phy = 1.0)
        ll = G.gaussian_marginal_loglik_contrasts(Y, Λ, σ; tree = tree_aug, σ²_phy = 0.8)
        @test isfinite(ll)
        ll_phy = G.gaussian_marginal_loglik_contrasts(Y, Λ, σ; tree = tree_aug, σ²_phy = 0.5,
            σ_phy = ones(p))
        @test isfinite(ll_phy)
    end

    @testset "contrasts vs dense (no phy block)" begin
        ll_d = G.gaussian_marginal_loglik(Y, Λ, σ)
        ll_c = G.gaussian_marginal_loglik_contrasts(Y, Λ, σ; tree = tree_aug, σ²_phy = 0.0)
        @test isfinite(ll_d)
        @test isfinite(ll_c)
    end

    @testset "gaussian_marginal_loglik_edge_phy" begin
        σ_phy = ones(p)
        ll_e = G.gaussian_marginal_loglik_edge_phy(Y, Λ, σ; phy = tree_edge,
            σ²_phy = 0.5, σ_phy = σ_phy)
        @test isfinite(ll_e)
        @test_throws ArgumentError G.gaussian_marginal_loglik_edge_phy(
            Y, Λ, σ; phy = tree_edge, σ²_phy = 0.5)
        Ybad = randn(rng, 3, n)
        @test_throws ArgumentError G.gaussian_marginal_loglik_edge_phy(
            Ybad, Λ, σ; phy = tree_edge, σ_phy = σ_phy)
    end

    @testset "felsenstein_contrasts" begin
        fc = G.felsenstein_contrasts(tree_aug)
        @test fc isa G.FelsensteinContrasts
        @test size(fc.U, 1) == p - 1
        @test length(fc.weights) == p - 1
        @test all(fc.weights .> 0)
        @test isfinite(fc.logabsdet_T)
        U2, w2 = G.felsenstein_contrast_matrix(newick)
        @test size(U2) == (p - 1, p)
        @test length(w2) == p - 1
    end
end