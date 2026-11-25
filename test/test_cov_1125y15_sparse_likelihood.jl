# Lane cursor-cloud-1125y15 — src/sparse_phy.jl, src/likelihood_sparse_phy.jl
using GLLVModels, Test, Random, LinearAlgebra, Distributions, SparseArrays

const G = GLLVModels

@testset "cov 1125y15 sparse phy parser and likelihood" begin
    @testset "Newick refusals" begin
        @test_throws ErrorException G.augmented_phy("((A:0.1,B:0.2)")
        @test_throws ErrorException G.augmented_phy("(A:0.0,B:0.2);")
        @test_throws ErrorException G.augmented_phy("(A:0.1,A:0.2);")
        @test_throws ErrorException G.augmented_phy("(A:0.1,B:0.2,C:0.3);")
    end

    @testset "make_phy and random_balanced_tree" begin
        phy = G.random_balanced_tree(4; branch_length = 0.15)
        @test phy.n_leaves == 4
        @test all(phy.branch_lengths .> 0)
        Σ = G.sigma_phy_dense(phy; σ²_phy = 1.1)
        @test size(Σ) == (4, 4)
        @test Σ ≈ Σ' rtol = 1e-12
        @test all(diag(Σ) .> 0)
    end

    @testset "_apply_phy_correlation ultrametric gate" begin
        ultra = G.augmented_phy("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
        corr = G._apply_phy_correlation(ultra; correlation = true)
        @test isfinite(corr.scale) && corr.scale > 0
        @test SparseArrays.nnz(corr.Q_topology) >= SparseArrays.nnz(ultra.Q_topology)
        nonultra = G.augmented_phy("(A:0.2,B:0.3);")
        @test_throws ArgumentError G._apply_phy_correlation(nonultra; correlation = true)
    end

    @testset "gaussian_marginal_loglik_sparse_phy vs dense (K_phy=1)" begin
        Random.seed!(15)
        phy = G.augmented_phy("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
        p, n, K_B = phy.n_leaves, 12, 1
        σ²_phy, σ_eps = 0.6, 0.4
        Λ_B = randn(p, K_B)
        Λ_phy = reshape(randn(p), p, 1)
        Σ_phy = G.sigma_phy_dense(phy; σ²_phy = σ²_phy)
        y = randn(p, n)
        ll_d = G.gaussian_marginal_loglik(y, Λ_B, σ_eps; Λ_phy = Λ_phy, Σ_phy = Σ_phy)
        ll_s = G.gaussian_marginal_loglik_sparse_phy(y, Λ_B, σ_eps; Λ_phy = Λ_phy, phy = phy, σ²_phy = σ²_phy)
        @test isfinite(ll_s)
        @test ll_s ≈ ll_d rtol = 1e-10 atol = 1e-8
    end

    @testset "sparse path with zero phylo loadings" begin
        phy = G.augmented_phy("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
        p, n = phy.n_leaves, 5
        y = randn(p, n)
        Λ_B = fill(0.01, p, 1)
        Λ_phy = zeros(p, 1)
        ll0 = G.gaussian_marginal_loglik_sparse_phy(y, Λ_B, 0.5; Λ_phy = Λ_phy, phy = phy, σ²_phy = 1.0)
        Σ = G.sigma_phy_dense(phy; σ²_phy = 1.0)
        ll_ref = G.gaussian_marginal_loglik(y, Λ_B, 0.5; Λ_phy = Λ_phy, Σ_phy = Σ)
        @test ll0 ≈ ll_ref rtol = 1e-10 atol = 1e-8
    end

    @testset "whitespace-tolerant Newick" begin
        a = G.augmented_phy("( A : 0.1 , B : 0.2 );")
        b = G.augmented_phy("(A:0.1,B:0.2);")
        @test a.leaf_names == b.leaf_names
        @test a.branch_lengths ≈ b.branch_lengths rtol = 1e-12
    end
end