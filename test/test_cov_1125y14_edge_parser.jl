# Lane cursor-cloud-1125y14 — src/edge_incidence.jl (parser guards, solvers)
using GLLVModels, Test, LinearAlgebra, Random

const G = GLLVModels

@testset "cov 1125y14 edge parser and solvers" begin
    @testset "edge_phy Newick guards" begin
        @test_throws ErrorException G.edge_phy("((A:0.1,B:0.2)")
        @test_throws ErrorException G.edge_phy("(A:0.1,B:0.2);extra")
        @test_throws ErrorException G.edge_phy("(A:0.1,A:0.2);")
        @test_throws ErrorException G.edge_phy("(A:0.1,B:0.2,C:0.3);")
        @test_throws ErrorException G.edge_phy("(A:0.0,B:0.2);")
    end

    @testset "edge_phy minimal tree" begin
        tree = G.edge_phy("(A:0.2,B:0.3);")
        @test tree.n_leaves == 2
        @test tree.n_nodes == 3
        @test tree.leaf_names == ["A", "B"]
        bl = tree.branch_lengths
        @test length(bl) == 2
        @test all(bl .> 0)
    end

    @testset "log_det_Q and solve_Q" begin
        tree = G.edge_phy("((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.1);")
        σ² = 0.4
        ld = G.log_det_Q(tree, σ²)
        @test isfinite(ld)
        b = randn(tree.n_nodes - 1)
        x = G.solve_Q(tree, σ², b)
        @test length(x) == tree.n_nodes - 1
        @test all(isfinite, x)
        @test_throws ArgumentError G.solve_Q(tree, σ², randn(tree.n_nodes))
    end

    @testset "sigma_phy_dense_edge symmetry" begin
        tree = G.edge_phy("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.1);")
        Σ = G.sigma_phy_dense_edge(tree, 1.0)
        @test Σ ≈ Σ' rtol = 1e-12
        @test all(diag(Σ) .> 0)
        @test size(Σ) == (tree.n_leaves, tree.n_leaves)
    end

    @testset "_node_depths" begin
        tree = G.edge_phy("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
        d = G._node_depths(tree, 1.0)
        @test length(d) == tree.n_nodes
        @test all(d .>= 0)
        @test d[tree.root_index] ≈ 0.0 atol = 1e-12
    end

    @testset "whitespace tolerance" begin
        t1 = G.edge_phy("( A : 0.1 , B : 0.2 );")
        t2 = G.edge_phy("(A:0.1,B:0.2);")
        @test t1.n_leaves == t2.n_leaves
        @test t1.leaf_names == t2.leaf_names
    end
end