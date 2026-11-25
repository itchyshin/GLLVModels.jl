using GLLVModels, Test, LinearAlgebra

@testset "cov 1125y7 edge incidence validation and operators" begin
    @testset "edge_phy parser errors" begin
        @test_throws ErrorException edge_phy("(A:0.1,B:0.2)")
        @test_throws ErrorException edge_phy("((A:0.1,B:0.2),(C:0.1,D:0.2),(E:0.1,F:0.2));")
        @test_throws ErrorException edge_phy("((A:0.1,B:0.2):0.3,(C:0.4,D:0.0):0.1);")
        @test_throws ErrorException edge_phy("((A:0.1,A:0.2):0.3,(C:0.4,D:0.5):0.1);")
        @test_throws ErrorException edge_phy("(A:0.0,B:0.2);")
    end

    @testset "valid ultrametric four-tip tree" begin
        phy = edge_phy("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.1);")
        @test phy.n_leaves == 4
        @test phy.n_nodes == 7
        @test length(phy.leaf_names) == 4
        @test phy.leaf_names == ["A", "B", "C", "D"]
        @test length(phy.branch_lengths) == phy.n_edges
        @test all(>(0), phy.branch_lengths)
    end

    @testset "log_det_Q closed form" begin
        phy = edge_phy("((A:0.2,B:0.2):0.1,(C:0.3,D:0.3):0.1);")
        σ² = 0.25
        analytic = GLLVModels.log_det_Q(phy, σ²)
        n_cond = phy.n_nodes - 1
        W = 1 ./ (σ² .* phy.branch_lengths)
        BW = Matrix(phy.B) * Diagonal(W)
        Q = BW * Matrix(phy.B)'
        root = phy.root_index
        keep = setdiff(1:phy.n_nodes, root)
        Qc = Q[keep, keep]
        dense = logdet(Symmetric(Qc))
        @test analytic ≈ dense rtol = 1e-10 atol = 1e-10
        @test isfinite(analytic)
    end

    @testset "Q_times_x dimensions and symmetry probe" begin
        phy = edge_phy("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.1);")
        σ² = 0.5
        x = randn(phy.n_nodes)
        @test_throws ArgumentError GLLVModels.Q_times_x(phy, σ², x[1:end-1])
        Qx = GLLVModels.Q_times_x(phy, σ², x)
        @test length(Qx) == phy.n_nodes
        onesv = ones(phy.n_nodes)
        q1 = GLLVModels.Q_times_x(phy, σ², onesv)
        @test norm(q1) < 1e-12
        @test dot(x, Qx) ≈ dot(Qx, x) rtol = 1e-12
    end

    @testset "sigma_phy_dense_edge matches ultrametric structure" begin
        phy = edge_phy("((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);")
        Σ = GLLVModels.sigma_phy_dense_edge(phy, 1.0)
        @test size(Σ) == (4, 4)
        @test all(diag(Σ) .> 0)
        @test Σ ≈ Σ' rtol = 0 atol = 1e-14
        @test Σ[1, 2] > Σ[1, 3]
        @test Σ[3, 4] == Σ[1, 2]
    end
end