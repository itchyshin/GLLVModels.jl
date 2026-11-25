# Lane cursor-cloud-1125y15 — src/node_gradient.jl (node-frame per-species tools)
using GLLVModels, Test, Random, LinearAlgebra, SparseArrays

const G = GLLVModels

@testset "cov 1125y15 node per-species gradient and BLUPs" begin
    @testset "build_node_perspecies structure" begin
        phy = G.augmented_phy("((A:0.1,B:0.1):0.2,(C:0.15,D:0.15):0.1);")
        p = phy.n_leaves
        σ_phy = fill(0.3, p)
        st = G.build_node_perspecies(phy, σ_phy, 0.25)
        @test st.phy.n_leaves == p
        @test length(st.σ_phy) == p
        @test st.σ²_eps ≈ 0.25
        @test st.chol_Qcond isa SparseArrays.CHOLMOD.Factor
        @test st.cΛ̃ isa SparseArrays.CHOLMOD.Factor
        @test st.nb == phy.n_total - 1
    end

    @testset "grad_node_perspecies finite values" begin
        Random.seed!(1515)
        phy = G.augmented_phy("((A:0.2,B:0.25):0.3,(C:0.3,D:0.35):0.15);")
        p = phy.n_leaves
        σ_phy = fill(0.35, p)
        st = G.build_node_perspecies(phy, σ_phy, 0.4)
        y = randn(p)
        g = G.grad_node_perspecies(st, y, 0.1)
        @test length(g) == p
        @test all(isfinite, g)
        @test norm(g) > 0
        g2 = G.grad_node_perspecies(st, y, 0.1)
        @test g ≈ g2 rtol = 0 atol = 0
    end

    @testset "node_blups finite and consistent scale" begin
        phy = G.augmented_phy("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
        st = G.build_node_perspecies(phy, fill(0.2, phy.n_leaves), 0.5)
        û, ẑ = G.node_blups(st, randn(phy.n_leaves), 0.0)
        @test length(û) == st.nb
        @test length(ẑ) == phy.n_leaves
        @test all(isfinite, û)
        @test all(isfinite, ẑ)
        @test maximum(abs, ẑ) <= maximum(abs, [0.1, -0.2, 0.05]) + 1.0
    end

    @testset "takahashi feeds node BLUP solve" begin
        phy = G.augmented_phy("(A:0.15,B:0.2);")
        st = G.build_node_perspecies(phy, [0.25, 0.3], 0.6)
        dg = G.takahashi_diag(st.cΛ̃)
        @test length(dg) == st.nb
        @test all(dg .> 0)
        @test all(isfinite, dg)
    end
end