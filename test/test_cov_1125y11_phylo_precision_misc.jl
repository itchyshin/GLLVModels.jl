# Lane cursor-cloud-1125y11 — phylo_precision.jl helpers
using GLLVModels, Test, LinearAlgebra

const _NW_ULTRA = "((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);"
const _NW_NONULTRA = "((A:0.1,B:0.2):0.2,(C:0.1,D:0.3):0.2);"

@testset "cov 1125y11 phylo precision helpers" begin
    phy = augmented_phy(_NW_ULTRA)

    @testset "_phylo_root_to_tip_heights" begin
        dist = GLLVModels._phylo_root_to_tip_heights(phy)
        @test length(dist) == phy.n_total
        @test dist[phy.root_index] == 0.0
        leaf_h = dist[phy.leaf_indices]
        @test all(>(0), leaf_h)
        @test maximum(abs.(leaf_h .- leaf_h[1])) <= sqrt(eps()) * max(1.0, maximum(abs.(leaf_h)))
    end

    @testset "ultrametric gate" begin
        h = GLLVModels._phylo_check_ultrametric_height(phy)
        @test h ≈ 0.3 rtol = 1e-12
        phy_bad = augmented_phy(_NW_NONULTRA)
        @test_throws ArgumentError GLLVModels._phylo_check_ultrametric_height(phy_bad)
    end

    pp_unscaled = PrecisionPhy(phy; correlation = false)
    @testset "PrecisionPhy correlation scaling" begin
        @test pp_unscaled.scale == 1.0
        pp1 = PrecisionPhy(phy; correlation = true)
        @test pp1.scale ≈ 0.3 rtol = 1e-12
        @test pp1.Q.nzval ≈ (0.3 ./ phy.scale) .* pp_unscaled.Q.nzval rtol = 1e-10
        phy_scaled = augmented_phy(_NW_ULTRA; correlation = true)
        pp2 = PrecisionPhy(phy_scaled; correlation = true)
        @test pp2.scale == phy_scaled.scale
        @test_throws ArgumentError PrecisionPhy(phy_scaled; correlation = false)
    end

    @testset "raw triplet constructor fields" begin
        pp = pp_unscaled
        @test pp.n_leaves == 4
        @test pp.n_aug == 6
        @test length(pp.species_aug_id) == 4
        @test pp.species_aug_id == collect(3:6)
        @test length(pp.node_labels) == pp.n_aug
        recomputed, shipped, diff = precision_logdet_check(pp)
        @test diff <= 1e-8
        @test shipped == pp.log_det
    end
end