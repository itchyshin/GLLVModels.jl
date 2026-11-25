# Lane cursor-cloud-1125y13 — src/joint_phylo_grouped_gaussian.jl gate/design helpers
using GLLVModels, Test, LinearAlgebra, SparseArrays

include(joinpath(@__DIR__, "fixtures", "destination_b_joint_phylo_grouping.jl"))

const G = GLLVModels

@testset "cov 1125y13 joint_phylo_grouped helpers" begin
    fixture = destination_b_joint_phylo_grouping_fixture()
    Q = sparse(fixture.Q)
    ii, jj, xx = findnz(Q)
    phy = PrecisionPhy(ii, jj, xx, fixture.n_aug, fixture.n_tip, fixture.node_labels,
        fixture.precision_logdet, fixture.applied_scale, fixture.species_aug_id)

    @testset "_joint_phylo_grouped_phy_gate" begin
        @test G._joint_phylo_grouped_phy_gate(phy, fixture.species_id, length(fixture.species_id)) === nothing
        bad_phy = PrecisionPhy(ii, jj, xx, phy.n_aug, phy.n_leaves, phy.node_labels,
            phy.log_det, phy.scale, [3, 3])
        @test_throws ArgumentError G._joint_phylo_grouped_phy_gate(bad_phy, [1, 2], 2)
    end

    @testset "_joint_phylo_grouped_phy_design" begin
        n, p = length(fixture.species_id), size(fixture.response, 1)
        factor = Matrix{Float64}(I, p, 1)
        D = G._joint_phylo_grouped_phy_design(n, p, phy, fixture.species_id, factor)
        @test size(D) == (n * p, phy.n_aug)
        @test nnz(D) > 0
    end

    @testset "_joint_phylo_grouped_prior" begin
        P = G._joint_phylo_grouped_prior(phy, 1, 2)
        @test size(P, 1) == size(P, 2)
        @test nnz(P) > 0
        P0 = G._joint_phylo_grouped_prior(phy, 1, 0)
        @test size(P0) == (phy.n_aug, phy.n_aug)
    end

    @testset "_joint_phylo_grouped_ordinary_design" begin
        inc = fixture.crossed_incidence
        var = fixture.grouped_variance
        designs = G._joint_phylo_grouped_ordinary_design(length(fixture.species_id),
            size(fixture.response, 1), [inc], [var])
        @test length(designs) == 1
        @test size(designs[1], 1) == length(fixture.species_id) * size(fixture.response, 1)
        @test_throws DimensionMismatch G._joint_phylo_grouped_ordinary_design(
            length(fixture.species_id), size(fixture.response, 1), [inc], [var[1:end-1]])
        n_obs = length(fixture.species_id)
        @test_throws DimensionMismatch G._joint_phylo_grouped_ordinary_design(
            n_obs, size(fixture.response, 1), [zeros(3, 2)], [var])
    end
end