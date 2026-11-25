using GLLVModels, Test, LinearAlgebra, SparseArrays

function _y7_valid_precision()
    Q = sparse([4.0 -1.0 -1.0 0.0;
                -1.0 3.0 0.0 -1.0;
                -1.0 0.0 3.0 0.0;
                0.0 -1.0 0.0 3.0])
    i, j, x = findnz(Q)
    return PrecisionPhy(i, j, x, 4, 2, ["a1", "a2", "s1", "s2"],
        logdet(cholesky(Symmetric(Matrix(Q)))), 1.25, [3, 4])
end

@testset "cov 1125y7 precision fit admission gates" begin
    @testset "valid payload passes and copies" begin
        phy = _y7_valid_precision()
        out = GLLVModels._validate_precision_fit_input(phy)
        @test out isa PrecisionPhy{Float64}
        @test out.n_leaves == phy.n_leaves
        @test out.scale == phy.scale
        @test out.log_det == phy.log_det
        @test out.species_aug_id == phy.species_aug_id
        @test out.Q !== phy.Q
        @test out.Q ≈ phy.Q
    end

    @testset "dimension and map errors" begin
        phy = _y7_valid_precision()
        bad = PrecisionPhy(phy.n_leaves, phy.n_aug, phy.Q, phy.log_det, phy.scale,
            phy.species_aug_id[1:end-1], phy.node_labels)
        @test_throws ArgumentError GLLVModels._validate_precision_fit_input(bad)
        bad2 = PrecisionPhy(phy.n_leaves, phy.n_aug, phy.Q, phy.log_det, phy.scale,
            [1, 1], phy.node_labels)
        @test_throws ArgumentError GLLVModels._validate_precision_fit_input(bad2)
        bad3 = PrecisionPhy(phy.n_leaves, phy.n_aug, sparse(phy.Q[1:3, 1:3]),
            phy.log_det, phy.scale, phy.species_aug_id, phy.node_labels[1:3])
        @test_throws ArgumentError GLLVModels._validate_precision_fit_input(bad3)
    end

    @testset "scale and logdet metadata" begin
        phy = _y7_valid_precision()
        @test_throws ArgumentError GLLVModels._validate_precision_fit_input(
            PrecisionPhy(phy.n_leaves, phy.n_aug, phy.Q, NaN, phy.scale,
                phy.species_aug_id, phy.node_labels))
        @test_throws ArgumentError GLLVModels._validate_precision_fit_input(
            PrecisionPhy(phy.n_leaves, phy.n_aug, phy.Q, phy.log_det, 0.0,
                phy.species_aug_id, phy.node_labels))
        wrong_ld = PrecisionPhy(phy.n_leaves, phy.n_aug, phy.Q, phy.log_det + 1.0,
            phy.scale, phy.species_aug_id, phy.node_labels)
        @test_throws ArgumentError GLLVModels._validate_precision_fit_input(wrong_ld)
    end

    @testset "labels must be nonempty" begin
        phy = _y7_valid_precision()
        labels = copy(phy.node_labels)
        labels[2] = ""
        bad = PrecisionPhy(phy.n_leaves, phy.n_aug, phy.Q, phy.log_det, phy.scale,
            phy.species_aug_id, labels)
        @test_throws ArgumentError GLLVModels._validate_precision_fit_input(bad)
    end
end