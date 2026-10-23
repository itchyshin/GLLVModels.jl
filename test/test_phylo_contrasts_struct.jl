using GLLVModels, Test, LinearAlgebra, SparseArrays, ForwardDiff

# Coverage for the FelsensteinContrasts struct builder and contrast_transform.

@testset "felsenstein_contrasts struct" begin
    newicks = [
        "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);",                      # balanced, 4 leaves
        "((((A:0.1,B:0.1):0.1,C:0.2):0.1,D:0.3):0.1,E:0.4);",          # ladder, 5 leaves
        "((A:0.3,B:0.7):0.2,(C:0.5,(D:0.1,E:0.9):0.4):0.6);",          # unequal branches
    ]

    for nw in newicks
        @testset "$nw" begin
            tree = augmented_phy(nw)
            p = tree.n_leaves
            fc = felsenstein_contrasts(tree)
            fc_str = felsenstein_contrasts(nw)

            # (a) Newick method matches AugmentedPhy method
            @test fc_str.U == fc.U
            @test fc_str.weights == fc.weights

            # (b) matches felsenstein_contrast_matrix
            U, w = felsenstein_contrast_matrix(tree)
            @test fc.U == U
            @test fc.weights == w
            @test size(fc.U) == (p - 1, p)

            # (c) log|det T| for T = [c_root'; U]
            T = vcat(fc.c_root', Matrix(fc.U))
            @test fc.logabsdet_T ≈ logabsdet(T)[1] atol = 1e-10

            # (d) root centroid weights form a convex combination
            @test all(fc.c_root .>= 0)
            @test sum(fc.c_root) ≈ 1

            # (e) T Σ T' is diagonal with (t_root, weights)
            Σ = sigma_phy_dense(tree; σ²_phy = 1.0)
            D = T * Σ * T'
            @test D ≈ Diagonal(vcat(fc.t_root, fc.weights)) atol = 1e-10

            # (f) contrast_transform is U * y and passes Duals through
            y = reshape(collect(1.0:(p * 3)), p, 3)
            @test contrast_transform(y, fc.U) == fc.U * y
            yd = ForwardDiff.Dual.(y, 1.0)
            out = contrast_transform(yd, fc.U)
            @test eltype(out) <: ForwardDiff.Dual
            @test ForwardDiff.value.(out) ≈ fc.U * y
        end
    end

    # (g) a one-leaf tree is rejected
    @testset "1-leaf tree throws" begin
        @test_throws ErrorException felsenstein_contrasts(augmented_phy("A:0.5;"))
        err = try
            felsenstein_contrasts(augmented_phy("A:0.5;")); nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("≥ 2 leaves", err.msg)
    end
end
