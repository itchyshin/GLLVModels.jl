using GLLVModels, Test, Random, LinearAlgebra

# Wave 2 sweep (w2-structured): NaN or Inf in the response of three fitters that have no
# missing-data route returned a fit with loglik = -Inf (or NaN), converged = false and no
# error. They now raise an ArgumentError naming the offending cells, like fit_branch_re.
@testset "non-finite response is refused (phylo Gaussian, SPDE Gaussian, relaxed clock)" begin
    nw = "((A:0.3,B:0.3):0.2,((C:0.2,D:0.2):0.1,(E:0.2,F:0.2):0.1):0.2);"
    rng = MersenneTwister(11)

    @testset "fit_phylo_gaussian" begin
        y = randn(rng, 6)
        @test isfinite(fit_phylo_gaussian(nw, y).negll)           # finite input still fits
        for bad in (NaN, Inf, -Inf)
            z = copy(y); z[2] = bad
            err = try fit_phylo_gaussian(nw, z); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("non-finite values at indices [2]", sprint(showerror, err))
        end
        # the AugmentedPhy method goes through the same check
        @test_throws ArgumentError fit_phylo_gaussian(augmented_phy(nw), [y[1:5]; NaN])
    end

    @testset "fit_spde_gaussian" begin
        co = rand(rng, 30, 2) .* 10
        nodes, tris = spde_mesh_delaunay(co)
        y = randn(rng, 30)
        @test isfinite(fit_spde_gaussian(y, nodes, tris, co).loglik)
        for bad in (NaN, Inf)
            z = copy(y); z[4] = bad
            err = try fit_spde_gaussian(z, nodes, tris, co); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("non-finite values at indices [4]", sprint(showerror, err))
        end
    end

    @testset "fit_relaxed_clock" begin
        ephy = edge_phy(nw)
        Y = randn(rng, 6, 4)
        for bad in (NaN, Inf)
            Z = copy(Y); Z[2, 3] = bad
            err = try fit_relaxed_clock(ephy, Z); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("non-finite values", sprint(showerror, err))
            @test occursin("first at (row, column) = (2, 3)", sprint(showerror, err))
        end
    end
end
