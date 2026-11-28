# Clearer messages for three inputs that already fail (#1165). Message-only: the exception type
# stays the same except the NaN mesh point, which was a raw Bowyer-Watson ErrorException blaming
# duplicate points and is now an ArgumentError naming the cause.

using GLLVModels, Test, Random

_msg(f) = try; f(); ""; catch e; sprint(showerror, e); end

@testset "Newick without branch lengths (#1165)" begin
    nolen = "((t1,t2),(t3,t4));"
    for (name, f) in (("augmented_phy", augmented_phy), ("edge_phy", edge_phy))
        m = _msg(() -> f(nolen))
        @test occursin("no branch lengths", m)
        @test occursin("branch length must be > 0", m)      # wording that was there is kept
        @test_throws ErrorException f(nolen)                # same exception type as before
    end
    # a real zero-length tip branch in a tree that has other lengths keeps the old message
    for f in (augmented_phy, edge_phy)
        m = _msg(() -> f("((t1:0.0,t2:1):1,(t3:1,t4:1):1);"))
        @test occursin("branch length must be > 0; node", m)
        @test !occursin("no branch lengths", m)
    end
    # a valid tree is unaffected
    @test augmented_phy("((t1:1,t2:1):1,(t3:1,t4:1):1);").n_leaves == 4
end

@testset "polytomy error points to the route that accepts it (#1165)" begin
    poly = "(t1:1,t2:1,t3:1,(t4:0.5,t5:0.5):0.5);"
    for f in (augmented_phy, edge_phy)
        m = _msg(() -> f(poly))
        @test startswith(m, "tree is not binary (got 7 nodes for 5 leaves; expected 9)")
        @test occursin("fit_phylo_latent_gllvm", m)
        @test_throws ErrorException f(poly)
    end
end

@testset "spde_mesh_delaunay names a non-finite coordinate (#1165)" begin
    co = rand(MersenneTwister(30), 30, 2) .* 10
    for bad in (NaN, Inf)
        c = copy(co); c[3, 1] = bad
        m = _msg(() -> GLLVModels.spde_mesh_delaunay(c))
        @test startswith(m, "ArgumentError")
        @test occursin("row 3", m)
        @test !occursin("duplicate", m)
    end
    nodes, tris = GLLVModels.spde_mesh_delaunay(co)         # finite points unchanged
    @test size(nodes, 1) == 30 && size(tris, 1) > 0
end
