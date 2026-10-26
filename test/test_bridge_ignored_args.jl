using GLLVModels, Test, Random

# Refs #741. bridge_fit must not silently drop arguments: each route either uses
# every supplied argument or throws ArgumentError.

const _IGN_NEWICK = "(((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1):0.1,((E:0.1,F:0.1):0.1,(G:0.1,H:0.1):0.1):0.1);"

@testset "bridge_fit rejects silently ignored arguments (#741)" begin
    pp = GLLVModels.PrecisionPhy(GLLVModels.augmented_phy(_IGN_NEWICK); correlation = false)
    p = pp.n_leaves
    Random.seed!(741)
    yv = randn(p)

    @testset "phylo diagnostic hook" begin
        # baseline: the supported call still works
        base = GLLVModels.bridge_fit(; y = yv, family = "gaussian", phylo = pp)
        @test base.diagnostic_only
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = yv, phylo = pp, d = 3)
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = yv, phylo = pp,
                                                         X = randn(p, 1, 2))
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = yv, phylo = pp,
                                                         X_lv = randn(1, 2))
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = yv, phylo = pp,
                                                         mask = trues(p, 1))
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = yv, phylo = pp, N = ones(p, 1))
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = yv, phylo = pp,
                                                         trait_names = string.(1:p))
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = yv, phylo = pp,
                                                         unit_names = ["u"])
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = yv, phylo = pp,
            options = Dict("ci_method" => "wald"))
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = yv, phylo = pp,
            options = Dict("reml" => true))
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = yv, phylo = pp,
            options = Dict("coef_fixed" => [true]))
        # an explicit no-op value is still accepted
        ok = GLLVModels.bridge_fit(; y = yv, phylo = pp, d = 1,
            options = Dict("ci_method" => "none", "reml" => false))
        @test ok.diagnostic_only
    end

    @testset "reml on routes that do not read it" begin
        Y = randn(4, 30) .+ [1.0, 2.0, 3.0, 4.0]
        Yc = rand(0:5, 4, 30)
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = Yc, family = "poisson", d = 1,
            options = Dict("reml" => true))
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = Y, family = "gaussian", d = 1,
            X_lv = randn(30, 1), options = Dict("reml" => true))
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = Y, family = "gaussian", d = 1,
            X = randn(4, 30, 2), options = Dict("reml" => true))
        # the supported REML route is untouched
        br = GLLVModels.bridge_fit(; y = Y, family = "gaussian", d = 1,
            options = Dict("reml" => true))
        @test br.model == "gaussian_reml_rr"
    end

    @testset "non-Dict options" begin
        Y = randn(4, 30)
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = Y, family = "gaussian", d = 1,
            options = (reml = true,))
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = yv, phylo = pp,
            options = (ci_method = "wald",))
    end
end
