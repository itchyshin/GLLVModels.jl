using GLLVModels, Test, LinearAlgebra, SparseArrays

function _pmv_y7_fixture()
    Q = sparse([4.0 -1.0 -1.0 0.0;
                -1.0 3.0 0.0 -1.0;
                -1.0 0.0 3.0 0.0;
                0.0 -1.0 0.0 3.0])
    i, j, x = findnz(Q)
    phy = PrecisionPhy(i, j, x, 4, 2, ["a1", "a2", "s1", "s2"],
        logdet(cholesky(Symmetric(Matrix(Q)))), 1.0, [3, 4])
    return phy
end

@testset "cov 1125y7 bridge precision multivariate options" begin
    phy = _pmv_y7_fixture()
    m, n_leaves = 4, phy.n_leaves
    base = Dict{String,Any}("species_id" => [1, 2, 1, 2])

    @testset "_bridge_pmv_option_dict" begin
        @test GLLVModels._bridge_pmv_option_dict(nothing) == Dict{String,Any}()
        d = GLLVModels._bridge_pmv_option_dict(Dict("species_id" => [1, 2]))
        @test haskey(d, "species_id")
        @test_throws ArgumentError GLLVModels._bridge_pmv_option_dict("not a dict")
        @test_throws ArgumentError GLLVModels._bridge_pmv_option_dict(
            Dict("species_id" => [1], "unknown_key" => 1))
        dup = Dict{String,Any}("species_id" => [1])
        dup["species_id"] = [2]
        @test GLLVModels._bridge_pmv_option_dict(dup)["species_id"] == [2]
    end

    @testset "_bridge_pmv_mode" begin
        @test GLLVModels._bridge_pmv_mode("bareLowRank") == :barelowrank
        @test GLLVModels._bridge_pmv_mode("  explicitUnique ") == :explicitunique
        @test_throws ArgumentError GLLVModels._bridge_pmv_mode("shared")
    end

    @testset "_bridge_pmv_iterations" begin
        @test GLLVModels._bridge_pmv_iterations(0) == 0
        @test GLLVModels._bridge_pmv_iterations(42.0) == 42
        @test_throws ArgumentError GLLVModels._bridge_pmv_iterations(-1)
        @test_throws ArgumentError GLLVModels._bridge_pmv_iterations(1.5)
        @test_throws ArgumentError GLLVModels._bridge_pmv_iterations(NaN)
    end

    @testset "_bridge_pmv_options happy path" begin
        opts = GLLVModels._bridge_pmv_options(base, m, n_leaves)
        @test opts.species_id == [1, 2, 1, 2]
        @test opts.mode == :barelowrank
        @test opts.iterations == 400
        @test opts.ci_method == "none"
        @test opts.residual_mode == :trait
        @test 0 < opts.ci_level < 1
        @test opts.g_tol > 0
    end

    @testset "_bridge_pmv_options validation" begin
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(Dict(), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            Dict("species_id" => "bad"), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            Dict("species_id" => [1.0, 2]), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            Dict("species_id" => [1, 2, 3]), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            Dict("species_id" => [0, 1]), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            Dict("species_id" => [1, 5]), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            merge(base, Dict("mode" => "shared")), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            merge(base, Dict("g_tol" => 0.0)), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            merge(base, Dict("g_tol" => Inf)), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            merge(base, Dict("iterations" => -3)), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            merge(base, Dict("ci_method" => "profile")), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            merge(base, Dict("ci_level" => 1.5)), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            merge(base, Dict("ci_level" => 0.0)), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            merge(base, Dict("phylo_model" => "sparse")), m, n_leaves)
        @test_throws ArgumentError GLLVModels._bridge_pmv_options(
            merge(base, Dict("residual_mode" => "bogus")), m, n_leaves)
    end

    @testset "_bridge_pmv_ci_flat" begin
        Y = randn(3, 4)
        phy = _pmv_y7_fixture()
        species_id = [1, 2, 1, 2]
        fit = GLLVModels.fit_precision_multivariate(Y, phy; rank = 1,
            species_id = species_id, iterations = 0, g_tol = 1e-4)
        none = GLLVModels._bridge_pmv_ci_flat(fit, "none", 0.95)
        @test none.status == "not_requested"
        @test isempty(none.target_names)
        wald = GLLVModels._bridge_pmv_ci_flat(fit, "wald", 0.95)
        @test wald.status isa String
        @test length(wald.target_names) == length(wald.estimate)
    end
end