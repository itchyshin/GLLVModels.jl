# Lane 1124d: source covariance construction / design validation; bridge option helpers.
using Test, GLLVModels, LinearAlgebra

const G = GLLVModels

@testset "1124d source_fit validation" begin
    C = [1.0 0.1; 0.1 1.2]
    P = [1.0 0.0; 1.0 0.0; 0.0 1.0]

    @testset "SourceCovariance(C, P) direct projection" begin
        s = SourceCovariance(C, P; mode = :indep)
        @test s.projection == P
        @test s.mode === :indep
    end

    @testset "SourceCovariance(C, P) refusals" begin
        @test_throws DimensionMismatch SourceCovariance(C, fill(0.0, 3, 3))
        @test_throws ArgumentError SourceCovariance(C, fill(Inf, 3, 2))
        @test_throws ArgumentError SourceCovariance([1.0 2.0; 2.0 1.0], P)
        @test_throws ArgumentError SourceCovariance(C, P; mode = :indep, unique = true)
        @test_throws ArgumentError SourceCovariance(C, P; mode = :dep, common = true)
        @test_throws ArgumentError SourceCovariance(C, P; mode = :latent, rank = true)
        @test_throws ArgumentError SourceCovariance(C, P; mode = :indep, rank = 2)
    end

    @testset "_source_fixed_sigma" begin
        @test G._source_fixed_sigma(0.5) == 0.5
        for bad in (0.0, -1.0, Inf, NaN, true, "x")
            @test_throws ArgumentError G._source_fixed_sigma(bad)
        end
    end

    @testset "_source_mean_design" begin
        p, n, q = 3, 4, 2
        D = G._source_mean_design(reshape(randn(p, n, q), p * n, q), p, n)
        @test size(D) == (p * n, q)
        D3 = G._source_mean_design(randn(p, n, q), p, n)
        @test size(D3) == (p * n, q)
        @test_throws DimensionMismatch G._source_mean_design(randn(p * n - 1, q), p, n)
        @test_throws DimensionMismatch G._source_mean_design(randn(p - 1, n, q), p, n)
        @test_throws ArgumentError G._source_mean_design("not a matrix", p, n)
        @test_throws ArgumentError G._source_mean_design(fill(NaN, p * n, q), p, n)
        @test_throws DimensionMismatch G._source_mean_design([1 1; 1 1], p, n)
    end

    @testset "_source_coefficient_names" begin
        @test G._source_coefficient_names(nothing, 3) == ["x1", "x2", "x3"]
        @test G._source_coefficient_names(nothing, 2; default_trait_names = true) == ["trait1", "trait2"]
        @test G._source_coefficient_names(["a", :b], 2) == ["a", :b]
        @test_throws DimensionMismatch G._source_coefficient_names(["a"], 2)
        @test_throws ArgumentError G._source_coefficient_names([""], 1)
        @test_throws ArgumentError G._source_coefficient_names(["a", "a"], 2)
    end
end

@testset "1124d bridge helpers" begin
    @testset "_bridge_names" begin
        @test G._bridge_names(nothing, 3, "trait") == ["trait1", "trait2", "trait3"]
        @test G._bridge_names(["a", "b", "c"], 3, "trait") == ["a", "b", "c"]
        @test_throws ArgumentError G._bridge_names(["a", "b"], 3, "unit")
    end

    @testset "_bridge_truthy" begin
        @test G._bridge_truthy(true)
        @test G._bridge_truthy(1)
        @test G._bridge_truthy(0) == false
        @test G._bridge_truthy("TRUE")
        @test G._bridge_truthy("no") == false
        @test G._bridge_truthy(nothing) == false
        @test G._bridge_truthy(:symbol) == false
    end

    @testset "_bridge_get" begin
        opts = Dict("foo" => 7, :bar => 8)
        @test G._bridge_get(opts, "foo", 0) == 7
        @test G._bridge_get(opts, "bar", 0) == 8
        @test G._bridge_get(opts, "missing", 99) == 99
        @test G._bridge_get(nothing, "x", 1) == 1
    end

    @testset "_bridge_family_key aliases" begin
        @test G._bridge_family_key("NB2") == "negbinomial"
        @test G._bridge_family_key("beta.binomial") == "betabinomial"
        @test_throws ArgumentError G._bridge_family_key("not_a_family")
    end
end