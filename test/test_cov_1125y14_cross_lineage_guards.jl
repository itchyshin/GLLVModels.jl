# Lane cursor-cloud-1125y14 — src/cross_lineage_extract.jl
using GLLVModels, Test

const G = GLLVModels

@testset "cov 1125y14 cross lineage guards" begin
    p = 4

    @testset "_cross_lineage_traits" begin
        idx, lab = G._cross_lineage_traits([1, 2], "row_traits", p, nothing)
        @test idx == [1, 2]
        names = ["a", "b", "c", "d"]
        idx2, lab2 = G._cross_lineage_traits(["b", "d"], "row_traits", p, names)
        @test idx2 == [2, 4]
        @test lab2 == ["b", "d"]
        @test_throws ArgumentError G._cross_lineage_traits(Int[], "row_traits", p, names)
        @test_throws ArgumentError G._cross_lineage_traits([1, 1], "row_traits", p, nothing)
        @test_throws ArgumentError G._cross_lineage_traits([5], "row_traits", p, nothing)
        @test_throws ArgumentError G._cross_lineage_traits(["x"], "row_traits", p, names)
        @test_throws ArgumentError G._cross_lineage_traits([1.0, 2.0], "row_traits", p, names)
    end

    @testset "_cross_lineage_scale" begin
        @test G._cross_lineage_scale(:shape) === nothing
        @test_throws ArgumentError G._cross_lineage_scale(:effect)
        @test_throws ArgumentError G._cross_lineage_scale(:unknown)
    end

    @testset "_matrix_inv_sqrt_rel" begin
        M = [1.0 0.2; 0.2 1.0]
        R = G._matrix_inv_sqrt_rel(M, 1e-12, "test")
        @test R ≈ R' rtol = 1e-12
        @test_throws ArgumentError G._matrix_inv_sqrt_rel([-1.0 0.0; 0.0 -1.0], 1e-12, "bad")
        Z = zeros(2, 2)
        @test_throws ArgumentError G._matrix_inv_sqrt_rel(Z, 1e-12, "zero")
    end

    @testset "extract_latent_scores unsupported" begin
        @test_throws ArgumentError G.extract_latent_scores(1)
        @test_throws ArgumentError G.extract_latent_scores("fit", zeros(2, 2))
        fit = fit_gaussian_gllvm(randn(3, 5); K = 1, iterations = 10)
        y = randn(3, 5)
        @test extract_latent_scores(fit, y; level = :unit_obs) === nothing
        @test_throws ArgumentError extract_latent_scores(fit, y; level = :foo)
    end
end