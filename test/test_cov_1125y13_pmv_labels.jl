# Lane cursor-cloud-1125y13 — src/phylo_latent_precision.jl
using GLLVModels, Test, LinearAlgebra, SparseArrays

const G = GLLVModels

@testset "cov 1125y13 pmv labels" begin
    q = sparse([4.0 -1.0 -1.0 0.0;
                -1.0 3.0 0.0 -1.0;
                -1.0 0.0 3.0 0.0;
                0.0 -1.0 0.0 3.0])
    i, j, x = findnz(q)
    labels = ["a1", "a2", "s1", "s2"]
    phy = PrecisionPhy(i, j, x, 4, 2, labels, logdet(cholesky(Symmetric(Matrix(q)))), 1.0, [3, 4])
    species_id = [1, 2, 1, 2]

    @testset "_pmv_default_labels" begin
        obs, tips = G._pmv_default_labels(phy, species_id)
        @test obs == ["s1", "s2", "s1", "s2"]
        @test tips == ["s1", "s2"]
    end

    @testset "PrecisionMultivariateFit property accessors" begin
        Y = randn(3, 4)
        fit = GLLVModels.fit_precision_multivariate(Y, phy; species_id = species_id, rank = 1,
            iterations = 30, g_tol = 1e-3)
        @test fit.species_labels == ["s1", "s2", "s1", "s2"]
        @test fit.tip_labels == ["s1", "s2"]
        @test :species_labels in propertynames(fit)
        @test_throws DimensionMismatch G._pmv_with_labels(fit, ["only"], labels)
        @test_throws DimensionMismatch G._pmv_with_labels(fit, fit.species_labels, ["one"])
    end
end