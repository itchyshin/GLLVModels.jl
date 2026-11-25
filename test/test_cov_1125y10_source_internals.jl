# Lane cursor-cloud-1125y10: source_fit.jl _source_nparams, trait design, trait covariances.
using Test, GLLVModels, LinearAlgebra

const G = GLLVModels

@testset "1125y10 source_fit internals" begin
    C = [1.0 0.1; 0.1 1.0]
    P = [1.0 0.0; 1.0 0.0; 0.0 1.0]
    p = 3
    sindep = SourceCovariance(C, P; mode = :indep)
    slab = SourceCovariance(C, P; mode = :latent, rank = 2)

    @testset "_source_nparams by mode" begin
        @test G._source_nparams(sindep, p) == p
        sindep_c = SourceCovariance(C, P; mode = :indep, common = true)
        @test G._source_nparams(sindep_c, p) == 1
        @test G._source_nparams(slab, p) == G.rr_theta_len(p, 2)
        slab_u = SourceCovariance(C, P; mode = :latent, rank = 1, unique = true)
        @test G._source_nparams(slab_u, p) == G.rr_theta_len(p, 1) + p
        slab_uc = SourceCovariance(C, P; mode = :latent, rank = 1, unique = true, common = true)
        @test G._source_nparams(slab_uc, p) == G.rr_theta_len(p, 1) + 1
        sdep = SourceCovariance(C, P; mode = :dep)
        @test G._source_nparams(sdep, p) == G.rr_theta_len(p, p)
        @test_throws ArgumentError G._source_nparams(
            SourceCovariance(C, P; mode = :latent, rank = 4), p)
    end

    @testset "_trait_mean_design" begin
        D = G._trait_mean_design(3, 4)
        @test size(D) == (12, 3)
        @test D[1, 1] == 1.0 && D[1, 2] == 0.0
        @test D[4, 1] == 1.0 && D[4, 2] == 0.0
        @test D[2, 2] == 1.0 && D[5, 2] == 1.0
        @test sum(D) == 12.0
        @test count(!iszero, D) == 12
    end

    @testset "_source_trait_covariances indep" begin
        sindep_c = SourceCovariance(C, P; mode = :indep, common = true)
        θ1 = [log(0.5)]
        θ2 = [log(0.4), log(0.6), log(0.8)]
        B1 = G._source_trait_covariances([sindep_c], p, θ1)
        @test B1[1] ≈ Diagonal(fill(0.25, p))
        B2 = G._source_trait_covariances([sindep], p, θ2)
        @test diag(B2[1]) ≈ exp.(2 .* θ2)
        @test_throws DimensionMismatch G._source_trait_covariances([sindep], p, θ2[1:2])
    end

    @testset "_source_trait_covariances latent" begin
        rank = 2
        nθ = G.rr_theta_len(p, rank)
        θ = fill(0.1, nθ)
        Bl = G._source_trait_covariances([slab], p, θ)
        Λ = G.unpack_lambda(θ, p, rank)
        @test Bl[1] ≈ Λ * Λ' rtol = 1e-12
    end

    @testset "_legacy_source_mean_metadata" begin
        D, shape, names = G._legacy_source_mean_metadata([1.0, 2.0, 3.0], 12)
        @test size(D) == (12, 3)
        @test shape == (3, 4)
        @test names == ["trait1", "trait2", "trait3"]
        @test_throws ArgumentError G._legacy_source_mean_metadata(Float64[], 0)
        @test_throws ArgumentError G._legacy_source_mean_metadata([1.0, 2.0], 7)
    end
end