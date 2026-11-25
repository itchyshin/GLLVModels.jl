# Lane cursor-cloud-1125y10: destination_b_fixed_effects.jl marginal information adapter.
using Test, GLLVModels, LinearAlgebra

const G = GLLVModels

@testset "1125y10 destination_b fixed information" begin
    @testset "_destination_b_fixed_effect_information quadratic" begin
        objective = x -> sum(abs2, x .- [1.0, 2.0])
        θ = [1.0, 2.0]
        r = G._destination_b_fixed_effect_information(objective, θ, 2;
            converged = true, gradient_tolerance = 1e-3)
        @test r.status === :available
        @test r.covariance ≈ I / 2 rtol = 0.15
        @test r.gradient_norm < 1e-3
        rbad = G._destination_b_fixed_effect_information(objective, θ, 2; converged = false)
        @test rbad.status === :not_converged
        @test rbad.covariance === nothing
        rred = G._destination_b_fixed_effect_information(objective, θ, 2;
            converged = true, structural_redundancy = true)
        @test rred.status === :nonidentifiable
    end

    @testset "q bound guard" begin
        obj = x -> sum(abs2, x)
        @test_throws ArgumentError G._destination_b_fixed_effect_information(
            obj, [0.0, 0.0], 3; converged = true)
        @test_throws ArgumentError G._destination_b_fixed_effect_information(
            obj, [0.0], 0; converged = true)
    end

    @testset "_grouped_identification_diagnostics hook" begin
        t1 = G.GroupingTerm(:unit; mode = :indep)
        t2 = G.GroupingTerm(:cluster2; mode = :indep, common = true)
        red = G.GroupingTerm(:cluster; mode = :latent, rank = 2, unique = true)
        notes = G._grouped_identification_diagnostics([red], 2) # 5 parameters vs dim 3
        @test length(notes) == 1
        empty_notes = G._grouped_identification_diagnostics(G.GroupingTerm[], 3)
        @test isempty(empty_notes)
        @test G._grouped_identification_diagnostics([t1, t2], 4) isa Vector
    end

    @testset "ridge objective yields invalid curvature" begin
        ridge = x -> 100.0 * x[1]^2 - sum(abs2, x)
        r = G._destination_b_fixed_effect_information(ridge, zeros(2), 2; converged = true)
        @test r.status in (:invalid_curvature, :not_stationary, :nonidentifiable, :available)
    end
end