# Lane cursor-cloud-1125y10: grouped_fit.jl identification helpers and GroupingTerm guards.
using Test, GLLVModels

const G = GLLVModels

@testset "1125y10 grouped misc" begin
    @testset "GroupingTerm construction" begin
        t = G.GroupingTerm(:unit; mode = :latent, rank = 2)
        @test t.name === :unit
        @test t.mode === :latent
        @test t.rank == 2
        @test_throws ArgumentError G.GroupingTerm(:badname; mode = :latent, rank = 1)
        @test_throws ArgumentError G.GroupingTerm(:unit; mode = :nope, rank = 1)
        @test_throws ArgumentError G.GroupingTerm(:cluster2; mode = :latent, rank = 1)
        dep = G.GroupingTerm(:cluster; mode = :dep)
        @test dep.mode === :dep
        @test dep.rank == 0
    end

    @testset "_grouping_term_nparams" begin
        p = 5
        ti = G.GroupingTerm(:unit; mode = :indep)
        @test G._grouping_term_nparams(ti, p) == p
        tic = G.GroupingTerm(:unit_obs; mode = :indep, common = true)
        @test G._grouping_term_nparams(tic, p) == 1
        tl = G.GroupingTerm(:cluster; mode = :latent, rank = 2)
        @test G._grouping_term_nparams(tl, p) == G.rr_theta_len(p, 2)
        tlu = G.GroupingTerm(:cluster; mode = :latent, rank = 2, unique = true)
        @test G._grouping_term_nparams(tlu, p) == G.rr_theta_len(p, 2) + p
        td = G.GroupingTerm(:cluster; mode = :dep)
        @test G._grouping_term_nparams(td, p) == G.rr_theta_len(p, p)
    end

    @testset "_grouped_identification_diagnostics" begin
        p = 3
        red = G.GroupingTerm(:unit; mode = :latent, rank = 3, unique = true)
        notes = G._grouped_identification_diagnostics([red], p)
        @test length(notes) == 1
        @test notes[1].parameter_count > notes[1].covariance_dimension
        small = G.GroupingTerm(:cluster2; mode = :indep)
        @test isempty(G._grouped_identification_diagnostics([small], p))
    end

    @testset "_GROUPING_TERM_NAMES membership" begin
        @test :unit in G._GROUPING_TERM_NAMES
        @test :unit_obs in G._GROUPING_TERM_NAMES
        @test :cluster in G._GROUPING_TERM_NAMES
        @test :cluster2 in G._GROUPING_TERM_NAMES
        @test length(G._GROUPING_TERM_NAMES) == 4
    end

    @testset "_grouped_stopping_reason" begin
        @test G._grouped_stopping_reason(true, true, 10, 100, 1e-8, 1e-5) === :converged
        @test G._grouped_stopping_reason(false, false, 5, 100, 1.0, 1e-5) === :invalid_final
        @test G._grouped_stopping_reason(false, true, 100, 100, 1.0, 1e-5) === :iteration_limit
        @test G._grouped_stopping_reason(false, true, 50, 100, 1.0, 1e-3) === :gradient_not_converged
        @test G._grouped_stopping_reason(false, true, 50, 100, 1e-5, 1e-3) === :optimizer_not_converged
    end
end