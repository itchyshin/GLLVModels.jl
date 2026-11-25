# Lane cursor-cloud-1125y15 — joint_phylo_grouped_fit.jl, precision_multivariate_fit.jl, phylo_glm.jl
using GLLVModels, Test, LinearAlgebra, Random, Distributions, SparseArrays

const G = GLLVModels

@testset "cov 1125y15 joint phylo layout and PMV unpack" begin
    terms = [G.GroupingTerm(:unit; mode = :indep, common = false)]
    @testset "_joint_phylo_grouped_fit_layout" begin
        lay = G._joint_phylo_grouped_fit_layout(3, 1, :barelowrank, 2, terms)
        @test length(lay.loading) == 3
        @test length(lay.residual) == 3
        @test lay.total > lay.residual[end]
        @test_throws ArgumentError G._joint_phylo_grouped_fit_layout(3, 0, :barelowrank, 2, terms)
        @test_throws ArgumentError G._joint_phylo_grouped_fit_layout(3, 1, :barelowrank, -1, terms)
        bad_terms = [G.GroupingTerm(:unit; mode = :indep, common = true)]
        @test_throws ArgumentError G._joint_phylo_grouped_fit_layout(3, 1, :barelowrank, 2, bad_terms)
    end

    @testset "_joint_phylo_grouped_fit_unpack" begin
        p, q, rank = 2, 2, 1
        lay = G._joint_phylo_grouped_fit_layout(p, rank, :barelowrank, q, terms)
        θ = zeros(lay.total)
        θ[lay.loading] .= 0.1
        θ[lay.residual] .= log(0.5)
        θ[lay.ordinary[1]] .= log(0.3)
        up = G._joint_phylo_grouped_fit_unpack(θ, p, rank, :barelowrank, q, terms)
        @test length(up.beta) == q
        @test all(up.residual_variance .> 0)
        @test_throws DimensionMismatch G._joint_phylo_grouped_fit_unpack(θ[1:end-1],
            p, rank, :barelowrank, q, terms)
        θ_bad = copy(θ); θ_bad[lay.residual[1]] = NaN
        @test_throws ArgumentError G._joint_phylo_grouped_fit_unpack(θ_bad, p, rank, :barelowrank, q, terms)
    end

    @testset "_pmv_mode and _pmv_residual_mode" begin
        @test G._pmv_mode(:barelowrank) === :barelowrank
        @test G._pmv_mode(:explicitunique) === :explicitunique
        @test_throws ArgumentError G._pmv_mode(:invalid)
        @test G._pmv_residual_mode(:trait) === :trait
        @test G._pmv_residual_mode(:shared) === :shared
        @test_throws ArgumentError G._pmv_residual_mode(:unit)
    end

    @testset "_precision_multivariate_unpack" begin
        d, q, rank = 3, 1, 1
        layout = G._pmv_layout(d, rank, :barelowrank, q; residual_mode = :trait)
        θ = zeros(layout.total)
        θ[layout.loading] .= 0.05
        θ[layout.residual] .= log(0.2) # exp(2·log(0.2)) = 0.04 variance (log-σ parameterisation)
        parts = G._precision_multivariate_unpack(θ, d, rank, :barelowrank, q)
        @test length(parts.beta) == q
        @test parts.residual_variance ≈ fill(0.04, d) rtol = 1e-12
        layout_s = G._pmv_layout(d, rank, :barelowrank, q; residual_mode = :shared)
        θs = zeros(layout_s.total)
        θs[layout_s.loading] .= 0.1
        θs[layout_s.residual] .= log(0.3)
        ps = G._precision_multivariate_unpack(θs, d, rank, :barelowrank, q;
            residual_mode = :shared)
        @test length(ps.residual_variance) == d
        @test ps.residual_variance[1] ≈ exp(2 * log(0.3)) rtol = 1e-12
        @test all(ps.residual_variance .≈ ps.residual_variance[1])
        @test_throws ArgumentError G._precision_multivariate_unpack(θs, d, d + 1,
            :barelowrank, q)
    end

    @testset "fit_phylo_glm dimension guards" begin
        phy = G.augmented_phy("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")
        Y = rand(Binomial(1, 0.4), 2, 5) # wrong p
        @test_throws ArgumentError G.fit_phylo_glm(Y, phy; family = Binomial(), iterations = 0)
        Y3 = rand(Binomial(1, 0.3), 3, 3)
        @test_throws ArgumentError G.fit_phylo_glm(Y3, phy; family = Poisson(), iterations = 0)
    end
end