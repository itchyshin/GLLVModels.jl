using Test, GLLVModels, Random, Distributions

@testset "Grouped structural identification diagnostics" begin
    redundant = [GroupingTerm(:unit; mode=:latent, rank=1, unique=true)]
    notes = GLLVModels._grouped_identification_diagnostics(redundant, 2)
    @test length(notes) == 1
    @test only(notes).source == :unit
    @test only(notes).parameter_count == 4
    @test only(notes).covariance_dimension == 3
    @test isempty(GLLVModels._grouped_identification_diagnostics(redundant, 3))
    @test isempty(GLLVModels._grouped_identification_diagnostics(
        [GroupingTerm(:unit; mode=:dep)], 2))
    @test isempty(GLLVModels._grouped_identification_diagnostics(
        [GroupingTerm(:unit; mode=:latent, rank=1)], 2))
    @test_logs (:warn, r"structurally non-identifiable") GLLVModels._grouped_warn_identification(redundant, 2)
    @test_logs (:warn, r"structurally non-identifiable") GLLVModels._warn_covariance_redundancy(:phylo, 4, 3)
    @test_logs GLLVModels._warn_covariance_redundancy(:phylo, 6, 6)
end


# Binomial with one trial per cell + latent + unique: unique variances are dropped
# automatically with an @info message (as R gllvmTMB does), not merely warned about.
@testset "Binomial one trial per cell: latent + unique drops Psi" begin
    rng = MersenneTwister(1118)
    p, n = 6, 300
    lam = [0.9, 0.7, -0.6, 0.8, 0.5, -0.4]
    z = randn(rng, n)
    eta = 0.2 .+ lam .* z' .+ 0.5 .* randn(rng, p, n)
    Y = Float64.(rand(rng, p, n) .< 1 ./ (1 .+ exp.(-eta)))
    fitu = fitnu = nothing
    @test_logs (:info, r"one trial per cell.*unique") match_mode=:any begin
        fitu = fit_gllvm(Y; family=Binomial(), grouping=[GroupingTerm(:unit; mode=:latent, rank=1, unique=true)],
            unit=collect(1:n), N=ones(p, n))
    end
    fitnu = fit_gllvm(Y; family=Binomial(), grouping=[GroupingTerm(:unit; mode=:latent, rank=1)],
        unit=collect(1:n), N=ones(p, n))
    @test all(!t.unique for t in fitu.terms)
    @test !any(occursin("unique", String(s)) for s in fitu.parameter_labels)
    @test length(fitu.parameters) == length(fitnu.parameters)
    @test fitu.converged
    @test fitu.hessian_positive_definite == fitnu.hessian_positive_definite
    @test isapprox(fitu.loglik, fitnu.loglik; atol=1e-6)
    # helper: only latent+unique, binomial, all-ones trials
    lu = [GroupingTerm(:unit; mode=:latent, rank=1, unique=true)]
    @test all(!t.unique for t in GLLVModels._grouped_drop_binary_unique(lu, ones(4, 10), :binomial))
    @test GLLVModels._grouped_drop_binary_unique(lu, fill(5.0, 4, 10), :binomial)[1].unique
    @test GLLVModels._grouped_drop_binary_unique(lu, ones(4, 10), :poisson)[1].unique
end
