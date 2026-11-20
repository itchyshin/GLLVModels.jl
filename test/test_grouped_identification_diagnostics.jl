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

# twin-gllvm-a-1118 (R gllvmTMB vs Julia): binomial with one trial per cell and
# latent + unique used to fit the unique variances silently (R leaves them out).
@testset "Binomial one trial per cell: latent + unique warns" begin
    latent_unique = [GroupingTerm(:unit; mode=:latent, rank=1, unique=true)]
    latent_only = [GroupingTerm(:unit; mode=:latent, rank=1)]
    @test_logs (:warn, r"one trial per cell") GLLVModels._grouped_warn_binary_unique(
        latent_unique, ones(4, 10), :binomial)
    @test_logs GLLVModels._grouped_warn_binary_unique(latent_unique, fill(5.0, 4, 10), :binomial)
    @test_logs GLLVModels._grouped_warn_binary_unique(latent_only, ones(4, 10), :binomial)
    @test_logs GLLVModels._grouped_warn_binary_unique(latent_unique, ones(4, 10), :poisson)
end

# Poisson unique-only (mode=:indep) used to stop at BFGS iteration 2: the first,
# unscaled BFGS step overflows exp(2 * log-sd), and the search never recovered
# (loglik 8 below the converged value here, 117 below R gllvmTMB on the lane's
# seed-1 data, converged=false).
@testset "Poisson unique-only grouped fit converges at default iterations" begin
    rng = MersenneTwister(1118)
    p, n = 8, 300
    b = 0.4 .* randn(rng, p)
    psi = 0.3 .+ 0.4 .* rand(rng, p)
    x = randn(rng, n)
    eta = b .+ 0.4 .* x' .+ psi .* randn(rng, p, n)
    Y = Float64.(rand.(rng, Poisson.(exp.(0.7 .* eta))))
    X = zeros(p, n, p + 1)
    for t in 1:p
        X[t, :, t] .= 1.0
        X[t, :, p + 1] = x
    end
    fit = fit_gllvm(Y; family=Poisson(), grouping=[GroupingTerm(:unit; mode=:indep)],
        unit=collect(1:n), X=X)
    @test fit.converged
    @test fit.stopping_reason === :converged
end
