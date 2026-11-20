# #1284: fit_gllvm(Normal()) must not report converged=true at a Heywood boundary.
# Mixed-scale Gaussian traits with a SHARED σ_eps force the small-variance traits' noise onto
# the shared uniqueness and absorb the large-scale trait into the latent variable (communality
# -> 1). Rule (BRIAN-1430-1117 / SHINICHI-0545): converged=true only at an interior optimum;
# a Heywood gets the :boundary status plus a warning naming the parameter.
# Names per SHINICHI-1753-1118: :converged, :boundary (Heywood is a boundary kind), :not_converged.

using GLLVModels, Test, Random, StableRNGs, Distributions

@testset "Normal Heywood status (#1284)" begin
    rng = StableRNG(1284)
    p, n = 9, 276
    # Eight same-scale traits with a weak factor, one trait on a ~100x larger scale.
    λ = [0.15, -0.1, -0.08, 0.1, -0.14, 0.02, -0.04, 0.0, 0.19]
    η = randn(rng, 1, n)
    Y = λ * η .+ randn(rng, p, n)
    Y[8, :] .= 90 .* Y[8, :] .+ 250

    fit = @test_logs (:warn, r"Heywood.*trait 8") match_mode = :any fit_gllvm(Y; family = Normal(), K = 1)
    @test fit.converged == false
    @test convergence_status(fit) === :boundary
    @test boundary_params(fit) == [(name = "trait 8", kind = :heywood)]
    @test heywood_traits(fit) == [8]

    # Control: same-scale data (no Heywood) stays converged, no warning.
    Yc = λ * η .+ randn(rng, p, n)
    fitc = @test_logs min_level = Base.CoreLogging.Warn fit_gllvm(Yc; family = Normal(), K = 1)
    @test convergence_status(fitc) === :converged
    @test fitc.converged
    @test isempty(heywood_traits(fitc))
    @test isempty(boundary_params(fitc))
end
