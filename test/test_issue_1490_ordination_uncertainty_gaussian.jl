using Test, Random, GLLVModels

@testset "ordination_uncertainty on GllvmFit (refs #1490)" begin
    Y = Float64.([25 0 15 2 1 0 2 0 1 3 15 16
                  10 2 20 6 20 6 7 11 1 0 1 13])
    fit = fit_gaussian_gllvm(Y; K = 1, iterations = 80)
    u = ordination_uncertainty(fit, Y; n_boot = 30, rng = MersenneTwister(1))
    @test size(u.scores) == (12, 1)
    @test all(isfinite, u.se)
end