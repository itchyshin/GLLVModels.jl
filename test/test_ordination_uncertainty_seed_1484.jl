using GLLVModels, Test, Random, Distributions

@testset "ordination_uncertainty: refuse seed= keyword (refs #1484)" begin
    rng = MersenneTwister(1)
    Y = [rand(rng, Poisson(3)) for j in 1:5, i in 1:20]
    fit = fit_gllvm(Y; family = Poisson(), K = 1, iterations = 80)

    err = nothing
    try
        ordination_uncertainty(fit, Y; n_boot = 10, seed = 1)
    catch e
        err = e
    end
    @test err isa ArgumentError
    msg = err.msg
    @test occursin("ordination_uncertainty", msg)
    @test occursin("seed", msg)
    @test occursin("rng", msg)
    @test occursin("Random.MersenneTwister", msg)

    u = ordination_uncertainty(fit, Y; n_boot = 10, rng = MersenneTwister(1))
    @test u.n_boot ≥ 2
    @test all(isfinite, u.scores)
end