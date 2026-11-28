using Test
using GLLVModels, Distributions, Random

@testset "issue 1494: bootstrap_ci refuses PoissonFit (refs #1494)" begin
    Random.seed!(602)
    Y = rand(Poisson(2), 4, 50)
    fit = fit_poisson_gllvm(Y; K = 1, iterations = 80)
    @test fit.converged
    confint(fit, Y; method = :bootstrap, n_boot = 10, seed = 1)
    err = try
        bootstrap_ci(fit; y = Y, n_boot = 10, seed = 1)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("confint(fit, Y; method=:bootstrap", sprint(showerror, err))
end