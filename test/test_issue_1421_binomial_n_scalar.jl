# Issue #1421: scalar N must not TypeError at the fit_gllvm keyword layer.
using Test, GLLVModels, Random, Distributions

@testset "fit_gllvm Binomial scalar N raises ArgumentError (#1421)" begin
    rng = MersenneTwister(3)
    p, n = 10, 60
    Y = Int.(rand(rng, Bernoulli(0.3), p, n))
    err = @test_throws ArgumentError fit_gllvm(Y; family = Binomial(), K = 1, N = 1, iterations = 50)
    @test occursin("$(p)×$(n)", sprint(showerror, err.value))
    @test occursin("matrix", sprint(showerror, err.value))
    @test fit_gllvm(Y; family = Binomial(), K = 1, iterations = 50) isa BinomialFit
end