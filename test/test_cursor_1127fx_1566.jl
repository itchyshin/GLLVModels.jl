# refs #1566 — loglikelihood(fit, Y) must not resolve to Distributions.jl
using GLLVModels, Test, Random, Distributions

@testset "loglikelihood(fit, Y) on PoissonFit (#1566)" begin
    Random.seed!(1127)
    Y = [rand(Poisson(2)) for i in 1:4, j in 1:12]
    fit = fit_poisson_gllvm(Y; K = 1)

    @test loglikelihood(fit) ≈ fit.loglik

    err = @test_throws ArgumentError loglikelihood(fit, Y)
    @test occursin("loglikelihood(fit)", string(err.value))
    @test !occursin("ReshapedDistribution", sprint(showerror, err.value))
end