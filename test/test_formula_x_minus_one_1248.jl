using Test, GLLVModels, StatsModels, Distributions, Random

@testset "gllvm refuses x - 1 as shifted covariate (refs 1248)" begin
    rng = MersenneTwister(7)
    p, n = 5, 60
    x = randn(rng, n)
    Y = [0.2j + 0.5x[i] + 0.3randn(rng) for j in 1:p, i in 1:n]
    D = (x = x,)
    f = eval(Meta.parse("@formula(y ~ x - 1)"))
    err = try; gllvm(f, Y, D; family = Normal(), K = 1); nothing; catch e; e; end
    @test err isa ArgumentError
    @test occursin("0 + x", err.msg)
    g0 = gllvm(@formula(y ~ 0 + x), Y, D; family = Normal(), K = 1)
    @test dof(g0) == 7
end