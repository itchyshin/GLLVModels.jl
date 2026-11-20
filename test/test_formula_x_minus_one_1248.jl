using Test
using GLLVModels, StatsModels, Random, Distributions

@testset "gllvm refuses x - 1 as shifted covariate (refs #1248)" begin
    rng = MersenneTwister(7)
    p, n = 5, 60
    x = randn(rng, n)
    Y = [0.2j + 0.5 * x[i] + 0.3 * randn(rng) for j in 1:p, i in 1:n]
    f_with = @formula(y ~ 1 + x)
    fit = gllvm(f_with, Y, (x = x,); K = 1, family = Normal())
    @test dof(fit) == 12
    f_shift = @formula(y ~ x - 1)
    @test_throws ArgumentError gllvm(f_shift, Y, (x = x,); K = 1, family = Normal())
end