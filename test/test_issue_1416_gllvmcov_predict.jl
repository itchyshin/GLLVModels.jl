using Test, Random, Distributions, StatsModels, GLLVModels

@testset "GllvmCovFit predict(fit, Y) refuses without X (refs #1416)" begin
    Random.seed!(2001)
    p, n, K = 4, 60, 1
    x = randn(n)
    Y = reshape([rand(Poisson(2)) for _ in 1:(p * n)], p, n)
    fit = gllvm(@formula(y ~ 1 + x), Y, (x = x,); family = Poisson(), K = K, iterations = 80)
    err = try
        predict(fit, Y)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("design `X`", sprint(showerror, err))
end