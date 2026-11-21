# refs #1247: a complete numeric covariate stored as Union{Missing,T} must match Vector{T} in dof and logLik.
using Test, GLLVModels, Random, Distributions, StatsModels

@testset "Union Missing numeric covariate reads as continuous (refs #1247)" begin
    rng = MersenneTwister(7)
    p, n = 5, 60
    x = randn(rng, n)
    xu = Vector{Union{Missing, Float64}}(x)
    eta = [0.2j + 0.5 * x[i] for j in 1:p, i in 1:n]
    Yp = [Float64(rand(rng, Poisson(exp(0.3e)))) for e in eta]
    Yg = eta .+ 0.5 .* randn(rng, p, n)
    xi = rand(rng, -2:2, n)
    xiu = Vector{Union{Missing, Int}}(xi)
    eta_i = [0.2j + 0.5 * xi[i] for j in 1:p, i in 1:n]
    Ypi = [Float64(rand(rng, Poisson(exp(0.3e)))) for e in eta_i]

    @testset "Poisson seed 7" begin
        a = gllvm(@formula(y ~ 1 + x), Yp, (x = x,); family = Poisson(), K = 1)
        b = gllvm(@formula(y ~ 1 + x), Yp, (x = xu,); family = Poisson(), K = 1)
        @test dof(a) == dof(b) == 11
        @test loglikelihood(a) == loglikelihood(b)
    end

    @testset "Normal seed 7" begin
        a = gllvm(@formula(y ~ 1 + x), Yg, (x = x,); K = 1)
        b = gllvm(@formula(y ~ 1 + x), Yg, (x = xu,); K = 1)
        @test dof(a) == dof(b) == 12
        @test loglikelihood(a) == loglikelihood(b)
    end

    @testset "Poisson integer Union{Missing,Int}" begin
        a = gllvm(@formula(y ~ 1 + x), Ypi, (x = xi,); family = Poisson(), K = 1)
        b = gllvm(@formula(y ~ 1 + x), Ypi, (x = xiu,); family = Poisson(), K = 1)
        @test dof(a) == dof(b)
        @test loglikelihood(a) == loglikelihood(b)
    end
end