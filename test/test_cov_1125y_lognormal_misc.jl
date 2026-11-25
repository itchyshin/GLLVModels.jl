# Coverage round 4: lognormal.jl guards and show not exercised by test_lognormal.jl alone.
using Test
using GLLVModels
using Distributions

if !isdefined(GLLVModels, :Lognormal)
    Base.include(GLLVModels, joinpath(@__DIR__, "..", "src", "families", "lognormal.jl"))
end

const G = GLLVModels

@testset "1125y lognormal misc" begin
  Y = exp.(randn(3, 10))
  Λ = 0.1 .* randn(3, 1)
  β = [0.1, 0.2, 0.3]

  @testset "marginal validation" begin
    @test_throws DimensionMismatch G.lognormal_marginal_loglik(Y, Λ, [0.1, 0.2], 0.5)
    Ybad = copy(Y)
    Ybad[2, 3] = 0.0
    @test_throws ArgumentError G.lognormal_marginal_loglik(Ybad, Λ, β, 0.5)
    @test G.default_link(G.Lognormal()) isa G.LogLink
  end

  @testset "fit guards" begin
    @test_throws ArgumentError G.fit_lognormal_gllvm(Y; K = 1, mask = trues(3, 10))
    @test_throws ArgumentError G.fit_lognormal_gllvm(Y; K = 1,
        lambda_constraint = fill(NaN, 3, 1))
    off = fill(1.0, 3, 10)
    off[1, 1] = NaN
    @test_throws ArgumentError G.fit_lognormal_gllvm(Y; K = 1, offset = off, iterations = 3)
  end

  @testset "LognormalFit show" begin
    fit = G.fit_lognormal_gllvm(Y; K = 1, iterations = 12)
    buf = sprint(show, fit)
    @test occursin("LognormalFit", buf)
    @test occursin("LogLink", buf)
    @test occursin("loglik", buf)
    fitnc = G.LognormalFit(β, Λ, 0.4, G.LogLink(), -10.0, false, 0, Float64[])
    @test occursin("NOT CONVERGED", sprint(show, fitnc))
  end

  @testset "laplace alias" begin
    ll = G.lognormal_marginal_loglik(Y, Λ, β, 0.5)
    @test G.lognormal_marginal_loglik_laplace(Y, Λ, β, 0.5) == ll
  end
end