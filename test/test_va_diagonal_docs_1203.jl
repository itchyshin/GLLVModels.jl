# VA fits use a diagonal site posterior; K=2 Gram differs from Laplace (#1203).

using GLLVModels, Random, LinearAlgebra, Test
using Distributions: Poisson as DPoisson

@testset "VA diagonal posterior vs Laplace (#1203)" begin
    rng = MersenneTwister(2)
    p, n, K = 8, 150, 2
    L = 0.7 .* randn(rng, p, K)
    b = 0.8 .+ 0.3 .* randn(rng, p)
    Y = [rand(rng, DPoisson(exp(b[t] + dot(L[t, :], z)))) for t in 1:p, z in eachcol(randn(rng, K, n))]
    la = fit_poisson_gllvm(Y; K = K, iterations = 60)
    va = fit_poisson_gllvm_va(Y; K = K, iterations = 60)
    @test la.converged
    @test va.converged
    G_la = la.Λ * la.Λ'
    G_va = va.Λ * va.Λ'
    @test norm(G_va - G_la) / norm(G_la) > 0.01
    tut = read(joinpath(@__DIR__, "..", "docs", "src", "tutorial.md"), String)
    @test occursin("Lambda.struc = \"diagonal\"", tut)
end