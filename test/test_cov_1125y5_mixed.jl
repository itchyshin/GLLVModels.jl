# Lane 1125y5: families/mixed.jl layout, marginal, accessors (not 1124d offset helpers).
using Test, GLLVModels, Random, Distributions, LinearAlgebra
const G = GLLVModels

@testset "1125y5 mixed family layout and loglik" begin
    fams = [Poisson(), Normal(0.0, 1.0)]
    di, nd = G._mixed_family_layout(fams)
    @test nd == 1
    @test di[2] == 1 && di[1] == 0

    p, n, K = 2, 8, 1
    Y = zeros(Float64, p, n)
    Y[1, :] .= rand(Poisson(2), n)
    Y[2, :] .= randn(n)
    N = ones(p, n)
    Λ = reshape([0.5, 0.4], p, K)
    β = [0.1, 0.0]
    links = [LogLink(), IdentityLink()]
    L = G.mixed_marginal_loglik_laplace(fams, links, Y, N, Λ, β; maxiter = 30)
    @test isfinite(L)

    fit = fit_mixed_gllvm(Y; families = fams, K = K, iterations = 12)
    @test fit isa G.MixedFamilyFit
    @test occursin("MixedFamilyFit", sprint(show, "text/plain", fit))

    lv = getLV(fit, Y)
    @test size(lv, 1) == n
    pred = predict(fit, Y; type = :response)
    @test size(pred) == size(Y)
    lr = link_residual(fit, Y)
    @test length(lr) == p
    com = communality(fit, Y; N = N)
    @test length(com) == p
    @test_throws ArgumentError predict(fit, Y; type = :linear)

    θ = vcat(β, G.pack_lambda(Λ), log(0.8))
    θp = G._mixed_marginal_loglik_packed(θ, Y, N, p, K, fams, links, di; maxiter = 20)
    @test isfinite(θp) || θp == -Inf
end