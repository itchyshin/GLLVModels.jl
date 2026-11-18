# Verification contract for per-response-column family dispatch (issue #98).
# `fit_gllvm(Y; family = [one family per row], K)` routes to `fit_mixed_gllvm`, the
# per-column entry point; the three checks below are the shared contract from the issue:
#   (a) a homogeneous families vector recovers the single-family fit (special case);
#   (b) a two-family problem with block-diagonal loadings factorises into its two
#       single-family blocks (marginal exactly, and each block's fit recovered);
#   (c) the gradient of the mixed objective matches finite differences to 1e-6,
#       per parameter block.
using Test
using GLLVModels
using Distributions
using ForwardDiff
using LinearAlgebra
using Random

const _MC = GLLVModels

function _mc_poisson_binomial(rng, p1, p2, n; K = 1)
    Λ = zeros(p1 + p2, K)
    Λ[1:p1, 1] .= 0.4 .+ 0.5 .* rand(rng, p1)
    Λ[(p1 + 1):end, K] .= 0.4 .+ 0.5 .* rand(rng, p2)
    β = vcat(fill(0.8, p1), fill(-0.2, p2))
    Z = randn(rng, K, n)
    η = β .+ Λ * Z
    Y = Matrix{Float64}(undef, p1 + p2, n)
    for j in 1:n, t in 1:p1
        Y[t, j] = rand(rng, Poisson(exp(η[t, j])))
    end
    for j in 1:n, t in (p1 + 1):(p1 + p2)
        Y[t, j] = rand(rng) < 1 / (1 + exp(-η[t, j])) ? 1.0 : 0.0
    end
    return Y, Λ, β
end

@testset "per-column family dispatch contract (#98)" begin
    @testset "(a) homogeneous families vector = single-family fit" begin
        rng = MersenneTwister(98)
        p, n = 6, 120
        Λ = 0.4 .+ 0.5 .* rand(rng, p, 1)
        β = fill(0.7, p)
        η = β .+ Λ * randn(rng, 1, n)
        Yp = Float64[rand(rng, Poisson(exp(η[t, j]))) for t in 1:p, j in 1:n]
        ref = fit_gllvm(Yp; family = Poisson(), K = 1)
        mix = fit_gllvm(Yp; family = fill(Poisson(), p), K = 1)
        @test mix.converged
        @test isapprox(mix.loglik, ref.loglik; atol = 1e-3)
        @test isapprox(mix.β, ref.β; atol = 5e-3)
        @test isapprox(abs.(mix.Λ), abs.(ref.Λ); atol = 5e-3)

        Yb = Float64[rand(rng) < 1 / (1 + exp(-η[t, j])) ? 1 : 0 for t in 1:p, j in 1:n]
        refb = fit_gllvm(Yb; family = Binomial(), K = 1)
        mixb = fit_gllvm(Yb; family = fill(Binomial(), p), K = 1)
        @test isapprox(mixb.loglik, refb.loglik; atol = 1e-3)
    end

    @testset "(b) two-family block-diagonal problem factorises" begin
        rng = MersenneTwister(981)
        p1, p2, n = 4, 4, 150
        Y, Λ, β = _mc_poisson_binomial(rng, p1, p2, n; K = 2)
        fams = vcat(fill(Poisson(), p1), fill(Binomial(), p2))
        links = [GLLVModels.default_link(f) for f in fams]
        N = ones(Int, p1 + p2, n)
        joint = mixed_marginal_loglik_laplace(fams, links, Y, N, Λ, β)
        r1 = 1:p1
        r2 = (p1 + 1):(p1 + p2)
        l1 = _MC.marginal_loglik_laplace(Poisson(), Y[r1, :], N[r1, :], Λ[r1, 1:1], β[r1],
                                         links[1])
        l2 = _MC.marginal_loglik_laplace(Binomial(), Y[r2, :], N[r2, :], Λ[r2, 2:2], β[r2],
                                         links[end])
        @test isapprox(joint, l1 + l2; rtol = 1e-8)

        # each block of the mixed fit recovers its own single-family fit
        mix = fit_gllvm(Y; family = fams, K = 2)
        ref1 = fit_gllvm(Y[r1, :]; family = Poisson(), K = 1)
        ref2 = fit_gllvm(Y[r2, :]; family = Binomial(), K = 1)
        @test mix.converged
        # the joint optimum cannot be below the sum of the block optima at the
        # same (block-diagonal) point; allow optimiser slack
        @test mix.loglik >= ref1.loglik + ref2.loglik - 0.05
        @test isapprox(mix.β[r1], ref1.β; atol = 0.15)
        @test isapprox(mix.β[r2], ref2.β; atol = 0.25)
    end

    @testset "(c) gradient vs finite differences, per block" begin
        rng = MersenneTwister(982)
        p1, p2, n = 3, 3, 60
        Y, _, _ = _mc_poisson_binomial(rng, p1, p2, n; K = 1)
        p = p1 + p2
        K = 1
        fams = Any[fill(Poisson(), p1); fill(Binomial(), p2)]
        links = [GLLVModels.default_link(f) for f in fams]
        N = ones(Int, p, n)
        disp_index, _ = _MC._mixed_family_layout(fams)
        θ = vcat(0.1 .* randn(rng, p), _MC.pack_lambda(0.3 .* randn(rng, p, K)))
        f(θ) = _MC._mixed_marginal_loglik_packed(θ, Y, N, p, K, fams, links, disp_index;
                                                 maxiter = 100, tol = 1e-12)
        g = ForwardDiff.gradient(f, θ)
        h = 1e-5
        gfd = map(eachindex(θ)) do i
            e = zeros(length(θ)); e[i] = h
            (f(θ .+ e) - f(θ .- e)) / (2h)
        end
        β_block1 = 1:p1
        β_block2 = (p1 + 1):p
        for blk in (β_block1, β_block2)
            @test maximum(abs, g[blk] .- gfd[blk]) <= 1e-6 * max(1, maximum(abs, gfd[blk]))
        end
        lam = (p + 1):length(θ)
        @test maximum(abs, g[lam] .- gfd[lam]) <= 1e-6 * max(1, maximum(abs, gfd[lam]))
    end

    @testset "(d) fit_gllvm vector routing: same fit, loud refusals" begin
        rng = MersenneTwister(983)
        Y, _, _ = _mc_poisson_binomial(rng, 3, 3, 60; K = 1)
        fams = vcat(fill(Poisson(), 3), fill(Binomial(), 3))
        a = fit_gllvm(Y; family = fams, K = 1)
        b = fit_mixed_gllvm(Y; families = fams, K = 1)
        @test a isa MixedFamilyFit
        @test a.loglik == b.loglik
        @test a.β == b.β
        # num_lv is the gllvm synonym for K
        @test fit_gllvm(Y; family = fams, num_lv = 1).loglik == a.loglik
        @test_throws DimensionMismatch fit_gllvm(Y; family = fams[1:5], K = 1)
        @test_throws ArgumentError fit_gllvm(Y; family = fams)                      # K missing
        @test_throws ArgumentError fit_gllvm(Y; family = fams, K = 1, row_eff = :fixed)
        @test_throws ArgumentError fit_gllvm(Y; family = fams, K = 1, disp_group = :species)
        @test_throws ArgumentError fit_gllvm(Y; family = fams, K = 1, pervar = true)
        @test_throws ArgumentError fit_gllvm(Y; family = fams, K = 1, weights = ones(6, 60))
        @test_throws ArgumentError fit_gllvm(Y; family = fams, K = 1, optimizer = :lbfgs)
        @test_throws ArgumentError fit_gllvm(Y; family = fams, K = 1, link = LogLink())
    end
end
