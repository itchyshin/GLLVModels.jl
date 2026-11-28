# Issue #1249: a masked Poisson K = 1 fit whose species groups share no site must not stop
# with one group's loadings at exactly 0 (converged = true, below the block-fit sum).
using Test, GLLVModels, Random
const G = GLLVModels

@testset "Poisson masked disjoint blocks (#1249)" begin
    for seed in (1, 3)
        rng = MersenneTwister(seed); p, n = 10, 60
        b = 0.3 .+ 1.2 .* rand(rng, p); L = 0.6 .* randn(rng, p); z = randn(rng, n)
        off = repeat(0.5 .* randn(rng, n)', p) .+ 0.3 .* randn(rng, p, n)
        eta = b .+ off .+ L * z'
        Y = [rand(rng, G.Distributions.Poisson(exp(min(eta[i, j], 8.0)))) for i in 1:p, j in 1:n]
        M = trues(p, n); M[1:5, 31:60] .= false; M[6:10, 1:30] .= false
        j = fit_gllvm(Y; family = G.Poisson(), K = 1, mask = M, offset = off)
        a = fit_gllvm(Y[1:5, 1:30]; family = G.Poisson(), K = 1, offset = off[1:5, 1:30])
        c = fit_gllvm(Y[6:10, 31:60]; family = G.Poisson(), K = 1, offset = off[6:10, 31:60])
        @test all(!iszero, vec(j.Λ))
        @test j.loglik >= a.loglik + c.loglik - 1e-6
    end
end
