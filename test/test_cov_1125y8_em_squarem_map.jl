# Lane cursor-cloud-1125y8: em_squarem.jl EM map and one accelerated step.
using Test, GLLVModels, LinearAlgebra, Random

const G = GLLVModels

@testset "cov 1125y8 em squarem map" begin
    rng = MersenneTwister(1125088)
    phy = random_balanced_tree(4; branch_length = 0.25)
    Σ = G.sigma_phy_dense(phy)
    p, n, K = phy.n_leaves, 20, 1
    y = randn(rng, p, n)
    Λ_B = fill(0.2, p, K)
    θ = G._pack_phylo(Λ_B, 0.6, fill(0.1, p))
    θ1, floor1 = G._em_map_phylo(θ, y, Σ, p, K)
    @test length(θ1) == length(θ)
    @test floor1 isa Bool
  @test all(isfinite, θ1)
    ll1 = G._loglik_phylo(θ, y, Σ, p, K)
    ll2 = G._loglik_phylo(θ1, y, Σ, p, K)
    @test isfinite(ll1) && isfinite(ll2)

    fit = G.em_fit_phylo_squarem(y, K, Σ; max_iter = 5, tol = 1e-6, safety_check = false, safety_warmstart = false)
    @test fit isa G.EMPhyloFit
    @test isfinite(fit.logLik)
    @test fit.n_iter >= 1
    @test length(fit.σ_phy) == p

    @test length(G._unpack_phylo(θ, p, K)[3]) == p
end