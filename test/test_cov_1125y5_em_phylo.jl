# Lane 1125y5: em_phylo.jl sparse operators and short EM trace.
using Test, GLLVModels, Random, LinearAlgebra, SparseArrays, Statistics
const G = GLLVModels

@testset "1125y5 em_phylo sparse operators" begin
    phy = augmented_phy("((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2);")
    p, K_B, n = phy.n_leaves, 1, 12
    Λ_B = reshape(collect(range(0.3, 0.7; length = p)), p, K_B)
    σ_phy = fill(0.5, p)
    s = G.build_AnB_sparse(Λ_B, 0.4, σ_phy, phy, n; σ²_phy = 0.5)
    @test s isa G.AnBSparseSolver
    rhs = randn(p)
    v = G.solve_AnB(s, rhs)
    @test length(v) == p
    @test all(isfinite, v)
    bl = G.blup_phylo_sparse(randn(p, n), Λ_B, 0.4, σ_phy, phy; σ²_phy = 0.5)
    @test length(bl) == p
    Λφw = σ_phy .* rhs
    μz = G.μ_z_from_components(s, 0.5, Λφw, n)
    @test length(μz) == p

    y = Λ_B * randn(K_B, n) .+ 0.3 .* randn(p, n)
    Σ_phy = G.sigma_phy_dense(phy; σ²_phy = 1.0)
    emf = em_fit_phylo(y, K_B, Σ_phy; max_iter = 8, tol = 1e-4)
    @test emf isa G.EMPhyloFit
    @test isfinite(emf.logLik)
    @test length(emf.loglik_trace) >= 2
    npar = G._nparams(emf)
    @test npar > 0
    Iobs = em_observed_information(emf, y, Σ_phy)
    @test size(Iobs.cov) == (npar, npar)
    @test Iobs.pd isa Bool

    floorσ = G._sigma_eps2_data_floor(var(vec(y)), n * p, p)
    @test floorσ > 0
    @test G._observed_information_is_pd(Iobs.cov) in (true, false)
end