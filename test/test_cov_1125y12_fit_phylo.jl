# Lane cursor-cloud-1125y12 — fit_phylo.jl
using GLLVModels, Test, Random

const _FP = GLLVModels

@testset "cov 1125y12 phylogenetic Gaussian one-trait fitter" begin
  @testset "_phylo_verdict plateau guard" begin
    @test _FP._phylo_verdict(true, 1e13) == (false, Inf)
    @test _FP._phylo_verdict(true, 100.0) == (true, 100.0)
    @test _FP._phylo_verdict(false, 50.0) == (false, 50.0)
    @test _FP._phylo_verdict(true, NaN) == (false, Inf)
  end

  rng = MersenneTwister(11251206)
  phy = augmented_phy("((A:0.2,B:0.3):0.4,(C:0.15,D:0.25):0.2);")
  p = phy.n_leaves
  y = randn(rng, p)

  @testset "fit_phylo_gaussian" begin
    fit = fit_phylo_gaussian(phy, y; iterations = 120, g_tol = 1e-5)
    @test fit isa PhyloGaussianFit
    @test isfinite(fit.negll)
    @test fit.σ²_eps > 0
    shown = sprint(show, fit)
    @test occursin("PhyloGaussianFit", shown)
  end

  @testset "dimension and type refusals" begin
    @test_throws DimensionMismatch fit_phylo_gaussian(phy, y[1:(end - 1)])
    @test_throws MethodError fit_phylo_gaussian(phy, randn(p, 2))
  end
end