# Lane cursor-cloud-1125y6: extractors.jl level/part dispatch beyond test_extractors.jl smoke.
using Test, GLLVModels, Random, LinearAlgebra

const G = GLLVModels

@testset "1125y6 extractors sigma parts" begin
  rng = MersenneTwister(112506)
  p, n = 3, 10
  Y = randn(rng, p, n)
  fit = fit_gaussian_gllvm(Y; K = 1, has_diag = true)

  @testset "_canonical_level" begin
    @test G._canonical_level(:unit) === :unit
    @test G._canonical_level(:B) === :unit
    @test G._canonical_level(:unit_obs) === :unit_obs
    @test G._canonical_level(:W) === :unit_obs
    @test G._canonical_level(:site) === :site
    @test_throws ArgumentError G._canonical_level(:phy)
  end

  @testset "_cov2cor edge diagonals" begin
    Σ = [0.0 0.1; 0.1 1.0]
    R = G._cov2cor(Σ)
    @test R[2, 2] == 1.0
    @test isnan(R[1, 1])
    Σ2 = [2.0 0.5; 0.5 4.0]
    @test G._cov2cor(Σ2) ≈ [1.0 0.5 / sqrt(8); 0.5 / sqrt(8) 1.0] rtol = 1e-12
  end

  @testset "extract_Sigma parts on GllvmFit" begin
    sh = extract_Sigma(fit; level = :unit, part = :shared)
    @test sh.part === :shared
    @test size(sh.Sigma) == (p, p)
    @test sh.Sigma ≈ fit.pars.Λ * fit.pars.Λ' atol = 1e-12
    uq = extract_Sigma(fit; level = :unit, part = :unique)
    @test uq.part === :unique
    @test length(uq.s) == p
    @test all(>=(0), uq.s)
    tot = extract_Sigma(fit; level = :unit, part = :total)
    @test haskey(tot, :R)
    @test tot.Sigma ≈ G._sigma_unit(fit) atol = 1e-12
    site = extract_Sigma(fit; level = :site)
    @test site.level === :site
    @test size(site.Sigma) == (p, p)
    @test_throws ArgumentError extract_Sigma(fit; level = :site, part = :shared)
    @test_throws ArgumentError extract_Sigma(fit; part = :nope)
  end

  @testset "TwoLevelFit extract_Sigma" begin
    ΛB = reshape([0.7, 0.6, 0.5], p, 1)
    σB = fill(0.1, p)
    ΛW = reshape([0.2, 0.3, 0.4], p, 1)
    σW = fill(0.05, p)
    ΣB = Matrix(ΛB * ΛB' + Diagonal(σB))
    ΣW = Matrix(ΛW * ΛW' + Diagonal(σW))
    tl = TwoLevelFit(ΛB, σB, ΛW, σW, ΣB, ΣW, n, -1.0, true, 1)
  sh = extract_Sigma(tl; level = :W, part = :shared)
    @test sh.level === :unit_obs
    @test sh.Sigma ≈ ΛW * ΛW' atol = 1e-12
    @test_throws ArgumentError extract_Sigma(tl; level = :site)
  end

  @testset "extract_Sigma_table shape" begin
    tab = extract_Sigma_table(fit; level = :B, part = :total)
    @test length(tab) == p * (p + 1) ÷ 2
    @test tab[1].trait_i == 1
    @test all(r -> r.trait_j >= r.trait_i, tab)
    @test_throws ArgumentError extract_Sigma_table(fit; part = :unique)
  end
end