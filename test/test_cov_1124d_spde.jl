# Lane 1124d: SPDE mesh / FEM / latent postfit argument validation (no long fits).
using Test, GLLVModels, LinearAlgebra, Distributions
const G = GLLVModels

@testset "1124d spde_mesh.jl validation" begin
  @test_throws ArgumentError spde_mesh_grid(randn(5, 3))
  @test_throws ArgumentError spde_mesh_grid(randn(3, 2); nx = 1)
  @test_throws ArgumentError spde_mesh_grid(randn(3, 2); ny = 1)
  @test_throws ArgumentError spde_mesh_grid((0.0, 1.0), (0.0, 1.0); nx = 1)
  nodes, tris = spde_mesh_grid(randn(4, 2); nx = 3, ny = 3)
  @test size(nodes, 2) == 2
  @test size(tris, 2) == 3
end

@testset "1124d spde_delaunay.jl validation" begin
  @test_throws ArgumentError spde_mesh_delaunay(randn(5, 3))
  @test_throws ArgumentError spde_mesh_delaunay(randn(2, 2))
  @test_throws ArgumentError spde_mesh_delaunay([1.0 NaN; 2.0 3.0])
end

@testset "1124d spde.jl validation" begin
  nodes = [0.0 0.0; 1.0 0.0; 0.0 1.0]
  tris = [1 2 3]
  C, G = spde_fem(nodes, tris)
  @test_throws ArgumentError spde_precision(C, G, 0.0, 1.0)
  @test_throws ArgumentError spde_precision(C, G, 1.0, -0.1)
  @test_throws ArgumentError spde_precision(C, G, 1.0, 1.0; α = 3)
  @test_throws ArgumentError spde_projector(nodes, tris, randn(3, 3); outside_tol = -1.0)
  bad_nodes = [0.0 0.0; 1.0 0.0]
  @test_throws ArgumentError spde_fem(bad_nodes, tris)
  @test_throws ArgumentError spde_fem(nodes, [1 2])
end

@testset "1124d spde_latent_postfit.jl type checks" begin
  fit = SPDELatentFit(
    zeros(2), ones(2, 1), 1.0, 1.0, NaN, LogLink(), Poisson(),
    -1.0, false, 0, [0.0 0.0; 1.0 0.0; 0.0 1.0], [1 2 3],
    zeros(Float64, 0, 0),
  )
  Y = rand(0:2, 2, 2)
  locs = [0.2 0.2; 0.5 0.5]
  @test_throws ArgumentError G.predict(fit, Y, locs; type = :mean)
  @test_throws ArgumentError G.predict_spatial(fit, Y, locs, locs; type = :nonsense)
end