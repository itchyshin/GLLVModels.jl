using GLLVModels, Test, Random

# Wave 2 sweep (w2-structured): a mesh built from points that include an exact duplicate
# leaves the duplicate as a vertex that belongs to no triangle. spde_fem then gave that
# node zero lumped mass and fit_spde_gaussian returned loglik = -Inf, converged = false
# with no error, although the spde_mesh_delaunay docstring says spde_fem throws.
@testset "spde_fem refuses a mesh node that belongs to no triangle" begin
    rng = MersenneTwister(30)
    co = rand(rng, 30, 2) .* 10
    nodes, tris = spde_mesh_delaunay(co)
    Cd, G = spde_fem(nodes, tris)
    @test all(>(0), Cd)                          # clean mesh unchanged

    cod = vcat(co, co[1:2, :])                   # two exact duplicates
    nd, td = spde_mesh_delaunay(cod)
    err = try spde_fem(nd, td); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("belong to no triangle", sprint(showerror, err))

    y = randn(rng, 32)
    err2 = try fit_spde_gaussian(y, nd, td, cod); nothing catch e; e end
    @test err2 isa ArgumentError

    # the clean mesh still fits
    @test isfinite(fit_spde_gaussian(randn(rng, 30), nodes, tris, co).loglik)
end
