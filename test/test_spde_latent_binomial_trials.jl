using GLLVModels, Test, LinearAlgebra, SparseArrays, Random
using Distributions: Binomial

# Tiny regular mesh, same construction as test_spde_latent_postfit.jl.
function _grid_mesh_bin(m, L)
    xs = range(0.0, L; length = m)
    N = m * m
    nodes = Matrix{Float64}(undef, N, 2)
    nodeid(i, j) = (j - 1) * m + i
    for j in 1:m, i in 1:m
        nodes[nodeid(i, j), 1] = xs[i]
        nodes[nodeid(i, j), 2] = xs[j]
    end
    tris = Matrix{Int}(undef, 2 * (m - 1) * (m - 1), 3)
    t = 0
    for j in 1:(m - 1), i in 1:(m - 1)
        a = nodeid(i, j); b = nodeid(i + 1, j)
        c = nodeid(i, j + 1); d = nodeid(i + 1, j + 1)
        t += 1; tris[t, :] = [a, b, d]
        t += 1; tris[t, :] = [a, d, c]
    end
    return nodes, tris
end

@testset "SPDE-latent Binomial postfit uses stored trial counts (#746)" begin
    Random.seed!(746)
    m, L = 4, 4.0
    nodes, tris = _grid_mesh_bin(m, L)
    Nn = m * m
    site_nodes = collect(1:4:Nn)          # 4 sites
    locs = nodes[site_nodes, :]
    M = size(locs, 1)
    p, K = 2, 1
    Ntr = fill(20.0, p, M)
    Y = [rand(Binomial(20, 0.45)) for _ in 1:p, _ in 1:M]

    fit = fit_spde_latent_gllvm(Y, nodes, tris, locs;
                                family = Binomial(), K = K, N = Ntr,
                                iterations = 20, newton_maxiter = 20)

    fam = GLLVModels._spde_make_family(fit.family, Float64[])
    Cdiag, G = spde_fem(nodes, tris)
    Qs = sparse(spde_precision(Cdiag, G, fit.κ, fit.τ; α = 2))
    A = spde_projector(nodes, tris, locs)
    U_ones = GLLVModels._spde_latent_mode(fam, Y, ones(p, M), fit.Λ, fit.β,
                                          fit.link, A, Qs)
    U_N = GLLVModels._spde_latent_mode(fam, Y, Ntr, fit.Λ, fit.β,
                                       fit.link, A, Qs)
    @test U_ones !== nothing
    @test U_N !== nothing
    Z_ones = A * U_ones
    Z_N = A * U_N
    # Trials above 1 must change the field mode; otherwise this test is vacuous.
    @test !(Z_N ≈ Z_ones)

    Z = GLLVModels.getLV(fit, Y, locs)
    @test Z ≈ Z_N
    @test !(Z ≈ Z_ones)

    η = GLLVModels.predict(fit, Y, locs; type = :link)
    @test η ≈ fit.β .+ fit.Λ * Z_N'

    η_spatial = GLLVModels.predict_spatial(fit, Y, locs, locs; type = :link)
    @test η_spatial ≈ η
end
