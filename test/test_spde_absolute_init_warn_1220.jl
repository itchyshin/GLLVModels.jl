# Issue #1220: absolute κ_init/τ_init defaults are poorly scaled on tiny or huge domains (#1220).
using GLLVModels, Test, Random

@testset "SPDE fits warn on O(1) κ_init/τ_init with non-O(1) domain (#1220)" begin
    m = 4
    xs = range(0.0, 1e-3; length = m)
    nodes = Matrix{Float64}(undef, m * m, 2)
    nid(i, j) = (j - 1) * m + i
    for j in 1:m, i in 1:m
        nodes[nid(i, j), :] = [xs[i], xs[j]]
    end
    tris = Matrix{Int}(undef, 2 * (m - 1)^2, 3)
    t = 0
    for j in 1:(m - 1), i in 1:(m - 1)
        a, b, c, d = nid(i, j), nid(i + 1, j), nid(i, j + 1), nid(i + 1, j + 1)
        t += 1; tris[t, :] = [a, b, d]
        t += 1; tris[t, :] = [a, d, c]
    end
    rng = MersenneTwister(12201)
    M = 8
    locs = nodes[rand(rng, 1:size(nodes, 1), M), :]
    y = sin.(4 .* locs[:, 1]) .+ 0.1 .* randn(rng, M)

    fit = @test_logs (:info, r"domain extent") match_mode = :any fit_spde_gaussian(
        y, nodes, tris, locs; g_tol = 1e8)
    @test fit isa GLLVModels.SPDEGaussianFit
end