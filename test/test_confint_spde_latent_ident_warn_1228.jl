# Issue #1228: Lambda and tau are scale-identified; confint should warn (#1228).
using GLLVModels, Test, Random

function _grid_mesh(m, L)
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

@testset "confint_spde_latent warns on Lambda/tau identification (#1228)" begin
    Random.seed!(1228)
    nodes, tris = _grid_mesh(5, 1.0)
    M, p = 20, 3
    locs = nodes[rand(1:size(nodes, 1), M), :]
    Y = 0.2 .+ randn(p, M) .* 0.3
    fit = fit_spde_latent_gllvm(Y, nodes, tris, locs; family = GLLVModels.Normal(), K = 1, iterations = 80)
    ci = @test_logs (:warn, r"scale-identified") match_mode = :any confint_spde_latent(fit, Y, locs)
    @test ci.pd_hessian isa Bool
end