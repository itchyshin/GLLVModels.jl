# Issue #1220: an SPDE fit whose optimiser ends at its start must not report converged = true.
# Narrow rule: Optim accepted no iteration, or returned the start vector. A `g_tol` far above the
# gradient scale makes Optim stop at iteration 0 at the start point, the smallest way to show it.
# Nothing else changes: a fit that moves keeps converged = true and its estimates.
using GLLVModels, Test, Random

@testset "#1220 SPDE fits that end at their start report converged = false" begin
    m = 4
    xs = range(0.0, 1.0; length = m)
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
    rng = MersenneTwister(1220)
    M, p = 12, 2
    locs = 0.1 .+ 0.8 .* rand(rng, M, 2)
    Y = [0.3 * i + sin(5 * locs[j, 1]) + 0.2 * randn(rng) for i in 1:p, j in 1:M]

    # Gaussian SPDE fitter: ends at its start vector (kappa = tau = 1 are the defaults).
    g0 = fit_spde_gaussian(Y[1, :], nodes, tris, locs; g_tol = 1e8)
    @test g0.iterations == 0
    @test g0.κ == 1.0 && g0.τ == 1.0
    @test isfinite(g0.loglik)
    @test !g0.converged

    # SPDE-latent fitter.
    l0 = fit_spde_latent_gllvm(Y, nodes, tris, locs; family = GLLVModels.Normal(), K = 1, g_tol = 1e8)
    @test l0.iterations == 0
    @test l0.κ == 1.0 && l0.τ == 1.0
    @test isfinite(l0.loglik)
    @test !l0.converged

    # Control: with the default tolerance both fits move and still report converged = true.
    g1 = fit_spde_gaussian(Y[1, :], nodes, tris, locs)
    @test g1.iterations > 0 && g1.converged && isfinite(g1.loglik)
    l1 = fit_spde_latent_gllvm(Y, nodes, tris, locs; family = GLLVModels.Normal(), K = 1)
    @test l1.iterations > 0 && l1.converged && isfinite(l1.loglik)
end
