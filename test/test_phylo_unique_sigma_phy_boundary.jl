# Issue #1179: em_fit_phylo / fit_gaussian_gllvm(has_phy_unique=true) must not
# report converged=true at σ_phy = 0 when a Σ_phy-scaled restart reaches a
# substantially higher log-likelihood on the same data and model.

using GLLVModels, Test, Random, LinearAlgebra, Statistics

function _issue1179_rand_tree(nt, seed, c = 1.0)
    rng = MersenneTwister(seed)
    nodes = [("t$i", 0.0) for i in 1:nt]
    while length(nodes) > 1
        a = splice!(nodes, rand(rng, 1:length(nodes)))
        b = splice!(nodes, rand(rng, 1:length(nodes)))
        h = max(a[2], b[2]) + 0.1 + rand(rng) * 0.5
        push!(nodes, ("(" * a[1] * ":" * string(round(c * (h - a[2]), digits = 13)) * "," *
                     b[1] * ":" * string(round(c * (h - b[2]), digits = 13)) * ")", h))
    end
    nodes[1][1] * ";"
end

simY(S, ds, n = 30) =
    0.8 .* (cholesky(Symmetric(S)).L * randn(MersenneTwister(ds), size(S, 1))) .+
    0.5 .* randn(MersenneTwister(ds + 10), size(S, 1), n)

@testset "phylo_unique σ_phy origin stall (#1179)" begin
    nt, ts, ds, c = 12, 5, 3, 0.25
    phy = augmented_phy(_issue1179_rand_tree(nt, ts, c))
    S0 = Matrix(sigma_phy_dense(phy; σ²_phy = 1.0))
    Y = simY(S0, ds)
    S = c .* S0
    f0 = fit_gaussian_gllvm(Y; K = 1)
    s0 = 0.5 / sqrt(mean(diag(S)))

    e = em_fit_phylo(Y, 1, S; tol = 1e-10, max_iter = 5000)
    g = fit_gaussian_gllvm(Y; K = 1, has_phy_unique = true, Σ_phy = S)
    g_scr = fit_gaussian_gllvm(Y; K = 1, has_phy_unique = true, Σ_phy = S,
                               σ_phy_init = 0.1, phy_unique_boundary_restart = false)
    g_best = fit_gaussian_gllvm(Y; K = 1, has_phy_unique = true, Σ_phy = S, σ_phy_init = s0)

    @test g_best.logLik > f0.logLik + 1.0
    @test maximum(abs.(g_best.pars.σ_phy)) > 0.05

    for fit in (e, g)
        stalled = maximum(abs.(fit isa EMPhyloFit ? fit.σ_phy : fit.pars.σ_phy)) < 1e-6 &&
                  abs((fit isa EMPhyloFit ? fit.logLik : fit.logLik) - f0.logLik) < 1e-3
        @test !stalled || !fit.converged
        if fit isa EMPhyloFit
            @test !stalled || fit.sigma_phy_boundary
        end
    end

    @test abs(g.logLik - g_best.logLik) < 0.5
    @test maximum(abs.(g.pars.σ_phy)) > 0.05
    @test g.converged

    # Explicit tiny start without restart may still stall; then converged must be false.
    if maximum(abs.(g_scr.pars.σ_phy)) < 1e-6 && abs(g_scr.logLik - f0.logLik) < 1e-3
        @test !g_scr.converged
    else
        @test g_scr.logLik > f0.logLik + 1.0
    end
end