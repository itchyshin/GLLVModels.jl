# Issue #747: the SPDE-latent (and phylo x Poisson X_lv) mode search used to return the
# last iterate as if converged when `maxiter` ran out, and the fit objective's bare
# `catch` turned every exception (including caller errors) into a sentinel.
using GLLVModels, Test, LinearAlgebra, SparseArrays, Random, Distributions

@testset "#747 SPDE-latent mode search reports non-convergence" begin
    Random.seed!(747)
    m = 5
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
    p, M, K = 4, 30, 1
    locs = rand(M, 2)
    Cdiag, G = spde_fem(nodes, tris)
    A = spde_projector(nodes, tris, locs)
    Q = spde_precision(Cdiag, G, 3.0, 1.0)
    Λ = reshape([0.9, -0.6, 0.7, 0.5], p, K)
    β = fill(0.8, p)
    Y = rand(0:9, p, M)
    Ntr = ones(Int, p, M)
    args = (Poisson(), Y, Ntr, Λ, β, LogLink(), A, sparse(Q))

    # A single Newton step is nowhere near `tol`: no mode, so no Laplace value.
    @test GLLVModels._spde_latent_mode(args...; maxiter = 1) === nothing
    @test spde_latent_marginal_loglik(args[1:7]..., Q; maxiter = 1) == -Inf

    # With enough iterations the search converges, and the value no longer
    # depends on the cap.
    ℓ50  = spde_latent_marginal_loglik(args[1:7]..., Q; maxiter = 50)
    ℓ200 = spde_latent_marginal_loglik(args[1:7]..., Q; maxiter = 200)
    @test isfinite(ℓ50)
    @test ℓ50 ≈ ℓ200 atol = 1e-8

    # The fit objective must not swallow a caller error: an extra trial-count
    # column used to become a 1e12 plateau and a silent NOT CONVERGED fit.
    @test_throws DimensionMismatch fit_spde_latent_gllvm(
        Y, nodes, tris, locs; family = Poisson(), K = K,
        N = ones(Int, p, M + 1), iterations = 3)
end

@testset "#747 phylo x Poisson X_lv mode search reports non-convergence" begin
    Random.seed!(747)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n, K, q_lv = 7, 1, 1
    beta = 0.15 .* randn(p)
    Lambda = reshape(range(0.15, 0.55; length = p), p, K)
    alpha_lv = reshape([0.35], q_lv, K)
    X_lv = reshape(collect(range(-1.2, 1.2; length = n)), n, q_lv)
    Y = rand(0:5, p, n)
    f(maxiter) = GLLVModels._phylo_poisson_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, 0.45, phy, X_lv; maxiter = maxiter, tol = 1e-10)
    @test f(1) == -Inf
    @test isfinite(f(120))
end
