using GLLVModels, Test, LinearAlgebra, SparseArrays, Random, Distributions

# Build a regular triangulated grid over [0, L]² (same construction as test_spde.jl).
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

@testset "SPDE latent GLLVModels (joint Laplace over the GMRF)" begin

    # ---- Anchor 1: i.i.d. reduction (Q = I, A = I) == per-site Laplace ------
    # With identity projector and identity precision the K latent fields become
    # independent N(0,I) per site, so the joint-Laplace marginal must equal the
    # validated independent-site `marginal_loglik_laplace` to machine precision.
    @testset "Q=I, A=I reduces to independent-site Laplace (Poisson)" begin
        Random.seed!(11)
        p, M, K = 4, 7, 2
        β = randn(p) .* 0.3
        Λ = randn(p, K) .* 0.4
        Y = rand(0:6, p, M)
        Aid = sparse(1.0I, M, M)          # sites coincide with nodes
        Qid = sparse(1.0I, M, M)          # standard-normal prior ⇒ z_s ~ N(0,I)

        ℓ_joint = spde_latent_marginal_loglik(Poisson(), Y, ones(Int, p, M),
                                              Λ, β, LogLink(), Aid, Qid;
                                              maxiter = 100, tol = 1e-12)
        ℓ_site = GLLVModels.marginal_loglik_laplace(Poisson(), Y, ones(Int, p, M),
                                         Λ, β, LogLink(); maxiter = 100, tol = 1e-12)
        @test isapprox(ℓ_joint, ℓ_site; atol = 1e-7, rtol = 1e-8)
    end

    # ---- Anchor 2: conjugate Gaussian == closed-form SPDE field marginal ----
    # Single species, identity link, Λ = 1: the latent SPDE field IS the Gaussian
    # spatial field of spde_fit.jl. The Laplace approximation is exact for a
    # conjugate Gaussian, so the joint-Laplace marginal must reproduce
    # `spde_gaussian_marginal_loglik` to machine precision.
    @testset "Gaussian/identity == spde_gaussian_marginal_loglik" begin
        Random.seed!(22)
        m, L = 11, 10.0
        nodes, tris = _grid_mesh(m, L)
        Cdiag, G = spde_fem(nodes, tris)
        κ, τ = 0.8, 1.3
        Q = spde_precision(Cdiag, G, κ, τ; α = 2)

        # M observation sites placed exactly on a subset of mesh nodes.
        site_nodes = [1, 13, 27, 40, 55, 73, 90, 110, 121]
        locs = nodes[site_nodes, :]
        A = spde_projector(nodes, tris, locs)
        Mn = length(site_nodes)

        σ2 = 0.45
        μ = 0.7
        y = randn(Mn) .* 0.9 .+ μ
        Y = reshape(y, 1, Mn)                       # p = 1 species

        ℓ_latent = spde_latent_marginal_loglik(Normal(0.0, sqrt(σ2)), Y,
                        ones(1, Mn), reshape([1.0], 1, 1), [μ], IdentityLink(), A, Q;
                        maxiter = 50, tol = 1e-12)
        ℓ_closed = spde_gaussian_marginal_loglik(y, A, Q, σ2; μ = μ)
        @test isapprox(ℓ_latent, ℓ_closed; atol = 1e-7, rtol = 1e-9)
    end

    # ---- Fit driver smoke test (Poisson, small mesh) -----------------------
    @testset "fit_spde_latent_gllvm runs (Poisson)" begin
        Random.seed!(33)
        m, L = 6, 5.0
        nodes, tris = _grid_mesh(m, L)
        N = m * m
        # Observe at a scatter of mesh nodes.
        site_nodes = collect(1:3:N)
        locs = nodes[site_nodes, :]
        Mn = length(site_nodes)
        p = 3
        Y = rand(0:4, p, Mn)

        fit = fit_spde_latent_gllvm(Y, nodes, tris, locs;
                                    family = Poisson(), K = 1,
                                    iterations = 30, newton_maxiter = 30)
        @test isfinite(fit.loglik)
        @test size(fit.Λ) == (p, 1)
        @test length(fit.β) == p
        @test fit.κ > 0 && fit.τ > 0
        @test isnan(fit.dispersion)              # Poisson has no dispersion
    end

    # ---- Dispersion families: NB → Poisson reduction (marginal anchor) -----
    # As r → ∞ the NB2 marginal must collapse to the Poisson marginal at the same
    # (Λ, β, A, Q). Deterministic — no fit.
    @testset "NB(r→∞) marginal == Poisson marginal" begin
        Random.seed!(44)
        p, M, K = 3, 6, 1
        β = randn(p) .* 0.3
        Λ = randn(p, K) .* 0.4
        Y = rand(0:5, p, M)
        Aid = sparse(1.0I, M, M)
        Qid = sparse(1.0I, M, M)

        ℓ_nb = spde_latent_marginal_loglik(NegativeBinomial(1e7, 0.5), Y,
                        ones(Int, p, M), Λ, β, LogLink(), Aid, Qid;
                        maxiter = 100, tol = 1e-12)
        ℓ_pois = spde_latent_marginal_loglik(Poisson(), Y, ones(Int, p, M),
                        Λ, β, LogLink(), Aid, Qid; maxiter = 100, tol = 1e-12)
        @test isapprox(ℓ_nb, ℓ_pois; atol = 1e-3)
    end

    # ---- Dispersion-family fit drivers: smoke (Gaussian σ², NB r) -----------
    @testset "fit_spde_latent_gllvm dispersion families run" begin
        Random.seed!(55)
        m, L = 6, 5.0
        nodes, tris = _grid_mesh(m, L)
        Nn = m * m
        site_nodes = collect(1:3:Nn)
        locs = nodes[site_nodes, :]
        Mn = length(site_nodes)

        # Gaussian (σ²) — single species.
        yG = reshape(randn(Mn) .+ 0.5, 1, Mn)
        fG = fit_spde_latent_gllvm(yG, nodes, tris, locs;
                                   family = Normal(), K = 1,
                                   iterations = 30, newton_maxiter = 30)
        @test isfinite(fG.loglik)
        @test fG.dispersion > 0                  # σ²

        # Negative binomial (r) — two species of counts.
        YN = rand(0:4, 2, Mn)
        fN = fit_spde_latent_gllvm(YN, nodes, tris, locs;
                                   family = NegativeBinomial(1.0, 0.5), K = 1,
                                   iterations = 30, newton_maxiter = 30)
        @test isfinite(fN.loglik)
        @test fN.dispersion > 0                  # r
    end
end
