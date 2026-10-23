using GLLVModels, Test, Random, LinearAlgebra

# Convergence sweep: #574 (ordinal logit false convergence) and #498 item 1
# (binomial runaway-loading diagnostic).

@testset "convergence sweep" begin

    @testset "#574 ordinal pertrait Laplace mode search is damped (no cliff)" begin
        # Mechanism: undamped Newton in `_ordinal_laplace_mode_pertrait` overshoots to a
        # far worse site mode (on random draws ~10% of the time, gaps up to ~470), so the
        # Laplace marginal has ~50-unit cliffs under 1e-5 parameter steps; L-BFGS then
        # stalls on a zero-length step and `Optim.converged` reports success tens of
        # logLik below the optimum. Property: the returned mode never has a lower
        # penalised site log-posterior than the z = 0 start Newton begins from.
        G = GLLVModels
        rng = MersenneTwister(1)
        p, K = 8, 2
        C = fill(6, p)
        n_bad = 0
        for _ in 1:300
            Λ = 2.5 .* randn(rng, p, K); β = randn(rng, p)
            τ = zeros(p, 5)
            for t in 1:p
                τ[t, :] = vcat(0.0, cumsum(rand(rng, 4) .* 2.0 .+ 0.1))
            end
            y = rand(rng, 1:6, p)
            logpost(z) = begin
                η = G._clamp_eta.(β .+ Λ * z)
                sum(log(max(G._ord_prob(y[t], η[t], G._trait_cutpoints(τ, C, t), LogitLink()), 1e-12))
                    for t in 1:p) - 0.5 * dot(z, z)
            end
            z = G._ordinal_laplace_mode_pertrait(y, Λ, β, τ, C, LogitLink())
            logpost(z) >= logpost(zeros(K)) - 1e-9 || (n_bad += 1)
        end
        @test n_bad == 0
    end

    @testset "#498 binomial runaway-loading diagnostic" begin
        G = GLLVModels
        healthy = [0.8 0.1; 0.5 0.4; 0.3 -0.2; -0.6 0.3; 0.2 0.5; -0.3 0.4]
        @test isempty(G._binomial_runaway_loadings(healthy))
        runaway = copy(healthy); runaway[3, 1] = 42.5            # relative ~ 80, absolute > 8
        @test G._binomial_runaway_loadings(runaway) == [3]
        @test_logs (:warn, r"runaway loadings on trait\(s\) 3") G._warn_runaway_loadings(runaway)
        # absolute arm: whole matrix inflated (ratio is blind to this)
        @test G._binomial_runaway_loadings(10 .* healthy) == [1]
        # negative control: no warning for a healthy matrix, and for a healthy fit
        @test_logs G._warn_runaway_loadings(healthy)
        Random.seed!(11)
        p, n, K = 6, 150, 1
        Λt = reshape([0.9, 0.7, 0.5, -0.6, 0.4, -0.8], p, K)
        Z = randn(K, n)
        Y = [rand() < 1 / (1 + exp(-(0.2 + Λt[t, 1] * Z[1, i]))) ? 1 : 0 for t in 1:p, i in 1:n]
        fit = @test_logs fit_binomial_gllvm(Y; K = 1)
        @test maximum(abs, fit.Λ) < 8
    end
end
