using GLLVModels, Test, Random, LinearAlgebra, Distributions

@testset "W tier and diag RE" begin
    @testset "K_W=0 has_diag=false reproduces J1 behaviour" begin
        Random.seed!(0)
        p, K, n = 4, 1, 60
        Λ_B = reshape([0.7, 0.5, 0.3, -0.2], p, K)
        y = Λ_B * randn(K, n) + 0.5 * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        @test fit.converged
        @test fit.pars.Λ_W === nothing
        @test fit.pars.σ²_B === nothing
        @test fit.pars.σ²_W === nothing
    end

    @testset "matches direct MvNormal with full Σ_y_site" begin
        # Build a known model with W tier + diag RE, compute log-lik two ways.
        Random.seed!(1)
        p, K_B, K_W, n = 4, 1, 1, 30
        Λ_B = reshape([0.8, 0.5, 0.3, -0.2], p, K_B)
        Λ_W = reshape([0.4, 0.3, 0.2, 0.1], p, K_W)
        σ_eps = 0.5
        σ²_B = [0.10, 0.20, 0.05, 0.15]
        σ²_W = [0.05, 0.10, 0.02, 0.08]
        # The W-tier scores are shared by all traits of a unit (as in the C++
        # twin, issue #135), so Λ_W adds its full cross-trait block.
        d_total = σ²_B .+ σ²_W .+ σ_eps^2
        Σ_y = Λ_B * Λ_B' + Λ_W * Λ_W' + Diagonal(d_total)
        d_dist = MvNormal(zeros(p), Symmetric(Σ_y))
        y = rand(d_dist, n)
        ll_direct = sum(logpdf(d_dist, y[:, s]) for s in 1:n)
        ll_ours = GLLVModels.gaussian_marginal_loglik(
            y, Λ_B, σ_eps;
            Λ_W = Λ_W, σ²_B = σ²_B, σ²_W = σ²_W
        )
        @test ll_ours ≈ ll_direct rtol = 1e-10
    end

    @testset "recovery: W tier + diag RE on a clean fixture" begin
        # p = 8 traits: with the W tier shared across traits (issue #135) the
        # K_B = 1, K_W = 1 model is a rank-2 factor model, which p = 5 leaves
        # with a single spare degree of freedom (Heywood cases on this seed).
        Random.seed!(2)
        p, K_B, K_W, n = 8, 1, 1, 1000
        Λ_B = reshape([0.7, 0.5, 0.4, -0.3, 0.2, 0.6, -0.5, 0.3], p, K_B)
        Λ_W = reshape([0.3, 0.4, 0.2, 0.3, 0.1, -0.4, 0.2, 0.5], p, K_W)
        σ_eps = 0.5
        σ²_B = fill(0.10, p)
        σ²_W = fill(0.05, p)
        # Simulate exactly as the engine (and the C++ twin) assembles η:
        #   y[t, s] = Λ_B[t, :] η_B[:, s] + Λ_W[t, :] η_W[:, s]
        #           + s_B[t, s] + s_W[t, s] + σ_eps ε[t, s]
        # with one W-tier score vector per unit, shared by all traits.
        η_B = randn(K_B, n)
        η_W = randn(K_W, n)
        s_B = sqrt.(σ²_B) .* randn(p, n)
        s_W = sqrt.(σ²_W) .* randn(p, n)
        y = Λ_B * η_B + Λ_W * η_W
        y += s_B + s_W + σ_eps * randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K_B, K_W = K_W, has_diag = true)
        @test fit.converged
        # Only Λ_B Λ_Bᵀ + Λ_W Λ_Wᵀ is identified in this one-unit-per-column
        # layout, so recovery is checked on the full per-unit covariance.
        Σ_true = Λ_B * Λ_B' + Λ_W * Λ_W' + Diagonal(σ²_B .+ σ²_W .+ σ_eps^2)
        Σ_hat  = fit.pars.Λ * fit.pars.Λ' + fit.pars.Λ_W * fit.pars.Λ_W' +
                 Diagonal(fit.pars.σ²_B .+ fit.pars.σ²_W .+ fit.pars.σ_eps^2)
        @test sigma_y_site(fit) ≈ Σ_hat
        # Diagonal recovery (per-trait observation variance)
        @test maximum(abs.(diag(Σ_hat) .- diag(Σ_true)) ./ diag(Σ_true)) < 0.20
        # Σ_y recovery (rotation invariant)
        @test norm(Σ_true - Σ_hat) / norm(Σ_true) < 0.15
    end

    @testset "AD-friendly" begin
        using ForwardDiff
        Random.seed!(3)
        p, K_B, K_W, n = 4, 1, 1, 30
        y = randn(p, n)
        # params layout: [log_σ_eps; log_σ_B (p); log_σ_W (p); θ_rr_B; θ_rr_W]
        rr_B = GLLVModels.rr_theta_len(p, K_B)
        rr_W = GLLVModels.rr_theta_len(p, K_W)
        n_params = 1 + 2 * p + rr_B + rr_W
        params₀ = zeros(n_params)
        params₀[1] = 0.0   # log_σ_eps
        params₀[2:(1 + p)] .= log(0.1)   # log_σ_B
        params₀[(2 + p):(1 + 2 * p)] .= log(0.1)  # log_σ_W
        params₀[(2 + 2 * p):(1 + 2 * p + rr_B)] .= GLLVModels.init_theta_rr(p, K_B)
        params₀[(2 + 2 * p + rr_B):end] .= GLLVModels.init_theta_rr(p, K_W)
        spec = (q = 0, p = p, K_B = K_B, K_W = K_W, has_diag = true)
        nll = params -> GLLVModels.gaussian_nll_packed(params, y; spec = spec)
        g = ForwardDiff.gradient(nll, params₀)
        @test all(isfinite, g)
        @test length(g) == n_params
    end
end
