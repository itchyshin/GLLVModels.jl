using GLLVModels, Test, Random, Distributions, Statistics

@testset "Delta-Gamma family" begin
    @testset "Λ = 0 reduces to independent two-part loglik (exact)" begin
        Random.seed!(160)
        p, K, n = 6, 2, 50
        βz = 0.4 .* randn(p)                       # occurrence logits
        βc = 0.5 .* randn(p)                       # positive log-mean
        α = 3.0
        π = inv.(1 .+ exp.(-βz))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            if rand() < π[t]
                μ = exp(βc[t])
                Y[t, s] = rand(Gamma(α, μ / α))
            end                                     # else stays 0 (absence)
        end

        ll = GLLVModels.delta_gamma_marginal_loglik_laplace(Y, zeros(p, K), βz, βc, α)
        ref = 0.0
        for t in 1:p, s in 1:n
            ref += Y[t, s] > 0 ? (log(π[t]) + logpdf(Gamma(α, exp(βc[t]) / α), Y[t, s])) :
                                 log(1 - π[t])
        end
        @test ll ≈ ref atol = 1e-8
    end

    @testset "K = 1 single site ≈ quadrature" begin
        Random.seed!(161)
        p, K = 6, 1
        βz = 0.3 .* randn(p)
        βc = 0.5 .* randn(p)
        α = 4.0
        Λc = reshape(0.4 .* randn(p), p, 1)
        ztrue = randn()
        π = inv.(1 .+ exp.(-βz))
        y = zeros(p)
        for t in 1:p
            if rand() < π[t]
                μ = exp(βc[t] + Λc[t, 1] * ztrue)
                y[t] = rand(Gamma(α, μ / α))
            end
        end
        Y = reshape(y, p, 1)
        ll_lap = GLLVModels.delta_gamma_marginal_loglik_laplace(Y, Λc, βz, βc, α)

        zs = range(-10, 10; length = 8001); dz = step(zs)
        marg = 0.0
        for z in zs
            lp = 0.0
            for t in 1:p
                πt = inv(1 + exp(-βz[t]))
                if y[t] > 0
                    μ = exp(βc[t] + Λc[t, 1] * z)
                    lp += log(πt) + logpdf(Gamma(α, μ / α), y[t])
                else
                    lp += log(1 - πt)
                end
            end
            marg += exp(lp) * pdf(Normal(), z) * dz
        end
        ll_quad = log(marg)
        # Gamma positive part is not Gaussian in η^c, so the Laplace carries an
        # O(curvature) error (here ≈0.17 nats) — loose tolerance, but it must track
        # the integral.
        @test ll_lap ≈ ll_quad atol = 0.3
    end

    @testset "fit_delta_gamma_gllvm recovers parameters" begin
        Random.seed!(162)
        p, K, n = 8, 2, 400
        βz_true = 0.5 .* randn(p) .+ 0.4          # occurrence logits (≈ 60% presence)
        βc_true = 0.5 .* randn(p)                 # positive log-mean
        Λc_true = 0.6 .* randn(p, K)
        α_true = 4.0
        Z = randn(K, n)
        ηc = βc_true .+ Λc_true * Z
        π = inv.(1 .+ exp.(-βz_true))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            if rand() < π[t]
                μ = exp(ηc[t, s])
                Y[t, s] = rand(Gamma(α_true, μ / α_true))
            end
        end

        # disp_group=:shared pinned: α_true is a single shared scalar, and fit.α is
        # compared below as a scalar (chained inequality) — :shared keeps this test's
        # pre-2026-09-24 behaviour unchanged now that :species is the public default.
        fit = fit_delta_gamma_gllvm(Y; K = K, disp_group = :shared)
        @test fit isa DeltaGammaFit
        @test isfinite(fit.loglik)
        @test cor(fit.βz, βz_true) > 0.8                                  # occurrence recovers well
        @test cor(vec(fit.Λc * fit.Λc'), vec(Λc_true * Λc_true')) > 0.7   # loadings (Gram)
        # The positive-block intercept, and especially the Gamma shape α, are only
        # weakly recovered here: the Laplace marginal biases dispersion parameters
        # (the motivation for VA — see ROADMAP) and the method-of-moments α₀ is
        # biased low because it cannot net out the latent-variable variance. We
        # therefore check direction/sanity for these, not accuracy.
        @test cor(fit.βc, βc_true) > 0.4                                  # positive log-mean (weak)
        @test 0 < fit.α < 50                                              # shape: positive & finite

        # post-fit surface
        P = predict(fit, Y; type = :response)
        @test size(P) == (p, n)
        @test all(P .>= 0)
        occ = predict(fit, Y; type = :occurrence)
        @test all(0 .< occ .< 1)
        R = residuals(fit, Y; rng = MersenneTwister(1))
        @test size(R) == (p, n)
        @test abs(mean(R)) < 0.3                  # ≈ N(0,1) under correct model
        @test getLV(fit, Y) |> size == (n, K)
        @test isfinite(aic(fit)) && isfinite(bic(fit, n))
    end

    # VA-based Wald standard errors for the two-part Delta-Gamma (objective=:va):
    # extends the VA-SE path (previously GLM-only) to a two-part family.
    @testset "VA-based standard errors (objective=:va)" begin
        Random.seed!(808)
        p, K, n = 4, 1, 150
        βz = 0.5 .* randn(p) .+ 0.3
        βc = 0.3 .* randn(p)
        α  = 3.0
        Λc = 0.4 .* randn(p, K)
        Y = zeros(p, n)
        for s in 1:n
            ηc = βc .+ Λc * randn(K)
            for t in 1:p
                if rand() < inv(1 + exp(-βz[t]))
                    Y[t, s] = rand(Gamma(α, exp(ηc[t]) / α))
                end
            end
        end
        # disp_group=:shared pinned: nterm below assumes a single α term.
        fit = fit_delta_gamma_gllvm(Y; K = K, disp_group = :shared)

        ci_va = confint(fit, Y; method = :wald, objective = :va)
        ci_la = confint(fit, Y; method = :wald, objective = :laplace)
        nterm = 2p + (p * K - div(K * (K - 1), 2)) + 1     # βz + βc + Λc + α
        @test length(ci_va.term) == nterm
        @test ci_va.method == :wald
        for i in eachindex(ci_va.term)
            if isfinite(ci_va.lower[i]) && isfinite(ci_va.upper[i])
                @test ci_va.lower[i] ≤ ci_va.estimate[i] ≤ ci_va.upper[i]
            end
        end
        # Same point estimates as the Laplace path (both evaluated at the MLE θ).
        @test ci_va.term == ci_la.term
        @test ci_va.estimate ≈ ci_la.estimate
    end

    # No-X public surfaces. Marker α is a tag payload (never read by fit_gllvm).
    @testset "no-X public surfaces: fit_gllvm and @formula" begin
        Random.seed!(163)
        p, K, n = 6, 1, 120
        βz = 0.4 .* randn(p) .+ 0.3
        βc = 0.4 .* randn(p)
        Λc = 0.5 .* randn(p, K)
        α = 3.0
        Z = randn(K, n)
        ηc = βc .+ Λc * Z
        π = inv.(1 .+ exp.(-βz))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            if rand() < π[t]
                μ = exp(ηc[t, s])
                Y[t, s] = rand(Gamma(α, μ / α))
            end
        end

        fn = fit_delta_gamma_gllvm(Y; K = K, iterations = 40)
        fu = fit_gllvm(Y; family = DeltaGamma(), K = K, iterations = 40)
        @test fu isa DeltaGammaFit
        @test fu.loglik ≈ fn.loglik atol = 1e-10
        @test fu.α ≈ fn.α atol = 1e-10

        @test fit_gllvm(Y; family = DeltaGamma(9.0), K = K, iterations = 40).α ≈
              fu.α atol = 1e-10
        @test DeltaGamma().α == 1.0

        ff = gllvm(@formula(y ~ 1), Y, (; temp = randn(n));
                   family = DeltaGamma(), K = K, iterations = 40)
        @test ff isa DeltaGammaFit
        @test ff.loglik ≈ fn.loglik atol = 1e-10
    end

    @testset "observed Laplace curvature (2026-08-24 fix)" begin
        # `_tp_pieces(::DeltaGamma)` returns the Fisher weight Wc = alpha, a CONSTANT.
        # TMB uses the observed joint Hessian, which for the Gamma/log positive part is
        # alpha*y/mu. TMB never faces this choice: MakeADFun(..., random=) differentiates
        # the coded nll, so it gets observed curvature structurally. GLLVModels.jl hand-codes
        # the weight and so must choose -- and used the same W for two different roles
        # (mode search AND log-det). Only the log-det needs observed curvature.
        Random.seed!(73)
        p, K, n = 5, 1, 120
        al = 2.0
        bz = [0.8, 0.4, 1.0, 0.2, 0.6]
        bc = log.([2.0, 3.0, 1.5, 2.5, 2.2])
        Lc = 0.25 .* randn(p, K)
        Z = randn(K, n)
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            if rand() < 1 / (1 + exp(-bz[t]))
                mu = exp(bc[t] + (Lc * Z[:, s])[1])
                Y[t, s] = rand(Gamma(al, mu / al))
            end
        end
        @test count(>(0), Y) > 0

        lf = delta_gamma_marginal_loglik_laplace(Y, Lc, bz, bc, al; hessian = :fisher)
        lo = delta_gamma_marginal_loglik_laplace(Y, Lc, bz, bc, al; hessian = :observed)
        @test isfinite(lf) && isfinite(lo)
        @test !isapprox(lo, lf; rtol = 1e-8)      # genuinely different objectives

        # The observed weight is taken from the already-verified Gamma implementation
        # in grouped_dispersion.jl rather than re-derived here.
        for (mu, y) in ((0.5, 4.0), (2.0, 0.3), (7.0, 9.0))
            @test GLLVModels._gamma_grouped_laplace_weight(:observed, GLLVModels.Gamma(al, 1.0),
                                                      mu, mu, y, GLLVModels.LogLink()) ≈ al * y / mu
        end

        # Fits converge under both and neither degenerates.
        fo = fit_delta_gamma_gllvm(Y; K = K)
        ff = fit_delta_gamma_gllvm(Y; K = K, hessian = :fisher)
        @test fo.converged && ff.converged
        @test !isapprox(fo.loglik, ff.loglik; rtol = 1e-8)
        @test sqrt(sum(abs2, fo.Λc)) < 10 && sqrt(sum(abs2, ff.Λc)) < 10

        # Fail loud: the objective's try/catch would otherwise turn a typo into a
        # large penalty and a converged-looking garbage fit.
        @test_throws ArgumentError fit_delta_gamma_gllvm(Y; K = K, hessian = :bogus)
    end

    @testset "the override does NOT touch other two-part families" begin
        # `_tp_observed_Wc` defaults to identity, so every family without a method is
        # bit-for-bit unchanged. DeltaLogNormal is the sharpest check: its positive part
        # is Gaussian in log y, so Wc = 1/sigma^2 is ALREADY the exact Hessian.
        Random.seed!(74)
        p, K, n = 5, 1, 120
        bz = [0.8, 0.4, 1.0, 0.2, 0.6]
        bc = log.([2.0, 3.0, 1.5, 2.5, 2.2])
        Lc = 0.25 .* randn(p, K)
        Z = randn(K, n)
        Yl = zeros(p, n)
        for t in 1:p, s in 1:n
            if rand() < 1 / (1 + exp(-bz[t]))
                Yl[t, s] = exp(bc[t] + (Lc * Z[:, s])[1] + 0.5 * randn())
            end
        end
        a = delta_lognormal_marginal_loglik_laplace(Yl, Lc, bz, bc, 0.5; hessian = :fisher)
        b = delta_lognormal_marginal_loglik_laplace(Yl, Lc, bz, bc, 0.5; hessian = :observed)
        @test a == b        # EXACT equality: the default override is the identity
    end

end
