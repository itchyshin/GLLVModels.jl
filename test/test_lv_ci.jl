using Test
using GLLVModels
using Random
using Statistics
using Distributions
using LinearAlgebra

# Wald CIs for the predictor-informed latent-score trait effects B_lv = Λ·α'
# (confint_lv_effects). Generative model is correctly specified (unit innovation,
# the same z_s ~ N(0,1) the estimator assumes).
@testset "X_lv Wald CIs — confint_lv_effects" begin
    p, n, K, q_lv = 5, 200, 1, 1
    X_lv  = reshape(collect(range(-1.2, 1.2; length = n)), n, q_lv)
    Λ     = reshape([0.6, -0.45, 0.35, 0.25, -0.2], p, K)
    alpha = reshape([0.7], q_lv, K)
    B_true = vec(Λ * alpha')

    function ztot(rng)
        vec(X_lv * alpha) .+ randn(rng, n)
    end

    # Contract assertions shared by every family.
    function _check(ci, fit)
        @test ci.term == ["B_lv[$t,1]" for t in 1:p]
        @test ci.level == 0.95
        @test ci.method == :wald
        @test ci.pd_hessian
        @test ci.estimate ≈ vec(extract_lv_effects(fit)) atol = 1e-10
        @test all(isfinite, ci.se) && all(>(0), ci.se)
        @test all(ci.lower .< ci.estimate .< ci.upper)
        # 95% Wald CI half-widths ≈ 1.96·se
        @test ci.upper .- ci.estimate ≈ 1.959963984540054 .* ci.se rtol = 1e-6
        # point estimate recovers truth to within a few SEs (sanity, not coverage)
        @test maximum(abs.(ci.estimate .- B_true) ./ ci.se) < 6
    end

    @testset "Poisson" begin
        Random.seed!(4101)
        β = log.([6.0, 4.0, 8.0, 5.0, 7.0])
        η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
        Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]
        fit = fit_poisson_gllvm(Y; K = K, X_lv = X_lv, β_init = β, λ_init = Λ,
                                alpha_lv_init = alpha, iterations = 200, g_tol = 1e-6)
        _check(confint_lv_effects(fit, Y, X_lv), fit)
    end

    @testset "Binomial logit and probit" begin
        for (link, N0) in ((LogitLink(), 40), (ProbitLink(), 40))
            Random.seed!(4202)
            β = [-0.6, -0.25, 0.05, 0.35, 0.65]
            η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
            μ = clamp.(GLLVModels.linkinv.(Ref(link), η), 1e-4, 1 - 1e-4)
            N = fill(N0, p, n)
            Y = [rand(Binomial(N[t, s], μ[t, s])) for t in 1:p, s in 1:n]
            fit = fit_binomial_gllvm(Y; K = K, N = N, link = link, X_lv = X_lv,
                                     β_init = β, λ_init = Λ, alpha_lv_init = alpha,
                                     iterations = 150, g_tol = 1e-6)
            _check(confint_lv_effects(fit, Y, X_lv; N = N), fit)
        end
    end

    @testset "NB2" begin
        Random.seed!(4303)
        β = log.([6.0, 4.0, 8.0, 5.0, 7.0]); r = 10.0
        η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
        Y = [rand(NegativeBinomial(r, r / (r + exp(η[t, s])))) for t in 1:p, s in 1:n]
        fit = fit_nb_gllvm(Y; K = K, X_lv = X_lv, β_init = β, λ_init = Λ,
                           alpha_lv_init = alpha, r_init = r, iterations = 200, g_tol = 1e-6)
        _check(confint_lv_effects(fit, Y, X_lv), fit)
    end

    @testset "Gamma" begin
        Random.seed!(4404)
        β = log.([2.0, 1.5, 3.0, 2.5, 1.8]); a = 6.0
        η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
        Y = [rand(Gamma(a, exp(η[t, s]) / a)) for t in 1:p, s in 1:n]
        fit = fit_gamma_gllvm(Y; K = K, X_lv = X_lv, β_init = β, λ_init = Λ,
                              alpha_lv_init = alpha, α_init = a, iterations = 200, g_tol = 1e-6)
        _check(confint_lv_effects(fit, Y, X_lv), fit)
    end

    @testset "Beta" begin
        Random.seed!(4505)
        β = [0.3, -0.5, 0.6, -0.2, 0.4]; φ = 15.0
        η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
        μ = 1.0 ./ (1.0 .+ exp.(-η))
        Y = [rand(Beta(μ[t, s] * φ, (1 - μ[t, s]) * φ)) for t in 1:p, s in 1:n]
        fit = fit_beta_gllvm(Y; K = K, X_lv = X_lv, β_init = β, λ_init = Λ,
                             alpha_lv_init = alpha, φ_init = φ, iterations = 200, g_tol = 1e-6)
        _check(confint_lv_effects(fit, Y, X_lv), fit)
    end

    @testset "Gaussian (closed-form, exact ForwardDiff Hessian)" begin
        Random.seed!(4606)
        zt = vec(X_lv * alpha) .+ randn(n)
        Yg = Λ * reshape(zt, 1, n) .+ 0.3 .* randn(p, n)   # centred: no β; B_lv = Λ·α
        fit = fit_gaussian_gllvm(Yg; K = K, X_lv = X_lv, iterations = 300)
        cw = confint_lv_effects(fit, Yg, X_lv)
        _check(cw, fit)
        ct = confint_lv_effects(fit, Yg, X_lv; method = :wald_t_unit)
        df = max(n - K - 1, 1)
        tcrit = quantile(TDist(df), 0.975)
        @test ct.method == :wald_t_unit
        @test ct.term == cw.term
        @test ct.level == cw.level
        @test ct.pd_hessian == cw.pd_hessian
        @test ct.estimate ≈ cw.estimate atol = 1e-10
        @test ct.se ≈ cw.se atol = 1e-10
        @test ct.upper .- ct.estimate ≈ tcrit .* ct.se rtol = 1e-6
        @test all((ct.upper .- ct.lower) .> (cw.upper .- cw.lower))
        @test_throws ArgumentError confint_lv_effects(fit_gaussian_gllvm(Yg; K = K), Yg, X_lv)
    end

    @testset "K = 2 (rotation-invariant B_lv)" begin
        Random.seed!(909)
        p2, n2 = 6, 240
        X2 = reshape(collect(range(-1.2, 1.2; length = n2)), n2, 1)
        Lam2 = [0.7 0.3; -0.5 0.4; 0.4 -0.3; 0.3 0.5; -0.25 0.2; 0.2 -0.4]
        a2 = reshape([0.7, -0.5], 1, 2)
        B2 = vec(Lam2 * a2')
        beta2 = log.([6.0, 4.0, 8.0, 5.0, 7.0, 4.5])
        Z = zeros(2, n2)
        for j in 1:n2
            Z[:, j] = vec(X2[j, :] .* vec(a2)) .+ randn(2)
        end
        eta = beta2 .+ Lam2 * Z
        Y2 = [rand(Poisson(exp(eta[t, j]))) for t in 1:p2, j in 1:n2]
        fit = fit_poisson_gllvm(Y2; K = 2, X_lv = X2, iterations = 400, g_tol = 1e-6)
        ci = confint_lv_effects(fit, Y2, X2)
        @test ci.term == ["B_lv[$t,1]" for t in 1:p2]
        @test ci.pd_hessian
        @test ci.estimate ≈ vec(extract_lv_effects(fit)) atol = 1e-10
        @test all(ci.lower .< ci.estimate .< ci.upper)
        @test all(isfinite, ci.se) && all(>(0), ci.se)
        @test cor(ci.estimate, B2) > 0.9   # recovers the rotation-stable truth
    end

    @testset "bootstrap method (Poisson)" begin
        Random.seed!(5151)
        β = log.([6.0, 4.0, 8.0, 5.0, 7.0])
        η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
        Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]
        fit = fit_poisson_gllvm(Y; K = K, X_lv = X_lv, β_init = β, λ_init = Λ,
                                alpha_lv_init = alpha, iterations = 150, g_tol = 1e-6)
        cb = confint_lv_effects(fit, Y, X_lv; method = :bootstrap, n_boot = 50, seed = 1)
        @test cb.method == :bootstrap
        @test cb.term == ["B_lv[$t,1]" for t in 1:p]
        @test cb.n_converged >= 35
        @test cb.estimate ≈ vec(extract_lv_effects(fit)) atol = 1e-10
        @test all(isfinite, cb.lower) && all(isfinite, cb.upper)
        @test all(cb.lower .< cb.upper)
        @test all(cb.lower .<= cb.estimate .<= cb.upper)   # point est near boot centre
        # :profile is now implemented (its own testset below); only an unknown
        # method should throw.
        @test_throws ArgumentError confint_lv_effects(fit, Y, X_lv; method = :nope)
    end

    @testset "bootstrap covers all families (+ Gaussian K=2 guards the transpose)" begin
        # Gaussian K=2, q_lv=1: the score mean is X_lv*alpha (q_lv x K = 1 x 2). A
        # transpose (alpha') would DimensionMismatch on every replicate -> n_converged=0.
        Random.seed!(7777)
        Lam2 = [0.6 0.3; -0.45 0.35; 0.4 -0.3; 0.3 0.4; -0.25 0.2]
        a2 = reshape([0.7, -0.4], 1, 2)
        Zg = zeros(2, n)
        for j in 1:n
            Zg[:, j] = vec(X_lv[j, :] .* vec(a2)) .+ randn(2)
        end
        Yg2 = Lam2 * Zg .+ 0.3 .* randn(p, n)
        fg2 = fit_gaussian_gllvm(Yg2; K = 2, X_lv = X_lv, iterations = 300)
        cbg = confint_lv_effects(fg2, Yg2, X_lv; method = :bootstrap, n_boot = 40, seed = 3)
        @test cbg.method == :bootstrap
        @test cbg.n_converged >= 25                    # == 0 under the transpose bug
        @test all(isfinite, cbg.lower) && all(isfinite, cbg.upper)
        @test all(cbg.lower .< cbg.upper)
        cbg_limited = confint_lv_effects(fg2, Yg2, X_lv; method = :bootstrap,
                                         n_boot = 10, seed = 33,
                                         bootstrap_iterations = 5)
        @test cbg_limited.method == :bootstrap
        @test cbg_limited.term == cbg.term
        @test cbg_limited.n_converged <= 10
        @test_throws ArgumentError confint_lv_effects(fg2, Yg2, X_lv;
                                                      method = :bootstrap,
                                                      n_boot = 2,
                                                      bootstrap_iterations = 0)

        boot_ok(cb) = cb.method == :bootstrap && cb.n_converged >= 10 &&
                      all(isfinite, cb.lower) && all(cb.lower .< cb.upper)
        @testset "binomial" begin
            Random.seed!(11); β = [-0.6, -0.25, 0.05, 0.35, 0.65]
            η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
            μ = clamp.(GLLVModels.linkinv.(Ref(LogitLink()), η), 1e-4, 1 - 1e-4); N = fill(40, p, n)
            Y = [rand(Binomial(N[t, s], μ[t, s])) for t in 1:p, s in 1:n]
            f = fit_binomial_gllvm(Y; K = 1, N = N, link = LogitLink(), X_lv = X_lv,
                                   β_init = β, λ_init = Λ, alpha_lv_init = alpha, iterations = 120)
            @test boot_ok(confint_lv_effects(f, Y, X_lv; N = N, method = :bootstrap, n_boot = 20, seed = 5))
        end
        @testset "nb2" begin
            Random.seed!(12); β = log.([6.0, 4, 8, 5, 7])
            η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
            Y = [rand(NegativeBinomial(10.0, 10.0 / (10.0 + exp(η[t, s])))) for t in 1:p, s in 1:n]
            f = fit_nb_gllvm(Y; K = 1, X_lv = X_lv, β_init = β, λ_init = Λ,
                             alpha_lv_init = alpha, r_init = 10.0, iterations = 150)
            @test boot_ok(confint_lv_effects(f, Y, X_lv; method = :bootstrap, n_boot = 20, seed = 6))
        end
        @testset "gamma" begin
            Random.seed!(13); β = log.([2.0, 1.5, 3, 2.5, 1.8])
            η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
            Y = [rand(Gamma(6.0, exp(η[t, s]) / 6.0)) for t in 1:p, s in 1:n]
            f = fit_gamma_gllvm(Y; K = 1, X_lv = X_lv, β_init = β, λ_init = Λ,
                                alpha_lv_init = alpha, α_init = 6.0, iterations = 150)
            @test boot_ok(confint_lv_effects(f, Y, X_lv; method = :bootstrap, n_boot = 20, seed = 7))
        end
        @testset "beta" begin
            Random.seed!(14); β = [0.3, -0.5, 0.6, -0.2, 0.4]
            η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
            μ = 1.0 ./ (1.0 .+ exp.(-η))
            Y = [rand(Beta(μ[t, s] * 15, (1 - μ[t, s]) * 15)) for t in 1:p, s in 1:n]
            f = fit_beta_gllvm(Y; K = 1, X_lv = X_lv, β_init = β, λ_init = Λ,
                               alpha_lv_init = alpha, φ_init = 15.0, iterations = 150)
            @test boot_ok(confint_lv_effects(f, Y, X_lv; method = :bootstrap, n_boot = 20, seed = 8))
        end
    end

    @testset "q_lv = 2 (multi-predictor latent score)" begin
        Random.seed!(2468)
        p2, n2, ql = 5, 220, 2
        xa = collect(range(-1.2, 1.2; length = n2))
        xb = collect(range(-1.0, 1.0; length = n2)) .^ 2 .- 0.4   # quadratic ⟂ linear
        X2 = hcat(xa, xb)                                          # n × 2
        Λ2 = reshape([0.6, -0.45, 0.35, 0.25, -0.2], p2, 1)
        a2 = reshape([0.7, -0.4], ql, 1)                          # q_lv × K = 2 × 1
        B2 = Λ2 * a2'                                             # p × 2
        β2 = log.([6.0, 4, 8, 5, 7])
        zt2 = vec(X2 * a2) .+ randn(n2)                          # X2(n×2)·a2(2×1) = n-score mean
        η = β2 .+ Λ2 * reshape(zt2, 1, n2)
        Y2 = [rand(Poisson(exp(η[t, s]))) for t in 1:p2, s in 1:n2]
        fit = fit_poisson_gllvm(Y2; K = 1, X_lv = X2, β_init = β2, λ_init = Λ2,
                                alpha_lv_init = a2, iterations = 200, g_tol = 1e-6)
        # Wald — 10 entries of vec(B_lv) (column-major over p×q_lv)
        ci = confint_lv_effects(fit, Y2, X2)
        @test ci.term == ["B_lv[$t,$c]" for c in 1:ql for t in 1:p2]
        @test ci.estimate ≈ vec(extract_lv_effects(fit)) atol = 1e-10
        @test ci.pd_hessian
        @test all(ci.lower .< ci.estimate .< ci.upper)
        @test cor(ci.estimate, vec(B2)) > 0.9
        # bootstrap — exercises the q_lv>1 vec→matrix paths
        cb = confint_lv_effects(fit, Y2, X2; method = :bootstrap, n_boot = 25, seed = 9)
        @test cb.term == ci.term
        @test cb.n_converged >= 12 && all(cb.lower .< cb.upper)
        # bridge round-trip: CI fields reshape to p × q_lv matching lv_effects
        br = bridge_fit(; y = Float64.(Y2), family = "poisson", d = 1, X_lv = X2,
                        options = Dict("ci_method" => "wald"))
        @test size(br.lv_effects_lower) == (p2, ql)
        @test all(br.lv_effects_lower .< br.lv_effects .< br.lv_effects_upper)
    end

    @testset "bridge ci == native, level, gate rejection" begin
        Random.seed!(3690)
        β = log.([6.0, 4, 8, 5, 7])
        η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
        Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]
        # identical defaults on both sides so the bridge's internal fit and this
        # native fit are the same deterministic optimum.
        fit = fit_poisson_gllvm(Y; K = K, X_lv = X_lv)
        ci = confint_lv_effects(fit, Y, X_lv)
        br = bridge_fit(; y = Float64.(Y), family = "poisson", d = 1, X_lv = X_lv,
                        options = Dict("ci_method" => "wald"))
        @test vec(br.lv_effects_lower) ≈ ci.lower atol = 1e-6
        @test vec(br.lv_effects_upper) ≈ ci.upper atol = 1e-6
        @test vec(br.lv_effects_se) ≈ ci.se atol = 1e-6
        @test br.lv_effects_ci_method == "wald" && br.lv_effects_ci_pd
        # non-default level: 90% interval is strictly narrower than 95%
        ci90 = confint_lv_effects(fit, Y, X_lv; level = 0.90)
        @test ci90.level == 0.90
        @test all((ci90.upper .- ci90.lower) .< (ci.upper .- ci.lower))
        # bridge admits only ci_method="wald" for X_lv
        @test_throws ArgumentError bridge_fit(; y = Float64.(Y), family = "poisson",
            d = 1, X_lv = X_lv, options = Dict("ci_method" => "profile"))
        @test_throws ArgumentError bridge_fit(; y = Float64.(Y), family = "poisson",
            d = 1, X_lv = X_lv, options = Dict("ci_method" => "bootstrap"))
    end

    @testset "argument guards" begin
        Random.seed!(4606)
        β = log.([6.0, 4.0, 8.0, 5.0, 7.0])
        η = β .+ Λ * reshape(ztot(Random.default_rng()), 1, n)
        Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]
        xfit = fit_poisson_gllvm(Y; K = K, X_lv = X_lv, β_init = β, λ_init = Λ,
                                 alpha_lv_init = alpha, iterations = 150, g_tol = 1e-6)
        plain = fit_poisson_gllvm(Y; K = K, iterations = 150)
        @test_throws ArgumentError confint_lv_effects(plain, Y, X_lv)            # no α_lv
        @test_throws ArgumentError confint_lv_effects(xfit, Y, X_lv; level = 1.0) # bad level
        @test_throws ArgumentError confint_lv_effects(xfit, Y, X_lv; method = :wald_t_unit) # Gaussian-only comparator
        @test_throws ArgumentError confint_lv_effects(xfit, Y, X_lv[1:(n - 1), :]) # row mismatch
    end

    @testset "profile method (Gaussian)" begin
        # Gaussian objective is AD-friendly → fast LBFGS constrained refits.
        Random.seed!(9090)
        zg = ztot(Random.default_rng())
        Yg = Λ * reshape(zg, 1, n) .+ 0.5 .* randn(p, n)
        fg = fit_gaussian_gllvm(Yg; K = K, X_lv = X_lv, iterations = 300)
        cwald = confint_lv_effects(fg, Yg, X_lv)                    # wald
        cprof = confint_lv_effects(fg, Yg, X_lv; method = :profile)
        @test cprof.method == :profile
        @test cprof.term == ["B_lv[$t,1]" for t in 1:p]
        @test cprof.level == 0.95
        @test cprof.estimate ≈ cwald.estimate atol = 1e-10         # same point estimate
        @test all(isnan, cprof.se)                                 # profile: no single se
        @test all(isfinite, cprof.lower) && all(isfinite, cprof.upper)
        @test all(cprof.lower .< cprof.estimate .< cprof.upper)    # brackets the MLE
        # profile ≈ wald widths in the well-identified interior (tracks mild
        # asymmetry rather than being forced symmetric).
        @test maximum(abs.((cprof.upper .- cprof.lower) .- (cwald.upper .- cwald.lower)) ./
                      (cwald.upper .- cwald.lower)) < 0.25

        selected = [2, 4]
        csel = confint_lv_effects(fg, Yg, X_lv; method = :profile,
                                  profile_indices = selected,
                                  profile_iterations = 1000,
                                  profile_g_tol = 1e-8,
                                  profile_max_expand = 40,
                                  profile_max_bisect = 40)
        @test csel.method == :profile
        @test csel.term == cprof.term[selected]
        @test csel.estimate ≈ cprof.estimate[selected] atol = 1e-10
        @test csel.lower ≈ cprof.lower[selected] atol = 1e-6
        @test csel.upper ≈ cprof.upper[selected] atol = 1e-6
        @test all(isnan, csel.se)
        @test_throws ArgumentError confint_lv_effects(fg, Yg, X_lv;
                                                      method = :profile,
                                                      profile_indices = Int[])
        @test_throws ArgumentError confint_lv_effects(fg, Yg, X_lv;
                                                      method = :profile,
                                                      profile_indices = [2, 2])
        @test_throws ArgumentError confint_lv_effects(fg, Yg, X_lv;
                                                      method = :profile,
                                                      profile_indices = [0])
        @test_throws ArgumentError confint_lv_effects(fg, Yg, X_lv;
                                                      method = :wald,
                                                      profile_indices = selected)
        @test_throws ArgumentError confint_lv_effects(fg, Yg, X_lv;
                                                      method = :profile,
                                                      profile_iterations = 0)
        @test_throws ArgumentError confint_lv_effects(fg, Yg, X_lv;
                                                      method = :profile,
                                                      profile_g_tol = Inf)
    end

    @testset "profile method (Poisson selected-entry canary)" begin
        # Gate 1 canary for ordinary non-Gaussian B_lv profile-LR: one selected
        # Poisson entry, finite endpoints, and known DGP truth inside the interval.
        # This is route evidence, not a coverage-calibration claim.
        Random.seed!(20260702)
        pc, nc, Kc = 2, 45, 1
        Xc = reshape(collect(range(-1.0, 1.0; length = nc)), nc, 1)
        Lc = reshape([0.55, -0.42], pc, Kc)
        ac = reshape([0.65], 1, Kc)
        Bc = vec(Lc * ac')
        βc = log.([6.0, 4.5])
        zc = vec(Xc * ac) .+ randn(nc)
        ηc = βc .+ Lc * reshape(zc, 1, nc)
        Yc = [rand(Poisson(exp(ηc[t, s]))) for t in 1:pc, s in 1:nc]
        fc = fit_poisson_gllvm(Yc; K = Kc, X_lv = Xc, β_init = βc,
                               λ_init = Lc, alpha_lv_init = ac,
                               iterations = 120, g_tol = 1e-6)
        idx = [1]
        cpr = confint_lv_effects(fc, Yc, Xc; method = :profile,
                                 profile_indices = idx,
                                 profile_iterations = 800,
                                 profile_max_expand = 20,
                                 profile_max_bisect = 24)
        @test cpr.method == :profile
        @test cpr.term == ["B_lv[1,1]"]
        @test cpr.estimate ≈ vec(extract_lv_effects(fc))[idx] atol = 1e-10
        @test all(isnan, cpr.se)
        @test all(isfinite, cpr.lower) && all(isfinite, cpr.upper)
        @test all(cpr.lower .< cpr.estimate .< cpr.upper)
        @test cpr.lower[1] <= Bc[idx[1]] <= cpr.upper[1]
    end

    @testset "profile method (Binomial logit selected-entry canary)" begin
        # Second ordinary non-Gaussian Gate 1 canary: Binomial logit shares the
        # selected-entry profile contract while also threading trial counts.
        Random.seed!(20260703)
        pc, nc, Kc = 2, 60, 1
        Xc = reshape(collect(range(-1.0, 1.0; length = nc)), nc, 1)
        Lc = reshape([0.45, -0.35], pc, Kc)
        ac = reshape([0.55], 1, Kc)
        Bc = vec(Lc * ac')
        βc = [-0.35, 0.25]
        Nc = fill(30, pc, nc)
        zc = vec(Xc * ac) .+ randn(nc)
        ηc = βc .+ Lc * reshape(zc, 1, nc)
        μc = 1.0 ./ (1.0 .+ exp.(-ηc))
        Yc = [rand(Binomial(Nc[t, s], clamp(μc[t, s], 1e-4, 1 - 1e-4)))
              for t in 1:pc, s in 1:nc]
        fc = fit_binomial_gllvm(Yc; K = Kc, N = Nc, link = LogitLink(),
                                X_lv = Xc, β_init = βc, λ_init = Lc,
                                alpha_lv_init = ac, iterations = 160,
                                g_tol = 1e-6)
        idx = [1]
        cpr = confint_lv_effects(fc, Yc, Xc; N = Nc, method = :profile,
                                 profile_indices = idx,
                                 profile_iterations = 900,
                                 profile_max_expand = 20,
                                 profile_max_bisect = 24)
        @test cpr.method == :profile
        @test cpr.term == ["B_lv[1,1]"]
        @test cpr.estimate ≈ vec(extract_lv_effects(fc))[idx] atol = 1e-10
        @test all(isnan, cpr.se)
        @test all(isfinite, cpr.lower) && all(isfinite, cpr.upper)
        @test all(cpr.lower .< cpr.estimate .< cpr.upper)
        @test cpr.lower[1] <= Bc[idx[1]] <= cpr.upper[1]
    end

    @testset "profile method (NB2 selected-entry canary)" begin
        # Third ordinary non-Gaussian Gate 1 canary: NB2 exercises the shared
        # dispersion path. The cell is small enough for local tests but keeps
        # fitted r away from the Poisson boundary.
        Random.seed!(20260711)
        pc, nc, Kc = 4, 45, 1
        Xc = reshape(collect(range(-1.0, 1.0; length = nc)), nc, 1)
        Lc = reshape([(-1)^i * 0.2 * (1 + 0.08i) for i in 1:pc], pc, Kc)
        ac = reshape([0.35], 1, Kc)
        Bc = vec(Lc * ac')
        βc = log.([8.0, 6.0, 7.0, 5.0])
        rc = 1.5
        zc = vec(Xc * ac) .+ randn(nc)
        ηc = βc .+ Lc * reshape(zc, 1, nc)
        Yc = [rand(NegativeBinomial(rc, rc / (rc + exp(ηc[t, s]))))
              for t in 1:pc, s in 1:nc]
        fc = fit_nb_gllvm(Yc; K = Kc, X_lv = Xc, β_init = βc,
                          λ_init = Lc, alpha_lv_init = ac, r_init = rc,
                          iterations = 160, g_tol = 1e-6)
        @test fc.converged
        @test 0.25 < fc.r < 10.0
        idx = [1]
        cpr = confint_lv_effects(fc, Yc, Xc; method = :profile,
                                 profile_indices = idx,
                                 profile_iterations = 160,
                                 profile_max_expand = 10,
                                 profile_max_bisect = 10)
        @test cpr.method == :profile
        @test cpr.term == ["B_lv[1,1]"]
        @test cpr.estimate ≈ vec(extract_lv_effects(fc))[idx] atol = 1e-10
        @test all(isnan, cpr.se)
        @test all(isfinite, cpr.lower) && all(isfinite, cpr.upper)
        @test all(cpr.lower .< cpr.estimate .< cpr.upper)
        @test cpr.lower[1] <= Bc[idx[1]] <= cpr.upper[1]
    end

    @testset "profile method (Gamma selected-entry canary)" begin
        # Fourth ordinary non-Gaussian Gate 1 canary: Gamma exercises the
        # positive continuous one-part profile route with a fitted shape away
        # from a near-deterministic limit.
        Random.seed!(20260722)
        pc, nc, Kc = 4, 45, 1
        Xc = reshape(collect(range(-1.0, 1.0; length = nc)), nc, 1)
        Lc = reshape([(-1)^i * 0.20 * (1 + 0.08i) for i in 1:pc], pc, Kc)
        ac = reshape([0.35], 1, Kc)
        Bc = vec(Lc * ac')
        βc = log.([3.0, 2.4, 2.7, 2.1])
        αc = 2.5
        zc = vec(Xc * ac) .+ randn(nc)
        ηc = βc .+ Lc * reshape(zc, 1, nc)
        Yc = [rand(Gamma(αc, exp(ηc[t, s]) / αc))
              for t in 1:pc, s in 1:nc]
        fc = fit_gamma_gllvm(Yc; K = Kc, X_lv = Xc, β_init = βc,
                             λ_init = Lc, alpha_lv_init = ac, α_init = αc,
                             iterations = 160, g_tol = 1e-6)
        @test fc.converged
        @test 0.5 < fc.α < 10.0
        idx = [1]
        cpr = confint_lv_effects(fc, Yc, Xc; method = :profile,
                                 profile_indices = idx,
                                 profile_iterations = 160,
                                 profile_max_expand = 10,
                                 profile_max_bisect = 10)
        @test cpr.method == :profile
        @test cpr.term == ["B_lv[1,1]"]
        @test cpr.estimate ≈ vec(extract_lv_effects(fc))[idx] atol = 1e-10
        @test all(isnan, cpr.se)
        @test all(isfinite, cpr.lower) && all(isfinite, cpr.upper)
        @test all(cpr.lower .< cpr.estimate .< cpr.upper)
        @test cpr.lower[1] <= Bc[idx[1]] <= cpr.upper[1]
    end

    @testset "profile method (Beta selected-entry canary)" begin
        # Fifth ordinary non-Gaussian Gate 1 canary: Beta exercises the
        # bounded-continuous one-part profile route with fitted precision away
        # from both degenerate and nearly uniform limits.
        Random.seed!(20260734)
        pc, nc, Kc = 5, 50, 1
        Xc = reshape(collect(range(-1.0, 1.0; length = nc)), nc, 1)
        Lc = reshape([(-1)^i * 0.16 * (1 + 0.08i) for i in 1:pc], pc, Kc)
        ac = reshape([0.35], 1, Kc)
        Bc = vec(Lc * ac')
        βc = [0.30, -0.20, 0.25, -0.15, 0.05]
        φc = 12.0
        zc = vec(Xc * ac) .+ randn(nc)
        ηc = βc .+ Lc * reshape(zc, 1, nc)
        μc = clamp.(1.0 ./ (1.0 .+ exp.(-ηc)), 1e-5, 1 - 1e-5)
        Yc = [rand(Beta(μc[t, s] * φc, (1 - μc[t, s]) * φc))
              for t in 1:pc, s in 1:nc]
        fc = fit_beta_gllvm(Yc; K = Kc, X_lv = Xc, β_init = βc,
                            λ_init = Lc, alpha_lv_init = ac, φ_init = φc,
                            iterations = 180, g_tol = 1e-6)
        @test fc.converged
        @test 2.0 < fc.φ < 50.0
        idx = [1]
        cpr = confint_lv_effects(fc, Yc, Xc; method = :profile,
                                 profile_indices = idx,
                                 profile_iterations = 180,
                                 profile_max_expand = 10,
                                 profile_max_bisect = 10)
        @test cpr.method == :profile
        @test cpr.term == ["B_lv[1,1]"]
        @test cpr.estimate ≈ vec(extract_lv_effects(fc))[idx] atol = 1e-10
        @test all(isnan, cpr.se)
        @test all(isfinite, cpr.lower) && all(isfinite, cpr.upper)
        @test all(cpr.lower .< cpr.estimate .< cpr.upper)
        @test cpr.lower[1] <= Bc[idx[1]] <= cpr.upper[1]
    end

    @testset "profile method (Ordinal logit selected-entry canary)" begin
        # Sixth ordinary non-Gaussian Gate 1 canary: the Julia-side shared-
        # cutpoint ordinal route has no intercept, so X_lv enters only through
        # the latent-score mean and the estimand remains B_lv = Lambda * alpha'.
        Random.seed!(20260744)
        pc, nc, Kc, Cc = 2, 60, 1, 4
        Xc = reshape(collect(range(-1.0, 1.0; length = nc)), nc, 1)
        Lc = reshape([0.50, -0.38], pc, Kc)
        ac = reshape([0.55], 1, Kc)
        Bc = vec(Lc * ac')
        τc = [-1.1, 0.05, 1.25]
        zc = vec(Xc * ac) .+ randn(nc)
        ηc = Lc * reshape(zc, 1, nc)
        Yc = Matrix{Int}(undef, pc, nc)
        for s in 1:nc, t in 1:pc
            probs = [GLLVModels._ord_prob(c, ηc[t, s], τc, LogitLink()) for c in 1:Cc]
            Yc[t, s] = rand(Categorical(probs ./ sum(probs)))
        end
        @test all([all(vec(sum(Yc .== c; dims = 2)) .> 0) for c in 1:Cc])

        fc = fit_ordinal_gllvm(Yc; K = Kc, X_lv = Xc, λ_init = Lc,
                               alpha_lv_init = ac, iterations = 500,
                               g_tol = 1e-6)
        @test fc.converged
        @test fc.alpha_lv !== nothing
        @test issorted(fc.τ)
        @test extract_lv_effects(fc) == fc.Λ * fc.alpha_lv'
        cwald = confint_lv_effects(fc, Yc, Xc)
        @test cwald.method == :wald
        @test cwald.term == ["B_lv[1,1]", "B_lv[2,1]"]
        @test all(isfinite, cwald.se)
        @test all(cwald.lower .< cwald.estimate .< cwald.upper)

        idx = [1]
        cpr = confint_lv_effects(fc, Yc, Xc; method = :profile,
                                 profile_indices = idx,
                                 profile_iterations = 220,
                                 profile_max_expand = 8,
                                 profile_max_bisect = 8)
        @test cpr.method == :profile
        @test cpr.term == ["B_lv[1,1]"]
        @test cpr.estimate ≈ vec(extract_lv_effects(fc))[idx] atol = 1e-10
        @test all(isnan, cpr.se)
        @test all(isfinite, cpr.lower) && all(isfinite, cpr.upper)
        @test all(cpr.lower .< cpr.estimate .< cpr.upper)
        @test cpr.lower[1] <= Bc[idx[1]] <= cpr.upper[1]
    end

    # GLM profile uses derivative-free NelderMead over the Laplace marginal and is
    # expensive (~1 min even for a tiny problem), so it is opt-in to keep CI fast.
    # Set GLLVM_SLOW_TESTS=true to run it. The fast Gaussian test above guards the
    # algorithm; this guards the GLM (Poisson) code path end to end.
    if get(ENV, "GLLVM_SLOW_TESTS", "false") == "true"
        @testset "profile method (Poisson, opt-in slow)" begin
            Random.seed!(4101)
            ps, ns = 3, 80
            Xs = reshape(collect(range(-1.2, 1.2; length = ns)), ns, 1)
            Ls = reshape([0.6, -0.45, 0.35], ps, 1)
            as = reshape([0.7], 1, 1)
            zs = vec(Xs * as) .+ randn(ns)
            βs = log.([6.0, 4.0, 8.0])
            ηs = βs .+ Ls * reshape(zs, 1, ns)
            Ys = [rand(Poisson(exp(ηs[t, s]))) for t in 1:ps, s in 1:ns]
            fs = fit_poisson_gllvm(Ys; K = 1, X_lv = Xs, β_init = βs, λ_init = Ls,
                                   alpha_lv_init = as, iterations = 200, g_tol = 1e-6)
            cpr = confint_lv_effects(fs, Ys, Xs; method = :profile)
            @test cpr.method == :profile
            @test all(isfinite, cpr.lower) && all(isfinite, cpr.upper)
            @test all(cpr.lower .< cpr.estimate .< cpr.upper)
        end
    end
end
