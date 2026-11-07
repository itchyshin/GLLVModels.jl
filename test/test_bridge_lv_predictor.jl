using Test
using GLLVModels
using Random
using Statistics
using Distributions

@testset "bridge predictor-informed latent-score X_lv" begin
    function _binary_xlv_fixture(link)
        Random.seed!(123)
        p, n, K, q_lv = 5, 140, 1, 1
        X_lv = reshape(collect(range(-1.2, 1.2; length = n)), n, q_lv)
        β = [-0.6, -0.25, 0.05, 0.35, 0.65]
        Λ = reshape([0.85, -0.65, 0.5, 0.35, -0.25], p, K)
        alpha_lv = reshape([0.7], q_lv, K)
        z_innov = randn(n)
        z_total = vec(X_lv * alpha_lv) .+ z_innov
        η = β .+ Λ * reshape(z_total, 1, n)
        μ = clamp.(GLLVModels.linkinv.(Ref(link), η), 1e-4, 1 - 1e-4)
        N = fill(40, p, n)
        Y = [rand(Binomial(N[t, s], μ[t, s])) for t in 1:p, s in 1:n]
        return Y, N, X_lv, β, Λ, alpha_lv
    end

    @testset "Gaussian X_lv bridge matches native centered oracle" begin
        Random.seed!(7311)
        p, n, K, q_lv = 4, 90, 1, 1
        X_lv = reshape(collect(range(-1.4, 1.4; length = n)), n, q_lv)
        α_trait = [0.7, -0.4, 0.2, 0.9]
        Λ_true = reshape([0.8, 0.45, -0.35, 0.25], p, K)
        alpha_lv_true = reshape([1.15], q_lv, K)
        Z_mean = X_lv * alpha_lv_true
        Z_innov = 0.25 .* randn(K, n)
        Y = α_trait .+ Λ_true * (Z_mean' .+ Z_innov) .+ 0.12 .* randn(p, n)

        br = bridge_fit(; y = Y, family = "gaussian", d = K, X_lv = X_lv)
        alpha_hat = vec(mean(Y; dims = 2))
        Yc = Y .- alpha_hat
        oracle = fit_gaussian_gllvm(Yc; K = K, X_lv = X_lv)

        @test br.model == "gaussian_xlv_rr"
        @test br.family == "gaussian"
        @test br.alpha ≈ alpha_hat atol = 0
        @test br.loadings ≈ getLoadings(oracle; rotate = true) atol = 1e-10
        @test br.scores ≈ getLV(oracle, Yc; X_lv = X_lv, component = :total,
                                rotate = true) atol = 1e-10
        @test br.scores_mean ≈ getLV(oracle, Yc; X_lv = X_lv, component = :mean,
                                     rotate = true) atol = 1e-10
        @test br.scores_innovation ≈ getLV(oracle, Yc; X_lv = X_lv,
                                           component = :innovation,
                                           rotate = true) atol = 1e-10
        @test br.lv_effects ≈ extract_lv_effects(oracle) atol = 1e-10
        @test br.alpha_lv ≈ extract_lv_effects(oracle; type = :axis_effect) atol = 1e-10
        @test br.sigma_eps ≈ oracle.pars.σ_eps atol = 0
        @test br.df == p + GLLVModels._nparams(oracle)
        @test occursin("predictor-informed latent-score", br.note)
        @test occursin("Confidence intervals", br.note)
    end

    @testset "Binomial X_lv packed objective matches offset Laplace core" begin
        p, n, K, q_lv = 3, 9, 1, 1
        X_lv = reshape(collect(range(-0.8, 0.8; length = n)), n, q_lv)
        β = [-0.25, 0.15, 0.4]
        Λ = reshape([0.55, -0.35, 0.2], p, K)
        alpha_lv = reshape([0.7], q_lv, K)
        N = fill(12, p, n)
        Y = [mod(t + 2s, 8) for t in 1:p, s in 1:n]
        params = vcat(β, vec(alpha_lv), GLLVModels.pack_lambda(Λ))
        for link in (LogitLink(), ProbitLink(), CLogLogLink())
            lv_offset = GLLVModels._lv_mean_eta(Λ, X_lv, alpha_lv)
            nll_xlv = GLLVModels.binomial_lv_nll_packed(
                params, Y, N, p, K, link; X_lv = X_lv, q_lv = q_lv)
            nll_offset = -GLLVModels.binomial_marginal_loglik_laplace(
                Y, N, Λ, β, link; offset = lv_offset)
            @test nll_xlv ≈ nll_offset atol = 1e-10
        end
    end

    @testset "Binomial X_lv native and bridge routes cover logit probit cloglog" begin
        link_cases = (
            ("binomial", LogitLink()),
            ("binomial_probit", ProbitLink()),
            ("binomial_cloglog", CLogLogLink()),
        )
        for (family_key, link) in link_cases
            Y, N, X_lv, β_true, Λ_true, alpha_true = _binary_xlv_fixture(link)
            fit = fit_binomial_gllvm(
                Y; K = 1, N = N, link = link, X_lv = X_lv,
                β_init = β_true, λ_init = Λ_true, alpha_lv_init = alpha_true,
                iterations = 80, g_tol = 1e-5)

            @test fit.converged
            @test isfinite(fit.loglik)
            @test size(fit.alpha_lv) == size(alpha_true)
            @test extract_lv_effects(fit) ≈ fit.Λ * fit.alpha_lv' atol = 1e-10
            @test extract_lv_effects(fit; type = :axis_effect) ≈ fit.alpha_lv atol = 1e-10
            @test cor(vec(extract_lv_effects(fit)), vec(Λ_true * alpha_true')) > 0.98

            Zmean = getLV(fit, Y; N = N, X_lv = X_lv, component = :mean,
                          rotate = false)
            Zinnovation = getLV(fit, Y; N = N, X_lv = X_lv,
                                component = :innovation, rotate = false)
            Ztotal = getLV(fit, Y; N = N, X_lv = X_lv, component = :total,
                           rotate = false)
            @test Zmean ≈ X_lv * fit.alpha_lv atol = 1e-10
            @test Ztotal ≈ Zmean .+ Zinnovation atol = 1e-10
            @test_throws ArgumentError getLV(fit, Y; N = N)
            @test all(isfinite, predict(fit, Y; N = N, X_lv = X_lv))
            @test_throws ArgumentError simulate(fit, size(Y, 2); N = N)
            Ysim = simulate(fit, size(Y, 2); N = N, X_lv = X_lv,
                            rng = MersenneTwister(41))
            @test size(Ysim) == size(Y)
            @test all(0 .<= Ysim .<= N)
            @test_throws ArgumentError confint(fit, Y; N = N)

            br = bridge_fit(; y = Float64.(Y), family = family_key, d = 1,
                            N = N, X_lv = X_lv)
            @test br.family == family_key
            @test br.model == "$(family_key)_xlv_rr"
            @test all(==(GLLVModels._bridge_link_name(link)), br.link)
            @test size(br.lv_effects) == (size(Y, 1), size(X_lv, 2))
            @test size(br.alpha_lv) == size(alpha_true)
            @test br.scores ≈ br.scores_mean .+ br.scores_innovation atol = 1e-10
            @test occursin("binomial C1", br.note)
            br_ci = bridge_fit(
                ; y = Float64.(Y), family = family_key, d = 1, N = N, X_lv = X_lv,
                options = Dict("ci_method" => "wald"))
            @test size(br_ci.lv_effects_lower) == size(br_ci.lv_effects)
            @test br_ci.lv_effects_ci_pd
            @test all(br_ci.lv_effects_lower .< br_ci.lv_effects .< br_ci.lv_effects_upper)
            @test all(br_ci.lv_effects_se .> 0)
            M = trues(size(Y))
            M[1, 1] = false
            @test_throws ArgumentError bridge_fit(
                ; y = Float64.(Y), family = family_key, d = 1, N = N, X_lv = X_lv,
                mask = M)
            @test_throws ArgumentError bridge_fit(
                ; y = Float64.(Y), family = family_key, d = 1, N = N, X_lv = X_lv,
                X = randn(size(Y, 1), size(Y, 2), 1))
        end
    end

    @testset "Poisson X_lv packed objective matches offset Laplace core" begin
        p, n, K, q_lv = 3, 9, 1, 1
        X_lv = reshape(collect(range(-0.8, 0.8; length = n)), n, q_lv)
        β = [0.4, 0.9, 0.6]
        Λ = reshape([0.45, -0.3, 0.2], p, K)
        alpha_lv = reshape([0.6], q_lv, K)
        Y = [mod(t + 2s, 6) for t in 1:p, s in 1:n]
        params = vcat(β, vec(alpha_lv), GLLVModels.pack_lambda(Λ))
        lv_offset = GLLVModels._lv_mean_eta(Λ, X_lv, alpha_lv)
        nll_xlv = GLLVModels.poisson_lv_nll_packed(
            params, Y, p, K, LogLink(); X_lv = X_lv, q_lv = q_lv)
        nll_offset = -GLLVModels.poisson_marginal_loglik_laplace(
            Y, Λ, β, LogLink(); offset = lv_offset)
        @test nll_xlv ≈ nll_offset atol = 1e-10
    end

    @testset "Poisson X_lv native and bridge route" begin
        Random.seed!(909)
        p, n, K, q_lv = 5, 160, 1, 1
        X_lv = reshape(collect(range(-1.1, 1.1; length = n)), n, q_lv)
        β = log.([6.0, 4.0, 8.0, 5.0, 7.0])
        Λ = reshape([0.5, -0.4, 0.3, 0.25, -0.2], p, K)
        alpha_lv = reshape([0.6], q_lv, K)
        z_innov = randn(n)
        z_total = vec(X_lv * alpha_lv) .+ z_innov
        η = β .+ Λ * reshape(z_total, 1, n)
        Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]

        fit = fit_poisson_gllvm(Y; K = 1, X_lv = X_lv,
            β_init = β, λ_init = Λ, alpha_lv_init = alpha_lv,
            iterations = 150, g_tol = 1e-5)

        @test fit.converged
        @test isfinite(fit.loglik)
        @test size(fit.alpha_lv) == size(alpha_lv)
        @test extract_lv_effects(fit) ≈ fit.Λ * fit.alpha_lv' atol = 1e-10
        @test extract_lv_effects(fit; type = :axis_effect) ≈ fit.alpha_lv atol = 1e-10
        @test cor(vec(extract_lv_effects(fit)), vec(Λ * alpha_lv')) > 0.95

        Zmean = getLV(fit, Y; X_lv = X_lv, component = :mean, rotate = false)
        Zinnovation = getLV(fit, Y; X_lv = X_lv, component = :innovation,
                            rotate = false)
        Ztotal = getLV(fit, Y; X_lv = X_lv, component = :total, rotate = false)
        @test Zmean ≈ X_lv * fit.alpha_lv atol = 1e-10
        @test Ztotal ≈ Zmean .+ Zinnovation atol = 1e-10
        @test_throws ArgumentError getLV(fit, Y)
        @test all(isfinite, predict(fit, Y; X_lv = X_lv))
        @test_throws ArgumentError simulate(fit, size(Y, 2))
        Ysim = simulate(fit, size(Y, 2); X_lv = X_lv, rng = MersenneTwister(7))
        @test size(Ysim) == size(Y)
        @test all(Ysim .>= 0)
        @test_throws ArgumentError confint(fit, Y)

        br = bridge_fit(; y = Float64.(Y), family = "poisson", d = 1, X_lv = X_lv)
        @test br.family == "poisson"
        @test br.model == "poisson_xlv_rr"
        @test size(br.lv_effects) == (size(Y, 1), size(X_lv, 2))
        @test size(br.alpha_lv) == size(alpha_lv)
        @test br.scores ≈ br.scores_mean .+ br.scores_innovation atol = 1e-10
        @test occursin("Poisson C1", br.note)
        br_ci = bridge_fit(
            ; y = Float64.(Y), family = "poisson", d = 1, X_lv = X_lv,
            options = Dict("ci_method" => "wald"))
        @test size(br_ci.lv_effects_lower) == size(br_ci.lv_effects)
        @test br_ci.lv_effects_ci_pd
        @test all(br_ci.lv_effects_lower .< br_ci.lv_effects .< br_ci.lv_effects_upper)
        M = trues(size(Y))
        M[1, 1] = false
        @test_throws ArgumentError bridge_fit(
            ; y = Float64.(Y), family = "poisson", d = 1, X_lv = X_lv, mask = M)
        @test_throws ArgumentError bridge_fit(
            ; y = Float64.(Y), family = "poisson", d = 1, X_lv = X_lv,
            X = randn(size(Y, 1), size(Y, 2), 1))
    end

    @testset "NB2 X_lv packed objective matches offset Laplace core" begin
        p, n, K, q_lv = 3, 9, 1, 1
        X_lv = reshape(collect(range(-0.8, 0.8; length = n)), n, q_lv)
        β = [0.5, 1.0, 0.7]
        Λ = reshape([0.45, -0.3, 0.2], p, K)
        alpha_lv = reshape([0.6], q_lv, K)
        r = 8.0
        Y = [mod(t + 2s, 6) for t in 1:p, s in 1:n]
        params = vcat(β, vec(alpha_lv), GLLVModels.pack_lambda(Λ), log(r))
        lv_offset = GLLVModels._lv_mean_eta(Λ, X_lv, alpha_lv)
        nll_xlv = GLLVModels.nb_lv_nll_packed(
            params, Y, p, K, LogLink(); X_lv = X_lv, q_lv = q_lv)
        nll_offset = -GLLVModels.nb_marginal_loglik_laplace(Y, Λ, β, r; offset = lv_offset)
        @test nll_xlv ≈ nll_offset atol = 1e-10
    end

    @testset "NB2 X_lv native and bridge route" begin
        Random.seed!(606)
        p, n, K, q_lv = 5, 160, 1, 1
        X_lv = reshape(collect(range(-1.1, 1.1; length = n)), n, q_lv)
        β = log.([6.0, 4.0, 8.0, 5.0, 7.0])
        Λ = reshape([0.5, -0.4, 0.3, 0.25, -0.2], p, K)
        alpha_lv = reshape([0.6], q_lv, K)
        r_true = 10.0
        z_innov = randn(n)
        z_total = vec(X_lv * alpha_lv) .+ z_innov
        η = β .+ Λ * reshape(z_total, 1, n)
        Y = [rand(NegativeBinomial(r_true, r_true / (r_true + exp(η[t, s]))))
             for t in 1:p, s in 1:n]

        fit = fit_nb_gllvm(Y; K = 1, X_lv = X_lv,
            β_init = β, λ_init = Λ, alpha_lv_init = alpha_lv, r_init = r_true,
            iterations = 200, g_tol = 1e-5)

        @test fit.converged
        @test isfinite(fit.loglik)
        @test fit.r > 0
        @test size(fit.alpha_lv) == size(alpha_lv)
        @test extract_lv_effects(fit) ≈ fit.Λ * fit.alpha_lv' atol = 1e-10
        @test extract_lv_effects(fit; type = :axis_effect) ≈ fit.alpha_lv atol = 1e-10
        @test cor(vec(extract_lv_effects(fit)), vec(Λ * alpha_lv')) > 0.9

        Zmean = getLV(fit, Y; X_lv = X_lv, component = :mean, rotate = false)
        Zinnovation = getLV(fit, Y; X_lv = X_lv, component = :innovation,
                            rotate = false)
        Ztotal = getLV(fit, Y; X_lv = X_lv, component = :total, rotate = false)
        @test Zmean ≈ X_lv * fit.alpha_lv atol = 1e-10
        @test Ztotal ≈ Zmean .+ Zinnovation atol = 1e-10
        @test_throws ArgumentError getLV(fit, Y)
        @test all(isfinite, predict(fit, Y; X_lv = X_lv))
        @test_throws ArgumentError simulate(fit, size(Y, 2))
        Ysim = simulate(fit, size(Y, 2); X_lv = X_lv, rng = MersenneTwister(8))
        @test size(Ysim) == size(Y)
        @test all(Ysim .>= 0)
        @test_throws ArgumentError confint(fit, Y)

        br = bridge_fit(; y = Float64.(Y), family = "negbinomial", d = 1, X_lv = X_lv)
        @test br.family == "negbinomial"
        @test br.model == "negbinomial_xlv_rr"
        @test size(br.lv_effects) == (size(Y, 1), size(X_lv, 2))
        @test size(br.alpha_lv) == size(alpha_lv)
        @test br.scores ≈ br.scores_mean .+ br.scores_innovation atol = 1e-10
        @test occursin("NB2 C1", br.note)
        br_ci = bridge_fit(
            ; y = Float64.(Y), family = "negbinomial", d = 1, X_lv = X_lv,
            options = Dict("ci_method" => "wald"))
        @test size(br_ci.lv_effects_lower) == size(br_ci.lv_effects)
        @test br_ci.lv_effects_ci_pd
        @test all(br_ci.lv_effects_lower .< br_ci.lv_effects .< br_ci.lv_effects_upper)
        M = trues(size(Y))
        M[1, 1] = false
        @test_throws ArgumentError bridge_fit(
            ; y = Float64.(Y), family = "negbinomial", d = 1, X_lv = X_lv, mask = M)
        @test_throws ArgumentError bridge_fit(
            ; y = Float64.(Y), family = "negbinomial", d = 1, X_lv = X_lv,
            X = randn(size(Y, 1), size(Y, 2), 1))
    end

    @testset "Gamma X_lv packed objective matches offset Laplace core" begin
        p, n, K, q_lv = 3, 9, 1, 1
        X_lv = reshape(collect(range(-0.8, 0.8; length = n)), n, q_lv)
        β = [0.5, 1.0, 0.7]
        Λ = reshape([0.4, -0.3, 0.2], p, K)
        alpha_lv = reshape([0.6], q_lv, K)
        α = 5.0
        Y = [0.5 + 0.3 * mod(t + 2s, 5) for t in 1:p, s in 1:n]
        params = vcat(β, vec(alpha_lv), GLLVModels.pack_lambda(Λ), log(α))
        lv_offset = GLLVModels._lv_mean_eta(Λ, X_lv, alpha_lv)
        nll_xlv = GLLVModels.gamma_lv_nll_packed(
            params, Y, p, K, LogLink(); X_lv = X_lv, q_lv = q_lv)
        nll_offset = -GLLVModels.gamma_marginal_loglik_laplace(Y, Λ, β, α; offset = lv_offset)
        @test nll_xlv ≈ nll_offset atol = 1e-10
    end

    @testset "Gamma X_lv native and bridge route" begin
        Random.seed!(707)
        p, n, K, q_lv = 5, 160, 1, 1
        X_lv = reshape(collect(range(-1.1, 1.1; length = n)), n, q_lv)
        β = log.([2.0, 1.5, 3.0, 2.5, 1.8])
        Λ = reshape([0.4, -0.3, 0.25, 0.2, -0.15], p, K)
        alpha_lv = reshape([0.6], q_lv, K)
        α_true = 6.0
        z_innov = randn(n)
        z_total = vec(X_lv * alpha_lv) .+ z_innov
        η = β .+ Λ * reshape(z_total, 1, n)
        Y = [rand(Gamma(α_true, exp(η[t, s]) / α_true)) for t in 1:p, s in 1:n]

        fit = fit_gamma_gllvm(Y; K = 1, X_lv = X_lv,
            β_init = β, λ_init = Λ, alpha_lv_init = alpha_lv, α_init = α_true,
            iterations = 200, g_tol = 1e-5)

        @test fit.converged
        @test isfinite(fit.loglik)
        @test fit.α > 0
        @test size(fit.alpha_lv) == size(alpha_lv)
        @test extract_lv_effects(fit) ≈ fit.Λ * fit.alpha_lv' atol = 1e-10
        @test extract_lv_effects(fit; type = :axis_effect) ≈ fit.alpha_lv atol = 1e-10
        @test cor(vec(extract_lv_effects(fit)), vec(Λ * alpha_lv')) > 0.9

        Zmean = getLV(fit, Y; X_lv = X_lv, component = :mean, rotate = false)
        Zinnovation = getLV(fit, Y; X_lv = X_lv, component = :innovation,
                            rotate = false)
        Ztotal = getLV(fit, Y; X_lv = X_lv, component = :total, rotate = false)
        @test Zmean ≈ X_lv * fit.alpha_lv atol = 1e-10
        @test Ztotal ≈ Zmean .+ Zinnovation atol = 1e-10
        @test_throws ArgumentError getLV(fit, Y)
        @test all(isfinite, predict(fit, Y; X_lv = X_lv))
        @test_throws ArgumentError simulate(fit, size(Y, 2))
        Ysim = simulate(fit, size(Y, 2); X_lv = X_lv, rng = MersenneTwister(9))
        @test size(Ysim) == size(Y)
        @test all(Ysim .> 0)
        @test_throws ArgumentError confint(fit, Y)

        br = bridge_fit(; y = Y, family = "gamma", d = 1, X_lv = X_lv)
        @test br.family == "gamma"
        @test br.model == "gamma_xlv_rr"
        @test size(br.lv_effects) == (size(Y, 1), size(X_lv, 2))
        @test size(br.alpha_lv) == size(alpha_lv)
        @test br.scores ≈ br.scores_mean .+ br.scores_innovation atol = 1e-10
        @test occursin("Gamma C1", br.note)
        br_ci = bridge_fit(
            ; y = Y, family = "gamma", d = 1, X_lv = X_lv,
            options = Dict("ci_method" => "wald"))
        @test size(br_ci.lv_effects_lower) == size(br_ci.lv_effects)
        @test br_ci.lv_effects_ci_pd
        @test all(br_ci.lv_effects_lower .< br_ci.lv_effects .< br_ci.lv_effects_upper)
        M = trues(size(Y))
        M[1, 1] = false
        @test_throws ArgumentError bridge_fit(
            ; y = Y, family = "gamma", d = 1, X_lv = X_lv, mask = M)
    end

    @testset "Beta X_lv packed objective matches offset Laplace core" begin
        p, n, K, q_lv = 3, 9, 1, 1
        X_lv = reshape(collect(range(-0.8, 0.8; length = n)), n, q_lv)
        β = [0.2, -0.3, 0.4]
        Λ = reshape([0.4, -0.3, 0.2], p, K)
        alpha_lv = reshape([0.6], q_lv, K)
        φ = 12.0
        Y = [0.2 + 0.12 * mod(t + 2s, 5) for t in 1:p, s in 1:n]
        params = vcat(β, vec(alpha_lv), GLLVModels.pack_lambda(Λ), log(φ))
        lv_offset = GLLVModels._lv_mean_eta(Λ, X_lv, alpha_lv)
        nll_xlv = GLLVModels.beta_lv_nll_packed(
            params, Y, p, K, LogitLink(); X_lv = X_lv, q_lv = q_lv)
        nll_offset = -GLLVModels.beta_marginal_loglik_laplace(Y, Λ, β, φ; offset = lv_offset)
        @test nll_xlv ≈ nll_offset atol = 1e-10
    end

    @testset "Beta X_lv native and bridge route" begin
        Random.seed!(808)
        p, n, K, q_lv = 5, 160, 1, 1
        X_lv = reshape(collect(range(-1.1, 1.1; length = n)), n, q_lv)
        β = [0.3, -0.5, 0.6, -0.2, 0.4]
        Λ = reshape([0.4, -0.3, 0.25, 0.2, -0.15], p, K)
        alpha_lv = reshape([0.6], q_lv, K)
        φ_true = 15.0
        z_innov = randn(n)
        z_total = vec(X_lv * alpha_lv) .+ z_innov
        η = β .+ Λ * reshape(z_total, 1, n)
        μ = 1.0 ./ (1.0 .+ exp.(-η))
        Y = [rand(Beta(μ[t, s] * φ_true, (1 - μ[t, s]) * φ_true)) for t in 1:p, s in 1:n]

        fit = fit_beta_gllvm(Y; K = 1, X_lv = X_lv,
            β_init = β, λ_init = Λ, alpha_lv_init = alpha_lv, φ_init = φ_true,
            iterations = 200, g_tol = 1e-5)

        @test fit.converged
        @test isfinite(fit.loglik)
        @test fit.φ > 0
        @test size(fit.alpha_lv) == size(alpha_lv)
        @test extract_lv_effects(fit) ≈ fit.Λ * fit.alpha_lv' atol = 1e-10
        @test extract_lv_effects(fit; type = :axis_effect) ≈ fit.alpha_lv atol = 1e-10
        @test cor(vec(extract_lv_effects(fit)), vec(Λ * alpha_lv')) > 0.9

        Zmean = getLV(fit, Y; X_lv = X_lv, component = :mean, rotate = false)
        Zinnovation = getLV(fit, Y; X_lv = X_lv, component = :innovation,
                            rotate = false)
        Ztotal = getLV(fit, Y; X_lv = X_lv, component = :total, rotate = false)
        @test Zmean ≈ X_lv * fit.alpha_lv atol = 1e-10
        @test Ztotal ≈ Zmean .+ Zinnovation atol = 1e-10
        @test_throws ArgumentError getLV(fit, Y)
        @test all(isfinite, predict(fit, Y; X_lv = X_lv))
        @test_throws ArgumentError simulate(fit, size(Y, 2))
        Ysim = simulate(fit, size(Y, 2); X_lv = X_lv, rng = MersenneTwister(10))
        @test size(Ysim) == size(Y)
        @test all(0 .< Ysim .< 1)
        @test_throws ArgumentError confint(fit, Y)

        br = bridge_fit(; y = Y, family = "beta", d = 1, X_lv = X_lv)
        @test br.family == "beta"
        @test br.model == "beta_xlv_rr"
        @test size(br.lv_effects) == (size(Y, 1), size(X_lv, 2))
        @test size(br.alpha_lv) == size(alpha_lv)
        @test br.scores ≈ br.scores_mean .+ br.scores_innovation atol = 1e-10
        @test occursin("Beta C1", br.note)
        br_ci = bridge_fit(
            ; y = Y, family = "beta", d = 1, X_lv = X_lv,
            options = Dict("ci_method" => "wald"))
        @test size(br_ci.lv_effects_lower) == size(br_ci.lv_effects)
        @test br_ci.lv_effects_ci_pd
        @test all(br_ci.lv_effects_lower .< br_ci.lv_effects .< br_ci.lv_effects_upper)
        M = trues(size(Y))
        M[1, 1] = false
        @test_throws ArgumentError bridge_fit(
            ; y = Y, family = "beta", d = 1, X_lv = X_lv, mask = M)
    end

    @testset "X_lv bridge unsupported combinations fail loudly" begin
        Y = randn(4, 45)
        X_lv = randn(45, 1)
        X = randn(4, 45, 1)

        @test_throws ArgumentError bridge_fit(; y = Y, family = "gaussian", d = 0,
                                              X_lv = X_lv)
        @test_throws ArgumentError bridge_fit(; y = Y, family = "gaussian", d = 1,
                                              X_lv = X_lv, X = X)
        # ci_method="wald" is now ADMITTED for X_lv (delta-method Wald on B_lv).
        @test haskey(bridge_fit(; y = Y, family = "gaussian", d = 1, X_lv = X_lv,
                                options = Dict("ci_method" => "wald")), :lv_effects_lower)
        @test_throws ArgumentError bridge_fit(; y = Y, family = "gaussian", d = 1,
                                              X_lv = randn(44, 1))
        @test_throws ArgumentError bridge_fit(; y = Y[1:2, :],
                                              family = ["gaussian", "poisson"],
                                              d = 1, X_lv = X_lv)
    end
end
