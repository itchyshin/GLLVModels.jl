using Test
using GLLVModels
using Random

@testset "bridge missing-response mask" begin
    Random.seed!(531)
    p, n, K = 3, 32, 1
    Y = Float64.(rand(0:4, p, n))
    mask = trues(p, n)
    mask[1, 2] = false
    mask[2, 9] = false
    mask[3, 17] = false

    Ysane = copy(Y)
    Ysane[.!mask] .= 0.0
    Ygarbage = copy(Y)
    Ygarbage[.!mask] .= 999.0

    @testset "Poisson mask parity and sentinel invariance" begin
        br = bridge_fit(; y = Ysane, family = "poisson", d = K, mask = mask)
        direct = fit_poisson_gllvm(round.(Int, Ysane); K = K, mask = mask)
        scores = getLV(direct, round.(Int, Ysane); rotate = true, mask = mask)

        @test br.nobs == count(mask)
        @test isapprox(br.loglik, direct.loglik; atol = 1e-8, rtol = 0)
        @test isapprox(br.alpha, direct.β; atol = 1e-8, rtol = 0)
        @test isapprox(br.loadings, getLoadings(direct; rotate = true); atol = 1e-8, rtol = 0)
        @test isapprox(br.scores, scores; atol = 1e-8, rtol = 0)

        # BIC must agree with the `nobs` field it is built from — R's p·n
        # observed-cell count (docs/dev-log/decisions/2026-09-01-maintainer-decisions-round1.md
        # #1), NOT the site count `n`. Before the fix `bic` used `log(n)`
        # while `nobs` already used `log(count(mask))`, so the two disagreed
        # whenever any cell was masked.
        @test isapprox(br.bic, br.df * log(br.nobs) - 2 * br.loglik; atol = 1e-8, rtol = 0)
        @test br.nobs != n
        @test !isapprox(br.bic, br.df * log(n) - 2 * br.loglik; atol = 1e-8, rtol = 0)

        br_garbage = bridge_fit(; y = Ygarbage, family = "poisson", d = K, mask = mask)
        @test isapprox(br_garbage.loglik, br.loglik; atol = 1e-8, rtol = 0)
        @test isapprox(br_garbage.alpha, br.alpha; atol = 1e-8, rtol = 0)
        @test isapprox(br_garbage.loadings, br.loadings; atol = 1e-8, rtol = 0)
        @test isapprox(br_garbage.scores, br.scores; atol = 1e-8, rtol = 0)

        br_ci = bridge_fit(; y = Ysane, family = "poisson", d = K, mask = mask,
                           options = Dict("ci_method" => "wald"))
        br_ci_garbage = bridge_fit(; y = Ygarbage, family = "poisson", d = K,
                                   mask = mask, options = Dict("ci_method" => "wald"))
        @test br_ci.ci_method == "wald"
        @test br_ci.ci_param_names == br_ci_garbage.ci_param_names
        @test isapprox(br_ci.ci_estimate, br_ci_garbage.ci_estimate; atol = 1e-8, rtol = 0)
        @test isapprox(br_ci.ci_lower, br_ci_garbage.ci_lower; atol = 1e-8, rtol = 0)
        @test isapprox(br_ci.ci_upper, br_ci_garbage.ci_upper; atol = 1e-8, rtol = 0)
        @test all(isfinite, br_ci.ci_estimate)
    end

    @testset "all-true mask is the complete-data bridge path" begin
        br_nomask = bridge_fit(; y = Y, family = "poisson", d = K)
        br_alltrue = bridge_fit(; y = Y, family = "poisson", d = K, mask = trues(p, n))
        @test br_alltrue.nobs == p * n
        @test br_alltrue.loglik == br_nomask.loglik
        @test br_alltrue.alpha == br_nomask.alpha
        @test br_alltrue.loadings == br_nomask.loadings
    end

    @testset "NB1 mask parity and sentinel invariance" begin
        br = bridge_fit(; y = Ysane, family = "nb1", d = K, mask = mask)
        direct = fit_nb1_gllvm_grouped(round.(Int, Ysane); K = K,
                                       group = collect(1:p), mask = mask)

        @test br.nobs == count(mask)
        @test isapprox(br.loglik, direct.loglik; atol = 1e-8, rtol = 0)
        @test isapprox(br.alpha, direct.β; atol = 1e-8, rtol = 0)
        @test br.dispersion_group_id == collect(1:p)
        @test isapprox(br.dispersion, direct.φ[direct.group]; atol = 1e-8, rtol = 0)
        @test isapprox(br.loadings, getLoadings(direct; rotate = true); atol = 1e-8, rtol = 0)
        @test br.scores isa Matrix{Float64}

        br_garbage = bridge_fit(; y = Ygarbage, family = "nb1", d = K, mask = mask)
        @test isapprox(br_garbage.loglik, br.loglik; atol = 1e-8, rtol = 0)
        @test isapprox(br_garbage.alpha, br.alpha; atol = 1e-8, rtol = 0)
        @test isapprox(br_garbage.dispersion, br.dispersion; atol = 1e-8, rtol = 0)
        @test isapprox(br_garbage.loadings, br.loadings; atol = 1e-8, rtol = 0)
        @test br_garbage.scores == br.scores
    end

    @testset "masked no-X CIs route for all admitted one-part non-Gaussian rows" begin
        mask_small = trues(2, 10)
        mask_small[1, 3] = false
        mask_small[2, 8] = false
        cases = [
            ("poisson", [1 2 0 3 1 4 2 0 1 3; 2 1 3 0 2 1 4 2 3 1], nothing),
            ("binomial", [1 0 1 1 0 1 0 1 1 0; 0 1 1 0 1 0 1 1 0 1], fill(1, 2, 10)),
            ("negbinomial", [2 3 1 4 2 5 3 1 2 4; 1 2 4 2 3 1 5 3 2 1], nothing),
            ("nb1", [2 3 1 4 2 5 3 1 2 4; 1 2 4 2 3 1 5 3 2 1], nothing),
            ("beta", [0.20 0.35 0.40 0.55 0.65 0.72 0.30 0.45 0.58 0.80;
                      0.75 0.62 0.50 0.38 0.28 0.18 0.85 0.70 0.55 0.42], nothing),
            ("gamma", [1.2 1.5 2.0 2.4 1.8 2.8 3.1 1.7 2.2 2.6;
                       2.1 2.4 1.6 1.9 2.7 3.2 2.5 1.4 2.0 2.9], nothing),
            ("betabinomial", [1 2 0 3 1 4 2 0 1 3; 2 1 3 0 2 1 4 2 3 1], fill(6, 2, 10)),
        ]
        for (family, Ycase, Ncase) in cases
            if Ncase === nothing
                br = bridge_fit(; y = Float64.(Ycase), family = family, d = 0,
                                mask = mask_small,
                                options = Dict("ci_method" => "wald"))
            else
                br = bridge_fit(; y = Float64.(Ycase), family = family, d = 0,
                                N = Ncase, mask = mask_small,
                                options = Dict("ci_method" => "wald"))
            end
            @test br.ci_method == "wald"
            @test br.nobs == count(mask_small)
            @test length(br.ci_param_names) == length(br.ci_estimate)
            @test length(br.ci_lower) == length(br.ci_estimate)
            @test length(br.ci_upper) == length(br.ci_estimate)
            @test all(isfinite, br.ci_estimate)
        end

        Yp = Float64.([1 2 1 3 2 4 1 2 3 1; 2 1 3 2 1 2 4 1 2 3])
        br_profile = bridge_fit(; y = Yp, family = "poisson", d = 0,
                                mask = mask_small,
                                options = Dict("ci_method" => "profile"))
        @test br_profile.ci_method == "profile"
        @test length(br_profile.ci_param_names) == 2

        br_boot = bridge_fit(; y = Yp, family = "poisson", d = 0,
                             mask = mask_small,
                             options = Dict("ci_method" => "bootstrap",
                                            "ci_nboot" => 6,
                                            "ci_seed" => 42))
        @test br_boot.ci_method == "bootstrap"
        @test length(br_boot.ci_param_names) == 2
    end

    @testset "ordinal_probit mask uses the probit ordinal bridge" begin
        Yo = [1 2 3 1 2 3 1 2 3 1 2 3
              2 3 1 2 3 1 2 3 1 2 3 1
              3 1 2 3 1 2 3 1 2 3 1 2]
        mo = trues(size(Yo))
        mo[1, 4] = false
        mo[3, 9] = false
        Yog = copy(Yo)
        Yog[.!mo] .= 1

        br = bridge_fit(; y = Float64.(Yog), family = "ordinal_probit", d = K, mask = mo)
        direct = fit_ordinal_gllvm_pertrait(Yog; K = K, link = ProbitLink(), mask = mo)

        @test br.family == "ordinal_probit"
        @test br.model == "ordinal_probit_rr"
        @test all(==("ProbitLink"), br.link)
        @test br.nobs == count(mo)
        @test br.cutpoint_mode == "per_trait"
        @test br.n_categories == direct.C
        @test isapprox(br.loglik, direct.loglik; atol = 1e-8, rtol = 0)
        @test isapprox(br.loadings, getLoadings(direct; rotate = true); atol = 1e-8, rtol = 0)
    end

    @testset "Gaussian mask parity and sentinel invariance (#912)" begin
        Random.seed!(912)
        pg, ng, Kg = 4, 40, 1
        Yg = randn(pg, 1) * randn(1, ng) .+ 0.5 .* randn(pg, ng) .+ [1.0, -2.0, 0.5, 3.0]
        Mg = trues(pg, ng)
        Mg[1, 3] = false
        Mg[2, 11] = false
        Mg[3, 20] = false
        Mg[4, 31] = false
        Mg[1, 32] = false
        Ygarbage_g = copy(Yg)
        Ygarbage_g[.!Mg] .= 999.0

        # The bridge's no-X convention centres each trait on its mean over the
        # OBSERVED cells, then fits the masked zero-mean model to the centred data.
        alpha_obs = [sum(Yg[t, Mg[t, :]]) / count(Mg[t, :]) for t in 1:pg]
        Yc = Yg .- alpha_obs
        Yc[.!Mg] .= 0.0
        direct = fit_gaussian_gllvm(Yc; K = Kg, mask = Mg)

        br = bridge_fit(; y = Yg, family = "gaussian", d = Kg, mask = Mg)
        @test br.family == "gaussian"
        @test br.nobs == count(Mg)
        @test isapprox(br.alpha, alpha_obs; atol = 1e-10, rtol = 0)
        @test isapprox(br.loglik, direct.logLik; atol = 1e-6, rtol = 0)
        @test br.converged == direct.converged
        @test isapprox(br.loadings, getLoadings(direct; rotate = true); atol = 1e-6, rtol = 0)

        br_garbage = bridge_fit(; y = Ygarbage_g, family = "gaussian", d = Kg, mask = Mg)
        @test isapprox(br_garbage.loglik, br.loglik; atol = 1e-8, rtol = 0)
        @test isapprox(br_garbage.alpha, br.alpha; atol = 1e-10, rtol = 0)
        @test isapprox(br_garbage.loadings, br.loadings; atol = 1e-8, rtol = 0)

        # An all-true mask is the complete-data path.
        br_nomask = bridge_fit(; y = Yg, family = "gaussian", d = Kg)
        br_alltrue = bridge_fit(; y = Yg, family = "gaussian", d = Kg, mask = trues(pg, ng))
        @test br_alltrue.loglik == br_nomask.loglik
        @test br_alltrue.alpha == br_nomask.alpha

        # Masked CIs are not wired for Gaussian yet and are refused, not dropped.
        @test_throws ArgumentError bridge_fit(; y = Yg, family = "gaussian", d = Kg,
                                              mask = Mg,
                                              options = Dict("ci_method" => "wald"))

        # A trait with no observed cell cannot be centred and is refused.
        Mbad = copy(Mg)
        Mbad[2, :] .= false
        @test_throws ArgumentError bridge_fit(; y = Yg, family = "gaussian", d = Kg,
                                              mask = Mbad)
    end

    @testset "unsupported masked bridge cells fail loudly" begin
        X = randn(p, n, 1)
        @test_throws ArgumentError bridge_fit(; y = Y, family = "poisson", d = K,
                                              mask = mask, X = X)
        # Gaussian masks are wired for the plain no-X route only (#912); the
        # REML, fixed-effect X and X_lv routes still refuse a mask loudly.
        @test_throws ArgumentError bridge_fit(; y = randn(p, n), family = "gaussian",
                                              d = K, mask = mask, X = X)
        @test_throws ArgumentError bridge_fit(; y = randn(p, n), family = "gaussian",
                                              d = K, mask = mask,
                                              options = Dict("reml" => true))
        @test_throws ArgumentError bridge_fit(; y = randn(p, n), family = "gaussian",
                                              d = K, mask = mask, X_lv = randn(n, 1))
        @test_throws ArgumentError bridge_fit(; y = Y, family = "ordinal", d = K,
                                              mask = mask,
                                              options = Dict("ci_method" => "wald"))
        @test_throws ArgumentError bridge_fit(; y = Y, family = ["poisson", "binomial"],
                                              d = K, mask = mask)
    end
end
