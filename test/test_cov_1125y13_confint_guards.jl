# Lane cursor-cloud-1125y13 — src/confint.jl parse/check helpers (beyond 1125x select)
using GLLVModels, Test, Random, LinearAlgebra

const G = GLLVModels

@testset "cov 1125y13 confint guards" begin
    rng = MersenneTwister(1125136)

    @testset "_require_complete_y" begin
        Y = [1.0 2.0; missing 4.0]
        @test_throws ArgumentError G._require_complete_y("confint", Y)
        Yn = [1.0 NaN; 3.0 4.0]
        @test_throws ArgumentError G._require_complete_y("bootstrap_ci", Yn)
        Yi = [1.0 Inf; 3.0 4.0]
        @test_throws ArgumentError G._require_complete_y("vcov", Yi)
        @test G._require_complete_y("confint", [1.0 2.0; 3.0 4.0]) === nothing
        @test G._require_complete_y("confint", nothing) === nothing
    end

    @testset "_confint_check_method and kwargs" begin
        @test_throws ArgumentError G._confint_check_method(:aghq, "PoissonFit", nothing)
        @test G._confint_check_method(:wald, "GllvmFit", nothing) === nothing
        kw = (; level = 0.9, parm = "log_σ_eps")
        @test G._confint_check_kwargs(kw, (:level, :parm), (), "GllvmFit", :wald) === nothing
        @test_throws ArgumentError G._confint_check_kwargs((; bogus = 1), (:level,), (), "GllvmFit", :wald)
    end

    p, n, K = 3, 12, 1
    Y = randn(rng, p, n)
    fit = fit_gaussian_gllvm(Y; K = K, iterations = 20)

    @testset "_confint_parse_derived and expand" begin
        pr = G._confint_parse_derived("communality[2]")
        @test pr.kind == :communality
        @test_throws ArgumentError G._confint_expand_derived(fit,
            G._confint_parse_derived("communality:unit[1]"))
        @test_throws ArgumentError G._confint_expand_derived(fit,
            G._confint_parse_derived("proportion:unknown[1]"))
        targets = G._confint_expand_derived(fit, G._confint_parse_derived("communality[1]"))
        @test length(targets) == 1
        @test targets[1].label == "communality[1]"
        all_targets = G._confint_expand_derived(fit, G._confint_parse_derived("communality"))
        @test length(all_targets) == p
        @test_throws ArgumentError G._confint_expand_derived(fit, G._confint_parse_derived("rho[1,1]"))
        pair = G._confint_expand_derived(fit, G._confint_parse_derived("rho[1,2]"))
        @test pair[1].label == "rho[1,2]"
    end

    @testset "_confint_lambda_term_names" begin
        terms = G._confint_lambda_term_names("loading_B", 3, 2)
        @test length(terms) == 5
        @test all(startswith.(terms, "loading_B"))
    end

    @testset "_gaussian_wald_covariance_from_hessian" begin
        H = [2.0 0.1; 0.1 3.0]
        V, se, pd = G._gaussian_wald_covariance_from_hessian(H, 2)
        @test pd
        @test all(se .> 0)
        Vbad, sebad, pdbad = G._gaussian_wald_covariance_from_hessian([1.0 2.0; 2.0 1.0], 2)
        @test !pdbad
        @test all(isnan, sebad)
    end
end