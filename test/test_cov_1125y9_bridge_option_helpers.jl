# Lane cursor-cloud-1125y9: bridge.jl option parsing helpers (outside open coverage lanes).
using Test, GLLVModels, LinearAlgebra

const G = GLLVModels

@testset "1125y9 bridge option helpers" begin
    @testset "_bridge_truthy" begin
        @test G._bridge_truthy(true)
        @test G._bridge_truthy(1)
        @test !G._bridge_truthy(0)
        @test G._bridge_truthy("TRUE")
        @test G._bridge_truthy(" yes ")
        @test !G._bridge_truthy("no")
        @test !G._bridge_truthy(nothing)
        @test !G._bridge_truthy(:symbol)
    end

    @testset "_bridge_get" begin
        opts = Dict("ci_method" => "wald", "foo" => 1)
        @test G._bridge_get(opts, "ci_method", "none") == "wald"
        @test G._bridge_get(opts, "missing", 42) == 42
        @test G._bridge_get(nothing, "x", 0) == 0
    end

    @testset "_bridge_names" begin
        @test G._bridge_names(nothing, 3, "t") == ["t1", "t2", "t3"]
        @test G._bridge_names(["a", "b"], 2, "t") == ["a", "b"]
        @test_throws ArgumentError G._bridge_names(["a"], 2, "t")
    end

    @testset "_bridge_corr_from_sigma" begin
        Σ = [4.0 1.0; 1.0 9.0]
        R = G._bridge_corr_from_sigma(Σ)
        @test R[1, 1] == 1.0 && R[2, 2] == 1.0
        @test R[1, 2] ≈ 1 / 6 rtol = 1e-12
        R0 = G._bridge_corr_from_sigma(zeros(2, 2))
        @test R0[1, 2] == 0.0
    end

    @testset "_bridge_ci_method and nboot" begin
        @test G._bridge_ci_method(Dict("ci_method" => "Profile")) == "profile"
        @test_throws ArgumentError G._bridge_ci_method(Dict("ci_method" => "bayes"))
        @test G._bridge_ci_nboot(Dict("ci_method" => "wald", "ci_nboot" => 200)) == 200
        @test_throws ArgumentError G._bridge_ci_nboot(Dict("ci_method" => "bootstrap", "ci_nboot" => 0))
        @test_throws ArgumentError G._bridge_ci_nboot(Dict("ci_method" => "bootstrap", "ci_nboot" => 2.5))
    end

    @testset "_bridge_family_key aliases" begin
        @test G._bridge_family_key("NB2") == "negbinomial"
        @test G._bridge_family_key("zero_inflated_negbin") == "zinb"
        @test_throws ArgumentError G._bridge_family_key("cauchy")
    end

    @testset "_bridge_rr_df" begin
        @test G._bridge_rr_df(5, 2) == 5 * 2 - 1
        @test G._bridge_rr_df(4, 1) == 4
    end
end