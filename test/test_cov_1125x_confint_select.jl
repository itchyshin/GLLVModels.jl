# Lane 1125x round 3: confint.jl term-name list and parm selector resolution.
using Test, GLLVModels, Random, LinearAlgebra

const G = GLLVModels

function _plain_gauss()
    rng = MersenneTwister(112503)
    p, n, K = 3, 25, 2
    Λ = 0.4 .* randn(rng, p, K)
    Y = Λ * randn(rng, K, n) .+ 0.3 .* randn(rng, p, n)
    return fit_gaussian_gllvm(Y; K = K), Y
end

@testset "1125x confint term names and parm selectors" begin
    @testset "_confint_lambda_term_names order" begin
        names = G._confint_lambda_term_names("Lambda_B", 4, 2)
        @test names == [
            "Lambda_B[1,1]", "Lambda_B[2,2]",
            "Lambda_B[2,1]", "Lambda_B[3,1]", "Lambda_B[4,1]",
            "Lambda_B[3,2]", "Lambda_B[4,2]",
        ]
    end

    fit, _ = _plain_gauss()
    terms, kinds = G._confint_all_term_names(fit)
    @test length(terms) == length(kinds)
    @test "sigma_eps" in terms
    @test all(k -> kinds[k] == :log_sd, findall(==("sigma_eps"), terms))

    @testset "structured Gaussian lists B/W/sigma_phy/Lambda tiers" begin
        rng = MersenneTwister(112504)
        p, n = 3, 20
        Σ_phy = [1.0 0.2 0.1; 0.2 1.0 0.15; 0.1 0.15 1.0]
        Y = 0.5 .* randn(rng, p, n)
        sfit = fit_gaussian_gllvm(Y; K = 1, has_diag = true, has_phy_unique = true, Σ_phy = Σ_phy)
        t2, k2 = G._confint_all_term_names(sfit)
        @test any(startswith(n, "sigma_B[") for n in t2)
        @test any(startswith(n, "sigma_W[") for n in t2)
        @test any(startswith(n, "sigma_phy[") for n in t2)
        @test count(n -> startswith(n, "Lambda_B["), t2) == p
        σb = findall(k -> k2[k] == :log_sd, 1:length(k2))
        @test !isempty(σb)
    end

    @testset "_confint_select_indices_one" begin
        @test G._confint_select_indices_one("sigma_eps", terms) ==
            [findfirst(==("sigma_eps"), terms)]
        idx_Λ = G._confint_select_indices_one("Lambda", terms)
        @test all(t -> startswith(t, "Lambda_"), terms[idx_Λ])
        idx_b1 = G._confint_select_indices_one("Lambda_B[1,1]", terms)
        @test terms[idx_b1[1]] == "Lambda_B[1,1]"
        @test G._confint_select_indices_one("Lambda:1,1", terms) == idx_b1
        @test G._confint_select_indices_one("beta", terms) ==
            findall(startswith("beta["), terms)
        @test_throws ArgumentError G._confint_select_indices_one("not_a_parm", terms)
    end

    @testset "_confint_select_indices vectors and nothing" begin
        @test G._confint_select_indices(nothing, terms) == collect(1:length(terms))
        @test G._confint_select_indices(:sigma_eps, terms) ==
            G._confint_select_indices_one("sigma_eps", terms)
        both = G._confint_select_indices(["sigma_eps", "Lambda_B"], terms)
        @test issubset(G._confint_select_indices_one("sigma_eps", terms), both)
        @test issubset(G._confint_select_indices_one("Lambda_B", terms), both)
        @test_throws ArgumentError G._confint_select_indices(42, terms)
    end
end