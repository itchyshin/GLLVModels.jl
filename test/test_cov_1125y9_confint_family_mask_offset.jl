# Lane cursor-cloud-1125y9: confint_family.jl mask/offset helpers (not covered in open PRs).
using Test, GLLVModels, Random, Distributions

const G = GLLVModels

@testset "1125y9 confint family mask and offset" begin
    @testset "_ci_mask" begin
        Y = rand(0:2, 3, 4)
        @test G._ci_mask(nothing, Y) === nothing
        @test G._ci_mask(trues(3, 4), Y) === nothing
        m = [true false true false; true true false true; false true true false]
        out = G._ci_mask(m, Y)
        @test out !== nothing
        @test size(out) == size(Y)
        @test !out[1, 2] && out[1, 1]
        @test_throws ArgumentError G._ci_mask(falses(3, 4), Y)
        @test_throws ArgumentError G._ci_mask(trues(2, 4), Y)
    end

    @testset "_ci_fill_masked and _ci_count_matrix" begin
        Y = [1.0 2.0; 3.0 4.0]
        m = [true false; true true]
        Yf = G._ci_fill_masked(Y, m, -99.0)
        @test Yf[1, 2] == -99.0
        @test Yf[1, 1] == 1.0
        @test G._ci_fill_masked(Y, nothing, 0.0) === Y
        Yc = G._ci_count_matrix([1.4 2.6; 3.2 4.8], m, 0)
        @test Yc == [1 0; 3 5]
    end

    @testset "_glm_lin_names pack order" begin
        names = G._glm_lin_names(3, 2)
        @test names[1:3] == ["beta[1]", "beta[2]", "beta[3]"]
        @test names[4] == "Lambda[1,1]"
        @test length(names) == 3 + G.rr_theta_len(3, 2)
    end

    @testset "_ci_offset shapes" begin
        p, n = 3, 5
        Y = zeros(p, n)
        O1 = G._ci_offset(0.25, Y, nothing)
        @test O1 == fill(0.25, p, n)
        O2 = G._ci_offset(reshape(1:p, p, 1), Y, nothing)
        @test vec(O2[:, 1]) == collect(1.0:p)
        O3 = G._ci_offset(reshape(1:n, 1, n), Y, nothing)
        @test vec(O3[1, :]) == collect(1.0:n)
        @test G._ci_offset([0.1, 0.2, 0.3], Y, nothing) == repeat([0.1, 0.2, 0.3], 1, n)
        Ysq = zeros(4, 4)
        @test_throws ArgumentError G._ci_offset([0.1, 0.2, 0.3, 0.4], Ysq, nothing)
        m = falses(p, n)
        m[2, 3] = true
        @test_throws ArgumentError G._ci_offset(fill(Inf, p, n), Y, m)
    end

    @testset "_ci_refuse_mask on unsupported fit" begin
        rng = MersenneTwister(1125091)
        Y = rand(rng, 0:3, 3, 12)
        fit = fit_poisson_gllvm(Y; K = 1, iterations = 12)
        m = trues(3, 12)
        m[1, 1] = false
        @test G._ci_refuse_mask(fit, nothing, Y) === nothing
        @test_throws ArgumentError G._ci_refuse_mask(fit, m, Y)
    end

    @testset "_binomial_ridge_ci_guard" begin
        bf = G.BinomialFit(
            zeros(2), zeros(2, 1), LogitLink(), -1.0, true, 5,
            nothing, Float64[], :fisher, nothing, nothing, 2.0, nothing,
        )
        @test_throws ArgumentError G._binomial_ridge_ci_guard(bf)
        bf2 = G.BinomialFit(
            zeros(2), zeros(2, 1), LogitLink(), -1.0, true, 5,
            nothing, Float64[], :fisher, nothing, nothing, Inf, nothing,
        )
        @test G._binomial_ridge_ci_guard(bf2) === nothing
    end
end