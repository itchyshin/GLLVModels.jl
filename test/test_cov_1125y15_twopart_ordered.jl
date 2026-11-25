# Lane cursor-cloud-1125y15 — src/families/twopart.jl, ordered_beta.jl, beta_hurdle.jl
using GLLVModels, Test, Random, Distributions, LinearAlgebra

const G = GLLVModels

@testset "cov 1125y15 twopart, ordered beta, beta hurdle" begin
    @testset "ordered_beta_logp and score" begin
        lp = G.ordered_beta_logp(0.3, 0.0, 0.0, 1.0, 2.0)
        @test isfinite(lp) && lp < 0
        sc, w = G._ob_score_weight(0.5, 0.1, 0.0, 1.0, 1.5)
        @test isfinite(sc) && isfinite(w) && w > 0
        @test all(isfinite, (G.ordered_beta_logp(0.2, 0.0, 0.0, 1.0, 1.0),
            G.ordered_beta_logp(0.5, 0.0, 0.0, 1.0, 1.0),
            G.ordered_beta_logp(0.8, 0.1, -0.1, 1.0, 1.2)))
        @test G.ordered_beta_logp(0.5, 0.0, 0.0, 1.0, 1.0) < 0
    end

    @testset "predict ordered beta type guard" begin
        rng = MersenneTwister(3)
        Y = clamp.(rand(rng, Beta(2, 2), 3, 12), 0.05, 0.95)
        fit = fit_ordered_beta_gllvm(Y; K = 1, iterations = 60, g_tol = 1e-3)
        @test predict(fit, Y; type = :mean) isa Matrix
        @test_throws ArgumentError predict(fit, Y; type = :response)
        @test isfinite(fit.loglik)
    end

    @testset "twopart _tp_pieces hurdle families" begin
        pieces = G._tp_pieces(G.HurdlePoisson(), 0, 0.0, 0.0)
        @test length(pieces) == 5
        @test pieces[3] ≈ 0.25 atol = 1e-12
        @test isfinite(pieces[4])
        ypos = G._tp_pieces(G.DeltaLogNormal(), 2, -1.0, 0.2)
        @test all(isfinite, ypos)
    end

    @testset "delta lognormal σ length guard via fitter" begin
        Y = abs.(randn(2, 6)) .+ 0.1
        Y[1, 3] = 0.0
        @test_throws ArgumentError fit_delta_lognormal_gllvm(Y; K = 1, iterations = 5, predictor = :bogus)
    end

    @testset "beta hurdle predict type guard" begin
        rng = MersenneTwister(7)
        Y = clamp.(rand(rng, Beta(2, 3), 4, 10), 1e-4, 1 - 1e-4)
        fit = fit_beta_hurdle_gllvm(Y; K = 1, iterations = 80, g_tol = 1e-3)
        @test predict(fit, Y; type = :response) isa Matrix
        @test_throws ArgumentError predict(fit, Y; type = :mean)
        shown = sprint(show, fit)
        @test occursin("BetaHurdle", shown)
    end

    @testset "twopart ZIP ICC helpers" begin
        icc_p = G._zi_Icc_pois(0.2, 1.5)
        icc_n = G._zi_Icc_nb(0.15, 2.0, 3.0)
        @test 0 <= icc_p <= 1
        @test 0 <= icc_n <= 1
        @test isfinite(icc_p) && isfinite(icc_n)
    end

    @testset "hurdle NB pieces at zero" begin
        ηz, ηc = -0.5, 0.1
        pieces0 = G._tp_pieces(G.HurdleNB(), 0, ηz, ηc)
        @test all(isfinite, pieces0)
        pieces3 = G._tp_pieces(G.HurdleNB(), 3, ηz, ηc)
        @test all(isfinite, pieces3)
        @test pieces3[5] <= pieces0[5] + 1.0
    end
end