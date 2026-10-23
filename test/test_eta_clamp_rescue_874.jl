# #874: a user `β_init` entry above the η clamp falls back to the default warm start.
# Since #873 (#869) a log-link objective returns its failure sentinel once an intercept
# (plus offset) is above `_ETA_CLAMP` = 30, so a start there could not move and the fit
# ended at loglik = -Inf, converged = false. Each fitter that accepts `β_init` now replaces
# such an entry by its default warm start and warns. Data: 6 species of
# mvabund::spider$abund (28 sites), as in test_eta_clamp_869.jl.
using Test, GLLVModels
const _G874 = GLLVModels

const _Y874 = [
    25 0 15 2 1 0 2 0 1 3 15 16 3 0 0 0 0 0 0 0 0 7 17 11 9 3 29 15;
    10 2 20 6 20 6 7 11 1 0 1 13 43 2 0 3 0 1 1 2 1 0 0 0 1 0 0 0;
    4 30 9 24 9 6 16 7 0 0 1 0 18 4 0 0 0 0 0 0 0 0 0 0 2 0 0 0;
    60 1 29 7 2 11 30 2 26 22 95 96 24 14 0 0 0 0 0 0 1 2 6 3 11 0 1 6;
    45 37 45 94 76 24 105 1 1 0 1 8 72 72 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
    57 65 66 86 91 63 118 30 2 1 4 13 97 94 25 28 23 25 22 22 18 1 1 0 16 1 0 2]
# Zero-truncated counts for the truncated NB2 fitters: the spider data put r at the 1e-6
# boundary there, so the default start is not converged. Fixed-quantile draws from a
# zero-truncated NB2 (μ_t = exp(0.5 + 0.4t), r = 3, u = frac(0.618034·(28t + s))).
const _YT874 = [
    6 3 1 4 2 1 3 1 5 2 1 4 2 7 3 1 4 2 1 3 2 6 3 1 4 2 10 3;
    2 7 3 1 5 2 9 4 2 6 3 1 4 2 7 3 1 5 3 11 4 2 6 3 1 5 2 8;
    5 2 8 4 1 6 3 10 5 2 7 3 13 6 2 9 4 1 7 3 11 5 2 8 4 17 6 3;
    13 7 2 10 5 17 8 3 12 6 1 9 4 15 7 3 11 5 20 8 4 13 6 2 10 5 16 8;
    5 17 9 43 13 6 20 10 3 15 8 26 12 5 18 9 2 14 7 22 11 4 16 8 32 13 6 19;
    15 5 22 11 35 17 7 25 13 2 20 10 30 16 6 23 12 41 18 8 27 14 4 21 11 33 17 7]

# one intercept at 35, the others at plain log-mean values (not the default warm start)
function _b874(Y)
    b = log.(vec(sum(Float64.(Y); dims = 2)) ./ size(Y, 2) .+ 0.5)
    b[1] = 35.0
    return b
end

_rescue_warned874(logs) = any(l -> occursin("above the η clamp", string(l.message)), logs)

@testset "#874 β_init above the η clamp falls back to the default warm start" begin
    @testset "helper" begin
        Z = [0.0 2.0; 1.0 3.0]                       # default warm start = row means [1, 2]
        @test _G874._rescue_beta_init([35.0, 0.5], Z, nothing) == [1.0, 0.5]
        @test_logs (:warn, r"β_init.*above the η clamp") _G874._rescue_beta_init([35.0, 0.5], Z, nothing; caller = "t")
        @test_logs _G874._rescue_beta_init([29.0, 0.5], Z, nothing)                 # below the cap: kept, silent
        @test _G874._rescue_beta_init([29.0, 0.5], Z, nothing) == [29.0, 0.5]
        @test _G874._rescue_beta_init([29.0, 0.5], Z, [2.0 0.0; 0.0 0.0]) == [1.0, 0.5]   # offset pushes it over
        @test _G874._rescue_beta_init([29.0, 0.5], Z, [2.0 0.0; 0.0 0.0];
                                      mask = [false true; true true]) == [29.0, 0.5]       # unobserved cell ignored
        @test _G874._rescue_beta_init([35.0, 0.5], Z, nothing; guarded = false) == [35.0, 0.5]  # non-log link
        @test _G874._rescue_beta_init([35.0], Z, nothing) == [35.0]                 # wrong length: the fitter's call
        # the replacement is bit-identical to the default warm start (`β_init = nothing`)
        Zr = log.(Float64.(_Y874) .+ 0.5)
        @test _G874._rescue_beta_init(fill(35.0, size(Zr, 1)), Zr, nothing) == vec(sum(Zr; dims = 2)) ./ size(Zr, 2)
    end

    @testset "fitters reach the default-start optimum" begin
        Y = _Y874
        Yt = _YT874                                  # zero-truncated responses
        Yf = Float64.(Y) .+ 0.5                      # positive responses
        Yn = Float64.(Y)                             # non-negative (Tweedie)
        grp = [1, 1, 2, 2, 3, 3]
        for (nm, D, fit) in [
            ("fit_poisson_gllvm K = 1",            Y,  (D, b) -> fit_poisson_gllvm(D; K = 1, β_init = b)),
            ("fit_nb_gllvm",                       Y,  (D, b) -> fit_nb_gllvm(D; K = 0, β_init = b)),
            ("fit_nb1_gllvm",                      Y,  (D, b) -> fit_nb1_gllvm(D; K = 0, β_init = b)),
            ("fit_nb_gllvm_grouped",               Y,  (D, b) -> fit_nb_gllvm_grouped(D; K = 0, group = grp, β_init = b)),
            ("fit_nb1_gllvm_grouped",              Y,  (D, b) -> fit_nb1_gllvm_grouped(D; K = 0, group = grp, β_init = b)),
            ("fit_compoisson_gllvm",               Y,  (D, b) -> fit_compoisson_gllvm(D; K = 0, β_init = b)),
            ("fit_gp1_gllvm K = 1",                Y,  (D, b) -> fit_gp1_gllvm(D; K = 1, β_init = b)),
            ("fit_censored_poisson_gllvm",         Y,  (D, b) -> fit_censored_poisson_gllvm(D; K = 0, β_init = b)),
            ("fit_truncated_poisson_gllvm",        Yt, (D, b) -> fit_truncated_poisson_gllvm(D; K = 0, β_init = b)),
            ("fit_truncated_nbinom2_gllvm",        Yt, (D, b) -> fit_truncated_nbinom2_gllvm(D; K = 0, β_init = b)),
            ("fit_truncated_nbinom2_gllvm_pertrait", Yt, (D, b) -> fit_truncated_nbinom2_gllvm_pertrait(D; K = 0, β_init = b)),
            ("fit_tweedie_gllvm",                  Yn, (D, b) -> fit_tweedie_gllvm(D; K = 0, β_init = b)),
            ("fit_gamma_gllvm",                    Yf, (D, b) -> fit_gamma_gllvm(D; K = 0, β_init = b)),
            ("fit_exponential_gllvm K = 1",        Yf, (D, b) -> fit_exponential_gllvm(D; K = 1, β_init = b)),
        ]
            f0 = redirect_stderr(devnull) do; fit(D, nothing); end
            logs, f = Test.collect_test_logs() do; fit(D, _b874(D)); end
            @test (nm, f0.converged) == (nm, true)
            @test (nm, f.converged, _rescue_warned874(logs)) == (nm, true, true)
            @test (nm, isapprox(f.loglik, f0.loglik; rtol = 1e-6)) == (nm, true)
        end
    end

    @testset "an offset that carries the intercept past the cap is rescued too" begin
        Y = _Y874
        O = fill(10.0, size(Y))
        b = _b874(Y) .- 10.0; b[1] = 25.0           # 25 + offset 10 = 35 > 30
        f0 = redirect_stderr(devnull) do; fit_nb_gllvm(Y; K = 0, offset = O); end
        logs, f = Test.collect_test_logs() do; fit_nb_gllvm(Y; K = 0, offset = O, β_init = b); end
        @test f.converged
        @test _rescue_warned874(logs)
        @test isapprox(f.loglik, f0.loglik; rtol = 1e-6)
    end

    @testset "non-log links and starts inside the cap are left alone" begin
        Yf = Float64.(_Y874) .+ 50.0
        b = fill(35.0, size(Yf, 1))                 # μ = 35 is an ordinary identity-link start
        logs, f = Test.collect_test_logs() do
            fit_gamma_gllvm(Yf; K = 0, link = _G874.IdentityLink(), β_init = b)
        end
        @test !_rescue_warned874(logs)
        @test isfinite(f.loglik)
        b29 = _b874(_Y874); b29[1] = 29.0
        logs, f = Test.collect_test_logs() do; fit_nb_gllvm(_Y874; K = 0, β_init = b29); end
        @test !_rescue_warned874(logs)
    end
end
