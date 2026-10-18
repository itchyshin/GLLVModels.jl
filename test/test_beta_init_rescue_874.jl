# #874: a user-supplied β_init entry above the η clamp (30) starts a log-link fit on the failure
# sentinel; since #873 it ended at loglik = -Inf, converged = false. The entry now falls back to
# the default warm start (with a warning), so the fit reaches the same optimum as the default start.
# Data: 6 species of mvabund::spider$abund (28 sites), shared with test_eta_clamp_869.jl.
using Test, GLLVModels
const _G874 = GLLVModels

const _Y874 = [
    25 0 15 2 1 0 2 0 1 3 15 16 3 0 0 0 0 0 0 0 0 7 17 11 9 3 29 15;
    10 2 20 6 20 6 7 11 1 0 1 13 43 2 0 3 0 1 1 2 1 0 0 0 1 0 0 0;
    4 30 9 24 9 6 16 7 0 0 1 0 18 4 0 0 0 0 0 0 0 0 0 0 2 0 0 0;
    60 1 29 7 2 11 30 2 26 22 95 96 24 14 0 0 0 0 0 0 1 2 6 3 11 0 1 6;
    45 37 45 94 76 24 105 1 1 0 1 8 72 72 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
    57 65 66 86 91 63 118 30 2 1 4 13 97 94 25 28 23 25 22 22 18 1 1 0 16 1 0 2]
const _P874 = size(_Y874, 1)

@testset "#874 β_init above the η clamp is rescued" begin
    Yi = _Y874
    Yf = Float64.(_Y874) .+ 0.5
    Ypos = max.(_Y874, 1)
    cases = [
        ("fit_poisson_gllvm",          Yi,   (Y; kw...) -> fit_poisson_gllvm(Y; K = 1, kw...)),
        ("fit_nb_gllvm",               Yi,   (Y; kw...) -> fit_nb_gllvm(Y; K = 1, kw...)),
        ("fit_nb1_gllvm",              Yi,   (Y; kw...) -> fit_nb1_gllvm(Y; K = 1, kw...)),
        ("fit_gp1_gllvm",              Yi,   (Y; kw...) -> fit_gp1_gllvm(Y; K = 1, kw...)),
        ("fit_compoisson_gllvm",       Yi,   (Y; kw...) -> fit_compoisson_gllvm(Y; K = 0, kw...)),   # K = 0: the K = 1 fit takes ~20 min
        ("fit_truncated_poisson_gllvm", Ypos, (Y; kw...) -> fit_truncated_poisson_gllvm(Y; K = 0, kw...)),   # K = 0: K = 1 has local optima, so the start matters
        ("fit_truncated_nbinom2_gllvm", Ypos, (Y; kw...) -> fit_truncated_nbinom2_gllvm(Y; K = 1, kw...)),
        ("fit_censored_poisson_gllvm", Yi,   (Y; kw...) -> fit_censored_poisson_gllvm(Y; K = 1, kw...)),
        ("fit_tweedie_gllvm",          Float64.(Yi), (Y; kw...) -> fit_tweedie_gllvm(Y; K = 1, kw...)),
        ("fit_gamma_gllvm",            Yf,   (Y; kw...) -> fit_gamma_gllvm(Y; K = 1, kw...)),
        ("fit_exponential_gllvm",      Yf,   (Y; kw...) -> fit_exponential_gllvm(Y; K = 1, kw...)),
    ]
    for (nm, Y, fit) in cases
        @testset "$nm" begin
            base = redirect_stderr(devnull) do; fit(Y); end
            b0 = copy(base.β); b0[1] = 35.0
            f = redirect_stderr(devnull) do; fit(Y; β_init = b0); end
            @test f.converged == base.converged
            @test isfinite(f.loglik)
            @test f.loglik ≈ base.loglik rtol = 1e-6
        end
    end

    @testset "warns once and names the entry" begin
        b0 = zeros(_P874); b0[2] = 35.0
        @test_logs (:warn, r"β_init entries \[2\] sit above the η clamp") fit_poisson_gllvm(_Y874; K = 1, β_init = b0)
    end

    @testset "offset counts toward the cap" begin
        off = fill(5.0, _P874, size(_Y874, 2))
        b0 = fill(26.0, _P874)
        f = redirect_stderr(devnull) do; fit_poisson_gllvm(_Y874; K = 1, offset = off, β_init = b0); end
        @test isfinite(f.loglik)
    end

    @testset "in-range β_init and non-log links are untouched" begin
        b0 = fill(1.0, _P874)
        @test _G874._rescue_β_init(b0, zeros(_P874), nothing, _G874.LogLink()) == b0
        b1 = fill(40.0, _P874)
        @test _G874._rescue_β_init(b1, zeros(_P874), nothing, _G874.LogitLink()) == b1
        @test _G874._rescue_β_init(nothing, zeros(_P874), nothing, _G874.LogLink()) == zeros(_P874)
    end
end
