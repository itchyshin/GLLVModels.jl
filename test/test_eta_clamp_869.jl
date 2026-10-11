# #869 items 1-2: log-link count/positive fitters must not report converged = true on the
# eta-clamp plateau. `_clamp_eta` caps eta at +-30, so once an intercept (plus offset) is above
# the cap the objective is exactly flat in it. Before this change, data on a scale where the
# MLE intercept is past the cap (counts ~1e14) made 14 of the fitters below report
# converged = true at a clamped, meaningless loglik; now they end at the failure sentinel
# (loglik = -Inf, converged = false). Data: 6 species of mvabund::spider$abund (28 sites).
using Test, GLLVModels, Distributions
const _G869 = GLLVModels

const _Y869 = [
    25 0 15 2 1 0 2 0 1 3 15 16 3 0 0 0 0 0 0 0 0 7 17 11 9 3 29 15;
    10 2 20 6 20 6 7 11 1 0 1 13 43 2 0 3 0 1 1 2 1 0 0 0 1 0 0 0;
    4 30 9 24 9 6 16 7 0 0 1 0 18 4 0 0 0 0 0 0 0 0 0 0 2 0 0 0;
    60 1 29 7 2 11 30 2 26 22 95 96 24 14 0 0 0 0 0 0 1 2 6 3 11 0 1 6;
    45 37 45 94 76 24 105 1 1 0 1 8 72 72 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
    57 65 66 86 91 63 118 30 2 1 4 13 97 94 25 28 23 25 22 22 18 1 1 0 16 1 0 2]
const _P869 = size(_Y869, 1)
const _X869 = reshape([sin(0.7 * (t + 3i)) for t in 1:_P869, i in 1:size(_Y869, 2)], _P869, size(_Y869, 2), 1)
const _BIG869 = 1e14

# A false positive is converged = true with a non-finite or clamped loglik; both must now be off.
_off869(f) = !f.converged

@testset "#869 eta-clamp plateau and cap-contact verdict" begin
    @testset "helpers" begin
        @test _G869._eta_cap_guarded(_G869.LogLink())
        @test !_G869._eta_cap_guarded(_G869.IdentityLink())
        @test !_G869._eta_cap_guarded(_G869.LogitLink())
        @test _G869._eta_above_clamp([31.0, 1.0], nothing)
        @test !_G869._eta_above_clamp([29.0, 1.0], nothing)
        @test _G869._eta_above_clamp([29.0, 1.0], [2.0 0.0; 0.0 0.0])         # offset pushes it over
        @test !_G869._eta_above_clamp([29.0], [Inf 0.0]; mask = [false true])  # unobserved +Inf ignored
        @test !_G869._eta_above_clamp([29.0], [NaN -Inf])
        @test _G869._eta_offset_max(zeros(2, 0), 1) == -Inf                    # p x 0 does not throw
        # contact verdict: a converged fit within the FD step of the cap is overturned (with a
        # warning), on log links only
        @test_logs (:warn, r"η clamp") !_G869._eta_cap_verdict(true, _G869.LogLink(), [29.995], nothing; caller = "t")
        @test _G869._eta_cap_verdict(true, _G869.LogLink(), [29.0], nothing)
        @test _G869._eta_cap_verdict(true, _G869.IdentityLink(), [29.995], nothing)
        @test !_G869._eta_cap_verdict(false, _G869.LogLink(), [1.0], nothing)
    end

    @testset "scale-1e14 data is not reported as converged" begin
        Yi = round.(Int, _Y869 .* _BIG869)
        Yf = Float64.(_Y869) .* _BIG869 .+ 0.5
        Ynn = Float64.(_Y869) .* _BIG869
        for (nm, thunk) in [
            ("fit_nb_gllvm",              () -> fit_nb_gllvm(Yi; K = 0)),
            ("fit_gp1_gllvm",             () -> fit_gp1_gllvm(Yi; K = 0)),
            ("fit_truncated_nbinom2_gllvm", () -> fit_truncated_nbinom2_gllvm(max.(Yi, 1); K = 0)),
            ("fit_zinb_gllvm",            () -> fit_zinb_gllvm(Yi; K = 0)),
            ("fit_hurdle_nb_gllvm",       () -> fit_hurdle_nb_gllvm(Yi; K = 0)),
            ("fit_gllvm_cov (NB)",        () -> fit_gllvm_cov(Yi; X = _X869, K = 0, family = _G869.NegativeBinomial())),
            ("fit_tweedie_gllvm",         () -> fit_tweedie_gllvm(Ynn; K = 0)),
            ("fit_tweedie_gllvm_grouped", () -> fit_tweedie_gllvm_grouped(Ynn; K = 0)),
            ("fit_gamma_gllvm",           () -> fit_gamma_gllvm(Yf; K = 0)),
            ("fit_gamma_gllvm_grouped",   () -> fit_gamma_gllvm_grouped(Yf; K = 0)),
            ("fit_exponential_gllvm",     () -> fit_exponential_gllvm(Yf; K = 0)),
            ("fit_delta_gamma_gllvm",     () -> fit_delta_gamma_gllvm(Ynn; K = 0)),
            ("fit_delta_lognormal_gllvm", () -> fit_delta_lognormal_gllvm(Ynn; K = 0)),
        ]
            f = redirect_stderr(devnull) do; thunk(); end
            @test (nm, _off869(f)) == (nm, true)
        end
    end

    @testset "a start past the cap is refused, not reported converged (K = 0, Gamma / Tweedie)" begin
        Yf = Float64.(_Y869) .+ 0.5
        b0 = log.(vec(sum(Yf; dims = 2)) ./ size(Yf, 2)); b0[1] = 35.0
        for thunk in (() -> fit_gamma_gllvm(Yf; K = 0, β_init = b0),
                      () -> fit_tweedie_gllvm(Float64.(_Y869); K = 0, β_init = b0))
            f = redirect_stderr(devnull) do; thunk(); end
            @test !f.converged
            @test f.loglik == -Inf
        end
    end

    @testset "fits that sit well inside the cap are unchanged" begin
        Yi = _Y869
        @test fit_nb_gllvm(Yi; K = 0).converged
        @test fit_gamma_gllvm(Float64.(Yi) .+ 0.5; K = 0).converged
        @test fit_poisson_gllvm(Yi; K = 0).converged
        @test fit_zinb_gllvm(Yi; K = 0).converged
        # an IdentityLink Gamma fit at a large intercept must not be touched by the log-link guard
        f = fit_gamma_gllvm(Float64.(Yi) .+ 50.0; K = 0, link = _G869.IdentityLink())
        @test isfinite(f.loglik)
    end
end
