# gllvm-parity-tag: P1
#
# R twin test for chibar2_pvalue()/variance_lrt() (src/boundary_inference.jl) against
# gllvmTMB's R/chibar.R at pin P1 (9539352f66f2db2cc26b1c393e67212a359b60c9).
#
# WHY THIS TEST EXISTS. gllvmTMB's chibar.R says, in its own header, that it is "a
# direct, faithful port" of these two GLLVModels.jl functions -- same closed-form
# Self & Liang (1987) / Stram & Lee (1994) chi-bar-square mixture weights, same scope
# note (independent boundary variance components, regular Fisher information
# elsewhere), same argument names (`LRT`, `q`; `ll_full`, `ll_reduced`, `n_boundary`).
# D-297's P1 case map still classes the pair `semantic_divergence` until a twin test
# proves equal outputs on real inputs -- a name match alone never counts. This file is
# that proof: every fixture value below was computed by calling the ACTUAL installed
# gllvmTMB package at the P1 pin (see fixtures/generate_chibar2_p1_fixture.R), not
# retyped from the R source, so an accidental drift in either side's formula shows up
# as a numeric mismatch here.
#
# SCOPE. Both are closed-form (chi-square survival function evaluations, or a scalar
# affine transform of two log-likelihoods) -- no fitting, no RNG, no RCall needed at
# test time. The tolerance is 1e-12, matching that closed-form character.
#
# ONE DOCUMENTED DIVERGENCE (not a mixture-weight/substance difference, so it does not
# block twinning per D-297): gllvmTMB's `chibar2_pvalue()`/`variance_lrt()` explicitly
# reject a non-numeric or `NA` log-likelihood/LRT (classed errors
# `gllvmTMB_chibar2_bad_LRT` / `gllvmTMB_variance_lrt_bad_loglik`). GLLVModels.jl's
# `chibar2_pvalue(LRT::Real, q::Integer)` does not special-case `NaN`: `LRT > 0` is
# `false` for `NaN`, so it silently returns a p-value of 1.0 instead of erroring. See
# the "documented divergence" testset below -- it records current behaviour, it does
# not assert parity with R for that one input class.

using GLLVModels, Test, TOML, SHA

const _CHIBAR2_P1_FIXTURE = joinpath(@__DIR__, "parity", "fixtures", "chibar2_p1_fixture.toml")
const _CHIBAR2_P1_FIXTURE_SHA256 =
    "eefc9095fd49b6692412db8fda8c85bf4836100eaf23e8db22cefa46c3a4b640"

@testset "chibar2_pvalue / variance_lrt vs gllvmTMB R/chibar.R @ P1" begin

    @testset "fixture integrity" begin
        @test isfile(_CHIBAR2_P1_FIXTURE)
        raw = read(_CHIBAR2_P1_FIXTURE)
        @test bytes2hex(sha256(raw)) == _CHIBAR2_P1_FIXTURE_SHA256
    end

    fixture = TOML.parsefile(_CHIBAR2_P1_FIXTURE)
    @test fixture["gllvmtmb_pin_sha"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"

    @testset "chibar2_pvalue(LRT, q) matches gllvmTMB to 1e-12" begin
        for row in fixture["chibar2_pvalue"]
            LRT, q, pvalue = row["LRT"], row["q"], row["pvalue"]
            @test chibar2_pvalue(LRT, q) ≈ pvalue atol = 1e-12
        end
    end

    @testset "variance_lrt() matches gllvmTMB to 1e-12" begin
        for row in fixture["variance_lrt"]
            r = variance_lrt(row["ll_full"], row["ll_reduced"]; n_boundary = row["n_boundary"])
            @test r.LRT ≈ row["LRT"] atol = 1e-12
            @test r.pvalue ≈ row["pvalue"] atol = 1e-12
            @test r.n_boundary == row["n_boundary"]
        end
    end

    @testset "refusals gllvmTMB and GLLVModels.jl both reach" begin
        # q must be >= 1 (gllvmTMB: class gllvmTMB_chibar2_bad_q).
        @test_throws ArgumentError chibar2_pvalue(3.84, 0)
        @test_throws ArgumentError chibar2_pvalue(3.84, -1)
        # gllvmTMB also refuses a non-integer q under the same class; Julia's method
        # signature (`q::Integer`) enforces the identical constraint one level earlier,
        # at dispatch, as a MethodError rather than the library's own ArgumentError.
        @test_throws MethodError chibar2_pvalue(3.84, 1.5)
    end

    @testset "documented divergence: NaN log-likelihood / LRT (not twinned)" begin
        # gllvmTMB errors (class gllvmTMB_chibar2_bad_LRT / gllvmTMB_variance_lrt_bad_loglik)
        # on a missing/NaN LRT or log-likelihood. GLLVModels.jl does not special-case NaN
        # here: `NaN > 0` is `false`, so both functions silently report "no evidence"
        # (p-value 1.0) instead of refusing. This test records that ACTUAL behaviour; it
        # is a known, minor divergence in input validation, not in the mixture formula
        # itself, and is out of scope for this twin (see file header).
        @test chibar2_pvalue(NaN, 1) == 1.0
        @test variance_lrt(NaN, -102.0).pvalue == 1.0
    end
end
