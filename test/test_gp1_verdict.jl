using GLLVModels, Test, TOML, SHA
using SpecialFunctions: loggamma
const GM = GLLVModels

# GP-1 verdict fix. On origin/main 863ee0f78, fit_gp1_gllvm on otherwise healthy
# GP-1 data (p = 4, n = 120, K = 1) with ONE cell set to the count 10^18 reported
# converged = true at loglik = +6795.99 (Julia 1.10.12) and +4939.22 (Julia
# 1.13.0). A Laplace log-marginal over a probability mass function cannot be
# positive. Cause: the direct GP-1 log-pmf subtracts terms of size y*log(y)
# (about 4e19 at y = 10^18), so its Float64 value near the per-site mode is
# rounding noise (measured +9216 where the 256-bit value is -59.8). This file
# covers: (1) the new _gp1_verdict helper rejects the recorded absurd states,
# (2) a live refit of the fixture never reports a converged positive loglik,
# (3) the log-pmf at huge y matches a 256-bit BigFloat reference, and (4) six
# healthy fits keep their origin/main loglik and converged flag.

const GP1V_FIXTURE_PATH = joinpath(@__DIR__, "fixtures", "gp1_verdict.toml")

_gp1v_sha(Y) = bytes2hex(sha256(reinterpret(UInt8, vec(Int64.(Y)))))

@testset "GP-1 verdict (huge-count log-pmf and impossible loglik)" begin

    fixture = TOML.parsefile(GP1V_FIXTURE_PATH)
    p, n, K = fixture["p"], fixture["n"], fixture["K"]
    healthy_keys = sort([k for k in keys(fixture) if startswith(k, "healthy_seed_")])

    @testset "fixture integrity" begin
        Y = Int64.(fixture["Y_column_major"])
        @test length(Y) == p * n
        @test _gp1v_sha(Y) == fixture["data_sha256"]
        @test maximum(Y) == 10^18
        @test length(healthy_keys) >= 5
        for key in healthy_keys
            y = Int64.(fixture[key]["Y_column_major"])
            @test length(y) == p * n
            @test _gp1v_sha(y) == fixture[key]["data_sha256"]
        end
    end

    @testset "verdict rejects the recorded absurd states (direct)" begin
        for key in ("main_julia_1_10_loglik", "main_julia_1_13_loglik")
            ll_main = fixture[key]
            @test ll_main > 0                    # the recorded red state
            conv, loglik, reason = GM._gp1_verdict(true, -ll_main)
            @test conv == false
            @test loglik == -Inf
            @test reason === :objective_impossible
        end
    end

    @testset "_gp1_verdict gates every failure mode and passes healthy points" begin
        @test GM._gp1_verdict(true, NaN) == (false, -Inf, :objective_impossible)
        @test GM._gp1_verdict(true, Inf) == (false, -Inf, :objective_impossible)
        @test GM._gp1_verdict(true, 1e12) == (false, -Inf, :objective_impossible)
        @test GM._gp1_verdict(true, 2e11) == (false, -Inf, :objective_impossible)
        @test GM._gp1_verdict(true, -1e-3) == (false, -Inf, :objective_impossible)
        @test GM._gp1_verdict(true, -1e4) == (false, -Inf, :objective_impossible)
        conv, loglik, reason = GM._gp1_verdict(true, 1259.0)
        @test conv && reason === :ok && loglik == -1259.0
        conv, loglik, reason = GM._gp1_verdict(false, 1259.0)
        @test !conv && reason === :ok && loglik == -1259.0
    end

    @testset "live refit never reports a converged positive loglik" begin
        Y = reshape(Int64.(fixture["Y_column_major"]), p, n)
        fit = GM.fit_gp1_gllvm(Y; K = K)
        @test !(fit.converged && !(fit.loglik <= GM._GP1_LOGLIK_MAX))
        @test !(fit.loglik > GM._GP1_LOGLIK_MAX)
        # With the stable log-pmf the fitted value is an ordinary negative number.
        @test isfinite(fit.loglik) && fit.loglik < 0
    end

    @testset "log-pmf at huge y matches a 256-bit BigFloat reference" begin
        function gp1_logpmf_big(α, μ, y)
            setprecision(BigFloat, 256) do
                a = big(α); m = big(μ); yb = big(y)
                g = 1 + a * m
                h = 1 + a * yb
                v = yb * (log(m) - log(g)) + (yb - 1) * log(h) - loggamma(yb + 1) - m * h / g
                return Float64(v)
            end
        end
        for α in (0.0360679774997897, 0.2, 1.0),
            μ in (1.0, 3.0, 1e6, 8.225841571505276e10, exp(30.0)),
            y in (10^6, 10^8, 10^12, 10^15, 10^18)
            got = GM._glm_logpdf(GM.GeneralizedPoisson1(α), μ, 1, y)
            ref = gp1_logpmf_big(α, μ, y)
            @test isfinite(got)
            @test got <= 0
            # rtol covers the huge-magnitude cells (up to ~4e19 in size); atol
            # covers the cells near the per-site mode, where the true value is
            # O(log y) and the direct formula was off by thousands.
            @test isapprox(got, ref; rtol = 1e-12, atol = 1e-6)
        end
        # The measured diagnosis point: +9216.0 on origin/main, -59.8232 in BigFloat.
        α0, μ0 = 0.0360679774997897, 8.225841571505276e10
        @test GM._glm_logpdf(GM.GeneralizedPoisson1(α0), μ0, 1, 10^18) ≈
              gp1_logpmf_big(α0, μ0, 10^18) atol = 1e-6
        # Just below the switch point the direct formula is still used and still
        # accurate against the same reference.
        for y in (10, 1000, 10^5, 999_999), μ in (3.0, 1e3, 1e6)
            got = GM._glm_logpdf(GM.GeneralizedPoisson1(0.2), μ, 1, y)
            @test isapprox(got, gp1_logpmf_big(0.2, μ, y); rtol = 1e-12, atol = 1e-6)
        end
    end

    @testset "healthy fits are unchanged vs origin/main (per Julia version)" begin
        vkey = "main_julia_1_$(VERSION.minor)"
        # The origin/main logliks were measured on macOS aarch64. Linux CI reaches
        # a different optimum on some seeds (seed 101: -1256.35 vs -1259.00 on
        # Julia 1.10), so the literal record only binds where it was measured.
        # Everywhere, the new log-pmf branch must be unreachable on this data, so
        # the fix cannot move these fits on any platform.
        on_record_platform = Sys.isapple() && Sys.ARCH === :aarch64
        for key in healthy_keys
            case = fixture[key]
            Y = reshape(Int64.(case["Y_column_major"]), p, n)
            @test maximum(Y) < GM._GP1_Y_STABLE
            fit = GM.fit_gp1_gllvm(Y; K = K)
            if on_record_platform && haskey(case, vkey * "_loglik")
                @test fit.converged == case[vkey * "_converged"]
                @test fit.loglik ≈ case[vkey * "_loglik"] atol = 1e-8
            else
                # No origin/main record for this Julia minor version or platform:
                # the optimum is version- and platform-dependent (seeds 101 and
                # 104 differ between 1.10 and 1.13), so only the plausibility of
                # the fit is checked.
                @test fit.converged
                @test isfinite(fit.loglik) && fit.loglik < 0
            end
        end
    end
end
