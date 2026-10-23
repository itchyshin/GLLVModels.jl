using GLLVModels, Test, TOML, SHA
using SpecialFunctions: loggamma
const GM = GLLVModels

# Issue #515: fit_beta_binomial_gllvm could report converged=true at a
# log-likelihood of about +7.18e54 with the Beta precision φ near 3.3e65 -- a
# numerically wrong (catastrophic loggamma cancellation, see
# betabinomial_logp's _BB_PHI_STABLE note in src/families/beta_binomial.jl)
# outer-optimiser divergence, silently reported as a plausible fit. This file
# covers: (1) the recorded absurd state is rejected by the new verdict helper,
# (2) betabinomial_logp stays numerically correct past the stabilisation
# threshold (checked against a 256-bit BigFloat reference), and (3) fits that
# were already healthy on origin/main keep the same loglik/converged flag.
#
# Both `fit_beta_binomial_gllvm_grouped` and `fit_beta_binomial_gllvm_grouped_cov`
# share the same underlying `betabinomial_logp` numerical exposure and the same
# missing verdict gate, but neither is covered by this file or by this PR's
# fix -- see the PR body.

const BB515_FIXTURE_PATH = joinpath(@__DIR__, "fixtures", "beta_binomial_verdict_515.toml")

@testset "Beta-binomial verdict (#515)" begin

    @testset "fixture integrity" begin
        fixture = TOML.parsefile(BB515_FIXTURE_PATH)
        Y = Int64.(fixture["Y_column_major"])
        N = Int64.(fixture["N_column_major"])
        @test length(Y) == fixture["p"] * fixture["n"]
        @test length(N) == fixture["p"] * fixture["n"]
        @test bytes2hex(sha256(reinterpret(UInt8, Y))) == fixture["data_sha256"]
        @test bytes2hex(sha256(reinterpret(UInt8, N))) == fixture["N_sha256"]
    end

    @testset "verdict rejects the recorded #515 absurd state (direct)" begin
        # On main @ 2847b5dbf, fit_beta_binomial_gllvm on this fixture's data
        # (seed 7005, loading sd 0.9) reported converged=true, loglik ≈
        # +7.18357e54, φ ≈ 3.32173e65 -- issue #515's own evidence table,
        # reproduced and confirmed against 2847b5dbf during this fix's
        # diagnosis. This particular dataset no longer diverges on origin/main
        # (#510's damped per-site search happens to dodge it), so this testset
        # exercises `_beta_binomial_verdict` directly against the literal
        # recorded (nll, φ) state. The live end-to-end case is the next
        # testset, on two datasets that still diverge on origin/main 1385b0490.
        fixture = TOML.parsefile(BB515_FIXTURE_PATH)
        nll = -fixture["main_2847b5dbf_loglik"]
        φ_absurd = fixture["main_2847b5dbf_phi"]
        conv, loglik, reason = GM._beta_binomial_verdict(true, nll, φ_absurd)
        @test conv == false
        @test reason === :objective_impossible
        # The impossible (positive) value is never passed on as a
        # log-likelihood (it would poison AIC/BIC/LRT downstream): -Inf, as
        # `_tweedie_verdict` does for `:objective_failed`.
        @test loglik == -Inf
    end

    @testset "live fit: datasets that diverge on origin/main 1385b0490" begin
        # On origin/main 1385b0490 each of these fits reported converged=true
        # at an impossible loglik: 9149 (1.81e10, φ 2.70e21) and 9187 (1.33e74,
        # φ 6.76e84) on Julia 1.10.12, 2 of 100 datasets at loading sd 4.5 in a
        # wide search (0 of 100 at sd 0.9); 9111 (1.52e33, φ 1.57e44) on Julia
        # 1.13.0, where 9149/9187 happen not to diverge. With the stabilised
        # log-pmf the outer search no longer runs φ into the cancellation
        # region and reaches an ordinary optimum on both versions; the verdict
        # gate is the backstop if it ever does.
        fixture = TOML.parsefile(BB515_FIXTURE_PATH)
        p, n = fixture["p"], fixture["n"]
        for key in ("live_seed_9111", "live_seed_9149", "live_seed_9187")
            case = fixture[key]
            y = Int64.(case["Y_column_major"])
            nn = Int64.(case["N_column_major"])
            @test bytes2hex(sha256(reinterpret(UInt8, y))) == case["data_sha256"]
            @test bytes2hex(sha256(reinterpret(UInt8, nn))) == case["N_sha256"]
            @test case["main_1385b0490_loglik"] > 0  # the recorded red state
            fit = GM.fit_beta_binomial_gllvm(reshape(y, p, n); K = 2,
                                             N = reshape(nn, p, n), iterations = 500)
            # Never an impossible value reported as a converged fit.
            @test !(fit.converged && !(fit.loglik <= GM._BB_LOGLIK_MAX))
            @test !(fit.converged && fit.φ >= GM._BB_PHI_STABLE)
            # And on these datasets the fit is in fact a plausible optimum.
            @test fit.converged
            @test isfinite(fit.loglik) && fit.loglik < 0
            @test 1 < fit.φ < 100
        end
    end

    @testset "_beta_binomial_verdict gates all three failure modes" begin
        # Non-finite nll, or the fail-penalty sentinel this file's own `negll`
        # closures already return on a failed evaluation: never a real value.
        @test GM._beta_binomial_verdict(true, NaN, 5.0) == (false, -Inf, :objective_impossible)
        @test GM._beta_binomial_verdict(true, Inf, 5.0) == (false, -Inf, :objective_impossible)
        @test GM._beta_binomial_verdict(true, 1e12, 5.0) == (false, -Inf, :objective_impossible)
        # A log-likelihood above 0 beyond rounding cannot be genuine.
        @test GM._beta_binomial_verdict(true, -1e-3, 5.0) == (false, -Inf, :objective_impossible)

        # φ at/beyond the stabilisation boundary, otherwise-plausible loglik:
        # reported but marked not converged (mirrors :power_at_boundary).
        conv, loglik, reason = GM._beta_binomial_verdict(true, 500.0, 2e6)
        @test !conv
        @test reason === :phi_at_boundary
        @test loglik ≈ -500.0

        # A genuinely healthy point: Optim's own flag passes straight through.
        conv, loglik, reason = GM._beta_binomial_verdict(true, 300.0, 5.0)
        @test conv && reason === :ok && loglik ≈ -300.0
        conv, loglik, reason = GM._beta_binomial_verdict(false, 300.0, 5.0)
        @test !conv && reason === :ok && loglik ≈ -300.0
    end

    @testset "betabinomial_logp at phi >= _BB_PHI_STABLE matches a BigFloat reference" begin
        # Direct-formula reference at 256-bit precision (same method used to
        # diagnose #515's cancellation: ~2e-10 error at φ=1e6, ~1e-3 at φ=1e12,
        # unbounded beyond that in Float64). betabinomial_logp itself, past the
        # threshold, no longer uses this formula (it returns the exact
        # Binomial(N, μ) limit) -- this test checks that limit is what the
        # BigFloat direct formula actually converges to at these φ.
        function bb_logp_big(y, N, μ, φ)
            setprecision(BigFloat, 256) do
                a = big(μ) * big(φ)
                b = (1 - big(μ)) * big(φ)
                yb = big(y)
                Nb = big(N)
                v = loggamma(a + b) + loggamma(a + yb) + loggamma(b + Nb - yb) -
                    loggamma(a) - loggamma(b) - loggamma(a + b + Nb) +
                    loggamma(Nb + 1) - loggamma(yb + 1) - loggamma(Nb - yb + 1)
                return Float64(v)
            end
        end

        link = LogitLink()
        cases = ((5, 12, 0.4), (0, 10, 0.05), (9, 9, 0.5), (3, 20, 0.8))
        for φ in (1e6, 1e8, 1e10), (y, N, μ) in cases
            η = log(μ / (1 - μ))
            got = GM.betabinomial_logp(y, η, N, φ; link = link)
            ref = bb_logp_big(y, N, μ, φ)
            @test isfinite(got)
            # The true beta-binomial log-pmf itself deviates from the exact
            # Binomial(N, μ) limit by O(1/φ) (the Beta's own variance is
            # μ(1-μ)/(φ+1)) -- measured here at up to ~5e-4 (φ=1e6), ~5e-6
            # (φ=1e8), ~5e-8 (φ=1e10) across these cases. `got` uses the
            # Binomial-limit formula past `_BB_PHI_STABLE`, so this is the
            # tolerance floor, not slack for a bug: a fixed atol would either
            # be too tight at φ=1e6 or too loose to mean anything at φ=1e10.
            @test got ≈ ref atol = 2e-3 * (1e6 / φ)
        end
    end

    @testset "healthy fits are unchanged (>=10, both loading scales)" begin
        # Ten datasets (loading sd 0.9 and 4.5, five each) on which origin/main
        # 1385b0490 reported converged=true at an ordinary optimum, with the
        # same loglik on Julia 1.10.12 and 1.13.0; stored literally because the
        # seeded draw differs across Julia versions. The fix must leave these
        # fits as they were.
        fixture = TOML.parsefile(BB515_FIXTURE_PATH)
        p, n = fixture["p"], fixture["n"]
        keys_ = sort([k for k in keys(fixture) if startswith(k, "healthy_seed_")])
        @test length(keys_) >= 10
        for key in keys_
            case = fixture[key]
            y = Int64.(case["Y_column_major"])
            nn = Int64.(case["N_column_major"])
            @test bytes2hex(sha256(reinterpret(UInt8, y))) == case["data_sha256"]
            @test bytes2hex(sha256(reinterpret(UInt8, nn))) == case["N_sha256"]
            fit = GM.fit_beta_binomial_gllvm(reshape(y, p, n); K = 2,
                                             N = reshape(nn, p, n), iterations = 500)
            @test fit.converged
            @test fit.loglik ≈ case["main_1385b0490_loglik"] atol = 1e-3
            @test fit.φ ≈ case["main_1385b0490_phi"] atol = 1e-2
        end
    end
end
