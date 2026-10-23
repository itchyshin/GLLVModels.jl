using GLLVModels, Test, TOML, SHA
const GM = GLLVModels

# Follow-up to issue #515 (PR #522). #522 gave fit_beta_binomial_gllvm a
# per-family verdict (`_beta_binomial_verdict`); the grouped routes
# fit_beta_binomial_gllvm_grouped and fit_beta_binomial_gllvm_grouped_cov got
# the stabilised log-pmf but kept the shared `_fit_verdict`, so a per-species
# fit could still report converged=true with a group's Beta precision far past
# the 1e6 stabilisation boundary. This file reuses #522's hash-verified
# literal fixture (test/fixtures/beta_binomial_verdict_515.toml) and covers:
# (1) the grouped verdict applied to recorded absurd states, (2) live grouped
# fits that reach the boundary on origin/main 52ed4281b, and (3) healthy
# grouped fits that must keep their loglik and converged flag.

const BBG515_FIXTURE_PATH = joinpath(@__DIR__, "fixtures", "beta_binomial_verdict_515.toml")

# Deterministic literal covariate for the grouped_cov route (no RNG, so it is
# the same on every Julia version).
_bbg515_X(p, n) = reshape(Float64[((7s + 3t) % 11) / 5 - 1 for t in 1:p, s in 1:n], p, n, 1)

function _bbg515_case(fixture, key)
    p, n = fixture["p"], fixture["n"]
    case = fixture[key]
    y = Int64.(case["Y_column_major"])
    nn = Int64.(case["N_column_major"])
    @test bytes2hex(sha256(reinterpret(UInt8, y))) == case["data_sha256"]
    @test bytes2hex(sha256(reinterpret(UInt8, nn))) == case["N_sha256"]
    return reshape(y, p, n), reshape(nn, p, n)
end

@testset "Beta-binomial grouped verdict (#515 follow-up)" begin

    @testset "_beta_binomial_grouped_verdict on recorded absurd states" begin
        fixture = TOML.parsefile(BBG515_FIXTURE_PATH)
        # #515's recorded state (main 2847b5dbf, ungrouped): loglik +7.18e54 at
        # φ 3.32e65. Placed in one group of an otherwise ordinary φ vector.
        nll = -fixture["main_2847b5dbf_loglik"]
        φg = [12.0, fixture["main_2847b5dbf_phi"], 8.0]
        @test GM._beta_binomial_grouped_verdict(nll, true, 57, φg) == (-Inf, false, 57)
        # A positive loglik is rejected even when every φ is ordinary.
        @test GM._beta_binomial_grouped_verdict(-1e-3, true, 5, [12.0, 8.0]) == (-Inf, false, 5)
        # The shared `_fit_verdict` plateau screen (nll >= 1e11) still applies:
        # the grouped verdict is strictly stronger, never weaker.
        @test GM._beta_binomial_grouped_verdict(5e11, true, 0, [12.0]) == (-Inf, false, 0)
        @test GM._beta_binomial_grouped_verdict(NaN, true, 3, [12.0]) == (-Inf, false, 3)
        # One group at the boundary: loglik kept, converged forced false.
        ll, conv, it = GM._beta_binomial_grouped_verdict(1614.07, true, 90, [25.0, 7.6e15, 10.0])
        @test ll ≈ -1614.07 && !conv && it == 90
        # Healthy: Optim's flag and the loglik pass straight through.
        @test GM._beta_binomial_grouped_verdict(300.0, true, 40, [12.0, 9.0]) == (-300.0, true, 40)
        @test GM._beta_binomial_grouped_verdict(300.0, false, 500, [12.0, 9.0]) == (-300.0, false, 500)
    end

    @testset "live: per-species fits that reach the phi boundary" begin
        # On origin/main 52ed4281b each of these per-species (group = 1:p) fits
        # of genuine beta-binomial data (φ_true = 12) reported converged=true
        # with one species' φ past `_BB_PHI_STABLE`, on both Julia 1.10.12 and
        # 1.13.0 (aarch64-apple-darwin, 2026-09-27):
        #   healthy_seed_9003 grouped      φ_max 7.63e15 (1.10) / 4.10e15 (1.13)
        #   healthy_seed_9003 grouped_cov  φ_max 8.54e11 (1.10) / 1.07e12 (1.13)
        #   healthy_seed_9103 grouped_cov  φ_max 2.93e12 (1.10) / 8.59e11 (1.13)
        # At that φ the log-pmf is exactly Binomial (flat in φ), so Optim's
        # zero-gradient stop is not evidence of an optimum in φ. The verdict
        # keeps the loglik (it is a genuine value) but reports converged=false.
        fixture = TOML.parsefile(BBG515_FIXTURE_PATH)
        p, n = fixture["p"], fixture["n"]
        X = _bbg515_X(p, n)
        cases = (("healthy_seed_9003", :grouped, -1614.0736190046484),
                 ("healthy_seed_9003", :grouped_cov, -1613.7418188600086),
                 ("healthy_seed_9103", :grouped_cov, -1189.3083638717906))
        for (key, route, ll_main) in cases
            y, nn = _bbg515_case(fixture, key)
            fit = route === :grouped ?
                GM.fit_beta_binomial_gllvm_grouped(y; K = 2, N = nn, group = collect(1:p),
                                                   iterations = 500) :
                GM.fit_beta_binomial_gllvm_grouped_cov(y; X = X, K = 2, N = nn,
                                                       group = collect(1:p), iterations = 500)
            @test maximum(fit.φ) >= GM._BB_PHI_STABLE   # the recorded state is reached
            @test !fit.converged                        # ... and no longer reported as converged
            @test fit.loglik ≈ ll_main atol = 1e-3      # the loglik itself is unchanged
        end
    end

    @testset "healthy grouped fits are unchanged (>=10, both routes)" begin
        # Twelve fits on which origin/main 52ed4281b reported converged=true
        # with every φ well below `_BB_PHI_STABLE` on both Julia 1.10.12 and
        # 1.13.0; expected logliks are the 1.10.12 values (1.13.0 agrees to
        # better than 2e-11 on all twelve). Fast single-group fits dominate to
        # keep this file light; the per-species ones have φ_max below 130.
        fixture = TOML.parsefile(BBG515_FIXTURE_PATH)
        p, n = fixture["p"], fixture["n"]
        X = _bbg515_X(p, n)
        one_ = ones(Int, p); pers = collect(1:p)
        cases = (("healthy_seed_9001", :grouped, one_, -1558.5437810832875),
                 ("healthy_seed_9002", :grouped, one_, -1624.6712944300982),
                 ("healthy_seed_9003", :grouped, one_, -1618.9694099727433),
                 ("healthy_seed_9004", :grouped, one_, -1581.7904815549573),
                 ("healthy_seed_9005", :grouped, one_, -1589.391737922303),
                 ("healthy_seed_9001", :grouped_cov, one_, -1558.1750818258242),
                 ("healthy_seed_9003", :grouped_cov, one_, -1618.5261054230882),
                 ("healthy_seed_9004", :grouped_cov, one_, -1579.6474311293991),
                 ("healthy_seed_9004", :grouped, pers, -1577.0693716306068),
                 ("healthy_seed_9005", :grouped, pers, -1588.036045120239),
                 ("healthy_seed_9001", :grouped_cov, pers, -1557.4048373954795),
                 ("healthy_seed_9005", :grouped_cov, pers, -1586.8245593362308))
        @test length(cases) >= 10
        for (key, route, grp, ll_main) in cases
            y, nn = _bbg515_case(fixture, key)
            fit = route === :grouped ?
                GM.fit_beta_binomial_gllvm_grouped(y; K = 2, N = nn, group = grp,
                                                   iterations = 500) :
                GM.fit_beta_binomial_gllvm_grouped_cov(y; X = X, K = 2, N = nn,
                                                       group = grp, iterations = 500)
            @test fit.converged
            @test maximum(fit.φ) < GM._BB_PHI_STABLE
            @test fit.loglik ≈ ll_main atol = 1e-6
        end
    end
end
