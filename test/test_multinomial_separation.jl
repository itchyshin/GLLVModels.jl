using GLLVModels, Test, TOML, SHA
const GM = GLLVModels

# Multinomial complete-separation verdict. On origin/main 0ce4a35aa,
# fit_multinomial_gllvm on completely separated data (n = 12, K = 3, one covariate
# that orders the categories) reported converged = true at loglik = -1.19e-5 while the
# slopes ran off toward infinity (max |theta| = 66.0 after 22 iterations; Julia 1.10.12
# and 1.13.0). Under complete separation no finite MLE exists, so "converged" is false
# by definition. This file covers: (1) the new _multinomial_verdict helper gates the
# recorded state and passes healthy values, (2) a live refit of the fixture reports
# converged = false with the loglik left as computed, and (3) six healthy fits keep
# their origin/main loglik and converged flag.

const MNSEP_FIXTURE_PATH = joinpath(@__DIR__, "fixtures", "multinomial_separation.toml")

_mnsep_sha(y, x) = bytes2hex(sha256(vcat(reinterpret(UInt8, Vector{Int64}(y)),
                                         reinterpret(UInt8, Vector{Float64}(x)))))

function _mnsep_fit(case)
    n, p = case["n"], case["p"]
    X = p == 0 ? nothing : reshape(Float64.(case["X_column_major"]), n, p)
    return GM.fit_multinomial_gllvm(reshape(Int.(case["y"]), 1, n); X = X,
                                    n_categories = case["n_categories"])
end

@testset "multinomial verdict (complete separation)" begin

    fixture = TOML.parsefile(MNSEP_FIXTURE_PATH)
    healthy_keys = sort([k for k in keys(fixture) if startswith(k, "healthy_seed_")])

    @testset "fixture integrity" begin
        @test _mnsep_sha(fixture["y"], fixture["X_column_major"]) == fixture["data_sha256"]
        @test length(fixture["y"]) == fixture["n"]
        @test length(healthy_keys) >= 5
        for key in healthy_keys
            case = fixture[key]
            @test length(case["y"]) == case["n"]
            @test length(case["X_column_major"]) == case["n"] * case["p"]
            @test _mnsep_sha(case["y"], case["X_column_major"]) == case["data_sha256"]
        end
    end

    @testset "_multinomial_verdict gates separation and passes healthy points" begin
        τ = GM._MN_SEPARATION_NLL
        # The recorded red state: the total nll bounds every per-observation term.
        for key in ("main_julia_1_10_loglik", "main_julia_1_13_loglik")
            ll_main = fixture[key]
            @test -ll_main < τ
            conv, loglik, reason = GM._multinomial_verdict(true, -ll_main, -ll_main)
            @test conv == false
            @test loglik == ll_main          # loglik stays as computed
            @test reason === :separation
        end
        @test GM._multinomial_verdict(true, NaN, 0.0) == (false, -Inf, :objective_impossible)
        @test GM._multinomial_verdict(true, 1e12, 0.0) == (false, -Inf, :objective_impossible)
        @test GM._multinomial_verdict(true, 206.8, 1.83) == (true, -206.8, :ok)
        @test GM._multinomial_verdict(false, 206.8, 1.83) == (false, -206.8, :ok)
        # Boundary: one observation fitted just short of the threshold is not flagged.
        @test GM._multinomial_verdict(true, 5.0, 2τ)[3] === :ok
        @test GM._multinomial_verdict(true, 5.0, τ / 2)[3] === :separation
    end

    @testset "live refit of separated data reports converged = false" begin
        fit = _mnsep_fit(fixture)
        @test fit.converged == false
        # The loglik is reported as computed: finite, negative, at the saturated bound.
        @test isfinite(fit.loglik)
        @test -GM._MN_SEPARATION_NLL < fit.loglik < 0
        @test occursin("NOT CONVERGED", sprint(show, fit))
    end

    @testset "healthy fits are unchanged vs origin/main (per Julia version)" begin
        vkey = "main_julia_1_$(VERSION.minor)"
        # The origin/main logliks were measured on macOS aarch64; the literal record
        # binds only there. Everywhere, a healthy fit must stay converged and finite.
        on_record_platform = Sys.isapple() && Sys.ARCH === :aarch64
        for key in healthy_keys
            case = fixture[key]
            fit = _mnsep_fit(case)
            @test fit.converged
            @test isfinite(fit.loglik) && fit.loglik < 0
            if on_record_platform && haskey(case, vkey * "_loglik")
                @test fit.converged == case[vkey * "_converged"]
                @test fit.loglik == case[vkey * "_loglik"]
            end
        end
    end
end
