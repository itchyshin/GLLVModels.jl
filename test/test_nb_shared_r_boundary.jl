using GLLVModels, Test, TOML, SHA
const GM = GLLVModels

# Shared-r NB2 boundary verdict. On origin/main 0ce4a35aa, fit_nb_gllvm (one shared
# dispersion r) on Poisson data (p = 5, n = 60, K = 1) reported converged = true with
# r = 1.712e7 (Julia 1.10.12) and r = 1.551e7 (Julia 1.13.0): the Poisson limit, where
# the likelihood is flat in log r and the optimizer's flag says nothing about r. The
# grouped NB fitters already force converged = false once any r_group leaves
# [1e-6, 1e6] (`_dispersion_group_boundary`, grouped_dispersion.jl); this file checks
# the shared-r fitter now does the same, with the loglik left as computed, and that
# six healthy NB2 fits (r_true 2 to 5) keep their origin/main loglik and flag.

const NBSR_FIXTURE_PATH = joinpath(@__DIR__, "fixtures", "nb_shared_r_boundary.toml")

_nbsr_sha(Y) = bytes2hex(sha256(reinterpret(UInt8, vec(Int64.(Y)))))

@testset "shared-r NB2 boundary verdict (Poisson limit)" begin

    fixture = TOML.parsefile(NBSR_FIXTURE_PATH)
    p, n, K = fixture["p"], fixture["n"], fixture["K"]
    healthy_keys = sort([k for k in keys(fixture) if startswith(k, "healthy_seed_")])
    vkey = "main_julia_1_$(VERSION.minor)"
    # The origin/main records were measured on macOS aarch64; Linux CI can reach a
    # different point on the flat log r ridge, so literal values bind only there.
    on_record_platform = Sys.isapple() && Sys.ARCH === :aarch64

    @testset "fixture integrity" begin
        Y = Int64.(fixture["Y_column_major"])
        @test length(Y) == p * n
        @test _nbsr_sha(Y) == fixture["data_sha256"]
        @test length(healthy_keys) >= 5
        for key in healthy_keys
            y = Int64.(fixture[key]["Y_column_major"])
            @test length(y) == p * n
            @test _nbsr_sha(y) == fixture[key]["data_sha256"]
        end
    end

    @testset "Poisson data: r at the boundary is never reported converged" begin
        Y = reshape(Int64.(fixture["Y_column_major"]), p, n)
        fit = GM.fit_nb_gllvm(Y; K = K)
        # Everywhere: a converged fit must have r inside [1e-6, 1e6].
        @test !(fit.converged && GM._dispersion_group_boundary([fit.r])[1])
        @test isfinite(fit.loglik) && fit.loglik < 0
        if on_record_platform && haskey(fixture, vkey * "_loglik")
            @test fixture[vkey * "_converged"]          # the recorded red state
            @test fixture[vkey * "_r"] > 1e6
            @test fit.r > 1e6
            @test fit.converged == false
            # The verdict only changes the flag: the loglik stays as computed.
            @test fit.loglik ≈ fixture[vkey * "_loglik"] atol = 1e-8
        end
    end

    @testset "healthy NB2 fits are unchanged vs origin/main (per Julia version)" begin
        for key in healthy_keys
            case = fixture[key]
            Y = reshape(Int64.(case["Y_column_major"]), p, n)
            fit = GM.fit_nb_gllvm(Y; K = K)
            @test 1e-6 <= fit.r <= 1e6
            if on_record_platform && haskey(case, vkey * "_loglik")
                @test fit.converged == case[vkey * "_converged"]
                @test fit.loglik ≈ case[vkey * "_loglik"] atol = 1e-8
            else
                @test fit.converged
                @test isfinite(fit.loglik) && fit.loglik < 0
            end
        end
    end
end
