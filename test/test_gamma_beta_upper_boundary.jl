using GLLVModels, Test, TOML, SHA
const GM = GLLVModels

# Gamma and Beta grouped fits: a large dispersion is not a boundary.
# On origin/main 0ce4a35aa, data drawn with true Gamma shape alpha = 1e8 (or Beta
# precision phi = 1e8) were estimated at 8.9e7 to 1.3e8, yet the grouped Gamma and
# Beta fitters reported converged = false, because `_dispersion_group_boundary`
# flags any dispersion above 1e6. That rule is right for NB r (the Poisson limit,
# where the likelihood is flat), but a large Gamma alpha or Beta phi is the
# near-deterministic end, which the data identify. Gamma and Beta grouped routes
# now flag only the lower end (`_dispersion_group_lower_boundary`); NB2 and NB1
# keep both ends. Literal data: test/fixtures/gamma_beta_upper_boundary.toml.

const GBUB_FIXTURE_PATH = joinpath(@__DIR__, "fixtures", "gamma_beta_upper_boundary.toml")

_gbub_sha(A) = bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(A)))))

function _gbub_data(fx, c)
    p, n, q = fx["p"], fx["n"], fx["q"]
    Y = reshape(Float64.(c["Y_column_major"]), p, n)
    X = haskey(c, "X_column_major") ? reshape(Float64.(c["X_column_major"]), p, n, q) : nothing
    return Y, X
end

function _gbub_fit(fx, c)
    Y, X = _gbub_data(fx, c)
    K = fx["K"]; g = Int.(c["group"])
    if c["family"] == "gamma"
        return X === nothing ? GM.fit_gamma_gllvm_grouped(Y; K = K, group = g) :
                               GM.fit_gamma_gllvm_grouped_cov(Y; X = X, K = K, group = g)
    elseif c["family"] == "beta"
        return X === nothing ? GM.fit_beta_gllvm_grouped(Y; K = K, group = g) :
                               GM.fit_beta_gllvm_grouped_cov(Y; X = X, K = K, group = g)
    else
        return GM.fit_nb_gllvm_grouped(round.(Int, Y); K = K, group = g)
    end
end

_gbub_disp(f) = hasproperty(f, :α) ? f.α : hasproperty(f, :φ) ? f.φ : f.r_group

@testset "Gamma/Beta grouped: a large dispersion is not a boundary" begin
    fx = TOML.parsefile(GBUB_FIXTURE_PATH)
    vkey = "main_julia_1_$(VERSION.minor)"
    # The origin/main logliks were measured on macOS aarch64; they bind only there.
    on_record_platform = Sys.isapple() && Sys.ARCH === :aarch64
    huge_keys = ["gamma_huge", "gamma_huge_cov", "beta_huge", "beta_huge_cov"]
    healthy_keys = sort([k for k in keys(fx) if startswith(k, "healthy_")])

    @testset "fixture integrity" begin
        @test length(healthy_keys) >= 5
        for key in vcat(huge_keys, ["gamma_lower", "nb_upper"], healthy_keys)
            c = fx[key]
            Y, X = _gbub_data(fx, c)
            @test length(c["Y_column_major"]) == fx["p"] * fx["n"]
            @test _gbub_sha(Y) == c["data_sha256"]
            X === nothing || @test _gbub_sha(X) == c["X_sha256"]
        end
    end

    @testset "boundary helpers" begin
        d = [1e-8, 1e-6, 1.0, 1e6, 1e8]
        @test GM._dispersion_group_boundary(d) == [true, false, false, false, true]
        @test GM._dispersion_group_lower_boundary(d) == [true, false, false, false, false]
    end

    @testset "positional constructors derive the per-family flag" begin
        β = zeros(2); Λ = zeros(2, 1); g = [1, 1]; γ = [0.0]; γf = [false]
        for (v, flag) in ((1e8, false), (1e-8, true), (5.0, false))
            @test GM.GammaGroupedFit(β, Λ, [v], g, GM.LogLink(), -1.0, true, 1).dispersion_boundary == [flag]
            @test GM.BetaGroupedFit(β, Λ, [v], g, GM.LogitLink(), -1.0, true, 1).dispersion_boundary == [flag]
            @test GM.GammaGroupedCovFit(β, γ, γf, Λ, [v], g, GM.LogLink(), -1.0, true, 1).dispersion_boundary == [flag]
            @test GM.BetaGroupedCovFit(β, γ, γf, Λ, [v], g, GM.LogitLink(), -1.0, true, 1).dispersion_boundary == [flag]
        end
        # NB2 and NB1 keep both ends.
        @test GM.NBGroupedFit(β, Λ, [1e8], g, GM.LogLink(), -1.0, true, 1).dispersion_boundary == [true]
        @test GM.NB1GroupedFit(β, Λ, [1e8], g, GM.LogLink(), -1.0, true, 1).dispersion_boundary == [true]
    end

    @testset "huge true shape / precision: no longer a boundary" begin
        for key in huge_keys
            c = fx[key]
            fit = _gbub_fit(fx, c)
            est = _gbub_disp(fit)
            @test c["main_julia_1_10_converged"] == false   # the recorded red state
            @test all(>(1e6), est)
            @test all(1e7 .< est .< 1e9)                      # close to the truth 1e8
            @test !any(fit.dispersion_boundary)
            # Gamma: was false on main only because of the boundary flag. Beta: also
            # needed the standard-error-scaled gradient test (`_beta_grouped_verdict`):
            # with φ above about 1e5 the raw gradient at a stationary point is 1 to 10
            # (intercept curvature about 1e9), so the #480 raw-gradient gate could never
            # pass. The diagonal Newton polish that goes with that test can move a Beta
            # optimum up slightly (beta_huge: +1.2e-4), never down.
            @test fit.converged
            @test isfinite(fit.loglik)
            if on_record_platform && haskey(c, vkey * "_loglik")
                if c["family"] == "gamma"
                    @test fit.loglik ≈ c[vkey * "_loglik"] atol = 1e-8   # verdict only
                else
                    @test fit.loglik >= c[vkey * "_loglik"] - 1e-8
                    @test fit.loglik - c[vkey * "_loglik"] < 1e-2
                end
            end
        end
    end

    @testset "Gamma lower end is still a boundary" begin
        fit = _gbub_fit(fx, fx["gamma_lower"])
        @test all(<(1e-6), fit.α)
        @test fit.dispersion_boundary == [true]
        @test fit.converged == false
    end

    @testset "NB2 upper end is still a boundary" begin
        fit = _gbub_fit(fx, fx["nb_upper"])
        @test all(>(1e6), fit.r_group)
        @test fit.dispersion_boundary == [true]
        @test fit.converged == false
    end

    @testset "healthy grouped Gamma/Beta fits unchanged vs origin/main" begin
        for key in healthy_keys
            c = fx[key]
            fit = _gbub_fit(fx, c)
            @test fit.converged
            @test isfinite(fit.loglik)
            @test !any(fit.dispersion_boundary)
            if on_record_platform && haskey(c, vkey * "_loglik")
                @test fit.converged == c[vkey * "_converged"]
                @test fit.loglik ≈ c[vkey * "_loglik"] atol = 1e-8
            end
        end
    end
end
