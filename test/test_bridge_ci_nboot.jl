using Test, GLLVModels, Random

# `ci_nboot` is read only by the bootstrap CI route. A value that is not a positive
# integer must be refused there, as the native Gaussian bootstrap already refuses
# `n_boot < 1`; on main the Laplace routes returned all-NaN intervals with an empty
# `ci_note` for ci_nboot = 0, and a fractional value died with an InexactError.
@testset "bridge_fit: ci_nboot must be a positive integer for bootstrap CIs" begin
    rng = MersenneTwister(5)
    Y = Float64.(rand(rng, 0:6, 4, 30))
    boot(nb; family = "poisson") = bridge_fit(; y = Y, family = family, d = 1,
        options = Dict("ci_method" => "bootstrap", "ci_nboot" => nb, "ci_seed" => 1))

    @testset "refused values" begin
        for bad in (0, -3, 2.7, NaN, Inf)
            @test_throws ArgumentError boot(bad)
        end
        err = try; boot(0); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin("ci_nboot", err.msg)
    end

    @testset "accepted values are unchanged" begin
        # R sends doubles: an integral Float64 behaves like the Int
        a = boot(20)
        b = boot(20.0)
        @test a.ci_lower == b.ci_lower
        @test a.ci_upper == b.ci_upper
        @test all(isfinite, a.ci_lower)
    end

    @testset "other CI methods do not read ci_nboot" begin
        w = bridge_fit(; y = Y, family = "poisson", d = 1,
                       options = Dict("ci_method" => "wald", "ci_nboot" => 0))
        @test w.ci_method == "wald"
        @test all(isfinite, w.ci_lower)
    end
end

# `ci_seed` went through a bare `Int(...)`, so 2.5, NaN or a string died with a raw
# InexactError or MethodError that named neither `bridge_fit` nor the option, whatever
# `ci_method` was (#1246). An integral double (what R sends) stays accepted.
@testset "bridge_fit: ci_seed must be an integer" begin
    rng = MersenneTwister(7)
    Y = Float64.(rand(rng, 0:6, 5, 60))
    fit(seed; method = "bootstrap", family = "poisson") = bridge_fit(; y = Y, family = family, d = 1,
        options = Dict("ci_method" => method, "ci_nboot" => 20, "ci_seed" => seed))

    @testset "refused values name bridge_fit and ci_seed" begin
        for bad in (2.5, NaN, Inf, "a", 1e30, nothing)
            for method in ("bootstrap", "wald", "none")
                err = try; fit(bad; method = method); nothing; catch e; e; end
                @test err isa ArgumentError
                @test occursin("bridge_fit: ci_seed", err.msg)
            end
        end
        err = try; fit(2.5; family = "gaussian"); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin("ci_seed", err.msg)
    end

    @testset "accepted values are unchanged" begin
        a = fit(1)
        b = fit(1.0)
        @test a.ci_lower == b.ci_lower
        @test a.ci_upper == b.ci_upper
        @test all(isfinite, a.ci_lower)
        d = bridge_fit(; y = Y, family = "poisson", d = 1, options = Dict("ci_method" => "wald"))
        @test d.ci_method == "wald"
    end
end
