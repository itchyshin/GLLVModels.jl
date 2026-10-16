using Test, GLLVModels, Random, Distributions

# #767: the Dunn–Smyth residual was `quantile(Normal(), clamp(u, 1e-12, 1 - 1e-12))` with
# `u` built from `cdf`, so every cell with a tail probability below 1e-12 got exactly
# +-7.034 (a count of 40 and a count of 100000 under a mean near 1 were indistinguishable).
# The residual is now computed in the nearer tail from `ccdf`, with a log-scale fallback
# when the tail probability itself underflows, so it stays finite and ordered.
@testset "Dunn–Smyth residuals are not capped at 7.034 (#767)" begin
    GM = GLLVModels
    cap = quantile(Normal(), 1 - 1e-12)

    @testset "Poisson fit: the issue's repro" begin
        Random.seed!(8)
        Y = rand(0:3, 4, 60)
        f = fit_poisson_gllvm(Y; K = 1)
        r = Dict{Int, Float64}()
        for v in (12, 40, 1000, 100000)
            Y2 = copy(Y); Y2[1, 1] = v
            r[v] = residuals(f, Y2; rng = MersenneTwister(1))[1, 1]
        end
        @test all(isfinite, values(r))
        @test r[12] < r[40] < r[1000] < r[100000]
        @test r[40] > cap
        @test r[1000] > cap
        # An ordinary cell is unchanged by the fix (|z| well below the old cap).
        @test abs(r[12]) < cap
    end

    @testset "NB2 fit: gross outliers stay ordered" begin
        Random.seed!(9)
        Y = rand(0:3, 4, 60)
        f = fit_nb_gllvm(Y; K = 1)
        r = Dict{Int, Float64}()
        for v in (40, 1000, 100000)
            Y2 = copy(Y); Y2[1, 1] = v
            r[v] = residuals(f, Y2; rng = MersenneTwister(1))[1, 1]
        end
        @test all(isfinite, values(r))
        @test r[40] < r[1000] < r[100000]
        @test r[1000] > cap
    end

    @testset "_norm_upper_z inverts the log upper tail" begin
        for lt in (-1.0, -50.0, -700.0, -709.0, -1e3, -1e4, -1e5)
            z = GM._norm_upper_z(lt)
            @test isfinite(z) && z > 0
            @test isapprox(logccdf(Normal(), z), lt; rtol = 1e-8)
        end
    end

    @testset "_ds_z agrees with quantile(Normal(), u) away from the tails" begin
        rng = MersenneTwister(767)
        for _ in 1:300
            lo = 0.9 * rand(rng)
            hi = lo + (1 - lo) * rand(rng)
            v = rand(rng)
            u = lo + (hi - lo) * v
            @test GM._ds_z(lo, hi, 1 - lo, 1 - hi, v) ≈ quantile(Normal(), u) atol = 1e-10
        end
        @test isnan(GM._ds_z(NaN, NaN, NaN, NaN, 0.5))
    end

    @testset "extreme cells are finite and increase with the observation" begin
        for (d, ys) in ((Poisson(1.5), (30, 60, 90)),
                        (NegativeBinomial(2.0, 0.5), (100, 400, 5000)),
                        (Binomial(100, 0.01), (40, 60, 90)))
            z = [GM._ds_z_discrete(d, y, 0.5) for y in ys]
            @test all(isfinite, z)
            @test issorted(z; lt = <=)
            @test z[end] > cap
        end
        # The upper tail underflows (ccdf == 0): the log-mass fallback still orders cells.
        d = Poisson(1.5)
        @test ccdf(d, 100000 - 1) == 0
        @test GM._ds_z_discrete(d, 100000, 0.5) > GM._ds_z_discrete(d, 5000, 0.5) > cap
        # Where the tail is representable, the fallback agrees with the exact value.
        direct = GM._ds_z_discrete(d, 60, 0.5)
        fallback = GM._ds_z(1.0, 1.0, 0.0, 0.0, 0.5; lmass = logpdf(d, 60))
        @test isapprox(direct, fallback; atol = 0.05)
        # Lower tail: a zero count under a huge mean.
        zl = [GM._ds_z_discrete(Poisson(m), 0, 0.5) for m in (50.0, 500.0, 5000.0)]
        @test all(isfinite, zl)
        @test issorted(zl; rev = true)
        @test zl[end] < -cap
    end

    @testset "two-part and ordinal pieces use the same route" begin
        # Hurdle-style cell: P(Y <= y) = (1 - pi) + pi F_trunc(y), complement pi * tail.
        πt = 0.7; d = Poisson(1.5); p0 = exp(-1.5); y = 80
        lo = (1 - πt) + πt * (cdf(d, y - 1) - p0) / (1 - p0)
        hi = (1 - πt) + πt * (cdf(d, y) - p0) / (1 - p0)
        clo = πt * ccdf(d, y - 1) / (1 - p0)
        chi = πt * ccdf(d, y) / (1 - p0)
        z = GM._ds_z(lo, hi, clo, chi, 0.5)
        @test isfinite(z) && z > cap
        # Continuous cell: Gamma observation far in the upper tail.
        zc = GM._ds_z_continuous(Gamma(2.0, 1.0), 80.0)
        @test isfinite(zc) && zc > cap
    end
end
