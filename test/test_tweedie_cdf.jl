using GLLVModels, Test, Random, QuadGK

# Direct coverage of `tweedie_cdf` (src/families/tweedie.jl): the exact atom at 0 plus a
# 200-point composite Simpson integral of the positive density. Part of #767.
#
# `BROKEN` marks the cases where the measured error exceeds the tolerance asserted for
# the case; each is `@test_broken` with its measured error in the comment, so a fix in
# src/ turns it into an "Unexpected Pass" and forces the marker to be dropped.

# Independent reference: atom + adaptive Gauss-Kronrod of the positive density.
_ref_cdf(y, μ, φ, p) = exp(tweedie_logpdf(0.0, μ, φ, p)) +
    first(quadgk(t -> exp(tweedie_logpdf(t, μ, φ, p)), 0.0, y; rtol = 1e-12, atol = 0.0))

# Tweedie draw (compound Poisson-Gamma), same construction as the bootstrap sampler.
_draw(rng, μ, φ, p) = GLLVModels._rand_tweedie(rng, μ, φ, p)

# Broken-aware assertion: `broken` cases record `@test_broken`, the rest `@test`.
function _check(ok::Bool, broken::Bool)
    if broken
        @test_broken ok
    else
        @test ok
    end
end

# Measured on main (Julia 1.13.2), (μ, φ, p) cases keyed:
#  - MONO: F decreases on the upper-tail grid: (1,1,1.5) min step -6.2e-5, (0.2,0.5,1.2)
#    min step -4.6e-5 (the 200-point grid is too coarse once y >> the density's support).
#  - REF: (5,2,1.8) returns 1.0 for every y > 0 (reference e.g. 0.034 at y = 1e-6, 0.296
#    at y = 1): p > 1.5 makes the density unbounded at 0 (Gamma shape (2-p)/(p-1) < 1),
#    and the Simpson endpoint f(1e-12 y) is huge, so the integral blows up and is clamped.
#  - MEAN: ∫(1-F) is 1.024 for μ = 1 (err 2.4e-2), 0.2023 for μ = 0.2 (err 2.3e-3),
#    0.0 for μ = 5 (err 5.0, F is 1 everywhere on y > 0).
#  - MC: for (5,2,1.8) the CDF is 1.0 where the empirical CDF is 0.33, 0.65, 0.84, 0.97
#    at q = 0.25μ, μ, 2μ, 4μ.
const BROKEN_MONO = Set([(1.0, 1.0, 1.5), (0.2, 0.5, 1.2)])
const BROKEN_TAIL = Set{Tuple}()
const BROKEN_REF = Set([(5.0, 2.0, 1.8)])
const BROKEN_MEAN = Set([(1.0, 1.0, 1.5), (0.2, 0.5, 1.2), (5.0, 2.0, 1.8)])
const BROKEN_MC = Set([(5.0, 2.0, 1.8)])

@testset "tweedie_cdf" begin
    cases = [(1.0, 1.0, 1.5), (0.2, 0.5, 1.2), (5.0, 2.0, 1.8)]

    @testset "y = 0 is the exact atom" begin
        for (μ, φ, p) in [cases; (3.0, 0.5, 1.3); (0.7, 2.0, 1.7)]
            @test tweedie_cdf(0.0, μ, φ, p) ≈ exp(-μ^(2 - p) / (φ * (2 - p))) rtol = 1e-12
        end
    end

    @testset "y < 0 pins current behaviour" begin
        # Documented, not ruled on: the CDF of a nonnegative variable is 0 below 0, but the
        # implementation returns the atom F0 = P(Y = 0) for every y <= 0.
        for (μ, φ, p) in cases
            F0 = exp(-μ^(2 - p) / (φ * (2 - p)))
            @test tweedie_cdf(-1.0, μ, φ, p) == tweedie_cdf(0.0, μ, φ, p)
            @test tweedie_cdf(-1e-9, μ, φ, p) ≈ F0 rtol = 1e-12
        end
    end

    @testset "monotone non-decreasing, within [0,1], upper tail" begin
        for (μ, φ, p) in cases
            grid = vcat(0.0, 10.0 .^ range(-6, log10(40 * μ^p * φ + 40 * μ), length = 200))
            F = [tweedie_cdf(y, μ, φ, p) for y in grid]
            @test all(0.0 .<= F .<= 1.0)
            _check(all(diff(F) .>= -1e-12), (μ, φ, p) in BROKEN_MONO)
            _check(tweedie_cdf(grid[end], μ, φ, p) > 1 - 1e-3, (μ, φ, p) in BROKEN_TAIL)
        end
    end

    @testset "agrees with quadrature reference (1e-6)" begin
        for (μ, φ, p) in cases
            for y in (1e-6, 1e-3, 0.1, 0.5 * μ, μ, 2μ, 5μ)
                err = abs(tweedie_cdf(y, μ, φ, p) - _ref_cdf(y, μ, φ, p))
                _check(err < 1e-6, (μ, φ, p) in BROKEN_REF)
            end
        end
    end

    @testset "tiny dispersion and tiny y" begin
        for (μ, φ, p) in [(1.0, 0.05, 1.5), (1.0, 0.01, 1.3)], y in (1e-6, 1e-3, 0.1, 0.5, 1.0, 2.0)
            err = abs(tweedie_cdf(y, μ, φ, p) - _ref_cdf(y, μ, φ, p))
            _check(err < 1e-6, false)
        end
    end

    @testset "integral of 1 - F recovers the mean (1e-3)" begin
        for (μ, φ, p) in cases
            ymax = 60 * μ^p * φ / μ + 60 * μ
            I = first(quadgk(y -> 1 - tweedie_cdf(y, μ, φ, p), 0.0, ymax; rtol = 1e-8))
            _check(abs(I - μ) < 1e-3, (μ, φ, p) in BROKEN_MEAN)
        end
    end

    @testset "empirical CDF of simulated draws" begin
        rng = MersenneTwister(20261019)
        N = 20_000
        for (μ, φ, p) in cases
            x = [_draw(rng, μ, φ, p) for _ in 1:N]
            for q in (0.0, 0.25 * μ, μ, 2μ, 4μ)
                emp = count(<=(q), x) / N
                F = tweedie_cdf(q, μ, φ, p)
                # 5 sd of a binomial proportion
                tol = 5 * sqrt(max(emp * (1 - emp), 1 / N) / N)
                _check(abs(emp - F) < tol, (μ, φ, p) in BROKEN_MC && q > 0)
            end
        end
    end
end
