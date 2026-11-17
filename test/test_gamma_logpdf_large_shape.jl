using GLLVModels, Test, Distributions
using SpecialFunctions: logabsgamma
const GM = GLLVModels

# Gamma conditional log-density at a very large shape. `logpdf(Gamma(a, μ/a), y)` subtracts two
# terms of size a·log a; at a ≈ 1e18 each is ~1e20 and Float64 leaves ~1e4 of rounding error per
# observation, so a flat-shape fit chased noise and reported a positive log-likelihood (the
# per-trait USairpollution fit gave +1.34e6 without --check-bounds=yes, true value about -1297.4).
# Reference: the same closed form in 320-bit BigFloat.

function _glp_ref(a, μ, y)
    setprecision(BigFloat, 320) do
        A = BigFloat(a); M = BigFloat(μ); Y = BigFloat(y)
        Float64(-first(logabsgamma(A)) - A * log(M / A) + (A - 1) * log(Y) - A * Y / M)
    end
end

@testset "Gamma log-density is accurate at large shape" begin
    for a in (1.0, 50.0, 999.0, 1001.0, 1e5, 1e8, 1e12, 1e18), (μ, y) in ((10.0, 10.0), (10.0, 12.5), (3.0, 0.4), (250.0, 244.0))
        got = GM._glm_logpdf(Gamma(a, 1.0), μ, 1, y)
        ref = _glp_ref(a, μ, y)
        # the error budget is a few ulps of the answer, not of a·log a
        @test isapprox(got, ref; atol = 1e-9 + 1e-12 * abs(ref))
    end
end

@testset "Gamma log-density below the switch is the original expression" begin
    for a in (0.3, 1.0, 7.0, 999.9), (μ, y) in ((10.0, 10.0), (3.0, 0.4))
        @test GM._glm_logpdf(Gamma(a, 1.0), μ, 1, y) === logpdf(Gamma(a, μ / a), y)
    end
end
