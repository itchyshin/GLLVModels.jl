using GLLVModels, Test, ForwardDiff
using SpecialFunctions: loggamma

# #872: betabinomial_logp used to switch to the Binomial(N, μ) log-pmf at
# φ >= 1e6, a step of order N²μ(1−μ)/(2φ) in the objective (0.079 per row at
# N = 1000). The pmf must be continuous in φ and equal the exact beta-binomial.
@testset "betabinomial_logp is continuous in φ (#872)" begin
    GM = GLLVModels
    # Exact reference: shift(x, n) = Σ_{k<n} log1p(k/x) has no cancellation.
    sh(x, n) = n == 0 ? 0.0 : sum(log1p(k / x) for k in 0:(n - 1))
    function bb_exact(y, N, μ, φ)
        a = μ * φ
        b = (1 - μ) * φ
        return loggamma(N + 1) - loggamma(y + 1) - loggamma(N - y + 1) +
               y * log(μ) + (N - y) * log(1 - μ) + sh(a, y) + sh(b, N - y) - sh(φ, N)
    end
    lp(y, η, N, φ) = GM.betabinomial_logp(y, η, N, φ)

    @testset "no step across the old φ = 1e6 switch" begin
        for N in (10, 100, 1000)
            y = round(Int, 0.3N)
            step = lp(y, 0.0, N, prevfloat(1e6)) - lp(y, 0.0, N, 1e6)
            @test abs(step) < 1e-10
        end
    end

    @testset "matches the exact pmf for φ from 0.01 to 1e300" begin
        for φ in (0.01, 1.0, 12.0, 1e2, 1e4, 1e6, 1e8, 1e12, 1e300),
            (y, N, μ) in ((300, 1000, 0.5), (5, 12, 0.4), (0, 10, 0.05), (9, 9, 0.5))
            η = log(μ / (1 - μ))
            @test lp(y, η, N, φ) ≈ bb_exact(y, N, μ, φ) atol = 1e-10
        end
    end

    @testset "AD score is finite and matches a finite difference past 1e6" begin
        y, N, φ = 300, 1000, 2e6
        g = ForwardDiff.derivative(η -> lp(y, η, N, φ), 0.1)
        h = 1e-5
        fd = (lp(y, 0.1 + h, N, φ) - lp(y, 0.1 - h, N, φ)) / (2h)
        @test isfinite(g)
        @test g ≈ fd rtol = 1e-5
    end
end
