using GLLVModels, Test

# NB1 expected Fisher information at large means. `_nb1_fisher_mu` started its pmf
# recursion at P(y = 0) = (1 + φ)^(-μ/φ), which underflows to exactly 0 once
# (μ/φ)·log(1 + φ) is above about 745 (mean counts of a few hundred and up). The loop then
# summed nothing, ran to its 10 000-term cap and returned the tail term ψ'(μ/φ)/φ², which
# is about (1 + φ) times the true information. The Fisher weight only sets the Newton step
# of the per-site mode search, so the mode (and the likelihood) did not move, but the step
# was a factor 1 + φ too short: a site at counts near 1e4 needed about 40 Fisher steps
# where the observed curvature needs 5, and each step cost a 10 000-term loop per species.
# Reference below: the same sum in 256-bit BigFloat from y = 0, which does not underflow.

function _nb1_fisher_big(μ, φ)
    setprecision(BigFloat, 256) do
        μb = BigFloat(μ); φb = BigFloat(φ)
        r = μb / φb; q = φb / (1 + φb)
        P = exp(-r * log1p(φb))
        S = zero(BigFloat); E = zero(BigFloat); tot = P
        y = 0
        while tot < 1 - big"1e-30" && y < 400_000      # all the mass, heavy upper tails included
            y += 1
            S += inv(r + y - 1)^2
            P *= q * (y - 1 + r) / y
            E += P * S; tot += P
        end
        return Float64(E / tot / φb^2)
    end
end

@testset "NB1 Fisher information at large means" begin
    @testset "matches a no-underflow BigFloat sum" begin
        for φ in (0.3, 1.0, 5.0), μ in (5.0, 300.0, 1500.0, 1e4)
            ref = _nb1_fisher_big(μ, φ)
            @test GLLVModels._nb1_fisher_mu(μ, φ) ≈ ref rtol = 1e-9
        end
        # large overdispersion: the mass passes the old 10 000-term cap with P(0) still finite
        for (μ, φ) in ((3000.0, 200.0), (800.0, 3000.0))
            @test GLLVModels._nb1_fisher_mu(μ, φ) ≈ _nb1_fisher_big(μ, φ) rtol = 1e-9
        end
    end

    @testset "degenerate trial points do not throw" begin
        # an L-BFGS line-search trial can push phi to Inf (r = 0) or to absurd values
        for φ in (Inf, 1e300, 1e12), μ in (1e4, 1e-12)
            @test GLLVModels._nb1_fisher_mu(μ, φ) isa Float64     # returns, whatever the value
        end
    end

    @testset "trigamma rise series agrees with the direct sum" begin
        for (r, y) in ((20.0, 65), (1e3, 500), (1e4, 12_000), (1e9, 20_000))
            direct = sum(inv(r + k)^2 for k in 0:(y - 1))
            @test GLLVModels._nb1_trigamma_rise(r, y) ≈ direct rtol = 1e-12
        end
    end

    @testset "Fisher mode search converges in few steps at counts near 1e4" begin
        p = 8
        μ0 = [9500.0, 10200.0, 10800.0, 9900.0, 11000.0, 10400.0, 9700.0, 10100.0]
        y = round.(μ0 .* [1.01, 0.98, 1.03, 0.99, 1.0, 1.02, 0.97, 1.01])
        β = log.(μ0); Λ = fill(0.05, p, 1)
        fams = [GLLVModels.NB1(1.0) for _ in 1:p]
        N = ones(Int, p)
        z, ok = GLLVModels._nb1_grouped_mode(fams, y, N, Λ, β, GLLVModels.LogLink(), :fisher;
                                             maxiter = 12)
        @test ok
    end
end
