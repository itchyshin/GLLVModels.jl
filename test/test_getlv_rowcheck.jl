using Test
using GLLVModels
using Random

# A Y with the wrong number of rows (traits) used to reach an @inbounds per-site mode
# search in getLV (and so predict) for these four fit types and was silently accepted
# (OrderedBetaFit, COMPoissonFit: a result with the fit's trait count came back) or failed
# with a DomainError / BoundsError at a garbage index (BetaBinomialFit,
# BetaBinomialGroupedFit). It is now a DimensionMismatch.
@testset "getLV refuses a Y with the wrong number of rows" begin
    Random.seed!(20261112)
    p, n = 4, 40
    z = randn(n)
    λ = [0.8, -0.5, 0.6, 0.4]
    η = 0.3 .+ λ * z'
    ybeta = clamp.(1 ./ (1 .+ exp.(-η)) .* 0.8 .+ 0.1 .* rand(p, n), 0.02, 0.98)
    ycnt = [rand() < 0.5 ? 0 : floor(Int, exp(η[i, j]) * 2 * rand() + 1) for i in 1:p, j in 1:n]
    N = fill(6, p, n)
    ybb = [sum(rand() < 1 / (1 + exp(-η[i, j])) for _ in 1:6) for i in 1:p, j in 1:n]

    cases = [
        ("OrderedBetaFit", fit_ordered_beta_gllvm(ybeta; K = 1), ybeta, (;)),
        ("COMPoissonFit", fit_compoisson_gllvm(ycnt; K = 1), ycnt, (;)),
        ("BetaBinomialFit", fit_beta_binomial_gllvm(ybb; N = N, K = 1), ybb, (; N = N)),
        ("BetaBinomialGroupedFit",
         fit_beta_binomial_gllvm_grouped(ybb; N = N, K = 1), ybb, (; N = N)),
    ]
    for (name, fit, Y, kw) in cases
        @testset "$name" begin
            Yw = Y[1:(end - 1), :]
            kww = haskey(kw, :N) ? (; N = N[1:(end - 1), :]) : kw
            @test_throws DimensionMismatch getLV(fit, Yw; kww...)
            @test_throws DimensionMismatch predict(fit, Yw; kww...)
            @test size(getLV(fit, Y; kw...)) == (n, 1)
        end
    end
end
