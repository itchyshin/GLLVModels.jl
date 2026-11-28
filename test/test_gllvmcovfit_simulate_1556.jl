using GLLVModels, Test, Random, Distributions

# No code change for #1556 — documents current MethodError until a package-wide
# refusal-only simulate(fit, n) pattern exists (see docs/dev-log/owed/2026-11-27-1556-simulate-gllvmcovfit-rec.md).

@testset "simulate(fit, n) on GllvmCovFit (refs #1556)" begin
    Random.seed!(1556)
    p, K, n = 3, 1, 15
    X = zeros(p, n, 1)
    @inbounds for t in 1:p, s in 1:n
        X[t, s, 1] = randn()
    end
    Y = [rand(Poisson(2.0)) for t in 1:p, s in 1:n]
    fit = fit_gllvm_cov(Y; family = Poisson(), X = X, K = K, iterations = 20)
    @test_throws MethodError simulate(fit, n)
    @test size(simulate(fit, X)) == (p, n)
end