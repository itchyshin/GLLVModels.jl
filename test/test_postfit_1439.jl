using GLLVModels, Test, Random, Distributions
@testset "GllvmCovFit keyword X (#1439)" begin
    Random.seed!(1911)
    p, K, n = 5, 1, 60
    x = randn(n)
    X = zeros(p, n, 1)
    X[:, :, 1] .= reshape(x, 1, n)
    Y = [rand(Poisson(exp(0.2 + 0.4 * x[s]))) for t in 1:p, s in 1:n]
    fit = fit_gllvm_cov(Y; family = NegativeBinomial(), X = X, K = K, iterations = 80)
    Pkw = predict(fit, Y; X = X, type = :response)
    Ppos = predict(fit, Y, X; type = :response)
    @test Pkw ≈ Ppos
    R = residuals(fit, Y; X = X)
    @test size(R) == size(Y)
    ord = ordination(fit, Y; X = X)
    @test size(ord.sites, 1) == n
end
