using GLLVModels, Test, Random, Distributions

@testset "bic(fit) on GllvmCovFit (refs #1558)" begin
    Random.seed!(1558)
    p, K, n = 3, 1, 20
    X = zeros(p, n, 1)
    x = randn(n)
    @inbounds for t in 1:p, s in 1:n
        X[t, s, 1] = x[s]
    end
    η = 0.1 .* randn(p, n) .+ 0.4 .* x'
    Y = [rand(Poisson(exp(η[t, s]))) for t in 1:p, s in 1:n]
    fit = fit_gllvm_cov(Y; family = Poisson(), X = X, K = K, iterations = 30)

    @test isfinite(aic(fit))

    err = try
        bic(fit)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("bic(fit, Y)", err.msg)

    pn = nobs(fit, Y)
    @test isfinite(bic(fit, Y))
    @test bic(fit, Y) ≈ bic(fit, pn)
end