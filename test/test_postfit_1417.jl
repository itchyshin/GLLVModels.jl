using GLLVModels, Test, Random
@testset "GaussianPerVarFit predict (#1417)" begin
    Random.seed!(2002)
    p, K, n = 3, 1, 40
    Y = randn(p, n)
    fit = fit_gaussian_pervar_gllvm(Y; K = K, iterations = 80)
    P = predict(fit, Y)
    @test size(P) == (p, n) && all(isfinite, P)
    x = randn(n)
    X = zeros(p, n, 2)
    for t in 1:p
        X[t, :, 1] .= 1
        X[t, :, 2] .= x
    end
    fitx = fit_gaussian_pervar_gllvm(Y; K = K, X = X, iterations = 80)
    @test size(predict(fitx, Y; X = X)) == (p, n)
end
