# fit_gllvm_speciescov on Beta data whose default start (dispersion 10) makes one site's
# Laplace mode search fail returned logLik = -Inf, iterations = 0, with no message (#1200).
# Data: 12 species x 100 sites, two covariates with species-specific slopes (generator in the
# issue). CRAN gllvm 2.0.15 (LA, logit link, shared dispersion) reaches logLik 639.274244344.

using GLLVModels, Test, DelimitedFiles
const _GM1200 = GLLVModels

@testset "Beta speciescov: a failing start dispersion is retried, not returned as -Inf (#1200)" begin
    dd = joinpath(@__DIR__, "data")
    Y = Matrix{Float64}(readdlm(joinpath(dd, "beta_speciescov_start_1200_Y.csv"), ','))
    x = Matrix{Float64}(readdlm(joinpath(dd, "beta_speciescov_start_1200_X.csv"), ','))
    p, n = size(Y)
    X = zeros(p, n, size(x, 2))
    for k in axes(x, 2), t in 1:p, s in 1:n
        X[t, s, k] = x[s, k]
    end
    fit = _GM1200.fit_gllvm_speciescov(Y; family = _GM1200.Beta(), X = X, K = 1)
    @test isfinite(fit.loglik)
    @test fit.converged
    @test fit.iterations > 0
    @test fit.loglik ≈ 639.274244344 atol = 1e-3
end
