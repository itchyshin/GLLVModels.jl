# #865 (siblings of fit_gllvm_cov): the SVD loadings start must be QR-rotated to
# lower-triangular form before pack_lambda, so the start's ΛΛ' equals the SVD's
# rank-K reconstruction for K >= 2 instead of its truncated upper triangle.
using Test, Random, LinearAlgebra, Distributions
using GLLVModels

@testset "SVD loadings start keeps ΛΛ' in family fitters (#865)" begin
    p, n = 8, 60
    for K in (2, 3)
        Random.seed!(865 + K)
        η = 0.3 .+ 0.4 .* (randn(p, K) * randn(K, n))
        function target(Z)                       # rank-K SVD fit of the centred link-scale data
            Zc = Z .- sum(Z; dims = 2) ./ n
            F = svd(Zc)
            L = F.U[:, 1:K] .* (F.S[1:K] ./ sqrt(n))'
            L * L'
        end
        Yt = η .+ randn(p, n)
        Ye = [rand(Exponential(exp(x))) for x in η]
        Yg = [rand(Gamma(2.0, exp(x) / 2)) for x in η]
        cases = (
            ("studentt", () -> fit_studentt_gllvm(Yt; K = K, iterations = 0), Yt),
            ("exponential", () -> fit_exponential_gllvm(Ye; K = K, iterations = 0), log.(max.(Ye, 1e-6))),
            ("gamma", () -> fit_gamma_gllvm(Yg; K = K, iterations = 0), log.(max.(Yg, 1e-6))),
        )
        for (nm, fitf, Z) in cases
            f = fitf()
            @test f.Λ * f.Λ' ≈ target(Z) rtol = 1e-6
        end
    end
end
