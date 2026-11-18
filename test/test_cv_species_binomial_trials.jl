using GLLVModels
using Test
using Random
using Distributions
using StableRNGs

# cv_gllvm(split = :species) with a Binomial trial matrix whose rows differ: the held-out
# log-density must use each cell's own trial count N[t, s], not the count of species 1.
@testset "cv_gllvm species split uses N[t, s] for the Binomial density" begin
    rng = StableRNG(20261112)
    p, n = 8, 60
    N = [2 + (t % 4) * 3 for t in 1:p, s in 1:n]            # rows differ: 2, 5, 8, 11, ...
    Λ = reshape(0.8 .* [1.0, -0.6, 0.5, -0.4, 0.7, -0.5, 0.3, 0.6], p, 1)
    β = 0.2 .* randn(rng, p)
    Y = zeros(Int, p, n)
    for s in 1:n
        z = randn(rng)
        for t in 1:p
            Y[t, s] = rand(rng, Binomial(N[t, s], 1 / (1 + exp(-(β[t] + Λ[t, 1] * z)))))
        end
    end
    res = cv_gllvm(Y; k_folds = 4, split = :species, family = Binomial(), K = 1,
                   N = N, rng = StableRNG(1))
    @test isfinite(res.loglik)
    @test all(isfinite, res.fold_logliks)
    # Rebuild the score from the public fields: predictions hold N * mu.
    ll = 0.0
    for t in 1:p, s in 1:n
        μ = res.predictions[t, s] / N[t, s]
        ll += logpdf(Binomial(N[t, s], μ), Y[t, s])
    end
    @test res.loglik ≈ ll rtol = 1e-10
end
