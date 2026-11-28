# (refs #1156) grouped Beta default hessian and refusal of :observed on non-logit links
using Test, GLLVModels, Random, Distributions
const G = GLLVModels

@testset "Beta grouped hessian default and non-logit links (#1156)" begin
    rng = MersenneTwister(11)
    p, n = 4, 40
    L = [0.8, 0.4, -0.5, 0.3]
    z = randn(rng, n)
    a = rand(rng, p) .* 0.7 .+ 0.3
    Y = [begin
            μ = cdf(Normal(), 0.5 * a[j] - 0.3 + 0.5 * L[j] * z[i])
            φ = 8.0
            clamp(rand(rng, Beta(μ * φ, (1 - μ) * φ)), 1e-4, 1 - 1e-4)
        end for j in 1:p, i in 1:n]

    f_probit = fit_gllvm(Y; family = G.Beta(), K = 1, link = G.ProbitLink())
    @test f_probit.converged
    @test isfinite(f_probit.loglik)
    @test f_probit.iterations > 0

  @test_throws ArgumentError fit_beta_gllvm_grouped(Y; K = 1, link = G.ProbitLink(), hessian = :observed)
end