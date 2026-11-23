using Test, GLLVModels, Random, Distributions, ForwardDiff

@testset "Beta observed Laplace weight respects y clamp" begin
    rng = MersenneTwister(1)
    p, n = 6, 60
    z = randn(rng, n)
    λ = 0.5 .+ 0.4 .* rand(rng, p)
    b = 0.3 .* randn(rng, p)
    Y = [let m = 1 / (1 + exp(-(b[j] + λ[j] * z[i])))
            rand(rng, Distributions.Beta(m * 10, (1 - m) * 10))
         end for j in 1:p, i in 1:n]
    fb = fit_gllvm(Y; family = GLLVModels.Beta(), K = 1)
  function pert(v)
        Y2 = copy(Y)
        for c in (5, 17, 29)
            Y2[c] = v
        end
        return GLLVModels.beta_marginal_loglik_laplace(Y2, fb.Λ, fb.β, 10.0)
    end
    @test isapprox(pert(1e-12), pert(1e-300); rtol = 0, atol = 1e-6)
    @test isfinite(pert(1e-300))
    Y2 = copy(Y)
    for c in (5, 17, 29)
        Y2[c] = 1e-300
    end
    @test isfinite(fit_gllvm(Y2; family = GLLVModels.Beta(), K = 1).loglik)
end