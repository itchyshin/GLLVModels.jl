using Test
using GLLVModels, Random, Statistics, Distributions

@testset "cv_gllvm residual_std docs and site-split behaviour (refs #1212)" begin
    doc = string(@doc(GLLVModels.CVResult))
    @test occursin("plug-in latent score", doc)
    @test occursin("latent 0", doc)
    rng = MersenneTwister(1)
    p, n = 8, 80
    Λt = [0.9 * (i == 1 ? 1.0 : (0.4 + 0.6 * (((i * 7 + 3) % 5) / 5)) * (iseven(i + 1) ? 1 : -1)) for i in 1:p]
    b = 0.3 .+ 0.3 .* randn(rng, p)
    Y = zeros(p, n)
    for s in 1:n
        z = randn(rng)
        for t in 1:p
            Y[t, s] = b[t] + Λt[t] * z + 0.6 * randn(rng)
        end
    end
  mask = rand(MersenneTwister(6), p, n) .> 0.08
    expected = sqrt(mean(1 .+ Λt .^ 2 ./ 0.36))
    res = cv_gllvm(Y; k_folds = 5, split = :site, family = Normal(), K = 1,
                   rng = MersenneTwister(11), mask = mask)
    @test res.residual_std > 1.2
    @test isapprox(res.residual_std, expected; rtol = 0.05)
end