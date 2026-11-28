# cv_gllvm site hold-out residual_std can exceed 1 under the true model (#1212).

using GLLVModels, Random, Statistics, Test

@testset "cv site-split residual_std calibration (#1212)" begin
    seed = 1
    rng = MersenneTwister(seed)
    p, n = 8, 80
    Λt = [0.9 * (i == 1 ? 1.0 : 0.5) for i in 1:p]
    b = 0.3 .+ 0.3 .* randn(rng, p)
    Y = zeros(p, n)
    for s in 1:n
        z = randn(rng)
        for t in 1:p
            Y[t, s] = b[t] + Λt[t] * z + 0.6 * randn(rng)
        end
    end
    mask = rand(MersenneTwister(seed + 5), p, n) .> 0.08
    cv = cv_gllvm(Y; K = 1, mask = mask, split = :site, k_folds = 5, rng = MersenneTwister(seed))
    @test cv.residual_std > 1.2
    cv_r = cv_gllvm(Y; K = 1, mask = mask, split = :random, k_folds = 5, rng = MersenneTwister(seed))
    @test cv_r.residual_std < cv.residual_std
end