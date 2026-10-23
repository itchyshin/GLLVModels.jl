using GLLVModels, Test, Random, Statistics

# Issue #770: predictive_check must forward trial counts to simulate, and a
# non-finite observed statistic must return NaN rather than p = 0.
@testset "predictive_check — binomial trials use N; NaN statistic is not p = 0" begin
    Random.seed!(12)
    p, n = 3, 24
    Ntrials = fill(8, p, n)
    Y = [rand(0:8) for _ in 1:p, _ in 1:n]
    fit = fit_binomial_gllvm(Y; K = 1, N = Ntrials, iterations = 15)

    pc = GLLVModels.predictive_check(fit, Y; nsim = 20, rng = MersenneTwister(1),
                                     N = Ntrials)
    mean_rows = findall(==("mean"), pc.stat)
    # Bernoulli draws are in {0,1}; trial counts of 8 put the mean above 1.
    @test all(>(1.0), pc.sim_mean[mean_rows])
    @test !all(iszero, pc.p_value)

    Yn = Float64.(Y)
    Yn[1, 1] = NaN
    pcn = GLLVModels.predictive_check(fit, Yn; nsim = 20, rng = MersenneTwister(1),
                                      N = Ntrials)
    t1_mean = findfirst(i -> pcn.stat[i] == "mean" && pcn.trait[i] == 1,
                        eachindex(pcn.stat))
    @test isnan(pcn.observed[t1_mean])
    @test isnan(pcn.p_value[t1_mean])

    mask = trues(p, n)
    mask[1, 1] = false
    pcm = GLLVModels.predictive_check(fit, Yn; nsim = 20, rng = MersenneTwister(1),
                                      N = Ntrials, mask = mask)
    @test isfinite(pcm.observed[t1_mean])
    @test isfinite(pcm.p_value[t1_mean])
    @test 0.0 <= pcm.p_value[t1_mean] <= 1.0
end
