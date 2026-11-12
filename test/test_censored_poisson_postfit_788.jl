using GLLVModels, Test, Random, Distributions, Statistics
const GM = GLLVModels

# #788: CensoredPoissonFit accepted `offset` at fit time but kept none and had no getLV / predict /
# residuals. It now stores the training offset (`fit.offset`) and has the three post-fit methods.
#
# The fit does not store the censoring pattern (it is data, not a parameter), so the post-fit
# methods take it the way the fitter does: `censored` (Bool p x n, `Y` holds the limit C where true)
# or `lower` / `upper`; with neither, every cell is an exact count.
#
# Response scale: `predict(...; type = :response)` is exp(eta), the mean of the underlying
# uncensored Poisson process, as gllvmTMB documents for fitted() / predict(type = "response").

p, n, K = 5, 60, 1
C = 4
rng = MersenneTwister(78802)
O = 0.5 .* randn(rng, p, n)
Λt = 0.5 .* randn(rng, p, K); Zt = randn(rng, n, K)
η0 = 0.9 .+ O .+ Λt * Zt'
Ytrue = [rand(rng, Poisson(exp(η0[t, s]))) for t in 1:p, s in 1:n]
cens = Ytrue .>= C
Y = ifelse.(cens, C, Ytrue)                      # limit C in censored cells
@assert 0.1 < mean(cens) < 0.6

f  = fit_censored_poisson_gllvm(Y; K = K, censored = cens, offset = O)
f0 = fit_censored_poisson_gllvm(Y; K = K, censored = cens)

@testset "CensoredPoissonFit stored offset and post-fit methods (#788)" begin
    @test f isa CensoredPoissonFit
    @test f.offset == O
    @test f0.offset === nothing

    # getLV: n x K, finite, default (stored) == explicit offset, differs from zero offset
    Z = getLV(f, Y; censored = cens, rotate = false)
    @test size(Z) == (n, K) && all(isfinite, Z)
    @test Z == getLV(f, Y; censored = cens, rotate = false, offset = O)
    @test maximum(abs, Z .- getLV(f, Y; censored = cens, rotate = false, offset = zeros(p, n))) > 1e-3
    @test size(getLV(f, Y; censored = cens)) == (n, K)
    # the censoring pattern matters: treating the limits as exact counts moves the modes
    @test maximum(abs, Z .- getLV(f, Y; rotate = false)) > 1e-4
    # lower / upper encoding gives the same scores as the Bool indicator
    L = copy(Y); U = copy(Y); U[cens] .= typemax(Int)
    @test getLV(f, Y; lower = L, upper = U, rotate = false) ≈ Z atol = 1e-12
    @test_throws ArgumentError getLV(f, Y; censored = cens, lower = L, upper = U)
    @test_throws ArgumentError getLV(f, Y; lower = L)

    # mode is a stationary point of the censored log-posterior: score equation at z
    Yc, Nc = GM.censored_bounds_to_YN(L, U)
    for s in (1, 7, 30)
        z = Z[s, :]
        lp(zz) = GM._laplace_mode_logpost(GM.CensoredPoisson(), Yc[:, s], Nc[:, s], f.Λ, f.β,
                                          f.link, zz; offset = O[:, s])
        h = 1e-5
        @test abs((lp(z .+ h) - lp(z .- h)) / 2h) < 1e-5
        @test lp(z) >= lp(z .+ 0.05) && lp(z) >= lp(z .- 0.05)
    end

    # predict: link = beta + O + Lambda z by hand; response = exp(eta)
    η = predict(f, Y; censored = cens, type = :link)
    @test size(η) == (p, n)
    @test η ≈ f.β .+ O .+ f.Λ * Z' atol = 1e-10
    @test η == predict(f, Y; censored = cens, type = :link, offset = O)
    @test maximum(abs, η .- predict(f, Y; censored = cens, type = :link, offset = zeros(p, n))) > 1e-3
    @test predict(f, Y; censored = cens) ≈ exp.(η) rtol = 1e-12
    @test predict(f, Y; censored = cens) == predict(f, Y; censored = cens, type = :response)
    @test_throws ArgumentError predict(f, Y; censored = cens, type = :mean)
    @test_throws ArgumentError predict(f, Y; censored = cens, type = :nonsense)

    # a different-size Y without an offset is refused; with one it works
    Ynew = Y[:, 1:10]
    @test_throws ArgumentError predict(f, Ynew; censored = cens[:, 1:10])
    @test size(predict(f, Ynew; censored = cens[:, 1:10], offset = O[:, 1:10])) == (p, 10)
    @test size(predict(f0, Ynew; censored = cens[:, 1:10], type = :link)) == (p, 10)
    # censored must match Y
    @test_throws ArgumentError getLV(f, Y; censored = cens[:, 1:10])

    # mask: masked cells are ignored whatever Y holds there
    msk = trues(p, n); msk[2, 5] = false; msk[4, 9] = false
    Ym = copy(Y); Ym[2, 5] = 12345; Ym[4, 9] = -1
    @test getLV(f, Ym; censored = cens, mask = msk, rotate = false) ≈
          getLV(f, Y; censored = cens, mask = msk, rotate = false) atol = 1e-12
    @test all(isnan, [residuals(f, Ym; censored = cens, mask = msk)[2, 5],
                      residuals(f, Ym; censored = cens, mask = msk)[4, 9]])
end

@testset "CensoredPoissonFit Dunn-Smyth residuals respect censoring (#788)" begin
    μ = predict(f, Y; censored = cens, type = :response)
    R = residuals(f, Y; censored = cens, rng = MersenneTwister(3))
    @test size(R) == (p, n) && all(isfinite, R)
    @test R == residuals(f, Y; censored = cens, rng = MersenneTwister(3))
    @test R == residuals(f, Y; censored = cens, rng = MersenneTwister(3), offset = O)
    # hand check: same draw order as the method (columns outer, rows inner)
    rng2 = MersenneTwister(3)
    Rh = similar(R)
    for s in 1:n, t in 1:p
        v = rand(rng2); d = Poisson(μ[t, s])
        u = cens[t, s] ? cdf(d, C - 1) + (1 - cdf(d, C - 1)) * v :
            cdf(d, Y[t, s] - 1) + (cdf(d, Y[t, s]) - cdf(d, Y[t, s] - 1)) * v
        Rh[t, s] = quantile(Normal(), u)
    end
    @test R ≈ Rh atol = 1e-8
    # a censored cell's residual lies above the lower end of its interval: z >= Phi^-1(F(C-1))
    for s in 1:n, t in 1:p
        cens[t, s] && @test R[t, s] >= quantile(Normal(), cdf(Poisson(μ[t, s]), C - 1)) - 1e-9
    end
    @test_throws ArgumentError residuals(f, Y; censored = cens, type = :pearson)

    # helper under the true law: censored and exact cells together are N(0, 1)
    rng3 = MersenneTwister(788)
    z = Float64[]
    for _ in 1:20000
        d = Poisson(rand(rng3, [0.4, 2.0, 6.0]))
        y = rand(rng3, d); v = rand(rng3)
        push!(z, y >= C ? GM._ds_z_censored(d, C, v) : GM._ds_z_discrete(d, y, v))
    end
    @test all(isfinite, z)
    @test abs(mean(z)) < 0.04 && abs(std(z) - 1) < 0.04
end

@testset "CensoredPoissonFit through extract_latent_scores / ordination (#788)" begin
    z = extract_latent_scores(f, Y; censored = cens)
    @test z == getLV(f, Y; censored = cens, rotate = false)
    @test extract_latent_scores(f, Y; censored = cens, offset = O) == z
    @test extract_latent_scores(f, Y; level = :unit_obs) === nothing
    @test size(getLV(f, Y; censored = cens, rotate = true)) == (n, K)
    @test rotation(f) isa AbstractMatrix
    @test getLoadings(f; rotate = false) == f.Λ
    @test isfinite(aic(f)) && isfinite(loglikelihood(f))
end

@testset "CensoredPoissonFit positional constructor without offset still works (#788)" begin
    g = CensoredPoissonFit(f.β, f.Λ, f.link, f.loglik, f.converged, f.iterations, f.theta_packed)
    @test g.offset === nothing
end
