using GLLVModels, Test, Random, Distributions, Statistics
const GM = GLLVModels

# #788: the zero-truncated count fits (TruncatedPoissonFit, TruncatedNegBin2Fit,
# TruncatedNegBin2PerTraitFit) accepted `offset` at fit time but kept none and had no getLV /
# predict / residuals. They now store the training offset (`fit.offset`) and have the three
# post-fit methods, with the same offset semantics as PoissonFit / NBFit: the default call uses the
# stored training offset, an explicit `offset =` overrides it.
#
# Response scale: `predict(...; type = :response)` is exp(eta), the same as gllvmTMB's
# predict(type = "response") for truncated_poisson / truncated_nbinom2 (its inverse-link rule is
# exp(eta) for every log-link family). `type = :mean` is the zero-truncated mean E[y | y >= 1].

function _ztcount_data(rng, p, n, K; r = nothing)
    O = 0.6 .* randn(rng, p, n)
    Λt = 0.5 .* randn(rng, p, K); Zt = randn(rng, n, K)
    η = 0.8 .+ O .+ Λt * Zt'
    function draw(μ, rt)
        while true
            y = rt === nothing ? rand(rng, Poisson(μ)) :
                rand(rng, NegativeBinomial(rt, rt / (rt + μ)))
            y >= 1 && return y
        end
    end
    Y = [draw(exp(η[t, s]), r === nothing ? nothing : (r isa AbstractVector ? r[t] : r))
         for t in 1:p, s in 1:n]
    return Y, O
end

p, n, K = 5, 50, 1
rng = MersenneTwister(78801)
Yp, Op = _ztcount_data(rng, p, n, K)
Yn, On = _ztcount_data(rng, p, n, K; r = 3.0)
Yt, Ot = _ztcount_data(rng, p, n, K; r = [2.0, 3.0, 4.0, 5.0, 6.0])

fits = Dict(
    "TruncatedPoissonFit" => (fit_truncated_poisson_gllvm(Yp; K = K, offset = Op),
                              fit_truncated_poisson_gllvm(Yp; K = K), Yp, Op),
    "TruncatedNegBin2Fit" => (fit_truncated_nbinom2_gllvm(Yn; K = K, offset = On),
                              fit_truncated_nbinom2_gllvm(Yn; K = K), Yn, On),
    "TruncatedNegBin2PerTraitFit" => (fit_truncated_nbinom2_gllvm_pertrait(Yt; K = K, offset = Ot),
                                      fit_truncated_nbinom2_gllvm_pertrait(Yt; K = K), Yt, Ot),
)

# zero-truncated mean by hand: mu / (1 - P(0))
_p0(::TruncatedPoissonFit, μ, t) = exp(-μ)
_p0(f::TruncatedNegBin2Fit, μ, t) = (f.r / (f.r + μ))^f.r
_p0(f::TruncatedNegBin2PerTraitFit, μ, t) = (f.r[t] / (f.r[t] + μ))^f.r[t]

@testset "zero-truncated count post-fit methods and stored offset (#788)" begin
    for name in sort(collect(keys(fits)))
        f, f0, Y, O = fits[name]
        @testset "$name" begin
            @test f isa getfield(GM, Symbol(name))
            # stored offset: the matrix it was trained with; `nothing` when none was given
            @test f.offset == O
            @test f0.offset === nothing

            # getLV: n x K, finite; default (stored) == explicit offset = O, differs from zero
            Z = getLV(f, Y; rotate = false)
            @test size(Z) == (n, K)
            @test all(isfinite, Z)
            @test Z == getLV(f, Y; rotate = false, offset = O)
            @test maximum(abs, Z .- getLV(f, Y; rotate = false, offset = zeros(p, n))) > 1e-3
            @test size(getLV(f, Y)) == (n, K)          # rotate = true default

            # predict: link = beta + O + Lambda z (hand), default == explicit O, != zero offset
            η = predict(f, Y; type = :link)
            @test size(η) == (p, n)
            @test η ≈ f.β .+ O .+ f.Λ * Z' atol = 1e-10
            @test η == predict(f, Y; type = :link, offset = O)
            @test maximum(abs, η .- predict(f, Y; type = :link, offset = zeros(p, n))) > 1e-3

            # response = exp(eta) (gllvmTMB predict(type = "response")); mean = truncated mean
            @test predict(f, Y; type = :response) ≈ exp.(η) rtol = 1e-12
            @test predict(f, Y) == predict(f, Y; type = :response)
            M = predict(f, Y; type = :mean)
            Mhand = [exp(η[t, s]) / (1 - _p0(f, exp(η[t, s]), t)) for t in 1:p, s in 1:n]
            @test M ≈ Mhand rtol = 1e-10
            @test all(M .> exp.(η))                     # truncation raises the mean
            @test all(M .> 1)                           # and a zero-truncated mean exceeds 1
            @test_throws ArgumentError predict(f, Y; type = :nonsense)

            # training offset does not apply to a different-size Y without an explicit offset
            Ynew = Y[:, 1:10]
            @test_throws ArgumentError predict(f, Ynew)
            @test size(predict(f, Ynew; offset = O[:, 1:10])) == (p, 10)
            # a fit without an offset needs none
            @test size(predict(f0, Ynew; type = :link)) == (p, 10)

            # residuals: right shape, finite, reproducible under a fixed rng, pearson deterministic
            R = residuals(f, Y; rng = MersenneTwister(1))
            @test size(R) == (p, n)
            @test all(isfinite, R)
            @test R == residuals(f, Y; rng = MersenneTwister(1))
            @test R == residuals(f, Y; rng = MersenneTwister(1), offset = O)
            Rp = residuals(f, Y; type = :pearson)
            @test size(Rp) == (p, n) && all(isfinite, Rp)
            @test Rp == residuals(f, Y; type = :pearson, offset = O)
            @test_throws ArgumentError residuals(f, Y; type = :nonsense)
        end
    end
end

@testset "positional constructors without the offset field still work (#788)" begin
    f, = fits["TruncatedPoissonFit"]
    g = TruncatedPoissonFit(f.β, f.Λ, f.link, f.loglik, f.converged, f.iterations, f.theta_packed)
    @test g.offset === nothing
    f, = fits["TruncatedNegBin2Fit"]
    g = TruncatedNegBin2Fit(f.β, f.Λ, f.r, f.link, f.loglik, f.converged, f.iterations,
                            f.theta_packed, f.min_site_eigen)
    @test g.offset === nothing && g.hessian === :observed
    g = TruncatedNegBin2Fit(f.β, f.Λ, f.r, f.link, f.loglik, f.converged, f.iterations,
                            f.theta_packed, f.min_site_eigen, :fisher)
    @test g.offset === nothing && g.hessian === :fisher
    f, = fits["TruncatedNegBin2PerTraitFit"]
    g = TruncatedNegBin2PerTraitFit(f.β, f.Λ, f.r, f.link, f.loglik, f.converged, f.iterations,
                                    f.theta_packed)
    @test g.offset === nothing && g.hessian === :observed
end

@testset "zero-truncated Dunn-Smyth helper (#788)" begin
    # Hand check: for y = 2 under zero-truncated Poisson(mu), u = F(1) + (F(2) - F(1)) v with
    # F(k) = (P(Y <= k) - P(0)) / (1 - P(0)).
    μ = 1.7; d = Poisson(μ)
    pm(k) = pdf(d, k)
    F(k) = k < 1 ? 0.0 : sum(pm, 1:k) / (1 - pm(0))
    for v in (0.1, 0.5, 0.9)
        u = F(1) + (F(2) - F(1)) * v
        @test GM._ds_z_zerotrunc(d, 2, v) ≈ quantile(Normal(), u) atol = 1e-10
    end
    # y = 1 spans [0, F(1)]
    @test GM._ds_z_zerotrunc(d, 1, 0.5) ≈ quantile(Normal(), F(1) * 0.5) atol = 1e-10

    # Under the true zero-truncated law the residuals are N(0, 1).
    rng = MersenneTwister(788)
    draw(d) = (while true; y = rand(rng, d); y >= 1 && return y; end)
    for d in (Poisson(0.4), Poisson(3.0), NegativeBinomial(2.0, 2.0 / (2.0 + 1.5)),
              NegativeBinomial(5.0, 5.0 / (5.0 + 0.3)))
        z = [GM._ds_z_zerotrunc(d, draw(d), rand(rng)) for _ in 1:20000]
        @test all(isfinite, z)
        @test abs(mean(z)) < 0.04
        @test abs(std(z) - 1) < 0.04
    end
end
