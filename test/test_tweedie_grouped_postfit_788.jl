using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra
const GM = GLLVModels

# #788: TweedieGroupedFit and TweediePerTraitPowerFit accepted `offset` (and `mask`) at fit time
# but kept no offset and had no getLV / predict / residuals. They now store the training offset
# (`fit.offset`) and have the three post-fit methods, mirroring the plain TweedieFit (types,
# defaults, Dunn-Smyth and Pearson residuals) plus the `mask` / `offset` / `modes` keywords that
# the other grouped-dispersion fits take.
#
# Model: y | z ~ Tweedie(mu, phi_t, power_t), mu = exp(beta + offset + Lambda z), phi_t = phi[group[t]],
# power_t = power (shared) or power[t] (per trait).

function _rtweedie(rng, μ, φ, pw)
    λ = μ^(2 - pw) / (φ * (2 - pw))
    k = rand(rng, Poisson(λ))
    k == 0 && return 0.0
    return rand(rng, Gamma(k * (2 - pw) / (pw - 1), φ * (pw - 1) * μ^(pw - 1)))
end

p, n, K = 4, 60, 1
rng = MersenneTwister(78804)
O = 0.3 .* randn(rng, p, n)
Λt = 0.6 .* randn(rng, p, K); Zt = randn(rng, n, K)
η0 = 0.8 .+ O .+ Λt * Zt'
Y = [_rtweedie(rng, exp(η0[t, s]), 1.2, 1.5) for t in 1:p, s in 1:n]
group = [1, 1, 2, 2]

f  = fit_tweedie_gllvm_grouped(Y; K = K, group = group, power = nothing, offset = O)
f0 = fit_tweedie_gllvm_grouped(Y; K = K, group = group)
fs = fit_tweedie_gllvm_grouped(Y; K = K, group = group, power_group = :species, offset = O)
M = trues(p, n); M[2, 5] = false; M[3, 17] = false
fm = fit_tweedie_gllvm_grouped(Y; K = K, group = group, offset = O, mask = M)

φvec(fit) = fit.φ[fit.group]
pwvec(fit) = fit.power isa AbstractVector ? fit.power : fill(fit.power, length(fit.group))

@testset "fit types and stored offset (#788)" begin
    @test f isa TweedieGroupedFit
    @test fs isa TweediePerTraitPowerFit
    @test f.offset == O && fs.offset == O && fm.offset == O
    @test f0.offset === nothing
    @test fit_gllvm(Y; family = GM.TweedieED(1.0, 1.5), K = K, disp_group = group, offset = O).offset == O
end

for (label, fit) in (("TweedieGroupedFit", f), ("TweediePerTraitPowerFit", fs))
    @testset "$label getLV, predict, residuals (#788)" begin
        φ_, pw_ = φvec(fit), pwvec(fit)
        Z = getLV(fit, Y; rotate = false)
        @test size(Z) == (n, K) && all(isfinite, Z)
        @test Z == getLV(fit, Y; rotate = false, offset = O)
        @test maximum(abs, Z .- getLV(fit, Y; rotate = false, offset = zeros(p, n))) > 1e-3
        @test getLV(fit, Y) ≈ Z * GM._svd_rotation(fit.Λ) atol = 1e-12

        # independent check: Z is the maximiser of the site log-posterior
        function lp(s, z)
            η = fit.β .+ O[:, s] .+ fit.Λ * z
            sum(GM.tweedie_logpdf(Y[t, s], exp(η[t]), φ_[t], pw_[t]) for t in 1:p) - sum(abs2, z) / 2
        end
        for s in (1, 12, 33, 60)
            z = Z[s, :]
            @test lp(s, z) >= lp(s, z .+ 0.02) - 1e-9 && lp(s, z) >= lp(s, z .- 0.02) - 1e-9
        end

        # predict: link by hand, response = exp(link), defaults and errors
        η = predict(fit, Y; type = :link)
        @test size(η) == (p, n)
        @test η ≈ fit.β .+ O .+ fit.Λ * Z' atol = 1e-10
        @test η == predict(fit, Y; type = :link, offset = O)
        @test maximum(abs, η .- predict(fit, Y; type = :link, offset = zeros(p, n))) > 1e-3
        @test predict(fit, Y) ≈ exp.(η) rtol = 1e-12
        @test predict(fit, Y) == predict(fit, Y; type = :response)
        @test_throws ArgumentError predict(fit, Y; type = :median)
        @test_throws ArgumentError predict(fit, Y; modes = :bad)
        ηt = predict(fit, Y; type = :link, modes = :training, offset = O .+ 0.3)
        @test ηt ≈ fit.β .+ (O .+ 0.3) .+ fit.Λ * Z' atol = 1e-10

        # new units need their own offset
        Ynew = Y[:, 1:10]
        @test_throws ArgumentError predict(fit, Ynew)
        @test_throws ArgumentError getLV(fit, Ynew)
        @test size(predict(fit, Ynew; offset = O[:, 1:10])) == (p, 10)
        @test size(getLV(fit, Ynew; offset = 0.2)) == (10, K)
        @test_throws DimensionMismatch getLV(fit, Ynew; offset = zeros(p, 11))

        # residuals: Pearson by hand; Dunn-Smyth reproducible, finite, roughly standard normal
        μ = predict(fit, Y)
        @test residuals(fit, Y; type = :pearson) ≈ (Y .- μ) ./ sqrt.(φ_ .* μ .^ reshape(pw_, p, 1)) rtol = 1e-10
        R1 = residuals(fit, Y; rng = MersenneTwister(1))
        @test R1 == residuals(fit, Y; type = :dunnsmyth, rng = MersenneTwister(1), offset = O)
        @test size(R1) == (p, n) && all(isfinite, R1)
        @test abs(mean(R1)) < 0.35 && 0.5 < std(R1) < 1.5
        # positive cells are deterministic PIT values (the atom at 0 is the only random part)
        pos = Y .> 0
        Ra = residuals(fit, Y; rng = MersenneTwister(1)); Rb = residuals(fit, Y; rng = MersenneTwister(2))
        @test Ra[pos] == Rb[pos]
        t, s = findfirst(pos).I
        @test Ra[t, s] ≈ quantile(Normal(), clamp(GM.tweedie_cdf(Y[t, s], μ[t, s], φ_[t], pw_[t]), 1e-12, 1 - 1e-12))
        @test maximum(abs, residuals(fit, Y; rng = MersenneTwister(1), offset = zeros(p, n)) .- Ra) > 1e-3
        @test_throws ArgumentError residuals(fit, Y; type = :deviance)

        # the public latent-score route
        @test extract_latent_scores(fit, Y) == getLV(fit, Y; rotate = false)
        @test extract_latent_scores(fit, Y; level = :unit_obs) === nothing
        @test isfinite(aic(fit)) && isfinite(loglikelihood(fit))
    end
end

@testset "masked Tweedie grouped fit (#788)" begin
    Zm = getLV(fm, Y; rotate = false, mask = M)
    @test size(Zm) == (n, K) && all(isfinite, Zm)
    # masked cells do not influence the scores or predictions
    Yj = copy(Y); Yj[2, 5] = 1e3; Yj[3, 17] = 1e3
    @test getLV(fm, Yj; rotate = false, mask = M) ≈ Zm atol = 1e-10
    @test predict(fm, Yj; type = :link, mask = M)[:, 5] ≈ predict(fm, Y; type = :link, mask = M)[:, 5] atol = 1e-10
    R = residuals(fm, Y; mask = M, rng = MersenneTwister(3))
    @test isnan(R[2, 5]) && isnan(R[3, 17]) && count(isnan, R) == 2
    Rp = residuals(fm, Y; type = :pearson, mask = M)
    @test isnan(Rp[2, 5]) && count(isnan, Rp) == 2
    # a masked fit's stored offset is finite here; the mask-free call still works on this fixture
    @test size(getLV(fm, Y)) == (n, K)
end

@testset "fit without an offset (#788)" begin
    @test size(getLV(f0, Y)) == (n, K)
    @test size(predict(f0, Y[:, 1:10]; type = :link)) == (p, 10)
    @test predict(f0, Y; type = :link) ≈ f0.β .+ f0.Λ * getLV(f0, Y; rotate = false)' atol = 1e-10
end

@testset "Tweedie grouped fits are in exactly one latent-score dispatch union (#788)" begin
    unions = (GM._ComponentAwareGllvmFit, GM._PlainGllvmFit, GM._PositionalArgGllvmFit)
    for T in (TweedieGroupedFit, TweediePerTraitPowerFit)
        @test count(U -> T <: U, unions) == 1
        @test T <: GM._PlainGllvmFit
    end
end

@testset "positional constructors without offset still work (#788)" begin
    g9 = TweedieGroupedFit(f.β, f.Λ, f.φ, f.power, f.group, f.link, f.loglik, f.converged, f.iterations)
    g10 = TweedieGroupedFit(f.β, f.Λ, f.φ, f.power, f.group, f.link, f.loglik, f.converged,
                            f.iterations, :fisher)
    g11 = TweedieGroupedFit(f.β, f.Λ, f.φ, f.power, f.group, f.link, f.loglik, f.converged,
                            f.iterations, :observed, true)
    @test g9.offset === nothing && g10.offset === nothing && g11.offset === nothing
    @test g10.hessian === :fisher && g11.power_fixed
    s10 = TweediePerTraitPowerFit(fs.β, fs.Λ, fs.φ, fs.power, fs.group, fs.link, fs.loglik,
                                  fs.converged, fs.iterations, fs.hessian)
    @test s10.offset === nothing
end
