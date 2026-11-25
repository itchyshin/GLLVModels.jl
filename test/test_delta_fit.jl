using GLLVModels, Test, Random, Distributions, Statistics

@testset "fit_delta_lognormal_gllvm" begin
    Random.seed!(140)
    p, K, n = 8, 2, 400
    βz_true = 0.5 .* randn(p) .+ 0.4         # occurrence logits (≈ 60% presence)
    βc_true = 0.5 .* randn(p)                # positive meanlog
    Λc_true = 0.6 .* randn(p, K)
    σ_true = 0.5
    Z = randn(K, n)
    ηc = βc_true .+ Λc_true * Z
    π = inv.(1 .+ exp.(-βz_true))
    Y = zeros(p, n)
    for t in 1:p, s in 1:n
        if rand() < π[t]
            Y[t, s] = exp(ηc[t, s] + σ_true * randn())
        end
    end

    # disp_group=:shared pinned: σ_true is a single shared scalar, and fit.σ is
    # compared below as a scalar (chained inequality) — :shared keeps this test's
    # pre-2026-09-24 behaviour unchanged now that :species is the public default.
    fit = fit_delta_lognormal_gllvm(Y; K = K, disp_group = :shared)
    @test fit isa DeltaLogNormalFit
    @test fit.converged
    @test cor(fit.βz, βz_true) > 0.8                                  # occurrence
    @test cor(fit.βc, βc_true) > 0.8                                  # positive meanlog
    @test cor(vec(fit.Λc * fit.Λc'), vec(Λc_true * Λc_true')) > 0.7   # loadings (Gram)
    @test 0.5 * σ_true < fit.σ < 2 * σ_true                           # log-scale SD

    # No-X public surfaces. `fit_gllvm` and `@formula` must reach exactly the
    # named Delta-lognormal fitter; the marker's σ is a tag payload (never read).
    @testset "no-X public surfaces: fit_gllvm and @formula" begin
        Random.seed!(141)
        p, K, n = 6, 1, 120
        βz = 0.4 .* randn(p) .+ 0.3
        βc = 0.4 .* randn(p)
        Λc = 0.5 .* randn(p, K)
        σ = 0.6
        Z = randn(K, n)
        ηc = βc .+ Λc * Z
        π = inv.(1 .+ exp.(-βz))
        Y = zeros(p, n)
        for t in 1:p, s in 1:n
            if rand() < π[t]
                Y[t, s] = exp(ηc[t, s] + σ * randn())
            end
        end

        fn = fit_delta_lognormal_gllvm(Y; K = K, iterations = 40)
        fu = fit_gllvm(Y; family = DeltaLogNormal(), K = K, iterations = 40)
        @test fu isa DeltaLogNormalFit
        @test fu.loglik ≈ fn.loglik atol = 1e-10
        @test fu.σ ≈ fn.σ atol = 1e-10

        # Tag payload: marker σ is never read — both constructors give the same fit.
        @test fit_gllvm(Y; family = DeltaLogNormal(9.0), K = K, iterations = 40).σ ≈
              fu.σ atol = 1e-10
        @test DeltaLogNormal().σ == 1.0

        ff = gllvm(@formula(y ~ 1), Y, (; temp = randn(n));
                   family = DeltaLogNormal(), K = K, iterations = 40)
        @test ff isa DeltaLogNormalFit
        @test ff.loglik ≈ fn.loglik atol = 1e-10
    end
end

@testset "DeltaLogNormal tiny positive response stays finite (#1243)" begin
    G = GLLVModels
    # Site level: zero occurrence loadings, equal occurrence prob 1/2, so the site marginal
    # has a closed form. y = 1e-300 puts the exact mode at |eta| ~ 82, outside the old ±30 clamp.
    p = 6
    rng = MersenneTwister(1); λ = 0.5 .+ 0.4 .* rand(rng, p); b = 0.3 .* randn(rng, p)
    σ = 0.8
    y = exp.(b .+ σ .* randn(rng, p)); y[1] = 1e-300
    r = log.(y) .- b
    A = 1 + sum(λ .^ 2) / σ^2
    zhat = sum(λ .* r) / σ^2 / A
    ll = sum(logpdf.(Normal.(b .+ λ .* zhat, σ), log.(y))) - sum(log.(y)) + p * log(0.5) -
         0.5 * zhat^2 - 0.5 * log(A)
    fams = G.DeltaLogNormal.(fill(σ, p))
    got = G.twopart_loglik_site(fams, y, zeros(p, 1), reshape(λ, p, 1), zeros(p), b)
    @test isfinite(got)
    @test got ≈ ll rtol = 1e-8
    # Fit level: three positive cells set to 1e-50 used to give logLik -Inf, converged = false.
    rng = MersenneTwister(1); p, n = 6, 60
    z = randn(rng, n); λ = 0.5 .+ 0.4 .* rand(rng, p); b = 0.3 .* randn(rng, p)
    Y = [rand(rng) < 0.25 ? 0.0 : exp(b[j] + λ[j] * z[i] + 0.8 * randn(rng)) for j in 1:p, i in 1:n]
    pos = findall(>(0), Y)
    Y[pos[[5, 17, 29]]] .= 1e-50
    r = fit_gllvm(Y; family = G.DeltaLogNormal(), K = 1)
    @test isfinite(r.loglik)
    @test r.converged
end
