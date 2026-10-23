using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra

# Issue #555: NBGroupedFit / NBGroupedCovFit had no predict, fitted, residuals
# (Dunn-Smyth) or getResidualCor methods, although they support getLV and
# confint. Every expected value below is computed independently of the new
# methods: a hand-written Newton search for the posterior latent mode, a
# hand-written NB2 CDF recursion, and cov2cor(Lambda Lambda') for the residual
# correlation (gllvmTMB's getResidualCor is extract_Sigma(link_residual = "none"),
# which is cov2cor(Lambda Lambda' + Psi) with Psi = 0 for these fits).

# Posterior mode of one site's latent vector for the NB2/log model with
# per-species size r[t], eta = eta0 + Lambda z, z ~ N(0, I). Newton with the
# observed information and a backtracking line search on the log-posterior.
function _nbpf_site_mode(y, Λ, η0, r; obs = trues(length(y)))
    K = size(Λ, 2)
    z = zeros(K)
    logpost(z) = begin
        η = η0 .+ Λ * z
        s = -0.5 * dot(z, z)
        for t in eachindex(y)
            obs[t] || continue
            μ = exp(η[t])
            s += logpdf(NegativeBinomial(r[t], r[t] / (r[t] + μ)), y[t])
        end
        s
    end
    for _ in 1:200
        μ = exp.(η0 .+ Λ * z)
        w = obs .* (r .* (y .- μ) ./ (r .+ μ))
        h = obs .* (r .* μ .* (r .+ y) ./ (r .+ μ) .^ 2)
        g = Λ' * w - z
        H = Λ' * (h .* Λ) + I
        stp = H \ g
        a = 1.0
        f0 = logpost(z)
        while logpost(z + a * stp) < f0 && a > 1e-10
            a /= 2
        end
        z = z + a * stp
        norm(a * stp) < 1e-13 && break
    end
    return z
end

# Hand linear predictor eta = eta0 + Lambda z-hat for every site (p x n).
function _nbpf_eta(Y, Λ, η0::AbstractMatrix, r; mask = trues(size(Y)))
    p, n = size(Y)
    η = Matrix{Float64}(undef, p, n)
    for s in 1:n
        z = _nbpf_site_mode(Y[:, s], Λ, η0[:, s], r; obs = mask[:, s])
        η[:, s] = η0[:, s] .+ Λ * z
    end
    return η
end

# NB2 CDF at integer y by the pmf recursion (Var = mu + mu^2 / r). Stable for
# the moderate means used here, including r = 1e12 (the Poisson limit).
function _nbpf_cdf(y, μ, r)
    y < 0 && return 0.0
    P = exp(-r * log1p(μ / r))
    s = P
    for k in 0:(y - 1)
        P *= (k + r) / (k + 1) * μ / (r + μ)
        s += P
    end
    return min(s, 1.0)
end

# Randomisation interval of the Dunn-Smyth residual of one cell, on the N(0,1) scale.
function _nbpf_ds_interval(y, μ, r)
    lo = quantile(Normal(), clamp(_nbpf_cdf(y - 1, μ, r), 1e-12, 1 - 1e-12))
    hi = quantile(Normal(), clamp(_nbpf_cdf(y, μ, r), 1e-12, 1 - 1e-12))
    return lo, hi
end

# Number of cells whose residual falls outside its hand-computed randomisation
# interval (with a small slack for the mode-search tolerance).
function _nbpf_n_outside(R, Y, M, rvec; slack = 1e-4)
    bad = 0
    for s in axes(Y, 2), t in axes(Y, 1)
        isnan(R[t, s]) && continue
        lo, hi = _nbpf_ds_interval(Y[t, s], M[t, s], rvec[t])
        (lo - slack <= R[t, s] <= hi + slack) || (bad += 1)
    end
    return bad
end

# Simulate NB2 counts with per-cell mean matrix M and per-species size r.
function _nbpf_sim(rng, M, r)
    p, n = size(M)
    return [rand(rng, NegativeBinomial(r[t], r[t] / (r[t] + M[t, s]))) for t in 1:p, s in 1:n]
end

# cov2cor(Lambda Lambda'), written out by hand.
function _nbpf_cor(Λ)
    Σ = Λ * Λ'
    p = size(Σ, 1)
    return [Σ[i, j] / sqrt(Σ[i, i] * Σ[j, j]) for i in 1:p, j in 1:p]
end

@testset "#555 NBGroupedFit / NBGroupedCovFit predict, fitted, residuals, getResidualCor" begin

    # ---- Fixture A: NBGroupedFit at known parameters (no optimiser involved) ----
    rngA = MersenneTwister(555)
    p, K, n = 8, 2, 40
    βA = 0.3 .* randn(rngA, p) .+ 1.0
    ΛA = 0.5 .* randn(rngA, p, K)
    ΛA[1, 2] = 0.0
    groupA = [2, 1, 2, 1, 2, 1, 2, 1]          # unsorted: the species -> group map must be used
    rgA = [0.7, 30.0]                          # group 1 very overdispersed, group 2 near-Poisson
    rvecA = rgA[groupA]
    ZA = randn(rngA, K, n)
    YA = _nbpf_sim(rngA, exp.(βA .+ ΛA * ZA), rvecA)
    fitA = NBGroupedFit(βA, ΛA, rgA, groupA, LogLink(), 0.0, true, 0)
    ηA = _nbpf_eta(YA, ΛA, repeat(βA, 1, n), rvecA)
    μA = exp.(ηA)

    @testset "NBGroupedFit: predict link / response and fitted" begin
        L = predict(fitA, YA; type = :link)
        @test size(L) == (p, n)
        @test L ≈ ηA atol = 1e-6
        R = predict(fitA, YA)                                  # default is :response
        @test size(R) == (p, n)
        @test R ≈ μA rtol = 1e-5
        @test all(>(0), R)
        @test predict(fitA, YA; type = :response) == R
        @test predict(fitA, YA; type = :mean) == R
        @test fitted(fitA, YA) == R
        @test_throws ArgumentError predict(fitA, YA; type = :nonsense)
    end

    @testset "NBGroupedFit: offset and mask are honoured like in getLV" begin
        O = 0.4 .* randn(MersenneTwister(1), p, n)
        ηO = _nbpf_eta(YA, ΛA, repeat(βA, 1, n) .+ O, rvecA)
        LO = predict(fitA, YA; type = :link, offset = O)
        @test LO ≈ ηO atol = 1e-6
        @test !isapprox(LO, predict(fitA, YA; type = :link); atol = 1e-3)
        # a site with every cell unobserved has its latent mode at the prior mean z = 0
        mask = trues(p, n)
        mask[:, 3] .= false
        mask[2, 7] = false
        ηM = _nbpf_eta(YA, ΛA, repeat(βA, 1, n), rvecA; mask = mask)
        LM = predict(fitA, YA; type = :link, mask = mask)
        @test LM ≈ ηM atol = 1e-6
        @test LM[:, 3] ≈ βA atol = 1e-8
    end

    @testset "NBGroupedFit: Dunn-Smyth and Pearson residuals" begin
        Rds = residuals(fitA, YA; rng = MersenneTwister(1))
        @test size(Rds) == (p, n)
        @test all(isfinite, Rds)
        # every residual sits inside its randomisation interval under the PER-GROUP size
        @test _nbpf_n_outside(Rds, YA, μA, rvecA) == 0
        # ... and the interval check has power: the wrong size for the same cells fails it
        @test _nbpf_n_outside(Rds, YA, μA, fill(rgA[1], p)) > 0
        # reproducible with a fixed rng, different with another
        @test residuals(fitA, YA; rng = MersenneTwister(1)) == Rds
        @test residuals(fitA, YA; rng = MersenneTwister(2)) != Rds
        Pr = residuals(fitA, YA; type = :pearson)
        @test Pr ≈ (YA .- μA) ./ sqrt.(μA .+ μA .^ 2 ./ rvecA) atol = 1e-4
        @test_throws ArgumentError residuals(fitA, YA; type = :deviance)
        # unobserved cells are NaN, observed cells are untouched
        mask = trues(p, n)
        mask[1, 1] = false
        mask[:, 5] .= false
        Rm = residuals(fitA, YA; mask = mask, rng = MersenneTwister(1))
        @test all(isnan, Rm[.!mask])
        @test all(isfinite, Rm[mask])
    end

    @testset "NBGroupedFit: Dunn-Smyth residuals are roughly N(0,1) on model-simulated data" begin
        rng = MersenneTwister(5551)
        p2, K2, n2 = 12, 1, 300
        β2 = 0.3 .* randn(rng, p2) .+ 1.2
        Λ2 = 0.5 .* randn(rng, p2, K2)
        g2 = repeat([1, 2, 3], 4)
        rg2 = [1.5, 6.0, 40.0]
        Z2 = randn(rng, K2, n2)
        Y2 = _nbpf_sim(rng, exp.(β2 .+ Λ2 * Z2), rg2[g2])
        fit2 = NBGroupedFit(β2, Λ2, rg2, g2, LogLink(), 0.0, true, 0)
        R2 = residuals(fit2, Y2; rng = MersenneTwister(3))
        @test all(isfinite, R2)
        @test abs(mean(R2)) < 0.1
        @test 0.85 < std(R2) < 1.1
        @test 0.02 < mean(abs.(R2) .> 1.96) < 0.09
        for g in 1:3                                   # no group is calibrated at another's expense
            Rg = R2[g2 .== g, :]
            @test abs(mean(Rg)) < 0.15
            @test 0.8 < std(Rg) < 1.15
        end
    end

    @testset "NBGroupedFit: a group at the Poisson limit gives finite, sensible residuals" begin
        # r_group >= 1e6 is how the package flags the Poisson limit, and fitted groups
        # do land there (r ~ 1e9 to 1e24): NegativeBinomial(r, r / (r + mu)) then has
        # prob rounded to 1 and a degenerate CDF, so the residual must not use it blindly.
        rgP = [2.0, 1e20]
        fitP = NBGroupedFit(βA, ΛA, rgP, groupA, LogLink(), 0.0, true, 0)
        μP = predict(fitP, YA)
        RP = residuals(fitP, YA; rng = MersenneTwister(4))
        @test all(isfinite, RP)
        @test maximum(abs, RP) < 5
        @test _nbpf_n_outside(RP, YA, μP, rgP[groupA]) == 0
        PP = residuals(fitP, YA; type = :pearson)
        @test all(isfinite, PP)
    end

    # ---- Fixture C: NBGroupedCovFit at known parameters, species-specific design ----
    rngC = MersenneTwister(556)
    pC, KC, nC, qC = 6, 1, 35, 2
    βC = 0.3 .* randn(rngC, pC) .+ 1.0
    ΛC = 0.5 .* randn(rngC, pC, KC)
    γC = [0.4, -0.3]
    groupC = [1, 1, 2, 2, 3, 3]
    rgC = [1.0, 5.0, 25.0]
    rvecC = rgC[groupC]
    XC = randn(rngC, pC, nC, qC)                   # differs across species and sites
    OC = [sum(XC[t, s, k] * γC[k] for k in 1:qC) for t in 1:pC, s in 1:nC]
    ZC = randn(rngC, KC, nC)
    YC = _nbpf_sim(rngC, exp.(βC .+ OC .+ ΛC * ZC), rvecC)
    fitC = NBGroupedCovFit(βC, γC, falses(qC), ΛC, rgC, groupC, LogLink(), 0.0, true, 0)
    ηC = _nbpf_eta(YC, ΛC, βC .+ OC, rvecC)
    μC = exp.(ηC)

    @testset "NBGroupedCovFit: predict link / response and fitted use the covariate design" begin
        L = predict(fitC, YC, XC; type = :link)
        @test size(L) == (pC, nC)
        @test L ≈ ηC atol = 1e-6
        R = predict(fitC, YC, XC)                              # default is :response
        @test R ≈ μC rtol = 1e-5
        @test predict(fitC, YC, XC; type = :response) == R
        @test predict(fitC, YC, XC; type = :mean) == R
        @test fitted(fitC, YC, XC) == R
        @test_throws ArgumentError predict(fitC, YC, XC; type = :nonsense)
        # the design matters: zero covariates give the intercept-and-latent-only predictor
        fit0 = NBGroupedFit(βC, ΛC, rgC, groupC, LogLink(), 0.0, true, 0)
        L0 = predict(fitC, YC, zero(XC); type = :link)
        @test L0 ≈ predict(fit0, YC; type = :link) atol = 1e-8
        @test !isapprox(L0, L; atol = 1e-2)
        # X gamma is the offset of the shared-size route
        @test predict(fit0, YC; type = :link, offset = OC) ≈ L atol = 1e-8
        # a design whose covariate count disagrees with gamma is refused
        @test_throws DimensionMismatch predict(fitC, YC, XC[:, :, 1:1])
    end

    @testset "NBGroupedCovFit: a fixed-at-zero coefficient ignores its covariate" begin
        γF = [0.4, 0.0]
        fitF = NBGroupedCovFit(βC, γF, [false, true], ΛC, rgC, groupC, LogLink(), 0.0, true, 0)
        OF = [XC[t, s, 1] * γF[1] for t in 1:pC, s in 1:nC]
        ηF = _nbpf_eta(YC, ΛC, βC .+ OF, rvecC)
        @test predict(fitF, YC, XC; type = :link) ≈ ηF atol = 1e-6
        XC2 = copy(XC)
        XC2[:, :, 2] .= 7.0
        @test predict(fitF, YC, XC2; type = :link) ≈ predict(fitF, YC, XC; type = :link) atol = 1e-8
    end

    @testset "NBGroupedCovFit: Dunn-Smyth and Pearson residuals" begin
        Rds = residuals(fitC, YC, XC; rng = MersenneTwister(1))
        @test size(Rds) == (pC, nC)
        @test all(isfinite, Rds)
        @test _nbpf_n_outside(Rds, YC, μC, rvecC) == 0
        @test _nbpf_n_outside(Rds, YC, μC, fill(rgC[1], pC)) > 0
        @test residuals(fitC, YC, XC; rng = MersenneTwister(1)) == Rds
        @test residuals(fitC, YC, XC; rng = MersenneTwister(2)) != Rds
        Pr = residuals(fitC, YC, XC; type = :pearson)
        @test Pr ≈ (YC .- μC) ./ sqrt.(μC .+ μC .^ 2 ./ rvecC) atol = 1e-4
        @test_throws ArgumentError residuals(fitC, YC, XC; type = :deviance)
        mask = trues(pC, nC)
        mask[2, 4] = false
        Rm = residuals(fitC, YC, XC; mask = mask, rng = MersenneTwister(1))
        @test isnan(Rm[2, 4])
        @test all(isfinite, Rm[mask])
    end

    @testset "NBGroupedCovFit: Dunn-Smyth residuals are roughly N(0,1) on model-simulated data" begin
        rng = MersenneTwister(5561)
        p2, n2 = 12, 300
        β2 = 0.3 .* randn(rng, p2) .+ 1.2
        Λ2 = 0.5 .* randn(rng, p2, 1)
        γ2 = [0.35, -0.25]
        g2 = repeat([1, 2, 3], 4)
        rg2 = [1.5, 6.0, 40.0]
        X2 = randn(rng, p2, n2, 2)
        O2 = [X2[t, s, 1] * γ2[1] + X2[t, s, 2] * γ2[2] for t in 1:p2, s in 1:n2]
        Z2 = randn(rng, 1, n2)
        Y2 = _nbpf_sim(rng, exp.(β2 .+ O2 .+ Λ2 * Z2), rg2[g2])
        fit2 = NBGroupedCovFit(β2, γ2, falses(2), Λ2, rg2, g2, LogLink(), 0.0, true, 0)
        R2 = residuals(fit2, Y2, X2; rng = MersenneTwister(3))
        @test all(isfinite, R2)
        @test abs(mean(R2)) < 0.1
        @test 0.85 < std(R2) < 1.1
        @test 0.02 < mean(abs.(R2) .> 1.96) < 0.09
    end

    @testset "getResidualCor: cov2cor(Lambda Lambda'), symmetric, unit diagonal" begin
        for (fit, Λ) in ((fitA, ΛA), (fitC, ΛC))
            pp = size(Λ, 1)
            Rc = getResidualCor(fit)
            @test Rc isa Matrix{Float64}
            @test size(Rc) == (pp, pp)
            @test issymmetric(Rc)
            @test all(==(1.0), diag(Rc))
            @test all(abs.(Rc) .<= 1 + 1e-12)
            @test Rc ≈ _nbpf_cor(Λ)
            @test getResidualCor(fit; level = :unit) == Rc
            @test getResidualCor(fit; level = :B) == Rc          # legacy alias
            @test_throws ArgumentError getResidualCor(fit; level = :site)
            @test_throws ArgumentError getResidualCor(fit; level = :nonsense)
        end
        # ΛΛ' is rotation invariant, so a rotated loadings matrix gives the same correlation
        Q = Matrix(qr(randn(MersenneTwister(9), K, K)).Q)
        fitAr = NBGroupedFit(βA, ΛA * Q, rgA, groupA, LogLink(), 0.0, true, 0)
        @test getResidualCor(fitAr) ≈ getResidualCor(fitA) atol = 1e-12
        # the sign convention of the loadings does not matter either
        fitCs = NBGroupedCovFit(βC, γC, falses(qC), -ΛC, rgC, groupC, LogLink(), 0.0, true, 0)
        @test getResidualCor(fitCs) ≈ getResidualCor(fitC) atol = 1e-12
    end

    # ---- The issue's call pattern on real fits (species-specific slopes by block expansion) ----
    @testset "fitted grouped NB2 fits: the calls from the issue now work and agree with eta from getLV" begin
        rng = MersenneTwister(20266928)
        p3, n3, K3 = 6, 70, 1
        β3 = 0.2 .+ 0.35 .* randn(rng, p3)
        Λ3 = 0.45 .* randn(rng, p3, K3)
        slope = 0.25 .* randn(rng, p3)
        x1 = randn(rng, n3)
        Xb = zeros(p3, n3, p3)                       # block expansion: slope of species t on column t
        for t in 1:p3
            Xb[t, :, t] .= x1
        end
        Z3 = randn(rng, K3, n3)
        M3 = exp.(β3 .+ slope .* x1' .+ Λ3 * Z3)
        Y3 = _nbpf_sim(rng, M3, fill(2.0, p3))

        f3 = fit_nb_gllvm_grouped_cov(Y3; X = Xb, K = K3, group = collect(1:p3))
        @test f3 isa NBGroupedCovFit
        Zhat = getLV(f3, Y3, Xb; rotate = false)
        ηhand = f3.β .+ reshape(reshape(Xb, p3 * n3, p3) * f3.γ, p3, n3) .+ f3.Λ * Zhat'
        @test predict(f3, Y3, Xb; type = :link) ≈ ηhand atol = 1e-8
        @test predict(f3, Y3, Xb) ≈ exp.(ηhand) rtol = 1e-8
        @test fitted(f3, Y3, Xb) ≈ exp.(ηhand) rtol = 1e-8
        R3 = residuals(f3, Y3, Xb; rng = MersenneTwister(1))
        @test size(R3) == (p3, n3)
        @test all(isfinite, R3)
        @test abs(mean(R3)) < 0.2
        @test 0.7 < std(R3) < 1.2
        C3 = getResidualCor(f3)
        @test size(C3) == (p3, p3) && issymmetric(C3)
        @test C3 ≈ _nbpf_cor(f3.Λ)

        f2 = fit_nb_gllvm_grouped(Y3; K = K3, group = collect(1:p3))
        @test f2 isa NBGroupedFit
        Z2hat = getLV(f2, Y3; rotate = false)
        η2 = f2.β .+ f2.Λ * Z2hat'
        @test predict(f2, Y3; type = :link) ≈ η2 atol = 1e-8
        @test fitted(f2, Y3) ≈ exp.(η2) rtol = 1e-8
        R2 = residuals(f2, Y3; rng = MersenneTwister(1))
        @test size(R2) == (p3, n3)
        @test all(isfinite, R2)
        @test abs(mean(R2)) < 0.2
        @test 0.7 < std(R2) < 1.2
        @test getResidualCor(f2) ≈ _nbpf_cor(f2.Λ)
    end
end
