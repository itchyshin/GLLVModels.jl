# test_x_covariate_parity.jl — shared site-X light logLik vs gllvmTMB
#
# Included by runparity.jl AFTER no-X cells. NEVER included by runtests.jl.
# Cohort 1: Gaussian / Binomial / Poisson with q=1 shared site covariate (#170).
# Cohort 2 (Arc 2): NB2 / Beta with q=1 shared site-X + per-trait dispersion,
# using Arc 1 fit_nb_gllvm_grouped_cov / fit_beta_gllvm_grouped_cov
# (group=collect(1:p), default hessian=:observed — twin API B under X, per
# docs/dev-log/decisions/2026-08-02-nb2-beta-x-dispersion-identity.md).
# Cohort 3 (Gamma Arc 2): Gamma with q=1 shared site-X + per-trait shape α,
# using fit_gamma_gllvm_grouped_cov (default hessian=:observed — twin API B,
# docs/dev-log/decisions/2026-08-03-gamma-x-dispersion-identity.md).
# Cohort 4 (Ordinal Arc 2): Ordinal with q=1 shared site-X + per-trait
# cutpoints (τ₁=0 / K−2), using fit_ordinal_gllvm_pertrait_cov + ProbitLink
# vs gllvmTMB ordinal_probit
# (docs/dev-log/decisions/2026-08-03-ordinal-x-cutpoint-identity.md).
# Cohort 5 (BetaBinomial Arc 1+2): BetaBinomial with q=1 shared site-X +
# per-trait precision φ + trials N, using fit_beta_binomial_gllvm_grouped_cov
# vs gllvmTMB betabinomial() (trials threaded via `weights`, API B under X;
# docs/dev-log/decisions/2026-08-05-betabinomial-x-dispersion-identity.md).
# Fence: X_lv, shared-φ-Julia-vs-per-trait-R, shared-cutpoint Option B.

using GLLVModels, RCall, Test, Random, LinearAlgebra, Statistics, Distributions

# Knuth sampler — matches test_poisson_parity.jl (no Distributions dep).
function _rand_poisson_x(λ::Float64)
    λ = clamp(λ, 0.0, 1e6)
    L = exp(-λ)
    k = 0
    prod = 1.0
    while true
        k += 1
        prod *= rand()
        prod <= L && return k - 1
    end
end

# Marsaglia–Tsang gamma sampler — matches test_negbin_parity.jl (no Distributions dep).
function _rand_gamma_x(shape::Float64, scale::Float64)
    if shape < 1.0
        return _rand_gamma_x(shape + 1.0, scale) * rand()^(1.0 / shape)
    end
    d = shape - 1.0 / 3.0
    c = 1.0 / sqrt(9.0 * d)
    while true
        z = randn()
        v = (1.0 + c * z)^3
        v <= 0 && continue
        u = rand()
        z2 = z * z
        u < 1.0 - 0.0331 * z2 * z2 && return d * v * scale
        logu = log(u)
        logu < 0.5 * z2 + d * (1.0 - v + log(v)) && return d * v * scale
    end
end

# Gamma–Poisson compound: NB2 with mean μ, dispersion r (Var = μ + μ²/r).
function _rand_nb2_x(μ::Float64, r::Float64)
    λ = _rand_gamma_x(r, μ / r)
    return _rand_poisson_x(λ)
end

# Johnk's algorithm — Beta(a,b) for a,b > 0, matches test_beta_parity.jl.
function _rand_beta_x(a::Float64, b::Float64)
    a = max(a, 1e-12)
    b = max(b, 1e-12)
    while true
        u = rand()
        v = rand()
        xu = u^(1 / a)
        yv = v^(1 / b)
        s = xu + yv
        if s <= 1.0 && s > 0.0
            return xu / s
        end
    end
end

@testset "Shared site-X light logLik: GLLVModels.jl vs gllvmTMB" begin

    @testset "Gaussian + shared X (q=1)" begin
        # Stored seed-420 draw (not re-simulated): Julia's seeded randn stream differs
        # between 1.10.12 and 1.13.2. x (demeaned) and y (centred per trait) come from
        # fixtures/frozen_rng_xcov_gaussian_s420.toml, generated on 1.10.12 by
        # fixtures/generate_frozen_rng_w15.jl from the original DGP: seed 420,
        # Λ = parity_loadings_p5k2(), γ = 0.55, x = randn(n) demeaned,
        # y = γ .* x' .+ Λ * randn(K, n) .+ 0.7 .* randn(p, n), then centred per trait.
        # They are stored AFTER demeaning/centring because `mean` differs at the last bit
        # between the two Julia versions (2.2e-16 here; the raw draws are identical).
        p, K, n = 5, 2, 30
        frozen = parity_load_frozen_rng("frozen_rng_xcov_gaussian_s420.toml";
            expected_sha = "7a8c62d79eeb1a0bb915dfbce0d898c403600d2c0adcaf919d1701a5d9a1ee13", expected_x_sha = "2a8fac260283493905c72d59b23e988ac51a1128bc6b45d322da4cd60cfb1b38")
        # Julia Gaussian+X has shared β only (no per-trait intercepts in Xβ).
        # Match no-X oracle: site x demeaned; Y centred per trait so R `0+trait`
        # intercepts do not shift the objective (J1 zero-mean alignment). Both are
        # already done in the stored arrays.
        x = frozen.x
        X = parity_site_design(x, p)
        y = frozen.Y

        jl_fit = fit_gaussian_gllvm(y; K = K, X = X)
        @test jl_fit.converged
        @test isfinite(jl_fit.logLik)
        jl_logL = jl_fit.logLik

        r = fit_gllvmtmb_parity_loglik_x(y, x, K; family = :gaussian)
        @test r.converged
        @test isfinite(r.logLik)

        print_parity_loglik(
            "Gaussian+X logLik oracle (seed=420, p=$p, K=$K, n=$n, q=1 shared)";
            jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
        )

        @testset "log-likelihood agreement (rtol=1e-6)" begin
            @test jl_logL ≈ r.logLik rtol = 1e-6
            @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
        end
    end

    @testset "Binomial + shared X (q=1, Bernoulli)" begin
        # Milder loadings + larger n than the first probe: seed=421 / n=30 hit an
        # R runaway-loading (Heywood) warning and a large ΔlogLik. Keep rtol=1e-6.
        Random.seed!(431)
        p, K, n = 5, 2, 80
        β = [-0.3, 0.0, 0.3, -0.15, 0.2]
        Λ = 0.35 .* parity_loadings_p5k2()
        γ = 0.5
        x = randn(n)
        X = parity_site_design(x, p)
        Z = randn(K, n)
        η = β .+ γ .* x' .+ Λ * Z
        Y = [rand() < 1 / (1 + exp(-η[t, s])) ? 1 : 0 for t in 1:p, s in 1:n]

        jl_fit = fit_gllvm_cov(Y; family = GLLVModels.Binomial(), X = X, K = K)
        @test jl_fit.converged
        @test isfinite(jl_fit.loglik)
        jl_logL = jl_fit.loglik

        r = fit_gllvmtmb_parity_loglik_x(Y, x, K; family = :binomial)
        @test r.converged
        @test isfinite(r.logLik)

        print_parity_loglik(
            "Binomial+X logLik oracle (seed=431, p=$p, K=$K, n=$n, q=1 shared)";
            jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
        )

        @testset "log-likelihood agreement (rtol=1e-6)" begin
            @test jl_logL ≈ r.logLik rtol = 1e-6
            @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
        end
    end

    @testset "Poisson + shared X (q=1)" begin
        Random.seed!(422)
        p, K, n = 5, 2, 30
        β = log.([2.5, 4.0, 2.0, 3.5, 3.0])
        Λ = 0.4 .* parity_loadings_p5k2()
        γ = 0.6
        x = randn(n)
        X = parity_site_design(x, p)
        Z = randn(K, n)
        η = β .+ γ .* x' .+ Λ * Z
        Y = [_rand_poisson_x(exp(clamp(η[t, s], -8.0, 8.0))) for t in 1:p, s in 1:n]

        jl_fit = fit_gllvm_cov(Y; family = GLLVModels.Poisson(), X = X, K = K)
        @test jl_fit.converged
        @test isfinite(jl_fit.loglik)
        jl_logL = jl_fit.loglik

        r = fit_gllvmtmb_parity_loglik_x(Y, x, K; family = :poisson)
        @test r.converged
        @test isfinite(r.logLik)

        print_parity_loglik(
            "Poisson+X logLik oracle (seed=422, p=$p, K=$K, n=$n, q=1 shared)";
            jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
        )

        @testset "log-likelihood agreement (rtol=1e-6)" begin
            @test jl_logL ≈ r.logLik rtol = 1e-6
            @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
        end
    end

    @testset "NB2 + shared X (q=1)" begin
        # Per-trait r (group=collect(1:p)) + shared site-X slope γ — twin API B
        # under X (docs/dev-log/decisions/2026-08-02-nb2-beta-x-dispersion-identity.md).
        # Default hessian=:observed (NOT :fisher — that's only for the Arc 1
        # identity tests vs shared fit_gllvm_cov, a different estimand).
        # DGP repair note: an initial K=2/r_true=6/n=60 draw left one trait's
        # per-trait r running to a near-Poisson boundary (Heywood-like — same
        # instability mode Fisher flagged for NB2+X), producing a genuine
        # optimizer disagreement (R "relative convergence" but a different
        # local optimum) rather than numerical noise. K=1 + strong
        # overdispersion (r_true=1.5) + n=120 keeps every per-trait r
        # well-identified (Δ ~1e-8 at seed=45).
        Random.seed!(45)
        p, K, n = 5, 1, 120
        β = log.([2.5, 3.0, 2.0, 2.8, 2.2])
        r_true = 1.5
        Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
        γ = 0.4
        x = randn(n)
        X = parity_site_design(x, p)
        Z = randn(K, n)
        η = β .+ γ .* x' .+ Λ * Z
        Y = [_rand_nb2_x(exp(clamp(η[t, s], -8.0, 8.0)), r_true) for t in 1:p, s in 1:n]

        jl_fit = fit_nb_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
        @test jl_fit isa NBGroupedCovFit
        @test jl_fit.converged
        @test isfinite(jl_fit.loglik)
        @test length(jl_fit.r_group) == p
        jl_logL = jl_fit.loglik

        r = fit_gllvmtmb_parity_loglik_x(Y, x, K; family = :negbinomial)
        @test r.converged
        @test isfinite(r.logLik)

        print_parity_loglik(
            "NB2+X logLik oracle (seed=45, p=$p, K=$K, n=$n, q=1 shared, per-trait r)";
            jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
        )

        @testset "log-likelihood agreement (rtol=1e-6)" begin
            @test jl_logL ≈ r.logLik rtol = 1e-6
            @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
        end
    end

    @testset "Beta + shared X (q=1)" begin
        # Per-trait φ (group=collect(1:p)) + shared site-X slope γ — twin API B
        # under X. Default hessian=:observed.
        # DGP repair note: an initial φ_true=12/n=60 draw left one trait's
        # per-trait φ running to a near-degenerate boundary (Heywood-like),
        # producing R "false convergence (8)" even though the reported logLik
        # itself still agreed to ~5e-8 relative. Milder loadings + φ_true=8 +
        # n=80 keeps every per-trait φ well-identified and R converges cleanly.
        Random.seed!(45)
        p, K, n = 5, 1, 80
        β = [0.30, -0.20, 0.25, -0.15, 0.05]
        φ_true = 8.0
        Λ = 0.1 .* parity_loadings_p5k2()[:, 1:K]
        γ = 0.35
        x = randn(n)
        X = parity_site_design(x, p)
        Z = randn(K, n)
        η = β .+ γ .* x' .+ Λ * Z
        Y = [
            begin
                μ = clamp(1 / (1 + exp(-η[t, s])), 1e-4, 1 - 1e-4)
                _rand_beta_x(μ * φ_true, (1 - μ) * φ_true)
            end
            for t in 1:p, s in 1:n
        ]

        jl_fit = fit_beta_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
        @test jl_fit isa BetaGroupedCovFit
        @test jl_fit.converged
        @test isfinite(jl_fit.loglik)
        @test length(jl_fit.φ) == p
        jl_logL = jl_fit.loglik

        r = fit_gllvmtmb_parity_loglik_x(Y, x, K; family = :beta)
        @test r.converged
        @test isfinite(r.logLik)

        print_parity_loglik(
            "Beta+X logLik oracle (seed=45, p=$p, K=$K, n=$n, q=1 shared, per-trait φ)";
            jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
        )

        @testset "log-likelihood agreement (rtol=1e-6)" begin
            @test jl_logL ≈ r.logLik rtol = 1e-6
            @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
        end
    end

    @testset "Gamma + shared X (q=1)" begin
        # Per-trait shape α (group=collect(1:p)) + shared site-X slope γ —
        # twin API B under X
        # (docs/dev-log/decisions/2026-08-03-gamma-x-dispersion-identity.md).
        # Default hessian=:observed (NOT :fisher — that's only for Arc 1
        # identity vs shared fit_gllvm_cov). Arc 2 unblocker: grouped Gamma
        # Laplace now defaults to TMB observed curvature (W=α y/μ); Fisher-only
        # was systematically Δ≈0.2–1 vs gllvmTMB.
        # DGP follows NB2/Beta Arc 2 lesson: K=1, mild loadings, n large enough
        # that every per-trait α stays interior (no Heywood / rtol widen).
        Random.seed!(46)
        p, K, n = 5, 1, 120
        β = log.([2.0, 2.5, 1.8, 2.2, 2.1])
        α_true = 2.5
        Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
        γ = 0.4
        x = randn(n)
        X = parity_site_design(x, p)
        Z = randn(K, n)
        η = β .+ γ .* x' .+ Λ * Z
        Y = [
            begin
                μ = exp(clamp(η[t, s], -4.0, 4.0))
                _rand_gamma_x(α_true, μ / α_true) + 1e-6
            end
            for t in 1:p, s in 1:n
        ]

        jl_fit = fit_gamma_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
        @test jl_fit isa GammaGroupedCovFit
        @test jl_fit.converged
        @test isfinite(jl_fit.loglik)
        @test length(jl_fit.α) == p
        jl_logL = jl_fit.loglik

        r = fit_gllvmtmb_parity_loglik_x(Y, x, K; family = :gamma)
        @test r.converged
        @test isfinite(r.logLik)

        print_parity_loglik(
            "Gamma+X logLik oracle (seed=46, p=$p, K=$K, n=$n, q=1 shared, per-trait α)";
            jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
        )

        @testset "log-likelihood agreement (rtol=1e-6)" begin
            @test jl_logL ≈ r.logLik rtol = 1e-6
            @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
        end
    end

    @testset "NB1 + shared X (q=1)" begin
        # Per-trait linear-variance φ (group=collect(1:p)) + shared site-X γ —
        # twin API B under X
        # (docs/dev-log/decisions/2026-08-05-nb1-x-dispersion-identity.md).
        # Default hessian=:observed matches TMB; identity cells use :fisher.
        Random.seed!(48)
        p, K, n = 5, 1, 120
        β = log.([1.8, 2.2, 1.6, 2.0, 1.9])
        φ_true = 0.85
        Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
        γ = 0.4
        x = randn(n)
        X = parity_site_design(x, p)
        Z = randn(K, n)
        η = β .+ γ .* x' .+ Λ * Z
        Y = Matrix{Int}(undef, p, n)
        for t in 1:p, s in 1:n
            μ = exp(clamp(η[t, s], -3.5, 3.5))
            Y[t, s] = rand(NegativeBinomial(μ / φ_true, 1 / (1 + φ_true)))
        end

        jl_fit = fit_nb1_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
        @test jl_fit isa NB1GroupedCovFit
        @test jl_fit.converged
        @test isfinite(jl_fit.loglik)
        @test length(jl_fit.φ) == p
        jl_logL = jl_fit.loglik

        r = fit_gllvmtmb_parity_loglik_x(Float64.(Y), x, K; family = :nb1)
        @test r.converged
        @test isfinite(r.logLik)

        print_parity_loglik(
            "NB1+X logLik oracle (seed=48, p=$p, K=$K, n=$n, q=1 shared, per-trait φ)";
            jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
        )

        @testset "log-likelihood agreement (rtol=1e-6)" begin
            @test jl_logL ≈ r.logLik rtol = 1e-6
            @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
        end
    end

    @testset "Ordinal-probit + shared X (q=1)" begin
        # Per-trait cutpoints (τ₁=0 / K−2 free) + shared site-X γ —
        # twin API under X
        # (docs/dev-log/decisions/2026-08-03-ordinal-x-cutpoint-identity.md).
        # ProbitLink matches gllvmTMB::ordinal_probit(). Mild loadings / K=1
        # to avoid Heywood; never widen rtol.
        Random.seed!(47)
        p, K, n, C = 5, 1, 80, 3
        β = [0.30, -0.20, 0.15, -0.10, 0.05]
        Λ = 0.35 .* reshape([0.8, 0.5, 0.3, -0.2, 0.1], p, K)
        τ = [0.0, 0.75]
        γ = 0.35
        x = randn(n)
        X = parity_site_design(x, p)
        Z = randn(K, n)
        η = β .+ γ .* x' .+ Λ * Z
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n, t in 1:p
            u = rand()
            Y[t, s] = u < GLLVModels._ord_F(τ[1] - η[t, s], ProbitLink()) ? 1 :
                      u < GLLVModels._ord_F(τ[2] - η[t, s], ProbitLink()) ? 2 : 3
        end

        jl_fit = fit_ordinal_gllvm_pertrait_cov(
            Y; X = X, K = K, link = ProbitLink(),
            g_tol = 1e-7, iterations = 1_000,
        )
        @test jl_fit isa OrdinalPerTraitCovFit
        @test jl_fit.converged
        @test isfinite(jl_fit.loglik)
        jl_logL = jl_fit.loglik

        r = fit_gllvmtmb_parity_loglik_x(Y, x, K; family = :ordinal)
        @test r.converged
        @test isfinite(r.logLik)

        print_parity_loglik(
            "Ordinal-probit+X logLik oracle (seed=47, p=$p, K=$K, n=$n, C=$C, q=1 shared, per-trait τ)";
            jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
        )

        @testset "log-likelihood agreement (rtol=1e-6)" begin
            @test jl_logL ≈ r.logLik rtol = 1e-6
            @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
        end
    end

    @testset "BetaBinomial + shared X (q=1)" begin
        # Per-trait Beta precision φ (group=collect(1:p)) + shared site-X γ +
        # trials N — twin API B under X
        # (docs/dev-log/decisions/2026-08-05-betabinomial-x-dispersion-identity.md).
        # Mirrors the NB2/Beta/Gamma/NB1 Arc 2 DGP shape: K=1, mild loadings,
        # trials N=8 (as in test/test_betabinomial_x_identity.jl), n large
        # enough that every per-trait φ stays interior (no Heywood / rtol
        # widen). Sum-of-Bernoulli binomial draw — small N keeps it cheap.
        Random.seed!(49)
        p, K, n = 5, 1, 120
        β = [0.30, -0.20, 0.25, -0.15, 0.05]
        φ_true = 8.0
        N = fill(8, p, n)
        Λ = 0.2 .* parity_loadings_p5k2()[:, 1:K]
        γ = 0.35
        x = randn(n)
        X = parity_site_design(x, p)
        Z = randn(K, n)
        η = β .+ γ .* x' .+ Λ * Z
        Y = Matrix{Int}(undef, p, n)
        for t in 1:p, s in 1:n
            μ = clamp(1 / (1 + exp(-η[t, s])), 1e-4, 1 - 1e-4)
            psucc = clamp(_rand_beta_x(μ * φ_true, (1 - μ) * φ_true), 1e-6, 1 - 1e-6)
            Y[t, s] = count(_ -> rand() < psucc, 1:N[t, s])
        end

        jl_fit = fit_beta_binomial_gllvm_grouped_cov(Y; X = X, K = K, N = N,
                                                      group = collect(1:p))
        @test jl_fit isa BetaBinomialGroupedCovFit
        @test jl_fit.converged
        @test isfinite(jl_fit.loglik)
        @test length(jl_fit.φ) == p
        jl_logL = jl_fit.loglik

        r = fit_gllvmtmb_parity_loglik_x(Y, x, K; family = :betabinomial, N = N)
        @test r.converged
        @test isfinite(r.logLik)

        print_parity_loglik(
            "BetaBinomial+X logLik oracle (seed=49, p=$p, K=$K, n=$n, q=1 shared, per-trait φ, N=8)";
            jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
        )

        @testset "log-likelihood agreement (rtol=1e-6)" begin
            @test jl_logL ≈ r.logLik rtol = 1e-6
            @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
        end
    end

end
