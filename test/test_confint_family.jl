using GLLVModels, Test, Random, Distributions, Statistics, LinearAlgebra

# Helper: simulate a Poisson GLLVModels dataset (p×n integer counts).
function _sim_poisson(p, K, n; seed = 11)
    Random.seed!(seed)
    β = 0.5 .* randn(p) .+ 1.0
    Λ = 0.5 .* randn(p, K)
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
        η = β .+ Λ * randn(K)
        for t in 1:p
            Y[t, s] = rand(Poisson(exp(η[t])))
        end
    end
    return Y, β, Λ
end

@testset "Non-Gaussian confidence intervals" begin
    @testset "Wald (Poisson)" begin
        Y, _, _ = _sim_poisson(5, 1, 140; seed = 21)
        fit = fit_poisson_gllvm(Y; K = 1)
        ci = confint(fit, Y; method = :wald)
        @test ci.method === :wald
        @test length(ci.term) == length(fit.β) + (5 * 1)   # β + Λ entries
        # estimate equals the MLE; finite intervals bracket it (a finite-difference
        # Hessian need not be globally PD, so we don't demand pd_hessian)
        @test ci.estimate[1] ≈ fit.β[1] atol = 1e-8
        fin = isfinite.(ci.se)
        @test any(fin)
        @test all(ci.lower[fin] .< ci.estimate[fin] .< ci.upper[fin])

        # parm subsetting
        ci_b1 = confint(fit, Y; method = :wald, parm = "beta[1]")
        @test ci_b1.term == ["beta[1]"]
        ci_b = confint(fit, Y; method = :wald, parm = "beta")
        @test length(ci_b.term) == 5
    end

    @testset "Profile (Poisson)" begin
        Y, _, _ = _sim_poisson(4, 1, 120; seed = 22)
        fit = fit_poisson_gllvm(Y; K = 1)
        ci = confint(fit, Y; method = :profile, parm = "beta[1]",
                     profile_iterations = 200,
                     profile_g_tol = 1e-4,
                     profile_max_expand = 20,
                     profile_max_bisect = 30)
        @test ci.method === :profile
        @test ci.status[1] in (:profile, :partial)
        @test isfinite(ci.lower[1]) || isfinite(ci.upper[1])   # at least one side bracketed
        isfinite(ci.lower[1]) && @test ci.lower[1] < ci.estimate[1]
        isfinite(ci.upper[1]) && @test ci.estimate[1] < ci.upper[1]
        @test_throws ArgumentError confint(fit, Y; method = :profile,
                                           parm = "beta[1]",
                                           profile_iterations = 0)
        @test_throws ArgumentError confint(fit, Y; method = :profile,
                                           parm = "beta[1]",
                                           profile_g_tol = Inf)

        # when both sides bracket, the profile interval should be in the Wald ballpark
        if ci.status[1] === :profile
            w = confint(fit, Y; method = :wald, parm = "beta[1]")
            @test isapprox(ci.lower[1], w.lower[1]; atol = 0.4)
            @test isapprox(ci.upper[1], w.upper[1]; atol = 0.4)
        end
    end

    @testset "Bootstrap (Poisson) — single- vs multi-core identical" begin
        Y, _, _ = _sim_poisson(4, 1, 120; seed = 23)
        fit = fit_poisson_gllvm(Y; K = 1)
        ci_serial = confint(fit, Y; method = :bootstrap, n_boot = 30, seed = 7, parallel = false)
        ci_par    = confint(fit, Y; method = :bootstrap, n_boot = 30, seed = 7, parallel = true)
        @test ci_serial.method === :bootstrap
        @test ci_serial.n_converged ≥ 12
        # per-replicate RNG seeding ⇒ results independent of threading
        @test ci_serial.lower == ci_par.lower
        @test ci_serial.upper == ci_par.upper
        @test ci_serial.n_converged == ci_par.n_converged
        # percentile bounds are ordered (they need NOT bracket the MLE: for K=1 the
        # loadings have a sign-flip non-identifiability, so bootstrap replicates mix
        # +Λ and −Λ and the interval can legitimately exclude the point estimate)
        @test all(ci_serial.lower .<= ci_serial.upper)
    end

    @testset "Dispersion CI on the natural scale (NB)" begin
        Random.seed!(24)
        p, K, n, r_true = 4, 1, 200, 6.0
        β = 0.4 .* randn(p) .+ 1.2
        Λ = 0.4 .* randn(p, K)
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n
            η = β .+ Λ * randn(K)
            for t in 1:p
                μ = exp(η[t])
                Y[t, s] = rand(NegativeBinomial(r_true, r_true / (r_true + μ)))
            end
        end
        fit = fit_nb_gllvm(Y; K = K)
        ci = confint(fit, Y; method = :wald, parm = "r")
        @test ci.term == ["r"]
        @test ci.estimate[1] ≈ fit.r atol = 1e-8
        # log-scale parameterisation ⇒ strictly positive natural-scale bounds
        if isfinite(ci.lower[1])
            @test 0 < ci.lower[1] < ci.estimate[1] < ci.upper[1]
        end
    end

    @testset "Gamma dispersion bracketed by Wald" begin
        Random.seed!(25)
        p, K, n, α_true = 4, 1, 200, 5.0
        β = 0.3 .* randn(p)
        Λ = 0.4 .* randn(p, K)
        Y = Matrix{Float64}(undef, p, n)
        for s in 1:n
            η = β .+ Λ * randn(K)
            for t in 1:p
                μ = exp(η[t])
                Y[t, s] = rand(Gamma(α_true, μ / α_true))
            end
        end
        fit = fit_gamma_gllvm(Y; K = K)
        ci = confint(fit, Y; method = :wald, parm = "alpha")
        @test ci.term == ["alpha"]
        @test ci.estimate[1] ≈ fit.α atol = 1e-8
        @test isfinite(ci.estimate[1])
    end

    @testset "bad method errors" begin
        Y, _, _ = _sim_poisson(3, 1, 60; seed = 26)
        fit = fit_poisson_gllvm(Y; K = 1)
        @test_throws ArgumentError confint(fit, Y; method = :nope)
    end

    @testset "Two-part: Hurdle-Poisson Wald + profile" begin
        Random.seed!(31)
        p, K, n = 4, 1, 160
        βz = 0.4 .* randn(p) .+ 0.5; βc = 0.3 .* randn(p) .+ 1.0
        Λc = 0.4 .* randn(p, K)
        Y = zeros(Int, p, n)
        for s in 1:n
            ηc = βc .+ Λc * randn(K)
            for t in 1:p
                if rand() < inv(1 + exp(-βz[t]))
                    y = 0; while y == 0; y = rand(Poisson(exp(ηc[t]))); end
                    Y[t, s] = y
                end
            end
        end
        fit = fit_hurdle_poisson_gllvm(Y; K = K)
        ci = confint(fit, Y; method = :wald)
        @test length(ci.term) == 2p + (p * K)         # βz + βc + Λ
        @test "betaz[1]" in ci.term && "betac[1]" in ci.term
        w = confint(fit, Y; method = :wald, parm = "betac[1]")
        @test w.term == ["betac[1]"]
        @test w.estimate[1] ≈ fit.βc[1] atol = 1e-8
        pr = confint(fit, Y; method = :profile, parm = "betac[1]")
        @test pr.method === :profile
        @test isfinite(pr.lower[1]) || isfinite(pr.upper[1])
        isfinite(pr.lower[1]) && @test pr.lower[1] < pr.estimate[1]
        isfinite(pr.upper[1]) && @test pr.estimate[1] < pr.upper[1]
    end

    @testset "Two-part: Delta-lognormal Wald σ on natural scale" begin
        Random.seed!(32)
        p, K, n = 4, 1, 200; σ_true = 0.5
        βz = 0.3 .* randn(p) .+ 0.6; βc = 0.4 .* randn(p)
        Λc = 0.4 .* randn(p, K)
        Y = zeros(p, n)
        for s in 1:n
            ηc = βc .+ Λc * randn(K)
            for t in 1:p
                rand() < inv(1 + exp(-βz[t])) && (Y[t, s] = exp(ηc[t] + σ_true * randn()))
            end
        end
        # disp_group=:shared pinned: this test checks the singular "sigma" Wald
        # term name, which only exists under :shared (:species names them
        # "sigma[t]").
        fit = fit_delta_lognormal_gllvm(Y; K = K, disp_group = :shared)
        ci = confint(fit, Y; method = :wald, parm = "sigma")
        @test ci.term == ["sigma"]
        @test ci.estimate[1] ≈ fit.σ atol = 1e-8
        if isfinite(ci.lower[1])
            @test 0 < ci.lower[1] < ci.estimate[1] < ci.upper[1]
        end
    end

    @testset "Two-part: ZIP bootstrap single- vs multi-core identical" begin
        Random.seed!(33)
        p, K, n = 4, 1, 140
        βz = 0.3 .* randn(p) .- 0.6; βc = 0.3 .* randn(p) .+ 1.2
        Λc = 0.4 .* randn(p, K)
        Y = zeros(Int, p, n)
        for s in 1:n
            ηc = βc .+ Λc * randn(K)
            for t in 1:p
                Y[t, s] = rand() < inv(1 + exp(-βz[t])) ? 0 : rand(Poisson(exp(ηc[t])))
            end
        end
        fit = fit_zip_gllvm(Y; K = K)
        a = confint(fit, Y; method = :bootstrap, n_boot = 20, seed = 5, parallel = false)
        b = confint(fit, Y; method = :bootstrap, n_boot = 20, seed = 5, parallel = true)
        @test a.lower == b.lower && a.upper == b.upper
        @test a.n_converged ≥ 6
    end

    @testset "Two-part: ZIB Wald + bootstrap (zero-inflated binomial)" begin
        Random.seed!(36)
        # ZIB bootstrap refits use finite-difference gradients through the
        # two-part Laplace likelihood. Keep this as a smoke, not a runtime gate.
        p, K, n, Ntr = 4, 1, 80, 8
        βz = 0.3 .* randn(p) .- 0.6; βc = 0.3 .* randn(p)
        Λc = 0.4 .* randn(p, K)
        Y = zeros(Int, p, n)
        for s in 1:n
            ηc = βc .+ Λc * randn(K)
            for t in 1:p
                μ = inv(1 + exp(-ηc[t]))
                Y[t, s] = rand() < inv(1 + exp(-βz[t])) ? 0 : rand(Binomial(Ntr, μ))
            end
        end
        fit = fit_zib_gllvm(Y; K = K, N = Ntr, iterations = 120)

        ci = confint(fit, Y; method = :wald)
        @test length(ci.term) == 2p + GLLVModels.rr_theta_len(p, K)   # βz + βc + Λc (no dispersion)
        @test ci.method == :wald
        for i in eachindex(ci.term)
            if isfinite(ci.lower[i]) && isfinite(ci.upper[i])
                @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
            end
        end

        # parametric bootstrap is deterministic in the seed (serial == parallel).
        a = confint(fit, Y; method = :bootstrap, n_boot = 10, seed = 5, parallel = false)
        b = confint(fit, Y; method = :bootstrap, n_boot = 10, seed = 5, parallel = true)
        @test a.lower == b.lower && a.upper == b.upper
        @test a.n_converged ≥ 6
    end

    @testset "Ordinal Wald + bootstrap (τ in natural scale)" begin
        Random.seed!(34)
        p, K, n, C = 4, 1, 220, 4
        Λ = 0.7 .* randn(p, K)
        τ = [-1.0, 0.0, 1.2]
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n
            η = Λ * randn(K)
            for t in 1:p
                u = rand(); cum = 0.0; cat = C
                for c in 1:C
                    Fhi = c == C ? 1.0 : inv(1 + exp(-(τ[c] - η[t])))
                    Flo = c == 1 ? 0.0 : inv(1 + exp(-(τ[c - 1] - η[t])))
                    cum += Fhi - Flo
                    if u <= cum
                        cat = c; break
                    end
                end
                Y[t, s] = cat
            end
        end
        fit = fit_ordinal_gllvm(Y; K = K)
        ci = confint(fit, Y; method = :wald)
        @test length(ci.term) == (p * K) + (C - 1)        # Λ + τ
        @test "tau[1]" in ci.term && "Lambda[1,1]" in ci.term
        # cutpoints are ordered; their Wald point estimates inherit that
        taus = [ci.estimate[findfirst(==("tau[$c]"), ci.term)] for c in 1:(C - 1)]
        @test issorted(taus)

        bo = confint(fit, Y; method = :bootstrap, n_boot = 20, seed = 3, parm = "Lambda")
        @test bo.method === :bootstrap
        @test bo.n_converged ≥ 5
    end

    @testset "Tweedie Wald + bootstrap (phi on natural scale, power fixed)" begin
        # Compound Poisson–Gamma draws (true zeros + positive part), mirroring the
        # data-generating loop in test/test_tweedie.jl.
        # Seed repair (2026-08-03): seed 35 converged with the fitted power p
        # pinned at its p_init=1.5 default, leaving a knife-edge-flat profile in
        # (φ, p) whose Hessian-based phi SE is ~1e-8 — finite on macOS/windows/
        # Julia 1 but flaked to NaN on Julia 1.10-ubuntu (LAPACK/BLAS LSB
        # differences flip the sign of a near-zero curvature term). Seed 3 lands
        # a draw where the fit moves well away from p_init (p≈1.28), giving a
        # well-conditioned phi SE (~0.09, four orders of magnitude larger) that
        # is robust across platforms — a setup repair, not a tolerance change.
        Random.seed!(3)
        p, K, n = 4, 1, 50
        β = log.(rand(p) .* 2 .+ 0.5)
        Λ = 0.4 .* randn(p, K)
        Y = zeros(p, n)
        for s in 1:n
            z = randn(K)
            for t in 1:p
                μ = exp(β[t] + dot(Λ[t, :], z))
                k = rand(Poisson(μ))
                Y[t, s] = k == 0 ? 0.0 : sum(rand(Gamma(2.0, μ / (2.0 * μ + 1e-9)), k))
            end
        end
        fit = fit_tweedie_gllvm(Y; K = K, iterations = 40)

        ci = confint(fit, Y; method = :wald)
        # β + Λ + a single dispersion term (φ); the power p is held fixed.
        @test length(ci.term) == p + GLLVModels.rr_theta_len(p, K) + 1
        @test ci.method === :wald
        @test "phi" in ci.term
        let pidx = findfirst(==("phi"), ci.term)
            @test ci.estimate[pidx] ≈ fit.φ atol = 1e-6
            @test isfinite(ci.se[pidx])           # φ gets a real Wald SE (Hessian non-singular)
        end
        # finite-SE intervals bracket their estimate (standard non-strict pattern).
        for i in eachindex(ci.term)
            if isfinite(ci.lower[i]) && isfinite(ci.upper[i])
                @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
            end
        end

        # Parametric bootstrap produces a sensible CI. (Serial-vs-parallel BITWISE
        # determinism is a framework property already asserted by the ZIP/ZIB
        # bootstrap tests; we don't re-assert exact equality here because Tweedie's
        # heavy marginal — Simpson quadrature over the compound density, threaded
        # BLAS — yields non-bitwise-identical reductions across execution contexts.)
        bo = confint(fit, Y; method = :bootstrap, n_boot = 8, seed = 5)
        @test bo.method === :bootstrap
        @test bo.n_converged ≥ 4
        for i in eachindex(bo.term)
            if isfinite(bo.lower[i]) && isfinite(bo.upper[i])
                @test bo.lower[i] ≤ bo.upper[i]
            end
        end
    end

    @testset "VA-based standard errors" begin
        Y, _, _ = _sim_poisson(4, 1, 140; seed = 41)
        fit = fit_poisson_gllvm_va(Y; K = 1)

        # Wald SEs from the variational (ELBO) marginal.
        ci = confint(fit, Y; method = :wald, objective = :va)
        @test ci.method === :wald
        @test length(ci.term) == length(fit.β) + (4 * 1)     # β + Λ entries
        @test ci.estimate[1] ≈ fit.β[1] atol = 1e-8
        fin = isfinite.(ci.se)
        @test any(fin)
        @test all(ci.lower[fin] .< ci.estimate[fin] .< ci.upper[fin])

        # coef_table forwards objective=:va through to the Wald confint.
        ct = coef_table(fit, Y; objective = :va)
        @test ct isa GllvmCoefTable
        ctfin = isfinite.(ct.std_error)
        @test any(ctfin)
        @test all(ct.lower[ctfin] .< ct.estimate[ctfin] .< ct.upper[ctfin])
        @test all(ct.z[ctfin] .≈ ct.estimate[ctfin] ./ ct.std_error[ctfin])

        # NB VA fit: positive natural-scale dispersion r interval.
        Random.seed!(42)
        p, K, n, r_true = 4, 1, 200, 6.0
        βr = 0.4 .* randn(p) .+ 1.2
        Λr = 0.4 .* randn(p, K)
        Yr = Matrix{Int}(undef, p, n)
        for s in 1:n
            η = βr .+ Λr * randn(K)
            for t in 1:p
                μ = exp(η[t])
                Yr[t, s] = rand(NegativeBinomial(r_true, r_true / (r_true + μ)))
            end
        end
        fnb = fit_nb_gllvm_va(Yr; K = K)
        cir = confint(fnb, Yr; method = :wald, objective = :va, parm = "r")
        @test cir.term == ["r"]
        @test cir.estimate[1] ≈ fnb.r atol = 1e-8
        if isfinite(cir.lower[1])
            @test 0 < cir.lower[1] < cir.estimate[1] < cir.upper[1]
        end

        # objective=:va restricted to method=:wald.
        @test_throws ArgumentError confint(fit, Y; method = :profile, objective = :va)

        # objective=:va unsupported for families without a VA marginal.
        Yo, _, _ = _sim_poisson(3, 1, 80; seed = 43)
        Yo = (Yo .% 4) .+ 1                                   # fold counts into 4 ordered categories
        ford = fit_ordinal_gllvm(Yo; K = 1)
        @test_throws ArgumentError confint(ford, Yo; method = :wald, objective = :va)
    end

    @testset "Beta-binomial Wald + bootstrap (phi on natural scale, N as data)" begin
        Random.seed!(45)
        p, K, n, Ntr, φ_true = 4, 1, 60, 8, 6.0
        β = 0.3 .* randn(p) .- 0.2
        Λ = 0.4 .* randn(p, K)
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n
            η = β .+ Λ * randn(K)
            for t in 1:p
                μ = clamp(inv(1 + exp(-η[t])), 1e-12, 1 - 1e-12)
                pdraw = clamp(rand(Beta(μ * φ_true, (1 - μ) * φ_true)), 1e-12, 1 - 1e-12)
                Y[t, s] = rand(Binomial(Ntr, pdraw))
            end
        end
        N = fill(Ntr, p, n)
        fit = fit_beta_binomial_gllvm(Y; K = K, N = N, iterations = 40)

        # N is taken as data through the `N` kwarg (it is not stored in the fit).
        ci = confint(fit, Y; method = :wald, N = N)
        # β + Λ + a single dispersion term (φ).
        @test length(ci.term) == p + GLLVModels.rr_theta_len(p, K) + 1
        @test ci.method === :wald
        @test "phi" in ci.term
        let pidx = findfirst(==("phi"), ci.term)
            @test ci.estimate[pidx] ≈ fit.φ atol = 1e-6
        end
        for i in eachindex(ci.term)
            if isfinite(ci.lower[i]) && isfinite(ci.upper[i])
                @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
            end
        end

        # Parametric bootstrap produces a sensible CI (no serial==parallel assertion).
        bo = confint(fit, Y; method = :bootstrap, n_boot = 8, seed = 5, N = N)
        @test bo.method === :bootstrap
        @test bo.n_converged ≥ 4
        for i in eachindex(bo.term)
            if isfinite(bo.lower[i]) && isfinite(bo.upper[i])
                @test bo.lower[i] ≤ bo.upper[i]
            end
        end
    end

    @testset "Beta-binomial grouped Wald (phi[g] on natural scale, N as data)" begin
        Random.seed!(47)
        p, K, n, Ntr, φ_true = 3, 1, 50, 8, 7.0
        β = 0.25 .* randn(p) .- 0.15
        Λ = 0.35 .* randn(p, K)
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n
            η = β .+ Λ * randn(K)
            for t in 1:p
                μ = clamp(inv(1 + exp(-η[t])), 1e-12, 1 - 1e-12)
                pdraw = clamp(rand(Beta(μ * φ_true, (1 - μ) * φ_true)), 1e-12, 1 - 1e-12)
                Y[t, s] = rand(Binomial(Ntr, pdraw))
            end
        end
        N = fill(Ntr, p, n)
        fit = fit_beta_binomial_gllvm_grouped(Y; K = K, N = N, group = collect(1:p),
                                              iterations = 80)
        @test fit isa BetaBinomialGroupedFit
        ci = confint(fit, Y; method = :wald, N = N)
        @test ci.method === :wald
        @test length(ci.term) == p + GLLVModels.rr_theta_len(p, K) + p
        @test all(("phi[$g]" in ci.term) for g in 1:p)
        for g in 1:p
            pidx = findfirst(==("phi[$g]"), ci.term)
            @test pidx !== nothing
            @test ci.estimate[pidx] ≈ fit.φ[g] atol = 1e-6
        end
        for i in eachindex(ci.term)
            if isfinite(ci.lower[i]) && isfinite(ci.upper[i])
                @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
            end
        end
        @test_throws ArgumentError confint(fit, Y; method = :nope, N = N)
    end

    @testset "Beta-binomial grouped_cov Wald (gamma + phi[g], N and X as data)" begin
        Random.seed!(48)
        p, K, n, q, Ntr, φ_true = 3, 1, 55, 1, 8, 8.0
        β = 0.2 .* randn(p) .+ 0.1
        γ = [0.35]
        Λ = 0.3 .* randn(p, K)
        X = randn(p, n, q)
        O = GLLVModels._build_offset(X, γ)
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n
            η = β .+ view(O, :, s) .+ Λ * randn(K)
            for t in 1:p
                μ = clamp(inv(1 + exp(-η[t])), 1e-12, 1 - 1e-12)
                pdraw = clamp(rand(Beta(μ * φ_true, (1 - μ) * φ_true)), 1e-12, 1 - 1e-12)
                Y[t, s] = rand(Binomial(Ntr, pdraw))
            end
        end
        N = fill(Ntr, p, n)
        fit = fit_beta_binomial_gllvm_grouped_cov(Y; X = X, K = K, N = N,
                                                  group = collect(1:p), iterations = 80)
        @test fit isa BetaBinomialGroupedCovFit
        @test_throws ArgumentError confint(fit, Y; method = :wald, N = N)
        ci = confint(fit, Y; method = :wald, N = N, X = X)
        @test ci.method === :wald
        @test "gamma[1]" in ci.term
        @test all(("phi[$g]" in ci.term) for g in 1:p)
        gidx = findfirst(==("gamma[1]"), ci.term)
        @test ci.estimate[gidx] ≈ fit.γ[1] atol = 1e-6
        for g in 1:p
            pidx = findfirst(==("phi[$g]"), ci.term)
            @test ci.estimate[pidx] ≈ fit.φ[g] atol = 1e-6
        end
        for i in eachindex(ci.term)
            if isfinite(ci.lower[i]) && isfinite(ci.upper[i])
                @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
            end
        end
    end

    @testset "ZIPCovFit Wald (dual γz/γc; missing X throws)" begin
        Random.seed!(49)
        p, K, n, q = 3, 1, 55, 1
        βz = 0.3 .* randn(p) .- 0.7
        γz = [0.3]; βc = 0.2 .* randn(p) .+ 0.5; γc = [0.4]
        Λc = 0.3 .* randn(p, K)
        X = randn(p, n, q)
        Oz = GLLVModels._build_offset(X, γz); Oc = GLLVModels._build_offset(X, γc)
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n
            z = randn(K)
            for t in 1:p
                π = inv(1 + exp(-(βz[t] + Oz[t, s])))
                μ = exp(clamp(βc[t] + Oc[t, s] + (Λc * z)[t], -4, 4))
                Y[t, s] = rand() < π ? 0 : rand(Poisson(μ))
            end
        end
        fit = fit_zip_gllvm_cov(Y; X = X, K = K, iterations = 120)
        @test fit isa ZIPCovFit
        @test_throws ArgumentError confint(fit, Y; method = :wald)
        ci = confint(fit, Y; method = :wald, X = X)
        @test ci.method === :wald
        @test "gammaz[1]" in ci.term
        @test "gammac[1]" in ci.term
        gz = findfirst(==("gammaz[1]"), ci.term)
        gc = findfirst(==("gammac[1]"), ci.term)
        @test ci.estimate[gz] ≈ fit.γz[1] atol = 1e-6
        @test ci.estimate[gc] ≈ fit.γc[1] atol = 1e-6
        for i in eachindex(ci.term)
            if isfinite(ci.lower[i]) && isfinite(ci.upper[i])
                @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
            end
        end
        # Optional Rung1: zero-X ZIPCovFit CI ≈ ZIPFit on shared intercept/loading terms.
        X0 = zeros(p, n, q)
        f0 = fit_zip_gllvm(Y; K = K, iterations = 120)
        fx0 = fit_zip_gllvm_cov(Y; X = X0, K = K, iterations = 120)
        ci0 = confint(f0, Y; method = :wald)
        cix0 = confint(fx0, Y; method = :wald, X = X0)
        shared = filter(t -> startswith(t, "betaz") || startswith(t, "betac") ||
                             startswith(t, "Lambda"), ci0.term)
        for t in shared
            i0 = findfirst(==(t), ci0.term)
            ix = findfirst(==(t), cix0.term)
            @test ix !== nothing
            @test ci0.estimate[i0] ≈ cix0.estimate[ix] atol = 1e-4
        end
    end

    @testset "ZINBCovFit Wald (dual γz/γc; shared r; missing X throws)" begin
        Random.seed!(50)
        p, K, n, q = 3, 1, 55, 1
        r = 7.0
        βz = 0.3 .* randn(p) .- 0.7
        γz = [0.3]; βc = 0.2 .* randn(p) .+ 0.5; γc = [0.4]
        Λc = 0.3 .* randn(p, K)
        X = randn(p, n, q)
        Oz = GLLVModels._build_offset(X, γz); Oc = GLLVModels._build_offset(X, γc)
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n
            z = randn(K)
            for t in 1:p
                π = inv(1 + exp(-(βz[t] + Oz[t, s])))
                μ = exp(clamp(βc[t] + Oc[t, s] + (Λc * z)[t], -4, 4))
                Y[t, s] = rand() < π ? 0 : rand(NegativeBinomial(r, r / (r + μ)))
            end
        end
        fit = fit_zinb_gllvm_cov(Y; X = X, K = K, iterations = 120)
        @test fit isa ZINBCovFit
        @test_throws ArgumentError confint(fit, Y; method = :wald)
        ci = confint(fit, Y; method = :wald, X = X)
        @test ci.method === :wald
        @test "gammaz[1]" in ci.term
        @test "gammac[1]" in ci.term
        @test "r" in ci.term
        gz = findfirst(==("gammaz[1]"), ci.term)
        gc = findfirst(==("gammac[1]"), ci.term)
        ri = findfirst(==("r"), ci.term)
        @test ci.estimate[gz] ≈ fit.γz[1] atol = 1e-6
        @test ci.estimate[gc] ≈ fit.γc[1] atol = 1e-6
        @test ci.estimate[ri] ≈ fit.r atol = 1e-6
        for i in eachindex(ci.term)
            if isfinite(ci.lower[i]) && isfinite(ci.upper[i])
                @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
            end
        end
        # Optional Rung1: zero-X ZINBCovFit CI ≈ ZINBFit on shared intercept/loading/r.
        X0 = zeros(p, n, q)
        f0 = fit_zinb_gllvm(Y; K = K, iterations = 120)
        fx0 = fit_zinb_gllvm_cov(Y; X = X0, K = K, iterations = 120)
        ci0 = confint(f0, Y; method = :wald)
        cix0 = confint(fx0, Y; method = :wald, X = X0)
        shared = filter(t -> startswith(t, "betaz") || startswith(t, "betac") ||
                             startswith(t, "Lambda") || t == "r", ci0.term)
        for t in shared
            i0 = findfirst(==(t), ci0.term)
            ix = findfirst(==(t), cix0.term)
            @test ix !== nothing
            @test ci0.estimate[i0] ≈ cix0.estimate[ix] atol = 1e-4
        end
    end

    @testset "Random row effect Wald + bootstrap (sigma_row + family dispersion)" begin
        Random.seed!(46)
        p, K, n, σ_true = 4, 1, 60, 0.6
        β = 0.4 .* randn(p) .+ 1.0
        Λ = 0.4 .* randn(p, K)
        Y = Matrix{Int}(undef, p, n)            # NB family: carries a dispersion r
        for s in 1:n
            ρ = σ_true * randn()
            η = β .+ ρ .+ Λ * randn(K)
            for t in 1:p
                μ = exp(η[t]); r = 5.0
                Y[t, s] = rand(NegativeBinomial(r, r / (r + μ)))
            end
        end
        fit = fit_row_random_gllvm(Y; family = NegativeBinomial(), K = K, iterations = 40)

        # Working vector [β; pack_lambda(Λ); log σ_row; log r] ⇒ β + Λ + sigma_row + r.
        ci = confint(fit, Y; method = :wald)
        @test length(ci.term) == p + GLLVModels.rr_theta_len(p, K) + 2
        @test ci.method === :wald
        @test "sigma_row" in ci.term && "r" in ci.term
        let sidx = findfirst(==("sigma_row"), ci.term)
            @test ci.estimate[sidx] ≈ fit.σ_row atol = 1e-6
        end
        for i in eachindex(ci.term)
            if isfinite(ci.lower[i]) && isfinite(ci.upper[i])
                @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
            end
        end

        # Parametric bootstrap produces a sensible CI (no serial==parallel assertion).
        bo = confint(fit, Y; method = :bootstrap, n_boot = 8, seed = 5)
        @test bo.method === :bootstrap
        @test bo.n_converged ≥ 4
        for i in eachindex(bo.term)
            if isfinite(bo.lower[i]) && isfinite(bo.upper[i])
                @test bo.lower[i] ≤ bo.upper[i]
            end
        end
    end

    # T14 F1 (docs/dev-log/core070/t14-nb2-wald-nan-diagnosis.md): when the joint
    # Wald Hessian is not PD, `_family_wald` now degrades PER PARAMETER instead of
    # NaN-ing every entry — conditioning out the KNOWN-boundary parameters (a
    # fit's `dispersion_boundary`) plus any further direction needed for the
    # remaining sub-Hessian to pass `isposdef`, and reporting `boundary_terms`.
    @testset "NB2 grouped-cov Wald: per-parameter boundary degradation (T14 F1)" begin
        @testset "seed-523 degenerate fixture: partial-NaN, not all-NaN" begin
            # The exact `_bx_sim(NegativeBinomial(), 3, 70, 1, 1; seed=523)`
            # fixture from test/test_bridge_x.jl.
            rng = Random.MersenneTwister(523)
            p, n, K, q = 3, 70, 1, 1
            β = 0.3 .* randn(rng, p)
            γ = 0.6 .* randn(rng, q)
            Λ = 0.4 .* randn(rng, p, K)
            x1 = randn(rng, n)
            X = zeros(p, n, q)
            for t in 1:p, s in 1:n
                X[t, s, 1] = x1[s]
            end
            O = GLLVModels._build_offset(X, γ)
            Z = randn(rng, K, n)
            η = β .+ O .+ Λ * Z
            Y = Matrix{Float64}(undef, p, n)
            for t in 1:p, s in 1:n
                η_ts = clamp(η[t, s], -6, 4)
                r = 8.0; μ = exp(η_ts)
                Y[t, s] = float(rand(rng, NegativeBinomial(r, r / (r + μ))))
            end
            Yi = round.(Int, Y)
            fit = fit_nb_gllvm_grouped_cov(Yi; X = X, K = K, group = collect(1:p))
            @test any(fit.dispersion_boundary)
            boundary_groups = findall(fit.dispersion_boundary)
            nonboundary_groups = findall(!, fit.dispersion_boundary)
            @test !isempty(nonboundary_groups)   # the fixture has a mix, not all-degenerate

            ci = confint(fit, Yi; method = :wald, X = X)
            @test ci.pd_hessian == false
            @test Set(ci.boundary_terms) == Set("r[$g]" for g in boundary_groups)

            # beta / gamma / Lambda / the non-boundary trait's r get FINITE bounds.
            finite_expected = vcat(["beta[$t]" for t in 1:p], ["gamma[1]"],
                                    GLLVModels._confint_lambda_term_names("Lambda", p, K),
                                    ["r[$g]" for g in nonboundary_groups])
            for name in finite_expected
                i = findfirst(==(name), ci.term)
                @test i !== nothing
                @test isfinite(ci.se[i])
                @test isfinite(ci.lower[i])
                @test isfinite(ci.upper[i])
                @test ci.lower[i] ≤ ci.estimate[i] ≤ ci.upper[i]
            end
            # the boundary trait(s)' r are NaN (Fisher information ~0 there).
            for g in boundary_groups
                i = findfirst(==("r[$g]"), ci.term)
                @test isnan(ci.se[i])
                @test isnan(ci.lower[i])
                @test isnan(ci.upper[i])
            end
        end

        @testset "forced boundary (deterministic): flagged r is conditioned out even when the joint Hessian is barely PD" begin
            # Independent of where the optimizer stops: start from the SHARED-
            # dispersion fixture (group = ones(p); one r, well-conditioned by
            # construction — per-trait NB dispersion is only weakly identified
            # against a free latent factor, which is why the seed-523 case is a
            # knife edge), then REPLACE that r by a Poisson-limit value. The
            # positional constructor derives the `dispersion_boundary` flag, and
            # confint must condition the flagged term out regardless of the
            # Cholesky outcome (T14 F1, 2026-09-03).
            rng = Random.MersenneTwister(9001)
            p, n, K = 3, 120, 1
            μ = exp.(1.2 .+ 0.3 .* randn(rng, p))
            Yi = [rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + μ[t]))) for t in 1:p, s in 1:n]
            X = reshape(randn(rng, p * n), p, n, 1)
            f0 = fit_nb_gllvm_grouped_cov(Yi; X = X, K = K, group = ones(Int, p))
            @test length(f0.r_group) == 1
            r2 = [1.0e12]
            f1 = GLLVModels.NBGroupedCovFit(f0.β, f0.γ, f0.γ_fixed, f0.Λ, r2, f0.group, f0.link,
                                       f0.loglik, f0.converged, f0.iterations)
            @test f1.dispersion_boundary == [true]
            ci = confint(f1, Yi; method = :wald, X = X)
            @test ci.pd_hessian == false
            @test "r[1]" in ci.boundary_terms
            i1 = findfirst(==("r[1]"), ci.term)
            @test isnan(ci.se[i1]) && isnan(ci.lower[i1]) && isnan(ci.upper[i1])
            # The contract, not a fixed finite set: the point is a forced
            # non-optimum, so the reduced Hessian may need one more direction
            # conditioned out (platform-dependent). Every term is therefore
            # either finite or NAMED in `boundary_terms` — never a huge
            # "finite" SE or an `Inf` bound (the F3-class defect).
            for (i, name) in enumerate(ci.term)
                conditioned = name in ci.boundary_terms
                @test conditioned || (isfinite(ci.se[i]) && isfinite(ci.lower[i]) && isfinite(ci.upper[i]))
                @test !conditioned || (isnan(ci.se[i]) && isnan(ci.lower[i]) && isnan(ci.upper[i]))
                @test !isinf(ci.lower[i]) && !isinf(ci.upper[i])
            end
            @test count(isfinite, ci.se) ≥ 1   # the degradation kept something, not all-NaN
        end

        @testset "PD fixture: bounds unaffected by the degradation branch" begin
            # group = ones(p): a single SHARED dispersion, matching the DGP —
            # well-conditioned by construction (mirrors the existing
            # "one group ≈ fit_nb_gllvm" precedent in test_grouped_dispersion.jl).
            Random.seed!(4901)
            p, K, n, q, r_true = 5, 1, 200, 1, 6.0
            β = 0.3 .* randn(p) .+ 1.0
            γ = [0.4]
            Λ = 0.35 .* randn(p, K)
            X = randn(p, n, q)
            O = GLLVModels._build_offset(X, γ)
            Y = Matrix{Int}(undef, p, n)
            for s in 1:n
                η = β .+ view(O, :, s) .+ Λ * randn(K)
                for t in 1:p
                    μ = exp(clamp(η[t], -6, 6))
                    Y[t, s] = rand(NegativeBinomial(r_true, r_true / (r_true + μ)))
                end
            end
            fit = fit_nb_gllvm_grouped_cov(Y; X = X, K = K, group = ones(Int, p))
            @test fit isa NBGroupedCovFit
            @test !any(fit.dispersion_boundary)
            ci = confint(fit, Y; method = :wald, X = X)
            @test ci.pd_hessian == true
            @test isempty(ci.boundary_terms)
            @test all(isfinite, ci.se)
            @test all(ci.lower .<= ci.estimate .<= ci.upper)
            # T14 F1 receipt: on a PD fixture the new `_family_wald` never enters
            # the `!pd` degradation branch, so its output is bit-identical to the
            # pre-fix code by construction — independently confirmed by running
            # this exact fixture through both the pre-fix and post-fix
            # `_family_wald` (via a temporary `git stash` of src/confint_family.jl
            # + src/families/grouped_dispersion.jl) on BOTH Julia 1.12.6 and
            # 1.10.12: identical `ci.lower`/`ci.upper` before vs after on each
            # version (not asserted here as hardcoded literals, since Julia's
            # `randn` stream for this seed legitimately differs between 1.10.12
            # and 1.12.6 — a real cross-version RNG difference, not a bug — so a
            # literal capture from one Julia version fails the other; verified in
            # this task's after-task report instead of baked into the test).
        end
    end
end
