# Laplace saturation health for Binomial fits (2026-08-28).
#
# WHY. Cloglog's doubly-exponential upper tail saturates at η ≈ 3.3 (logit
# needs ~35), so the Laplace saturation ridge — where the log-det weight
# collapses and the penalty is deleted — is reachable at moderate ‖Λ‖. On the
# diagnosed runaway (campaign seed-1 small cloglog) the optimizer climbs +74.8
# loglik units of pure approximation error and reports `converged = true` with
# ‖Λ̂‖ ≈ 27 against a truth of 0.9. The link derivatives are FD-verified
# correct: this is an intrinsic approximation pathology, so the remedy is a
# DIAGNOSTIC (report saturation, never touch the objective or `converged`).
#
# The campaign DGP is embedded VERBATIM (Xoshiro, not StableRNGs — the
# campaign's own convention) so the acceptance case is the diagnosed one.

using GLLVModels, Test, Random, Distributions, LinearAlgebra

@testset "Laplace saturation health (Binomial)" begin

    @testset "T1: constructed thresholds and mask exclusion" begin
        p, n = 3, 4
        Y = [1 0 1 0; 0 1 0 1; 1 1 0 0]
        N = ones(Int, p, n)
        Λ = zeros(p, 1)
        # β chosen so trait 1 sits past cloglog's upper saturation (η=5 ⇒
        # μ = 1−exp(−e^5) ≈ 1 within 1e-12), trait 2 benign, trait 3 at −40
        # (past the η-clamp).
        β = [5.0, 0.0, -40.0]
        sat = GLLVModels._laplace_saturation_health(Y, N, Λ, β, GLLVModels.CLogLogLink(), :fisher)
        @test sat.n_obs == p * n
        @test sat.n_clamp == 2 * n          # traits 1 and 3, every site
        @test sat.n_wcollapse >= n          # trait 1's weight is collapsed
        # masking trait 1 removes its cells from every count
        mask = trues(p, n); mask[1, :] .= false
        satm = GLLVModels._laplace_saturation_health(Y, N, Λ, β, GLLVModels.CLogLogLink(), :fisher; mask = mask)
        @test satm.n_obs == 2 * n
        @test satm.n_clamp == n             # only trait 3 remains
    end

    @testset "T2: the diagnosed cloglog runaway fires the diagnostic" begin
        # campaign DGP verbatim (cell_binlinks.jl, fam=cloglog, regime=small,
        # seed=1): p=5, n=60, lam=0.4; invlink(cloglog, η) = 1−exp(−e^η).
        rng = Xoshiro(1)
        p, n, lam = 5, 60, 0.4
        β = 0.3 .* randn(rng, p)
        Λ = reshape(lam .* randn(rng, p), p, 1)
        Z = randn(rng, 1, n)
        H = β .+ Λ * Z
        Y = zeros(Int, p, n)
        for t in 1:p, s in 1:n
            μ = clamp(1 - exp(-exp(H[t, s])), 1e-12, 1 - 1e-12)
            Y[t, s] = rand(rng, Bernoulli(μ)) ? 1 : 0
        end
        f = @test_logs (:warn, r"saturation region") match_mode = :any begin
            GLLVModels.fit_binomial_gllvm(Y; K = 1, link = GLLVModels.CLogLogLink())
        end
        @test !f.converged                     # optimizer may stop, but runaway loadings veto converged (#1467)
        @test norm(f.Λ) > 5.0                  # the runaway is real
        @test f.saturation !== nothing
        @test f.saturation.n_clamp > 0
        @test occursin("SATURATED", sprint(show, f))
    end

    @testset "T3: benign fits stay quiet" begin
        Random.seed!(207)
        p, n = 5, 60
        β = 0.2 .* randn(p); Λ = reshape(0.3 .* randn(p), p, 1); Z = randn(1, n)
        H = β .+ Λ * Z
        # (i) benign cloglog truth
        Yc = [rand(Bernoulli(clamp(1 - exp(-exp(H[t, s])), 1e-6, 1 - 1e-6))) ? 1 : 0
              for t in 1:p, s in 1:n]
        fc = GLLVModels.fit_binomial_gllvm(Yc; K = 1, link = GLLVModels.CLogLogLink())
        if fc.saturation !== nothing && fc.saturation.n_clamp == 0 && fc.saturation.n_wcollapse == 0
            @test !occursin("SATURATED", sprint(show, fc))
        end
        # (ii) logit and probit own-DGP fits: populated health, quiet
        Yl = [rand(Bernoulli(1 / (1 + exp(-H[t, s])))) ? 1 : 0 for t in 1:p, s in 1:n]
        fl = GLLVModels.fit_binomial_gllvm(Yl; K = 1)
        @test fl.saturation !== nothing && fl.saturation.n_obs == p * n
        @test fl.saturation.n_clamp == 0
        # (iii) extreme prevalence, trivial Λ: saturation MAY occur through the
        # intercepts alone — the hedged warning covers it; assert only that the
        # diagnostic runs and the counts are consistent (this is the D3 case:
        # the trigger reports, it does not adjudicate).
        Ye = [rand(Bernoulli(t == 1 ? 0.999 : 0.5)) ? 1 : 0 for t in 1:p, s in 1:n]
        fe = GLLVModels.fit_binomial_gllvm(Ye; K = 1, link = GLLVModels.CLogLogLink())
        @test fe.saturation === nothing || fe.saturation.n_clamp <= fe.saturation.n_obs
    end

    @testset "T4: compat construction and plateau guard" begin
        # 9-arg compat tier (the VA route's shape): saturation = "not computed"
        f9 = GLLVModels.BinomialFit([0.0], zeros(1, 1), GLLVModels.LogitLink(), -1.0, true, 3,
                               nothing, Float64[], :fisher)
        @test f9.saturation === nothing
        # 8-arg tier: default hessian AND nothing-saturation
        f8 = GLLVModels.BinomialFit([0.0], zeros(1, 1), GLLVModels.LogitLink(), -1.0, true, 3,
                               nothing, Float64[])
        @test f8.hessian === GLLVModels._default_hessian(Binomial(), GLLVModels.LogitLink())
        @test f8.saturation === nothing
    end
end
