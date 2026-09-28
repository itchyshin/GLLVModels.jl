using GLLVModels, Test, Random, LinearAlgebra, ForwardDiff, Statistics

# Two-level Gaussian simulator: y_:,obs = Λ_B z_B,i + s_B,i + Λ_W z_W,obs + s_W,obs
# (μ = 0). The between draw (z_B,i, s_B,i) is shared across an individual'\''s obs;
# the within draw (z_W,obs, s_W,obs) is fresh per observation.
function _sim_twolevel(rng, Λ_B, σ²_B, Λ_W, σ²_W, nindiv, nobs)
    p, K_B = size(Λ_B); K_W = size(Λ_W, 2)
    indiv = repeat(1:nindiv, inner = nobs)
    Y = zeros(p, nindiv * nobs)
    sdB = sqrt.(σ²_B); sdW = sqrt.(σ²_W)
    col = 0
    for i in 1:nindiv
        bcontrib = Λ_B * randn(rng, K_B) .+ sdB .* randn(rng, p)
        for _ in 1:nobs
            col += 1
            Y[:, col] = bcontrib .+ Λ_W * randn(rng, K_W) .+ sdW .* randn(rng, p)
        end
    end
    return Y, indiv
end

@testset "Gaussian two-level (between/within-individual) reduced-rank" begin
    # ------------------------------------------------------------------
    # GATE 1: the marginal matches a direct dense Σ_i = I⊗Σ_W + J⊗Σ_B build,
    # summed over individuals (the rotation trick must equal the brute-force
    # block-MVN log-density).
    # ------------------------------------------------------------------
    @testset "rotation-trick marginal == dense block-MVN (rtol 1e-9)" begin
        Random.seed!(70001)
        p, K_B, K_W = 4, 2, 1
        Λ_B = 0.7 .* randn(p, K_B); Λ_W = 0.5 .* randn(p, K_W)
        σ²_B = 0.3 .+ 0.4 .* rand(p); σ²_W = 0.3 .+ 0.4 .* rand(p)
        indiv = repeat(1:6, inner = 3)               # 6 individuals, 3 obs each
        y = randn(p, length(indiv))
        ll = twolevel_marginal_loglik(y, indiv, Λ_B, σ²_B, Λ_W, σ²_W)

        Σ_B = Λ_B * Λ_B' + Diagonal(σ²_B)
        Σ_W = Λ_W * Λ_W' + Diagonal(σ²_W)
        codes, _ = GLLVModels._code_grouping(indiv); L = maximum(codes)
        ll_dense = 0.0
        for g in 1:L
            idx = findall(==(g), codes); ni = length(idx)
            yi = vec(y[:, idx])                       # stacked p·ni (obs-major)
            Σi = kron(Matrix(I, ni, ni), Σ_W) + kron(ones(ni, ni), Σ_B)
            cΣ = cholesky(Symmetric(Σi))
            ll_dense += -0.5 * (ni * p * log(2π) + logdet(cΣ) + dot(yi, cΣ \ yi))
        end
        @test isapprox(ll, ll_dense; rtol = 1e-9)
    end

    # ------------------------------------------------------------------
    # GATE 1b: at extreme variance ratios the Woodbury Σ_W⁻¹ quadratic form
    # loses all precision and can come out NEGATIVE (impossible for a PD
    # Σ_W), which turned the log-likelihood into a huge POSITIVE number.
    # The optimiser then "converged" there (parametric-bootstrap replicates
    # of the CORE070 CI-ROUTE-011 fixture reported loglik ~ +1e22 with
    # repeatability ~ 0, collapsing the ICC bootstrap lower bound). The
    # marginal must never overstate the exact (BigFloat, dense) value.
    # Parameters are one such replicate fit, verbatim.
    # ------------------------------------------------------------------
    @testset "marginal never overstates exact value at extreme variance ratios" begin
        Λ_B = reshape([-0.9133173413239226, -0.0009085060802649791,
                       0.08140689784664193, 7.110538275801313], 4, 1)
        σ²_B = [2.7880521598580705e-15, 6.153275246278945e-22,
                1.0177929500613746e-19, 4.056049533090739e-21]
        Λ_W = reshape([-0.6832983940881012, -2.493929718953794,
                       4.425680962343325, 3.546306331228814], 4, 1)
        σ²_W = [2.2310698803382298e15, 1.736860948771967e-37,
                1.5458132733345844e14, 2.89155236951954e11]
        y = randn(MersenneTwister(1), 4, 12)
        indiv = repeat(1:4, inner = 3)
        ll = twolevel_marginal_loglik(y, indiv, Λ_B, σ²_B, Λ_W, σ²_W)

        # Exact reference: the same rotation-trick decomposition in BigFloat
        # with dense solves (no Woodbury cancellation).
        SB = big.(Λ_B) * big.(Λ_B)' + Diagonal(big.(σ²_B))
        SW = big.(Λ_W) * big.(Λ_W)' + Diagonal(big.(σ²_W))
        ll_exact = big(0)
        for g in 1:4
            Yi = big.(y[:, indiv .== g]); ni = size(Yi, 2)
            mi = vec(sum(Yi, dims = 2)) ./ ni; Yic = Yi .- mi
            M = SW + ni * SB
            ll_exact += -(ni * 4 * log(2 * big(pi)) + logdet(M) + (ni - 1) * logdet(SW) +
                          ni * dot(mi, M \ mi) + sum(Yic .* (SW \ Yic))) / 2
        end
        @test ll <= Float64(ll_exact) + 1e-8 * abs(Float64(ll_exact))
    end

    # ------------------------------------------------------------------
    # GATE 2: AD gradient of the packed NLL matches a central finite
    # difference to ≤ 1e-6.
    # ------------------------------------------------------------------
    @testset "FD-gradient of the packed NLL ≤ 1e-6" begin
        Random.seed!(70002)
        p, K_B, K_W = 5, 2, 1
        indiv = repeat(1:8, inner = 4)
        y = randn(p, length(indiv))
        codes, _ = GLLVModels._code_grouping(indiv); L = maximum(codes)
        ind_idx = [findall(==(g), codes) for g in 1:L]
        rrB = GLLVModels.rr_theta_len(p, K_B); rrW = GLLVModels.rr_theta_len(p, K_W)
        nll = θ -> begin
            Λ_B, σ²_B, Λ_W, σ²_W = GLLVModels._twolevel_unpack(θ, p, K_B, K_W)
            -GLLVModels._twolevel_loglik(y, ind_idx, Λ_B, σ²_B, Λ_W, σ²_W)
        end
        θ = vcat(0.3 .* randn(rrB), log.(fill(0.4, p)),
                 0.3 .* randn(rrW), log.(fill(0.5, p)))
        g_ad = ForwardDiff.gradient(nll, θ)
        h = 1e-6; g_fd = similar(θ)
        for i in eachindex(θ)
            θp = copy(θ); θp[i] += h
            θm = copy(θ); θm[i] -= h
            g_fd[i] = (nll(θp) - nll(θm)) / (2h)
        end
        @test maximum(abs.(g_ad .- g_fd)) ≤ 1e-6
    end

    # ------------------------------------------------------------------
    # GATE 3: recovery of a known Σ_B / Σ_W, per-trait repeatability, and
    # per-level communality. Averaged over a few replicates (the MC mean is
    # ~√nrep tighter than any single fit) so the assertion reflects the
    # MCSE-level recovery — single-seed scatter at 300 individuals × 5 obs is
    # genuine sampling error, not bias. The full 30-replicate MCSE study (all
    # targets |z| ≤ 2.5, unbiased) is run outside the package suite.
    # ------------------------------------------------------------------
    @testset "recovery of Σ_B / Σ_W + repeatability + communality" begin
        p, K_B, K_W = 4, 1, 1
        Λ_B = reshape([1.2, 0.9, -0.7, 0.5], p, 1)
        Λ_W = reshape([0.6, 0.4, 0.8, 0.3], p, 1)
        σ²_B = [0.30, 0.40, 0.25, 0.35]
        σ²_W = [0.50, 0.45, 0.55, 0.40]
        Σ_B_true = Λ_B * Λ_B' + Diagonal(σ²_B)
        Σ_W_true = Λ_W * Λ_W' + Diagonal(σ²_W)
        R_true   = [Σ_B_true[t, t] / (Σ_B_true[t, t] + Σ_W_true[t, t]) for t in 1:p]
        cB_true  = [(Λ_B * Λ_B')[t, t] / Σ_B_true[t, t] for t in 1:p]
        cW_true  = [(Λ_W * Λ_W')[t, t] / Σ_W_true[t, t] for t in 1:p]

        nrep = 30; nindiv = 500; nobs = 5
        rng = MersenneTwister(70003)
        ΣB̄ = zeros(p, p); ΣW̄ = zeros(p, p)
        R̄ = zeros(p); cB̄ = zeros(p); cW̄ = zeros(p)
        local last_fit
        for _ in 1:nrep
            Y, indiv = _sim_twolevel(rng, Λ_B, σ²_B, Λ_W, σ²_W, nindiv, nobs)
            fit = fit_twolevel_gaussian(Y, indiv; K_B = K_B, K_W = K_W)
            @test fit.converged
            @test fit.nindiv == nindiv
            ΣB̄ .+= fit.Σ_B ./ nrep; ΣW̄ .+= fit.Σ_W ./ nrep
            R̄ .+= repeatability(fit) ./ nrep
            cB̄ .+= communality_B(fit) ./ nrep
            cW̄ .+= communality_W(fit) ./ nrep
            last_fit = fit
        end

        # Σ diagonals (per-trait total between / within variances). The between
        # diagonal has only nindiv realisations behind it, so it carries the
        # widest MCSE; the within diagonal (nindiv·nobs obs) is far tighter.
        # These atols are smoke-test bounds, NOT the MCSE study (that runs at
        # higher nrep with |z| ≤ 2.5 outside the package suite). They are sized
        # to absorb cross-Julia-version / cross-arch BLAS scatter seen on the CI
        # grid: on Julia 1.12 the between off-diagonal drifts ~0.04 and R̄ ~0.011
        # vs Julia 1.10 (same fixed MersenneTwister(70003) seed, deterministic
        # PPCA-init fit — the difference is LAPACK, not sampling). nrep was raised
        # 10→30 so the MC mean is √3 tighter; the bounds then add headroom on top.
        @test isapprox(diag(ΣB̄), diag(Σ_B_true); atol = 0.10)
        @test isapprox(diag(ΣW̄), diag(Σ_W_true); atol = 0.03)
        # Off-diagonals (the syndrome / state correlation building blocks).
        @test isapprox(ΣB̄[1, 2], Σ_B_true[1, 2]; atol = 0.06)
        @test isapprox(ΣW̄[1, 2], Σ_W_true[1, 2]; atol = 0.03)
        # Derived quantities.
        @test isapprox(R̄, R_true; atol = 0.03)
        @test isapprox(cB̄, cB_true; atol = 0.06)
        @test isapprox(cW̄, cW_true; atol = 0.04)

        # Correlation matrices are valid (unit diagonal, symmetric, in range).
        C_B = correlation_B(last_fit); C_W = correlation_W(last_fit)
        @test all(isapprox.(diag(C_B), 1.0))
        @test all(isapprox.(diag(C_W), 1.0))
        @test C_B ≈ C_B'
        @test C_W ≈ C_W'
        @test all(abs.(C_B) .<= 1.0 + 1e-8)
        @test all(abs.(C_W) .<= 1.0 + 1e-8)
    end
end
