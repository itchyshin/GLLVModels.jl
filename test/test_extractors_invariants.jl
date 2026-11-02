using GLLVModels, Test, Random, LinearAlgebra

# Algebraic invariants of the residual covariance / rotated-loadings / cutpoint /
# repeatability extractors in src/extractors.jl, on small seeded fits. These
# complement (and do not duplicate) test/test_extractors.jl, which pins the
# extractors against the generics they forward to.

function _inv_ordinal_data(seed, p, n, C)
    Random.seed!(seed)
    η = 0.7 .* randn(p) .* randn(1, n)
    τ_true = collect(range(-1.0, 1.0; length = C - 1))
    Y = Matrix{Int}(undef, p, n)
    for j in 1:n, i in 1:p
        c = 1
        while c < C && η[i, j] > τ_true[c]
            c += 1
        end
        Y[i, j] = c
    end
    return Y
end

function _inv_twolevel_fit(seed)
    Random.seed!(seed)
    p, K_B, K_W, nindiv, nobs = 3, 1, 1, 40, 4
    Λ_B = 0.6 .* randn(p, K_B); Λ_W = 0.4 .* randn(p, K_W)
    σ²_B = 0.1 .+ 0.1 .* rand(p); σ²_W = 0.1 .+ 0.1 .* rand(p)
    indiv = repeat(1:nindiv, inner = nobs)
    Y = zeros(p, nindiv * nobs)
    col = 0
    for i in 1:nindiv
        b = Λ_B * randn(K_B) .+ sqrt.(σ²_B) .* randn(p)
        for _ in 1:nobs
            col += 1
            Y[:, col] = b .+ Λ_W * randn(K_W) .+ sqrt.(σ²_W) .* randn(p)
        end
    end
    return fit_twolevel_gaussian(Y, indiv; K_B = K_B, K_W = K_W)
end

function _inv_cov_checks(Σ, R, Σ_hand)
    @test Σ ≈ Σ_hand atol = 1e-10
    @test Σ ≈ Σ' atol = 1e-12
    @test minimum(eigvals(Symmetric(Σ))) > -1e-10
    @test all(isapprox.(diag(R), 1.0; atol = 1e-12))
    @test all(abs.(R) .<= 1 + 1e-10)
    d = sqrt.(diag(Σ))
    @test R ≈ Σ ./ (d * d') atol = 1e-10
    @test R ≈ GLLVModels._cov2cor(Σ) atol = 1e-12
end

@testset "extractor invariants (residual cov/cor, rotation, cutpoints, repeatability)" begin

    @testset "Gaussian p=4 K=2 n=60: residual cov = ΛΛ' + diag(ψ)" begin
        Random.seed!(2031)
        p, K, n = 4, 2, 60
        Y = 0.7 .* randn(p, K) * randn(K, n) .+ 0.5 .* randn(p, n)
        fit = fit_gaussian_gllvm(Y; K = K, has_diag = true)
        Λ = getLoadings(fit; rotate = false)
        ψ = collect(Float64, fit.pars.σ²_B)
        Σ_hand = Λ * Λ' + Diagonal(ψ)

        Σ = extract_residual_cov(fit)
        R = extract_residual_cor(fit)
        @test size(Σ) == (p, p)
        _inv_cov_checks(Σ, R, Σ_hand)

        # gllvm-style aliases agree value for value.
        @test getResidualCov(fit) == Σ
        @test getResidualCov(fit; level = :unit) == extract_residual_cov(fit; level = :unit)
        @test getResidualCor(fit) == R

        # Rotation does not change ΛΛ'.
        Λr = getLoadings(fit; rotate = true)
        @test Λr * Λr' ≈ Λ * Λ' atol = 1e-10

        @test_throws ArgumentError extract_residual_cov(fit; level = :bogus)
        @test_throws ArgumentError extract_residual_cor(fit; level = :bogus)
        @test_throws ArgumentError getResidualCov(fit; level = :bogus)

        # Without a diagonal the unit tier is exactly ΛΛ'.
        fit0 = fit_gaussian_gllvm(Y; K = K)
        Λ0 = getLoadings(fit0; rotate = false)
        @test extract_residual_cov(fit0) ≈ Λ0 * Λ0' atol = 1e-10
    end

    @testset "extract_rotated_loadings: ΛrotΛrot' = ΛΛ', R orthogonal" begin
        Random.seed!(2032)
        p, K, n = 4, 2, 60
        Yg = 0.7 .* randn(p, K) * randn(K, n) .+ 0.5 .* randn(p, n)
        fit_g = fit_gaussian_gllvm(Yg; K = K)

        Random.seed!(2033)
        p3, n3 = 4, 120
        η = 0.5 .* randn(p3, 2) * randn(2, n3)
        Yp = [rand(GLLVModels.Distributions.Poisson(exp(η[i, j] + 0.5))) for i in 1:p3, j in 1:n3]
        fit_p = fit_poisson_gllvm(Yp; K = 2)

        for fit in (fit_g, fit_p)
            rl = extract_rotated_loadings(fit)
            Λraw = getLoadings(fit; rotate = false)
            @test size(rl.Λ) == size(Λraw)
            @test size(rl.R) == (2, 2)
            @test rl.Λ * rl.Λ' ≈ Λraw * Λraw' atol = 1e-10
            @test rl.R' * rl.R ≈ Matrix(I, 2, 2) atol = 1e-10
            @test rl.R * rl.R' ≈ Matrix(I, 2, 2) atol = 1e-10
            @test rl.Λ ≈ Λraw * rl.R atol = 1e-10
        end
    end

    @testset "TwoLevelFit: residual cov/cor, :W alias, repeatability" begin
        fit = _inv_twolevel_fit(2034)
        p = size(fit.Λ_B, 1)
        for (lvl, Λ, ψ) in ((:unit, fit.Λ_B, fit.σ²_B), (:unit_obs, fit.Λ_W, fit.σ²_W))
            Σ = extract_residual_cov(fit; level = lvl)
            R = extract_residual_cor(fit; level = lvl)
            _inv_cov_checks(Σ, R, Λ * Λ' + Diagonal(ψ))
        end
        @test getResidualCov(fit; level = :unit) == extract_residual_cov(fit; level = :unit)
        @test getResidualCov(fit; level = :W) ≈ extract_residual_cov(fit; level = :W)
        @test getResidualCov(fit; level = :W) ≈ extract_residual_cov(fit; level = :unit_obs)
        @test getResidualCov(fit; level = :W) ≈ fit.Σ_W atol = 1e-10
        @test_throws ArgumentError extract_residual_cov(fit; level = :bogus)
        @test_throws ArgumentError getResidualCov(fit; level = :bogus)

        r = extract_repeatability(fit)
        @test r == repeatability(fit)
        @test length(r) == p
        @test all(0 .<= r .<= 1)
        vB = diag(fit.Σ_B); vW = diag(fit.Σ_W)
        @test r ≈ vB ./ (vB .+ vW) atol = 1e-10
    end

    @testset "extract_cutpoints: ordering, copy semantics, shape" begin
        C = 4
        Y = _inv_ordinal_data(2035, 4, 200, C)

        fit = fit_ordinal_gllvm(Y; K = 1)
        cp = extract_cutpoints(fit)
        @test length(cp.τ) == C - 1
        @test cp.C == C
        @test all(diff(cp.τ) .> 0)
        τ_before = copy(fit.τ)
        cp.τ .= 0.0
        @test fit.τ == τ_before
        @test cp.τ !== fit.τ
        cp2 = extract_cutpoints(fit)
        @test cp2.τ == τ_before

        fitp = fit_ordinal_gllvm_pertrait(Y; K = 1)
        cpp = extract_cutpoints(fitp)
        p = size(Y, 1)
        @test size(cpp.τ, 1) == p
        @test all(2 .<= cpp.C .<= C)
        @test size(cpp.τ, 2) == maximum(cpp.C) - 1
        for t in 1:p
            @test all(diff(cpp.τ[t, 1:(cpp.C[t] - 1)]) .> 0)
        end
        τp_before = copy(fitp.τ)
        C_before = copy(fitp.C)
        cpp.τ .= 0.0
        cpp.C .= 0
        @test isequal(fitp.τ, τp_before)
        @test fitp.C == C_before
        @test cpp.τ !== fitp.τ
        @test cpp.C !== fitp.C
    end
end
