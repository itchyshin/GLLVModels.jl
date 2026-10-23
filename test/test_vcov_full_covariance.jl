# vcov(fit, Y) returns the full inverse observed information, not Diagonal(se^2)
# (itchyshin/GLLVModels.jl#652). R-vs-Julia agreement on a gllvmTMB P1 fit is in
# test_namespace_numeric_p1_twin.jl; this file pins the structure on both code
# paths (legacy Gaussian GllvmFit and the shared non-Gaussian _family_wald) and the
# non-positive-definite convention.
using GLLVModels, Test, Random, LinearAlgebra, Distributions
using GLLVModels: StatsAPI, vcov, ForwardDiff

@testset "vcov returns the full covariance" begin
    @testset "Gaussian GllvmFit" begin
        rng = MersenneTwister(7)
        p, K, n = 5, 1, 80
        Y = 0.7 .* randn(rng, p, K) * randn(rng, K, n) .+ 0.5 .* randn(rng, p, n)
        fit = fit_gaussian_gllvm(Y; K = K)
        V = vcov(fit, Y)
        ci = confint(fit, Y)
        @test V isa Matrix{Float64}
        @test size(V) == (length(ci.term), length(ci.term))
        @test diag(V) == ci.se .^ 2                    # diagonal unchanged, bit for bit
        @test issymmetric(V)
        @test maximum(abs, V - Diagonal(diag(V))) > 0  # off-diagonals no longer zero
        H = ForwardDiff.hessian(GLLVModels._confint_reconstruct_nll(fit, Y, nothing, nothing),
                                fit.pars.θ_packed)
        @test V ≈ inv(Symmetric((H + H') / 2)) rtol = 1e-10
        sel = findall(startswith("Lambda"), ci.term)
        @test vcov(fit, Y; parm = "Lambda") == V[sel, sel]
        @test vcov(fit; y = Y) == V
    end

    @testset "non-Gaussian (_family_wald path)" begin
        rng = MersenneTwister(21)
        p, n = 5, 140
        β = 0.5 .* randn(rng, p) .+ 1.0
        Λ = 0.5 .* randn(rng, p, 1)
        Y = [rand(rng, Poisson(exp(β[t] + Λ[t] * z))) for t in 1:p, z in randn(rng, n)]
        fit = fit_poisson_gllvm(Y; K = 1)
        ci = confint(fit, Y; method = :wald)
        V = vcov(fit, Y)
        @test V isa Matrix{Float64}
        @test size(V) == (length(ci.term), length(ci.term))
        fin = isfinite.(ci.se)
        @test diag(V)[fin] == ci.se[fin] .^ 2
        @test all(isnan, diag(V)[.!fin])
        @test issymmetric(V[fin, fin])
        @test maximum(abs, V[fin, fin] - Diagonal(diag(V)[fin])) > 0
        @test vcov(fit, Y; parm = "beta") == V[1:p, 1:p]
        @test_throws ArgumentError vcov(fit, Y; method = :profile)
    end

    @testset "non-PD convention: invalid variance -> NaN row and column" begin
        # Hessian with an indefinite 2x2 block and a well-identified third parameter:
        # Cholesky fails, the conditioned-out direction gets NaN rows/columns, the
        # remaining block is the inverse of its sub-Hessian, and nothing is regularised.
        H = [1.0 2.0 0.0; 2.0 1.0 0.0; 0.0 0.0 4.0]
        ad = GLLVModels._FamilyCI(zeros(3), θ -> 0.0, ["a", "b", "c"], fill(:linear, 3),
                                  rng -> nothing, y -> nothing)
        r = GLLVModels._family_wald(ad, [1, 2, 3], 0.95; hessian = H, covariance = true)
        @test !r.pd_hessian
        V = r.covariance
        bad = findall(isnan, r.se)
        good = findall(isfinite, r.se)
        @test !isempty(bad) && 3 in good
        @test all(isnan, V[bad, :]) && all(isnan, V[:, bad])
        @test V[3, 3] == r.se[3]^2 && V[3, 3] ≈ 0.25
        # the default call shape is unchanged (no covariance field)
        @test !haskey(GLLVModels._family_wald(ad, [1, 2, 3], 0.95; hessian = H), :covariance)
    end
end
