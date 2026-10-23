# Observation weights (slice W4-1a): the shared Laplace kernel, the Poisson fit, the shape rules
# and the refusals. R semantics (gllvmTMB 9539352f6, src/gllvmTMB.cpp): a weight multiplies its
# cell's conditional log-density; 0 drops the cell; fractional weights are allowed. The fit-level
# parity with gllvmTMB is test/test_weights_twin_p1.jl.
using Test
using GLLVModels
using Distributions: Poisson, NegativeBinomial, Binomial, Normal
using Random

@testset "observation weights" begin
    rng = Xoshiro(7)
    p, n, K = 4, 30, 1
    lam = [0.6, -0.4, 0.5, 0.3]; mu = [0.8, 0.4, 0.6, 1.0]
    z = randn(rng, n)
    Y = [rand(rng, Poisson(exp(mu[t] + lam[t] * z[s]))) for t in 1:p, s in 1:n]
    Λ = reshape([0.5, -0.3, 0.4, 0.2], p, 1); β = [0.7, 0.3, 0.5, 0.9]
    marg(Yx, Λx, βx; kw...) = GLLVModels.poisson_marginal_loglik_laplace(Yx, Λx, βx; kw...)

    @testset "kernel: weights generalise the mask" begin
        @test marg(Y, Λ, β; weights = ones(p, n)) ≈ marg(Y, Λ, β) atol = 1e-10
        W0 = ones(p, n); W0[2, 5] = 0; W0[3, 11] = 0
        M = trues(p, n); M[2, 5] = false; M[3, 11] = false
        @test marg(Y, Λ, β; weights = W0) ≈ marg(Y, Λ, β; mask = M) atol = 1e-10
        # a weight at a masked cell is never used: NaN or any value there leaves the value unchanged
        Wn = ones(p, n); Wn[2, 5] = NaN; Wn[3, 11] = 7.0
        @test marg(Y, Λ, β; weights = Wn, mask = M) == marg(Y, Λ, β; weights = ones(p, n), mask = M)
        @test marg(Y, Λ, β; weights = Wn, mask = M) ≈ marg(Y, Λ, β; mask = M) atol = 1e-10
        # weight 2 on trait 2 = trait 2 entered twice (conditionally independent copies with the
        # same intercept and loading): an exact identity of the Laplace marginal.
        W2 = ones(p, n); W2[2, :] .= 2
        Yd = vcat(Y, Y[2:2, :]); Λd = vcat(Λ, Λ[2:2, :]); βd = vcat(β, β[2])
        @test marg(Y, Λ, β; weights = W2) ≈ marg(Yd, Λd, βd) atol = 1e-9
        # fractional weights are accepted and move the value
        Wf = fill(0.5, p, n)
        @test isfinite(marg(Y, Λ, β; weights = Wf))
        @test abs(marg(Y, Λ, β; weights = Wf) - marg(Y, Λ, β)) > 1
    end

    @testset "Poisson fit" begin
        f0 = fit_gllvm(Y; family = Poisson(), K = K)
        @test f0.weights === nothing
        f1 = fit_gllvm(Y; family = Poisson(), K = K, weights = 1.0)
        @test f1.weights == ones(p, n)
        @test isapprox(f1.loglik, f0.loglik; atol = 1e-6)
        @test isapprox(f1.β, f0.β; atol = 1e-4)
        W0 = ones(p, n); W0[1, 3] = 0; W0[4, 20] = 0
        M = trues(p, n); M[1, 3] = false; M[4, 20] = false
        fz = fit_gllvm(Y; family = Poisson(), K = K, weights = W0)
        fm = fit_gllvm(Y; family = Poisson(), K = K, mask = M)
        @test isapprox(fz.loglik, fm.loglik; atol = 1e-6)
        @test isapprox(fz.β, fm.β; atol = 1e-4)
        Wc = 0.5 .+ rand(Xoshiro(3), p, n)
        fw = fit_gllvm(Y; family = Poisson(), K = K, weights = Wc)
        @test fw.converged
        # the stored objective is the weighted marginal at the estimate
        @test isapprox(marg(Y, fw.Λ, fw.β; weights = Wc), fw.loglik; atol = 1e-6)
        @test isapprox(fit_poisson_gllvm(Y; K = K, weights = Wc).loglik, fw.loglik; atol = 1e-8)
        # post-fit: weighted modes on the training data; intervals refused
        Z = getLV(fw, Y; rotate = false)
        @test size(Z) == (n, K)
        @test Z[:, 1] ≈ [GLLVModels._laplace_mode(Poisson(), Y[:, s], ones(Int, p), fw.Λ, fw.β,
                                                    LogLink(); weights = Wc[:, s])[1] for s in 1:n]
        @test_throws ArgumentError getLV(fw, Y[:, 1:10])
        @test_throws ArgumentError confint(fw, Y)
        @test_throws ArgumentError loglikelihood(fw)            # gllvmTMB's logLik() aborts too
        @test_throws ArgumentError aic(fw)
        @test loglikelihood(f0) == f0.loglik
        @test size(predict(fw, Y)) == (p, n)
        # formula front end, no covariates
        data = (x = randn(Xoshiro(5), n),)
        fg = gllvm(@formula(y ~ 1), Y, data; family = Poisson(), K = K, weights = Wc)
        @test isapprox(fg.loglik, fw.loglik; atol = 1e-8)
    end

    @testset "shapes" begin
        nw(w; kw...) = GLLVModels._normalize_weights(w, p, n; kw...)
        u = collect(1.0:n)
        @test nw(nothing) === nothing
        @test nw(2) == fill(2.0, p, n)
        @test nw(u) == repeat(u', p, 1)                         # one weight per unit
        @test nw(reshape(u, 1, n)) == repeat(u', p, 1)
        @test nw(reshape([1.0, 2, 3, 4], p, 1)) == repeat([1.0, 2, 3, 4], 1, n)
        Wm = rand(Xoshiro(1), p, n)
        @test nw(Wm) == Wm && nw(Wm) !== Wm
        @test_throws ArgumentError nw(ones(p))                  # per-trait needs reshape(w, p, 1)
        @test_throws ArgumentError nw(ones(n, p))               # units × traits: permutedims
        @test_throws ArgumentError nw(-1.0)
        @test_throws ArgumentError nw(fill(NaN, p, n))
        @test_throws ArgumentError nw("1")
        # unobserved cells: any value, stored as 0
        Wn = ones(p, n); Wn[2, 3] = NaN
        M = trues(p, n); M[2, 3] = false
        @test nw(Wn; mask = M)[2, 3] == 0.0
        Ym = Matrix{Union{Missing, Int}}(Y); Ym[2, 3] = missing
        Wmiss = Matrix{Union{Missing, Float64}}(ones(p, n)); Wmiss[2, 3] = missing
        @test nw(Wmiss; Y = Ym)[2, 3] == 0.0
        @test_throws ArgumentError nw(Wmiss)                    # observed there: refused
        fmiss = fit_gllvm(Ym; family = Poisson(), K = K, weights = Wmiss)
        fmask = fit_gllvm(Y; family = Poisson(), K = K, mask = M)
        @test isapprox(fmiss.loglik, fmask.loglik; atol = 1e-6)
    end

    @testset "refusals" begin
        Yb = min.(Y, 3); Nb = fill(3, p, n)
        Yg = randn(Xoshiro(2), p, n)
        @test_throws ArgumentError fit_gllvm(Yb; family = Binomial(), K = K, N = Nb, weights = 1.0)
        @test_throws ArgumentError fit_gllvm(Y; family = NegativeBinomial(1.0, 0.5), K = K, weights = 1.0)
        @test_throws ArgumentError fit_gllvm(Yg; family = Normal(), K = K, weights = 1.0)
        @test_throws ArgumentError fit_gllvm(Y; family = Poisson(), K = K, row_eff = :fixed, weights = 1.0)
        @test_throws ArgumentError fit_gllvm(Y; family = Poisson(), K = K, disp_group = :species, weights = 1.0)
        @test_throws ArgumentError fit_gllvm(Y; family = Poisson(), weights = 1.0)
        @test_throws ArgumentError fit_gllvm(Y; family = Poisson(), K = K, aghq = 3, weights = 1.0)
        @test_throws ArgumentError fit_gllvm(Y; family = Poisson(), K = K, X_lv = randn(n, 1), weights = 1.0)
        @test_throws ArgumentError fit_poisson_gllvm(Y; K = K, aghq = 3, weights = 1.0)
        @test_throws ArgumentError fit_poisson_gllvm(Y; K = K, X_lv = randn(n, 1), weights = 1.0)
        data = (x = randn(Xoshiro(5), n),)
        @test_throws ArgumentError gllvm(@formula(y ~ 1 + x), Y, data; family = Poisson(), K = K, weights = 1.0)
        @test_throws ArgumentError gllvm(@formula(y ~ 1), Yg, data; family = Normal(), K = K, weights = 1.0)
        @test_throws ArgumentError gllvm(@formula(y ~ 1), Y, data; family = NegativeBinomial(1.0, 0.5), K = K, weights = 1.0)
        # the refusal names what blocks the route
        msg = try
            fit_gllvm(Yb; family = Binomial(), K = K, N = Nb, weights = 1.0); ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("trial", msg) && occursin("N", msg)
    end
end
