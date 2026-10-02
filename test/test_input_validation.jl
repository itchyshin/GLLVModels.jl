using GLLVModels, Test, Random, LinearAlgebra

# Input-validation sweep (issues #134 #150 #151 #153 #159 #162 #139 #141).
@testset "input validation sweep" begin

    @testset "#134 phylo loadings without Σ_phy throw" begin
        Random.seed!(1)
        p, n = 4, 6
        y = randn(p, n)
        Λ_B = 0.3 .* randn(p, 1)
        L = 0.2 .* randn(p, 1)
        @test_throws ArgumentError GLLVModels.gaussian_marginal_loglik(y, Λ_B, 0.5; Λ_phy = L)
        @test_throws ArgumentError GLLVModels.gaussian_marginal_loglik(y, Λ_B, 0.5; σ_phy = fill(0.2, p))
        # valid inputs are unchanged
        @test isfinite(GLLVModels.gaussian_marginal_loglik(y, Λ_B, 0.5))
        @test isfinite(GLLVModels.gaussian_marginal_loglik(y, Λ_B, 0.5; Λ_phy = L, Σ_phy = Matrix(1.0I, p, p)))
    end

    @testset "#150 Λ_B first dim must equal p" begin
        Random.seed!(2)
        p, n = 4, 6
        y = randn(p, n)
        @test_throws ArgumentError GLLVModels.gaussian_marginal_loglik(
            y, 0.3 .* randn(p - 1, 1), 0.5;
            Λ_phy = 0.2 .* randn(p, 1), Σ_phy = Matrix(1.0I, p, p))
        @test_throws ArgumentError GLLVModels.gaussian_marginal_loglik(
            y, 0.3 .* randn(p - 1, 1), 0.5)
    end

    @testset "#151 low_rank_chol requires positive d" begin
        Λ = [1.0 0.0; 0.5 0.5; 0.2 0.1]
        @test_throws ArgumentError GLLVModels.low_rank_chol(Λ, [1.0, 0.0, 2.0])
        @test_throws ArgumentError GLLVModels.low_rank_chol(Λ, [1.0, -0.1, 2.0])
        @test GLLVModels.low_rank_chol(Λ, [1.0, 0.1, 2.0]) isa GLLVModels.LowRankPlusDiagChol
    end

    @testset "#153 per-node binary check in augmented_phy" begin
        # one polytomy (A,B,C) + one unary node (D) net to 2p-1 = 9 nodes
        @test_throws ErrorException GLLVModels.augmented_phy(
            "((A:1,B:1,C:1):1,((D:1):1,E:1):1);")
        @test_throws ErrorException GLLVModels.augmented_phy("((A:1,B:1,C:1):1,D:1);")
        @test GLLVModels.augmented_phy("((A:0.1,B:0.2):0.3,C:0.5);").n_leaves == 3
    end

    @testset "#159 size(F, i) follows Base for i > 2" begin
        F = GLLVModels.low_rank_chol([1.0 0.0; 0.5 0.5; 0.2 0.1], [1.0, 0.1, 2.0])
        @test size(F, 1) == 3
        @test size(F, 2) == 3
        @test size(F, 3) == 1
        @test size(F, 7) == 1
        @test_throws BoundsError size(F, 0)
    end

    @testset "#162 edge wrapper validates before building Σ_phy" begin
        p = 400
        nwk = "(L1:1.0,L2:1.0);"
        for i in 3:p
            nwk = "(" * nwk[1:end-1] * ":1.0,L$i:1.0);"
        end
        phy = GLLVModels.edge_phy(nwk)
        y = randn(p, 2)
        Λ_B = randn(p, 1)
        f() = try
            GLLVModels.gaussian_marginal_loglik_edge_phy(y, Λ_B, 0.5; phy = phy)
        catch e
            e
        end
        @test f() isa ArgumentError
        @test (@allocated f()) < p^2 * 8 ÷ 4   # far below a dense p×p matrix
    end

    @testset "#139 bridge_fit accepts integral doubles for d" begin
        Random.seed!(3)
        Y = randn(4, 30) .+ 0.5 .* randn(1, 30)
        b1 = GLLVModels.bridge_fit(; y = Y, family = "gaussian", d = 1)
        b2 = GLLVModels.bridge_fit(; y = Y, family = "gaussian", d = 1.0)
        @test b2.loglik ≈ b1.loglik
        @test_throws ArgumentError GLLVModels.bridge_fit(; y = Y, family = "gaussian", d = 1.5)
    end

    @testset "#141 correlation() with a degenerate-variance trait" begin
        Random.seed!(4)
        y = randn(3, 40)
        fit = fit_gaussian_gllvm(y; K = 1)
        Λ0 = copy(fit.pars.Λ); Λ0[2, :] .= 0.0
        pars = merge(fit.pars, (Λ = Λ0, σ_eps = 0.0))
        fit.pars.σ²_B === nothing || (pars = merge(pars, (σ²_B = zero(fit.pars.σ²_B),)))
        fit.pars.σ²_W === nothing || (pars = merge(pars, (σ²_W = zero(fit.pars.σ²_W),)))
        bad = GLLVModels.GllvmFit(fit.model, pars, fit.logLik, fit.n_iter,
                                  fit.converged, fit.optim_result, fit.cputime)
        R = @test_logs (:warn, r"non-positive variance") GLLVModels.correlation(bad)
        @test isnan(R[2, 1]) && isnan(R[1, 2]) && isnan(R[2, 3])
        @test isnan(R[2, 2])
        @test R[1, 1] == 1.0
        # a healthy fit is unchanged and silent
        R2 = @test_logs GLLVModels.correlation(fit)
        @test all(isfinite, R2)
        @test all(==(1.0), diag(R2))
    end

    @testset "#141 cases the old correlation() got wrong" begin
        Random.seed!(5)
        y = randn(3, 40)
        fit = fit_gaussian_gllvm(y; K = 1, has_diag = true)
        @test fit.pars.σ²_B !== nothing
        mk(Λ, σ²_B) = GLLVModels.GllvmFit(fit.model,
            merge(fit.pars, (Λ = Λ, σ_eps = 0.0, σ²_B = σ²_B,
                             σ²_W = fit.pars.σ²_W === nothing ? nothing : zero(fit.pars.σ²_W))),
            fit.logLik, fit.n_iter, fit.converged, fit.optim_result, fit.cputime)
        quiet(f) = Test.@test_logs min_level = Base.CoreLogging.Error f()
        # (a) Σ[i,i]·Σ[j,j] underflows to 0 although each variance is positive:
        # the old code returned Inf (1e-200 / sqrt(0)).
        tiny = mk(fill(1e-100, 3, 1), zeros(3))
        Ra = quiet(() -> GLLVModels.correlation(tiny))
        @test all(isfinite, Ra)
        @test all(≈(1.0), Ra)
        # (b) tiny negative round-off variance: the old code hit sqrt of a
        # negative product (DomainError).
        σ²neg = [0.0, -1e-18, 0.0]
        neg = mk(reshape([0.5, 0.0, 0.4], 3, 1), σ²neg)
        Rb = quiet(() -> GLLVModels.correlation(neg))
        @test isnan(Rb[2, 1]) && isnan(Rb[1, 2]) && isnan(Rb[2, 2])
        @test isfinite(Rb[1, 3]) && Rb[1, 3] ≈ 1.0
    end
end
