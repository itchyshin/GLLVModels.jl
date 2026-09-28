using GLLVModels, Test, Random, LinearAlgebra, Distributions, SparseArrays, Statistics

# Louis (1982) / Supplemented EM (Meng & Rubin 1991) observed information at
# the EM MLE — see `em_observed_information` in `src/em_phylo.jl`. The function
# computes I_obs = (I − DM) · I_complete via ForwardDiff on the EM map's
# Jacobian and the expected complete-data log-likelihood, then back-transforms
# σ_ε's SE via the delta method (the only non-identity link in the packing).
#
# This test pulls in the EM file directly (matches `test_em_phylo.jl`'s
# convention: em_phylo.jl is NOT wired into the GLLVModels module). Guard the
# include so that when the full suite runs (test_em_phylo.jl already pulled
# em_phylo.jl into Main), we reuse the existing definitions rather than
# re-`include`-ing — a second include rebinds the file's types in Main and
# breaks `GLLVModels.AugmentedPhy` method dispatch.
isdefined(Main, :em_observed_information) ||
    include(joinpath(@__DIR__, "..", "src", "em_phylo.jl"))

function _sim_phylo_unique_louis(tree, Λ_B, σ_phy, σ_eps, n; seed = 0)
    Random.seed!(seed)
    Σ_phy = GLLVModels.sigma_phy_dense(tree; σ²_phy = 1.0)
    p, K_B = size(Λ_B)
    η_B = randn(K_B, n)
    φ   = cholesky(Symmetric(Σ_phy)).L * randn(p)
    z   = σ_phy .* φ
    y   = Λ_B * η_B .+ reshape(z, p, 1) .+ σ_eps .* randn(p, n)
    return y, Σ_phy
end

# Dense fit at the exact MLE. SEM equals the observed information only at a
# fixed point. The default stopping rule (relative f_tol = 1e-10) halts these
# fits after 21 to 29 LBFGS steps with a gradient near 3e-3, and at that point
# the SEM-vs-Hessian gap on the weakly identified σ_phy terms is 8e-4 to 2e-3.
# That is optimiser slack, not the round-off these gates bound; it also moved
# with test order (1.056e-3 at p = 10 in the full suite, 9.2e-4 standalone).
# Converged to a gradient near 5e-6, every gap below is under 1e-6.
_louis_dense_fit(y, Σ) = fit_gaussian_gllvm(y; K = 1, has_phy_unique = true, Σ_phy = Σ,
                                            f_tol = 0.0, x_tol = 0.0, g_tol = 1e-8,
                                            iterations = 5_000)

@testset "EM observed information (Louis / Supplemented EM)" begin

    # -- p = 6 fixture (K_B = 1, n = 400) --------------------------------------
    # CORRECTED 2026-08-26: the seed-30 optimum is NOT all-positive. Measured:
    # σ_phy = [-0.323, 0.551, 0.360, 0.347, 0.777, 1.618] against a truth of +0.9,
    # and ADEMP over 40 replicates shows a ~62% systematic UNDERESTIMATE (mean 0.343)
    # with 28% of components negative. Starting from the truth converges to the same
    # point, so it is not a basin artefact. See docs/dev-log/check-log.md 2026-08-26
    # and docs/dev-log/pending/sigma-phy-recovery-ademp.jl.
    tree1   = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p1      = tree1.n_leaves
    Λ_B1    = reshape([0.8, 0.6, 0.4, -0.3, 0.5, -0.2], p1, 1)
    σ_phy1  = fill(0.9, p1)
    n1      = 400
    y1, Σ1  = _sim_phylo_unique_louis(tree1, Λ_B1, σ_phy1, 0.5, n1; seed = 30)

    @testset "SE PRIMARY GATE: EM-SEM SEs match dense-Hessian SEs (p=6)" begin
        # Dense fit + its Wald SEs (ForwardDiff Hessian of the marginal NLL).
        fit  = _louis_dense_fit(y1, Σ1)
        ci   = confint(fit; y = y1, Σ_phy = Σ1)
        @test ci.pd_hessian

        # Evaluate SEM at the DENSE MLE (the asymptotic-equivalence gate is the
        # SEM-vs-Hessian comparison at a common point; EM warm-started at the
        # dense MLE still drifts ~1e-6 in parameter space before its first
        # convergence check, which on poorly-identified σ_phy directions inflates
        # the SE relative gap. Constructing a synthetic EMPhyloFit at the dense
        # MLE pins the comparison point; this is a legitimate use of
        # `em_observed_information`, which is a generic info-at-θ evaluator).
        emf_at_dense = EMPhyloFit(Matrix{Float64}(fit.pars.Λ),
                                  float(fit.pars.σ_eps),
                                  Vector{Float64}(fit.pars.σ_phy),
                                  fit.logLik, 0, true,
                                  Float64[fit.logLik], zeros(p1), zeros(p1))
        info = em_observed_information(emf_at_dense, y1, Σ1)
        @test info.pd

        @test length(info.term) == length(ci.term)
        @test info.term == ci.term

        # Relative tolerance on SE: ≤ 1e-3 (asymptotically equivalent; SEM is
        # algebraically exact at the MLE, so divergence is FP round-off in the
        # ForwardDiff Jacobian of the EM map vs ForwardDiff Hessian of the
        # marginal NLL — both evaluated at the same θ̂).
        for i in eachindex(ci.se)
            @test isfinite(info.se[i])
            @test isfinite(ci.se[i])
            rel = abs(info.se[i] - ci.se[i]) / max(abs(ci.se[i]), eps())
            @test rel ≤ 1e-3
        end
    end

    @testset "EM converged at its own MLE: SEs still match within 2e-3" begin
        # Independent of the same-point comparison above, the SEM SEs at the
        # EM's own fixed point should agree with the dense Hessian SEs. With
        # both fits converged, the EM and dense optima coincide to round-off.
        fit = _louis_dense_fit(y1, Σ1)
        ci  = confint(fit; y = y1, Σ_phy = Σ1)
        emf = em_fit_phylo(y1, 1, Σ1;
                           λ_init = fit.pars.Λ,
                           σ_eps_init = fit.pars.σ_eps,
                           σ_phy_init = fit.pars.σ_phy,
                           tol = 1e-12, max_iter = 50_000)
        info = em_observed_information(emf, y1, Σ1)
        for i in eachindex(ci.se)
            rel = abs(info.se[i] - ci.se[i]) / max(abs(ci.se[i]), eps())
            @test rel ≤ 2e-3
        end
    end

    # -- p = 10 fixture (larger) -----------------------------------------------
    tree2  = GLLVModels.augmented_phy("(((((A:0.2,B:0.2):0.2,C:0.4):0.2,(D:0.3,E:0.3):0.3):0.1," *
                           "((F:0.2,G:0.2):0.3,H:0.5):0.1):0.1,(I:0.4,J:0.4):0.2);")
    p2     = tree2.n_leaves
    Λ_B2   = reshape([0.7, 0.5, -0.4, 0.3, 0.6, -0.5, 0.4, 0.2, 0.8, -0.3], p2, 1)
    σ_phy2 = fill(0.8, p2)
    n2     = 500
    y2, Σ2 = _sim_phylo_unique_louis(tree2, Λ_B2, σ_phy2, 0.5, n2; seed = 30)

    @testset "SE PRIMARY GATE: EM-SEM SEs match dense-Hessian SEs (p=10)" begin
        fit = _louis_dense_fit(y2, Σ2)
        ci  = confint(fit; y = y2, Σ_phy = Σ2)
        @test ci.pd_hessian

        emf_at_dense = EMPhyloFit(Matrix{Float64}(fit.pars.Λ),
                                  float(fit.pars.σ_eps),
                                  Vector{Float64}(fit.pars.σ_phy),
                                  fit.logLik, 0, true,
                                  Float64[fit.logLik], zeros(p2), zeros(p2))
        info = em_observed_information(emf_at_dense, y2, Σ2)
        @test info.pd
        @test info.term == ci.term

        for i in eachindex(ci.se)
            @test isfinite(info.se[i])
            @test isfinite(ci.se[i])
            rel = abs(info.se[i] - ci.se[i]) / max(abs(ci.se[i]), eps())
            @test rel ≤ 1e-3
        end
    end

    @testset "EM-converged SEs vs dense (p=10) within 2e-3" begin
        fit = _louis_dense_fit(y2, Σ2)
        ci  = confint(fit; y = y2, Σ_phy = Σ2)
        emf = em_fit_phylo(y2, 1, Σ2;
                           λ_init = fit.pars.Λ,
                           σ_eps_init = fit.pars.σ_eps,
                           σ_phy_init = fit.pars.σ_phy,
                           tol = 1e-12, max_iter = 50_000)
        info = em_observed_information(emf, y2, Σ2)
        for i in eachindex(ci.se)
            rel = abs(info.se[i] - ci.se[i]) / max(abs(ci.se[i]), eps())
            # Louis (EM-converged) observed information vs the dense ForwardDiff
            # Hessian. The 0.00219 gap CI once saw here came from the dense fit
            # stopping early, not from EM-vs-LBFGS optima; with _louis_dense_fit
            # both routes meet at the same MLE, so the titled 2e-3 gate holds.
            @test rel ≤ 2e-3
        end
    end

    @testset "info matrix is symmetric and PD at an interior MLE" begin
        emf  = em_fit_phylo(y1, 1, Σ1; tol = 1e-12, max_iter = 50_000)
        info = em_observed_information(emf, y1, Σ1)
        @test info.pd
        @test maximum(abs.(info.info .- info.info')) < 1e-8
        @test all(diag(info.info) .> 0)
    end

    @testset "K_B > 1 raises an informative error (out of scope)" begin
        # SEM on the raw vec(Λ_B) parameterisation gives a singular I_obs along
        # the rotation directions; QR-rotation onto the strict-lower orbit
        # before packing is not yet implemented. We keep the function honest
        # by raising rather than returning garbage.
        Random.seed!(99)
        Λ_B22 = randn(p1, 2)
        for k in 1:2, i in 1:(k - 1)
            Λ_B22[i, k] = 0.0
        end
        y22, Σ22 = _sim_phylo_unique_louis(tree1, Λ_B22, fill(0.7, p1), 0.5, 200; seed = 1)
        emf22 = em_fit_phylo(y22, 2, Σ22; tol = 1e-12, max_iter = 50_000)
        @test_throws ArgumentError em_observed_information(emf22, y22, Σ22)
    end
end
