# Targeted coverage for src/[g-o]*.jl validation and contract branches (refs coverage 27 Nov).
using Test, GLLVModels, LinearAlgebra, SparseArrays, Optim

function _jci_phy()
    Q = sparse([2.0 -0.5; -0.5 2.0])
    return GLLVModels.PrecisionPhy(findnz(Q)..., 2, 2, ["t1", "t2"],
        logdet(cholesky(Symmetric(Matrix(Q)))), 1.0, [1, 2])
end

@testset "joint covariance input validation (#coverage-go)" begin
    phy = _jci_phy()
  L = reshape([0.6, 0.4, 0.3], 3, 1)
    good_term = [GLLVModels.GroupingTerm(:unit; mode = :indep, common = false)]
    good_inc = [GLLVModels._grouped_incidence([1, 1, 2, 2], 4)]

    @test_throws ArgumentError GLLVModels._joint_covariance_identification(
        phy, Int[], good_term, good_inc, L)
    @test_throws ArgumentError GLLVModels._joint_covariance_identification(
        phy, ["a", "b"], good_term, good_inc, L)
    @test_throws DimensionMismatch GLLVModels._joint_covariance_identification(
        phy, [1, 2, 1, 2], good_term, SparseMatrixCSC{Float64, Int}[], L)
    bad_term = [GLLVModels.GroupingTerm(:unit; mode = :dep, common = false)]
    @test_throws ArgumentError GLLVModels._joint_covariance_identification(
        phy, [1, 2, 1, 2], bad_term, good_inc, L)
    dup_terms = [GLLVModels.GroupingTerm(:unit; mode = :indep, common = false),
                 GLLVModels.GroupingTerm(:unit; mode = :indep, common = false)]
    @test_throws ArgumentError GLLVModels._joint_covariance_identification(
        phy, [1, 2, 1, 2], dup_terms, [good_inc[1], good_inc[1]], L)
    @test_throws ArgumentError GLLVModels._joint_covariance_identification(
        phy, [1, 2, 1, 2], good_term, [fill(1.0, 4, 2)], L)
    @test_throws DimensionMismatch GLLVModels._joint_covariance_identification(
        phy, [1, 2, 1, 2], good_term, [sparse(fill(1.0, 3, 2))], L)
    nan_inc = sparse([1, 2, 3, 4], [1, 1, 2, 2], [1.0, NaN, 1.0, 1.0], 4, 2)
    @test_throws ArgumentError GLLVModels._joint_covariance_identification(
        phy, [1, 2, 1, 2], good_term, [nan_inc], L)
    @test_throws DimensionMismatch GLLVModels._joint_covariance_identification(
        phy, [1, 2, 1, 2], good_term, good_inc,
        reshape([0.5, 0.4, 0.3], 3, 1); phylo_unique_variance = [0.1, 0.2])
    @test_throws ArgumentError GLLVModels._joint_covariance_identification(
        phy, [1, 2, 1, 2], good_term, good_inc, zeros(0, 1))
end

@testset "link_residual scalar contracts (#coverage-go)" begin
    @test GLLVModels.link_residual(GLLVModels.Binomial(), GLLVModels.CLogLogLink(),
                                   0.5, nothing) ≈ π^2 / 6
    @test GLLVModels.link_residual(GLLVModels.Poisson(), GLLVModels.LogLink(),
                                   0.0, nothing) == 0.0
    @test GLLVModels.link_residual(GLLVModels.Poisson(), GLLVModels.LogLink(),
                                   -1.0, nothing) == 0.0
    @test_throws ArgumentError GLLVModels.link_residual(
        GLLVModels.Binomial(), GLLVModels.IdentityLink(), 0.5, nothing)
end

@testset "link_residual masked trait means (#coverage-go)" begin
    μ = [1.0 2.0 3.0 10.0; 1.0 1.0 1.0 1.0]
    mask = trues(size(μ))
    mask[1, 4] = false
    m_full = GLLVModels._masked_trait_mean(μ, nothing)
    m_masked = GLLVModels._masked_trait_mean(μ, mask)
    @test m_full[1] ≈ 4.0
    @test m_masked[1] ≈ 2.0
    σ_full = GLLVModels.link_residual(GLLVModels.Poisson(), GLLVModels.LogLink(),
                                      m_full[1], nothing)
    σ_masked = GLLVModels.link_residual(GLLVModels.Poisson(), GLLVModels.LogLink(),
                                        m_masked[1], nothing)
    @test σ_masked > σ_full
    @test m_masked[2] ≈ m_full[2]
end

@testset "marginal target interval argument guards (#coverage-go)" begin
    f(x) = sum(abs2, x)
    t = [(name = "a", value = x -> x[1], transform = :identity)]
    @test_throws ArgumentError GLLVModels._marginal_target_intervals(
        f, [0.0], [t..., (name = "a", value = x -> x[2], transform = :identity)];
        converged = true)
    @test_throws ArgumentError GLLVModels._marginal_target_intervals(
        f, [0.0], t; converged = true, gradient_tolerance = 0.0)
end

@testset "loading profile pin-matrix validation (#coverage-go)" begin
    @test_throws ArgumentError GLLVModels._lambda_constraint_is_pinned(nothing, 0, 1)
    @test_throws ArgumentError GLLVModels._lambda_constraint_is_pinned(nothing, 2, 0)
    @test_throws ArgumentError GLLVModels._lambda_constraint_is_pinned(zeros(2, 3), 2, 2)
    @test_throws ArgumentError GLLVModels._enumerate_free_lambda_entries(
        nothing, 2, 2; entries = reshape([1], 1, 1))
    @test_throws ArgumentError GLLVModels._enumerate_free_lambda_entries(
        nothing, 2, 2; entries = [9 1])
end

@testset "optimizer selection helpers (#coverage-go)" begin
    @test GLLVModels._refuse_optimizer(nothing, "route", "reason") === nothing
    @test_throws ArgumentError GLLVModels._refuse_optimizer(:lbfgs, "route", "reason")
    @test_throws ArgumentError GLLVModels._optimizer_method(
        :nelder_mead, Optim.LBFGS(); route = "coverage-go")
    stages = GLLVModels.OptimizerStage[]
    empty!(stages)
    @test_throws ArgumentError GLLVModels._optimizer_record(nothing, stages)
    Y = randn(3, 12)
    fit = fit_gaussian_gllvm(Y; K = 1, optimizer = :bfgs, iterations = 5)
    @test_throws ArgumentError GLLVModels._check_optimizer_refit(fit, "profile_ci")
end

@testset "lv_targets rank guard (#coverage-go)" begin
    X = [1.0 1.0; 2.0 2.0; 3.0 3.0]
    Z = randn(3, 1)
    Λ = reshape([0.5, 0.4, 0.3], 3, 1)
    @test_throws ArgumentError GLLVModels._eta_realized_lv_effects(X, Z, Λ)
end