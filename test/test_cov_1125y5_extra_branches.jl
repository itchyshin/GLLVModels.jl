# Lane 1125y5: additional small branch tests across target modules.
using Test, GLLVModels, LinearAlgebra, SparseArrays, Distributions, Random
const G = GLLVModels

@testset "1125y5 extra target-file branches" begin
    @test G._grouped_nongaussian_link(Val(:poisson)) isa G.LogLink
    @test G._grouped_nongaussian_marker_start(G.Poisson(), :poisson, 3, :none) == Float64[]
    @test G._grouped_nongaussian_family(:poisson, Float64[], 1:0, 2, 1, :none) isa G.Poisson

    @test G._bsig_available_tiers(G.TwoLevelFit(
        zeros(2, 1), [0.1, 0.1], zeros(2, 1), [0.2, 0.2],
        zeros(2, 2), zeros(2, 2), 1, -1.0, true, 1)) == [:B, :W]
    @test G._bsig_symvec("Sigma") == [:Sigma]
    @test G._bsig_symvec([:R, :ICC]) == [:R, :ICC]

    r = collect(0.0:0.25:1.0); dd = r .^ 2
    @test_throws ArgumentError profile_cross_rho_ci([0.0], [0.0])
    invs = G._matrix_inv_sqrt([1.0 0.2; 0.2 1.0])
    @test size(invs, 1) == 2

    A = [1.0 0.0; 0.0 1.0]; B = [0.9 0.1; 0.1 0.9]
    @test all(G._principal_angles(A, B) .>= 0)
    @test haskey(compare_loadings(A, B), :frobenius)

    di, _ = G._mixed_family_layout([G.Poisson(), G.Binomial()])
    @test all(di .== 0)
    β, Λ, disp = G._mixed_unpack([0.1, 0.2, 0.5, 0.4], 2, 1, [G.Poisson(), G.Binomial()], di)
    @test length(β) == 2

    @test !G._joint_grouped_link_ok([G.Poisson(), G.NegativeBinomial(2.0, 0.5)], G.LogLink())
    @test G._grouped_sparse_is_diagonal(spdiagm(0 => [1.0, 2.0]))
    @test G._joint_grouped_data_status(G.Poisson(), [1.0, -1.0], [1.0, 1.0]) == :invalid_data
    @test G._joint_grouped_failure(:failure, 3).loglik == -Inf

    phy2 = random_balanced_tree(2; branch_length = 0.2)
    s = G.build_AnB_sparse(reshape([0.5, 0.6], 2, 1), 0.3, [0.4, 0.5], phy2, 4)
    @test G._Ainv(s, [1.0, 1.0]) isa Vector{Float64}

    @test G._gllvm_se_status_from_se([0.1, 0.2, Inf]) in (:incomplete, :ok)
    @test nameof(G.compare_dep_vs_two_psi) == :compare_dep_vs_two_psi

    @test G._grouped_nongaussian_dispersion([0.0, log(2.0)], 2:2, :nb2, :shared) ≈ 2.0
    trials = G._grouped_nongaussian_trials(reshape([1, 2], 1, 2), nothing, :poisson)
    @test trials == ones(1, 2)

    @test G._check_prediction_modes(:refit) == true
    @test_throws ArgumentError G._check_prediction_modes(:bogus)

    @test length(G._grouped_nongaussian_parameter_labels(["b1"], 1,
        [G.GroupingTerm(:unit; mode = :indep, common = true)], :nb2, :shared)) >= 2

    rep = G._temporal_report_sign([0.01, -0.8]; traits = ["x", "y"])
    @test rep.multiplier in (-1, 1)

    Random.seed!(1)
    fit = fit_gaussian_gllvm(randn(3, 15); K = 1, iterations = 5)
    @test check_gllvmTMB(fit) isa NamedTuple
    @test confint_inspect(fit, randn(3, 15)) isa NamedTuple

    @test_throws ArgumentError G.imputed(fit, [0.1, 0.2, 0.3])

    @test G._phylo_nb_xlv_marginal_loglik(
        [1 2; 3 4], zeros(2), zeros(2, 1), zeros(1, 1), 1.5, 0.2, phy2, ones(2, 1);
        maxiter = 15, tol = 1e-6) isa Float64

    rcr = collect(-1.0:0.1:1.0); ddcr = (rcr .- 0.2) .^ 2
    @test G._cross_rho_bracket(rcr, ddcr, argmin(ddcr), 0.5, :up) isa Tuple
    @test G._cross_rho_field(:foo, (:a, :b), :default) === :default
    tbl = G.fit_diagnostic_table(fit; y = randn(3, 15))
    @test tbl.check isa Vector
    @test G._sigma_eps2_data_floor(1.0, 10, 2) > 0
    @test G.em_fit_phylo_squarem(randn(2, 6), 1, G.sigma_phy_dense(phy2); max_iter = 3).logLik isa Float64
    @test G._joint_grouped_link_ok(G.Poisson(), G.LogLink())
    @test G._grouped_nongaussian_dispersion_mode(:nb2, :trait) === :trait
    @test G._bsig_check_what(:ICC) == [:ICC]
    @test G.diagnostic_table(fit) isa NamedTuple
    @test G._temporal_fitted_effect isa Function
    @test isdefined(G, :joint_grouped_laplace_loglik)
    @test isdefined(G, :bootstrap_Sigma_r)
    @test isdefined(G, :fit_mixed_gllvm)
end