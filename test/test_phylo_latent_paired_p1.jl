# gllvm-parity-tag: P1
# Replay of the phylo_latent twin receipts against gllvmTMB P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9, 0.7.1): A14 (STRUCT-PHY-TREE-RR,
# STRUCT-PHY-DENSE-RR) and A15 (COV-PHYLO-LATENT-RSZ). Julia-only: the R values
# are recorded in committed, SHA-256-guarded JSON produced by
# tools/phylo_latent/r_reference_p1.R from a private P1 build. The Julia side is
# recomputed live from the stored response. A missing receipt is a counted
# skip, never a pass.
using Test, LinearAlgebra, SHA, GLLVModels

include(joinpath(@__DIR__, "..", "tools", "phylo_latent", "compare_phylo_latent_p1.jl"))

const _PLP1_DIR = joinpath(@__DIR__, "..", "docs", "dev-log", "core070", "phylo-latent-p1")
const _PLP1_SHA = Dict(
    "a14-fixture.json" => "9a4c2e1f87a3fbb519400fff8a99fdfb6d66b249d473b0e25833b93b78e7e13c",
    "struct_phy_tree_rr/r-receipt.json" => "1e98555bbf7ff610778d92e50741f32d6b60b25dc8bda2986315fe20ee5ae239",
    "struct_phy_tree_rr/julia-receipt.json" => "3e59be3c26bb87774591593e6fac367f12b623a5cafaf4f37eac35a407de3b01",
    "struct_phy_dense_rr/r-receipt.json" => "01cffdff7d6841634b8d0d66e1aa9f4af85e487c148c6ea27d5e8f7482bd7b7e",
    "struct_phy_dense_rr/julia-receipt.json" => "a1effb82b3d6120890c76f2b561720175be8113b1095a712bdcc4055ed032d4d",
    "a15-fixture.json" => "f3bace851f60f337a8274859644135a84792b9e9ab35c07f757a7ade336706d4",
    "cov_phylo_latent_rsz/r-receipt.json" => "b84eb446dd51bc08983e2bebdced361344824ae2c895266460b89074fa0cf2fb",
    "cov_phylo_latent_rsz/julia-receipt.json" => "84b6751a23f325af2c66177fbfd8e136b088a2366794be4447687517871a1c71",
)
const _PLP1_DLL_SHA = "cba0f54d5492f0c6d0c5474c19281e3b58d5e2b0709e536684619558aaf55b8d"

_plp1(path) = joinpath(_PLP1_DIR, path)
_plp1_present(paths...) = all(p -> haskey(_PLP1_SHA, p) && isfile(_plp1(p)), paths)

# `stationarity_gap`: A15 only. Both engines stop on the objective's numerical
# floor above the 1e-4 cross-gradient bar: R's own nlminb optimum has max |AD
# gradient| 4.0e-3, and the Julia LBFGS stops at max |FD gradient| 2.0e-4, so
# Julia reports converged = false under its absolute default g_tol = 1e-5 (one
# Newton step from there lowers the objective by 1.8e-10 and the gradient to
# 6e-6). The primary A15 receipt is the cross objective (abs 1e-8) and logLik
# (rtol 1e-6); the gradients carry explicit looser bounds with measured
# headroom, and the non-convergence is asserted as recorded, not hidden.
# The recorded Julia receipt predates the fitter's Newton polish (PR #547 CI
# fix): the live refit now takes that Newton step itself and reports
# converged = true under the unchanged g_tol = 1e-5 (measured max |FD
# gradient| 6.2e-6, logLik within 3e-13 relative of R's, Linux OpenBLAS).
const _PLP1_A15_BOUNDS = (r_gradient_at_julia = 1e-3,   # measured 2.0e-4
                          julia_gradient_at_r = 1e-2,   # measured 4.0e-3
                          julia_gradient_norm = 1e-3,   # measured 2.0e-4
                          beta_abs = 1e-4)              # measured 1.6e-5
function _plp1_case(case, fixture; refit::Bool, stationarity_gap::Bool = false)
    data = "$(fixture)-fixture.json"
    rpath, jpath = "$(case)/r-receipt.json", "$(case)/julia-receipt.json"
    if !_plp1_present(data, rpath, jpath)
        @test_skip "receipt for $(case) absent"
        return
    end
    for p in (data, rpath, jpath)
        @test bytes2hex(sha256(read(_plp1(p)))) == _PLP1_SHA[p]
    end
    fx = pl_read_fixture(_plp1(data))
    rr = pl_read_json(_plp1(rpath))
    jr = pl_read_json(_plp1(jpath))
    n_traits = size(fx.Y, 1)
    K = fx.rank
    n_rr = n_traits * K - K * (K - 1) ÷ 2

    # R provenance: P1 private build, one recorded optimum, R's coordinate names.
    @test rr.source_pin == PL_P1_SOURCE_PIN
    @test rr.package_version == "0.7.1"
    @test rr.dll_sha256 == _PLP1_DLL_SHA
    @test rr.status == "recorded" && jr.status == "recorded"
    @test rr.data_sha256 == fx.data_sha256 == jr.data_sha256
    @test rr.data_file_sha256 == _PLP1_SHA[data]
    @test rr.julia_receipt_sha256 == _PLP1_SHA[jpath]
    @test collect(String, rr.parameter_names) ==
        vcat(fill("b_fix", n_traits), "log_sigma_eps", fill("theta_rr_phy", n_rr))
    @test rr.convergence == 0
    @test rr.route == String(pl_route(case))

    # Structure: the twin's precision against R's shipped bundle.
    phy, species_id = pl_precision(fx, case)
    @test phy.n_aug == rr.n_aug_phy
    @test abs(-phy.log_det - rr.log_det_A_phy_rr) <= PL_P1_TOL.logdet_abs
    Q_r = zeros(rr.n_aug_phy, rr.n_aug_phy)
    for (i, j, x) in zip(rr.Ainv_triplets.i, rr.Ainv_triplets.j, rr.Ainv_triplets.x)
        Q_r[i, j] = x
    end
    labels_r = String.(collect(rr.Ainv_node_labels))
    tips_r = [findfirst(==(t), labels_r) for t in phy.node_labels[phy.species_aug_id]]
    @test all(!isnothing, tips_r)
    # Tip covariance implied by each engine's precision (node order differs
    # only among internal nodes). Norm-relative: two dense inverses of an
    # order-198 precision with short edges carry ~1e-12 roundoff per entry.
    @test isapprox(inv(Q_r)[tips_r, tips_r],
        inv(Matrix(phy.Q))[phy.species_aug_id, phy.species_aug_id]; rtol = 1e-10)
    order = [findfirst(==(t), fx.tips) for t in phy.node_labels[phy.species_aug_id]]
    C_expected = fx.vcv[order, order] + (pl_route(case) === :vcv ? 1e-8 : 0.0) * I
    @test isapprox(inv(Matrix(phy.Q))[phy.species_aug_id, phy.species_aug_id], C_expected;
        rtol = 1e-10)
    @test sort(unique(rr.species_aug_id_zero_based .+ 1)) == sort(unique(phy.species_aug_id[species_id]))

    # Both optima and both cross objectives.
    r_theta = Float64.(collect(rr.theta_hat))
    j_theta = Float64.(collect(jr.theta_julia_order))
    julia_at_r = pl_julia_objective(fx, case, pl_r_to_julia(r_theta, n_traits))
    julia_at_j = pl_julia_objective(fx, case, j_theta)
    @test abs(julia_at_r - rr.objective) <= PL_P1_TOL.cross_abs
    @test abs(rr.cross.r_objective_at_julia_theta - julia_at_j) <= PL_P1_TOL.cross_abs
    @test Float64.(collect(rr.cross.julia_theta_r_order)) == pl_julia_to_r(j_theta, n_traits)
    g = zeros(length(j_theta))
    objective = th -> pl_julia_objective(fx, case, th)
    GLLVModels._pmv_fd_gradient!(g, objective, pl_r_to_julia(r_theta, n_traits))
    if stationarity_gap
        @test rr.cross.r_gradient_max_abs_at_julia_theta <= _PLP1_A15_BOUNDS.r_gradient_at_julia
        @test maximum(abs, g) <= _PLP1_A15_BOUNDS.julia_gradient_at_r
        @test !jr.converged && jr.stopping_reason == "gradient_not_converged" &&
            jr.gradient_norm <= _PLP1_A15_BOUNDS.julia_gradient_norm
        @test maximum(abs.(Float64.(collect(jr.beta)) .- r_theta[1:n_traits])) <=
            _PLP1_A15_BOUNDS.beta_abs
    else
        @test rr.cross.r_gradient_max_abs_at_julia_theta <= PL_P1_TOL.cross_gradient
        @test maximum(abs, g) <= PL_P1_TOL.cross_gradient
        @test jr.converged
    end
    @test isapprox(-jr.objective, rr.loglik; rtol = PL_P1_TOL.loglik_rtol)

    # Estimates at the recorded Julia optimum against R's.
    Sigma_r = _pl_matrix(rr.Sigma_phy)
    @test isapprox(_pl_matrix(jr.Sigma_phy), Sigma_r; rtol = PL_P1_TOL.estimate_rtol)
    stationarity_gap ||
        @test isapprox(Float64.(collect(jr.beta)), r_theta[1:n_traits]; rtol = PL_P1_TOL.estimate_rtol)
    @test isapprox(jr.sigma_eps2, exp(2 * r_theta[n_traits + 1]); rtol = PL_P1_TOL.estimate_rtol)

    if refit
        # Live Julia refit from the stored response (this platform, this build).
        fit = pl_fit(fx, case)
        if stationarity_gap
            # Live refit after the Newton polish: converged under the default
            # g_tol = 1e-5; the recorded receipt above keeps its stall.
            @test fit.converged && fit.stopping_reason === :converged
            @test fit.gradient_norm <= 1e-5
            @test maximum(abs.(fit.beta .- r_theta[1:n_traits])) <= _PLP1_A15_BOUNDS.beta_abs
        else
            @test fit.converged
            @test isapprox(fit.beta, r_theta[1:n_traits]; rtol = PL_P1_TOL.estimate_rtol)
        end
        @test isapprox(fit.loglik, rr.loglik; rtol = PL_P1_TOL.loglik_rtol)
        @test isapprox(fit.loading * fit.loading', Sigma_r; rtol = PL_P1_TOL.estimate_rtol)
        @test isapprox(fit.residual_variance[1], exp(2 * r_theta[n_traits + 1]);
            rtol = PL_P1_TOL.estimate_rtol)
    else
        @test_skip "live A15 refit (set GLLVM_PHYLO_LATENT_A15_REFIT=1)"
    end
end

@testset "phylo_latent P1 paired receipts" begin
    @testset "A14 STRUCT-PHY-TREE-RR" begin
        _plp1_case("struct_phy_tree_rr", "a14"; refit = true)
    end
    @testset "A14 STRUCT-PHY-DENSE-RR" begin
        _plp1_case("struct_phy_dense_rr", "a14"; refit = true)
    end
    @testset "A14 tree and dense routes agree on each side (test-phylo-hadfield.R:99-107)" begin
        if _plp1_present("struct_phy_tree_rr/r-receipt.json", "struct_phy_dense_rr/r-receipt.json",
                "struct_phy_tree_rr/julia-receipt.json", "struct_phy_dense_rr/julia-receipt.json")
            for side in ("r", "julia")
                t = pl_read_json(_plp1("struct_phy_tree_rr/$(side)-receipt.json"))
                d = pl_read_json(_plp1("struct_phy_dense_rr/$(side)-receipt.json"))
                lt = side == "r" ? t.loglik : -t.objective
                ld = side == "r" ? d.loglik : -d.objective
                @test isapprox(lt, ld; rtol = 1e-4)
                @test isapprox(_pl_matrix(t.Sigma_phy), _pl_matrix(d.Sigma_phy); rtol = 1e-4)
            end
        else
            @test_skip "A14 receipts absent"
        end
    end
    @testset "A14 in-keyword Ainv = inv(C) is R's dense route (R/brms-sugar.R:3311-3319)" begin
        if _plp1_present("a14-fixture.json", "struct_phy_dense_rr/r-receipt.json")
            fx = pl_read_fixture(_plp1("a14-fixture.json"))
            rr = pl_read_json(_plp1("struct_phy_dense_rr/r-receipt.json"))
            fit = fit_phylo_latent_gllvm(fx.Y, fx.species; d = fx.rank, Ainv = inv(fx.vcv),
                tip_labels = fx.tips)
            @test fit.converged
            @test fit.phy.n_aug == rr.n_aug_phy == 8
            @test abs(-fit.phy.log_det - rr.log_det_A_phy_rr) <= PL_P1_TOL.logdet_abs
            @test isapprox(fit.loglik, rr.loglik; rtol = PL_P1_TOL.loglik_rtol)
            r_theta = Float64.(collect(rr.theta_hat))
            n_traits = size(fx.Y, 1)
            @test abs(pl_julia_objective(fx, "struct_phy_dense_rr",
                pl_r_to_julia(r_theta, n_traits)) - rr.objective) <= PL_P1_TOL.cross_abs
            @test isapprox(fit.loading * fit.loading', _pl_matrix(rr.Sigma_phy);
                rtol = PL_P1_TOL.estimate_rtol)
        else
            @test_skip "A14 dense receipt absent"
        end
    end
    @testset "A15 COV-PHYLO-LATENT-RSZ" begin
        _plp1_case("cov_phylo_latent_rsz", "a15"; refit = true, stationarity_gap = true)
    end
end
