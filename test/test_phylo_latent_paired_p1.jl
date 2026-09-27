# gllvm-parity-tag: P1
# Replay of the phylo_latent twin receipts against gllvmTMB P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9, 0.7.1): A14 (STRUCT-PHY-TREE-RR,
# STRUCT-PHY-DENSE-RR) and A15 (COV-PHYLO-LATENT-RSZ). Julia-only: the R values
# are recorded in committed, SHA-256-guarded JSON produced by
# tools/phylo_latent/r_reference_p1.R from a private P1 build. The Julia side is
# recomputed live from the stored response. A missing receipt is a counted
# skip, never a pass.
using Test, LinearAlgebra, JSON3, SHA, GLLVModels

include(joinpath(@__DIR__, "..", "tools", "phylo_latent", "compare_phylo_latent_p1.jl"))

const _PLP1_DIR = joinpath(@__DIR__, "..", "docs", "dev-log", "core070", "phylo-latent-p1")
const _PLP1_SHA = Dict(
    "a14-fixture.json" => "9a4c2e1f87a3fbb519400fff8a99fdfb6d66b249d473b0e25833b93b78e7e13c",
    "struct_phy_tree_rr/r-receipt.json" => "c550ddbab35b4ba952dbda470f7e879898fb43fd665f5a7c144b10ec2c2c91f4",
    "struct_phy_tree_rr/julia-receipt.json" => "7da419e71edc2268395e84a9687ded92fa466902691374d68e3c53b0f430becd",
    "struct_phy_dense_rr/r-receipt.json" => "b7d728eb30e27c207ba05d4867730052fc75ebe82db1cac15cc3e738d03fc3d5",
    "struct_phy_dense_rr/julia-receipt.json" => "700f14677d5498d832d900ea7939d8ed023c90948f1603891029da18401cd288",
    "a15-fixture.json" => "f3bace851f60f337a8274859644135a84792b9e9ab35c07f757a7ade336706d4",
    "cov_phylo_latent_rsz/r-receipt.json" => "5121fbfeb20421149e9c535ea9220458d531f8ec0cfaa71c5b010a8597a7e40e",
    "cov_phylo_latent_rsz/julia-receipt.json" => "9840f0a17840e438f66c5df97b5325188f15d0e05789191a14b6188f04054bab",
)
const _PLP1_DLL_SHA = "cba0f54d5492f0c6d0c5474c19281e3b58d5e2b0709e536684619558aaf55b8d"

_plp1(path) = joinpath(_PLP1_DIR, path)
_plp1_present(paths...) = all(p -> haskey(_PLP1_SHA, p) && isfile(_plp1(p)), paths)

# `stationarity_gap`: A15 only. Both engines stop on the objective's numerical
# floor above the 1e-4 cross-gradient bar: R's own nlminb optimum has max |AD
# gradient| 4.0e-3, and the Julia LBFGS stops at max |FD gradient| 2.0e-4, so
# Julia reports converged = false under its default g_tol = 1e-5 (one Newton
# step from there lowers the objective by 1.8e-10 and the gradient to 6e-6).
# Recorded as @test_broken, never widened; see the A15 README.
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
    rr = JSON3.read(read(_plp1(rpath), String))
    jr = JSON3.read(read(_plp1(jpath), String))
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
        @test_broken rr.cross.r_gradient_max_abs_at_julia_theta <= PL_P1_TOL.cross_gradient
        @test_broken maximum(abs, g) <= PL_P1_TOL.cross_gradient
        @test_broken jr.converged
    else
        @test rr.cross.r_gradient_max_abs_at_julia_theta <= PL_P1_TOL.cross_gradient
        @test maximum(abs, g) <= PL_P1_TOL.cross_gradient
        @test jr.converged
    end
    @test isapprox(-jr.objective, rr.loglik; rtol = PL_P1_TOL.loglik_rtol)

    # Estimates at the recorded Julia optimum against R's.
    Sigma_r = _pl_matrix(rr.Sigma_phy)
    @test isapprox(_pl_matrix(jr.Sigma_phy), Sigma_r; rtol = PL_P1_TOL.estimate_rtol)
    @test isapprox(Float64.(collect(jr.beta)), r_theta[1:n_traits]; rtol = PL_P1_TOL.estimate_rtol)
    @test isapprox(jr.sigma_eps2, exp(2 * r_theta[n_traits + 1]); rtol = PL_P1_TOL.estimate_rtol)

    if refit
        # Live Julia refit from the stored response (this platform, this build).
        fit = pl_fit(fx, case)
        stationarity_gap ? (@test_broken fit.converged) : (@test fit.converged)
        @test isapprox(fit.loglik, rr.loglik; rtol = PL_P1_TOL.loglik_rtol)
        @test isapprox(fit.loading * fit.loading', Sigma_r; rtol = PL_P1_TOL.estimate_rtol)
        @test isapprox(fit.beta, r_theta[1:n_traits]; rtol = PL_P1_TOL.estimate_rtol)
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
                t = JSON3.read(read(_plp1("struct_phy_tree_rr/$(side)-receipt.json"), String))
                d = JSON3.read(read(_plp1("struct_phy_dense_rr/$(side)-receipt.json"), String))
                lt = side == "r" ? t.loglik : -t.objective
                ld = side == "r" ? d.loglik : -d.objective
                @test isapprox(lt, ld; rtol = 1e-4)
                @test isapprox(_pl_matrix(t.Sigma_phy), _pl_matrix(d.Sigma_phy); rtol = 1e-4)
            end
        else
            @test_skip "A14 receipts absent"
        end
    end
    @testset "A15 COV-PHYLO-LATENT-RSZ" begin
        _plp1_case("cov_phylo_latent_rsz", "a15";
            refit = get(ENV, "GLLVM_PHYLO_LATENT_A15_REFIT", "") == "1",
            stationarity_gap = true)
    end
end
