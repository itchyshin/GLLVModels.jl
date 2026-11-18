# #1048: a line search that stalls at the rounding floor while the Newton decrement
# g'delta is below 1e-12 is convergence. `tol = 0` asks for a sub-rounding step, so
# the search can only end by that stall (or by maxiter, which reports a failure):
# before the rule it returned (nothing, nothing, nothing) at this recorded seed.
using GLLVModels, Test, Random, LinearAlgebra

@testset "Phylo x NB xLV mode search: rounding-floor stall is convergence (#1048)" begin
    Random.seed!(1)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 28
    K = 1
    beta = 0.12 .* randn(p)
    Lambda = reshape(range(0.08, 0.34; length = p), p, K)
    alpha_lv = reshape([0.32], 1, K)
    X_lv = reshape(collect(range(-1.1, 1.1; length = n)), n, 1)
    r = 3.0
    Y = Float64.(rand(0:6, p, n))
    mean_eta = GLLVModels._lv_mean_eta(Lambda, X_lv, alpha_lv)
    Q_cond, leaf_pos = GLLVModels._phylo_qcond(phy)
    Q = Matrix(Q_cond ./ 0.5)

    eps0, u0, chol0 = GLLVModels._phylo_nb_xlv_mode(Y, beta, Lambda, alpha_lv, r,
        mean_eta, Q, leaf_pos; maxiter = 200, tol = 1e-10)
    @test eps0 !== nothing
    eps, u, cholH = GLLVModels._phylo_nb_xlv_mode(Y, beta, Lambda, alpha_lv, r,
        mean_eta, Q, leaf_pos; maxiter = 200, tol = 0.0)
    @test eps !== nothing
    @test cholH !== nothing
    @test isapprox(eps, eps0; atol = 1e-6)
    @test isapprox(u, u0; atol = 1e-6)
end
