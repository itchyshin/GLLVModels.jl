using GLLVModels, Test, Random, LinearAlgebra

@testset "Phylo x Ordinal xLV mode search flags a maxiter exit (#747)" begin
    Random.seed!(747)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 8
    K = 1
    Lambda = reshape(range(0.07, 0.25; length = p), p, K)
    alpha_lv = reshape([0.28], 1, K)
    tau = [-1.1, 0.05, 1.25]
    X_lv = reshape(collect(range(-1.0, 1.0; length = n)), n, 1)
    Y = fill(2, p, n)
    mean_eta = GLLVModels._lv_mean_eta(Lambda, X_lv, alpha_lv)
    Q_cond, leaf_pos = GLLVModels._phylo_qcond(phy)
    Q = Matrix(Q_cond ./ 0.5)

    # One Newton step cannot reach the mode: a failed search, not a non-mode.
    @test GLLVModels._phylo_ordinal_xlv_mode(Y, Lambda, alpha_lv, tau,
        mean_eta, Q, leaf_pos, LogitLink(); maxiter = 1) ==
        (nothing, nothing, nothing)
    @test GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y, Lambda, alpha_lv, tau, 0.45, phy, X_lv; maxiter = 1) == -Inf

    # A converged search still returns the mode.
    eps, u, cholH = GLLVModels._phylo_ordinal_xlv_mode(Y, Lambda, alpha_lv, tau,
        mean_eta, Q, leaf_pos, LogitLink(); maxiter = 120, tol = 1e-10)
    @test eps !== nothing
    @test u !== nothing
    @test cholH !== nothing
    @test isfinite(GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y, Lambda, alpha_lv, tau, 0.45, phy, X_lv; maxiter = 120))
end
