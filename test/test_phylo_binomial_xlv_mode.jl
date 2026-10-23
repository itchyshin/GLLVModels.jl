using GLLVModels, Test, Random, LinearAlgebra

@testset "Phylo x Binomial xLV mode search flags a maxiter exit (#747)" begin
    Random.seed!(747)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 7
    K = 1
    beta = 0.12 .* randn(p)
    Lambda = reshape(range(0.08, 0.34; length = p), p, K)
    alpha_lv = reshape([0.32], 1, K)
    X_lv = reshape(collect(range(-1.1, 1.1; length = n)), n, 1)
    N = fill(14.0, p, n)
    Y = Float64.(rand(0:14, p, n))
    mean_eta = GLLVModels._lv_mean_eta(Lambda, X_lv, alpha_lv)
    Q_cond, leaf_pos = GLLVModels._phylo_qcond(phy)
    Q = Matrix(Q_cond ./ 0.5)

    # One Newton step cannot reach the mode: a failed search, not a non-mode.
    @test GLLVModels._phylo_binomial_xlv_mode(Y, N, beta, Lambda, mean_eta, Q,
        leaf_pos; maxiter = 1) == (nothing, nothing, nothing)

    # A converged search still returns the mode.
    eps, u, cholH = GLLVModels._phylo_binomial_xlv_mode(Y, N, beta, Lambda,
        mean_eta, Q, leaf_pos; maxiter = 120, tol = 1e-10)
    @test eps !== nothing
    @test u !== nothing
    @test cholH !== nothing
end
