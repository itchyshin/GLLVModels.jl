# Phylo x NB2 predictor-informed latent fit must not stop early (refs #1129, #1048).
#
# On the data of the NB `B_eta_realized` canary the fit used to stop after 3 iterations with
# stop_reason == :gradient_at_failure_value, 8.5 log-likelihood units below the optimum of the plain
# model (sigma2_phy fixed at 0): one finite-difference neighbour hit a rounding-floor Newton stall in the
# mode search, the objective returned its 1e12 failure value there, and the gradient became about 5e16.
# The rounding-floor rule (a failed line search with Newton decrement below 1e-12 is convergence) removes
# the stall. The fitted log-likelihood must be at least that of the plain model, which is nested in it.
using GLLVModels, Test, Random, LinearAlgebra
using Distributions: NegativeBinomial

@testset "Phylo x NB2 xLV fit does not stop short of the nested plain model (refs #1129)" begin
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 28
    K = 1
    q_lv = 1
    X_lv = reshape(collect(range(-1.0, 1.0; length = n)), n, q_lv)
    beta = log.([6.5, 5.8, 6.2, 5.4, 6.0, 5.6])
    Lambda = reshape([0.18, -0.15, 0.16, -0.13, 0.11, -0.10], p, K)
    alpha_lv = reshape([0.40], q_lv, K)
    r = 1.6
    for seed in (743, 744, 746)      # main stopped after 3, 5 and 9 iterations at these seeds
        Random.seed!(seed)
        Z_truth = X_lv * alpha_lv + 0.05 .* randn(n, K)
        eta = beta .+ Lambda * transpose(Z_truth)
        Y = [rand(NegativeBinomial(r, r / (r + exp(eta[t, s])))) for t in 1:p, s in 1:n]
        fit = GLLVModels._fit_phylo_nb_xlv(
            Y, phy; K = K, X_lv = X_lv,
            beta_init = beta, Lambda_init = Lambda, alpha_lv_init = alpha_lv,
            r_init = r, sigma2_phy_init = 0.30,
            iterations = 600, g_tol = 1e-5,
            newton_maxiter = 120, newton_tol = 1e-10)
        nll_plain = th -> GLLVModels.nb_lv_nll_packed(th, Y, p, K, LogLink(); X_lv = X_lv, q_lv = q_lv,
                                                      maxiter = 120, tol = 1e-10)
        th0 = vcat(beta, vec(alpha_lv), GLLVModels.pack_lambda(Lambda), log(r))
        res = GLLVModels.Optim.optimize(nll_plain, th0, GLLVModels.Optim.LBFGS(),
            GLLVModels.Optim.Options(g_tol = 1e-6, iterations = 3000); autodiff = :finite)
        ll_plain = -GLLVModels.Optim.minimum(res)
        @test isfinite(fit.loglik)
        # on main the gap was 8.53, 5.59 and 1.76; with the rule it is below 1e-3
        @test fit.loglik >= ll_plain - 1e-2
    end
end
