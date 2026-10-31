using GLLVModels, Test, Random, LinearAlgebra
using Distributions: Gamma

function _dense_leaf_phylo_gamma_xlv_loglik(Y, beta, Lambda, alpha_lv,
        alpha_shape, sigma2_phy, phy, X_lv; maxiter = 120, tol = 1e-10)
    p, n = size(Y)
    K = size(Lambda, 2)
    keep = filter(i -> i != phy.root_index, 1:phy.n_total)
    Qc = Matrix(phy.Q_topology[keep, keep])
    leaf_pos = [(lp = phy.leaf_indices[t]; phy.root_index < lp ? lp - 1 : lp) for t in 1:p]
    Sigma_a = sigma2_phy .* (inv(Qc)[leaf_pos, leaf_pos])
    Pa = inv(Sigma_a)
    mean_eta = GLLVModels._lv_mean_eta(Lambda, X_lv, alpha_lv)
    fam = Gamma(alpha_shape, 1.0)

    eps = zeros(n, K)
    a = zeros(p)
    n_z = n * K
    local H
    for _ in 1:maxiter
        grad = zeros(n_z + p)
        H = zeros(n_z + p, n_z + p)
        @inbounds for s in 1:n
            for k in 1:K
                idx = (s - 1) * K + k
                grad[idx] -= eps[s, k]
                H[idx, idx] += 1.0
            end
        end
        @views begin
            grad[(n_z + 1):end] .-= Pa * a
            H[(n_z + 1):end, (n_z + 1):end] .+= Pa
        end

        @inbounds for s in 1:n, t in 1:p
            eta_ts = beta[t] + mean_eta[t, s] + a[t]
            for k in 1:K
                eta_ts += Lambda[t, k] * eps[s, k]
            end
            eta_c = GLLVModels._clamp_eta(eta_ts)
            mu_ts = exp(eta_c)
            me_ts = GLLVModels.mu_eta(LogLink(), eta_c)
            score_ts = GLLVModels._glm_score(fam, mu_ts, 1, me_ts, Y[t, s])
            weight_ts = GLLVModels._glm_weight(fam, mu_ts, 1, me_ts)
            aidx = n_z + t
            grad[aidx] += score_ts
            H[aidx, aidx] += weight_ts
            for k in 1:K
                zidx_k = (s - 1) * K + k
                ltk = Lambda[t, k]
                grad[zidx_k] += ltk * score_ts
                cross = weight_ts * ltk
                H[zidx_k, aidx] += cross
                H[aidx, zidx_k] += cross
                for j in 1:K
                    zidx_j = (s - 1) * K + j
                    H[zidx_k, zidx_j] += weight_ts * ltk * Lambda[t, j]
                end
            end
        end
        delta = cholesky(Symmetric(H)) \ grad
        @inbounds for s in 1:n, k in 1:K
            eps[s, k] += delta[(s - 1) * K + k]
        end
        @inbounds for t in 1:p
            a[t] += delta[n_z + t]
        end
        maximum(abs, delta) < tol && break
    end

    # Newton above uses expected (Fisher) weights, as the sparse evaluator's mode
    # search does. The Laplace log-det must use the OBSERVED information at the
    # mode, matching the evaluator's `_default_hessian` (= :observed for this
    # non-canonical log link).
    H = zeros(n_z + p, n_z + p)
    @inbounds for s in 1:n, k in 1:K
        H[(s - 1) * K + k, (s - 1) * K + k] += 1.0
    end
    @views H[(n_z + 1):end, (n_z + 1):end] .+= Pa
    @inbounds for s in 1:n, t in 1:p
        eta_ts = beta[t] + mean_eta[t, s] + a[t]
        for k in 1:K
            eta_ts += Lambda[t, k] * eps[s, k]
        end
        eta_c = GLLVModels._clamp_eta(eta_ts)
        mu_ts = exp(eta_c)
        me_ts = GLLVModels.mu_eta(LogLink(), eta_c)
        weight_ts = GLLVModels._glm_obs_weight(fam, mu_ts, 1, me_ts, Y[t, s],
                                               LogLink(), eta_c)
        aidx = n_z + t
        H[aidx, aidx] += weight_ts
        for k in 1:K
            zidx_k = (s - 1) * K + k
            ltk = Lambda[t, k]
            H[zidx_k, aidx] += weight_ts * ltk
            H[aidx, zidx_k] += weight_ts * ltk
            for j in 1:K
                H[zidx_k, (s - 1) * K + j] += weight_ts * ltk * Lambda[t, j]
            end
        end
    end

    q = -0.5 * sum(abs2, eps) - 0.5 * dot(a, Pa * a)
    @inbounds for s in 1:n, t in 1:p
        eta_ts = beta[t] + mean_eta[t, s] + a[t]
        for k in 1:K
            eta_ts += Lambda[t, k] * eps[s, k]
        end
        mu_ts = exp(GLLVModels._clamp_eta(eta_ts))
        q += GLLVModels._glm_logpdf(fam, mu_ts, 1, Y[t, s])
    end
    return q + 0.5 * logdet(cholesky(Symmetric(Pa))) -
           0.5 * logdet(cholesky(Symmetric(H)))
end

@testset "Phylo x Gamma predictor-informed LV S1 likelihood" begin
    Random.seed!(752)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 7
    K = 1
    q_lv = 1
    beta = log.([2.8, 2.5, 3.0, 2.3, 2.7, 2.4])
    Lambda = reshape(range(0.07, 0.28; length = p), p, K)
    alpha_lv = reshape([0.32], q_lv, K)
    X_lv = reshape(collect(range(-1.1, 1.1; length = n)), n, q_lv)
    alpha_shape = 2.4
    eta = beta .+ Lambda * transpose(X_lv * alpha_lv)
    Y = [rand(Gamma(alpha_shape, exp(eta[t, s]) / alpha_shape)) for t in 1:p, s in 1:n]

    ll_joint0 = GLLVModels._phylo_gamma_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, alpha_shape, 1e-8, phy, X_lv;
        maxiter = 120, tol = 1e-10)
    theta_plain = vcat(beta, vec(alpha_lv), GLLVModels.pack_lambda(Lambda), log(alpha_shape))
    ll_plain = -GLLVModels.gamma_lv_nll_packed(
        theta_plain, Y, p, K, LogLink(); X_lv = X_lv, q_lv = q_lv,
        maxiter = 120, tol = 1e-10)
    @test isapprox(ll_joint0, ll_plain; atol = 2e-3)

    sigma2 = 0.40
    Lambda0 = zeros(p, K)
    ll_phylo_only = GLLVModels._phylo_gamma_xlv_marginal_loglik(
        Y, beta, Lambda0, alpha_lv, alpha_shape, sigma2, phy, X_lv;
        maxiter = 120, tol = 1e-10)
    ll_phylo_glm = phylo_glm_marginal_loglik(
        Gamma(alpha_shape, 1.0), Y, ones(Int, p, n), beta, sigma2, phy;
        link = LogLink(), maxiter = 120, tol = 1e-10)
    @test isapprox(ll_phylo_only, ll_phylo_glm; atol = 1e-7)

    ll_sparse_aug = GLLVModels._phylo_gamma_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, alpha_shape, sigma2, phy, X_lv;
        maxiter = 120, tol = 1e-10)
    ll_dense_leaf = _dense_leaf_phylo_gamma_xlv_loglik(
        Y, beta, Lambda, alpha_lv, alpha_shape, sigma2, phy, X_lv)
    @test isapprox(ll_sparse_aug, ll_dense_leaf; atol = 1e-6)
    @test isfinite(ll_sparse_aug)

    @test GLLVModels._phylo_gamma_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, alpha_shape, 0.0, phy, X_lv) == -Inf
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_marginal_loglik(
        Y[1:5, :], beta, Lambda, alpha_lv, alpha_shape, sigma2, phy, X_lv)
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, 0.0, sigma2, phy, X_lv)
    Y_zero = copy(Y)
    Y_zero[1, 1] = 0.0
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_marginal_loglik(
        Y_zero, beta, Lambda, alpha_lv, alpha_shape, sigma2, phy, X_lv)
    Y_nan = copy(Y)
    Y_nan[1, 1] = NaN
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_marginal_loglik(
        Y_nan, beta, Lambda, alpha_lv, alpha_shape, sigma2, phy, X_lv)
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, alpha_shape, sigma2, phy, X_lv[1:(end - 1), :])
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_marginal_loglik(
        Y, beta, Lambda, ones(2, K), alpha_shape, sigma2, phy, X_lv)
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, alpha_shape, sigma2, phy, X_lv; link = IdentityLink())
end

@testset "Phylo x Gamma B_eta_realized selected-entry canary" begin
    Random.seed!(753)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 28
    K = 1
    q_lv = 1
    X_lv = reshape(collect(range(-1.0, 1.0; length = n)), n, q_lv)
    beta = log.([3.0, 2.5, 2.8, 2.3, 2.7, 2.4])
    Lambda = reshape([0.18, -0.15, 0.16, -0.13, 0.11, -0.10], p, K)
    alpha_lv = reshape([0.40], q_lv, K)
    sigma2 = 0.30
    alpha_shape = 2.5
    epsilon = 0.05 .* randn(n, K)
    Z_truth = X_lv * alpha_lv + epsilon
    eta = beta .+ Lambda * transpose(Z_truth)
    Y = [rand(Gamma(alpha_shape, exp(eta[t, s]) / alpha_shape)) for t in 1:p, s in 1:n]

    fit = GLLVModels._fit_phylo_gamma_xlv(
        Y, phy; K = K, X_lv = X_lv,
        beta_init = beta, Lambda_init = Lambda, alpha_lv_init = alpha_lv,
        alpha_shape_init = alpha_shape, sigma2_phy_init = sigma2,
        iterations = 800, g_tol = 1e-5,
        newton_maxiter = 120, newton_tol = 1e-10)
    @test fit.converged
    @test isfinite(fit.loglik)
    @test fit.sigma2_phy > 0
    @test 0.5 < fit.alpha_shape < 20.0

    eta_target = vec(GLLVModels._eta_realized_lv_effects(X_lv, Z_truth, Lambda))
    prof = GLLVModels._phylo_gamma_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [5], eta_target;
        level = 0.95, profile_iterations = 700,
        newton_maxiter = 120, newton_tol = 1e-10)
    @test prof.method == :profile_eta_realized
    @test prof.term == ["B_eta_realized[5,1]"]
    @test all(isnan, prof.se)
    @test prof.endpoint_status == [:profile]
    @test all(isfinite, prof.lower)
    @test all(isfinite, prof.upper)
    @test prof.lower[1] < prof.estimate[1] < prof.upper[1]
    @test prof.lower[1] <= prof.target[1] <= prof.upper[1]
    @test isfinite(prof.lr_deviance[1])
    @test prof.lr_deviance[1] <= prof.lr_cutoff[1]
    @test prof.constrained_error[1] < 1e-3
    @test prof.covered == [true]
    @test prof.alpha_shape_ok
    @test prof.pd_hessian

    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, Int[], eta_target)
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1, 1], eta_target)
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [p + 1], eta_target)
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target[1:(end - 1)])
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target; profile_iterations = 0)
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target; endpoint_step = -0.1)
    @test_throws ArgumentError GLLVModels._phylo_gamma_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target; alpha_shape_bounds = (1.0, 0.5))
end
