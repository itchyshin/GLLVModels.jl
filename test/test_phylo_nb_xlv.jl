using GLLVModels, Test, Random, LinearAlgebra
using Distributions: NegativeBinomial

function _dense_leaf_phylo_nb_xlv_loglik(Y, beta, Lambda, alpha_lv, r,
        sigma2_phy, phy, X_lv; maxiter = 120, tol = 1e-10)
    p, n = size(Y)
    K = size(Lambda, 2)
    keep = filter(i -> i != phy.root_index, 1:phy.n_total)
    Qc = Matrix(phy.Q_topology[keep, keep])
    leaf_pos = [(lp = phy.leaf_indices[t]; phy.root_index < lp ? lp - 1 : lp) for t in 1:p]
    Sigma_a = sigma2_phy .* (inv(Qc)[leaf_pos, leaf_pos])
    Pa = inv(Sigma_a)
    mean_eta = GLLVModels._lv_mean_eta(Lambda, X_lv, alpha_lv)
    fam = NegativeBinomial(r, 0.5)

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

@testset "Phylo x NB2 predictor-informed LV S1 likelihood" begin
    Random.seed!(742)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 7
    K = 1
    q_lv = 1
    beta = log.([4.2, 3.7, 4.5, 3.4, 4.0, 3.8])
    Lambda = reshape(range(0.08, 0.32; length = p), p, K)
    alpha_lv = reshape([0.30], q_lv, K)
    X_lv = reshape(collect(range(-1.1, 1.1; length = n)), n, q_lv)
    r = 1.7
    eta = beta .+ Lambda * transpose(X_lv * alpha_lv)
    Y = [rand(NegativeBinomial(r, r / (r + exp(eta[t, s])))) for t in 1:p, s in 1:n]

    ll_joint0 = GLLVModels._phylo_nb_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, r, 1e-8, phy, X_lv;
        maxiter = 120, tol = 1e-10)
    theta_plain = vcat(beta, vec(alpha_lv), GLLVModels.pack_lambda(Lambda), log(r))
    ll_plain = -GLLVModels.nb_lv_nll_packed(
        theta_plain, Y, p, K, LogLink(); X_lv = X_lv, q_lv = q_lv,
        maxiter = 120, tol = 1e-10)
    @test isapprox(ll_joint0, ll_plain; atol = 2e-3)

    sigma2 = 0.40
    Lambda0 = zeros(p, K)
    ll_phylo_only = GLLVModels._phylo_nb_xlv_marginal_loglik(
        Y, beta, Lambda0, alpha_lv, r, sigma2, phy, X_lv;
        maxiter = 120, tol = 1e-10)
    ll_phylo_glm = phylo_glm_marginal_loglik(
        NegativeBinomial(r, 0.5), Y, ones(Int, p, n), beta, sigma2, phy;
        link = LogLink(), maxiter = 120, tol = 1e-10)
    @test isapprox(ll_phylo_only, ll_phylo_glm; atol = 1e-7)

    ll_sparse_aug = GLLVModels._phylo_nb_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, r, sigma2, phy, X_lv;
        maxiter = 120, tol = 1e-10)
    ll_dense_leaf = _dense_leaf_phylo_nb_xlv_loglik(
        Y, beta, Lambda, alpha_lv, r, sigma2, phy, X_lv)
    @test isapprox(ll_sparse_aug, ll_dense_leaf; atol = 1e-6)
    @test isfinite(ll_sparse_aug)

    @test GLLVModels._phylo_nb_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, r, 0.0, phy, X_lv) == -Inf
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_marginal_loglik(
        Y[1:5, :], beta, Lambda, alpha_lv, r, sigma2, phy, X_lv)
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, 0.0, sigma2, phy, X_lv)
    Y_frac = Float64.(Y)
    Y_frac[1, 1] += 0.5
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_marginal_loglik(
        Y_frac, beta, Lambda, alpha_lv, r, sigma2, phy, X_lv)
    Y_neg = copy(Y)
    Y_neg[1, 1] = -1
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_marginal_loglik(
        Y_neg, beta, Lambda, alpha_lv, r, sigma2, phy, X_lv)
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, r, sigma2, phy, X_lv[1:(end - 1), :])
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_marginal_loglik(
        Y, beta, Lambda, ones(2, K), r, sigma2, phy, X_lv)
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_marginal_loglik(
        Y, beta, Lambda, alpha_lv, r, sigma2, phy, X_lv; link = IdentityLink())
end

@testset "Phylo x NB2 B_eta_realized selected-entry canary" begin
    Random.seed!(743)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 28
    K = 1
    q_lv = 1
    X_lv = reshape(collect(range(-1.0, 1.0; length = n)), n, q_lv)
    beta = log.([6.5, 5.8, 6.2, 5.4, 6.0, 5.6])
    Lambda = reshape([0.18, -0.15, 0.16, -0.13, 0.11, -0.10], p, K)
    alpha_lv = reshape([0.40], q_lv, K)
    sigma2 = 0.30
    r = 1.6
    epsilon = 0.05 .* randn(n, K)
    Z_truth = X_lv * alpha_lv + epsilon
    eta = beta .+ Lambda * transpose(Z_truth)
    Y = [rand(NegativeBinomial(r, r / (r + exp(eta[t, s])))) for t in 1:p, s in 1:n]

    fit = GLLVModels._fit_phylo_nb_xlv(
        Y, phy; K = K, X_lv = X_lv,
        beta_init = beta, Lambda_init = Lambda, alpha_lv_init = alpha_lv,
        r_init = r, sigma2_phy_init = sigma2,
        iterations = 600, g_tol = 1e-5,
        newton_maxiter = 120, newton_tol = 1e-10)
    @test fit.converged
    @test isfinite(fit.loglik)
    @test fit.sigma2_phy > 0
    @test 0.2 < fit.r < 50.0

    eta_target = vec(GLLVModels._eta_realized_lv_effects(X_lv, Z_truth, Lambda))
    prof = GLLVModels._phylo_nb_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target;
        level = 0.95, profile_iterations = 700,
        newton_maxiter = 120, newton_tol = 1e-10)
    @test prof.method == :profile_eta_realized
    @test prof.term == ["B_eta_realized[1,1]"]
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
    @test prof.r_ok
    @test prof.pd_hessian

    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, Int[], eta_target)
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1, 1], eta_target)
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [p + 1], eta_target)
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target[1:(end - 1)])
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target; profile_iterations = 0)
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target; endpoint_step = -0.1)
    @test_throws ArgumentError GLLVModels._phylo_nb_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target; r_bounds = (1.0, 0.5))
end
