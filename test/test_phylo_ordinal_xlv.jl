using GLLVModels, Test, Random, LinearAlgebra
using Distributions: Categorical

function _dense_leaf_phylo_ordinal_xlv_loglik(Y, Lambda, alpha_lv,
        tau, sigma2_phy, phy, X_lv; maxiter = 120, tol = 1e-10)
    p, n = size(Y)
    K = size(Lambda, 2)
    keep = filter(i -> i != phy.root_index, 1:phy.n_total)
    Qc = Matrix(phy.Q_topology[keep, keep])
    leaf_pos = [(lp = phy.leaf_indices[t]; phy.root_index < lp ? lp - 1 : lp) for t in 1:p]
    Sigma_a = sigma2_phy .* (inv(Qc)[leaf_pos, leaf_pos])
    Pa = inv(Sigma_a)
    mean_eta = GLLVModels._lv_mean_eta(Lambda, X_lv, alpha_lv)

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
            eta_ts = mean_eta[t, s] + a[t]
            for k in 1:K
                eta_ts += Lambda[t, k] * eps[s, k]
            end
            score_ts, weight_ts = GLLVModels._ord_score_weight(
                Int(Y[t, s]), GLLVModels._clamp_eta(eta_ts), tau, LogitLink())
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
        q0 = _dense_leaf_phylo_ordinal_xlv_logpost(
            Y, Lambda, alpha_lv, tau, mean_eta, Pa, eps, a)
        accepted = false
        step = 1.0
        for _half in 1:30
            eps_trial = copy(eps)
            a_trial = copy(a)
            @inbounds for s in 1:n, k in 1:K
                eps_trial[s, k] += step * delta[(s - 1) * K + k]
            end
            @inbounds for t in 1:p
                a_trial[t] += step * delta[n_z + t]
            end
            q1 = _dense_leaf_phylo_ordinal_xlv_logpost(
                Y, Lambda, alpha_lv, tau, mean_eta, Pa, eps_trial, a_trial)
            if isfinite(q1) && q1 >= q0
                eps = eps_trial
                a = a_trial
                accepted = true
                break
            end
            step *= 0.5
        end
        accepted || break
        step * maximum(abs, delta) < tol && break
    end

    q = _dense_leaf_phylo_ordinal_xlv_logpost(
        Y, Lambda, alpha_lv, tau, mean_eta, Pa, eps, a)
    return q + 0.5 * logdet(cholesky(Symmetric(Pa))) -
           0.5 * logdet(cholesky(Symmetric(H)))
end

function _dense_leaf_phylo_ordinal_xlv_logpost(
        Y, Lambda, alpha_lv, tau, mean_eta, Pa, eps, a)
    p, n = size(Y)
    K = size(Lambda, 2)
    q = -0.5 * sum(abs2, eps) - 0.5 * dot(a, Pa * a)
    @inbounds for s in 1:n, t in 1:p
        eta_ts = mean_eta[t, s] + a[t]
        for k in 1:K
            eta_ts += Lambda[t, k] * eps[s, k]
        end
        q += log(max(GLLVModels._ord_prob(Int(Y[t, s]), GLLVModels._clamp_eta(eta_ts),
                                     tau, LogitLink()), 1e-12))
    end
    return q
end

function _simulate_phylo_ordinal_xlv(Lambda, alpha_lv, tau, X_lv; rng = Random.default_rng(),
        epsilon_scale = 1.0, a_phy = nothing)
    p = size(Lambda, 1)
    n = size(X_lv, 1)
    K = size(Lambda, 2)
    C = length(tau) + 1
    epsilon = epsilon_scale .* randn(rng, n, K)
    Z_truth = X_lv * alpha_lv + epsilon
    eta = Lambda * transpose(Z_truth)
    a_phy === nothing || (eta = eta .+ a_phy)
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n, t in 1:p
        probs = [GLLVModels._ord_prob(c, eta[t, s], tau, LogitLink()) for c in 1:C]
        Y[t, s] = rand(rng, Categorical(probs ./ sum(probs)))
    end
    return Y, Z_truth
end

@testset "Phylo x shared-cutpoint Ordinal predictor-informed LV S1 likelihood" begin
    rng = MersenneTwister(20260745)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 8
    K = 1
    q_lv = 1
    Lambda = reshape(range(0.07, 0.25; length = p), p, K)
    alpha_lv = reshape([0.28], q_lv, K)
    tau = [-1.1, 0.05, 1.25]
    C = length(tau) + 1
    X_lv = reshape(collect(range(-1.0, 1.0; length = n)), n, q_lv)
    Y, _ = _simulate_phylo_ordinal_xlv(Lambda, alpha_lv, tau, X_lv;
                                       rng = rng, epsilon_scale = 0.25)

    ll_joint0 = GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y, Lambda, alpha_lv, tau, 1e-8, phy, X_lv;
        maxiter = 120, tol = 1e-10)
    theta_plain = vcat(vec(alpha_lv), GLLVModels.pack_lambda(Lambda),
                       GLLVModels._phylo_ordinal_tau_to_psi(tau))
    ll_plain = -GLLVModels.ordinal_lv_nll_packed(
        theta_plain, Y, p, K, LogitLink(), C; X_lv = X_lv, q_lv = q_lv,
        maxiter = 120, tol = 1e-10)
    @test isapprox(ll_joint0, ll_plain; atol = 3e-3)

    sigma2 = 0.35
    ll_sparse_aug = GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y, Lambda, alpha_lv, tau, sigma2, phy, X_lv;
        maxiter = 120, tol = 1e-10)
    ll_dense_leaf = _dense_leaf_phylo_ordinal_xlv_loglik(
        Y, Lambda, alpha_lv, tau, sigma2, phy, X_lv)
    @test isapprox(ll_sparse_aug, ll_dense_leaf; atol = 1e-6)
    @test isfinite(ll_sparse_aug)

    Lambda0 = zeros(p, K)
    ll_sparse_phylo_only = GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y, Lambda0, alpha_lv, tau, sigma2, phy, X_lv;
        maxiter = 120, tol = 1e-10)
    ll_dense_phylo_only = _dense_leaf_phylo_ordinal_xlv_loglik(
        Y, Lambda0, alpha_lv, tau, sigma2, phy, X_lv)
    @test isapprox(ll_sparse_phylo_only, ll_dense_phylo_only; atol = 1e-6)

    @test GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y, Lambda, alpha_lv, tau, 0.0, phy, X_lv) == -Inf
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y[1:5, :], Lambda, alpha_lv, tau, sigma2, phy, X_lv)
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y, Lambda, alpha_lv, [0.0, -0.1, 1.0], sigma2, phy, X_lv)
    Y_zero = copy(Y)
    Y_zero[1, 1] = 0
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y_zero, Lambda, alpha_lv, tau, sigma2, phy, X_lv)
    Y_high = copy(Y)
    Y_high[1, 1] = C + 1
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y_high, Lambda, alpha_lv, tau, sigma2, phy, X_lv)
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y, Lambda, alpha_lv, tau, sigma2, phy, X_lv[1:(end - 1), :])
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y, Lambda, ones(2, K), tau, sigma2, phy, X_lv)
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_marginal_loglik(
        Y, Lambda, alpha_lv, tau, sigma2, phy, X_lv; link = IdentityLink())
end

@testset "Phylo x shared-cutpoint Ordinal B_eta_realized selected-entry canary" begin
    rng = MersenneTwister(20260746)
    phy = GLLVModels.augmented_phy("(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 32
    K = 1
    q_lv = 1
    X_lv = reshape(collect(range(-1.0, 1.0; length = n)), n, q_lv)
    Lambda = reshape([0.18, -0.15, 0.16, -0.13, 0.11, -0.10], p, K)
    alpha_lv = reshape([0.40], q_lv, K)
    tau = [-1.1, 0.05, 1.25]
    sigma2 = 0.25
    # Draw the phylogenetic intercept (one value per leaf, shared across sites) from
    # N(0, sigma2 * C_leaf), the same covariance the model uses, so sigma2_phy is
    # not driven to the zero boundary by data that carry no phylogenetic signal (#1048).
    keep = filter(i -> i != phy.root_index, 1:phy.n_total)
    Qc = Matrix(phy.Q_topology[keep, keep])
    leaf_pos = [(lp = phy.leaf_indices[t]; phy.root_index < lp ? lp - 1 : lp) for t in 1:p]
    Sigma_a = Symmetric(sigma2 .* inv(Qc)[leaf_pos, leaf_pos])
    a_phy = cholesky(Sigma_a).L * randn(rng, p)
    Y, Z_truth = _simulate_phylo_ordinal_xlv(Lambda, alpha_lv, tau, X_lv;
                                             rng = rng, epsilon_scale = 0.05, a_phy = a_phy)
    @test all([all(vec(sum(Y .== c; dims = 2)) .> 0) for c in 1:(length(tau) + 1)])

    fit = GLLVModels._fit_phylo_ordinal_xlv(
        Y, phy; K = K, X_lv = X_lv,
        Lambda_init = Lambda, alpha_lv_init = alpha_lv,
        tau_init = tau, sigma2_phy_init = sigma2,
        iterations = 500, g_tol = 1e-5,
        newton_maxiter = 80, newton_tol = 1e-9)
    @test fit.converged
    @test isfinite(fit.loglik)
    @test fit.sigma2_phy > 0
    @test all(diff(fit.tau) .> 0)

    eta_target = vec(GLLVModels._eta_realized_lv_effects(X_lv, Z_truth, Lambda))
    prof = GLLVModels._phylo_ordinal_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [6], eta_target;
        level = 0.95, profile_iterations = 5000,
        profile_max_expand = 5, profile_max_bisect = 6,
        newton_maxiter = 80, newton_tol = 1e-9)
    @test prof.method == :profile_eta_realized
    @test prof.term == ["B_eta_realized[6,1]"]
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
    @test prof.cutpoints_ordered
    @test prof.pd_hessian

    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, Int[], eta_target)
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1, 1], eta_target)
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [p + 1], eta_target)
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target[1:(end - 1)])
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target; profile_iterations = 0)
    @test_throws ArgumentError GLLVModels._phylo_ordinal_xlv_profile_eta_realized(
        fit, Y, phy, X_lv, [1], eta_target; endpoint_step = -0.1)
end
