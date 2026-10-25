# Internal S1 proof for the phylo x shared-cutpoint Ordinal x
# predictor-informed LV target.
#
# Model:
#   epsilon_s ~ N(0, I_K)
#   u         ~ N(0, sigma_phy^2 * Q_cond^{-1})
#   eta[t,s] = Lambda[t,:]' * (X_lv[s,:] * alpha_lv + epsilon_s) +
#              u[leaf_pos[t]]
#   P(Y[t,s] <= c | eta[t,s]) = logistic(tau[c] - eta[t,s])
#
# There is no per-trait intercept: shared ordered cutpoints carry the category
# levels, matching the native Julia shared-cutpoint OrdinalFit route. This is
# deliberately not exported. It is a reduction-tested likelihood surface for a
# structural-source non-Gaussian LV canary, not R grammar support.

function _phylo_ordinal_xlv_validate(Y::AbstractMatrix, Lambda::AbstractMatrix,
        alpha_lv::AbstractMatrix, tau::AbstractVector, phy::AugmentedPhy,
        X_lv::AbstractMatrix, link::Link)
    link isa LogitLink ||
        throw(ArgumentError("phylo Ordinal X_lv S1 is currently logit-link only"))
    all(isfinite, float.(tau)) || throw(ArgumentError("ordinal cutpoints must be finite"))
    length(tau) >= 1 ||
        throw(ArgumentError("ordinal cutpoints must contain at least one threshold"))
    all(diff(float.(tau)) .> 0) ||
        throw(ArgumentError("ordinal cutpoints must be strictly increasing"))
    p, n = size(Y)
    p == phy.n_leaves ||
        throw(ArgumentError("size(Y,1)=$p must equal phy.n_leaves=$(phy.n_leaves)"))
    size(Lambda, 1) == p ||
        throw(ArgumentError("Lambda rows ($(size(Lambda, 1))) must equal size(Y,1)=$p"))
    K = size(Lambda, 2)
    K > 0 || throw(ArgumentError("Lambda must have at least one latent column"))
    size(X_lv, 1) == n ||
        throw(ArgumentError("X_lv rows ($(size(X_lv, 1))) must equal size(Y,2)=$n"))
    size(X_lv, 2) == size(alpha_lv, 1) ||
        throw(ArgumentError(
            "X_lv columns ($(size(X_lv, 2))) must equal alpha_lv rows ($(size(alpha_lv, 1)))"))
    size(alpha_lv, 2) == K ||
        throw(ArgumentError("alpha_lv columns ($(size(alpha_lv, 2))) must equal Lambda columns ($K)"))
    C = length(tau) + 1
    @inbounds for y in Y
        yi = Int(y)
        yi == y && 1 <= yi <= C ||
            throw(ArgumentError("ordinal response category $y is outside 1:$C"))
    end
    return p, n, K, C
end

function _phylo_ordinal_xlv_logpost(Y, Lambda, alpha_lv, tau,
        mean_eta, Q, leaf_pos, eps, u, link::Link)
    p, n = size(Y)
    K = size(Lambda, 2)
    q = -0.5 * sum(abs2, eps) - 0.5 * dot(u, Q * u)
    @inbounds for s in 1:n, t in 1:p
        eta_ts = mean_eta[t, s] + u[leaf_pos[t]]
        for k in 1:K
            eta_ts += Lambda[t, k] * eps[s, k]
        end
        q += log(max(_ord_prob(Int(Y[t, s]), _clamp_eta(eta_ts), tau, link), 1e-12))
    end
    return q
end

function _phylo_ordinal_xlv_grad_hessian(Y, Lambda, alpha_lv, tau,
        mean_eta, Q, leaf_pos, eps, u, link::Link)
    p, n = size(Y)
    K = size(Lambda, 2)
    n_z = n * K
    n_block = length(u)
    m = n_z + n_block
    grad = zeros(Float64, m)
    H = zeros(Float64, m, m)

    @inbounds for s in 1:n
        for k in 1:K
            idx = (s - 1) * K + k
            grad[idx] -= eps[s, k]
            H[idx, idx] += 1.0
        end
    end
    @views begin
        grad[(n_z + 1):m] .-= Q * u
        H[(n_z + 1):m, (n_z + 1):m] .+= Q
    end

    @inbounds for s in 1:n, t in 1:p
        eta_ts = mean_eta[t, s] + u[leaf_pos[t]]
        for k in 1:K
            eta_ts += Lambda[t, k] * eps[s, k]
        end
        score_ts, weight_ts = _ord_score_weight(
            Int(Y[t, s]), _clamp_eta(eta_ts), tau, link)
        uidx = n_z + leaf_pos[t]

        grad[uidx] += score_ts
        H[uidx, uidx] += weight_ts
        for k in 1:K
            zidx_k = (s - 1) * K + k
            ltk = Lambda[t, k]
            grad[zidx_k] += ltk * score_ts
            cross = weight_ts * ltk
            H[zidx_k, uidx] += cross
            H[uidx, zidx_k] += cross
            for j in 1:K
                zidx_j = (s - 1) * K + j
                H[zidx_k, zidx_j] += weight_ts * ltk * Lambda[t, j]
            end
        end
    end
    return grad, H
end

function _phylo_ordinal_xlv_mode(Y, Lambda, alpha_lv, tau,
        mean_eta, Q, leaf_pos, link::Link; maxiter::Integer = 80,
        tol::Real = 1e-9)
    n = size(Y, 2)
    K = size(Lambda, 2)
    eps = zeros(Float64, n, K)
    u = zeros(Float64, size(Q, 1))
    local H
    mode_found = false

    for _ in 1:maxiter
        grad, H = _phylo_ordinal_xlv_grad_hessian(
            Y, Lambda, alpha_lv, tau, mean_eta, Q, leaf_pos, eps, u, link)
        cholH = try
            cholesky(Symmetric(H))
        catch
            return nothing, nothing, nothing
        end
        delta = cholH \ grad
        all(isfinite, delta) || return nothing, nothing, nothing

        q0 = _phylo_ordinal_xlv_logpost(
            Y, Lambda, alpha_lv, tau, mean_eta, Q, leaf_pos, eps, u, link)
        accepted = false
        step = 1.0
        n_z = n * K
        max_delta = maximum(abs, delta)
        for _half in 1:30
            eps_trial = copy(eps)
            u_trial = copy(u)
            @inbounds for s in 1:n, k in 1:K
                eps_trial[s, k] += step * delta[(s - 1) * K + k]
            end
            @inbounds for j in eachindex(u_trial)
                u_trial[j] += step * delta[n_z + j]
            end
            q1 = _phylo_ordinal_xlv_logpost(
                Y, Lambda, alpha_lv, tau, mean_eta, Q, leaf_pos,
                eps_trial, u_trial, link)
            if isfinite(q1) && q1 >= q0
                eps = eps_trial
                u = u_trial
                accepted = true
                break
            end
            step *= 0.5
        end
        accepted || return nothing, nothing, nothing
        if step * max_delta < tol
            mode_found = true
            break
        end
    end
    # `maxiter` exhausted: (eps, u) is not the mode, so a Laplace value built from it
    # would depend on `maxiter`. Report it as a failed search, like a non-SPD Hessian.
    mode_found || return nothing, nothing, nothing

    _, H = _phylo_ordinal_xlv_grad_hessian(
        Y, Lambda, alpha_lv, tau, mean_eta, Q, leaf_pos, eps, u, link)
    cholH = try
        cholesky(Symmetric(H))
    catch
        return nothing, nothing, nothing
    end
    return eps, u, cholH
end

"""
    _phylo_ordinal_xlv_marginal_loglik(Y, Lambda, alpha_lv, tau, sigma2_phy, phy, X_lv; kwargs...)

Internal Laplace marginal log-likelihood for the phylo x shared-cutpoint
Ordinal x predictor-informed latent-score S1 proof. The only admitted link is
`LogitLink`. The integrated variables are the site-score innovations and the
augmented-tree phylogenetic random intercept; shared ordered cutpoints are outer
nuisance parameters in the point wrapper, not interval targets.
"""
function _phylo_ordinal_xlv_marginal_loglik(Y::AbstractMatrix,
        Lambda::AbstractMatrix, alpha_lv::AbstractMatrix, tau::AbstractVector,
        sigma2_phy::Real, phy::AugmentedPhy, X_lv::AbstractMatrix;
        link::Link = LogitLink(), maxiter::Integer = 80, tol::Real = 1e-9)
    _phylo_ordinal_xlv_validate(Y, Lambda, alpha_lv, tau, phy, X_lv, link)
    sigma2_phy > 0 || return -Inf

    Yc = Matrix{Int}(Y)
    Lambda_c = Matrix{Float64}(Lambda)
    alpha_c = Matrix{Float64}(alpha_lv)
    tau_c = Vector{Float64}(tau)
    X_c = Matrix{Float64}(X_lv)
    mean_eta = _lv_mean_eta(Lambda_c, X_c, alpha_c)

    Q_cond, leaf_pos = _phylo_qcond(phy)
    Q = Matrix(Q_cond ./ float(sigma2_phy))
    cholQ = try
        cholesky(Symmetric(Q))
    catch
        return -Inf
    end

    eps_hat, u_hat, cholH = _phylo_ordinal_xlv_mode(
        Yc, Lambda_c, alpha_c, tau_c, mean_eta, Q, leaf_pos, link;
        maxiter = maxiter, tol = tol)
    eps_hat === nothing && return -Inf
    q = _phylo_ordinal_xlv_logpost(
        Yc, Lambda_c, alpha_c, tau_c, mean_eta, Q, leaf_pos, eps_hat,
        u_hat, link)
    return q + 0.5 * logdet(cholQ) - 0.5 * logdet(cholH)
end

function _phylo_ordinal_tau_to_psi(tau::AbstractVector)
    all(isfinite, float.(tau)) || throw(ArgumentError("tau_init must be finite"))
    all(diff(float.(tau)) .> 0) ||
        throw(ArgumentError("tau_init must be strictly increasing"))
    psi = similar(Vector{Float64}(tau))
    psi[1] = tau[1]
    @inbounds for c in 2:length(tau)
        psi[c] = log(tau[c] - tau[c - 1])
    end
    return psi
end

function _phylo_ordinal_xlv_unpack_packed(theta::AbstractVector, p::Integer,
        K::Integer, q_lv::Integer, C::Integer)
    rr = rr_theta_len(p, K)
    n_expected = q_lv * K + rr + C
    length(theta) == n_expected || throw(ArgumentError(
        "theta length ($(length(theta))) must equal $n_expected " *
        "(alpha_lv=$(q_lv * K) + rr=$rr + cutpoints=$(C - 1) + log_sigma2=1)"))

    cursor = 0
    alpha_vec = @view theta[(cursor + 1):(cursor + q_lv * K)]
    alpha_lv = reshape(alpha_vec, q_lv, K)
    cursor += q_lv * K
    theta_rr = @view theta[(cursor + 1):(cursor + rr)]
    Lambda = unpack_lambda(theta_rr, p, K)
    cursor += rr
    tau = _unpack_cutpoints(@view theta[(cursor + 1):(cursor + C - 1)])
    cursor += C - 1
    sigma2_phy = exp(theta[cursor + 1])
    return Lambda, alpha_lv, tau, sigma2_phy
end

function _phylo_ordinal_xlv_effects_from_packed(theta::AbstractVector,
        p::Integer, K::Integer, q_lv::Integer, C::Integer)
    Lambda, alpha_lv, _, _ = _phylo_ordinal_xlv_unpack_packed(theta, p, K, q_lv, C)
    return vec(Matrix(Lambda * transpose(alpha_lv)))
end

function _phylo_ordinal_xlv_nll_packed(theta::AbstractVector, Y::AbstractMatrix,
        p::Integer, K::Integer, C::Integer, phy::AugmentedPhy,
        X_lv::AbstractMatrix; q_lv::Integer, maxiter::Integer = 80,
        tol::Real = 1e-9)
    Lambda, alpha_lv, tau, sigma2_phy =
        _phylo_ordinal_xlv_unpack_packed(theta, p, K, q_lv, C)
    ll = _phylo_ordinal_xlv_marginal_loglik(Y, Lambda, alpha_lv, tau,
                                             sigma2_phy, phy, X_lv;
                                             maxiter = maxiter, tol = tol)
    return isfinite(ll) ? -ll : 1e12
end

function _phylo_ordinal_xlv_theta0(Y::AbstractMatrix, K::Integer,
        X_lv::AbstractMatrix; Lambda_init = nothing, alpha_lv_init = nothing,
        tau_init = nothing, sigma2_phy_init::Real = 0.5)
    p, n = size(Y)
    q_lv = size(X_lv, 2)
    K > 0 || throw(ArgumentError("K must be positive"))
    q_lv > 0 || throw(ArgumentError("X_lv must have at least one column"))
    sigma2_phy_init > 0 ||
        throw(ArgumentError("sigma2_phy_init must be positive"))
    C = tau_init === nothing ? maximum(Int.(Y)) : length(tau_init) + 1
    C >= 2 || throw(ArgumentError("ordinal response needs >= 2 categories; got $C"))

    Lambda0 = Lambda_init === nothing ?
        unpack_lambda(init_theta_rr(p, K), p, K) :
        Matrix{Float64}(Lambda_init)
    size(Lambda0) == (p, K) ||
        throw(ArgumentError("Lambda_init size $(size(Lambda0)) must equal ($p, $K)"))

    alpha0 = alpha_lv_init === nothing ? zeros(Float64, q_lv, K) :
        Matrix{Float64}(alpha_lv_init)
    size(alpha0) == (q_lv, K) ||
        throw(ArgumentError("alpha_lv_init size $(size(alpha0)) must equal ($q_lv, $K)"))

    psi0 = if tau_init === nothing
        counts = zeros(Int, C)
        @inbounds for y in Y
            yi = Int(y)
            1 <= yi <= C ||
                throw(ArgumentError("ordinal response category $y is outside 1:$C"))
            counts[yi] += 1
        end
        cum = cumsum(counts ./ sum(counts))
        tau0 = [log(clamp(cum[c], 1e-3, 1 - 1e-3) /
                    (1 - clamp(cum[c], 1e-3, 1 - 1e-3))) for c in 1:(C - 1)]
        psi0 = similar(tau0)
        psi0[1] = tau0[1]
        @inbounds for c in 2:(C - 1)
            psi0[c] = log(max(tau0[c] - tau0[c - 1], 1e-3))
        end
        psi0
    else
        _phylo_ordinal_tau_to_psi(Vector{Float64}(tau_init))
    end

    return vcat(vec(alpha0), pack_lambda(Lambda0), psi0,
                log(float(sigma2_phy_init))), C
end

"""
    _fit_phylo_ordinal_xlv(Y, phy; K, X_lv, kwargs...) -> NamedTuple

Private point-route wrapper for the phylo x shared-cutpoint Ordinal x
predictor-informed LV S1 canary. This truth-startable optimiser is intentionally
not exported; it exists only to support local selected-entry `B_eta_realized`
profile-LR diagnostics.
"""
function _fit_phylo_ordinal_xlv(Y::AbstractMatrix, phy::AugmentedPhy;
        K::Integer, X_lv::AbstractMatrix,
        Lambda_init = nothing, alpha_lv_init = nothing, tau_init = nothing,
        sigma2_phy_init::Real = 0.5, iterations::Integer = 200,
        g_tol::Real = 1e-6, newton_maxiter::Integer = 80,
        newton_tol::Real = 1e-9)
    p, n = size(Y)
    p == phy.n_leaves ||
        throw(ArgumentError("size(Y,1)=$p must equal phy.n_leaves=$(phy.n_leaves)"))
    size(X_lv, 1) == n ||
        throw(ArgumentError("X_lv rows ($(size(X_lv, 1))) must equal size(Y,2)=$n"))
    q_lv = size(X_lv, 2)
    theta0, C = _phylo_ordinal_xlv_theta0(Y, K, X_lv;
                                          Lambda_init = Lambda_init,
                                          alpha_lv_init = alpha_lv_init,
                                          tau_init = tau_init,
                                          sigma2_phy_init = sigma2_phy_init)
    nll = theta -> begin
        v = try
            _phylo_ordinal_xlv_nll_packed(theta, Y, p, K, C, phy, X_lv;
                                          q_lv = q_lv,
                                          maxiter = newton_maxiter,
                                          tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(nll, theta0, ls,
                         Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    theta_hat = Optim.minimizer(res)
    Lambda_hat, alpha_hat, tau_hat, sigma2_hat =
        _phylo_ordinal_xlv_unpack_packed(theta_hat, p, K, q_lv, C)
    return (Lambda = Matrix(Lambda_hat), alpha_lv = Matrix(alpha_hat),
            tau = Vector{Float64}(tau_hat), C = C, sigma2_phy = sigma2_hat,
            loglik = _fit_verdict(res)[1], converged = _fit_verdict(res)[2],
            iterations = Optim.iterations(res), theta_packed = collect(theta_hat))
end

function _phylo_ordinal_xlv_profile_indices(indices, nb::Integer)
    idx = collect(Int, indices)
    isempty(idx) && throw(ArgumentError("profile entries must not be empty"))
    for i in idx
        1 <= i <= nb || throw(ArgumentError("profile entry $i outside 1:$nb"))
    end
    length(unique(idx)) == length(idx) ||
        throw(ArgumentError("profile entries must be unique"))
    return idx
end

function _phylo_ordinal_xlv_constrained_refit(nll::Function,
        theta_start::AbstractVector, p::Integer, K::Integer, q_lv::Integer,
        C::Integer, entry::Integer, target::Real; profile_iterations::Integer,
        penalty_weights)
    theta_c = collect(Float64, theta_start)
    last_res = nothing
    for weight in penalty_weights
        obj = function (theta)
            val = try nll(theta) catch; return 1e12 end
            isfinite(val) || return 1e12
            b = _phylo_ordinal_xlv_effects_from_packed(theta, p, K, q_lv, C)[entry]
            return val + 0.5 * weight * (b - target)^2
        end
        last_res = try
            Optim.optimize(obj, theta_c, Optim.NelderMead(),
                           Optim.Options(iterations = profile_iterations,
                                         f_reltol = 1e-8, x_abstol = 1e-8))
        catch
            nothing
        end
        last_res === nothing && break
        theta_c = Optim.minimizer(last_res)
    end
    val_c = try nll(theta_c) catch; NaN end
    b_c = try
        _phylo_ordinal_xlv_effects_from_packed(theta_c, p, K, q_lv, C)[entry]
    catch
        NaN
    end
    return (nll = val_c, theta = collect(theta_c), effect = b_c,
            constraint_error = abs(b_c - target),
            # `converged` is the refit's own verdict (#752). Meeting the
            # constraint says nothing about the other parameters, so
            # feasibility is `constraint_error`, a separate field. Same rule
            # as the Poisson/binomial/NB/gamma xlv refits.
            converged = last_res !== nothing && Optim.converged(last_res))
end

function _phylo_ordinal_xlv_profile_eta_realized(fit, Y::AbstractMatrix,
        phy::AugmentedPhy, X_lv::AbstractMatrix, entries::AbstractVector{Int},
        eta_realized_truth::AbstractVector{<:Real};
        level::Real = 0.95, profile_iterations::Integer = 600,
        penalty_weights = (1e2, 1e3, 1e4, 1e5, 1e6),
        profile_endpoints::Bool = true,
        endpoint_step::Union{Nothing, Real} = nothing,
        profile_max_expand::Integer = 12,
        profile_max_bisect::Integer = 18,
        constraint_tol::Real = 1e-3,
        newton_maxiter::Integer = 80, newton_tol::Real = 1e-9)
    0 < level < 1 || throw(ArgumentError("level must be in (0,1); got $level"))
    profile_iterations > 0 ||
        throw(ArgumentError("profile_iterations must be positive; got $profile_iterations"))
    profile_max_expand > 0 ||
        throw(ArgumentError("profile_max_expand must be positive; got $profile_max_expand"))
    profile_max_bisect > 0 ||
        throw(ArgumentError("profile_max_bisect must be positive; got $profile_max_bisect"))
    constraint_tol > 0 ||
        throw(ArgumentError("constraint_tol must be positive; got $constraint_tol"))
    endpoint_step !== nothing && (!(isfinite(endpoint_step)) || endpoint_step <= 0) &&
        throw(ArgumentError("endpoint_step must be positive and finite; got $endpoint_step"))
    p, K = size(fit.Lambda)
    q_lv = size(X_lv, 2)
    C = fit.C
    nb = p * q_lv
    idx = _phylo_ordinal_xlv_profile_indices(entries, nb)
    length(eta_realized_truth) == nb || throw(ArgumentError(
        "eta_realized_truth length ($(length(eta_realized_truth))) must equal p*q_lv=$nb"))
    theta0 = collect(Float64, fit.theta_packed)
    nll = theta -> _phylo_ordinal_xlv_nll_packed(theta, Y, p, K, C, phy, X_lv;
                                                 q_lv = q_lv,
                                                 maxiter = newton_maxiter,
                                                 tol = newton_tol)
    nll0 = nll(theta0)
    cutoff = quantile(Chisq(1), level)
    b_hat = _phylo_ordinal_xlv_effects_from_packed(theta0, p, K, q_lv, C)

    terms = String[]
    estimates = Float64[]
    targets = Float64[]
    deviances = Float64[]
    constraint_errors = Float64[]
    converged = Bool[]
    lowers = Float64[]
    uppers = Float64[]
    endpoint_status = Symbol[]
    for entry in idx
        target = float(eta_realized_truth[entry])
        refit_truth = _phylo_ordinal_xlv_constrained_refit(
            nll, theta0, p, K, q_lv, C, entry, target;
            profile_iterations = profile_iterations,
            penalty_weights = penalty_weights)
        val_c = refit_truth.nll
        lr = isfinite(val_c) ? 2 * (val_c - nll0) : NaN
        row = ((entry - 1) % p) + 1
        col = ((entry - 1) ÷ p) + 1
        push!(terms, "B_eta_realized[$row,$col]")
        push!(estimates, b_hat[entry])
        push!(targets, target)
        push!(deviances, lr)
        push!(constraint_errors, refit_truth.constraint_error)
        push!(converged, refit_truth.converged)

        if profile_endpoints
            step = endpoint_step === nothing ?
                max(abs(b_hat[entry] - target),
                    0.05 * max(1.0, abs(b_hat[entry])), 0.02) :
                float(endpoint_step)
            theta_lower = copy(theta0)
            theta_upper = copy(theta0)
            function dev_at(c, start)
                refit = _phylo_ordinal_xlv_constrained_refit(
                    nll, start, p, K, q_lv, C, entry, c;
                    profile_iterations = profile_iterations,
                    penalty_weights = penalty_weights)
                ok = refit.converged && isfinite(refit.nll) &&
                     refit.constraint_error <= constraint_tol
                return ok ? 2 * (refit.nll - nll0) : NaN, refit.theta
            end
            function dev_lower(c)
                D, theta_new = dev_at(c, theta_lower)
                isfinite(D) && (theta_lower = theta_new)
                return D
            end
            function dev_upper(c)
                D, theta_new = dev_at(c, theta_upper)
                isfinite(D) && (theta_upper = theta_new)
                return D
            end
            lo = _profile_bisect_side(dev_lower, b_hat[entry], -step, cutoff;
                                      max_expand = profile_max_expand,
                                      max_bisect = profile_max_bisect)
            hi = _profile_bisect_side(dev_upper, b_hat[entry], step, cutoff;
                                      max_expand = profile_max_expand,
                                      max_bisect = profile_max_bisect)
            push!(lowers, lo)
            push!(uppers, hi)
            push!(endpoint_status, isnan(lo) && isnan(hi) ? :failed :
                                   isnan(lo) || isnan(hi) ? :partial :
                                   :profile)
        else
            push!(lowers, NaN)
            push!(uppers, NaN)
            push!(endpoint_status, :not_requested)
        end
    end

    covered = [isfinite(lr) && lr <= cutoff for lr in deviances]
    cutpoints_ordered = all(diff(fit.tau) .> 0)
    return (term = terms, estimate = estimates, target = targets,
            lower = lowers, upper = uppers,
            se = fill(NaN, length(idx)), lr_deviance = deviances,
            lr_cutoff = fill(cutoff, length(idx)),
            constrained_error = constraint_errors,
            constrained_converged = converged, covered = covered,
            endpoint_status = endpoint_status,
            cutpoints_ordered = cutpoints_ordered,
            level = level, method = :profile_eta_realized,
            pd_hessian = cutpoints_ordered && all(converged) &&
                         all(s -> s === :profile || s === :not_requested,
                             endpoint_status))
end
