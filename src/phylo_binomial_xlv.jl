# Internal S1 proof for the phylo x Binomial x predictor-informed LV target.
#
# Model:
#   epsilon_s ~ N(0, I_K)
#   u         ~ N(0, sigma_phy^2 * Q_cond^{-1})
#   eta[t,s] = beta[t] + Lambda[t,:]' * (X_lv[s,:] * alpha_lv + epsilon_s) +
#              u[leaf_pos[t]]
#   Y[t,s]   ~ Binomial(N[t,s], logistic(eta[t,s]))
#
# This is deliberately not exported. It is a reduction-tested likelihood surface
# for a structural-source non-Gaussian LV canary, not R grammar support.

function _phylo_binomial_xlv_validate(Y::AbstractMatrix, N::AbstractMatrix,
        beta::AbstractVector, Lambda::AbstractMatrix, alpha_lv::AbstractMatrix,
        phy::AugmentedPhy, X_lv::AbstractMatrix, link::Link)
    link isa LogitLink ||
        throw(ArgumentError("phylo Binomial X_lv S1 is currently logit-link only"))
    p, n = size(Y)
    p == phy.n_leaves ||
        throw(ArgumentError("size(Y,1)=$p must equal phy.n_leaves=$(phy.n_leaves)"))
    size(N) == (p, n) || throw(DimensionMismatch("N must be $(p)×$(n)"))
    all(isfinite, float.(N)) || throw(ArgumentError("N must be finite"))
    all(x -> isinteger(float(x)), N) ||
        throw(ArgumentError("N must contain integer-valued trial counts"))
    all(N .> 0) || throw(ArgumentError("N must be positive"))
    all(isfinite, float.(Y)) || throw(ArgumentError("Y must be finite"))
    all(x -> isinteger(float(x)), Y) ||
        throw(ArgumentError("Y must contain integer-valued successes"))
    all(Y .>= 0) || throw(ArgumentError("Y must be non-negative"))
    all(Y .<= N) || throw(ArgumentError("Y must not exceed N"))
    length(beta) == p ||
        throw(ArgumentError("length(beta)=$(length(beta)) must equal size(Y,1)=$p"))
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
    return p, n, K
end

function _phylo_binomial_xlv_logpost(Y, N, beta, Lambda, mean_eta, Q, leaf_pos, eps, u)
    p, n = size(Y)
    K = size(Lambda, 2)
    q = -0.5 * sum(abs2, eps) - 0.5 * dot(u, Q * u)
    @inbounds for s in 1:n, t in 1:p
        eta_ts = beta[t] + mean_eta[t, s] + u[leaf_pos[t]]
        for k in 1:K
            eta_ts += Lambda[t, k] * eps[s, k]
        end
        eta_c = _clamp_eta(eta_ts)
        mu_ts = _clamp_mu(Binomial(), linkinv(LogitLink(), eta_c))
        q += _glm_logpdf(Binomial(), mu_ts, N[t, s], Y[t, s])
    end
    return q
end

function _phylo_binomial_xlv_grad_hessian(Y, N, beta, Lambda, mean_eta, Q,
        leaf_pos, eps, u; hessian::Symbol = :fisher)
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
        eta_ts = beta[t] + mean_eta[t, s] + u[leaf_pos[t]]
        for k in 1:K
            eta_ts += Lambda[t, k] * eps[s, k]
        end
        eta_c = _clamp_eta(eta_ts)
        mu_ts = _clamp_mu(Binomial(), linkinv(LogitLink(), eta_c))
        me_ts = mu_eta(LogitLink(), eta_c)
        score_ts = _glm_score(Binomial(), mu_ts, N[t, s], me_ts, Y[t, s])
        # Role separation. Default `:fisher` leaves the Newton loop exactly as it
        # was — the mode search wants expected information (W ≥ 0, so the step is
        # always well-defined). Only the FINAL assembly, at the converged mode,
        # asks for `:observed`, which is what the log-det needs to match TMB.
        weight_ts = hessian === :fisher ?
            _glm_weight(Binomial(), mu_ts, N[t, s], me_ts) :
            _glm_obs_weight(Binomial(), mu_ts, N[t, s], me_ts, Y[t, s], LogitLink(), eta_c)
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

function _phylo_binomial_xlv_mode(Y, N, beta, Lambda, mean_eta, Q, leaf_pos;
        maxiter::Integer = 80, tol::Real = 1e-9)
    n = size(Y, 2)
    K = size(Lambda, 2)
    eps = zeros(Float64, n, K)
    u = zeros(Float64, size(Q, 1))
    local H
    mode_found = false

    for _ in 1:maxiter
        grad, H = _phylo_binomial_xlv_grad_hessian(Y, N, beta, Lambda, mean_eta,
                                                   Q, leaf_pos, eps, u)
        cholH = try
            cholesky(Symmetric(H))
        catch
            return nothing, nothing, nothing
        end
        delta = cholH \ grad
        all(isfinite, delta) || return nothing, nothing, nothing

        q0 = _phylo_binomial_xlv_logpost(Y, N, beta, Lambda, mean_eta, Q,
                                         leaf_pos, eps, u)
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
            q1 = _phylo_binomial_xlv_logpost(Y, N, beta, Lambda, mean_eta, Q,
                                             leaf_pos, eps_trial, u_trial)
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

    _, H = _phylo_binomial_xlv_grad_hessian(Y, N, beta, Lambda, mean_eta, Q,
                                            leaf_pos, eps, u;
        hessian = _default_hessian(Binomial(), LogitLink()))
    cholH = try
        cholesky(Symmetric(H))
    catch
        return nothing, nothing, nothing
    end
    return eps, u, cholH
end

"""
    _phylo_binomial_xlv_marginal_loglik(Y, N, beta, Lambda, alpha_lv, sigma2_phy, phy, X_lv; kwargs...)

Internal Laplace marginal log-likelihood for the phylo x Binomial x
predictor-informed latent-score S1 proof. The only admitted link is
`LogitLink`. The integrated variables are the site-score innovations and the
augmented-tree phylogenetic random intercept. This function is intentionally
private and reduction-tested before any fitter, bridge route, or R grammar is
exposed.
"""
function _phylo_binomial_xlv_marginal_loglik(Y::AbstractMatrix, N::AbstractMatrix,
        beta::AbstractVector, Lambda::AbstractMatrix, alpha_lv::AbstractMatrix,
        sigma2_phy::Real, phy::AugmentedPhy, X_lv::AbstractMatrix;
        link::Link = LogitLink(), maxiter::Integer = 80, tol::Real = 1e-9)
    _phylo_binomial_xlv_validate(Y, N, beta, Lambda, alpha_lv, phy, X_lv, link)
    sigma2_phy > 0 || return -Inf

    Yc = Matrix{Float64}(Y)
    Nc = Matrix{Float64}(N)
    beta_c = Vector{Float64}(beta)
    Lambda_c = Matrix{Float64}(Lambda)
    alpha_c = Matrix{Float64}(alpha_lv)
    X_c = Matrix{Float64}(X_lv)
    mean_eta = _lv_mean_eta(Lambda_c, X_c, alpha_c)

    Q_cond, leaf_pos = _phylo_qcond(phy)
    Q = Matrix(Q_cond ./ float(sigma2_phy))
    cholQ = try
        cholesky(Symmetric(Q))
    catch
        return -Inf
    end

    eps_hat, u_hat, cholH = _phylo_binomial_xlv_mode(Yc, Nc, beta_c, Lambda_c,
                                                     mean_eta, Q, leaf_pos;
                                                     maxiter = maxiter, tol = tol)
    eps_hat === nothing && return -Inf
    q = _phylo_binomial_xlv_logpost(Yc, Nc, beta_c, Lambda_c, mean_eta, Q,
                                    leaf_pos, eps_hat, u_hat)
    return q + 0.5 * logdet(cholQ) - 0.5 * logdet(cholH)
end

function _phylo_binomial_xlv_unpack_packed(theta::AbstractVector, p::Integer,
        K::Integer, q_lv::Integer)
    rr = rr_theta_len(p, K)
    n_expected = p + q_lv * K + rr + 1
    length(theta) == n_expected || throw(ArgumentError(
        "theta length ($(length(theta))) must equal $n_expected " *
        "(p=$p + alpha_lv=$(q_lv * K) + rr=$rr + log_sigma2=1)"))

    cursor = 0
    beta = @view theta[(cursor + 1):(cursor + p)]
    cursor += p
    alpha_vec = @view theta[(cursor + 1):(cursor + q_lv * K)]
    alpha_lv = reshape(alpha_vec, q_lv, K)
    cursor += q_lv * K
    theta_rr = @view theta[(cursor + 1):(cursor + rr)]
    Lambda = unpack_lambda(theta_rr, p, K)
    cursor += rr
    sigma2_phy = exp(theta[cursor + 1])
    return beta, Lambda, alpha_lv, sigma2_phy
end

function _phylo_binomial_xlv_effects_from_packed(theta::AbstractVector,
        p::Integer, K::Integer, q_lv::Integer)
    _, Lambda, alpha_lv, _ = _phylo_binomial_xlv_unpack_packed(theta, p, K, q_lv)
    return vec(Matrix(Lambda * transpose(alpha_lv)))
end

function _phylo_binomial_xlv_nll_packed(theta::AbstractVector, Y::AbstractMatrix,
        N::AbstractMatrix, p::Integer, K::Integer, phy::AugmentedPhy,
        X_lv::AbstractMatrix; q_lv::Integer, maxiter::Integer = 80,
        tol::Real = 1e-9)
    beta, Lambda, alpha_lv, sigma2_phy =
        _phylo_binomial_xlv_unpack_packed(theta, p, K, q_lv)
    ll = _phylo_binomial_xlv_marginal_loglik(Y, N, beta, Lambda, alpha_lv,
                                             sigma2_phy, phy, X_lv;
                                             maxiter = maxiter, tol = tol)
    return isfinite(ll) ? -ll : 1e12
end

# The objective `_fit_phylo_binomial_xlv` hands to Optim: any exception or non-finite value
# at a line-search trial step becomes the sentinel, as in the Gamma and NB2 fitters (#944).
function _phylo_binomial_xlv_nll_guarded(theta::AbstractVector, Y::AbstractMatrix,
        N::AbstractMatrix, p::Integer, K::Integer, phy::AugmentedPhy,
        X_lv::AbstractMatrix; q_lv::Integer, maxiter::Integer = 80,
        tol::Real = 1e-9)
    v = try
        _phylo_binomial_xlv_nll_packed(theta, Y, N, p, K, phy, X_lv;
                                       q_lv = q_lv, maxiter = maxiter, tol = tol)
    catch e
        e isa InterruptException && rethrow()
        return 1e12
    end
    return isfinite(v) ? v : 1e12
end

function _phylo_binomial_xlv_theta0(Y::AbstractMatrix, N::AbstractMatrix,
        K::Integer, X_lv::AbstractMatrix; beta_init = nothing,
        Lambda_init = nothing, alpha_lv_init = nothing,
        sigma2_phy_init::Real = 0.5)
    p, n = size(Y)
    size(N) == (p, n) || throw(DimensionMismatch("N must be $(p)×$(n)"))
    q_lv = size(X_lv, 2)
    K > 0 || throw(ArgumentError("K must be positive"))
    q_lv > 0 || throw(ArgumentError("X_lv must have at least one column"))
    sigma2_phy_init > 0 ||
        throw(ArgumentError("sigma2_phy_init must be positive"))

    beta0 = beta_init === nothing ?
        [(m = clamp((sum(Y[t, s] for s in 1:n) + 0.5) /
                    (sum(N[t, s] for s in 1:n) + 1), 1e-4, 1 - 1e-4);
          log(m / (1 - m)))
         for t in 1:p] :
        collect(Float64, beta_init)
    length(beta0) == p ||
        throw(ArgumentError("beta_init length ($(length(beta0))) must equal size(Y,1)=$p"))

    Lambda0 = Lambda_init === nothing ?
        unpack_lambda(init_theta_rr(p, K), p, K) :
        Matrix{Float64}(Lambda_init)
    size(Lambda0) == (p, K) ||
        throw(ArgumentError("Lambda_init size $(size(Lambda0)) must equal ($p, $K)"))

    alpha0 = alpha_lv_init === nothing ? zeros(Float64, q_lv, K) :
        Matrix{Float64}(alpha_lv_init)
    size(alpha0) == (q_lv, K) ||
        throw(ArgumentError("alpha_lv_init size $(size(alpha0)) must equal ($q_lv, $K)"))

    return vcat(beta0, vec(alpha0), pack_lambda(Lambda0), log(float(sigma2_phy_init)))
end

# Verdict shared by the phylogenetic xlv point fitters (binomial, NB2, Gamma). #505 class:
# `Optim.converged(res)` also fires on a zero-length line-search step or an f/x stop, so the
# `_fit_verdict(res)` pass-through reported `converged = true` at finite-difference gradients
# of order 1-100. This adds the same stall test `fit_phylo_gaussian` uses (`_phylo_g_met`:
# a loose relative bound, not the optimiser's own `g_tol`). A gradient at or near the 1e12
# failure value (a finite-difference step landing on the objective sentinel) is its own
# reason, not a generic "gradient not small".
function _phylo_xlv_verdict(res, g_tol::Real)
    loglik, optim_converged, iterations = _fit_verdict(res)
    gres = Optim.g_residual(res)
    reason = if !isfinite(loglik)
        :objective_failure
    elseif !isfinite(gres) || gres >= _NLL_FAIL_THRESHOLD
        :gradient_at_failure_value
    elseif !_phylo_g_met(res, g_tol)
        :gradient_not_small
    elseif !optim_converged
        :not_converged
    else
        :converged
    end
    return (loglik = loglik, converged = reason === :converged,
            iterations = iterations, reason = reason)
end

"""
    _fit_phylo_binomial_xlv(Y, N, phy; K, X_lv, kwargs...) -> NamedTuple

Private point-route wrapper for the phylo x Binomial x predictor-informed LV S1
canary. This truth-startable optimiser is intentionally not exported; it exists
only to support local selected-entry `B_eta_realized` profile-LR diagnostics.
"""
function _fit_phylo_binomial_xlv(Y::AbstractMatrix, N::AbstractMatrix,
        phy::AugmentedPhy; K::Integer, X_lv::AbstractMatrix,
        beta_init = nothing, Lambda_init = nothing, alpha_lv_init = nothing,
        sigma2_phy_init::Real = 0.5,
        iterations::Integer = 200, g_tol::Real = 1e-6,
        newton_maxiter::Integer = 80, newton_tol::Real = 1e-9)
    p, n = size(Y)
    p == phy.n_leaves ||
        throw(ArgumentError("size(Y,1)=$p must equal phy.n_leaves=$(phy.n_leaves)"))
    size(N) == (p, n) || throw(DimensionMismatch("N must be $(p)×$(n)"))
    size(X_lv, 1) == n ||
        throw(ArgumentError("X_lv rows ($(size(X_lv, 1))) must equal size(Y,2)=$n"))
    q_lv = size(X_lv, 2)
    theta0 = _phylo_binomial_xlv_theta0(Y, N, K, X_lv;
                                        beta_init = beta_init,
                                        Lambda_init = Lambda_init,
                                        alpha_lv_init = alpha_lv_init,
                                        sigma2_phy_init = sigma2_phy_init)
    nll = theta -> _phylo_binomial_xlv_nll_guarded(theta, Y, N, p, K, phy, X_lv;
                                                   q_lv = q_lv,
                                                   maxiter = newton_maxiter,
                                                   tol = newton_tol)
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    res = Optim.optimize(nll, theta0, ls,
                         Optim.Options(g_tol = g_tol, iterations = iterations);
                         autodiff = :finite)
    theta_hat = Optim.minimizer(res)
    beta_hat, Lambda_hat, alpha_hat, sigma2_hat =
        _phylo_binomial_xlv_unpack_packed(theta_hat, p, K, q_lv)
    verdict = _phylo_xlv_verdict(res, g_tol)
    return (beta = collect(beta_hat), Lambda = Matrix(Lambda_hat),
            alpha_lv = Matrix(alpha_hat), sigma2_phy = sigma2_hat,
            loglik = verdict.loglik, converged = verdict.converged,
            iterations = Optim.iterations(res), theta_packed = collect(theta_hat),
            stop_reason = verdict.reason)
end

function _phylo_binomial_xlv_profile_indices(indices, nb::Integer)
    idx = collect(Int, indices)
    isempty(idx) && throw(ArgumentError("profile entries must not be empty"))
    for i in idx
        1 <= i <= nb || throw(ArgumentError("profile entry $i outside 1:$nb"))
    end
    length(unique(idx)) == length(idx) ||
        throw(ArgumentError("profile entries must be unique"))
    return idx
end

function _phylo_binomial_xlv_constrained_refit(nll::Function,
        theta_start::AbstractVector, p::Integer, K::Integer, q_lv::Integer,
        entry::Integer, target::Real; profile_iterations::Integer,
        penalty_weights)
    theta_c = collect(Float64, theta_start)
    last_res = nothing
    for weight in penalty_weights
        obj = function (theta)
            val = try nll(theta) catch; return 1e12 end
            isfinite(val) || return 1e12
            b = _phylo_binomial_xlv_effects_from_packed(theta, p, K, q_lv)[entry]
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
        _phylo_binomial_xlv_effects_from_packed(theta_c, p, K, q_lv)[entry]
    catch
        NaN
    end
    return (nll = val_c, theta = collect(theta_c), effect = b_c,
            constraint_error = abs(b_c - target),
            converged = last_res !== nothing && Optim.converged(last_res))
end

function _phylo_binomial_xlv_profile_eta_realized(fit, Y::AbstractMatrix,
        N::AbstractMatrix, phy::AugmentedPhy, X_lv::AbstractMatrix,
        entries::AbstractVector{Int}, eta_realized_truth::AbstractVector{<:Real};
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
    nb = p * q_lv
    idx = _phylo_binomial_xlv_profile_indices(entries, nb)
    length(eta_realized_truth) == nb || throw(ArgumentError(
        "eta_realized_truth length ($(length(eta_realized_truth))) must equal p*q_lv=$nb"))
    theta0 = collect(Float64, fit.theta_packed)
    nll = theta -> _phylo_binomial_xlv_nll_packed(theta, Y, N, p, K, phy, X_lv;
                                                  q_lv = q_lv,
                                                  maxiter = newton_maxiter,
                                                  tol = newton_tol)
    nll0 = nll(theta0)
    cutoff = quantile(Chisq(1), level)
    b_hat = _phylo_binomial_xlv_effects_from_packed(theta0, p, K, q_lv)

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
        refit_truth = _phylo_binomial_xlv_constrained_refit(
            nll, theta0, p, K, q_lv, entry, target;
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
                refit = _phylo_binomial_xlv_constrained_refit(
                    nll, start, p, K, q_lv, entry, c;
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
    return (term = terms, estimate = estimates, target = targets,
            lower = lowers, upper = uppers,
            se = fill(NaN, length(idx)), lr_deviance = deviances,
            lr_cutoff = fill(cutoff, length(idx)),
            constrained_error = constraint_errors,
            constrained_converged = converged, covered = covered,
            endpoint_status = endpoint_status,
            level = level, method = :profile_eta_realized,
            pd_hessian = all(converged) &&
                         all(s -> s === :profile || s === :not_requested,
                             endpoint_status))
end
