# L26 — Julia-only zero-inflated / hurdle Wald coverage sweep (extends #926).
# Core: truth, data generator, fit + Wald (+ delta-method ΛΛ') extraction.
using GLLVModels, Random, Distributions, LinearAlgebra, Printf, ForwardDiff
const G = GLLVModels

const P = 10
const K = 2
const FAMILIES = ("ZIP", "ZINB", "HurdlePoisson")
const PIS = (0.05, 0.2, 0.5)
const NS = (100, 300, 1000)
const BASESEED = 20261016

# True loadings (lower-triangular, positive diagonal) and count intercepts, shared by all cells.
const LAMBDA_TRUE = [ 0.8  0.0;
                      0.5  0.6;
                     -0.4  0.5;
                      0.6 -0.3;
                      0.3  0.6;
                     -0.5  0.4;
                      0.7 -0.2;
                      0.2 -0.5;
                     -0.3  0.5;
                      0.4  0.3]
const BETAC_TRUE = collect(range(0.5, 1.5; length = P))   # count log-mean intercepts
const R_TRUE = 2.0                                        # ZINB NB2 size (shared)

# cell = (family-1)*9 + (pi-1)*3 + n_index; n innermost so each (family, pi) n-series is consecutive.
function cell_spec(cell::Integer)
    1 <= cell <= 27 || throw(ArgumentError("cell must be 1..27"))
    c = cell - 1
    fi = c ÷ 9 + 1; pi_i = (c % 9) ÷ 3 + 1; ni = c % 3 + 1
    return (cell = cell, family = FAMILIES[fi], pi = PIS[pi_i], n = NS[ni])
end

seed_of(cell, rep) = BASESEED + 10000 * cell + rep

logit(x) = log(x / (1 - x))

# Truth for betaz on each family's own scale:
#   ZIP / ZINB: betaz = logit(P(structural zero)) = logit(pi)
#   Hurdle-Poisson: betaz = logit(P(y > 0)) = logit(1 - pi)  (pi = hurdle zero probability)
betaz_true(family, π) = family == "HurdlePoisson" ? logit(1 - π) : logit(π)

# Exact inverse-CDF zero-truncated Poisson draw (one uniform per draw).
function rand_ztpois(rng::AbstractRNG, μ::Float64)
    u = rand(rng)
    p0 = exp(-μ)
    # target: P(Y <= k | Y >= 1) >= u  <=>  F(k) >= p0 + u (1 - p0)
    t = p0 + u * (1 - p0)
    k = 0; pk = p0; F = p0
    while F < t
        k += 1
        pk *= μ / k
        F += pk
        k > 10_000 && break
    end
    return max(k, 1)
end

"""
    simulate_cell(cell, rep) -> (Y::Matrix{Int} p×n, Z::Matrix K×n, seed)

Generator (same structure as the fitted model, Λz = 0):
  z_s ~ N(0, I_K);  η^c_{ts} = βc_t + (Λ z_s)_t
  ZIP:  y = 0 w.p. π, else Poisson(exp η^c)
  ZINB: y = 0 w.p. π, else NB2(mean exp η^c, size r = 2)
  Hurdle-Poisson: y = 0 w.p. π, else zero-truncated Poisson(exp η^c)
RNG: Xoshiro(seed); Z drawn first (randn K×n), then sites s = 1..n, species t = 1..p,
each cell draws one uniform for the zero part and then (if not zero) the count.
"""
function simulate_cell(cell::Integer, rep::Integer)
    sp = cell_spec(cell)
    seed = seed_of(cell, rep)
    rng = Xoshiro(seed)
    n = sp.n; π = sp.pi
    Z = randn(rng, K, n)
    Y = zeros(Int, P, n)
    for s in 1:n
        ηc = BETAC_TRUE .+ LAMBDA_TRUE * view(Z, :, s)
        for t in 1:P
            μ = exp(ηc[t])
            u = rand(rng)
            if u < π
                Y[t, s] = 0
            elseif sp.family == "ZIP"
                Y[t, s] = rand(rng, Poisson(μ))
            elseif sp.family == "ZINB"
                Y[t, s] = rand(rng, NegativeBinomial(R_TRUE, R_TRUE / (R_TRUE + μ)))
            else
                Y[t, s] = rand_ztpois(rng, μ)
            end
        end
    end
    return Y, Z, seed
end

function fit_family(family::AbstractString, Y)
    family == "ZIP"  && return G.fit_zip_gllvm(Y; K = K)
    family == "ZINB" && return G.fit_zinb_gllvm(Y; K = K)
    family == "HurdlePoisson" && return G.fit_hurdle_poisson_gllvm(Y; K = K)
    error("unknown family $family")
end

# Unique entries (i >= j) of Λ Λ' as a function of packed Λ.
function llt_vec(θΛ::AbstractVector)
    Λ = G.unpack_lambda(θΛ, P, K)
    S = Λ * Λ'
    return [S[i, j] for j in 1:P for i in j:P]
end
const LLT_NAMES = ["LLt[$i,$j]" for j in 1:P for i in j:P]
const LLT_TRUE = llt_vec(G.pack_lambda(LAMBDA_TRUE))

function central_grad(f, x; rel = 1e-5)
    g = similar(x)
    for i in eachindex(x)
        h = rel * max(abs(x[i]), 1.0)
        xp = copy(x); xp[i] += h; xm = copy(x); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    return g
end

"""
    run_one(cell, rep) -> (receipt::NamedTuple, params::Vector{NamedTuple})

Fit with the package's named fitter, then Wald intervals and the observed-information
covariance from ONE finite-difference Hessian — exactly what `confint(fit, Y; method=:wald)`
and `vcov(fit, Y)` return (they share `_family_confint_setup` + `_family_wald`).
ΛΛ' intervals: delta method, Var = J V_Λ J', J = ∂ vech(ΛΛ')/∂ pack(Λ) (ForwardDiff).
"""
function run_one(cell::Integer, rep::Integer)
    sp = cell_spec(cell)
    Y, Z, seed = simulate_cell(cell, rep)
    t0 = time()
    fit = fit_family(sp.family, Y)
    t_fit = time() - t0
    t1 = time()
    ad, _ = G._family_confint_setup(fit, Y; method = :wald, level = 0.95, N = nothing, X = nothing,
                                    mask = nothing, objective = :fit, newton_maxiter = 100, newton_tol = 1e-9)
    m = length(ad.θ)
    W = G._family_wald(ad, collect(1:m), 0.95; covariance = true)
    t_wald = time() - t1
    g = central_grad(ad.nll, ad.θ)
    z = quantile(Normal(), 0.975)
    rr = G.rr_theta_len(P, K)
    iΛ = (2P + 1):(2P + rr)
    V = W.covariance
    est_llt = llt_vec(ad.θ[iΛ])
    J = ForwardDiff.jacobian(llt_vec, ad.θ[iΛ])
    VΛ = V[iΛ, iΛ]
    se_llt = all(isfinite, VΛ) ? sqrt.(max.(diag(J * VΛ * J'), 0.0)) : fill(NaN, length(est_llt))
    bz_true = betaz_true(sp.family, sp.pi)
    params = NamedTuple[]
    for t in 1:P
        for (nm, tr, idx) in (("betaz[$t]", bz_true, t), ("betac[$t]", BETAC_TRUE[t], P + t))
            lo, hi, se = W.lower[idx], W.upper[idx], W.se[idx]
            push!(params, (param = nm, truth = tr, est = ad.θ[idx], se = se, lower = lo, upper = hi,
                           covered = isfinite(se) ? Int(lo <= tr <= hi) : -1))
        end
    end
    for (k, nm) in enumerate(LLT_NAMES)
        e = est_llt[k]; se = se_llt[k]; tr = LLT_TRUE[k]
        lo = isfinite(se) ? e - z * se : NaN; hi = isfinite(se) ? e + z * se : NaN
        push!(params, (param = nm, truth = tr, est = e, se = se, lower = lo, upper = hi,
                       covered = isfinite(se) ? Int(lo <= tr <= hi) : -1))
    end
    if sp.family == "ZINB"
        idx = m   # "r" on log scale in θ; Wald interval exp-transformed
        push!(params, (param = "r", truth = R_TRUE, est = exp(ad.θ[idx]), se = W.se[idx],
                       lower = W.lower[idx], upper = W.upper[idx],
                       covered = isfinite(W.se[idx]) ? Int(W.lower[idx] <= R_TRUE <= W.upper[idx]) : -1))
    end
    n_nan_se = count(!isfinite, W.se)
    receipt = (cell = cell, family = sp.family, pi = sp.pi, n = sp.n, p = P, K = K, rep = rep, seed = seed,
               status = "ok", converged = fit.converged, iterations = fit.iterations, logLik = fit.loglik,
               pd_hessian = W.pd_hessian, n_nan_se = n_nan_se, n_boundary_terms = length(W.boundary_terms),
               grad_maxabs = maximum(abs, g), r_hat = sp.family == "ZINB" ? fit.r : NaN,
               prop_zero = count(==(0), Y) / length(Y),
               t_fit_s = t_fit, t_wald_s = t_wald, t_total_s = time() - t0, error = "")
    return receipt, params
end
