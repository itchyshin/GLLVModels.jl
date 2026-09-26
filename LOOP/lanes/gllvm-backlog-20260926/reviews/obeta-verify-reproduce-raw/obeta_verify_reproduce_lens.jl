# Independent reproduction + optimum-validity check for the ordered-beta
# sibling-screen finding (2026-09-26). Reproduces 3 flagged seeds (2003, 2005,
# 2010) and 2 clean seeds (2001, 2008) from sibling_screen.jl's route_ordered_beta
# EXACTLY (same RNG stream, same DGP, same fitter call), then for each:
#   - recomputes the loglik at theta_hat via the package's OWN public
#     ordered_beta_marginal_loglik_laplace wrapper called directly (independent
#     of fit.loglik, which comes from Optim.minimum/_fit_verdict)
#   - checks parameter validity (cutpoint order, finite precision, bounded loadings)
#   - computes a central-FD gradient at theta_hat AND at the best restart point
#   - refits from the TRUE generating parameters with the SAME optimizer config
#     the fitter itself uses, and compares that result to both the first fit and
#     the restart.
using GLLVModels, LinearAlgebra, Random, Statistics, Distributions, Printf
const GM = GLLVModels
const Optim = GM.Optim
const T0 = time()

logistic(x) = 1 / (1 + exp(-x))

function lowtri(rng, p, K, sd)
    Λ = sd .* randn(rng, p, K)
    for j in 1:K, i in 1:(j - 1)
        Λ[i, j] = 0.0
    end
    for j in 1:K
        Λ[j, j] = abs(Λ[j, j]) + 0.3 * sd
    end
    return Λ
end

function rand_orderedbeta(rng, η, c0, c1, φ)
    p0 = logistic(c0 - η); p1 = logistic(η - c1)
    u = rand(rng)
    u < p0 && return 0.0
    u > 1 - p1 && return 1.0
    μ = clamp(logistic(η), 1e-10, 1 - 1e-10)
    return rand(rng, Distributions.Beta(μ * φ, (1 - μ) * φ))
end

function fdgrad(f, θ; h = 1e-6)
    g = similar(θ, Float64)
    for i in eachindex(θ)
        hi = h * max(1.0, abs(θ[i]))
        θp = copy(θ); θp[i] += hi
        θm = copy(θ); θm[i] -= hi
        g[i] = (f(θp) - f(θm)) / (2hi)
    end
    return g
end

bt_ls() = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
fd_run(negll, θ0; g_tol = 1e-5, iterations = 500) =
    Optim.optimize(negll, θ0, bt_ls(), Optim.Options(g_tol = g_tol, iterations = iterations);
                   autodiff = :finite)

perturb(rng, θ) = θ .+ (0.3 .* abs.(θ) .+ 0.1) .* randn(rng, length(θ))

function unpack_theta(θ, p, K, rr)
    β = θ[1:p]
    Λ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
    c0 = θ[p + rr + 1]
    c1 = c0 + exp(θ[p + rr + 2])
    φ = exp(θ[p + rr + 3])
    return β, Λ, c0, c1, φ
end

function valid_params(θ, p, K, rr)
    β, Λ, c0, c1, φ = unpack_theta(θ, p, K, rr)
    finite_ok = all(isfinite, β) && all(isfinite, Λ) && isfinite(c0) && isfinite(c1) && isfinite(φ)
    order_ok = finite_ok && (c1 > c0)
    prec_ok = finite_ok && (φ > 0)
    bound_ok = finite_ok && (maximum(abs, Λ) < 1e6) && (maximum(abs, β) < 1e6)
    ok = finite_ok && order_ok && prec_ok && bound_ok
    return ok, (finite_ok ? c1 - c0 : NaN), (finite_ok ? φ : NaN), (finite_ok ? maximum(abs, Λ) : NaN)
end

function verify_seed(seed; p = 5, n = 40, K = 2, sd = 0.5, c0true = -1.0, c1true = 1.0,
        φtrue = 8.0, βlo = -0.3, βhi = 0.3)
    rng = MersenneTwister(seed)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [rand_orderedbeta(rng, β[t] + dot(Λ[t, :], Z[:, i]), c0true, c1true, φtrue) for t in 1:p, i in 1:n]

    t0 = time(); fit = fit_ordered_beta_gllvm(Y; K = K); tfit = time() - t0
    rr = GM.rr_theta_len(p, K)
    negll = θ -> begin
        β_, Λ_, c0_, c1_, φ_ = unpack_theta(θ, p, K, rr)
        v = try
            -GM.ordered_beta_marginal_loglik_laplace(Y, Λ_, β_, c0_, c1_, φ_;
                                                      mask = nothing, maxiter = 100, tol = 1e-9)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ), fit.c0, log(fit.c1 - fit.c0), log(fit.φ))
    θtrue = vcat(β, GM.pack_lambda(Λ), c0true, log(c1true - c0true), log(φtrue))

    # --- independent recompute of loglik at theta_hat (NOT fit.loglik) ---
    negll_hat = negll(θhat)
    valid_delta_hat = abs(negll_hat + fit.loglik)   # should be ~0 if fit.loglik is honest

    okhat, gaphat, φhat_, maxabs_hat = valid_params(θhat, p, K, rr)
    ghat = fdgrad(negll, θhat)
    gscaled_hat = maximum(abs.(ghat) .* (1 .+ abs.(θhat)))

    # --- restart from perturbed theta_hat and from the TRUE generating params,
    #     same optimizer config the fitter used ---
    rngp = MersenneTwister(seed + 9000)
    θp0 = perturb(rngp, θhat)
    rp = try fd_run(negll, θp0) catch e; nothing end
    rt = try fd_run(negll, θtrue) catch e; nothing end
    llp = rp === nothing ? -Inf : -Optim.minimum(rp)
    llt = rt === nothing ? -Inf : -Optim.minimum(rt)
    θp_min = rp === nothing ? nothing : Optim.minimizer(rp)
    θt_min = rt === nothing ? nothing : Optim.minimizer(rt)
    best_is_true = llt >= llp
    θbest = best_is_true ? θt_min : θp_min
    llbest = max(llp, llt)
    drestart = llbest - fit.loglik

    okbest, gapbest, φbest_, maxabs_best = θbest === nothing ? (false, NaN, NaN, NaN) : valid_params(θbest, p, K, rr)
    gbest = θbest === nothing ? fill(NaN, length(θhat)) : fdgrad(negll, θbest)
    gscaled_best = θbest === nothing ? NaN : maximum(abs.(gbest) .* (1 .+ abs.(θbest)))
    negll_best_direct = θbest === nothing ? NaN : negll(θbest)
    valid_delta_best = θbest === nothing ? NaN : abs(negll_best_direct + llbest)

    # --- refit from the TRUE DGP parameters (== rt above): which solution does it reach? ---
    ok_truestart, _, _, _ = θt_min === nothing ? (false, NaN, NaN, NaN) : valid_params(θt_min, p, K, rr)
    g_truestart = θt_min === nothing ? fill(NaN, length(θhat)) : fdgrad(negll, θt_min)
    gscaled_truestart = θt_min === nothing ? NaN : maximum(abs.(g_truestart) .* (1 .+ abs.(θt_min)))
    dist_to_firstfit = abs(llt - fit.loglik)
    dist_to_restart = abs(llt - llbest)
    closer_to = dist_to_firstfit <= dist_to_restart ? "first_fit" : "restart"

    @printf("\nseed=%d  conv=%s  tfit=%.2fs\n", seed, fit.converged, tfit)
    @printf("  first fit:      ll=%.4f  valid|negll(that)-(-ll)|=%.2e  paramsOK=%s (gap=%.4g phi=%.4g maxL=%.4g)\n",
            fit.loglik, valid_delta_hat, okhat, gaphat, φhat_, maxabs_hat)
    @printf("  first-fit grad: gmax=%.3e gscaled=%.3e  (stationary iff gscaled ~1e-5 or below)\n",
            maxabs(ghat), gscaled_hat)
    @printf("  restart best:   source=%s ll=%.4f  drestart(vs first fit)=%+.4e\n",
            best_is_true ? "true-start" : "perturbed-start", llbest, drestart)
    @printf("  restart grad:   gmax=%.3e gscaled=%.3e  paramsOK=%s (gap=%.4g phi=%.4g maxL=%.4g)  valid|negll-(-ll)|=%.2e\n",
            maxabs(gbest), gscaled_best, okbest, gapbest, φbest_, maxabs_best, valid_delta_best)
    @printf("  true-start refit: ll=%.4f  gscaled=%.3e paramsOK=%s | |ll-firstfit|=%.4e |ll-restart|=%.4e -> closer_to=%s\n",
            llt, gscaled_truestart, ok_truestart, dist_to_firstfit, dist_to_restart, closer_to)
    flush(stdout)

    return (seed = seed, conv = fit.converged, ll_first = fit.loglik, valid_delta_hat = valid_delta_hat,
            okhat = okhat, gscaled_hat = gscaled_hat,
            ll_restart = llbest, drestart = drestart, gscaled_restart = gscaled_best, okbest = okbest,
            ll_truestart = llt, gscaled_truestart = gscaled_truestart, ok_truestart = ok_truestart,
            dist_to_firstfit = dist_to_firstfit, dist_to_restart = dist_to_restart, closer_to = closer_to)
end

maxabs(x) = isempty(x) ? 0.0 : maximum(abs, x)

println("Optim ", pkgversion(Optim), "  GLLVModels at ", pathof(GLLVModels))

FLAGGED_SEEDS = [2003, 2005, 2010]
CLEAN_SEEDS = [2001, 2008]

println("\n==== FLAGGED SEEDS ====")
results_flagged = [verify_seed(s) for s in FLAGGED_SEEDS]

println("\n==== CLEAN SEEDS ====")
results_clean = [verify_seed(s) for s in CLEAN_SEEDS]

@printf("\ntotal elapsed %.1f s\n", time() - T0)
