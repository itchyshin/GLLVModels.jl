# Sibling screen for the #477 bug class: fit_beta_gllvm_grouped
# (src/families/grouped_dispersion.jl, GLLVModels.jl worktree gllvm-nb2-finite-20260924).
#
# Question: does the public fitter stop BELOW the optimum of its own objective, the way
# fit_nb_gllvm_grouped did before _nb_boundary_restart? A stall is declared only when an
# alternative start reaches a log-likelihood more than 1e-3 above the fitter's, under the
# SAME objective (replicated negll below, validated against fit.loglik).
#
# Replication (read from the source, fit_beta_gllvm_grouped, lines ~748-802):
#   Yc   = _sanitize_missing(Y, 0.5); msk = _resolve_obs_mask(mask, Y)
#   Zemp = logit(clamp(Yc, 1e-6, 1-1e-6)); β0 = row means; Λ0 = SVD(Zemp .- β0) scaled S/sqrt(n)
#   θ0   = [β0; pack_lambda(Λ0); fill(log(10), G)]
#   negll(θ) = -beta_grouped_marginal_loglik_laplace(Yc, Λ, β, φvec; link, mask=msk,
#               offset, hessian=:observed, maxiter=100, tol=1e-9); try/catch and non-finite -> 1e12
#   optimizer: LBFGS(linesearch = BackTracking(order=3)), Options(g_tol=1e-5, iterations=500),
#              autodiff = :finite. NO boundary restart in this fitter.
#
# Design: p = 5, K = 2, n = 80, Λ = 0.30 .* [0.8 0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3],
# β = [-1.5, -0.75, 0, 0.75, 1.5] (μ ≈ 0.18..0.82 on the logit link), z_i ~ N(0, I_2),
# y_ti ~ Beta(μ_ti φ, (1-μ_ti) φ) with φ shared across traits.
#   datasets 1-5 : φ = 2   (near the maximal-variance / near-Bernoulli end: the four outer
#                           traits have a shape parameter < 1, J-shaped; the middle trait is uniform)
#   datasets 6-10: φ = 50  (well inside the parameter space)
# RNG: Random.MersenneTwister(20260924 + d) (StableRNGs is not in the test/parity environment).

using GLLVModels, LinearAlgebra, Random, Printf
import Distributions
const GM = GLLVModels
const Optim = GM.Optim

const p, K, n = 5, 2, 80
const Λtrue = 0.30 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]
const βtrue = [-1.5, -0.75, 0.0, 0.75, 1.5]
const PHI_LEVELS = [2.0, 2.0, 2.0, 2.0, 2.0, 50.0, 50.0, 50.0, 50.0, 50.0]
const YCLAMP = 1e-12

logistic(x) = 1 / (1 + exp(-x))

function simulate(d)
    rng = MersenneTwister(20260924 + d)
    φ = PHI_LEVELS[d]
    Y = Matrix{Float64}(undef, p, n)
    nclamp = 0
    for i in 1:n
        z = randn(rng, K)
        η = βtrue .+ Λtrue * z
        for t in 1:p
            μ = logistic(η[t])
            y = rand(rng, Distributions.Beta(μ * φ, (1 - μ) * φ))
            yc = clamp(y, YCLAMP, 1 - YCLAMP)
            nclamp += (yc != y)
            Y[t, i] = yc
        end
    end
    return Y, φ, nclamp
end

# ---- exact replication of the fitter's pieces (mask = nothing, offset = nothing) ----
function replicate(Y; group = collect(1:size(Y, 1)), link = GM.LogitLink(),
                   hessian = :observed, g_tol = 1e-5, iterations = 500,
                   newton_maxiter = 100, newton_tol = 1e-9)
    p_, n_ = size(Y)
    rr = GM.rr_theta_len(p_, K)
    labels = sort(unique(group)); G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p_]
    msk = GM._resolve_obs_mask(nothing, Y)
    Yc = GM._sanitize_missing(Y, 0.5)
    Zemp = [GM.linkfun(link, clamp(float(Yc[t, i]), 1e-6, 1 - 1e-6)) for t in 1:p_, i in 1:n_]
    GM._mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n_
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p_, K)
    for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n_))
    end
    θ0 = vcat(β0, GM.pack_lambda(Λ0), fill(log(10.0), G))
    function negll(θ)
        β = θ[1:p_]
        Λ = GM.unpack_lambda(θ[(p_ + 1):(p_ + rr)], p_, K)
        φg = exp.(θ[(p_ + rr + 1):(p_ + rr + G)])
        φvec = [φg[gidx[t]] for t in 1:p_]
        v = try
            -GM.beta_grouped_marginal_loglik_laplace(Yc, Λ, β, φvec; link = link, mask = msk,
                                                     offset = nothing, hessian = hessian,
                                                     maxiter = newton_maxiter, tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    return (; θ0, negll, ls, opts, first_logd = p_ + rr + 1, rr, G)
end

theta_of(fit) = vcat(fit.β, GM.pack_lambda(fit.Λ), log.(fit.φ))
fmt(v) = join([@sprintf("%.4g", x) for x in v], ",")

function main()
    ds = parse.(Int, split(get(ENV, "DATASETS", "1,2,3,4,5,6,7,8,9,10"), ","))
    println("# fit_beta_gllvm_grouped sibling screen (#477 bug class)")
    println("# p=$p K=$K n=$n  beta=", βtrue, "  Lambda=0.30*[...]  phi levels 2 (d1-5), 50 (d6-10)")
    println("# stall rule: best_alt_loglik - fitter_loglik > 1e-3 under the same replicated objective")
    rows = []
    t_all = time()
    for d in ds
        Y, φtrue, nclamp = simulate(d)
        R = replicate(Y)
        t0 = time()
        fit = fit_beta_gllvm_grouped(Y; K = K, group = collect(1:p))
        tfit = time() - t0
        θ̂ = theta_of(fit)
        f_at_fit = R.negll(θ̂)
        mismatch = abs(f_at_fit - (-fit.loglik))
        valid = mismatch <= 1e-6
        bd = findall(fit.dispersion_boundary)
        # truth value of the objective (reference only)
        θtrue = vcat(βtrue, GM.pack_lambda(Λtrue), fill(log(φtrue), p))
        ll_true = -R.negll(θtrue)
        # (b) restart from returned point with boundary groups' log-dispersion = 0
        ll_b = NaN; φ_b = Float64[]
        if !isempty(bd)
            θs = copy(θ̂); θs[R.first_logd - 1 .+ bd] .= 0.0
            rb = Optim.optimize(R.negll, θs, R.ls, R.opts; autodiff = :finite)
            ll_b = -Optim.minimum(rb); φ_b = exp.(Optim.minimizer(rb)[R.first_logd:end])
        end
        # (a) fresh start = fitter's θ0 with all log-dispersions = 0
        θa = copy(R.θ0); θa[R.first_logd:end] .= 0.0
        ra = Optim.optimize(R.negll, θa, R.ls, R.opts; autodiff = :finite)
        ll_a = -Optim.minimum(ra); φ_a = exp.(Optim.minimizer(ra)[R.first_logd:end])
        alts = [("a_fresh_logphi0", ll_a)]
        isnan(ll_b) || push!(alts, ("b_restart_boundary0", ll_b))
        ibest = argmax([x[2] for x in alts])
        best_name, best_ll = alts[ibest]
        gain = best_ll - fit.loglik
        stall = valid && gain > 1e-3
        @printf("\n[d%02d] phi_true=%g  clamped_y=%d  fit_time=%.1fs\n", d, φtrue, nclamp, tfit)
        @printf("  fitter: loglik=%.6f converged=%s iters=%d phi_hat=[%s] boundary_groups=%s\n",
                fit.loglik, fit.converged, fit.iterations, fmt(fit.φ), string(bd))
        @printf("  harness: negll(theta_hat)=%.9f  -fit.loglik=%.9f  |diff|=%.3e  valid=%s\n",
                f_at_fit, -fit.loglik, mismatch, valid)
        @printf("  objective at truth: loglik=%.6f\n", ll_true)
        @printf("  (a) fresh theta0 with log phi=0: loglik=%.6f conv=%s iters=%d phi=[%s] boundary=%s\n",
                ll_a, Optim.converged(ra), Optim.iterations(ra), fmt(φ_a),
                string(findall(GM._dispersion_group_boundary(φ_a))))
        if isnan(ll_b)
            println("  (b) not run: no boundary groups")
        else
            @printf("  (b) restart boundary groups at log phi=0: loglik=%.6f phi=[%s]\n", ll_b, fmt(φ_b))
        end
        @printf("  best_alt=%s loglik=%.6f  gain=%.6f  STALL=%s\n", best_name, best_ll, gain, stall)
        push!(rows, (d = d, phi = φtrue, fit_ll = fit.loglik, mismatch = mismatch, valid = valid,
                     boundary = bd, ll_a = ll_a, ll_b = ll_b, best = best_ll, which = best_name,
                     gain = gain, stall = stall))
        flush(stdout)
    end
    println("\n# SUMMARY")
    println("dataset\tphi\tfitter_ll\tmismatch\tvalid\tboundary\tll_a\tll_b\tbest_alt\tgain\tstall")
    for r in rows
        @printf("d%02d\t%g\t%.6f\t%.2e\t%s\t%s\t%.6f\t%s\t%s\t%.6f\t%s\n", r.d, r.phi, r.fit_ll,
                r.mismatch, r.valid, string(r.boundary), r.ll_a,
                isnan(r.ll_b) ? "NA" : @sprintf("%.6f", r.ll_b), r.which, r.gain, r.stall)
    end
    @printf("harness_valid=%s n_with_boundary=%d n_stalls=%d max_gain=%.6g max_mismatch=%.3e total_time=%.1fs\n",
            all(r.valid for r in rows), count(r -> !isempty(r.boundary), rows),
            count(r -> r.stall, rows), maximum(r.gain for r in rows),
            maximum(r.mismatch for r in rows), time() - t_all)
end

main()
