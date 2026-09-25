# Sibling screen for the #477 bug class: fit_gamma_gllvm_grouped.
# Read-only use of GLLVModels; replicates the fitter's negll and optimizer
# (src/families/grouped_dispersion.jl, fit_gamma_gllvm_grouped) exactly.
using GLLVModels, Distributions, Random, LinearAlgebra, Printf, Logging
const GM = GLLVModels
const Optim = GM.Optim

# ---- exact replication of fit_gamma_gllvm_grouped's setup, negll, optimizer ----
function build_problem(Y::AbstractMatrix; K::Integer,
        group::AbstractVector{<:Integer} = collect(1:size(Y, 1)),
        link = GM.LogLink(), mask = nothing, offset = nothing,
        hessian::Symbol = :observed,
        g_tol::Real = 1e-5, iterations::Integer = 500,
        newton_maxiter::Integer = 100, newton_tol::Real = 1e-9)
    p, n = size(Y)
    rr = GM.rr_theta_len(p, K)
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]
    msk = GM._resolve_obs_mask(mask, Y)
    Yc  = GM._sanitize_missing(Y, 1.0)
    Zemp = log.(max.(Yc, 1e-6))
    offset === nothing || (Zemp .-= offset)
    GM._mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(β0, GM.pack_lambda(Λ0), fill(log(2.0), G))
    function negll(θ)
        β = θ[1:p]
        Λ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        αg = exp.(θ[(p + rr + 1):(p + rr + G)])
        αvec = [αg[gidx[t]] for t in 1:p]
        v = try
            -GM.gamma_grouped_marginal_loglik_laplace(Yc, Λ, β, αvec; link = link, mask = msk,
                                                      offset = offset, hessian = hessian,
                                                      maxiter = newton_maxiter,
                                                      tol = newton_tol)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    run(θstart) = Optim.optimize(negll, θstart, ls, opts; autodiff = :finite)
    return (; negll, θ0, run, p, rr, G, first_log = p + rr + 1)
end

theta_of(fit) = vcat(fit.β, GM.pack_lambda(fit.Λ), log.(fit.α))
bnd(θ, first_log) = findall(GM._dispersion_group_boundary(exp.(θ[first_log:end])))

# ---- simulation design ----
const p, K, n = 5, 2, 80
const Λtrue = 0.30 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]
const βtrue = [0.5, 1.0, -0.5, 1.5, 0.0]          # log-mean intercepts: means ~0.6 to 4.5
# Two shape levels, shared across all 5 traits within a dataset:
#   α = 2   (seeds 1-5): moderate overdispersion, far from either limit (CV ≈ 0.71)
#   α = 200 (seeds 6-10): close to the near-deterministic limit (α→∞, Var→0; CV ≈ 0.07),
#            where residual noise is comparable to or smaller than the LV signal
#            (LV sd per trait on the log scale 0.09-0.24), i.e. Heywood / α→∞ territory.
function simulate(seed, α)
    rng = MersenneTwister(seed)
    Z = randn(rng, K, n)
    η = βtrue .+ Λtrue * Z
    μ = exp.(η)
    Y = [rand(rng, Gamma(α, μ[t, i] / α)) for t in 1:p, i in 1:n]
    return Y
end

const TOL_STALL = 1e-3
results = []
t_start = time()
println("fit_gamma_gllvm_grouped sibling screen (#477 class); p=$p K=$K n=$n; group = 1:p")
println("julia ", VERSION, "; threads ", Threads.nthreads())
for ds in 1:10
    αtrue = ds <= 5 ? 2.0 : 200.0
    Y = simulate(ds, αtrue)
    prob = build_problem(Y; K = K, group = collect(1:p))
    t0 = time()
    fit = with_logger(NullLogger()) do
        GM.fit_gamma_gllvm_grouped(Y; K = K, group = collect(1:p))
    end
    tfit = time() - t0
    θf = theta_of(fit)
    obj_at_fit = prob.negll(θf)
    harness_diff = abs(obj_at_fit - (-fit.loglik))
    # extra optimizer-replication check: rerun our copy from θ0
    rep = prob.run(prob.θ0)
    rep_diff = abs(-Optim.minimum(rep) - fit.loglik)
    bd = findall(fit.dispersion_boundary)
    # (b) boundary restart from returned point
    ll_b = NaN; bd_b = Int[]
    if !isempty(bd)
        θs = copy(θf); θs[prob.first_log - 1 .+ bd] .= 0.0
        rb = prob.run(θs)
        ll_b = -Optim.minimum(rb); bd_b = bnd(Optim.minimizer(rb), prob.first_log)
    end
    # (a) fresh start = θ0 with all log-dispersions 0
    θa = copy(prob.θ0); θa[prob.first_log:end] .= 0.0
    ra = prob.run(θa)
    ll_a = -Optim.minimum(ra); bd_a = bnd(Optim.minimizer(ra), prob.first_log)
    alts = [(ll_a, "a_fresh_logalpha0"), (ll_b, "b_boundary_restart")]
    alts = filter(x -> isfinite(x[1]), alts)
    best = alts[argmax(first.(alts))]
    gain = best[1] - fit.loglik
    stall = gain > TOL_STALL
    push!(results, (; ds, αtrue, fit_ll = fit.loglik, α̂ = fit.α, bd, harness_diff, rep_diff,
                    ll_a, bd_a, α̂a = exp.(Optim.minimizer(ra)[prob.first_log:end]),
                    ll_b, bd_b, best_ll = best[1], which = best[2], gain, stall,
                    conv = fit.converged, iters = fit.iterations, tfit))
    @printf("ds %2d  α_true=%6.1f  fit_ll=%.6f  iters=%d conv=%s  α̂=%s  boundary=%s\n",
            ds, αtrue, fit.loglik, fit.iterations, fit.converged,
            string(round.(fit.α; sigdigits = 4)), string(bd))
    @printf("       harness |negll(θ̂)+loglik|=%.3e  replicated-run |Δll|=%.3e\n",
            harness_diff, rep_diff)
    @printf("       (a) ll=%.6f α̂=%s boundary=%s\n", ll_a,
            string(round.(exp.(Optim.minimizer(ra)[prob.first_log:end]); sigdigits = 4)),
            string(bd_a))
    @printf("       (b) ll=%s boundary=%s\n", isnan(ll_b) ? "skipped (no boundary group)" :
            @sprintf("%.6f", ll_b), string(bd_b))
    @printf("       best alt=%s ll=%.6f gain=%.3e  STALL=%s  (fit %.1fs, elapsed %.1fs)\n",
            best[2], best[1], gain, stall, tfit, time() - t_start)
end

println("\n==== SUMMARY ====")
harness_valid = all(r -> r.harness_diff < 1e-6, results)
println("harness_valid (all |negll(θ̂) + fit.loglik| < 1e-6): ", harness_valid,
        "  max diff = ", maximum(r -> r.harness_diff, results))
println("optimizer replication max |Δll| (our run from θ0 vs public fit): ",
        maximum(r -> r.rep_diff, results))
println("n_datasets = ", length(results))
println("n_with_boundary (fitter) = ", count(r -> !isempty(r.bd), results))
println("n_stalls (gain > $TOL_STALL) = ", count(r -> r.stall, results))
println("max_gain = ", maximum(r -> r.gain, results))
for r in results
    @printf("ds %2d α=%5.0f fit_ll=%.4f best_alt=%.4f (%s) gain=%+.3e bd_fit=%s bd_a=%s stall=%s\n",
            r.ds, r.αtrue, r.fit_ll, r.best_ll, r.which, r.gain, string(r.bd), string(r.bd_a), r.stall)
end
println("total elapsed ", round(time() - t_start; digits = 1), " s")
