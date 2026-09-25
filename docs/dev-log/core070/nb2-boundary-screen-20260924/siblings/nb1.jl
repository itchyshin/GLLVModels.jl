# Sibling screen for the #477 bug class: fit_nb1_gllvm_grouped
# (src/families/grouped_dispersion.jl, GLLVModels.jl worktree gllvm-nb2-finite-20260924,
#  HEAD b4cb24df5).
#
# Question: does the public fitter stop BELOW the optimum of its own objective, the way
# fit_nb_gllvm_grouped did before _nb_boundary_restart? A stall is declared only when an
# alternative start reaches a log-likelihood more than 1e-3 above the fitter's, under the
# SAME objective (replicated negll below, validated against fit.loglik).
#
# NB1 parameterisation (src/families/negbin1.jl): y ~ NegativeBinomial(r = μ/φ, p = 1/(1+φ)),
# E[y] = μ, Var = μ(1+φ). φ -> 0 is the Poisson limit; φ -> ∞ is flat overdispersion.
#
# Replication (read from the source, fit_nb1_gllvm_grouped, lines 1555-1610):
#   Yc   = Integer.(_sanitize_missing(Y, 0)); msk = _resolve_obs_mask(mask, Y)
#   Zemp = log(max(Y + 0.5, 1e-4)); β0 = row means; Λ0 = SVD(Zemp .- β0) scaled S/sqrt(n)
#   θ0   = [β0; pack_lambda(Λ0); fill(log(1.0), G)]      <-- NOTE: log φ0 = 0 already
#   negll(θ) = -nb1_grouped_marginal_loglik_laplace(Yc, Λ, β, φvec; link, mask=msk,
#               offset, hessian=:observed, maxiter=100, tol=1e-9); try/catch and non-finite -> 1e12
#   optimizer: LBFGS(linesearch = BackTracking(order=3)), Options(g_tol=1e-5, iterations=500),
#              autodiff = :finite. NO boundary restart in this fitter.
#
# Because the fitter's own θ0 already sets every log φ to 0, the task's alternative (a)
# ("fitter's θ0 with all log-dispersions = 0") is IDENTICAL to the fitter's start: it is a
# deterministic replay, useful only as a replication check. A supplementary start (c),
# θ0 with all log φ = log(10) (the NB2 / Beta siblings' convention), is added so the screen
# still probes a different dispersion start. The stall verdict required by the brief uses
# (a) and (b); (c) is reported separately.
#
# Design: p = 5, K = 2, n = 80, Λ = 0.30 .* [0.8 0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3],
# β = [0.0, 0.5, 1.0, 1.5, 2.0] (μ ≈ 1.0 .. 7.4 counts on the log link), z_i ~ N(0, I_2),
# y_ti ~ NB1(μ_ti, φ) with φ shared across traits.
#   datasets 1-5 : φ = 0.05 (near the Poisson limit: Var = 1.05 μ)
#   datasets 6-10: φ = 2.0  (well inside: Var = 3 μ)
# RNG: Random.MersenneTwister(20260924 + d) (StableRNGs is not in the test/parity environment).

using GLLVModels, LinearAlgebra, Random, Printf
import Distributions
const GM = GLLVModels
const Optim = GM.Optim

const p, K, n = 5, 2, 80
const Λtrue = 0.30 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]
const βtrue = [0.0, 0.5, 1.0, 1.5, 2.0]
const PHI_LEVELS = [0.05, 0.05, 0.05, 0.05, 0.05, 2.0, 2.0, 2.0, 2.0, 2.0]

function simulate(d)
    rng = MersenneTwister(20260924 + d)
    φ = PHI_LEVELS[d]
    Y = Matrix{Int}(undef, p, n)
    for i in 1:n
        z = randn(rng, K)
        η = βtrue .+ Λtrue * z
        for t in 1:p
            μ = exp(η[t])
            Y[t, i] = rand(rng, Distributions.NegativeBinomial(μ / φ, 1 / (1 + φ)))
        end
    end
    return Y, φ
end

# ---- exact replication of the fitter's pieces (mask = nothing, offset = nothing) ----
function replicate(Y; group = collect(1:size(Y, 1)), link = GM.LogLink(),
                   hessian = :observed, g_tol = 1e-5, iterations = 500,
                   newton_maxiter = 100, newton_tol = 1e-9, mask = nothing, offset = nothing)
    p, n = size(Y)
    rr = GM.rr_theta_len(p, K)
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]
    msk = GM._resolve_obs_mask(mask, Y)
    Yc = Integer.(GM._sanitize_missing(Y, 0))
    Zemp = [GM.linkfun(link, max(Yc[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    offset === nothing || (Zemp .-= offset)
    GM._mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    θ0 = vcat(β0, GM.pack_lambda(Λ0), fill(log(1.0), G))
    function negll(θ)
        β = θ[1:p]
        Λ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        φg = exp.(θ[(p + rr + 1):(p + rr + G)])
        φvec = [φg[gidx[t]] for t in 1:p]
        v = try
            -GM.nb1_grouped_marginal_loglik_laplace(Yc, Λ, β, φvec; link = link, mask = msk,
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
    return (; negll, θ0, ls, opts, first_logphi = p + rr + 1, G, rr)
end

runopt(R, θstart) = Optim.optimize(R.negll, θstart, R.ls, R.opts; autodiff = :finite)
phis(R, θ) = exp.(θ[R.first_logphi:end])
fmt(v) = "[" * join([@sprintf("%.4g", x) for x in v], ",") * "]"

const DATASETS = isempty(ARGS) ? collect(1:10) : parse.(Int, ARGS)
const SKIP_A = get(ENV, "SKIP_A_RERUN", "0") == "1"
const SUPP_BUDGET = parse(Float64, get(ENV, "SUPP_BUDGET", "1e9"))   # s; skip (c) after this
const STOP_BUDGET = parse(Float64, get(ENV, "STOP_BUDGET", "1e9"))   # s; start no new dataset after this
rows = []
t_all = time()
for d in DATASETS
    if time() - t_all > STOP_BUDGET
        println("\n[d", lpad(d, 2, '0'), "] NOT RUN: compute cap reached (", round(time() - t_all; digits = 1), "s)")
        continue
    end
    Y, φ = simulate(d)
    R = replicate(Y)

    t0 = time()
    fit = fit_nb1_gllvm_grouped(Y; K = K, group = collect(1:p))
    tfit = time() - t0
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ), log.(fit.φ))
    nll_at_hat = R.negll(θhat)
    mismatch = abs(nll_at_hat - (-fit.loglik))
    valid = mismatch <= 1e-6
    bd = findall(GM._dispersion_group_boundary(fit.φ))

    # objective at the data-generating parameters (context only)
    ll_truth = -R.negll(vcat(βtrue, GM.pack_lambda(Λtrue), fill(log(φ), p)))

    # (a) fresh start: fitter's θ0 with all log φ = 0 (identical to θ0 for this fitter).
    # With SKIP_A_RERUN=1 the (deterministic) replay is not re-run: its start is asserted
    # identical to θ0 and its result is the fitter's (replay verified bit-identical on d01, d06).
    θa = copy(R.θ0); θa[R.first_logphi:end] .= 0.0
    a_is_theta0 = θa == R.θ0
    if SKIP_A && a_is_theta0
        ra = nothing
        ll_a = fit.loglik
        bd_a = bd
    else
        ra = runopt(R, θa)
        ll_a = -Optim.minimum(ra)
        bd_a = findall(GM._dispersion_group_boundary(phis(R, Optim.minimizer(ra))))
    end

    # (b) restart once from the returned point with boundary groups' log φ = 0
    ll_b = NaN; rb = nothing
    if !isempty(bd)
        θb = copy(θhat); θb[R.first_logphi - 1 .+ bd] .= 0.0
        rb = runopt(R, θb)
        ll_b = -Optim.minimum(rb)
    end

    # (c) SUPPLEMENTARY: fitter's θ0 with all log φ = log(10) (sibling convention).
    # Skipped once the loop has used SUPP_BUDGET seconds (compute cap).
    rc = nothing; ll_c = NaN; bd_c = Int[]
    if time() - t_all < SUPP_BUDGET
        θc = copy(R.θ0); θc[R.first_logphi:end] .= log(10.0)
        rc = runopt(R, θc)
        ll_c = -Optim.minimum(rc)
        bd_c = findall(GM._dispersion_group_boundary(phis(R, Optim.minimizer(rc))))
    end

    cands = [("a_fresh_logphi0", ll_a)]
    isnan(ll_b) || push!(cands, ("b_restart_boundary0", ll_b))
    ibest = argmax([c[2] for c in cands])
    best_name, best_ll = cands[ibest]
    gain = best_ll - fit.loglik
    stall = valid && gain > 1e-3
    gain_c = ll_c - fit.loglik
    stall_c = valid && gain_c > 1e-3

    println("\n[d", lpad(d, 2, '0'), "] phi_true=", φ, "  fit_time=", round(tfit; digits = 1), "s",
            "  ybar=", fmt(vec(sum(Y; dims = 2)) ./ n))
    println("  fitter: loglik=", @sprintf("%.6f", fit.loglik), " converged=", fit.converged,
            " iters=", fit.iterations, " phi_hat=", fmt(fit.φ), " boundary_groups=", bd)
    println("  harness: negll(theta_hat)=", @sprintf("%.9f", nll_at_hat), "  -fit.loglik=",
            @sprintf("%.9f", -fit.loglik), "  |diff|=", @sprintf("%.3e", mismatch), "  valid=", valid)
    println("  objective at truth: loglik=", @sprintf("%.6f", ll_truth))
    if ra === nothing
        println("  (a) fresh theta0 with log phi=0 (identical to fitter theta0: ", a_is_theta0,
                "): NOT re-run (deterministic replay of the fitter); loglik=", @sprintf("%.6f", ll_a),
                " boundary=", bd_a)
    else
        println("  (a) fresh theta0 with log phi=0 (identical to fitter theta0: ", a_is_theta0, "): loglik=",
                @sprintf("%.6f", ll_a), " conv=", Optim.converged(ra), " iters=", Optim.iterations(ra),
                " phi=", fmt(phis(R, Optim.minimizer(ra))), " boundary=", bd_a)
    end
    if rb === nothing
        println("  (b) not applicable: no boundary groups")
    else
        println("  (b) restart boundary groups at log phi=0: loglik=", @sprintf("%.6f", ll_b),
                " conv=", Optim.converged(rb), " iters=", Optim.iterations(rb),
                " phi=", fmt(phis(R, Optim.minimizer(rb))))
    end
    if rc === nothing
        println("  (c) SUPPLEMENTARY: skipped (compute cap)")
    else
        println("  (c) SUPPLEMENTARY theta0 with log phi=log(10): loglik=", @sprintf("%.6f", ll_c),
                " conv=", Optim.converged(rc), " iters=", Optim.iterations(rc),
                " phi=", fmt(phis(R, Optim.minimizer(rc))), " boundary=", bd_c,
                "  gain_c=", @sprintf("%.6f", gain_c), "  stall_c=", stall_c)
    end
    println("  best_alt(a,b)=", best_name, " loglik=", @sprintf("%.6f", best_ll),
            "  gain=", @sprintf("%.6f", gain), "  STALL=", stall)
    push!(rows, (; d, φ, fitter_ll = fit.loglik, mismatch, valid, bd, ll_a, ll_b, best_name,
                 best_ll, gain, stall, ll_c, gain_c, stall_c))
    flush(stdout)
end

println("\n# SUMMARY")
println("dataset\tphi\tfitter_ll\tmismatch\tvalid\tboundary\tll_a\tll_b\tbest_alt\tgain\tstall\tll_c_supp\tgain_c\tstall_c")
for r in rows
    println("d", lpad(r.d, 2, '0'), "\t", r.φ, "\t", @sprintf("%.6f", r.fitter_ll), "\t",
            @sprintf("%.2e", r.mismatch), "\t", r.valid, "\t", r.bd, "\t",
            @sprintf("%.6f", r.ll_a), "\t", isnan(r.ll_b) ? "NA" : @sprintf("%.6f", r.ll_b), "\t",
            r.best_name, "\t", @sprintf("%.6f", r.gain), "\t", r.stall, "\t",
            @sprintf("%.6f", r.ll_c), "\t", @sprintf("%.6f", r.gain_c), "\t", r.stall_c)
end
println("harness_valid=", all(r -> r.valid, rows),
        " n_with_boundary=", count(r -> !isempty(r.bd), rows),
        " n_stalls=", count(r -> r.stall, rows),
        " max_gain=", maximum(r -> r.gain, rows),
        " max_mismatch=", @sprintf("%.3e", maximum(r -> r.mismatch, rows)),
        " n_stalls_supp_c=", count(r -> r.stall_c, rows),
        " n_supp_c_run=", count(r -> !isnan(r.gain_c), rows),
        " max_gain_c=", (any(r -> !isnan(r.gain_c), rows) ?
                         maximum(r.gain_c for r in rows if !isnan(r.gain_c)) : NaN),
        " total_time=", round(time() - t_all; digits = 1), "s")
