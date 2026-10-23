# Sibling screen for the #477 bug class: fit_nb_gllvm_grouped_cov
# (src/families/grouped_dispersion.jl, GLLVModels.jl worktree gllvm-nb2-finite-20260924,
#  HEAD b4cb24df5 + uncommitted edits at screen time; those edits later committed as 025df3fca. fit_nb_gllvm_grouped_cov itself unchanged).
#
# Question: does the public fitter stop BELOW the optimum of its own objective, the way
# fit_nb_gllvm_grouped did before _nb_boundary_restart? A stall is declared only when an
# alternative start reaches a log-likelihood more than 1e-3 above the fitter's, under the
# SAME objective (replicated negll below, validated against fit.loglik to 1e-6).
#
# NB2 parameterisation: y ~ NegativeBinomial(r, r/(r+μ)), E[y] = μ, Var = μ + μ²/r.
# r -> ∞ is the Poisson limit (upper boundary 1e6); r -> 0 is extreme overdispersion.
#
# Replication (read from the source, fit_nb_gllvm_grouped_cov, lines 506-570):
#   X_fit = X (γ_fixed = nothing -> no column dropped), q = 1, rr = rr_theta_len(p, K)
#   Yc   = Integer.(_sanitize_missing(Y, 0)); msk = _resolve_obs_mask(nothing, Y)
#   Zemp = log(max(Y + 0.5, 1e-4)); _mask_warmstart!; β0 = row means;
#   Λ0 = SVD(Zemp .- β0), columns U[:,j] * S[j]/sqrt(n)
#   θ0   = [β0; zeros(q); pack_lambda(Λ0); fill(log(10.0), G)]
#   negll(θ) = -nb_grouped_marginal_loglik_laplace(Yc, Λ, β, rvec; link = LogLink(), mask = msk,
#               offset = _build_offset(X_fit, γ), hessian = :observed, maxiter = 100, tol = 1e-9);
#               try/catch -> 1e12; non-finite -> 1e12
#   optimizer: LBFGS(linesearch = BackTracking(order = 3)),
#              Options(g_tol = 1e-5, iterations = 500), autodiff = :finite.
#   NO _nb_boundary_restart call in this fitter (contrast fit_nb_gllvm_grouped, line 412).
#
# Design: p = 5, K = 2, n = 80, Λ = 0.30 .* [0.8 0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3],
# β = [0.0, 0.5, 1.0, 1.5, 2.0] (μ ≈ 1.0 .. 7.4 counts at x = 0 on the log link),
# one shared site covariate x_s ~ N(0,1) with slope γ = 0.3 (X[t,s,1] = x[s]),
# z_s ~ N(0, I_2), y_ts ~ NB2(μ_ts, r) with r shared across traits.
#   datasets 1-5 : r = 50  (near the Poisson limit: at μ = 1 the excess variance μ²/r is 2%;
#                           at μ = 7.4 it is 15% -- low-mean traits look Poisson)
#   datasets 6-10: r = 2   (well inside: Var = μ + μ²/2)
# RNG: Random.MersenneTwister(20260924 + d) (StableRNGs is not in the test/parity environment).
# Draw order per dataset: x (n), then per site z (K) followed by y (p).
#
# Alternatives, both run through the replicated objective and optimizer:
#   (b) restart from the fitter's returned θ̂ with the boundary groups' log r set to 0
#       (only when at least one group is at the boundary; otherwise N/A)
#   (a) fresh start = the fitter's θ0 with every log r = 0 (instead of log 10)
# stall := max(alt loglik) > fit.loglik + 1e-3.

using GLLVModels, LinearAlgebra, Random, Printf
import Distributions
const GM = GLLVModels
const Optim = GM.Optim

const p, K, n = 5, 2, 80
const Λtrue = 0.30 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]
const βtrue = [0.0, 0.5, 1.0, 1.5, 2.0]
const γtrue = 0.3
const R_LEVELS = [50.0, 50.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0, 2.0]
const G_TOL, ITERS, NEWTON_MAXITER, NEWTON_TOL = 1e-5, 500, 100, 1e-9

function simulate(d)
    rng = MersenneTwister(20260924 + d)
    r = R_LEVELS[d]
    x = randn(rng, n)
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
        z = randn(rng, K)
        η = βtrue .+ γtrue * x[s] .+ Λtrue * z
        for t in 1:p
            μ = exp(η[t])
            Y[t, s] = rand(rng, Distributions.NegativeBinomial(r, r / (r + μ)))
        end
    end
    X = Array{Float64}(undef, p, n, 1)
    for t in 1:p, s in 1:n
        X[t, s, 1] = x[s]
    end
    return Y, X, r
end

# Exact replica of the fitter's setup + negll closure (group = 1:p, γ_fixed = nothing,
# link = LogLink(), mask = nothing, hessian = :observed, defaults for the rest).
function build_problem(Y, X; group = collect(1:size(Y, 1)))
    link = GM.LogLink()
    p_, n_ = size(Y)
    γ_fixed_mask = GM._fixed_zero_mask(nothing, size(X, 3), "γ_fixed")
    X_fit, _ = GM._slice_fixed_X(X, γ_fixed_mask)
    q = size(X_fit, 3)
    rr = GM.rr_theta_len(p_, K)
    labels = sort(unique(group))
    G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p_]
    msk = GM._resolve_obs_mask(nothing, Y)
    Yc = Integer.(GM._sanitize_missing(Y, 0))
    Zemp = [GM.linkfun(link, max(Yc[t, i] + 0.5, 1e-4)) for t in 1:p_, i in 1:n_]
    GM._mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n_
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p_, K)
    for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n_))
    end
    θ0 = vcat(β0, zeros(q), GM.pack_lambda(Λ0), fill(log(10.0), G))
    function negll(θ)
        β = θ[1:p_]
        γ = θ[(p_ + 1):(p_ + q)]
        Λ = GM.unpack_lambda(θ[(p_ + q + 1):(p_ + q + rr)], p_, K)
        rg = exp.(θ[(p_ + q + rr + 1):(p_ + q + rr + G)])
        rvec = [rg[gidx[t]] for t in 1:p_]
        O = GM._build_offset(X_fit, γ)
        v = try
            -GM.nb_grouped_marginal_loglik_laplace(Yc, Λ, β, rvec; link = link, mask = msk,
                                                   offset = O, hessian = :observed,
                                                   maxiter = NEWTON_MAXITER, tol = NEWTON_TOL)
        catch
            return 1e12
        end
        return isfinite(v) ? v : 1e12
    end
    first_log_r = p_ + q + rr + 1
    return (; negll, θ0, q, rr, G, first_log_r)
end

run_opt(negll, θs) = Optim.optimize(negll, θs,
    Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)),
    Optim.Options(g_tol = G_TOL, iterations = ITERS); autodiff = :finite)

theta_of_fit(fit) = vcat(fit.β, fit.γ[.!fit.γ_fixed], GM.pack_lambda(fit.Λ), log.(fit.r_group))

fmt(v) = join([@sprintf("%.4g", x) for x in v], ", ")

function main(ds)
    println("fit_nb_gllvm_grouped_cov sibling screen  (Julia ", VERSION, ", threads ",
            Threads.nthreads(), ")")
    println("p=$p K=$K n=$n  β=$(βtrue)  γ=$(γtrue)  r levels: d1-5 r=50, d6-10 r=2")
    rows = []
    valid = true
    for d in ds
        Y, X, rtrue = simulate(d)
        pr = build_problem(Y, X)
        t_fit = @elapsed fit = GM.fit_nb_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
        θf = theta_of_fit(fit)
        rep = -pr.negll(θf)
        diff = abs(rep - fit.loglik)
        ok = diff <= 1e-6
        valid &= ok
        bd = findall(fit.dispersion_boundary)
        # (b) restart from θ̂ with boundary groups' log r = 0
        llb = NaN; bdb = Int[]
        if !isempty(bd)
            θb = copy(θf); θb[pr.first_log_r - 1 .+ bd] .= 0.0
            rb = run_opt(pr.negll, θb)
            llb = -Optim.minimum(rb)
            bdb = findall(GM._dispersion_group_boundary(exp.(Optim.minimizer(rb)[pr.first_log_r:end])))
        end
        # (a) fresh start: θ0 with all log r = 0
        θa = copy(pr.θ0); θa[pr.first_log_r:end] .= 0.0
        t_a = @elapsed ra = run_opt(pr.negll, θa)
        lla = -Optim.minimum(ra)
        rga = exp.(Optim.minimizer(ra)[pr.first_log_r:end])
        bda = findall(GM._dispersion_group_boundary(rga))
        best, which = isnan(llb) || lla >= llb ? (lla, "a") : (llb, "b")
        gain = best - fit.loglik
        stall = ok && gain > 1e-3
        push!(rows, (; d, rtrue, ll = fit.loglik, conv = fit.converged, iters = fit.iterations,
                     bd, llb, bdb, lla, bda, best, which, gain, stall, ok))
        @printf("\n[d=%2d r_true=%g] fit: loglik=%.6f conv=%s iters=%d time=%.1fs\n",
                d, rtrue, fit.loglik, fit.converged, fit.iterations, t_fit)
        @printf("   harness check: replicated -negll(θ̂)=%.9f |diff|=%.2e %s\n",
                rep, diff, ok ? "OK" : "MISMATCH")
        println("   fit r_group = [", fmt(fit.r_group), "]  γ̂ = ", fmt(fit.γ),
                "  boundary groups = ", bd)
        if isnan(llb)
            println("   (b) N/A (no boundary group)")
        else
            @printf("   (b) restart boundary log r=0: loglik=%.6f  Δ=%+.6f  boundary after=%s\n",
                    llb, llb - fit.loglik, string(bdb))
        end
        @printf("   (a) fresh start log r=0:     loglik=%.6f  Δ=%+.6f  conv=%s  time=%.1fs  r=[%s]  boundary=%s\n",
                lla, lla - fit.loglik, Optim.converged(ra), t_a, fmt(rga), string(bda))
        @printf("   best alt (%s) gain = %+.6f  -> %s\n", which, gain,
                stall ? "STALL" : "no stall")
    end
    println("\n==== SUMMARY ====")
    println("harness_valid = ", valid)
    println("n_datasets = ", length(rows))
    println("n_with_boundary = ", count(r -> !isempty(r.bd), rows))
    println("n_stalls = ", count(r -> r.stall, rows))
    println("max_gain = ", maximum(r -> r.gain, rows))
    println("d | r_true | fit_ll | boundary | ll_b | ll_a | best(which) | gain | stall")
    for r in rows
        @printf("%2d | %4g | %.4f | %s | %s | %.4f | %.4f(%s) | %+.2e | %s\n",
                r.d, r.rtrue, r.ll, string(r.bd), isnan(r.llb) ? "NA" : @sprintf("%.4f", r.llb),
                r.lla, r.best, r.which, r.gain, r.stall)
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__   # run only when executed directly (not when included)
    ds = isempty(ARGS) ? collect(1:10) : parse.(Int, ARGS)
    main(ds)
end
