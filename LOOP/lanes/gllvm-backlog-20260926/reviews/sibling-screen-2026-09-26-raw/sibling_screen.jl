# Sibling screen (2026-09-26): silent inner-search failure / false-converged-flag
# audit for the 8 families NOT covered by docs/dev-log/core070/class-audit-20260924/
# (mixed, beta-binomial, NB1 [per-trait, non-grouped], GP-1, COM-Poisson, ordered
# beta, Student-t, ordinal [per-trait]). Method mirrors that audit's
# fit_verdict_classB_probe.jl: reconstruct each fitter's negll via its OWN public
# *_marginal_loglik_laplace wrapper (same packing/kwargs the fitter itself used),
# then (a) FD-gradient at theta_hat (Class B: converged=true at a non-stationary
# point), (b) bump the inner Newton maxiter/tol far past the fitter's defaults at
# the SAME theta_hat and look for a lower negll (Class A: inner mode search didn't
# reach its own optimum), (c) restart from a perturbed theta_hat and from the true
# generating theta with the SAME optimizer settings the fitter used, and look for
# a materially higher loglik despite converged=true (Class B, global sense).
#
# Run: cd ~/hsq_work/gllvm-sibling-screen-20260926/repo && \
#   JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
#   ~/.juliaup/bin/julia +1.10.12 --startup-file=no --project=test/parity \
#   ~/hsq_work/gllvm-sibling-screen-20260926/sibling_screen.jl

using GLLVModels, LinearAlgebra, Random, Statistics, Distributions, Printf, SpecialFunctions
const GM = GLLVModels
const Optim = GM.Optim
const FD = GM.ForwardDiff
const T0 = time()
elapsed() = time() - T0

maxabs(x) = isempty(x) ? 0.0 : maximum(abs, x)
logistic(x) = 1 / (1 + exp(-x))

function fdgrad(f, θ; h = 1e-5)
    g = similar(θ, Float64)
    for i in eachindex(θ)
        hi = h * max(1.0, abs(θ[i]))
        θp = copy(θ); θp[i] += hi
        θm = copy(θ); θm[i] -= hi
        g[i] = (f(θp) - f(θm)) / (2hi)
    end
    return g
end

# Lower-triangular Λ (positive diagonal) — matches pack_lambda/unpack_lambda's
# identifiability convention (copied from the prior audit's probe_common.jl).
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

bt_ls() = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
fd_run(negll, θ0; g_tol = 1e-5, iterations = 500) =
    Optim.optimize(negll, θ0, bt_ls(), Optim.Options(g_tol = g_tol, iterations = iterations);
                   autodiff = :finite)

# Inverse-CDF samplers for the two families with no `rand` in the package
# (both distributions are documented above the fit function; formulas copied
# from the family file's own logpdf).
function rand_compoisson(rng, η, ν; kmax = 2000)
    u = rand(rng); c = 0.0
    for k in 0:kmax
        c += exp(GM.compoisson_logpdf(k, η, ν))
        c >= u && return k
    end
    return kmax
end

function rand_gp1(rng, η, α)
    μ = exp(clamp(η, -30, 30))
    return GM._rand_gp1(rng, GM.GeneralizedPoisson1(α), μ)
end

function rand_betabinom(rng, μ, N, φ)
    p = clamp(rand(rng, Distributions.Beta(μ * φ, (1 - μ) * φ)), 1e-10, 1 - 1e-10)
    return rand(rng, Distributions.Binomial(N, p))
end

function rand_orderedbeta(rng, η, c0, c1, φ)
    p0 = logistic(c0 - η); p1 = logistic(η - c1)
    u = rand(rng)
    u < p0 && return 0.0
    u > 1 - p1 && return 1.0
    μ = clamp(logistic(η), 1e-10, 1 - 1e-10)
    return rand(rng, Distributions.Beta(μ * φ, (1 - μ) * φ))
end

const ROWS = NamedTuple[]

# perturb θ in the SAME (transformed / unconstrained) packing used by the
# fitter — every constrained nuisance parameter (dispersion, precision, cutpoint
# gap) is already log- or otherwise unconstrained-packed, so an unconstrained
# additive+multiplicative jitter is safe generically.
perturb(rng, θ) = θ .+ (0.3 .* abs.(θ) .+ 0.1) .* randn(rng, length(θ))

function finish_row!(; family, seed, tfit, conv, ll, negll, θhat, θtrue, runner, inner_tight, rng)
    valid = abs(negll(θhat) + ll)               # sanity: reimplementation vs fitter's own loglik
    g1 = fdgrad(negll, θhat)
    gmax = maxabs(g1)
    gscaled = maximum(abs.(g1) .* (1 .+ abs.(θhat)))   # scale-aware gradient norm
    flagB = conv && gscaled > 1e-3
    v_default = negll(θhat)
    v_tight = inner_tight(θhat)
    dinner = v_default - v_tight                # >0: default inner-search settings undershot
    inner_flag = dinner > 0.05
    θp = perturb(rng, θhat)
    rp = try runner(θp) catch; nothing end
    llp = rp === nothing ? -Inf : -Optim.minimum(rp)
    rt = try runner(θtrue) catch; nothing end
    llt = rt === nothing ? -Inf : -Optim.minimum(rt)
    best_restart = max(llp, llt)
    drestart = best_restart - ll
    restart_flag = conv && drestart > 0.1
    row = (family = family, seed = seed, tfit = tfit, conv = conv, ll = ll, valid = valid,
           gmax = gmax, gscaled = gscaled, flagB = flagB,
           dinner = dinner, inner_flag = inner_flag,
           ll_restart_pert = llp, ll_restart_true = llt, drestart = drestart,
           restart_flag = restart_flag)
    push!(ROWS, row)
    @printf("  %-12s seed=%-5d t=%5.2fs conv=%-5s ll=%10.4f valid|Δ|=%.1e | |g|=%.2e gscaled=%.2e flagB=%-5s | dinner=%.2e innerFlag=%-5s | restart Δ=%+.3e flagRestart=%-5s\n",
            family, seed, tfit, conv, ll, valid, gmax, gscaled, flagB, dinner, inner_flag,
            drestart, restart_flag)
    flush(stdout)
    return row
end

# --------------------------------------------------------------------------------
# 1. NB1 (per-trait, non-grouped) — negbin1.jl fit_nb1_gllvm
# --------------------------------------------------------------------------------
function route_negbin1(seed; p = 5, n = 40, K = 2, sd = 0.7, φtrue = 1.2, βlo = 0.5, βhi = 1.5)
    rng = MersenneTwister(seed)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [rand(rng, Distributions.NegativeBinomial(exp(clamp(β[t] + dot(Λ[t, :], Z[:, i]), -30, 30)) / φtrue,
                                                  1 / (1 + φtrue))) for t in 1:p, i in 1:n]
    link = GM.LogLink()
    t0 = time(); fit = fit_nb1_gllvm(Y; K = K); tfit = time() - t0
    rr = GM.rr_theta_len(p, K); hessian = fit.hessian
    mk(mi, tol) = θ -> begin
        β_ = θ[1:p]; Λ_ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K); φ_ = exp(θ[p + rr + 1])
        v = try
            -GM.nb1_marginal_loglik_laplace(Y, Λ_, β_, φ_; link = link, mask = nothing,
                                            offset = nothing, hessian = hessian, maxiter = mi, tol = tol)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    negll = mk(100, 1e-9); inner_tight = mk(1000, 1e-12)
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ), log(fit.φ))
    θtrue = vcat(β, GM.pack_lambda(Λ), log(φtrue))
    runner = θs -> fd_run(negll, θs)
    finish_row!(family = "negbin1", seed = seed, tfit = tfit, conv = fit.converged, ll = fit.loglik,
                negll = negll, θhat = θhat, θtrue = θtrue, runner = runner, inner_tight = inner_tight,
                rng = MersenneTwister(seed + 9000))
end

# --------------------------------------------------------------------------------
# 2. GP-1 — gp1.jl fit_gp1_gllvm (profile over α; reconstruct at the FITTED α)
# --------------------------------------------------------------------------------
function route_gp1(seed; p = 5, n = 40, K = 2, sd = 0.5, αtrue = 0.3, βlo = 0.3, βhi = 1.0)
    rng = MersenneTwister(seed)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [rand_gp1(rng, clamp(β[t] + dot(Λ[t, :], Z[:, i]), -30, 30), αtrue) for t in 1:p, i in 1:n]
    link = GM.LogLink()
    t0 = time(); fit = fit_gp1_gllvm(Y; K = K); tfit = time() - t0
    rr = GM.rr_theta_len(p, K); hessian = fit.hessian; N1 = ones(Int, p, n)
    mk(mi, tol) = θ -> begin
        β_ = θ[1:p]; Λ_ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        v = try
            -GM.marginal_loglik_laplace(GM.GeneralizedPoisson1(fit.α), Y, N1, Λ_, β_, link;
                                        hessian = hessian, mask = nothing, offset = nothing,
                                        maxiter = mi, tol = tol)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    negll = mk(100, 1e-9); inner_tight = mk(1000, 1e-12)
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ))
    θtrue = vcat(β, GM.pack_lambda(Λ))
    runner = θs -> fd_run(negll, θs)
    finish_row!(family = "gp1", seed = seed, tfit = tfit, conv = fit.converged, ll = fit.loglik,
                negll = negll, θhat = θhat, θtrue = θtrue, runner = runner, inner_tight = inner_tight,
                rng = MersenneTwister(seed + 9000))
end

# --------------------------------------------------------------------------------
# 3. COM-Poisson — com_poisson.jl fit_compoisson_gllvm
# --------------------------------------------------------------------------------
function route_compoisson(seed; p = 5, n = 40, K = 2, sd = 0.5, νtrue = 0.6, βlo = 0.5, βhi = 1.2)
    rng = MersenneTwister(seed)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [rand_compoisson(rng, clamp(β[t] + dot(Λ[t, :], Z[:, i]), -30, 30), νtrue) for t in 1:p, i in 1:n]
    link = GM.LogLink()
    t0 = time(); fit = fit_compoisson_gllvm(Y; K = K); tfit = time() - t0
    rr = GM.rr_theta_len(p, K)
    mk(mi, tol) = θ -> begin
        β_ = θ[1:p]; Λ_ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K); ν_ = exp(θ[p + rr + 1])
        v = try
            -GM.compoisson_marginal_loglik_laplace(Y, Λ_, β_, ν_; mask = nothing, link = link,
                                                    maxiter = mi, tol = tol)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    negll = mk(100, 1e-9); inner_tight = mk(1000, 1e-12)
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ), log(fit.ν))
    θtrue = vcat(β, GM.pack_lambda(Λ), log(νtrue))
    runner = θs -> fd_run(negll, θs)
    finish_row!(family = "compoisson", seed = seed, tfit = tfit, conv = fit.converged, ll = fit.loglik,
                negll = negll, θhat = θhat, θtrue = θtrue, runner = runner, inner_tight = inner_tight,
                rng = MersenneTwister(seed + 9000))
end

# --------------------------------------------------------------------------------
# 4. Beta-binomial — beta_binomial.jl fit_beta_binomial_gllvm
# --------------------------------------------------------------------------------
function route_beta_binomial(seed; p = 5, n = 40, K = 2, sd = 0.5, φtrue = 6.0, Ntr = 10,
        βlo = -0.5, βhi = 0.5)
    rng = MersenneTwister(seed)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Nm = fill(Ntr, p, n)
    Y = [rand_betabinom(rng, logistic(β[t] + dot(Λ[t, :], Z[:, i])), Ntr, φtrue) for t in 1:p, i in 1:n]
    link = GM.LogitLink()
    t0 = time(); fit = fit_beta_binomial_gllvm(Y; K = K, N = Nm); tfit = time() - t0
    rr = GM.rr_theta_len(p, K)
    mk(mi, tol) = θ -> begin
        β_ = θ[1:p]; Λ_ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K); φ_ = exp(θ[p + rr + 1])
        v = try
            -GM.betabinomial_marginal_loglik_laplace(Integer.(Y), Nm, Λ_, β_, φ_; mask = nothing,
                                                      link = link, maxiter = mi, tol = tol)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    negll = mk(100, 1e-9); inner_tight = mk(1000, 1e-12)
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ), log(fit.φ))
    θtrue = vcat(β, GM.pack_lambda(Λ), log(φtrue))
    runner = θs -> fd_run(negll, θs)
    finish_row!(family = "beta_binomial", seed = seed, tfit = tfit, conv = fit.converged, ll = fit.loglik,
                negll = negll, θhat = θhat, θtrue = θtrue, runner = runner, inner_tight = inner_tight,
                rng = MersenneTwister(seed + 9000))
end

# --------------------------------------------------------------------------------
# 5. Ordered beta — ordered_beta.jl fit_ordered_beta_gllvm
# --------------------------------------------------------------------------------
function route_ordered_beta(seed; p = 5, n = 40, K = 2, sd = 0.5, c0true = -1.0, c1true = 1.0,
        φtrue = 8.0, βlo = -0.3, βhi = 0.3)
    rng = MersenneTwister(seed)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [rand_orderedbeta(rng, β[t] + dot(Λ[t, :], Z[:, i]), c0true, c1true, φtrue) for t in 1:p, i in 1:n]
    t0 = time(); fit = fit_ordered_beta_gllvm(Y; K = K); tfit = time() - t0
    rr = GM.rr_theta_len(p, K)
    mk(mi, tol) = θ -> begin
        β_ = θ[1:p]; Λ_ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        c0_ = θ[p + rr + 1]; c1_ = c0_ + exp(θ[p + rr + 2]); φ_ = exp(θ[p + rr + 3])
        v = try
            -GM.ordered_beta_marginal_loglik_laplace(Y, Λ_, β_, c0_, c1_, φ_; mask = nothing,
                                                      maxiter = mi, tol = tol)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    negll = mk(100, 1e-9); inner_tight = mk(1000, 1e-12)
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ), fit.c0, log(fit.c1 - fit.c0), log(fit.φ))
    θtrue = vcat(β, GM.pack_lambda(Λ), c0true, log(c1true - c0true), log(φtrue))
    runner = θs -> fd_run(negll, θs)
    finish_row!(family = "ordered_beta", seed = seed, tfit = tfit, conv = fit.converged, ll = fit.loglik,
                negll = negll, θhat = θhat, θtrue = θtrue, runner = runner, inner_tight = inner_tight,
                rng = MersenneTwister(seed + 9000))
end

# --------------------------------------------------------------------------------
# 6. Student-t (shared dispersion, ν estimated) — studentt.jl fit_studentt_gllvm
# --------------------------------------------------------------------------------
function route_studentt(seed; p = 5, n = 40, K = 2, sd = 0.6, νtrue = 5.0, σtrue = 0.7,
        βlo = -0.5, βhi = 0.5)
    rng = MersenneTwister(seed)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [β[t] + dot(Λ[t, :], Z[:, i]) + σtrue * rand(rng, Distributions.TDist(νtrue))
         for t in 1:p, i in 1:n]
    link = GM.IdentityLink()
    t0 = time(); fit = fit_studentt_gllvm(Y; K = K); tfit = time() - t0
    rr = GM.rr_theta_len(p, K); hessian = fit.hessian
    mk(mi, tol) = θ -> begin
        β_ = θ[1:p]; Λ_ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        σ_ = exp(θ[p + rr + 1]); ν_ = 1.0 + exp(θ[p + rr + 2])
        v = try
            -GM.studentt_marginal_loglik_laplace(Y, Λ_, β_, σ_; ν = ν_, link = link,
                                                 hessian = hessian, maxiter = mi, tol = tol)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    negll = mk(100, 1e-9); inner_tight = mk(1000, 1e-12)
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ), log(fit.σ), log(fit.ν - 1.0))
    θtrue = vcat(β, GM.pack_lambda(Λ), log(σtrue), log(νtrue - 1.0))
    runner = θs -> fd_run(negll, θs)
    finish_row!(family = "studentt", seed = seed, tfit = tfit, conv = fit.converged, ll = fit.loglik,
                negll = negll, θhat = θhat, θtrue = θtrue, runner = runner, inner_tight = inner_tight,
                rng = MersenneTwister(seed + 9000))
end

# --------------------------------------------------------------------------------
# 7. Ordinal (per-trait cutpoints) — ordinal.jl fit_ordinal_gllvm_pertrait
# --------------------------------------------------------------------------------
function route_ordinal(seed; p = 5, n = 40, K = 2, sd = 0.7, Ccat = 4, gap = 1.0)
    rng = MersenneTwister(seed)
    Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n); β = 0.3 .* randn(rng, p)
    τtrue = [gap * (c - 1) for c in 1:(Ccat - 1)]
    Y = [begin
            η = β[t] + dot(Λ[t, :], Z[:, i]); u = rand(rng); c = Ccat
            for k in 1:(Ccat - 1)
                if u <= logistic(τtrue[k] - η)
                    c = k; break
                end
            end
            c
         end for t in 1:p, i in 1:n]
    link = GM.LogitLink()
    t0 = time(); fit = fit_ordinal_gllvm_pertrait(Y; K = K, link = link); tfit = time() - t0
    rr = GM.rr_theta_len(p, K); C = fit.C; ncut = sum(C .- 2)
    mk(mi, tol) = θ -> begin
        β_ = θ[1:p]; Λ_ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        τ_ = GM._unpack_cutpoints_pertrait(θ[(p + rr + 1):(p + rr + ncut)], C)
        v = try
            -GM.ordinal_marginal_loglik_laplace_pertrait(Y, Λ_, β_, τ_, C; link = link,
                                                          mask = nothing, maxiter = mi, tol = tol)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    negll = mk(100, 1e-9); inner_tight = mk(1000, 1e-12)
    ψtruth = Float64[]
    for tt in 1:p, c in 2:(C[tt] - 1)
        push!(ψtruth, log(gap))
    end
    # τ[t,1] is pinned to 0 by _unpack_cutpoints_pertrait; ψ[t,c] = log(τ[t,c]-τ[t,c-1]),
    # c = 2..C[t]-1 (matches _pack_initial_ordinal_pertrait's own packing).
    ψhat = Float64[]
    for tt in 1:p
        prev = 0.0
        for c in 2:(C[tt] - 1)
            push!(ψhat, log(max(fit.τ[tt, c] - prev, 1e-9)))
            prev = fit.τ[tt, c]
        end
    end
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ), ψhat)
    θtrue = vcat(β, GM.pack_lambda(Λ), ψtruth)
    runner = θs -> fd_run(negll, θs)
    finish_row!(family = "ordinal", seed = seed, tfit = tfit, conv = fit.converged, ll = fit.loglik,
                negll = negll, θhat = θhat, θtrue = θtrue, runner = runner, inner_tight = inner_tight,
                rng = MersenneTwister(seed + 9000))
end

# --------------------------------------------------------------------------------
# 8. Mixed-family — mixed.jl fit_mixed_gllvm (Poisson, Binomial, Gamma, Beta traits)
# --------------------------------------------------------------------------------
function route_mixed(seed; n = 40, K = 2, sd = 0.5, βlo = -0.3, βhi = 0.3, gam_k = 4.0, beta_phi = 6.0)
    rng = MersenneTwister(seed)
    families = Any[Poisson(), Binomial(), Gamma(), Beta()]
    p = length(families)
    links = [GM.default_link(f) for f in families]
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Nm = ones(Int, p, n)
    Y = Matrix{Float64}(undef, p, n)
    for i in 1:n
        for t in 1:p
            η = clamp(β[t] + dot(Λ[t, :], Z[:, i]), -20, 20)
            μ = GM.linkinv(links[t], η)
            fam = families[t]
            Y[t, i] = if fam isa Poisson
                float(rand(rng, Poisson(max(μ, 1e-8))))
            elseif fam isa Binomial
                float(rand(rng, Bernoulli(clamp(μ, 1e-8, 1 - 1e-8))))
            elseif fam isa Gamma
                rand(rng, Distributions.Gamma(gam_k, max(μ, 1e-8) / gam_k))
            else # Beta
                μc = clamp(μ, 1e-8, 1 - 1e-8)
                rand(rng, Distributions.Beta(μc * beta_phi, (1 - μc) * beta_phi))
            end
        end
    end
    t0 = time(); fit = fit_mixed_gllvm(Y; families = families, K = K, N = Nm); tfit = time() - t0
    rr = GM.rr_theta_len(p, K)
    disp_index, n_disp = GM._mixed_family_layout(families)
    mk(mi, tol) = θ -> begin
        v = try
            -GM._mixed_marginal_loglik_packed(θ, Y, Nm, p, K, families, links, disp_index;
                                              maxiter = mi, tol = tol)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    negll = mk(100, 1e-9); inner_tight = mk(1000, 1e-12)
    logdisp0 = zeros(Float64, n_disp)
    for t in 1:p
        disp_index[t] > 0 && (logdisp0[disp_index[t]] = isnan(fit.dispersion[t]) ? 0.0 : log(fit.dispersion[t]))
    end
    logdisptrue = zeros(Float64, n_disp)
    for t in 1:p
        disp_index[t] > 0 && (logdisptrue[disp_index[t]] = t == 3 ? log(gam_k) : log(beta_phi))
    end
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ), logdisp0)
    θtrue = vcat(β, GM.pack_lambda(Λ), logdisptrue)
    runner = θs -> fd_run(negll, θs)
    finish_row!(family = "mixed", seed = seed, tfit = tfit, conv = fit.converged, ll = fit.loglik,
                negll = negll, θhat = θhat, θtrue = θtrue, runner = runner, inner_tight = inner_tight,
                rng = MersenneTwister(seed + 9000))
end

# --------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------
const ONLY = get(ENV, "SCREEN_ONLY", "")
const NDATA = parse(Int, get(ENV, "SCREEN_N", "10"))
const SEEDS = 2001:(2000 + NDATA)

const ROUTES = [("negbin1", route_negbin1), ("gp1", route_gp1), ("compoisson", route_compoisson),
                ("beta_binomial", route_beta_binomial), ("ordered_beta", route_ordered_beta),
                ("studentt", route_studentt), ("ordinal", route_ordinal), ("mixed", route_mixed)]

println("Optim ", pkgversion(Optim), "  GLLVModels at ", pathof(GLLVModels))
println("seeds ", first(SEEDS), ":", last(SEEDS), " (n=", NDATA, ")")
for (rname, rf) in ROUTES
    (isempty(ONLY) || occursin(ONLY, rname)) || continue
    println("--- ", rname, " ---")
    for seed in SEEDS
        try
            rf(seed)
        catch e
            @printf("  %-12s seed=%-5d ERROR %s\n", rname, seed, sprint(showerror, e)[1:min(end, 400)])
        end
    end
end

println("\n==== SUMMARY by family ====")
for r in unique(getfield.(ROWS, :family))
    rows = filter(x -> x.family == r, ROWS)
    nfit = length(rows)
    nconv = count(x -> x.conv, rows)
    nB = count(x -> x.flagB, rows)
    nInner = count(x -> x.inner_flag, rows)
    nRestart = count(x -> x.restart_flag, rows)
    maxvalid = isempty(rows) ? NaN : maximum(x.valid for x in rows)
    maxgap_restart = isempty(rows) ? NaN : maximum(x.drestart for x in rows)
    maxgap_inner = isempty(rows) ? NaN : maximum(x.dinner for x in rows)
    @printf("%-14s fits=%d conv=%d | flagB(scaled|g|>1e-3)=%d | innerFlag(Δinner>0.05)=%d (max %.3g) | restartFlag(Δ>0.1)=%d (max %.3g) | max valid|Δ|=%.1e\n",
            r, nfit, nconv, nB, nInner, maxgap_inner, nRestart, maxgap_restart, maxvalid)
end
@printf("total elapsed %.1f s\n", elapsed())
