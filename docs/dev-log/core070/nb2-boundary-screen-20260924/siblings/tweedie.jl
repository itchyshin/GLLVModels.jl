# Sibling screen for the #477 bug class: fit_tweedie_gllvm_grouped.
# Read-only against the repository; writes only under /tmp/claude-503/sibling-screen/.
#
# Usage: julia ... tweedie.jl [probe | full | budget <seconds>]
#
# Design (stated choices):
#   p = 5 traits, K = 2, n = 80 sites, log link.
#   Intercepts β = [-2, -1, 0, 1, 2]  (μ ≈ 0.14 … 7.4, the natural biomass range).
#   Loadings Λ = 0.30 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3].
#   True power 1.5, shared across traits (the fitter's default power_group = :shared).
#   True dispersion φ shared across the 5 traits within a dataset, at two levels:
#     datasets 1-5: φ = 0.05  (near the φ -> 0 near-deterministic / Heywood limit:
#                              CV ≈ 0.2 at μ = 1, no zeros at all);
#     datasets 6-10: φ = 5.0  (the low-mean traits sit near the all-zeros limit:
#                              P(y = 0) ≈ 0.86 for trait 1, ≈ 0.34 for trait 5).
#   Seed for dataset d: MersenneTwister(4770 + d). StableRNGs is not in test/parity.
#   Fitted with per-trait groups: group = collect(1:p), every other argument default.

const T_START = time()
using GLLVModels, Random, Distributions, LinearAlgebra, Printf
const GM = GLLVModels
const Optim = GM.Optim

const MODE = length(ARGS) >= 1 ? ARGS[1] : "full"

const P = 5; const K = 2; const N = 80
const BETA = [-2.0, -1.0, 0.0, 1.0, 2.0]
const LAMBDA = 0.30 .* [0.8 0.0; 0.5 0.6; 0.3 -0.4; -0.2 0.5; 0.1 0.3]
const POWER = 1.5
phi_true(d) = d <= 5 ? 0.05 : 5.0
seed_of(d) = 4770 + d

# Compound Poisson-Gamma draw (Dunn & Smyth 2005): N ~ Poisson(λ),
# λ = μ^{2-p}/(φ(2-p)); y = 0 if N = 0, else Gamma(N α, φ(p-1)μ^{p-1}), α = (2-p)/(p-1).
function rtweedie(rng, μ, φ, p)
    λ = μ^(2 - p) / (φ * (2 - p))
    m = rand(rng, Poisson(λ))
    m == 0 && return 0.0
    return rand(rng, Gamma(m * (2 - p) / (p - 1), φ * (p - 1) * μ^(p - 1)))
end

function simulate(d)
    rng = MersenneTwister(seed_of(d))
    Z = randn(rng, K, N)
    η = BETA .+ LAMBDA * Z
    φ = phi_true(d)
    Y = [rtweedie(rng, exp(η[t, i]), φ, POWER) for t in 1:P, i in 1:N]
    return Y
end

# ---- EXACT replication of fit_tweedie_gllvm_grouped's θ0 / negll / optimiser for the
# default arguments (power = nothing, power_group = :shared, power_init = 1.5,
# link = LogLink(), mask = nothing, offset = nothing, hessian = :observed,
# g_tol = 1e-5, iterations = 500, newton_maxiter = 100, newton_tol = 1e-9).
function replicate(Y::AbstractMatrix, K::Integer, group::AbstractVector{<:Integer})
    p, n = size(Y)
    link = GM.LogLink(); mask = nothing; offset = nothing; hessian = :observed
    newton_maxiter = 100; newton_tol = 1e-9
    spec = GM._tweedie_power_spec(nothing, :shared, p)
    power0 = spec.fixed ? spec.values : GM._tweedie_power_init(1.5, p, spec.mode)
    rr = GM.rr_theta_len(p, K)
    labels = sort(unique(group)); G = length(labels)
    gidx = [findfirst(==(group[t]), labels) for t in 1:p]
    msk = GM._resolve_obs_mask(mask, Y)
    Yc  = GM._sanitize_missing(Y, 1e-6)
    Zemp = log.(Yc .+ GM._tweedie_log_offset(Yc, msk))
    offset === nothing || (Zemp .-= offset)
    GM._mask_warmstart!(Zemp, msk)
    β0 = vec(sum(Zemp; dims = 2)) ./ n
    Zc = Zemp .- β0
    F = svd(Zc); kk = min(K, length(F.S))
    Λ0 = zeros(p, K)
    @inbounds for j in 1:kk
        Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n))
    end
    ξ0 = spec.nfree == 0 ? Float64[] :
         spec.mode === :shared ? [GM._tweedie_xi(power0[1])] : GM._tweedie_xi.(power0)
    θ0 = vcat(β0, GM.pack_lambda(Λ0), fill(log(1.0), G), ξ0)
    function negll(θ)
        β = θ[1:p]
        Λ = GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K)
        φg = exp.(θ[(p + rr + 1):(p + rr + G)])
        φvec = [φg[gidx[t]] for t in 1:p]
        ξ = spec.nfree == 0 ? Float64[] : @view θ[(p + rr + G + 1):end]
        pw = if spec.nfree == 0
            spec.values
        elseif spec.mode === :shared
            fill(GM._tweedie_power(ξ[1]), p)
        else
            GM._tweedie_power.(ξ)
        end
        v = try
            -GM.tweedie_grouped_marginal_loglik_laplace(Yc, Λ, β, φvec, pw; link = link,
                                                        mask = msk, offset = offset,
                                                        hessian = hessian,
                                                        maxiter = newton_maxiter,
                                                        tol = newton_tol)
        catch
            return GM._TWEEDIE_FAIL_PENALTY
        end
        return isfinite(v) ? v : GM._TWEEDIE_FAIL_PENALTY
    end
    return (; θ0, negll, p, rr, G, gidx)
end

const LS = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
const OPTS = Optim.Options(g_tol = 1e-5, iterations = 500)
run_opt(negll, θs) = Optim.optimize(negll, θs, LS, OPTS; autodiff = :finite)

theta_of_fit(fit) = vcat(fit.β, GM.pack_lambda(fit.Λ), log.(fit.φ), GM._tweedie_xi(fit.power))

function unpack_summary(θ, R)
    φ = exp.(θ[(R.p + R.rr + 1):(R.p + R.rr + R.G)])
    ξ = θ[end]
    return φ, GM._tweedie_power(ξ), ξ
end

fmtv(v) = "[" * join([@sprintf("%.4g", x) for x in v], ", ") * "]"

function screen_one(d; alts = true)
    Y = simulate(d)
    group = collect(1:P)
    nz = [count(==(0.0), Y[t, :]) for t in 1:P]
    t0 = time()
    fit = GM.fit_tweedie_gllvm_grouped(Y; K = K, group = group)
    t_fit = time() - t0
    R = replicate(Y, K, group)
    θhat = theta_of_fit(fit)
    obj_at_fit = R.negll(θhat)
    gap = abs(obj_at_fit - (-fit.loglik))
    valid = isfinite(fit.loglik) && gap < 1e-6
    bd = GM._dispersion_group_boundary(fit.φ)
    bdg = findall(bd)
    @printf("\n=== dataset %d (seed %d, true φ = %.3g, power = %.2f) ===\n",
            d, seed_of(d), phi_true(d), POWER)
    println("  zeros per trait (of $N): ", nz)
    @printf("  fitter: loglik = %.6f  converged = %s  iters = %d  time = %.1fs\n",
            fit.loglik, fit.converged, fit.iterations, t_fit)
    println("  fitter φ̂ = ", fmtv(fit.φ), "  power̂ = ", @sprintf("%.4f", fit.power))
    println("  boundary groups (φ outside [1e-6, 1e6]): ", isempty(bdg) ? "none" : bdg); flush(stdout)
    @printf("  harness check: negll(θ̂) = %.9f, -loglik = %.9f, |gap| = %.3e -> %s\n",
            obj_at_fit, -fit.loglik, gap, valid ? "VALID" : "INVALID")
    row = Dict{Symbol,Any}(:d => d, :seed => seed_of(d), :phi_true => phi_true(d),
        :fit_ll => fit.loglik, :conv => fit.converged, :iters => fit.iterations,
        :phi_hat => fit.φ, :power_hat => fit.power, :bd => bdg, :valid => valid,
        :gap => gap, :t_fit => t_fit)
    alts || return row
    # (b) restart once from the returned point, boundary groups' log φ set to 0
    #     (no boundary groups -> a plain restart from the returned point).
    θb = copy(θhat)
    for g in bdg
        θb[R.p + R.rr + g] = 0.0
    end
    t1 = time()
    rb = run_opt(R.negll, θb)
    ll_b = -Optim.minimum(rb)
    # (a) fresh start = the fitter's θ0 with all log φ = 0. NOTE: the fitter's θ0
    #     already has log φ = log(1.0) = 0, so (a) is the fitter's own start; it doubles
    #     as an end-to-end replication check (should reproduce fit.loglik exactly).
    θa = copy(R.θ0)
    θa[(R.p + R.rr + 1):(R.p + R.rr + R.G)] .= 0.0
    ra = run_opt(R.negll, θa)
    ll_a = -Optim.minimum(ra)
    t_alt = time() - t1
    φb, pwb, ξb = unpack_summary(Optim.minimizer(rb), R)
    φa, pwa, ξa = unpack_summary(Optim.minimizer(ra), R)
    @printf("  (b) restart-from-returned: loglik = %.6f  gain = %+.3e  |ξ| = %.3g  φ = %s\n",
            ll_b, ll_b - fit.loglik, abs(ξb), fmtv(φb))
    @printf("  (a) fresh θ0, log φ = 0  : loglik = %.6f  gain = %+.3e  |ξ| = %.3g  φ = %s\n",
            ll_a, ll_a - fit.loglik, abs(ξa), fmtv(φa))
    @printf("  (a) reproduces fitter bit-for-bit: %s   alt time = %.1fs\n",
            ll_a == fit.loglik, t_alt)
    best_ll, which = ll_b >= ll_a ? (ll_b, "b") : (ll_a, "a")
    # an alternative point counts only if it is itself a valid Tweedie point
    alt_ok = which == "b" ? abs(ξb) <= GM._TWEEDIE_XI_MAX : abs(ξa) <= GM._TWEEDIE_XI_MAX
    stall = valid && alt_ok && (best_ll - fit.loglik > 1e-3)
    @printf("  best alt = (%s) %.6f, gain = %+.3e -> %s\n", which, best_ll,
            best_ll - fit.loglik, stall ? "STALL" : "no stall")
    merge!(row, Dict(:ll_a => ll_a, :ll_b => ll_b, :best => best_ll, :which => which,
                     :gain => best_ll - fit.loglik, :stall => stall,
                     :a_reproduces => ll_a == fit.loglik, :t_alt => t_alt))
    return row
end


function summarize(rows)
    println("\n=== SUMMARY (datasets completed within budget) ===")
    @printf("%-3s %-7s %-14s %-6s %-12s %-14s %-11s %-3s %s\n", "d", "φtrue", "fit_ll",
            "conv", "boundary", "best_alt", "gain", "alt", "verdict")
    for r in rows
        @printf("%-3d %-7.3g %-14.6f %-6s %-12s %-14.6f %+-11.3e %-3s %s\n", r[:d],
                r[:phi_true], r[:fit_ll], r[:conv], isempty(r[:bd]) ? "none" : string(r[:bd]),
                r[:best], r[:gain], r[:which], r[:stall] ? "STALL" : "ok")
    end
    isempty(rows) && return
    println("harness_valid (completed datasets): ", all(r -> r[:valid], rows),
            "   max |gap| = ", @sprintf("%.3e", maximum(r -> r[:gap], rows)))
    println("n_completed = ", length(rows), "   n_with_boundary = ",
            count(r -> !isempty(r[:bd]), rows), "   n_stalls = ", count(r -> r[:stall], rows),
            "   max gain = ", @sprintf("%.3e", maximum(r -> r[:gain], rows)))
    println("(a) reproduced the fitter bit-for-bit on ", count(r -> r[:a_reproduces], rows),
            "/", length(rows), " completed datasets")
end

const BUDGET = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 150.0
@printf("[%.0fs] package loaded\n", time() - T_START); flush(stdout)
for d in (6, 1)
    Y = simulate(d); R = replicate(Y, K, collect(1:P))
    R.negll(R.θ0)
    t = @elapsed for _ in 1:3; R.negll(R.θ0); end
    @printf("[%.0fs] negll cost at θ0, dataset %d (φ = %.3g): %.1f ms/eval; one FD gradient ≈ %d evals\n",
            time() - T_START, d, phi_true(d), 1000 * t / 3, 2 * length(R.θ0)); flush(stdout)
end
rows = Any[]
last_cost = 0.0
for d in (6, 7, 8, 9, 10, 1, 2, 3, 4, 5)
    el = time() - T_START
    if el + 1.5 * last_cost > BUDGET
        @printf("[%.0fs] budget guard: not starting dataset %d (budget %.0fs)\n", el, d, BUDGET)
        continue
    end
    t0 = time()
    push!(rows, screen_one(d))
    global last_cost = time() - t0
    @printf("[%.0fs] dataset %d done in %.1fs\n", time() - T_START, d, last_cost); flush(stdout)
end
summarize(rows)
