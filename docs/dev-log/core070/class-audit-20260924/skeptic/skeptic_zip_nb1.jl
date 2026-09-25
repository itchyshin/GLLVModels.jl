# Skeptic check (independent of the probe harness): is fit_zip_gllvm / fit_nb1_gllvm_grouped
# returning converged=true at a point that is NOT stationary for a correctly evaluated
# objective, and is a better optimum real (not an artifact of a failed inner mode)?
#
# Independent pieces: my own log-densities (ForwardDiff for eta-derivatives), my own
# guarded Newton mode search (line search, multistart), my own Laplace assembly, a different
# outer line search (HagerZhang) for the re-optimisation.
using GLLVModels, LinearAlgebra, Random, Statistics, Distributions, Printf
const GM = GLLVModels
const Optim = GM.Optim
const FDiff = GM.ForwardDiff
const T0 = time()
logistic(x) = 1 / (1 + exp(-x))

# ---------------- DGP (copied: data generation is not under test) ----------------
function lowtri(rng, p, K, sd)
    Λ = sd .* randn(rng, p, K)
    for j in 1:K, i in 1:(j - 1); Λ[i, j] = 0.0; end
    for j in 1:K; Λ[j, j] = abs(Λ[j, j]) + 0.3 * sd; end
    Λ
end
function dgp_zip(seed; p = 5, n = 80, K = 2, sd = 0.7, π0 = 0.25, βlo = 0.5, βhi = 1.5)
    rng = MersenneTwister(seed)
    βz = fill(log(π0 / (1 - π0)), p); βc = βlo .+ (βhi - βlo) .* rand(rng, p)
    Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [rand(rng) < π0 ? 0 : rand(rng, Poisson(exp(clamp(βc[t] + dot(Λ[t, :], Z[:, i]), -30, 30))))
         for t in 1:p, i in 1:n]
    Y, βz, βc, Λ
end
function dgp_nb1(seed; p = 5, n = 80, K = 2, sd = 0.7, φ = 1.0, βlo = 0.5, βhi = 1.5)
    rng = MersenneTwister(seed)
    β = βlo .+ (βhi - βlo) .* rand(rng, p); Λ = lowtri(rng, p, K, sd); Z = randn(rng, K, n)
    Y = [begin μ = exp(clamp(β[t] + dot(Λ[t, :], Z[:, i]), -30, 30))
             rand(rng, NegativeBinomial(μ / φ, 1 / (1 + φ))) end for t in 1:p, i in 1:n]
    Y, β, Λ
end

# ---------------- my own log-densities (functions of the count-part eta) ----------------
zip_logf(y, η, πz) = y == 0 ? log(πz + (1 - πz) * exp(-exp(η))) :
                              log1p(-πz) + y * η - exp(η) - GM.loggamma(y + 1.0)
function nb1_logf(y, η, φ)
    μ = exp(η); r = μ / φ
    GM.loggamma(y + r) - GM.loggamma(r) - GM.loggamma(y + 1.0) - r * log1p(φ) + y * log(φ / (1 + φ))
end
d1(f, η) = FDiff.derivative(f, η)
d2(f, η) = FDiff.derivative(x -> FDiff.derivative(f, x), η)

# Expected count-part information for ZIP by brute-force summation (checks the package's
# closed form independently; used for the log-det because that is the package's DEFINITION).
function zip_Iexp_sum(η, πz)
    μ = exp(η); ymax = ceil(Int, μ + 30 * sqrt(μ) + 50); acc = 0.0
    for y in 0:ymax
        lp = zip_logf(y, η, πz); acc += exp(lp) * d1(x -> zip_logf(y, x, πz), η)^2
    end
    acc
end
function zip_Iexp_closed(η, πz)   # re-derived here: E[s^2], s = score in eta
    μ = exp(η); e = exp(-μ); P0 = πz + (1 - πz) * e
    max((1 - πz) * (μ - e * μ^2) + (1 - πz)^2 * e^2 * μ^2 / P0, 1e-12)
end

# ---------------- guarded Newton mode search for h(z) = sum logf(β+Λz) - z'z/2 ----------------
function site_h(fs, β, Λ, z)
    η = β .+ Λ * z
    sum(fs[t](η[t]) for t in eachindex(β)) - 0.5 * dot(z, z)
end
function mode_guarded(fs, β, Λ; z0 = zeros(size(Λ, 2)), tol = 1e-10, maxit = 500)
    z = copy(z0); K = length(z); hz = site_h(fs, β, Λ, z)
    for it in 1:maxit
        η = β .+ Λ * z
        g1 = [d1(fs[t], η[t]) for t in eachindex(η)]
        g2 = [d2(fs[t], η[t]) for t in eachindex(η)]
        g = Λ' * g1 .- z
        maximum(abs, g) < tol && return (z, hz, maximum(abs, g), it, :ok)
        H = Λ' * (g2 .* Λ) - I          # Hessian of h (negative definite if concave)
        C = cholesky(Symmetric(-H); check = false)
        dir = issuccess(C) ? C \ g : g ./ max(1.0, maximum(abs, g))   # Newton or gradient ascent
        t = 1.0; improved = false
        for _ in 1:60
            zn = z .+ t .* dir; hn = site_h(fs, β, Λ, zn)
            if isfinite(hn) && hn >= hz - 1e-14 * abs(hz)
                z = zn; hz = hn; improved = true; break
            end
            t /= 2
        end
        improved || return (z, hz, maximum(abs, g), it, :stalled)
    end
    η = β .+ Λ * z; g = Λ' * [d1(fs[t], η[t]) for t in eachindex(η)] .- z
    (z, hz, maximum(abs, g), maxit, :maxit)
end
function mode_multistart(fs, β, Λ)
    K = size(Λ, 2); best = mode_guarded(fs, β, Λ)
    starts = [zeros(K)]
    for k in 1:K, s in (-2.5, 2.5); v = zeros(K); v[k] = s; push!(starts, v); end
    nmodes = 1; zs = [best[1]]
    for s in starts[2:end]
        r = mode_guarded(fs, β, Λ; z0 = s)
        if all(norm(r[1] - zz) > 1e-4 for zz in zs) && r[5] == :ok
            push!(zs, r[1]); nmodes += 1
        end
        r[2] > best[2] + 1e-10 && (best = r)
    end
    best, nmodes
end

# ---------------- Laplace assembly (package's definitions of W, my own everything else) ----------------
function zip_site(y, βz, βc, Λ; multistart = false)
    πs = logistic.(βz)
    fs = [η -> zip_logf(y[t], η, πs[t]) for t in eachindex(y)]
    r, nm = multistart ? mode_multistart(fs, βc, Λ) : (mode_guarded(fs, βc, Λ), 1)
    z = r[1]; η = βc .+ Λ * z
    W = [zip_Iexp_closed(η[t], πs[t]) for t in eachindex(y)]
    v = r[2] - 0.5 * logdet(Symmetric(Λ' * (W .* Λ) + I))
    (v = v, z = z, gres = r[3], status = r[5], nmodes = nm)
end
function nb1_site(y, β, Λ, φ; multistart = false)
    fs = [η -> nb1_logf(y[t], η, φ[t]) for t in eachindex(y)]
    r, nm = multistart ? mode_multistart(fs, β, Λ) : (mode_guarded(fs, β, Λ), 1)
    z = r[1]; η = β .+ Λ * z
    W = [-d2(fs[t], η[t]) for t in eachindex(y)]      # observed curvature (hessian=:observed)
    v = r[2] - 0.5 * logdet(Symmetric(Λ' * (W .* Λ) + I))
    (v = v, z = z, gres = r[3], status = r[5], nmodes = nm)
end

function unpack_zip(θ, p, K); rr = GM.rr_theta_len(p, K)
    θ[1:p], θ[p+1:2p], GM.unpack_lambda(θ[2p+1:2p+rr], p, K); end
function unpack_nb1(θ, p, K); rr = GM.rr_theta_len(p, K)
    θ[1:p], GM.unpack_lambda(θ[p+1:p+rr], p, K), exp.(θ[p+rr+1:p+rr+p]); end

my_nll_zip(Y, θ, K; ms = false) = begin p = size(Y, 1); βz, βc, Λ = unpack_zip(θ, p, K)
    -sum(zip_site(view(Y, :, i), βz, βc, Λ; multistart = ms).v for i in axes(Y, 2)) end
my_nll_nb1(Y, θ, K; ms = false) = begin p = size(Y, 1); β, Λ, φ = unpack_nb1(θ, p, K)
    -sum(nb1_site(view(Y, :, i), β, Λ, φ; multistart = ms).v for i in axes(Y, 2)) end

pkg_site_zip(y, βz, βc, Λ) = GM.twopart_loglik_site(GM.ZIPoisson(), y, zeros(size(Λ)), Λ, βz, βc)
pkg_site_nb1(y, β, Λ, φ) = GM._nb1_grouped_loglik_site([GM.NB1(float(x)) for x in φ], y, ones(Int, length(y)),
                                                       Λ, β, GM.LogLink(); hessian = :observed)

function cfd(f, θ, h)
    g = zeros(length(θ))
    for i in eachindex(θ)
        e = zeros(length(θ)); e[i] = h
        g[i] = (f(θ .+ e) - f(θ .- e)) / (2h)
    end
    g
end

function site_compare(tag, Y, θ, K, which)
    p = size(Y, 1)
    diffs = Float64[]; nmm = 0; bad = 0; pv = 0.0; mv = 0.0
    for i in axes(Y, 2)
        y = view(Y, :, i)
        if which == :zip
            βz, βc, Λ = unpack_zip(θ, p, K)
            m = zip_site(y, βz, βc, Λ; multistart = true); pk = pkg_site_zip(y, βz, βc, Λ)
        else
            β, Λ, φ = unpack_nb1(θ, p, K)
            m = nb1_site(y, β, Λ, φ; multistart = true); pk = pkg_site_nb1(y, β, Λ, φ)
        end
        push!(diffs, pk - m.v); pv += pk; mv += m.v
        m.nmodes > 1 && (nmm += 1); m.status == :ok || (bad += 1)
    end
    big = count(d -> abs(d) > 1e-6, diffs)
    @printf("  [%s] pkg total ll %.4f | my robust ll %.4f | pkg-my = %+.4f | sites |pkg-my|>1e-6: %d (max %+.3e, min %+.3e) | multimodal sites %d | my-mode not ok %d\n",
            tag, pv, mv, pv - mv, big, maximum(diffs), minimum(diffs), nmm, bad)
    diffs
end

# ================= ZIP base-s1 =================
println("== ZIP formula check: closed-form expected info vs brute-force sum ==")
for (η, πz) in ((-1.0, 0.25), (0.5, 0.25), (1.5, 0.4), (2.5, 0.1), (0.0, 0.7))
    @printf("  eta=%.1f pi=%.2f closed=%.10f sum=%.10f\n", η, πz, zip_Iexp_closed(η, πz), zip_Iexp_sum(η, πz))
end

for (label, seed) in (("zip base-s1", 101),)
    println("\n== ", label, " (seed ", seed, ") ==")
    Y, βz_t, βc_t, Λ_t = dgp_zip(seed); p, n = size(Y); K = 2
    t = time(); fit = fit_zip_gllvm(Y; K = K); @printf("  fit_zip_gllvm: converged=%s loglik=%.4f iterations=%d (%.1fs)\n", fit.converged, fit.loglik, fit.iterations, time() - t)
    θhat = vcat(fit.βz, fit.βc, GM.pack_lambda(fit.Λc))
    d = site_compare("theta_hat", Y, θhat, K, :zip)
    f = θ -> my_nll_zip(Y, θ, K)
    for h in (1e-3, 1e-4, 1e-5)
        g = cfd(f, θhat, h); @printf("  my smooth objective: central-FD max|grad| at theta_hat, h=%.0e: %.4f (argmax %d)\n", h, maximum(abs, g), argmax(abs.(g)))
    end
    # descent test without trusting any gradient magnitude: step along -g of MY objective
    g = cfd(f, θhat, 1e-4); f0 = f(θhat)
    for s in (1e-4, 1e-3, 1e-2, 3e-2)
        @printf("  my nll(theta_hat - %.0e*g/|g|) - my nll(theta_hat) = %+.5f\n", s, f(θhat .- s .* g ./ norm(g)) - f0)
    end
    # re-optimise MY objective from theta_hat with a DIFFERENT line search (HagerZhang)
    t = time()
    res = Optim.optimize(f, θhat, Optim.LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 300, time_limit = 120); autodiff = :finite)
    θ2 = Optim.minimizer(res)
    @printf("  re-opt (my objective, LBFGS+HagerZhang): my ll %.4f -> %.4f (Δ %+.4f), g_conv=%s gres=%.2e it=%d (%.1fs)\n",
            -f0, -Optim.minimum(res), f0 - Optim.minimum(res), res.stopped_by.g_converged, Optim.g_residual(res), Optim.iterations(res), time() - t)
    site_compare("theta_reopt", Y, θ2, K, :zip)
    pk2 = GM.zip_marginal_loglik_laplace(Y, unpack_zip(θ2, p, K)[3], θ2[1:p], θ2[p+1:2p])
    @printf("  package's own objective at my re-opt point: %.4f (vs fit.loglik %.4f, Δ %+.4f)\n", pk2, fit.loglik, pk2 - fit.loglik)
    @printf("  max|Δθ| theta_reopt vs theta_hat: %.3f ; |Λc| frob hat %.3f reopt %.3f truth %.3f\n",
            maximum(abs, θ2 .- θhat), norm(fit.Λc * fit.Λc'), norm(unpack_zip(θ2,p,K)[3]*unpack_zip(θ2,p,K)[3]'), norm(Λ_t*Λ_t'))
    @printf("  βz hat %s\n  βz re-opt %s (truth %.3f)\n", round.(fit.βz; digits = 3), round.(θ2[1:p]; digits = 3), βz_t[1])
    # user-level: does giving the inner loop more iterations change the fitter's answer?
    t = time(); fit2 = fit_zip_gllvm(Y; K = K, newton_maxiter = 2000)
    @printf("  fit_zip_gllvm(newton_maxiter=2000): converged=%s loglik=%.4f iterations=%d (%.1fs)\n", fit2.converged, fit2.loglik, fit2.iterations, time() - t)
    @printf("  elapsed %.0fs\n", time() - T0)
end

# ================= NB1 base-s1..s3 =================
for (label, seed) in (("nb1 base-s2", 102), ("nb1 base-s1", 101), ("nb1 base-s3", 103))
    time() - T0 > 380 && (println("budget: skipping ", label); continue)
    println("\n== ", label, " (seed ", seed, ") ==")
    Y, β_t, Λ_t = dgp_nb1(seed); p, n = size(Y); K = 2
    t = time(); fit = fit_nb1_gllvm_grouped(Y; K = K, group = collect(1:p))
    @printf("  fit_nb1_gllvm_grouped: converged=%s loglik=%.4f iterations=%d (%.1fs)\n", fit.converged, fit.loglik, fit.iterations, time() - t)
    θhat = vcat(fit.β, GM.pack_lambda(fit.Λ), log.(fit.φ))
    site_compare("theta_hat", Y, θhat, K, :nb1)
    f = θ -> my_nll_nb1(Y, θ, K)
    for h in (1e-3, 1e-4, 1e-5)
        g = cfd(f, θhat, h); @printf("  my smooth objective: central-FD max|grad| at theta_hat, h=%.0e: %.3e (argmax %d)\n", h, maximum(abs, g), argmax(abs.(g)))
    end
    # package objective FD gradient for contrast
    fp = θ -> -GM.nb1_grouped_marginal_loglik_laplace(Y, unpack_nb1(θ,p,K)[2], θ[1:p], unpack_nb1(θ,p,K)[3]; hessian = :observed)
    for h in (1e-4, 1e-5, 1e-6)
        g = cfd(fp, θhat, h); @printf("  PACKAGE objective: central-FD max|grad| at theta_hat, h=%.0e: %.3e\n", h, maximum(abs, g))
    end
    t = time()
    res = Optim.optimize(f, θhat, Optim.LBFGS(), Optim.Options(g_tol = 1e-5, iterations = 200, time_limit = 60); autodiff = :finite)
    @printf("  re-opt my objective from theta_hat: my ll %.5f -> %.5f (Δ %+.5f), g_conv=%s gres=%.2e it=%d (%.1fs) max|Δθ|=%.4f\n",
            -f(θhat), -Optim.minimum(res), f(θhat) - Optim.minimum(res), res.stopped_by.g_converged, Optim.g_residual(res), Optim.iterations(res), time() - t,
            maximum(abs, Optim.minimizer(res) .- θhat))
    @printf("  elapsed %.0fs\n", time() - T0)
end
@printf("total %.0fs\n", time() - T0)
