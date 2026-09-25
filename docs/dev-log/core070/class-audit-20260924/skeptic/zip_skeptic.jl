# Independent skeptic check of CLASS A / CLASS B claims for fit_zip_gllvm.
# Own ZIP density, own robust mode search (trust-region Newton on the exact
# log-posterior), own brute-force grid quadrature of the exact marginal, own FD.
using Pkg; Pkg.activate("/tmp/claude-503/audit/env"; io = devnull)
using GLLVModels, Random, Distributions, LinearAlgebra, Printf, Optim, ForwardDiff
const G = GLLVModels
lgt(x) = 1 / (1 + exp(-x))
clampη(x) = clamp(x, -30.0, 30.0)
logfact(y) = y <= 1 ? 0.0 : sum(log, 2:y)

# own ZIP log density (same η clamp as the package, nothing else shared)
function zip_logf(y, ηz, ηc)
    π = lgt(ηz); μ = exp(ηc)
    y == 0 ? log(π + (1 - π) * exp(-μ)) : log1p(-π) + y * ηc - μ - logfact(y)
end
qpost(y, Λ, βz, βc, z) = sum(zip_logf(y[t], clampη(βz[t]), clampη(βc[t] + dot(view(Λ, t, :), z))) for t in eachindex(y)) - 0.5 * dot(z, z)

# own expected information for η^c, by summing over the support (independent of _zi_Icc_pois)
function Icc_num(ηz, ηc)
    μ = exp(ηc); ymax = ceil(Int, μ + 25 * sqrt(μ) + 40)
    acc = 0.0
    for y in 0:ymax
        lf = zip_logf(y, ηz, ηc)
        s = ForwardDiff.derivative(e -> zip_logf(y, ηz, e), ηc)
        acc += exp(lf) * s^2
    end
    max(acc, 1e-12)
end

function robust_mode(y, Λ, βz, βc, zstarts)
    K = size(Λ, 2); best = nothing; bestv = -Inf
    for z0 in zstarts
        all(isfinite, z0) || continue
        z0c = clamp.(z0, -8.0, 8.0)
        r = try
            Optim.optimize(z -> -qpost(y, Λ, βz, βc, z), z0c, Optim.NewtonTrustRegion(),
                           Optim.Options(g_tol = 1e-10, iterations = 500); autodiff = :forward)
        catch; nothing; end
        r === nothing && continue
        v = -Optim.minimum(r)
        if v > bestv; bestv = v; best = Optim.minimizer(r); end
    end
    best
end

# "correct" site value: package's intended formula (Laplace, expected-info logdet) at the TRUE mode
function correct_site(y, Λ, βz, βc; zpkg = nothing)
    K = size(Λ, 2)
    starts = Any[zeros(K)]; zpkg === nothing || push!(starts, zpkg)
    ẑ = robust_mode(y, Λ, βz, βc, starts)
    W = [Icc_num(clampη(βz[t]), clampη(βc[t] + dot(view(Λ, t, :), ẑ))) for t in eachindex(y)]
    A = Λ' * (W .* Λ) + I
    return qpost(y, Λ, βz, βc, ẑ) - 0.5 * logdet(Symmetric(A)), ẑ
end

# exact marginal log p(y_s) by brute-force 2-D trapezoid on a fixed wide grid (K = 2 only)
const GRID = collect(-7.0:0.04:7.0)
function exact_site(y, Λ, βz, βc)
    m = -Inf; vals = Vector{Float64}(undef, length(GRID)^2); k = 0
    for a in GRID, b in GRID
        k += 1
        v = sum(zip_logf(y[t], clampη(βz[t]), clampη(βc[t] + Λ[t, 1] * a + Λ[t, 2] * b)) for t in eachindex(y)) - 0.5 * (a^2 + b^2)
        vals[k] = v; m = max(m, v)
    end
    h = GRID[2] - GRID[1]
    return m + log(sum(exp.(vals .- m)) * h^2) - log(2π)
end

unpackθ(θ, p, K) = (θ[1:p], θ[(p + 1):(2p)], G.unpack_lambda(θ[(2p + 1):end], p, K))
function pkg_negll(Y, θ, p, K)
    βz, βc, Λ = unpackθ(θ, p, K)
    v = try
        -G.zip_marginal_loglik_laplace(Y, Λ, βz, βc; hessian = :observed, maxiter = 100, tol = 1e-9)
    catch
        1e12
    end
    isfinite(v) ? v : 1e12
end
function correct_ll(Y, θ, p, K)
    βz, βc, Λ = unpackθ(θ, p, K)
    s = 0.0
    for j in axes(Y, 2)
        y = Y[:, j]
        zpkg = G._twopart_mode(G.ZIPoisson(), y, zeros(p, K), Λ, βz, βc; maxiter = 100, tol = 1e-9)
        s += correct_site(y, Λ, βz, βc; zpkg = zpkg)[1]
    end
    s
end
function exact_ll(Y, θ, p, K)
    βz, βc, Λ = unpackθ(θ, p, K)
    sum(exact_site(Y[:, j], Λ, βz, βc) for j in axes(Y, 2))
end

function census(Y, θ, p, K; label = "")
    βz, βc, Λ = unpackθ(θ, p, K)
    nonconv = 0; nmaxit = 0; worst = (gap = 0.0, j = 0, pkg = 0.0, cor = 0.0, g = 0.0)
    pkgsum = 0.0; corsum = 0.0; nmat = 0; nhigh = 0
    for j in axes(Y, 2)
        y = Y[:, j]
        z100 = G._twopart_mode(G.ZIPoisson(), y, zeros(p, K), Λ, βz, βc; maxiter = 100, tol = 1e-9)
        z101 = G._twopart_mode(G.ZIPoisson(), y, zeros(p, K), Λ, βz, βc; maxiter = 101, tol = 1e-9)
        gq = maximum(abs, ForwardDiff.gradient(z -> qpost(y, Λ, βz, βc, z), z100))
        ran_out = z100 != z101     # 101st iteration still moved z ⇒ loop ended on maxiter, not tol
        pv = G.twopart_loglik_site(G.ZIPoisson(), y, zeros(p, K), Λ, βz, βc; maxiter = 100, tol = 1e-9)
        cv, _ = correct_site(y, Λ, βz, βc; zpkg = z100)
        pkgsum += pv; corsum += cv
        gq > 1e-4 && (nonconv += 1); ran_out && (nmaxit += 1)
        abs(cv - pv) > 1e-3 && (nmat += 1); pv - cv > 1e-3 && (nhigh += 1)
        (cv - pv) > worst.gap && (worst = (gap = cv - pv, j = j, pkg = pv, cor = cv, g = gq))
    end
    @printf("  census %-6s: non-mode sites (|∇q|>1e-4) %d/%d, ran out of iterations %d; pkg ll %.3f, correct ll %.3f, gap(correct-pkg) %.4g; material sites %d (pkg spuriously higher %d)\n",
            label, nonconv, size(Y, 2), nmaxit, pkgsum, corsum, corsum - pkgsum, nmat, nhigh)
    worst.j > 0 && @printf("           worst site %d: pkg %.4g vs correct %.4g, |∇q| at pkg z = %.3g\n", worst.j, worst.pkg, worst.cor, worst.g)
    return pkgsum, corsum
end

function fdgrad(f, θ, h)
    g = similar(θ)
    for i in eachindex(θ)
        e = zeros(length(θ)); e[i] = h * max(1.0, abs(θ[i]))
        g[i] = (f(θ .+ e) - f(θ .- e)) / (2e[i])
    end
    g
end

function run_case(name, Y, θtrue, K; do_opt_correct = true)
    p, n = size(Y)
    println("\n==== ", name, "  (p=$p, n=$n, K=$K, zeros=$(round(mean(Y .== 0); digits = 3)), max y=$(maximum(Y)))")
    t0 = time()
    f = fit_zip_gllvm(Y; K = K)
    @printf("  PUBLIC FIT: loglik %.3f  converged=%s  iterations=%d  (%.1f s)\n", f.loglik, f.converged, f.iterations, time() - t0)
    θ̂ = vcat(f.βz, f.βc, G.pack_lambda(f.Λc))
    @printf("  validity: pkg_negll(θ̂) + loglik = %.3g\n", pkg_negll(Y, θ̂, p, K) + f.loglik)
    census(Y, θ̂, p, K; label = "θ̂")
    census(Y, θtrue, p, K; label = "truth")
    ex̂ = exact_ll(Y, θ̂, p, K); ext = exact_ll(Y, θtrue, p, K)
    @printf("  EXACT marginal (grid quadrature): at θ̂ %.3f, at truth %.3f  (truth - θ̂ = %.3f)\n", ex̂, ext, ext - ex̂)
    fpk = θ -> pkg_negll(Y, θ, p, K)
    for h in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
        @printf("  FD max|∇negll| pkg objective, h=%.0e: %.4g\n", h, maximum(abs, fdgrad(fpk, θ̂, h)))
    end
    fco = θ -> -correct_ll(Y, θ, p, K)
    for h in (1e-5, 1e-4)
        @printf("  FD max|∇negll| CORRECT objective, h=%.0e: %.4g\n", h, maximum(abs, fdgrad(fco, θ̂, h)))
    end
    # warm restart on the package objective, identical optimiser settings
    ls = Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3))
    r = Optim.optimize(fpk, θ̂, ls, Optim.Options(g_tol = 1e-5, iterations = 500); autodiff = :finite)
    @printf("  warm restart (pkg objective): it=%d converged=%s x_conv=%s f_conv=%s g_conv=%s g_residual=%.4g, Δll=%.4g\n",
            Optim.iterations(r), Optim.converged(r), Optim.x_converged(r), Optim.f_converged(r), Optim.g_converged(r),
            Optim.g_residual(r), -(Optim.minimum(r)) - f.loglik)
    if do_opt_correct
        t1 = time()
        r2 = Optim.optimize(fco, θ̂, ls, Optim.Options(g_tol = 1e-4, iterations = 40, time_limit = 150); autodiff = :finite)
        θc = Optim.minimizer(r2)
        @printf("  optimise CORRECT objective from θ̂ (≤40 it, %.0f s): correct ll %.3f → %.3f; pkg ll at new point %.3f; exact ll %.3f → %.3f; max|Δθ| %.3g\n",
                time() - t1, correct_ll(Y, θ̂, p, K), -Optim.minimum(r2), -pkg_negll(Y, θc, p, K), ex̂, exact_ll(Y, θc, p, K), maximum(abs, θc .- θ̂))
    end
    return f
end

function lower_rot_own(Λ)       # rotate to the package's lower-triangular gauge (marginal is rotation-invariant)
    F = qr(Matrix(Λ')); L = Λ * Matrix(F.Q)
    for k in 1:size(L, 2); L[k, k] < 0 && (L[:, k] .*= -1); end
    for k in 1:size(L, 2), i in 1:(k - 1); L[i, k] = 0.0; end
    L
end

# Dataset A: re-create the probe's ZIP seed-101 dataset (same DGP recipe) to test its specific numbers.
function sim_probe(seed; p = 5, n = 80, K = 2, sd = 0.8, βc_mean = 1.0)
    rng = MersenneTwister(seed)
    Λ = sd .* randn(rng, p, K); βz = randn(rng, p) .* 0.5 .- 1.2; βc = randn(rng, p) .* 0.5 .+ βc_mean
    Z = randn(rng, K, n); _ = randn(rng, n)
    Y = zeros(Int, p, n)
    for t in 1:p, s in 1:n
        rand(rng) < lgt(βz[t]) && continue
        Y[t, s] = rand(rng, Poisson(exp(βc[t] + dot(Λ[t, :], Z[:, s]))))
    end
    Y, vcat(βz, βc, G.pack_lambda(lower_rot_own(Λ)))
end
# Dataset B: my own independent DGP (different RNG, different draws order, fixed designed parameters).
function sim_own(seed; p = 5, n = 80, K = 2, sd = 0.8)
    rng = Xoshiro(seed)
    βz = fill(-1.2, p) .+ 0.4 .* randn(rng, p); βc = 1.0 .+ 0.4 .* randn(rng, p)
    Λ = sd .* randn(rng, p, K)
    Y = zeros(Int, p, n)
    for s in 1:n
        z = randn(rng, K)
        for t in 1:p
            Y[t, s] = rand(rng) < lgt(βz[t]) ? 0 : rand(rng, Poisson(exp(βc[t] + dot(Λ[t, :], z))))
        end
    end
    Y, vcat(βz, βc, G.pack_lambda(lower_rot_own(Λ)))
end

which = isempty(ARGS) ? "A" : ARGS[1]
if which == "A"
    Y, θt = sim_probe(101); run_case("A: probe recipe, ZIP seed 101, sd 0.8", Y, θt, 2)
else
    for sd in (0.8,)
        for seed in parse.(Int, split(ARGS[2], ","))
            Y, θt = sim_own(seed; sd = sd); run_case("B: own DGP Xoshiro($seed), sd $sd", Y, θt, 2; do_opt_correct = false)
        end
    end
end
