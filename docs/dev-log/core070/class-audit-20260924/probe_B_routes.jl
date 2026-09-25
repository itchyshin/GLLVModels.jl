# Class B end-to-end: fit through the PUBLIC route, then measure the gradient of the
# fit's own objective (via the package's _family_ci adapter) at the returned point.
using GLLVModels, LinearAlgebra, Random, Distributions, StatsModels
const G = GLLVModels
function fdgrad(f, θ; h = 1e-5)
    g = similar(θ)
    for i in eachindex(θ)
        e = h * max(1.0, abs(θ[i])); θp = copy(θ); θm = copy(θ); θp[i] += e; θm[i] -= e
        g[i] = (f(θp) - f(θm)) / (2e)
    end
    return g
end
logistic(x) = 1 / (1 + exp(-x))
function sim(route, seed; p = 8, n = 60, K = 2, sc = 1.0)
    rng = MersenneTwister(seed)
    Λ = sc .* randn(rng, p, K); β = randn(rng, p) .* 0.5; Z = randn(rng, K, n); x = randn(rng, n)
    η = β .+ Λ * Z
    if route == :poisson_formula_X
        Y = [rand(rng, Poisson(exp(clamp(η[t, s] + 0.5 * x[s], -5, 5)))) for t in 1:p, s in 1:n]
        return Y, x
    elseif route in (:nb1_default, :poisson_default)
        route == :poisson_default && return ([rand(rng, Poisson(exp(clamp(η[t, s], -5, 5)))) for t in 1:p, s in 1:n], nothing)
        φ = exp.(randn(rng, p) .* 0.5)
        return ([rand(rng, NegativeBinomial(exp(clamp(η[t, s], -5, 5)) / φ[t], 1 / (1 + φ[t]))) for t in 1:p, s in 1:n], nothing)
    elseif route == :zip_default
        πz = logistic.(randn(rng, p) .- 1)
        return ([rand(rng) < πz[t] ? 0 : rand(rng, Poisson(exp(clamp(η[t, s], -5, 5)))) for t in 1:p, s in 1:n], nothing)
    elseif route == :betabinom_default
        φ = exp.(randn(rng, p) .* 0.5 .+ 1.5); N = fill(10, p, n)
        Y = [rand(rng, BetaBinomial(10, clamp(logistic(η[t, s]), 1e-3, 1 - 1e-3) * φ[t], clamp(1 - logistic(η[t, s]), 1e-3, 1 - 1e-3) * φ[t])) for t in 1:p, s in 1:n]
        return Y, N
    elseif route == :orderedbeta_default
        Y = [begin u = rand(rng); p0 = 1 - logistic(η[t, s] + 1); p1 = logistic(η[t, s] - 1)
                u < p0 ? 0.0 : u < p0 + p1 ? 1.0 : clamp(rand(rng, Beta(clamp(logistic(η[t, s]), 1e-3, 1 - 1e-3) * 10, clamp(1 - logistic(η[t, s]), 1e-3, 1 - 1e-3) * 10)), 1e-4, 1 - 1e-4) end
             for t in 1:p, s in 1:n]
        return Y, nothing
    elseif route == :studentt_fixed_nu
        Y = [η[t, s] + 0.5 * rand(rng, TDist(4.0)) for t in 1:p, s in 1:n]
        return Y, nothing
    end
end
function runroute(route, seed)
    Y, extra = sim(route, seed)
    t0 = time()
    fit, ci = if route == :poisson_formula_X
        p, n = size(Y); X = reshape(repeat(extra', p), p, n, 1)
        f = gllvm(@formula(y ~ 1 + x), Y, (x = extra,); family = Poisson(), K = 2)
        f, G._family_ci(f, Float64.(Y); X = X)
    elseif route == :nb1_default
        f = fit_gllvm(Y; family = G.NB1(), K = 2); f, G._family_ci(f, Float64.(Y))
    elseif route == :poisson_default
        f = fit_gllvm(Y; family = Poisson(), K = 2); f, G._family_ci(f, Float64.(Y))
    elseif route == :zip_default
        f = fit_gllvm(Y; family = G.ZIPoisson(), K = 2); f, G._family_ci(f, Float64.(Y))
    elseif route == :betabinom_default
        f = fit_gllvm(Y; family = G.BetaBinom(), K = 2, N = extra); f, G._family_ci(f, Float64.(Y); N = extra)
    elseif route == :orderedbeta_default
        f = fit_gllvm(Y; family = G.OrderedBeta(), K = 2); f, G._family_ci(f, Y)
    elseif route == :studentt_fixed_nu
        f = fit_gllvm(Y; family = G.StudentTFamily(4.0), K = 2); f, G._family_ci(f, Y)
    end
    el = time() - t0
    g = fdgrad(ci.nll, ci.θ)
    gn = maximum(abs, g)
    flag = fit.converged && gn > 1e-3 ? "DISHONEST(conv=true, |g|>100*g_tol)" : ""
    println(rpad(string(route), 22), " seed=$seed  type=$(nameof(typeof(fit)))  converged=$(fit.converged)  iters=$(fit.iterations)  loglik=$(round(fit.loglik; digits=3))  nll_adapter=$(round(ci.nll(ci.θ); digits=3))  max|grad|=$(round(gn; sigdigits=3))  t=$(round(el; digits=1))s  $flag")
    flush(stdout)
end
routes = Symbol.(split(get(ENV, "ROUTES", "poisson_default,poisson_formula_X,nb1_default,zip_default,betabinom_default,orderedbeta_default,studentt_fixed_nu"), ","))
seeds = parse.(Int, split(get(ENV, "SEEDS", "1,2,3,4"), ","))
for r in routes, s in seeds
    try
        runroute(r, s)
    catch e
        println(rpad(string(r), 22), " seed=$s  ERROR: ", sprint(showerror, e)[1:min(end, 300)])
    end
end
