# Screen NB2 finite-dispersion candidates on both engines (developer use only).
using GLLVModels, RCall, Test, Random, SHA, TOML, LinearAlgebra, Printf
include(joinpath(pwd(), "test/parity/parity_helpers.jl"))
function _rand_poisson(λ::Float64)
    λ = clamp(λ, 0.0, 1e6)
    L = exp(-λ)
    k = 0
    prod = 1.0
    while true
        k += 1
        prod *= rand()
        prod <= L && return k - 1
    end
end

# Gamma–Poisson compound: NB2 with mean μ, dispersion r (Var = μ + μ²/r).
function _rand_gamma(shape::Float64, scale::Float64)
    if shape < 1.0
        return _rand_gamma(shape + 1.0, scale) * rand()^(1.0 / shape)
    end
    d = shape - 1.0 / 3.0
    c = 1.0 / sqrt(9.0 * d)
    while true
        x = randn()
        v = (1.0 + c * x)^3
        v <= 0 && continue
        u = rand()
        x2 = x * x
        u < 1.0 - 0.0331 * x2 * x2 && return d * v * scale
        logu = log(u)
        logu < 0.5 * x2 + d * (1.0 - v + log(v)) && return d * v * scale
    end
end

function _rand_nb2(μ::Float64, r::Float64)
    λ = _rand_gamma(r, μ / r)
    return _rand_poisson(λ)
end

function draw(seed, r_true, n=80)
    Random.seed!(seed)
    p, K = 5, 2
    β = log.([2.5, 3.0, 2.0, 2.8, 2.2])
    Λ = 0.30 .* parity_loadings_p5k2()
    Z = randn(K, n); η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        Y[t, s] = _rand_nb2(exp(clamp(η[t, s], -8.0, 8.0)), r_true)
    end
    return Y
end

const Optim = GLLVModels.Optim
const GM = GLLVModels
# Same objective and optimizer as fit_nb_gllvm_grouped (per-trait groups, log link, :observed), start θ0 given.
function nbfit(Y, K, θ0)
    p, n = size(Y); rr = GM.rr_theta_len(p, K)
    negll(θ) = begin
        v = try
            -GM.nb_grouped_marginal_loglik_laplace(Y, GM.unpack_lambda(θ[p+1:p+rr], p, K), θ[1:p], exp.(θ[p+rr+1:end]);
                hessian = :observed, maxiter = 100, tol = 1e-9)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    res = Optim.optimize(negll, θ0, Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking(order = 3)),
                         Optim.Options(g_tol = 1e-7, iterations = 800); autodiff = :finite)
    return Optim.minimizer(res), -Optim.minimum(res), Optim.converged(res)
end
function default_start(Y, K, logr)
    p, n = size(Y)
    Z = [log(max(Y[t, i] + 0.5, 1e-4)) for t in 1:p, i in 1:n]
    β0 = vec(sum(Z; dims = 2)) ./ n; F = svd(Z .- β0)
    Λ0 = zeros(p, K); for j in 1:K; Λ0[:, j] = F.U[:, j] .* (F.S[j] / sqrt(n)); end
    vcat(β0, GM.pack_lambda(Λ0), fill(logr, p))
end
datasets = [(51, 1.0), (52, 2.0), (46, 1.0), (45, 2.0), (45, 1.5), (46, 2.0)]
println("Julia ", VERSION)
for (seed, r_true) in datasets
    Y = draw(seed, r_true); p = 5; K = 2; rr = GM.rr_theta_len(p, K)
    rfit = fit_gllvmtmb_parity_loglik(Y, 2; family = :negbinomial); Rll = rfit.logLik
    nb(θ) = count(x -> !(1e-6 <= x <= 1e6), exp.(θ[p+rr+1:end]))
    θc, llc, _ = nbfit(Y, K, default_start(Y, K, log(10.0)))
    θb, llb = θc, llc
    bd = findall(x -> !(1e-6 <= x <= 1e6), exp.(θc[p+rr+1:end]))
    if !isempty(bd)
        θs = copy(θc); θs[p+rr .+ bd] .= 0.0
        θr, llr, _ = nbfit(Y, K, θs); llr > llb + 1e-6 && ((θb, llb) = (θr, llr))
    end
    θa, lla, _ = nbfit(Y, K, default_start(Y, K, 0.0))
    θd, lld = θb, llb                      # (d) = (b) plus a fresh start at r = 1 when the boundary was hit
    if !isempty(bd) && lla > lld + 1e-6
        θd, lld = θa, lla
    end
    @printf("seed=%d r=%.1f | R %.4f | current %+.1e (bd %d) | (b) %+.1e (bd %d) | (a) %+.1e (bd %d) | (d) %+.1e (bd %d)\n",
        seed, r_true, Rll, llc - Rll, nb(θc), llb - Rll, nb(θb), lla - Rll, nb(θa), lld - Rll, nb(θd))
end
