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

function draw(seed, r_true, n)
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

println("Julia ", VERSION)
for seed in 45:52
    n = 200; r_true = 1.0
    Y = draw(seed, r_true, n); p = 5; K = 2; rr = GM.rr_theta_len(p, K)
    fit = fit_gllvm(Y; family = GM.NegativeBinomial(), K = 2, g_tol = 1e-7, iterations = 800)
    rf = fit_gllvmtmb_parity_loglik(Y, 2; family = :negbinomial)
    rr_R = rcopy(Vector{Float64}, R"exp(as.numeric(fit_r$tmb_obj$env$parList(fit_r$opt$par)$log_phi_nbinom2))")
    θ = vcat(fit.β, GM.pack_lambda(fit.Λ), log.(fit.r_group))
    pushed = Float64[]
    for t in 1:p
        θs = copy(θ); θs[p + rr + t] = 30.0
        _, ll, _ = nbfit(Y, K, θs); push!(pushed, ll)
    end
    gain = maximum(pushed) - fit.loglik
    ok = !any(fit.dispersion_boundary) && fit.converged && abs(fit.loglik - rf.logLik) <= 1e-6 * abs(rf.logLik) &&
         maximum(abs.(fit.r_group .- rr_R) ./ rr_R) <= 1e-3 && gain <= 1e-6
    @printf("%s seed=%d n=%d | jl %.6f conv=%s bd=%d | R %.6f maxrelr=%.1e | best push gain %+.3e (trait %d)\n",
        ok ? "INTERIOR" : "reject  ", seed, n, fit.loglik, fit.converged, count(fit.dispersion_boundary), rf.logLik,
        maximum(abs.(fit.r_group .- rr_R) ./ rr_R), gain, argmax(pushed))
end
