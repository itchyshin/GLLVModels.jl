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

function draw(seed, r_true)
    Random.seed!(seed)
    p, K, n = 5, 2, 80
    β = log.([2.5, 3.0, 2.0, 2.8, 2.2])
    Λ = 0.30 .* parity_loadings_p5k2()
    Z = randn(K, n); η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        Y[t, s] = _rand_nb2(exp(clamp(η[t, s], -8.0, 8.0)), r_true)
    end
    return Y
end
println("Julia ", VERSION)
for (seed, r_true) in ((46, 1.0), (45, 2.0), (45, 1.0))
    Y = draw(seed, r_true)
    jl = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = 2, g_tol = 1e-7, iterations = 800)
    r = fit_gllvmtmb_parity_loglik(Y, 2; family = :negbinomial)
    rr = rcopy(Vector{Float64}, R"exp(as.numeric(fit_r$tmb_obj$env$parList(fit_r$opt$par)$log_phi_nbinom2))")
    # Julia's own objective at R's optimum (same parameterisation as nb2_health.jl)
    rl = rcopy(Matrix{Float64}, R"as.matrix(fit_r$tmb_obj$report(fit_r$tmb_obj$env$last.par)$Lambda_B)")
    rb = rcopy(Vector{Float64}, R"as.numeric(fit_r$tmb_obj$env$parList(fit_r$opt$par)$b_fix)")
    jl_at_r = GLLVModels.nb_grouped_marginal_loglik_laplace(Y, rl, rb, rr; hessian = :observed, maxiter = 100, tol = 1e-9)
    @printf("seed=%d r_true=%.1f\n  jl.loglik=%.6f  R.logLik=%.6f  (jl - R = %.3e)\n  julia loglik at R's optimum = %.6f\n  jl r = %s\n  R  r = %s\n  jl iterations=%d converged=%s\n",
        seed, r_true, jl.loglik, r.logLik, jl.loglik - r.logLik, jl_at_r, string(round.(jl.r_group; sigdigits=4)), string(round.(rr; sigdigits=4)), jl.iterations, jl.converged)
end
