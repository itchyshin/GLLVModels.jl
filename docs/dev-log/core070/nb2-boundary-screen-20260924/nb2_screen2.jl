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
println("Julia ", VERSION)
for n in (80, 200), r_true in (1.0, 2.0), seed in 45:52
    Y = draw(seed, r_true, n)
    jl = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = 2, g_tol = 1e-7, iterations = 800)
    r = fit_gllvmtmb_parity_loglik(Y, 2; family = :negbinomial)
    rg = rcopy(Float64, R"max(abs(as.numeric(fit_r$tmb_obj$gr(fit_r$opt$par))))")
    rr = rcopy(Vector{Float64}, R"exp(as.numeric(fit_r$tmb_obj$env$parList(fit_r$opt$par)$log_phi_nbinom2))")
    code = rcopy(Int, R"as.integer(fit_r$opt$convergence)")
    dll = abs(jl.loglik - r.logLik) / abs(r.logLik)
    ok = jl.converged && !any(jl.dispersion_boundary) && all(x -> x < 1e3, rr) && code == 0 && rg <= 1e-4 && dll <= 1e-6
    @printf("%s n=%d r=%.1f seed=%d | jl_conv=%s bd=%d | R code=%d grad=%.1e maxr_R=%.3g | rel_dll=%.1e | max_rel_r=%.1e\n",
        ok ? "PASS" : "fail", n, r_true, seed, jl.converged, count(jl.dispersion_boundary), code, rg, maximum(rr), dll,
        maximum(abs.(jl.r_group .- rr) ./ rr))
end
