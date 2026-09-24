# test_negbin_parity.jl — NB2 GLLVModels logLik vs gllvmTMB
#
# Included by runparity.jl. NEVER included by test/runtests.jl.
# Same-model bar: per-trait intercepts + latent unique=FALSE (no Ψ) +
# per-trait NB2 dispersion (R default log_phi_nbinom2[p]; Julia #132).
#
# Julia parity entry: plain fit_gllvm(...; family=NegativeBinomial()) defaults
# to per-trait φ (disp_group=:species → NBGroupedFit). Shared-r remains
# fit_nb_gllvm (named). Inventory #132 / default-route-phi-20260801.

using GLLVModels, RCall, Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "nb2_health.jl"))

# parity_helpers.jl is included once by runparity.jl

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

@testset "NB2 GLLVModels parity: GLLVModels.jl vs gllvmTMB" begin
    Random.seed!(45)
    p, K, n = 5, 2, 80
    β = log.([2.5, 3.0, 2.0, 2.8, 2.2])
    # Shared true r (Var = μ + μ²/r); both engines estimate per-trait φ/r (R default).
    # Mild loadings keep Laplace modes stable (Poisson parity lesson).
    r_true = 4.0
    Λ = 0.30 .* parity_loadings_p5k2()
    Z = randn(K, n)
    η = β .+ Λ * Z
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = exp(clamp(η[t, s], -8.0, 8.0))
        Y[t, s] = _rand_nb2(μ, r_true)
    end
    # The loop above records how the data were drawn (Julia 1.12+). Julia 1.10
    # draws different numbers from this seed, so the fit uses the stored draw.
    Y = parity_nb2_original_Y()

    # Public default route — twin-aligned with gllvmTMB default nbinom2().
    jl_fit = fit_gllvm(Y; family = GLLVModels.NegativeBinomial(), K = K,
                       g_tol = 1e-7, iterations = 800)
    @test jl_fit isa NBGroupedFit
    @test jl_fit.converged
    @test isfinite(jl_fit.loglik)
    @test length(jl_fit.r_group) == p
    jl_logL = jl_fit.loglik

    r = parity_nb2_health(Y, K, jl_fit)
    @testset "original model and complete fit health" begin
        d = r.health
        @test d["hessian"] == "observed"
        @test d["native_nfree"] == d["r_nfree"] == 19
        @test d["r_packing_delta"] <= 1e-12
        @test d["native_gradient_max"] <= 1e-4
        @test d["r_gradient_max"] <= 1e-4
        @test d["fd_stability"] <= 1e-4
        @test d["native_objective_delta"] <= 1e-8
        @test abs(d["samepoint_delta"]) <= 1e-6
        @test abs(d["r_objective"] + d["r_loglik"]) <= 1e-8
        @test all(isfinite,d["native_parameters"]) && all(isfinite,d["r_parameters"])
    end
    @test r.converged
    @test isfinite(r.logLik)

    print_parity_loglik(
        "NB2 logLik oracle (seed=45, p=$p, K=$K, n=$n, per-trait φ via fit_gllvm default)";
        jl_logL = jl_logL, r_logL = r.logLik, r_obj = r.objective,
    )

    @testset "log-likelihood agreement (rtol=1e-6)" begin
        @test jl_logL ≈ r.logLik rtol = 1e-6
        @test r.logLik ≈ -r.objective rtol = 0 atol = 1e-10
    end
end
