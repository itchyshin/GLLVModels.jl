# Known-DGP recovery for the gllvmTMB-semantics zero-inflated route
# (zi_poisson(), zi_nbinom2(), zi_binomial(); src/families/zi_twin.jl), AGENTS.md
# design rule 1. Julia-only: simulated data from StableRNGs. This file is deliberately
# NOT tagged `# gllvm-parity-tag: P1`: it needs StableRNGs from test/Project.toml, so it
# runs under Pkg.test() only, not in the P1 twin job (which runs `julia --project=.`).
# The R-pinned literal NB2 regression datasets (reviewer seed 12, boundary seed 12,
# breakdown seed 6) live in the P1-tagged test/test_zi_twin.jl.
#
# Setting: p = 4 traits, n = 350 sites, K = 1, zero inflation in [0.15, 0.30],
# loadings of magnitude 0.4 to 0.6, NB2 dispersion in [1, 2]. Bounds: max |zi error|
# < 0.08, max |beta error| < 0.15, max |Lambda Lambda' error| < 0.25, NB2 max
# |log phi error| < 0.5. These are recovery bounds for one dataset each, not coverage
# claims. The NB2 cells use larger count means (beta 1.6 to 2.2) than the Poisson
# cells: at means near e^1 and phi near 1 the NB2 count-zero probability is large
# and the structural-zero probability and phi trade off; measured on two draws with
# beta = [1.2, 0.8, 1.4, 1.0], phi = [1.0, 1.5, 1.2, 1.0], R at P1 reached the same
# optima as Julia (logLik to 1e-4) with zi errors 0.10 and 0.15 (one trait's zi at
# 0) and log phi error 0.65, so that is an identifiability property of the model at
# this n, not a recovery failure of the route. At the reviewed setting 7 of 20 R
# fits also put one trait's phi at its upper boundary.
using Test
using GLLVModels
using StableRNGs
using Distributions
using LinearAlgebra

function _zir_simulate(fam, seed; p = 4, n = 350)
    rng = StableRNG(seed)
    β = fam isa ZiBinomial ? [0.2, -0.3, 0.5, 0.1] :
        fam isa ZiNbinom2 ? [2.0, 1.6, 2.2, 1.8] : [1.2, 0.8, 1.4, 1.0]
    zi = [0.25, 0.15, 0.30, 0.20]
    λ = [0.6, -0.5, 0.4, 0.5]
    phi = [1.5, 2.0, 1.0, 2.0]
    u = randn(rng, n)
    N = rand(rng, 3:8, p, n)
    Y = zeros(Int, p, n)
    for t in 1:p, s in 1:n
        η = β[t] + λ[t] * u[s]
        structural = rand(rng) < zi[t]
        y = fam isa ZiPoisson ? rand(rng, Poisson(exp(η))) :
            fam isa ZiNbinom2 ? rand(rng, NegativeBinomial(phi[t], phi[t] / (phi[t] + exp(η)))) :
                                rand(rng, Binomial(N[t, s], 1 / (1 + exp(-η))))
        Y[t, s] = structural ? 0 : y
    end
    return Y, N, (β = β, zi = zi, λ = λ, phi = phi)
end

@testset "zi_* known-DGP recovery (p = 4, n = 350, K = 1)" begin
    for (name, fam) in (("zi_poisson", zi_poisson()), ("zi_nbinom2", zi_nbinom2()),
                        ("zi_binomial", zi_binomial()))
        for seed in (101, 102)
            @testset "$name seed $seed" begin
                Y, N, truth = _zir_simulate(fam, seed)
                fit = fam isa ZiBinomial ? fit_gllvm(Y; family = fam, K = 1, trials = N) :
                                           fit_gllvm(Y; family = fam, K = 1)
                LLt = truth.λ * truth.λ'
                e_zi = maximum(abs.(fit.zi .- truth.zi))
                e_beta = maximum(abs.(fit.β .- truth.β))
                e_LLt = maximum(abs.(fit.Λ * fit.Λ' .- LLt))
                e_phi = fam isa ZiNbinom2 ? maximum(abs.(log.(fit.phi) .- log.(truth.phi))) : 0.0
                mineig = fit.min_site_eigen
                @info "zi recovery $name seed $seed" fit.loglik e_zi e_beta e_LLt e_phi mineig
                @test fit.converged
                @test e_zi < 0.08
                @test e_beta < 0.15
                @test e_LLt < 0.25
                @test e_phi < 0.5
                # Spurious-maximum guard. A breakdown maximum sits hundreds of units
                # above the (Laplace) log-likelihood at the true parameters (+298 on the
                # reviewer case); a genuine maximum exceeds it by about half a
                # chi-square on 16 parameters, well under 25.
                ll_truth = zi_marginal_loglik_laplace(fam, Float64.(Y), reshape(truth.λ, 4, 1),
                    truth.β, log.(truth.zi ./ (1 .- truth.zi));
                    phi = fam isa ZiNbinom2 ? truth.phi : nothing,
                    trials = fam isa ZiBinomial ? N : nothing)
                @test fit.loglik - ll_truth < 25
                @test mineig > GLLVModels.ZI_LAPLACE_EIGMIN_FLOOR
            end
        end
    end
end
