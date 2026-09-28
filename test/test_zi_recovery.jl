# Known-DGP recovery for the gllvmTMB-semantics zero-inflated route
# (zi_poisson(), zi_nbinom2(), zi_binomial(); src/families/zi_twin.jl), AGENTS.md
# design rule 1. Julia-only: simulated data from StableRNGs, plus two literal NB2
# datasets (sha256-guarded CSVs, R's P1 fit recorded in test/fixtures/zi_recovery_cases.toml)
# that exercise the Laplace breakdown the route guards against.
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
using SHA
using TOML

const _ZIR_DIR = joinpath(@__DIR__, "fixtures")

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

function _zir_load_csv(path, p, n)
    Y = zeros(Int, p, n)
    for line in readlines(path)[2:end]
        isempty(line) && continue
        r = parse.(Int, split(line, ","))
        Y[r[2], r[1]] = r[3]
    end
    return Y
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

@testset "zi_nbinom2 literal regression cases (Laplace breakdown)" begin
    cases = TOML.parsefile(joinpath(_ZIR_DIR, "zi_recovery_cases.toml"))
    for key in ("nb2_reviewer_seed12", "nb2_boundary_seed12")
        @testset "$key" begin
            c = cases[key]
            path = joinpath(_ZIR_DIR, c["data_file"])
            @test bytes2hex(sha256(read(path))) == c["data_sha256"]
            Y = _zir_load_csv(path, Int(c["n_traits"]), Int(c["n_unit"]))
            fit = fit_gllvm(Y; family = zi_nbinom2(), K = 1)
            llR = Float64(c["r_loglik"])
            mineig = fit.min_site_eigen
            @info "zi literal case $key" fit.loglik llR mineig fit.phi
            @test fit.converged
            # The spurious point on this surface has a HIGHER Laplace value than R's
            # optimum (-3012.80 against -3311.45 on the reviewer case), so "not above
            # R" is the guard; "not below R" checks the route still finds R's optimum.
            @test abs(fit.loglik - llR) < 1e-4
            @test mineig > GLLVModels.ZI_LAPLACE_EIGMIN_FLOOR
            @test maximum(abs.(fit.zi .- Float64.(c["r_zi"]))) < 1e-3
        end
    end

    # A draw where the Laplace surface itself leads into the breakdown region (R stops
    # there with convergence = 1): Julia must not report a converged fit.
    @testset "nb2_breakdown_seed6" begin
        c = cases["nb2_breakdown_seed6"]
        path = joinpath(_ZIR_DIR, c["data_file"])
        @test bytes2hex(sha256(read(path))) == c["data_sha256"]
        Y = _zir_load_csv(path, 4, 400)
        fit = @test_logs (:warn, r"Laplace breakdown guard") match_mode = :any fit_gllvm(
            Y; family = zi_nbinom2(), K = 1)
        @info "zi literal case nb2_breakdown_seed6" fit.loglik fit.min_site_eigen fit.converged
        @test !fit.converged
        @test fit.min_site_eigen < 1.1 * GLLVModels.ZI_LAPLACE_EIGMIN_FLOOR
    end

    # The guard itself: at the reviewer case's spurious point the site precision is
    # near singular, so the guarded marginal refuses it; unguarded, it reports the
    # inflated Laplace value that R's objective also returns there.
    c = cases["nb2_reviewer_seed12"]
    Y = _zir_load_csv(joinpath(_ZIR_DIR, c["data_file"]), 4, 400)
    sp = c["spurious_point"]
    args = (zi_nbinom2(), Float64.(Y), reshape(Float64.(sp["Lambda"]), 4, 1),
            Float64.(sp["beta"]), Float64.(sp["logit_zi"]))
    @test zi_marginal_loglik_laplace(args...; phi = Float64.(sp["phi"])) == -Inf
    ll_unguarded = zi_marginal_loglik_laplace(args...; phi = Float64.(sp["phi"]),
                                              eigmin_floor = -Inf)
    @test ll_unguarded > Float64(c["r_loglik"]) + 100
end
