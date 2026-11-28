using GLLVModels, Test, Random, LinearAlgebra, Distributions, Optim

# #505 (the remaining #485 class), `fit_nb1_gllvm_grouped_cov`: Optim can stop on a
# zero-length line-search step, count that as f_converged and report `converged = true`
# while the gradient is still large. `fit_nb1_gllvm_grouped` already judges the fit with
# `_nb1_grouped_g_met`; the covariate route must do the same.
#
# Platform robustness (as in test_fit_verdict_gradient.jl): a live-RNG fixture is never
# asserted to land on one outcome; the relation "converged => small independent
# gradient" is checked over several seeds. The unit contract of the helper is covered
# in test_fit_verdict_gradient.jl.

function _nb1cov505_fixture(seed; p = 5, n = 80, K = 2, sd = 0.7, φ = 1.0)
    rng = MersenneTwister(seed)
    β = 0.5 .+ rand(rng, p)
    Λ = sd .* randn(rng, p, K)
    for j in 1:K, i in 1:(j - 1)
        Λ[i, j] = 0.0
    end
    for j in 1:K
        Λ[j, j] = abs(Λ[j, j]) + 0.3 * sd
    end
    X = randn(rng, p, n, 1)
    γ = [0.4]
    Z = randn(rng, K, n)
    Y = [begin
            μ = exp(clamp(β[t] + dot(Λ[t, :], Z[:, i]) + γ[1] * X[t, i, 1], -30, 30))
            rand(rng, NegativeBinomial(μ / φ, 1 / (1 + φ)))
         end for t in 1:p, i in 1:n]
    return Y, X
end

# Largest central-difference gradient of the NB1 grouped-cov negative log-likelihood at
# the fitted point, from the public marginal, not from Optim's own bookkeeping.
function _nb1cov505_max_grad(Y, X, fit)
    p, K = size(fit.Λ)
    q = length(fit.γ)
    rr = GLLVModels.rr_theta_len(p, K)
    G = length(fit.φ)
    θ = vcat(fit.β, fit.γ, GLLVModels.pack_lambda(fit.Λ), log.(fit.φ))
    function f(θ)
        β_ = θ[1:p]
        γ_ = θ[(p + 1):(p + q)]
        Λ_ = GLLVModels.unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        φg_ = exp.(θ[(p + q + rr + 1):(p + q + rr + G)])
        φvec = [φg_[fit.group[t]] for t in 1:p]
        return -GLLVModels.nb1_grouped_marginal_loglik_laplace(Y, Λ_, β_, φvec;
                   offset = GLLVModels._build_offset(X, γ_), hessian = fit.hessian)
    end
    h = 1e-5
    return maximum(eachindex(θ)) do j
        e = zeros(length(θ)); e[j] = h
        abs(f(θ .+ e) - f(θ .- e)) / (2h)
    end
end

@testset "#505 NB1 grouped-cov: a zero-length-step stop is not convergence" begin
    @testset "relation over several seeds: fit.converged implies a small gradient" begin
        for seed in 101:112
            Y, X = _nb1cov505_fixture(seed)
            fit = GLLVModels.fit_nb1_gllvm_grouped_cov(Y; X = X, K = 2, group = collect(1:5))
            if fit.converged
                g = _nb1cov505_max_grad(Y, X, fit)
                @test (seed, g < 1e-2) == (seed, true)
            end
        end
    end
end
