using GLLVModels, Test, Random, Distributions, LinearAlgebra
const GM = GLLVModels

# #505 (scalar NB1): fit_nb1_gllvm trusted Optim.converged, which also fires on a
# zero-length step far from a stationary point. The grouped fitters judge converged
# with the gradient test; the scalar fitter now does too (same `_nb1_grouped_g_met`).

# max |central-difference gradient| of the scalar NB1 negative log-likelihood at the fit
function _nb1_scalar_gmax(fit, Y, K)
    p = size(Y, 1)
    rr = GM.rr_theta_len(p, K)
    θ = vcat(fit.β, GM.pack_lambda(fit.Λ), log(fit.φ))
    f(θ) = -GM.nb1_marginal_loglik_laplace(Y, GM.unpack_lambda(θ[(p + 1):(p + rr)], p, K),
        θ[1:p], exp(θ[p + rr + 1]); hessian = fit.hessian)
    h = 1e-5
    g = map(eachindex(θ)) do i
        e = zeros(length(θ)); e[i] = h
        (f(θ .+ e) - f(θ .- e)) / 2h
    end
    return maximum(abs, g), f(θ)
end

@testset "fit_nb1_gllvm: converged requires a small gradient (#505)" begin
    p, K, n, φ = 8, 2, 30, 0.05
    nconv = 0
    for seed in 1:12
        Random.seed!(seed)
        β = 0.3 .* randn(p)
        Λ = 0.5 .* randn(p, K)
        Z = randn(K, n)
        Y = [rand(NegativeBinomial(exp(β[t] + dot(Λ[t, :], Z[:, s])) / φ, 1 / (1 + φ)))
             for t in 1:p, s in 1:n]
        fit = GM.fit_nb1_gllvm(Y; K = K)
        if fit.converged
            nconv += 1
            gmax, nll = _nb1_scalar_gmax(fit, Y, K)
            @test gmax <= 10 * max(1e-5, 1e-5 * abs(nll))
        end
    end
    @test nconv >= 1   # the property is not vacuous
end
