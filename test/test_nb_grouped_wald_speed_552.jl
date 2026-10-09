# #552: per-species NB2 (`disp.group`), with and without covariates, was slow.
#   - The default Wald CI took the O(m²)-call `_fd_hessian` of the Laplace objective
#     (m = 71 on the issue's vegan::mite fit: about 10,000 Laplace passes, 37 s on main,
#     over 20 min at the reported commit). The NB2 grouped adapters now carry the
#     fitter's exact gradient and the Wald Hessian differences that (2m calls; 2 s).
#   - `fit_nb_gllvm_grouped` (no covariates) optimised with a finite-difference
#     gradient (494 s at p = 30 against tmb's 7 s); it now uses the exact gradient the
#     covariate fitter already used.
using Test, GLLVModels, Random, Distributions, LinearAlgebra

function _sim_nb_552(rng, p, n, K; r = 2.0, poisson = Int[])
    Λ = 0.6 .* randn(rng, p, K); β = 1.0 .+ 0.5 .* randn(rng, p); Z = randn(rng, K, n)
    W = randn(rng, n)
    X = zeros(p, n, p)
    for t in 1:p, s in 1:n
        X[t, s, t] = W[s]
    end
    γ = 0.3 .* randn(rng, p)
    rv = fill(r, p); rv[poisson] .= 1e8
    Y = [rand(rng, NegativeBinomial(rv[t], rv[t] / (rv[t] + exp(β[t] + γ[t] * W[s] +
                                                               dot(Λ[t, :], Z[:, s])))))
         for t in 1:p, s in 1:n]
    return Y, X
end

@testset "#552 NB2 grouped: Wald Hessian from the exact gradient" begin
    rng = MersenneTwister(552)
    p, n, K = 4, 50, 1
    Y, X = _sim_nb_552(rng, p, n, K; poisson = [4])
    Yf = Float64.(Y)

    @testset "covariate fit" begin
        fit = fit_nb_gllvm_grouped_cov(Y; X = X, K = K, group = collect(1:p))
        ad = GLLVModels._family_ci(fit, Yf; X = X)
        @test ad.grad !== nothing
        Hg = GLLVModels._grad_fd_hessian(ad.grad, ad.θ)
        Hf = GLLVModels._fd_hessian(ad.nll, ad.θ)
        @test Hg !== nothing
        @test maximum(abs.(Hg .- Hf)) <= 1e-4 * maximum(abs.(Hf))
        # The Wald step never evaluates the objective once the gradient route works:
        # an adapter whose `nll` throws still returns the same intervals.
        sel = collect(eachindex(ad.θ))
        ref = GLLVModels._family_wald(ad, sel, 0.95; hessian = Hf)
        boom = GLLVModels._FamilyCI(ad.θ, θ -> error("nll called"), ad.names, ad.kinds,
                                    ad.simulate, ad.refit, ad.boundary, ad.grad)
        new = GLLVModels._family_wald(boom, sel, 0.95)
        ok = isfinite.(ref.se)
        @test ok == isfinite.(new.se)
        @test any(ok)
        @test isapprox(new.se[ok], ref.se[ok]; rtol = 1e-3)
        # A group at the Poisson limit is conditioned out, as before.
        if fit.dispersion_boundary[4]
            @test !isfinite(new.se[end])
        end
        ci = confint(fit, Yf; method = :wald, X = X)
        @test isapprox(ci.se[ok], ref.se[ok]; rtol = 1e-3)
    end

    @testset "no-covariate fit" begin
        fit = fit_nb_gllvm_grouped(Y; K = K, group = collect(1:p))
        ad = GLLVModels._family_ci(fit, Yf)
        @test ad.grad !== nothing
        Hg = GLLVModels._grad_fd_hessian(ad.grad, ad.θ)
        Hf = GLLVModels._fd_hessian(ad.nll, ad.θ)
        @test maximum(abs.(Hg .- Hf)) <= 1e-4 * maximum(abs.(Hf))
    end

    @testset "failed gradient falls back to _fd_hessian" begin
        fit = fit_nb_gllvm_grouped(Y; K = K, group = collect(1:p))
        ad = GLLVModels._family_ci(fit, Yf)
        bad = GLLVModels._FamilyCI(ad.θ, ad.nll, ad.names, ad.kinds, ad.simulate,
                                   ad.refit, ad.boundary, θ -> nothing)
        @test GLLVModels._family_hessian(bad) == GLLVModels._fd_hessian(ad.nll, ad.θ)
    end
end

@testset "#552 no-covariate NB2 grouped gradient (with offset) matches finite differences" begin
    rng = MersenneTwister(5520)
    p, n, K = 4, 30, 1
    Y, _ = _sim_nb_552(rng, p, n, K)
    O = 0.2 .* randn(rng, p, n)
    rr = GLLVModels.rr_theta_len(p, K)
    θ = vcat(log.(vec(sum(Y; dims = 2)) ./ n .+ 0.5), 0.3 .* ones(rr), log.([1.0, 3.0, 10.0, 30.0]))
    nll(θv) = -GLLVModels.nb_grouped_marginal_loglik_laplace(
        Y, GLLVModels.unpack_lambda(θv[(p + 1):(p + rr)], p, K), θv[1:p],
        exp.(θv[(p + rr + 1):end]); offset = O)
    g = GLLVModels._nb_grouped_cov_negll_grad(Y, zeros(p, n, 0), θ, p, 0, K, rr, p,
                                              collect(1:p), GLLVModels.LogLink(), nothing,
                                              :observed, 100, 1e-9; offset = O)
    @test g !== nothing
    h = 1e-5
    gfd = [(nll(θ .+ h .* (1:length(θ) .== i)) - nll(θ .- h .* (1:length(θ) .== i))) / (2h)
           for i in eachindex(θ)]
    @test isapprox(g, gfd; rtol = 1e-5, atol = 1e-6)
end

@testset "#552 no-covariate NB2 grouped fit uses the exact gradient" begin
    rng = MersenneTwister(5521)
    p, n, K = 5, 60, 1
    Y, _ = _sim_nb_552(rng, p, n, K; poisson = [2])
    fit = fit_nb_gllvm_grouped(Y; K = K, group = collect(1:p))
    @test fit.converged
    ad = GLLVModels._family_ci(fit, Float64.(Y))
    g = ad.grad(ad.θ)
    @test g !== nothing
    @test maximum(abs, g) < 1e-3
end
