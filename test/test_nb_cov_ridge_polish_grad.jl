using GLLVModels, Test, Random, Distributions, Optim, LinearAlgebra
const GMR = GLLVModels

# Gradient branch of `_nb_poisson_ridge_polish` (#615 x #658), reached through
# `fit_nb_gllvm_grouped_cov`. Small NB2 data with covariates; species 1-2 are nearly
# Poisson (r = 1e6), so a capped fit stalls with r > 1e3 and the polish fires.
@testset "NB2 grouped-cov ridge polish uses the exact gradient" begin
    rng = MersenneTwister(615)
    p, n, K, q = 6, 50, 1, 1
    X = randn(rng, p, n, q)
    z = randn(rng, n)
    Λt = [0.6, -0.4, 0.3, 0.5, -0.2, 0.4]
    rt = [1e6, 1e6, 3.0, 2.0, 4.0, 3.0]
    η = [0.5 + 0.3 * X[t, s, 1] + Λt[t] * z[s] for t in 1:p, s in 1:n]
    Y = [rand(rng, NegativeBinomial(rt[t], rt[t] / (rt[t] + exp(η[t, s])))) for t in 1:p, s in 1:n]
    group = collect(1:p)
    cap = 40

    # (a) the polish ran: the capped fit did not converge on its own, and a species
    # now sits exactly at the polish's fixed value r = 1e10.
    fit = GMR.fit_nb_gllvm_grouped_cov(Y; X = X, K = K, group = group, iterations = cap)
    @test any(r -> isapprox(r, 1e10; rtol = 1e-9), fit.r_group)
    @test fit.iterations > cap && fit.converged   # polish adds its iterations and converges
    @test isfinite(fit.loglik)

    # Rebuild the fitter's objective and exact gradient to compare the two polish routes.
    rr = GMR.rr_theta_len(p, K); G = p; gidx = group
    Yc = Integer.(Y); msk = GMR._resolve_obs_mask(nothing, Y)
    function negll(θ)
        γ = θ[(p + 1):(p + q)]
        Λ = GMR.unpack_lambda(θ[(p + q + 1):(p + q + rr)], p, K)
        rv = exp.(θ[(p + q + rr + 1):(p + q + rr + G)])
        v = try
            -GMR.nb_grouped_marginal_loglik_laplace(Yc, Λ, θ[1:p], rv; link = LogLink(),
                mask = msk, offset = GMR._build_offset(X, γ), hessian = :observed,
                maxiter = 100, tol = 1e-9)
        catch
            return 1e12
        end
        isfinite(v) ? v : 1e12
    end
    grad(θ) = GMR._nb_grouped_cov_negll_grad(Yc, X, θ, p, q, K, rr, G, gidx, LogLink(), msk,
                                             :observed, 100, 1e-9)
    ls = GMR._COV_BFGS()
    opts = Optim.Options(g_tol = 1e-5, iterations = cap)
    θ0 = vcat(fit.β, fit.γ, GMR.pack_lambda(fit.Λ), log.(min.(fit.r_group, 1e6)))
    # a stalled start: species 1-2 at r = 1e5 (> 1e3), the rest at the fit's values
    θ0[(p + q + rr + 1):(p + q + rr + 2)] .= log(1e5)
    res = GMR._optimize_with_analytic(negll, grad, θ0, ls, Optim.Options(g_tol = 1e-12, iterations = 2))
    first_log_r = p + q + rr + 1
    @test !Optim.converged(res)
    @test all(>(1e3), exp.(Optim.minimizer(res)[first_log_r:first_log_r + 1]))

    # (b) restricted exact gradient vs central finite difference on the free coordinates
    θs = copy(Optim.minimizer(res)); fixed = first_log_r - 1 .+ [1, 2]
    θs[fixed] .= log(1e10); free = setdiff(eachindex(θs), fixed)
    sub(x) = negll(setindex!(copy(θs), x, free))
    gx = grad(θs)[free]
    x0 = θs[free]; h = 1e-6
    gfd = [(sub(setindex!(copy(x0), x0[i] + h, i)) - sub(setindex!(copy(x0), x0[i] - h, i))) / 2h
           for i in eachindex(x0)]
    @test maximum(abs.(gx .- gfd)) / max(1, maximum(abs.(gfd))) < 1e-6

    # (c) gradient polish is no worse than the finite-difference polish
    θg, fg, cg, _ = GMR._nb_poisson_ridge_polish(negll, res, ls, opts, first_log_r; grad = grad)
    θf, ff, cf, _ = GMR._nb_poisson_ridge_polish(negll, res, ls, opts, first_log_r)
    @test θg[fixed] == fill(log(1e10), 2)      # polish fired on the gradient route
    @test fg <= ff + 1e-6
end
