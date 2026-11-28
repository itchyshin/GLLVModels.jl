using GLLVModels, Test, Random, LinearAlgebra, Logging

# #1185: a constrained refit that beats the reported logLik (negative deviance)
# must warn, naming the routine. Forced by lowering the stored logLik.
@testset "profile: better-optimum warning (#1185)" begin
    Random.seed!(9)
    p, K, n = 6, 1, 60
    y = reshape(randn(p) .* 0.6, p, K) * randn(K, n) + 0.5 * randn(p, n)
    fit = fit_gaussian_gllvm(y; K = K)
    lowered = GLLVModels.GllvmFit(fit.model, fit.pars, fit.logLik - 6.0, fit.n_iter,
                                  fit.converged, fit.optim_result, fit.cputime,
                                  fit.integration, fit.optimizer_info)
    @test_logs (:warn, r"profile_ci: .*not the maximum") match_mode = :any begin
        GLLVModels.profile_ci(lowered, 1; y = y)
    end
    @test_logs min_level = Logging.Warn GLLVModels.profile_ci(fit, 1; y = y)
end

# #1185: profile_ci_derived warns through the same helper.
@testset "profile_ci_derived: better-optimum warning (#1185)" begin
    Random.seed!(9)
    p, K, n = 6, 1, 60
    y = reshape(randn(p) .* 0.6, p, K) * randn(K, n) + 0.5 * randn(p, n)
    fit = fit_gaussian_gllvm(y; K = K)
    lowered = GLLVModels.GllvmFit(fit.model, fit.pars, fit.logLik - 6.0, fit.n_iter,
                                  fit.converged, fit.optim_result, fit.cputime,
                                  fit.integration, fit.optimizer_info)
    g = θ -> θ[1]
    @test_logs (:warn, r"profile_ci_derived: .*not the maximum") match_mode = :any begin
        GLLVModels.profile_ci_derived(lowered, g; y = y)
    end
end

# #1185: the family route (confint(method = :profile)) warns when the reported
# optimum is displaced (here: a shifted intercept, so a constrained refit beats it).
@testset "confint(method = :profile): better-optimum warning (#1185)" begin
    Random.seed!(11)
    p, K, n = 5, 1, 50
    Y = [rand() < 0.5 ? 0 : rand(1:6) for _ in 1:p, _ in 1:n]
    fit = fit_poisson_gllvm(Y; K = K)
    β = copy(fit.β); β[1] += 1.0
    ll_bad = GLLVModels.poisson_marginal_loglik_laplace(Y, fit.Λ, β, fit.link;
                                                        hessian = fit.hessian)
    bad = GLLVModels.PoissonFit(β, fit.Λ, fit.link, ll_bad, fit.converged,
                                fit.iterations, fit.alpha_lv, fit.theta_packed, fit.hessian,
                                fit.integration, fit.offset, fit.weights, fit.optimizer_info)
    @test_logs (:warn, r"not the maximum") match_mode = :any begin
        GLLVModels.confint(bad, Y; method = :profile, parm = "beta[1]")
    end
end
