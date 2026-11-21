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
