# refs #780: confint on a fit that did not converge warns and marks source_converged = false;
# Wald no longer reports pd_hessian = true when the supplied fit was non-converged.
using Test, GLLVModels, Random, Distributions

@testset "confint annotates a non-converged supplied fit (refs #780)" begin
    Random.seed!(3)
    Y = rand(Poisson(3), 5, 60)
    fit = fit_poisson_gllvm(Y; K = 1, iterations = 2)
    @test convergence_status(fit) === :not_converged

    @test_logs (:warn,) confint(fit, Y; method = :wald)
    ci = confint(fit, Y; method = :wald)
    @test ci.source_converged == false
    @test ci.pd_hessian == false

    ci_p = confint(fit, Y; method = :profile, parm = "beta[1]")
    @test ci_p.source_converged == false

    ci_b = confint(fit, Y; method = :bootstrap, n_boot = 20, seed = 1)
    @test ci_b.source_converged == false
end