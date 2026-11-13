using GLLVModels, Test, Random, Logging
# Deterministic reproduction of the unmoved-start case: n_adapt=1 gives each AGHQ start a
# single adaptation pass, so the outer driver ends usable but unmoved and unconverged
# (stop_reason :adaptation_cap). No dependence on BLAS kernel or Julia RNG stream.
@testset "Ordinal AGHQ: unmoved nonconverged run falls back to Laplace" begin
    G = GLLVModels
    p, n = 5, 40
    Y = [1 + ((3t + 7s + (t * s) % 5) % 4 > 1) + ((5t + 2s + (t + s) % 3) % 7 > 4) for t in 1:p, s in 1:n]
    base = G.fit_ordinal_gllvm_pertrait(Y; K = 1)
    fit = @test_logs (:warn, r"did not move from its start") match_mode = :any G.fit_ordinal_gllvm_pertrait_aghq(Y; K = 1, aghq = 3, aghq_control = (n_adapt = 1,))
    @test fit.integration.actual === :laplace
    # The record carries the run's own stop reason and stays distinguishable from the
    # no-usable-run fallback (result.usable) and from a moved run (parameter_shift).
    @test fit.integration.reason === :adaptation_cap
    @test fit.integration.reason === fit.integration.result.selected.stop_reason
    @test fit.integration.result.usable && fit.integration.result.selected.parameter_shift == 0
    @test fit.Λ == base.Λ && fit.β == base.β && fit.loglik == base.loglik
end
