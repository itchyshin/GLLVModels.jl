using GLLVModels, Test, Logging
# Deterministic fit-level test of the AGHQ ordinal failure fallbacks (unmoved and moved).
# The failure is injected through the test-only `_adapt_hook` seam of
# `fit_ordinal_gllvm_pertrait_aghq` (default `identity`, so production behaviour is
# unchanged): the hook wraps the site-adaptation callback and throws on one chosen call.
# The data are pure arithmetic (no RNG), well-conditioned (Laplace max|Λ| < 3, converged), and
# the AGHQ run is single-start, so the call sequence does not depend on BLAS kernel or Julia RNG.
function _ord_det_fixture(p = 6, n = 60)
    lam = [0.9, 1.3, 0.7, 1.1, 1.5, 0.8]; b = [-0.3, 0.2, 0.0, 0.4, -0.2, 0.1]
    z = [((s * 0.6180339887498949) % 1 - 0.5) * 3.4 for s in 1:n]
    Y = zeros(Int, p, n)
    for t in 1:p, s in 1:n
        u = clamp(abs(sin(12.9898t + 78.233s) * 43758.5453) % 1, 1e-3, 1 - 1e-3)
        eta = b[t] + lam[t] * z[s] + log(u / (1 - u))
        Y[t, s] = 1 + (eta > -0.7) + (eta > 0.9)
    end
    return Y
end
# Throw on exactly the `fail_call`-th adaptation call (1-based); every other call passes through.
function _fail_once(fail_call)
    count = Ref(0)
    return adapt -> theta -> begin
        count[] += 1
        count[] == fail_call && error("injected AGHQ quadrature failure (call $(count[]))")
        adapt(theta)
    end
end
@testset "Ordinal AGHQ: failure fallbacks (deterministic, fit level)" begin
    G = GLLVModels
    Y = _ord_det_fixture()
    base = G.fit_ordinal_gllvm_pertrait(Y; K = 1)
    @test base.converged && maximum(abs, base.Λ) < 5
    ctl = (multistart = false,)
    fit1(; kw...) = G.fit_ordinal_gllvm_pertrait_aghq(Y; K = 1, aghq = 3, kw...)

    @testset "failure after the run has moved: retain Laplace, record :adaptation_failed" begin
        fit = @test_logs (:warn, r"stopped on a failure before converging") match_mode = :any fit1(aghq_control = ctl, _adapt_hook = _fail_once(5))
        run = fit.integration.result.runs[1]
        # preconditions: the injected error ended a usable run that had already moved
        @test run.stop_reason === :adaptation_failed
        @test occursin("injected AGHQ quadrature failure", string(run.error))
        @test run.usable && !run.converged && run.parameter_shift > 0
        # outcome
        @test fit.integration.actual === :laplace
        @test fit.integration.reason === :adaptation_failed
        @test fit.converged == base.converged
        @test fit.Λ == base.Λ && fit.β == base.β && fit.loglik == base.loglik
    end

    @testset "failure before any movement: unmoved branch also retains Laplace" begin
        fit = @test_logs (:warn, r"did not move from its start") match_mode = :any fit1(aghq_control = ctl, _adapt_hook = _fail_once(3))
        run = fit.integration.result.runs[1]
        @test run.stop_reason === :adaptation_failed && run.usable && run.parameter_shift == 0
        @test fit.integration.actual === :laplace && fit.integration.reason === :adaptation_failed
        @test fit.Λ == base.Λ && fit.β == base.β && fit.loglik == base.loglik
    end

    @testset "negative control: no injected failure keeps the AGHQ fit" begin
        fit = fit1(aghq_control = ctl)
        @test fit.integration.actual === :aghq
        @test fit.integration.reason !== :adaptation_failed
        @test fit.loglik != base.loglik
    end

    @testset "negative control: a moved run ended by the adaptation budget is not replaced" begin
        fit = fit1(aghq_control = (multistart = false, n_adapt = 4))
        run = fit.integration.result.runs[1]
        @test run.stop_reason === :adaptation_cap && run.parameter_shift > 0 && !run.converged
        @test fit.integration.actual === :aghq && fit.integration.reason === :adaptation_cap
    end

    @testset "identity hook is a no-op" begin
        a = fit1(aghq_control = ctl); b = fit1(aghq_control = ctl, _adapt_hook = identity)
        @test a.theta_packed == b.theta_packed && a.loglik == b.loglik
    end
end
