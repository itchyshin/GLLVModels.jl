using GLLVModels, Test, Logging
# Deterministic unmoved-start case for the non-ordinal AGHQ fitters: n_adapt=1 gives each
# AGHQ start one adaptation pass, so every run ends usable, unmoved (parameter_shift == 0)
# and unconverged (stop_reason :adaptation_cap). Data are arithmetic (no RNG stream).
const _G = GLLVModels
const _UF_P, _UF_N = 5, 40
_uf_cnt(t, s) = (3t + 7s + (t * s) % 5) % 6 + ((t + s) % 4 == 0 ? 4 : 0)
_uf_pos(t, s) = (3t + 7s) % 5 == 0 ? 0.0 : 0.4 + 0.3 * ((3t + 7s) % 5) + 0.1 * ((t * s) % 3)
const _UF_CASES = [
    ("Poisson",     () -> [_uf_cnt(t, s) for t in 1:_UF_P, s in 1:_UF_N],
        (Y; kw...) -> fit_poisson_gllvm(Y; K = 1, kw...)),
    ("Binomial",    () -> [Int((3t + 7s + (t * s) % 5) % 7 > 2) for t in 1:_UF_P, s in 1:_UF_N],
        (Y; kw...) -> fit_binomial_gllvm(Y; K = 1, kw...)),
    ("NB2",         () -> [_uf_cnt(t, s) for t in 1:_UF_P, s in 1:_UF_N],
        (Y; kw...) -> _G.fit_nb_gllvm_grouped_aghq(Y; K = 1, kw...)),
    ("Tweedie",     () -> [_uf_pos(t, s) for t in 1:_UF_P, s in 1:_UF_N],
        (Y; kw...) -> _G.fit_tweedie_gllvm_grouped_aghq(Y; K = 1, power = 1.5, kw...)),
    ("DeltaGamma",  () -> [_uf_pos(t, s) for t in 1:_UF_P, s in 1:_UF_N],
        (Y; kw...) -> _G.fit_delta_gamma_gllvm_aghq(Y; K = 1, kw...)),
    ("Gaussian",    () -> [sin(t) * cos(s) + 0.1 * ((t * s) % 7) for t in 1:_UF_P, s in 1:_UF_N],
        (Y; kw...) -> fit_gaussian_gllvm(Y; K = 1, kw...)),
]
@testset "AGHQ siblings: unmoved nonconverged run falls back" begin
@testset "$name" for (name, mk, fitf) in _UF_CASES
    Y = mk()
    fit = @test_logs (:warn, r"did not move from its start") match_mode = :any fitf(Y; aghq = 3, aghq_control = (n_adapt = 1,))
    @test fit.integration.actual === :laplace
    @test fit.integration.reason === :adaptation_cap
    @test fit.integration.reason === fit.integration.result.selected.stop_reason
    @test fit.integration.result.usable
    @test all(r -> r.parameter_shift == 0 && !r.converged, fit.integration.result.runs)
end
end
@testset "AGHQ Laplace-retention policy predicates" begin
    run(shift, conv, reason) = (parameter_shift = shift, converged = conv, stop_reason = reason)
    for r in (:adaptation_cap, :no_merit_descent, :optimizer_failed, :adaptation_failed,
              :warm_start_stagnation, :final_stationarity_lost)
        @test _G._aghq_unmoved_unconverged(run(0.0, false, r))
        @test !_G._aghq_unmoved_unconverged(run(0.5, false, r))
    end
    @test !_G._aghq_unmoved_unconverged(run(0.0, true, :converged))
    for r in (:adaptation_failed, :optimizer_failed, :finalization_failed)
        @test _G._aghq_failure_stop(run(5.0, false, r))
        @test !_G._aghq_failure_stop(run(5.0, true, r))
    end
    for r in (:adaptation_cap, :no_merit_descent, :objective_stagnation,
              :warm_start_stagnation, :final_stationarity_lost)
        @test !_G._aghq_failure_stop(run(5.0, false, r))
    end
end
