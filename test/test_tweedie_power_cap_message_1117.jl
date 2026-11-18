using GLLVModels, Test, Random, Logging

# #1117: `fit_tweedie_gllvm_grouped(; power_group = :species)` that stops at the iteration
# cap used to say nothing. It now warns, only when at least one fitted power is within
# 1e-3 of 1 or 2. The 1e-3 distance is a message trigger, not a fit tolerance.
@testset "Tweedie per-species power: iteration-cap message (#1117)" begin
    Random.seed!(1117)
    p, n = 6, 20
    Y = [GLLVModels._tweedie_sample(exp(0.2 * randn()), 1.0, 1.5, Random.default_rng())
         for _ in 1:p, _ in 1:n]
    run(; kw...) = fit_tweedie_gllvm_grouped(Y; K = 1, power_group = :species,
                                             newton_maxiter = 20, kw...)
    nedge(f) = count(q -> min(q - 1, 2 - q) < 1e-3, f.power)
    quiet(f) = with_logger(f, NullLogger())

    @testset "cap hit with powers at the edge: one extra warning" begin
        pinit = [1.99999, 1.00001, fill(1.5, p - 2)...]
        fit = quiet(() -> run(; iterations = 5, power_init = pinit))
        @test fit isa GLLVModels.TweediePerTraitPowerFit
        @test fit.iterations == 5 && !fit.converged
        k = nedge(fit)
        @test k >= 1
        @test_logs (:warn, Regex("stopped at the iteration cap.*$k of $p fitted powers.*iterations = 2000")) match_mode = :any run(; iterations = 5, power_init = pinit)
    end

    @testset "cap hit, no power near the edge: no message" begin
        fit = quiet(() -> run(; iterations = 5))
        @test fit.iterations == 5 && !fit.converged
        @test nedge(fit) == 0
        @test_logs run(; iterations = 5)
    end

    @testset "converged fit with edge powers: no message" begin
        pinit = [1.99999, 1.00001, fill(1.5, p - 2)...]
        fit = quiet(() -> run(; iterations = 5, power_init = pinit, g_tol = 1e6))
        @test fit.converged && fit.iterations < 5
        @test_logs run(; iterations = 5, power_init = pinit, g_tol = 1e6)
    end
end
