# Lane cursor-cloud-1125y6: optimizer_selection.jl internal helpers (not full fit spy).
using Test, GLLVModels, Optim, Random

const G = GLLVModels

@testset "1125y6 optimizer selection helpers" begin
    @testset "_optimizer_method" begin
        @test G._optimizer_method(nothing, Optim.LBFGS(); route = "t") isa Optim.LBFGS
        @test G._optimizer_method(:lbfgs, Optim.BFGS(); route = "t") isa Optim.LBFGS
        @test G._optimizer_method(:bfgs, Optim.LBFGS(); route = "t") isa Optim.BFGS
        m = Optim.LBFGS()
        @test G._optimizer_method(m, Optim.BFGS(); route = "t") === m
        @test_throws ArgumentError G._optimizer_method(:adam, Optim.LBFGS(); route = "fit")
    end

    @testset "_refuse_optimizer" begin
        @test G._refuse_optimizer(nothing, "route", "reason") === nothing
        @test_throws ArgumentError G._refuse_optimizer(:bfgs, "em_fit", "EM slice")
    end

    @testset "OptimizerStage / Record" begin
        res = Optim.optimize(x -> sum(abs2, x), zeros(2), Optim.LBFGS())
        stages = G.OptimizerStage[]
        G._optimizer_stage!(stages, :main, Optim.LBFGS(), :forward, (; g_tol = 1e-8), res)
        @test length(stages) == 1
        @test stages[1].accepted
        rec = G._optimizer_record(:lbfgs, stages; derivative = :forward)
        @test rec.requested === :lbfgs
        @test rec.derivative === :forward
        @test G._last_accepted_optimizer(stages) isa Optim.LBFGS
        fit_dummy = fit_gaussian_gllvm(randn(2, 6); K = 1, iterations = 3, optimizer = :lbfgs)
        oi = G.optimizer_info(fit_dummy)
        @test oi === nothing || oi isa G.OptimizerRecord
    end

    @testset "_check_optimizer_refit" begin
        rng = MersenneTwister(5)
        Y = randn(rng, 3, 12)
        fit0 = fit_gaussian_gllvm(Y; K = 1)
        @test G._check_optimizer_refit(fit0, "profile_ci") === nothing
        info = G.optimizer_info(fit0)
        if info !== nothing && info.requested !== nothing
            @test_throws ArgumentError G._check_optimizer_refit(fit0, "profile_ci")
        end
        fit_opt = fit_gaussian_gllvm(Y; K = 1, optimizer = :lbfgs)
        @test_throws ArgumentError G._check_optimizer_refit(fit_opt, "bootstrap_ci")
    end
end