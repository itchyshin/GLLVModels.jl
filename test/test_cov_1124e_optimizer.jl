# Coverage: outer-optimizer receipts in src/optimizer_selection.jl.
# Error branches and record assembly. No fit.
using Test
using GLLVModels
using Optim

grab(f) = try
    f()
    nothing
catch e
    e
end

@testset "optimizer selection helpers (1124e)" begin
    @test GLLVModels.optimizer_info(1) === nothing
    @test GLLVModels.optimizer_info((other = 1,)) === nothing

    @test GLLVModels._refuse_optimizer(nothing, "fit_x", "this route") === nothing
    e = grab(() -> GLLVModels._refuse_optimizer(:lbfgs, "fit_x", "this route"))
    @test e isa ArgumentError
    @test occursin("fit_x: an explicit outer optimizer is not supported for this route", e.msg)

    default = Optim.LBFGS()
    @test GLLVModels._optimizer_method(nothing, default; route = "fit_y") === default
    @test GLLVModels._optimizer_method(:lbfgs, default; route = "fit_y") isa Optim.LBFGS
    @test GLLVModels._optimizer_method(:bfgs, default; route = "fit_y") isa Optim.BFGS
    kept = Optim.BFGS()
    @test GLLVModels._optimizer_method(kept, default; route = "fit_y") === kept
    lbfgs = Optim.LBFGS()
    @test GLLVModels._optimizer_method(lbfgs, Optim.BFGS(); route = "fit_y") === lbfgs
    e = grab(() -> GLLVModels._optimizer_method(:newton, default; route = "fit_y"))
    @test e isa ArgumentError
    @test occursin("fit_y: optimizer must be nothing, :lbfgs, :bfgs", e.msg)
    @test occursin("got String", grab(() -> GLLVModels._optimizer_method("lbfgs", default; route = "fit_y")).msg)

    stages = GLLVModels.OptimizerStage[]
    @test GLLVModels._last_accepted_optimizer(stages) === nothing
    GLLVModels._optimizer_stage!(stages, :main, Optim.LBFGS(), :finite, (; g_tol = 1e-5,), :rejected;
                                 accepted = false)
    @test length(stages) == 1
    @test stages[1].name === :main
    @test stages[1].accepted == false
    @test stages[1].result === :rejected
    @test GLLVModels._last_accepted_optimizer(stages) === nothing
    e = grab(() -> GLLVModels._optimizer_record(nothing, stages))
    @test e isa ArgumentError
    @test occursin("cannot record an optimizer before an outer solve", e.msg)

    forced = GLLVModels._optimizer_record(nothing, stages; actual = Optim.BFGS())
    @test forced.actual == Optim.BFGS
    @test forced.requested === nothing
    @test forced.derivative === :unknown

    GLLVModels._optimizer_stage!(stages, :restart, Optim.BFGS(), :forward, (; iterations = 3,), :ok)
    @test GLLVModels._last_accepted_optimizer(stages) isa Optim.BFGS
    rec = GLLVModels._optimizer_record(:bfgs, stages; controls = (; iterations = 3,),
                                       derivative = :forward)
    @test rec.requested === :bfgs
    @test rec.actual == Optim.BFGS
    @test rec.derivative === :forward
    @test rec.controls == (; iterations = 3,)
    @test length(rec.stages) == 2
    @test rec.stages[1].accepted == false
    @test rec.stages[2].accepted == true
    @test GLLVModels.optimizer_info((optimizer_info = rec,)) === rec

    @test GLLVModels._check_optimizer_refit((other = 1,), "profile") === nothing
    @test GLLVModels._check_optimizer_refit((optimizer_info = nothing,), "profile") === nothing
    @test GLLVModels._check_optimizer_refit((optimizer_info = forced,), "profile") === nothing
    e = grab(() -> GLLVModels._check_optimizer_refit((optimizer_info = rec,), "bootstrap"))
    @test e isa ArgumentError
    @test occursin("bootstrap is not available after an explicit optimizer selection", e.msg)
    @test occursin("optimizer=nothing", e.msg)
end
