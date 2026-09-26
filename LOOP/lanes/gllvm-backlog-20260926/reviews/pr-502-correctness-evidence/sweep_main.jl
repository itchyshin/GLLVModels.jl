using Test, GLLVModels, Optim, Printf
const OUT = open(joinpath(@__DIR__, "sweep_main_verdicts.log"), "w")
const CUR = Ref("")
@eval GLLVModels function _fit_verdict(res)
    gres = Optim.g_residual(res); gt = Optim.g_tol(res); nll = Optim.minimum(res)
    thr = max(gt, gt * abs(nll))
    old = Optim.converged(res); new = old && _gradient_criterion_met(res)
    Base.println($(OUT), join((($(CUR))[], @sprintf("%.4g", nll), @sprintf("%.3e", gres), @sprintf("%.3e", thr),
        @sprintf("%.3e", gres / thr), string(Int(res.stopped_by.x_converged), Int(res.stopped_by.f_converged), Int(res.stopped_by.g_converged)),
        Optim.iterations(res), Int(old), Int(new)), "\t")); flush($(OUT))
    return _fit_verdict(Optim.minimum(res), old, Optim.iterations(res))
end
files = split(ENV["SWEEP_FILES"])
testdir = joinpath(ENV["WT"], "test")
cd(testdir)
for f in files
    CUR[] = f
    t = @elapsed begin
        ts = try
            @testset "$f" begin
                Base.include(Module(), joinpath(testdir, f))
            end
        catch e
            println("FILE-ERROR $f: ", sprint(showerror, e)[1:min(end, 300)])
        end
    end
    println("DONE $f in $(round(t, digits=1))s"); flush(stdout)
end
close(OUT)
