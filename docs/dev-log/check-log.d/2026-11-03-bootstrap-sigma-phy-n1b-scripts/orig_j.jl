# N1b: original Julia fit, simulate paired batch (MersenneTwister(2001)), refit with wrapper, dump.
using GLLVModels, DelimitedFiles, Printf, Random, LinearAlgebra
const G = GLLVModels
ds = ARGS[1]; root = ARGS[2]; od = ARGS[3]; mode = length(ARGS) >= 4 ? ARGS[4] : "full"
mkpath(od)
dd = joinpath(root, "data", "d" * ds)
Y = readdlm(joinpath(dd, "Y.csv"), ',', Float64)
sid = Int.(vec(readdlm(joinpath(dd, "sid.csv"), ',')))
pp = PrecisionPhy(G.augmented_phy(strip(read(joinpath(dd, "tree.nwk"), String))))
quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
t0 = time()
fit = quiet() do
    G.fit_precision_multivariate(Y, pp; rank=1, mode=:explicitunique, residual_mode=:shared, species_id=sid)
end
flat(fb) = begin
    e = Dict(G._bsig_entries(fb, [:phy], [:Sigma, :R, :communality]))
    vcat(vec(e[:Sigma_phy]), vec(e[:R_phy]), vec(e[:communality_phy]))
end
@printf("orig fit_s %.1f loglik %.12g conv %s hess_pd %s grad %.3g\n", time()-t0, fit.loglik, fit.converged, fit.hessian_positive_definite, fit.gradient_norm)
open(joinpath(od, "origJ_$(ds).txt"), "w") do io
    println(io, "converged=", fit.converged); println(io, "hess_pd=", fit.hessian_positive_definite)
    println(io, "loglik=", @sprintf("%.15g", fit.loglik)); println(io, "gradient_norm=", fit.gradient_norm)
    println(io, "unique=", join([@sprintf("%.15g", v) for v in fit.phylo_unique_variance], ","))
    println(io, "flat=", join([@sprintf("%.17g", v) for v in flat(fit)], ","))
end
mode == "orig" && (println("DONE orig"); exit(0))
Ys = G._bsig_pmv_simulate_batch(Random.MersenneTwister(2001), fit, 999)
open(joinpath(od, "allY_$(ds).csv"), "w") do io
    for b in 1:999
        println(io, join([2001; b; [@sprintf("%.17g", v) for v in vec(Ys[b])]], ","))
    end
end
println("Y written"); flush(stdout)
nanrow = fill(NaN, 55)
J = fill(NaN, 999, 55)
open(joinpath(od, "jinfo_$(ds).csv"), "w") do io
    println(io, "b,ok,conv_driver,reason,iters,gradnorm,loglik,hess_pd,thrown")
    for b in 1:999
        b % 50 == 0 && (println("rep $b ", time()-t0); flush(stdout))
        r = try
            quiet() do
                G.fit_precision_multivariate(Ys[b], fit.phy; rank=fit.rank, mode=fit.mode, residual_mode=fit.residual_mode,
                    species_id=fit.species_id, X=fit.mean_design, coefficient_names=fit.coefficient_names, iterations=3000)
            end
        catch e
            e
        end
        if r isa G.PrecisionMultivariateFit
            # wrapper rule: finite objective and FD gradient max-norm <= 1e-3 (driver default is 1e-5, iterations 400)
            ok = isfinite(r.loglik) && isfinite(r.gradient_norm) && r.gradient_norm <= 1e-3
            if ok
                try J[b, :] = flat(r) catch; ok = false end
            end
            println(io, join([b, ok, r.converged, r.stopping_reason, r.iterations, @sprintf("%.6g", r.gradient_norm), @sprintf("%.12f", r.loglik), r.hessian_positive_definite, false], ","))
        else
            println(io, join([b, false, false, replace(string(typeof(r)), "," => ";"), 0, "NaN", "NaN", false, true], ","))
        end
        flush(io)
    end
end
open(joinpath(od, "jdraws_$(ds).csv"), "w") do io
    for b in 1:999
        println(io, join([@sprintf("%.17g", v) for v in J[b, :]], ","))
    end
end
@printf("DONE total_s %.1f\n", time()-t0)
