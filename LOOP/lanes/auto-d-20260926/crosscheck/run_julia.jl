using Random, Distributions, DelimitedFiles
using GLLVModels

include(joinpath(dirname(@__DIR__), "pilot", "pilot_sim.jl"))

outdir = @__DIR__

datasets = [
    ("poisson",  100, 8, 2, MersenneTwister(101), Poisson()),
    ("binomial", 120, 8, 1, MersenneTwister(102), Binomial()),
    ("gaussian",  80, 8, 2, MersenneTwister(103), Normal()),
]

open(joinpath(outdir, "julia_results.csv"), "w") do io
    println(io, "dataset,K,status,loglik,nparams,aic,bic")
    for (fam, n, p, K, rng, famobj) in datasets
        Y = simulate(fam, n, p, K, rng)
        # save as CSV: species x sites
        writedlm(joinpath(outdir, "Y_$(fam).csv"), Y, ',')

        sel = select_lv(Y; family = famobj, Kmax = 4)
        println("== $fam ==")
        show(stdout, MIME("text/plain"), sel)
        println()
        println("chosen K = ", sel.best_k)

        for a in sel.attempts
            idx = findfirst(==(a.K), sel.K)
            npar = idx === nothing ? "" : sel.nparams[idx]
            aicv = idx === nothing ? "" : sel.aic[idx]
            bicv = idx === nothing ? "" : sel.bic[idx]
            println(io, "$fam,$(a.K),$(a.status),$(a.loglik),$npar,$aicv,$bicv")
        end
    end
end
println("DONE")
