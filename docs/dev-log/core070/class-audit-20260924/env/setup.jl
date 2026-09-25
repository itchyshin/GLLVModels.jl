using Pkg
Pkg.activate(@__DIR__)
Pkg.offline(true)
Pkg.develop(path="/Users/z3437171/local-scratch/gllvm-nb2-finite-20260924")
t = time()
@eval using GLLVModels
println("loaded in ", round(time()-t, digits=1), " s")
