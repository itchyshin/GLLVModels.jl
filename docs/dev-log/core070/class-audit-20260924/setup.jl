using Pkg
Pkg.activate("/tmp/claude-503/audit/env")
Pkg.develop(path="/Users/z3437171/local-scratch/gllvm-nb2-finite-20260924")
Pkg.add(["Optim", "Distributions", "ForwardDiff"])
Pkg.precompile()
using GLLVModels
println("OK ", pathof(GLLVModels))
using Optim
println("Optim ", pkgversion(Optim))
