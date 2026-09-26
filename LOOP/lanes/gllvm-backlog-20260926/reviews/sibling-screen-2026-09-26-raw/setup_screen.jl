using Pkg
Pkg.activate(joinpath(homedir(), "hsq_work/gllvm-sibling-screen-20260926/env"))
Pkg.develop(path = joinpath(homedir(), "hsq_work/gllvm-sibling-screen-20260926/repo"))
Pkg.add(["Optim", "Distributions", "ForwardDiff", "SpecialFunctions"])
Pkg.precompile()
using GLLVModels
println("OK ", pathof(GLLVModels))
using Optim
println("Optim ", pkgversion(Optim))
