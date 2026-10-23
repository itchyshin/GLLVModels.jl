# Writes test/fixtures/isdm/julia_estimates_p1.toml: Julia's estimate for each
# paired iSDM case, read by export_p1_fixtures.R (stage `xobj`) to evaluate R's
# own objective at Julia's optimum. Run from the repository root:
#   julia --project=. test/fixtures/isdm/export_julia_estimates.jl
using GLLVModels, Distributions, Printf
include(joinpath(@__DIR__, "isdm_fixture_io.jl"))

fmt(v) = "[" * join((@sprintf("%.17g", x) for x in v), ", ") * "]"
sfmt(v) = "[" * join(("\"$x\"" for x in v), ", ") * "]"
io = IOBuffer()
println(io, "# Julia estimates for the iSDM P1 paired cases. Written by")
println(io, "# test/fixtures/isdm/export_julia_estimates.jl. Do not edit.")
println(io, "julia_version = \"$(VERSION)\"")
println(io, "commit = \"$(strip(read(`git -C $(pkgdir(GLLVModels)) rev-parse HEAD`, String)))\"")
for name in ISDM_CASES
    c = isdm_case(name)
    t0 = time()
    ft = fit_isdm_gllvm(c.formula, read_isdm_csv(c.csv); family = c.family,
                        trait = :trait, unit = :cell_id)
    println(io, "\n[julia.$name]")
    println(io, "loglik = ", @sprintf("%.17g", ft.loglik))
    println(io, "converged = ", ft.converged)
    println(io, "cells_converged = ", all(ft.cell_converged))
    println(io, "iterations = ", ft.iterations)
    println(io, "wall_seconds = ", @sprintf("%.3f", time() - t0))
    println(io, "b_fix_names = ", sfmt(ft.b_names))
    println(io, "b_fix = ", fmt(ft.b_fix))
    println(io, "K = ", size(ft.Λ, 2))
    if size(ft.Λ, 2) > 0
        println(io, "theta_rr_B = ", fmt(GLLVModels.pack_lambda(ft.Λ)))
    end
    println(stderr, name, ": loglik ", ft.loglik, " converged ", ft.converged)
end
write(joinpath(@__DIR__, "julia_estimates_p1.toml"), take!(io))
