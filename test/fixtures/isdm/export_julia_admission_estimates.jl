# Writes test/fixtures/isdm/julia_estimates_admission_p1.toml: Julia's estimate for
# each admission-twin case, read by export_admission_twins_p1.R (stage `xobj`).
# Run from the repository root:
#   julia --project=. test/fixtures/isdm/export_julia_admission_estimates.jl
using GLLVModels, Distributions, Printf, TOML
include(joinpath(@__DIR__, "isdm_fixture_io.jl"))
include(joinpath(@__DIR__, "isdm_admission_cases.jl"))

fmt(v) = "[" * join((@sprintf("%.17g", x) for x in v), ", ") * "]"
sfmt(v) = "[" * join(("\"$x\"" for x in v), ", ") * "]"
io = IOBuffer()
println(io, "# Julia estimates for the iSDM admission twins. Written by")
println(io, "# test/fixtures/isdm/export_julia_admission_estimates.jl. Do not edit.")
println(io, "julia_version = \"$(VERSION)\"")
println(io, "commit = \"$(strip(read(`git -C $(pkgdir(GLLVModels)) rev-parse HEAD`, String)))\"")
for name in ISDM_ADMISSION_CASES
    c = isdm_admission_case(name)
    ft = fit_isdm_gllvm(c.formula, read_isdm_csv(c.csv); family = c.family, trait = :trait, unit = :cell_id)
    println(io, "\n[julia.$name]")
    println(io, "loglik = ", @sprintf("%.17g", ft.loglik))
    println(io, "converged = ", ft.converged)
    println(io, "cells_converged = ", all(ft.cell_converged))
    println(io, "iterations = ", ft.iterations)
    println(io, "b_fix_names = ", sfmt(ft.b_names))
    println(io, "b_fix = ", fmt(ft.b_fix))
    println(io, "K = ", size(ft.Λ, 2))
    size(ft.Λ, 2) > 0 && println(io, "theta_rr_B = ", fmt(GLLVModels.pack_lambda(ft.Λ)))
    println(stderr, name, ": loglik ", ft.loglik, " converged ", ft.converged, " cols ", ft.b_names)
end
write(joinpath(@__DIR__, "julia_estimates_admission_p1.toml"), take!(io))
