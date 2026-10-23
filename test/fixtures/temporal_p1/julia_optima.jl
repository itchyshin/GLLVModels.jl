# Write Julia's optimum for every fit in fits.toml to julia_optima.csv, so the
# R generator's `cross` stage can evaluate R's objective there (cross-objective,
# Julia to R). Run from the repository root:
#   julia --project=. test/fixtures/temporal_p1/julia_optima.jl
using GLLVModels, Printf
include(joinpath(@__DIR__, "fixture_helpers.jl"))
F = temporal_p1_load("fits.toml")
open(temporal_p1_path("julia_optima.csv"), "w") do io
    println(io, "fit_id,index,name,value")
    for c in F["fits"]
        f = temporal_p1_fit(F, c; g_tol=TEMPORAL_P1_GTOL, iterations=TEMPORAL_P1_ITERATIONS)
        for (i, (n, v)) in enumerate(zip(f.parameter_names, f.parameters))
            @printf(io, "%s,%d,%s,%.17g\n", c["id"], i, n, v)
        end
    end
end
