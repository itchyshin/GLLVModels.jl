# Write Julia's optimum for every composed fit in composed.toml to
# julia_optima_composed.csv, so the slice 2 R generator's `cross` stage can
# evaluate R's objective there (cross-objective, Julia to R). Run from the
# repository root:
#   julia --project=. test/fixtures/temporal_p1/julia_optima_composed.jl
using GLLVModels, Printf
include(joinpath(@__DIR__, "fixture_helpers.jl"))
C = temporal_p1_load("composed.toml")
open(temporal_p1_path("julia_optima_composed.csv"), "w") do io
    println(io, "fit_id,index,name,value")
    for c in C["fits"]
        f = temporal_p1_composed_fit(C, c; g_tol=TEMPORAL_P1_GTOL, iterations=TEMPORAL_P1_ITERATIONS)
        for (i, (n, v)) in enumerate(zip(f.parameter_names, f.parameters))
            @printf(io, "%s,%d,%s,%.17g\n", c["id"], i, n, v)
        end
    end
end
