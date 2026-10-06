# Captures origin/main's outputs for the weights byte-identity guard. NOT run by the test suite.
# Run against a detached worktree of origin/main (which has no `weights` keyword), single-threaded:
#   git worktree add --detach /path/main-wt <main sha>
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=/path/main-wt \
#       test/fixtures/weights_byte_identity/gen_main.jl <main sha>
# Writes main_capture.toml next to this file: per case, each numeric field's Float64 bit patterns.
using TOML
include(joinpath(@__DIR__, "cases.jl"))
sha = length(ARGS) >= 1 ? ARGS[1] : "unknown"
pathof(GLLVModels) |> p -> occursin("GLLVM.jl-weights", p) && error("run against the origin/main worktree, not the branch: $p")
GLLVModels.LinearAlgebra.BLAS.set_num_threads(1)
out = Dict{String, Any}("main_commit" => sha, "fingerprint" => _wbi_fingerprint(),
                        "glvmodels_path" => pathof(GLLVModels))
for (name, thunk) in _wbi_cases()
    out[name] = _wbi_record(thunk((;)))
    println(name, " captured")
end
open(joinpath(@__DIR__, "main_capture.toml"), "w") do io
    println(io, "# Weights byte-identity guard: origin/main outputs, captured by gen_main.jl. Do not hand-edit.")
    TOML.print(io, out; sorted = true)
end
