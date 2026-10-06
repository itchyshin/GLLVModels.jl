# CI-ROUTE-011 Monte-Carlo replicates, Julia side (ruling N4, vault D-319).
#
# Reads the fixture the R side wrote (r-mc.json: Y_tl and `individual`), fits it natively with
# fit_twolevel_gaussian(Y_tl, individual; K_B = 1, K_W = 1) as tools/core070_surface_conversion_batch.jl
# does, and for each seed s runs repeatability_ci(fit_tl, Y_tl, individual; method = :bootstrap,
# nsim = 200, seed = s), recording endpoints, the point estimate, the structural check and wall time.
# The rule it is judged by (receipts/inference/ci-route-011-mc/rule.json) was committed before any run.
#
#   julia --project=<env> tools/core070_ci_route_011_mc.jl <repo-root> <r-mc.json> <destination-file> <seed> [<seed> ...]

using GLLVModels, JSON3, SHA

length(ARGS) >= 4 || error("usage: julia core070_ci_route_011_mc.jl <repo-root> <r-mc.json> <destination-file> <seed>...")
repo_root, r_path, dest = abspath(ARGS[1]), ARGS[2], ARGS[3]
seeds = parse.(Int, ARGS[4:end])
isfile(dest) && error("destination exists (refusing to overwrite): $dest")
realpath(Base.pkgdir(GLLVModels)) == realpath(repo_root) ||
    error("GLLVModels is loaded from $(Base.pkgdir(GLLVModels)), not from $repo_root")

r = JSON3.read(read(r_path, String))
fx = r.fixture
p_tl, n_ind, reps = Int(fx.p), Int(fx.n_individual), Int(fx.reps_per_individual)
Y_tl = reshape(Float64.(fx.y), p_tl, n_ind * reps)
individual = Int.(fx.individual)
fit_tl = fit_twolevel_gaussian(Y_tl, individual; K_B = 1, K_W = 1)
point = GLLVModels.extract_repeatability(fit_tl)

replicates = Any[]
for s in seeds
    t0 = time()
    rec = try
        ci = GLLVModels.repeatability_ci(fit_tl, Y_tl, individual; method = :bootstrap, nsim = 200, seed = s)
        lower = Float64[x.lower for x in ci]; upper = Float64[x.upper for x in ci]
        Dict{String, Any}("ok" => true, "lower" => lower, "upper" => upper,
                          "finite" => all(isfinite, lower) && all(isfinite, upper),
                          "ordered" => all(lower .<= upper), "brackets_point" => all(lower .<= point .<= upper))
    catch e
        Dict{String, Any}("ok" => false, "error" => sprint(showerror, e))
    end
    rec["seed"] = s
    rec["elapsed_seconds"] = round(time() - t0; digits = 2)
    push!(replicates, rec)
    println("seed $s: $(rec["elapsed_seconds"]) s ok=$(rec["ok"])")
end

open(dest, "w") do io
    JSON3.pretty(io, Dict{String, Any}(
        "schema" => "core070-ci-route-011-mc-julia/v1", "julia_version" => string(VERSION),
        "r_input_sha256" => bytes2hex(open(SHA.sha256, r_path)),
        "source_pins" => Dict(f => bytes2hex(open(SHA.sha256, joinpath(repo_root, f)))
                              for f in ("src/twolevel.jl", "tools/core070_ci_route_011_mc.jl")),
        "point" => point, "seeds" => seeds, "replicates" => replicates))
    println(io)
end
println("CI_ROUTE_011_MC_JULIA_DONE")
