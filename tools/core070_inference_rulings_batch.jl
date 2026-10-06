# Inference rows under the signed rulings of 2026-10-05 (vault D-319): what Julia's PUBLIC
# confint(fit, y; parm, method) does for the rows ruling B (defaults and fallbacks), ruling C
# (withdrawn profile), ruling 3 (CI-ROUTE-029, the rho default) and the CI-ROUTE-034 fisher_z
# alias name. Same two fixtures and seeds as tools/core070_inference_post709_batch.jl.
#
#   withdrawn rows (023, 030, 037)  method = :profile for communality, rho, proportion: the
#                                   exception type and full message, paired with a :wald
#                                   control on the same fit and parm that must return an interval
#   default rows (015, 043, 046,    the call with NO method: which solver ran and the `method` it
#     055, 058, 061, 065, 068)      reports (Julia's default), plus the same parm with
#                                   method = :profile (R's default route), to show whether a Julia
#                                   user can reach R's default by asking for it
#   fallback rows (067, 070)        method = :bootstrap for beta and sigma_eps: whether Julia runs a
#                                   bootstrap (R falls back to Wald with a message)
#   CI-ROUTE-029                    rho with no method (Julia's default)
#   CI-ROUTE-034                    rho with method = :fisher_z, and whether it is identical to :wald
#
# Nothing here is compared with R: the R side of every row is read from the tracked P1 route
# probe by tools/core070_behaviour_receipts.py and tools/core070_inference_p1_receipts.py.
#
# Usage (the project must provide GLLVModels from <repo-root>):
#   julia --project=<env> tools/core070_inference_rulings_batch.jl <repo-root> <destination>
# <destination> must not exist.

using GLLVModels, Random, LinearAlgebra, SHA

length(ARGS) == 2 || error("usage: julia core070_inference_rulings_batch.jl <repo-root> <destination>")
repo_root = abspath(ARGS[1])
destination = ARGS[2]
isdir(destination) && error("destination already exists (refusing to overwrite retained evidence): $destination")
realpath(Base.pkgdir(GLLVModels)) == realpath(repo_root) ||
    error("GLLVModels is loaded from $(Base.pkgdir(GLLVModels)), not from $repo_root")
mkpath(destination)

sha256_file(path) = bytes2hex(open(SHA.sha256, path))
pinned_files = ["src/confint.jl", "src/confint_profile.jl", "src/confint_bootstrap.jl",
                "src/confint_derived.jl", "src/confint_derived_wald.jl",
                "src/families/aghq_gaussian_fit.jl", "src/fit.jl",
                "tools/core070_inference_rulings_batch.jl"]
source_pins = Dict(f => sha256_file(joinpath(repo_root, f)) for f in pinned_files)

# ---- Fixture A: AGHQ Gaussian-record fit (same seed and construction as the post-#709 batch) ----
Random.seed!(20260901)
p, n, K, q = 3, 30, 1, 1
βA = [0.5, -0.3, 0.2]
ΛA = reshape([0.6, 0.4, -0.5], p, K)
XA = zeros(p, n, q)
for s in 1:n, t in 1:p
    XA[t, s, 1] = randn()
end
YA = zeros(p, n)
for s in 1:n
    z = randn(K)
    for t in 1:p
        YA[t, s] = βA[t] + 0.3 * XA[t, s, 1] + (ΛA * z)[t] + 0.3 * randn()
    end
end
fitA = fit_gaussian_gllvm(YA; K = K, X = XA, aghq = 3)
fitA.converged || error("fixture A did not converge")
GLLVModels._has_gaussian_record(fitA) || error("fixture A must be an AGHQ Gaussian record")

# ---- Fixture S: plain Gaussian fit with B/W tiers and a phylogenetic block ----
Random.seed!(20260902)
Σ_phy = [1.0 0.2 0.1; 0.2 1.0 0.15; 0.1 0.15 1.0]
βS = [0.4, -0.2, 0.1]
ΛS = reshape([0.5, 0.3, -0.4], p, K)
YS = zeros(p, n)
for s in 1:n
    z = randn(K)
    for t in 1:p
        YS[t, s] = βS[t] + (ΛS * z)[t] + 0.2 * randn() + 0.15 * randn()
    end
end
fitS = fit_gaussian_gllvm(YS; K = K, has_diag = true, has_phy_unique = true, Σ_phy = Σ_phy)
fitS.converged || error("fixture S did not converge")
GLLVModels._has_gaussian_record(fitS) && error("fixture S was expected to be a plain fit")

# The solver that produced a result, from its shape (as in the post-#709 batch).
function route_tag(result)
    f = propertynames(result)
    :n_converged in f && return :bootstrap
    :se_transformed in f && return :wald_derived
    (:se in f && :pd_hessian in f) && return :wald_packed
    :method in f && return :profile
    return :unknown
end

describe_error(e) = Dict{String, Any}("error_type" => string(typeof(e)), "error_message" => sprint(showerror, e))

function observe(thunk)
    out = Dict{String, Any}()
    try
        res = thunk()
        out["outcome"] = "result"
        out["route_tag"] = string(route_tag(res))
        out["result_method"] = hasproperty(res, :method) ? string(res.method) : ""
        out["finite"] = all(isfinite, res.lower) && all(isfinite, res.upper)
        hasproperty(res, :transform) && (out["transform"] = string.(res.transform))
        return out, res
    catch e
        out["outcome"] = "error"
        merge!(out, describe_error(e))
        return out, nothing
    end
end

PS = "CORE070-INFERENCE-"
results = Dict{String, Any}[]
t_all = time()
function record!(sid, case, target, group, requested, call, thunk; extra = Dict{String, Any}())
    t0 = time()
    obs, res = observe(thunk)
    out = Dict{String, Any}("source_id" => "inference/" * sid, "case_id" => PS * case, "target" => target,
                            "group" => group, "requested_method" => requested, "julia_call" => call)
    merge!(out, obs)
    for (k, f) in extra
        o, _ = observe(f[2])
        out[k] = merge(Dict{String, Any}("call" => f[1]), o)
    end
    out["elapsed_seconds"] = round(time() - t0; digits = 2)
    push!(results, out)
    return res
end

# ---- ruling C: withdrawn profile (refusal with a :wald control) ----
for (sid, case, tg, parm) in [("CI-ROUTE-023", "COMMUNALITY-CI-METHOD-ROUTE", "communality", "communality[1]"),
                              ("CI-ROUTE-030", "RHO-CI-METHOD-ROUTE", "rho", "rho[1,2]"),
                              ("CI-ROUTE-037", "PROPORTION-CI-METHOD-ROUTE", "proportion", "proportion:shared[1]")]
    record!(sid, case, tg, "withdrawn_profile", "profile",
            "confint(fitS, YS; parm=\"$parm\", method=:profile, Σ_phy)",
            () -> confint(fitS, YS; parm = parm, method = :profile, Σ_phy = Σ_phy);
            extra = Dict("control" => ("confint(fitS, YS; parm=\"$parm\", method=:wald, Σ_phy)",
                                       () -> confint(fitS, YS; parm = parm, method = :wald, Σ_phy = Σ_phy))))
end

# ---- ruling B: defaults (no method), with R's default route (profile) asked for explicitly ----
for (sid, case, tg, parm, fit, Y, fixture) in [
        ("CI-ROUTE-015", "PHYLO-SIGNAL-CI-METHOD-ROUTE", "phylo_signal", "phylo_signal[1]", fitS, YS, "S"),
        ("CI-ROUTE-043", "SIGMA-B-CI-METHOD-ROUTE", "sigma_B", "sigma_B[1]", fitS, YS, "S"),
        ("CI-ROUTE-046", "SIGMA-W-CI-METHOD-ROUTE", "sigma_W", "sigma_W[1]", fitS, YS, "S"),
        ("CI-ROUTE-055", "SIGMA-B-CI-METHOD-ROUTE", "sigma_B", "sigma_B[1]", fitS, YS, "S"),
        ("CI-ROUTE-058", "SIGMA-W-CI-METHOD-ROUTE", "sigma_W", "sigma_W[1]", fitS, YS, "S"),
        ("CI-ROUTE-061", "SIGMA-PHY-CI-METHOD-ROUTE", "sigma_phy", "sigma_phy[1]", fitS, YS, "S"),
        ("CI-ROUTE-065", "BETA-CI-METHOD-ROUTE", "beta", "beta[1]", fitA, YA, "A"),
        ("CI-ROUTE-068", "SIGMA-EPS-CI-METHOD-ROUTE", "sigma_eps", "sigma_eps", fitA, YA, "A")]
    if fixture == "S"
        call = "confint(fitS, YS; parm=\"$parm\", Σ_phy)"
        thunk = () -> confint(fit, Y; parm = parm, Σ_phy = Σ_phy)
        pcall = "confint(fitS, YS; parm=\"$parm\", method=:profile, Σ_phy)"
        pthunk = () -> confint(fit, Y; parm = parm, method = :profile, Σ_phy = Σ_phy)
    else
        call = "confint(fitA, YA; parm=\"$parm\", X=XA)"
        thunk = () -> confint(fit, Y; parm = parm, X = XA)
        pcall = "confint(fitA, YA; parm=\"$parm\", method=:profile, X=XA)"
        pthunk = () -> confint(fit, Y; parm = parm, method = :profile, X = XA)
    end
    record!(sid, case, tg, "default", "DEFAULT", call, thunk;
            extra = Dict("explicit_profile" => (pcall, pthunk)))
end

# ---- ruling B: fallbacks (R falls back from bootstrap to Wald for fixed effects and sigma_eps) ----
for (sid, case, tg, parm, seed) in [("CI-ROUTE-067", "BETA-BOOTSTRAP-FALLBACK-DIVERGENCE", "beta", "beta[1]", 21),
                                    ("CI-ROUTE-070", "SIGMA-EPS-BOOTSTRAP-FALLBACK-DIVERGENCE", "sigma_eps", "sigma_eps", 22)]
    record!(sid, case, tg, "fallback", "bootstrap",
            "confint(fitA, YA; parm=\"$parm\", method=:bootstrap, X=XA, n_boot=6, seed=$seed)",
            () -> confint(fitA, YA; parm = parm, method = :bootstrap, X = XA, n_boot = 6, seed = seed))
end

# ---- ruling 3 and CI-ROUTE-034: rho default and the fisher_z alias ----
w = record!("CI-ROUTE-029", "RHO-CI-METHOD-ROUTE", "rho", "default", "DEFAULT",
            "confint(fitS, YS; parm=\"rho[1,2]\", Σ_phy)", () -> confint(fitS, YS; parm = "rho[1,2]", Σ_phy = Σ_phy))
z = record!("CI-ROUTE-034", "RHO-CI-METHOD-ROUTE", "rho", "fisher_z_alias", "fisher-z",
            "confint(fitS, YS; parm=\"rho[1,2]\", method=:fisher_z, Σ_phy)",
            () -> confint(fitS, YS; parm = "rho[1,2]", method = :fisher_z, Σ_phy = Σ_phy))
wald = confint(fitS, YS; parm = "rho[1,2]", method = :wald, Σ_phy = Σ_phy)
results[end]["identical_to_wald"] = z !== nothing && isequal(z, wald)
results[end]["identical_to_wald_call"] = "isequal(confint(fitS, YS; parm=\"rho[1,2]\", method=:fisher_z, Σ_phy), confint(fitS, YS; parm=\"rho[1,2]\", method=:wald, Σ_phy))"
results[end]["interval"] = z === nothing ? nothing : Dict("estimate" => z.estimate[1], "lower" => z.lower[1], "upper" => z.upper[1])
results[end-1]["identical_to_wald"] = w !== nothing && isequal(w, wald)
runtime_seconds = round(time() - t_all; digits = 1)

# ---- negative controls: the classifier and the refusal check must be able to fail ----
nc = Dict{String, Any}(
    "NEG-PROFILE-WITHDRAWN-FOR-PHYLO-SIGNAL" => Dict("behaved" => (try
        confint(fitS, YS; parm = "phylo_signal[1]", method = :profile, Σ_phy = Σ_phy)
        true
    catch
        false
    end)),
    "NEG-FISHER-Z-ACCEPTED-FOR-COMMUNALITY" => Dict("behaved" => (try
        confint(fitS, YS; parm = "communality[1]", method = :fisher_z, Σ_phy = Σ_phy)
        false
    catch e
        e isa ArgumentError && occursin("fisher_z", sprint(showerror, e))
    end)),
)
negatives_ok = all(v["behaved"] for v in values(nc))

function row_ok(r)
    g = r["group"]
    g == "withdrawn_profile" && return r["outcome"] == "error" && r["error_type"] == "ArgumentError" &&
        occursin("withdrawn", r["error_message"]) && r["control"]["outcome"] == "result" && r["control"]["finite"]
    g == "default" && return r["outcome"] == "result"
    g == "fallback" && return r["outcome"] == "result"
    g == "fisher_z_alias" && return r["outcome"] == "result" && r["identical_to_wald"] === true
    return false
end
all_ok = negatives_ok && all(row_ok, results)

# ---- JSON ----
_esc(s::AbstractString) = replace(replace(replace(String(s), "\\" => "\\\\"), "\"" => "\\\""), "\n" => "\\n")
function wj(io::IO, x; indent::Int = 0)
    pad, pad1 = "  "^indent, "  "^(indent + 1)
    if x isa AbstractDict
        println(io, "{")
        ks = sort!(collect(keys(x)))
        for (i, k) in enumerate(ks)
            print(io, pad1, "\"", _esc(string(k)), "\": ")
            wj(io, x[k]; indent = indent + 1)
            println(io, i == length(ks) ? "" : ",")
        end
        print(io, pad, "}")
    elseif x isa AbstractVector
        if isempty(x)
            print(io, "[]")
        else
            println(io, "[")
            for (i, v) in enumerate(x)
                print(io, pad1)
                wj(io, v; indent = indent + 1)
                println(io, i == length(x) ? "" : ",")
            end
            print(io, pad, "]")
        end
    elseif x isa AbstractString
        print(io, "\"", _esc(x), "\"")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        print(io, isfinite(x) ? x : "null")
    elseif x === nothing
        print(io, "null")
    else
        print(io, "\"", _esc(string(x)), "\"")
    end
end
function wjf(path, x)
    open(path, "w") do io
        wj(io, x)
        println(io)
    end
end

results_path = joinpath(destination, "inference-rulings-results.json")
wjf(results_path, Dict(
    "status" => all_ok ? "PASS" : "FAIL",
    "area" => "inference-rulings",
    "scope" => "CORE070_INFERENCE_RULINGS_BATCH",
    "case_count" => length(results),
    "julia_version" => string(VERSION),
    "gllvm_pkg_version" => string(Base.pkgversion(GLLVModels)),
    "source_pins" => source_pins,
    "runtime_seconds" => runtime_seconds,
    "negative_controls" => nc,
    "cases" => results))
wjf(joinpath(destination, "receipt.json"), Dict(
    "status" => all_ok ? "PASS" : "FAIL",
    "scope" => "CORE070_INFERENCE_RULINGS_BATCH",
    "source_pins" => source_pins,
    "case_count" => length(results),
    "expected_case_source_ids" => [r["source_id"] for r in results],
    "results_sha256" => sha256_file(results_path),
    "julia_runtime" => string(VERSION)))
println("INFERENCE_RULINGS_BATCH ", length(results), " rows, runtime ", runtime_seconds, " s, negatives ",
        negatives_ok ? "OK" : "BROKEN")
println("CORE070_INFERENCE_RULINGS_BATCH_", all_ok ? "PASS" : "FAIL")
exit(all_ok ? 0 : 1)
