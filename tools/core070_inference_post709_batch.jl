# Post-#709 re-measurement of the inference routing and error-class rows (W2-1).
#
# PR #709 made the public Gaussian `confint(fit, y; parm, method, ...)` either run the
# method it is asked for or refuse it with an ArgumentError, and routed the derived
# parm names (communality[t], icc[t], rho[i,j], proportion:<component>[t],
# phylo_signal[t]) to the transformed-Wald, derived-profile and derived-bootstrap
# functions. The 61c5eda48 batches (tools/core070_inference_batch.jl and
# tools/core070_inference_remainder_batch.jl) called internal functions directly,
# because the public call ignored `method` on a structured fit and had no derived
# routes. This script makes the same requests through the public `confint` only, on
# the same two fixtures (same seeds), and records for each row what a Julia user gets:
#
#   route rows     which solver ran (the shape of the returned NamedTuple) and the
#                  `method` field it reports
#   refusal rows   the exception type and the full message for a method the call
#                  does not support, paired with a VALID-method control on the same
#                  fit and parm that must return an interval
#
# It also records the rows that stay unbound (CI-ROUTE-015, 023, 029, 030, 037), so the
# reason is a measurement and not a claim. Nothing here is compared with R: the R side
# of every row is read from the tracked P1 route probe and oracle by
# tools/core070_behaviour_receipts.py.
#
# Usage (the project must provide GLLVModels from <repo-root>, for example a scratch
# environment that `Pkg.develop`s it):
#   julia --project=<env> tools/core070_inference_post709_batch.jl <repo-root> <destination>
# <destination> must not exist.

using GLLVModels, Random, LinearAlgebra, SHA

length(ARGS) == 2 || error("usage: julia core070_inference_post709_batch.jl <repo-root> <destination>")
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
                "tools/core070_inference_post709_batch.jl"]
source_pins = Dict(f => sha256_file(joinpath(repo_root, f)) for f in pinned_files)

# ---- Fixture A: AGHQ Gaussian-record fit (same seed and construction as the 61c5eda48 batch) ----
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

# The solver that produced a result, from its shape (as in the 61c5eda48 batch).
function route_tag(result)
    f = propertynames(result)
    :n_converged in f && return :bootstrap
    :se_transformed in f && return :wald_derived
    (:se in f && :pd_hessian in f) && return :wald_packed
    :method in f && return :profile
    return :unknown
end

struct Row
    source_id::String
    case_id::String
    target::String
    requested::String          # the method string the row asks for ("DEFAULT" when none)
    kind::String               # "route" | "refusal"
    call::String               # the public call, as text
    thunk::Function
    control_call::String       # "" for a route row
    control::Function
    reported::Bool             # false: observed for the record only, the row is not bound by it
end

rt(sid, case, target, req, call, thunk; reported = true) =
    Row(sid, case, target, req, "route", call, thunk, "", () -> nothing, reported)
rf(sid, case, target, req, call, thunk, ctl_call, control) =
    Row(sid, case, target, req, "refusal", call, thunk, ctl_call, control, true)

PS = "CORE070-INFERENCE-"
rows = Row[]

# ---- derived and Sigma routes, through the public confint ----
sS(parm, m; kw...) = () -> confint(fitS, YS; parm = parm, method = m, Σ_phy = Σ_phy, kw...)
dS(parm) = () -> confint(fitS, YS; parm = parm, Σ_phy = Σ_phy)
push!(rows, rt("CI-ROUTE-015", PS * "PHYLO-SIGNAL-CI-METHOD-ROUTE", "phylo_signal", "DEFAULT",
               "confint(fitS, YS; parm=\"phylo_signal[1]\", Σ_phy)", dS("phylo_signal[1]")))
push!(rows, rt("CI-ROUTE-016", PS * "PHYLO-SIGNAL-CI-METHOD-ROUTE", "phylo_signal", "profile",
               "confint(fitS, YS; parm=\"phylo_signal[1]\", method=:profile, Σ_phy)", sS("phylo_signal[1]", :profile)))
push!(rows, rt("CI-ROUTE-018", PS * "PHYLO-SIGNAL-CI-METHOD-ROUTE", "phylo_signal", "bootstrap",
               "confint(fitS, YS; parm=\"phylo_signal[1]\", method=:bootstrap, Σ_phy, n_boot=6, seed=11)",
               sS("phylo_signal[1]", :bootstrap; n_boot = 6, seed = 11)))
push!(rows, rt("CI-ROUTE-022", PS * "COMMUNALITY-CI-METHOD-ROUTE", "communality", "DEFAULT",
               "confint(fitS, YS; parm=\"communality[1]\", Σ_phy)", dS("communality[1]")))
push!(rows, rt("CI-ROUTE-023", PS * "COMMUNALITY-CI-METHOD-ROUTE", "communality", "profile",
               "confint(fitS, YS; parm=\"communality[1]\", method=:profile, Σ_phy)", sS("communality[1]", :profile)))
push!(rows, rt("CI-ROUTE-025", PS * "COMMUNALITY-CI-METHOD-ROUTE", "communality", "bootstrap",
               "confint(fitS, YS; parm=\"communality[1]\", method=:bootstrap, Σ_phy, n_boot=6, seed=9)",
               sS("communality[1]", :bootstrap; n_boot = 6, seed = 9)))
push!(rows, rt("CI-ROUTE-029", PS * "RHO-CI-METHOD-ROUTE", "rho", "DEFAULT",
               "confint(fitS, YS; parm=\"rho[1,2]\", Σ_phy)", dS("rho[1,2]")))
push!(rows, rt("CI-ROUTE-030", PS * "RHO-CI-METHOD-ROUTE", "rho", "profile",
               "confint(fitS, YS; parm=\"rho[1,2]\", method=:profile, Σ_phy)", sS("rho[1,2]", :profile)))
push!(rows, rt("CI-ROUTE-032", PS * "RHO-CI-METHOD-ROUTE", "rho", "bootstrap",
               "confint(fitS, YS; parm=\"rho[1,2]\", method=:bootstrap, Σ_phy, n_boot=6, seed=10)",
               sS("rho[1,2]", :bootstrap; n_boot = 6, seed = 10)))
push!(rows, rt("CI-ROUTE-036", PS * "PROPORTION-CI-METHOD-ROUTE", "proportion", "DEFAULT",
               "confint(fitS, YS; parm=\"proportion:shared[1]\", Σ_phy)", dS("proportion:shared[1]")))
push!(rows, rt("CI-ROUTE-037", PS * "PROPORTION-CI-METHOD-ROUTE", "proportion", "profile",
               "confint(fitS, YS; parm=\"proportion:shared[1]\", method=:profile, Σ_phy)",
               sS("proportion:shared[1]", :profile)))
push!(rows, rt("CI-ROUTE-039", PS * "PROPORTION-CI-METHOD-ROUTE", "proportion", "bootstrap",
               "confint(fitS, YS; parm=\"proportion:shared[1]\", method=:bootstrap, Σ_phy, n_boot=6, seed=12)",
               sS("proportion:shared[1]", :bootstrap; n_boot = 6, seed = 12)))
for (sid, case, tg, parm, seed) in [("CI-ROUTE-045", "SIGMA-B", "sigma_B", "sigma_B[1]", 4),
                                    ("CI-ROUTE-057", "SIGMA-B", "sigma_B", "sigma_B[1]", 5),
                                    ("CI-ROUTE-048", "SIGMA-W", "sigma_W", "sigma_W[1]", 6),
                                    ("CI-ROUTE-060", "SIGMA-W", "sigma_W", "sigma_W[1]", 7),
                                    ("CI-ROUTE-063", "SIGMA-PHY", "sigma_phy", "sigma_phy[1]", 8)]
    push!(rows, rt(sid, PS * case * "-CI-METHOD-ROUTE", tg, "bootstrap",
                   "confint(fitS, YS; parm=\"$parm\", method=:bootstrap, Σ_phy, n_boot=6, seed=$seed)",
                   sS(parm, :bootstrap; n_boot = 6, seed = seed)))
end

# ---- refusals, each with a valid-method control on the same fit and parm ----
FZ = Symbol("fisher-z")
function refusal_rows!(rows, target, case, parm, specs)
    ctl = () -> confint(fitS, YS; parm = parm, method = :wald, Σ_phy = Σ_phy)
    for (sid, meth) in specs
        sym = meth == "fisher-z" ? FZ : Symbol(meth)
        push!(rows, rf(sid, PS * case * "-CI-UNSUPPORTED-METHOD-REJECT", target, meth,
                       "confint(fitS, YS; parm=\"$parm\", method=Symbol(\"$meth\"), Σ_phy)",
                       () -> confint(fitS, YS; parm = parm, method = sym, Σ_phy = Σ_phy),
                       "confint(fitS, YS; parm=\"$parm\", method=:wald, Σ_phy)", ctl))
    end
end
refusal_rows!(rows, "icc", "ICC", "icc[1]",
              [("CI-ROUTE-012", "wald_asym"), ("CI-ROUTE-013", "fisher-z"), ("CI-ROUTE-014", "bogus")])
refusal_rows!(rows, "phylo_signal", "PHYLO-SIGNAL", "phylo_signal[1]",
              [("CI-ROUTE-019", "wald_asym"), ("CI-ROUTE-020", "fisher-z"), ("CI-ROUTE-021", "bogus")])
refusal_rows!(rows, "communality", "COMMUNALITY", "communality[1]",
              [("CI-ROUTE-026", "wald_asym"), ("CI-ROUTE-027", "fisher-z"), ("CI-ROUTE-028", "bogus")])
refusal_rows!(rows, "rho", "RHO", "rho[1,2]", [("CI-ROUTE-033", "wald_asym"), ("CI-ROUTE-035", "bogus")])
refusal_rows!(rows, "proportion", "PROPORTION", "proportion:shared[1]",
              [("CI-ROUTE-040", "wald_asym"), ("CI-ROUTE-041", "fisher-z"), ("CI-ROUTE-042", "bogus")])
let ctl = () -> confint(fitA, YA; parm = "Lambda_B[1,1]", method = :wald)
    for (sid, meth) in [("CI-ROUTE-006", "fisher-z"), ("CI-ROUTE-007", "bogus")]
        sym = Symbol(meth)
        push!(rows, rf(sid, PS * "LAMBDA-CI-UNSUPPORTED-METHOD-REJECT", "lambda", meth,
                       "confint(fitA, YA; parm=\"Lambda_B[1,1]\", method=Symbol(\"$meth\"))",
                       () -> confint(fitA, YA; parm = "Lambda_B[1,1]", method = sym),
                       "confint(fitA, YA; parm=\"Lambda_B[1,1]\", method=:wald)", ctl))
    end
end

# ---- run ----
function describe_error(e)
    return Dict{String, Any}("error_type" => string(typeof(e)), "error_message" => sprint(showerror, e))
end
function run_row(r::Row)
    t0 = time()
    out = Dict{String, Any}("source_id" => "inference/" * r.source_id, "case_id" => r.case_id, "target" => r.target,
                            "requested_method" => r.requested, "kind" => r.kind, "julia_call" => r.call,
                            "reported" => r.reported)
    try
        res = r.thunk()
        out["outcome"] = "result"
        out["route_tag"] = string(route_tag(res))
        out["result_method"] = hasproperty(res, :method) ? string(res.method) : ""
        out["n_terms"] = length(res.term)
    catch e
        out["outcome"] = "error"
        merge!(out, describe_error(e))
    end
    if r.kind == "refusal"
        out["control_call"] = r.control_call
        try
            c = r.control()
            out["control_outcome"] = "result"
            out["control_route_tag"] = string(route_tag(c))
            out["control_result_method"] = hasproperty(c, :method) ? string(c.method) : ""
            out["control_finite"] = all(isfinite, c.lower) && all(isfinite, c.upper)
        catch e
            out["control_outcome"] = "error"
            out["control_error"] = sprint(showerror, e)
        end
    end
    out["elapsed_seconds"] = round(time() - t0; digits = 2)
    return out
end
t_all = time()
results = [run_row(r) for r in rows]
runtime_seconds = round(time() - t_all; digits = 1)

# ---- negative controls: the classifier and the refusal check must be able to fail ----
wald_probe = confint(fitS, YS; parm = "communality[1]", method = :wald, Σ_phy = Σ_phy)
boot_probe = confint(fitS, YS; parm = "communality[1]", method = :bootstrap, Σ_phy = Σ_phy, n_boot = 6, seed = 9)
nc = Dict{String, Any}(
    "NEG-WALD-CARRIES-N_CONVERGED" => Dict("behaved" => !(:n_converged in propertynames(wald_probe))),
    "NEG-BOOTSTRAP-CARRIES-SE" => Dict("behaved" => !(:se_transformed in propertynames(boot_probe))),
    "NEG-BOGUS-METHOD-ACCEPTED" => Dict("behaved" => (try
        confint(fitS, YS; parm = "communality[1]", method = :bogus, Σ_phy = Σ_phy)
        false
    catch e
        e isa ArgumentError && occursin("bogus", sprint(showerror, e))
    end)),
)
negatives_ok = all(v["behaved"] for v in values(nc))

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

all_ok = negatives_ok && all(r -> r["kind"] == "route" ? r["outcome"] == "result" :
                                  (r["outcome"] == "error" && r["control_outcome"] == "result"), results)
results_path = joinpath(destination, "inference-post709-results.json")
wjf(results_path, Dict(
    "status" => all_ok ? "PASS" : "FAIL",
    "area" => "inference-post709",
    "scope" => "CORE070_INFERENCE_POST709_BATCH",
    "case_count" => length(results),
    "julia_version" => string(VERSION),
    "gllvm_pkg_version" => string(Base.pkgversion(GLLVModels)),
    "source_pins" => source_pins,
    "runtime_seconds" => runtime_seconds,
    "negative_controls" => nc,
    "cases" => results))
receipt_path = joinpath(destination, "receipt.json")
wjf(receipt_path, Dict(
    "status" => all_ok ? "PASS" : "FAIL",
    "scope" => "CORE070_INFERENCE_POST709_BATCH",
    "source_pins" => source_pins,
    "case_count" => length(results),
    "expected_case_source_ids" => ["inference/" * r.source_id for r in rows],
    "results_sha256" => sha256_file(results_path),
    "julia_runtime" => string(VERSION)))
println("INFERENCE_POST709_BATCH ", length(results), " rows, runtime ", runtime_seconds, " s, negatives ",
        negatives_ok ? "OK" : "BROKEN")
println("CORE070_INFERENCE_POST709_BATCH_", all_ok ? "PASS" : "FAIL")
exit(all_ok ? 0 : 1)
