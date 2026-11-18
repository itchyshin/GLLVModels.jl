# P1 public-door half of the first-seven behavioural checkpoint.
# Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/first_seven_behaviour_J.jl OUT.tsv
using GLLVModels, Distributions
using LinearAlgebra
using Sockets
using SHA
using Printf

length(ARGS) == 2 || error("usage: first_seven_behaviour_J.jl OUT.tsv GLLVM_COMMIT_AT_LAUNCH")
out = ARGS[1]
glvmodels_commit = ARGS[2]
actual_commit = readchomp(`git rev-parse HEAD`)
glvmodels_commit == actual_commit || error("launch commit does not match executed checkout HEAD")
runner_sha256 = bytes2hex(sha256(read(@__FILE__)))
package_source = realpath(pathof(GLLVModels))
startswith(package_source, realpath(joinpath(@__DIR__, "..", "src"))) || error("GLLVModels did not load from this checkout")
src_diff_sha256 = bytes2hex(sha256(read(pipeline(`git diff --binary HEAD -- src`))))
get(ENV, "OPENBLAS_NUM_THREADS", "") == "1" || error("OPENBLAS_NUM_THREADS=1 is required")
get(ENV, "OMP_NUM_THREADS", "") == "1" || error("OMP_NUM_THREADS=1 is required")
get(ENV, "JULIA_NUM_THREADS", "") == "4" || error("JULIA_NUM_THREADS=4 is required")
const IDS = ["CORE070-FIRST7-CHECK-AUTO-RESIDUAL", "CORE070-FIRST7-ISDM-COUNT",
    "CORE070-FIRST7-ISDM-EXTRA-SOURCE", "CORE070-FIRST7-ISDM-MISSING-IN-TRAIT",
    "CORE070-FIRST7-ISDM-MISSING-SOURCE", "CORE070-FIRST7-ISDM-WRAPPER-LAW"]
const CALLS = Dict(IDS[1] => "fit_gaussian_gllvm(diagnostic.Y; K=1, X=diagnostic.trait_X); check_auto_residual(fit)",
    IDS[2] => "fit_isdm_gllvm(formula, data; family=isdm_sources(count=Poisson(), detect=Poisson()))",
    IDS[3] => "fit_isdm_gllvm(formula, unknown; family=fam())",
    IDS[4] => "fit_isdm_gllvm(formula, missing_trait; family=fam())",
    IDS[5] => "fit_isdm_gllvm(formula, missing_source; family=fam())",
    IDS[6] => "isdm_sources(gbif=Poisson(), survey=isdm_source(Binomial(); observation=:(~ access)))")

function record(f, id)
    try
        v = f()
        if v isa NamedTuple && haskey(v, :coherent)
            return (id, "RETURN", "check_auto_residual", string(v.coherent), join(v.messages, " | "))
        elseif v isa NamedTuple && haskey(v, :count_admitted)
            return (id, "RETURN", "IsdmFit", v.count_admitted ? "wrong-count-route" : "all-count-nonmixed", "")
        elseif hasproperty(v, :converged)
            return (id, "RETURN", string(typeof(v)), string(v.converged), "public fit returned")
        end
        return (id, "RETURN", string(typeof(v)), "", "public call returned")
    catch e
        msg = sprint(showerror, e)
        firstline = first(split(msg, '\n'))
        return (id, "ERROR", string(nameof(typeof(e))), "", firstline)
    end
end

function diagnostic_fixture()
    n = 12
    values = sin.(collect(1:(2n)) ./ 3)
    Y = permutedims(reshape(values, n, 2))
    # Canonical X remains the shared site-support vector used by the paired
    # R fixture hash; trait_X is the explicit p × n × p design for 0 + trait.
    X = ones(n, 1)
    trait_X = zeros(Float64, 2, n, 2)
    for t in 1:2
        trait_X[t, :, t] .= 1.0
    end
    ordinal_values = 1 .+ (collect(1:n) .% 3)
    ordinal = repeat(reshape(ordinal_values, 1, :), 2, 1)
    return (; Y, X, trait_X, ordinal)
end

# Deterministic long panel; negatives mutate one public input condition.
function panel()
    rows = [(t, s, u) for t in ("a", "b") for s in ("count", "detect") for u in ("u1", "u2")]
    trait, source, unit = [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows]
    value = [s == "count" ? Float64(1 + i % 4) : Float64(i % 2) for (i, s) in enumerate(source)]
    log_support = collect(range(0.05, 0.4; length=length(rows)))
    for t in ("a", "b"), s in ("count", "detect")
        any((trait .== t) .& (source .== s)) || error("panel lacks trait/source cell $t/$s")
        for u in ("u1", "u2")
            count((trait .== t) .& (source .== s) .& (unit .== u)) == 1 ||
                error("panel cell is not unique: $t/$s/$u")
        end
    end
    return (; trait, isdm_source=source, unit, value, log_support)
end
panel_input_rows(d) = [join((d.trait[i], d.isdm_source[i], d.unit[i], @sprintf("%.8f", d.value[i]),
    @sprintf("%.8f", d.log_support[i])), '\t') for i in eachindex(d.value)]
diagnostic_input_rows(f) = vcat(
    ["gaussian_Y\t$(trait)\t" * join((@sprintf("%.8f", f.Y[t,j]) for j in axes(f.Y,2)), '\t') for (t,trait) in enumerate(("a", "b"))],
    ["gaussian_X\t" * join((@sprintf("%.8f", f.X[j,1]) for j in axes(f.X,1)), '\t')],
    ["ordinal_Y\t$(trait)\t" * join((string(f.ordinal[t,j]) for j in axes(f.ordinal,2)), '\t') for (t,trait) in enumerate(("a", "b"))])
function case_input_sha256(case_id, model, rows)
    payload = join(vcat(["CORE070-FIRST7-INPUT-v1", "case\t" * case_id,
        "model\t" * model], rows), "\n")
    return bytes2hex(sha256(codeunits(payload)))
end
function case_fixture_hashes(d, diagnostic, unknown, missing_trait, missing_source)
    ids = IDS
    specs = ["gaussian:0+trait+latent(0+trait|unit,d=1,unique=false);trait_intercepts",
        "isdm:count=poisson-log,detect=poisson-log",
        "isdm:count=poisson-log,detect=binomial-cloglog",
        "isdm:count=poisson-log,detect=binomial-cloglog",
        "isdm:count=poisson-log,detect=binomial-cloglog",
        "isdm_sources:gbif=poisson-log,survey=wrapped-binomial-logit-observation-access"]
    fixtures = [diagnostic_input_rows(diagnostic), panel_input_rows(d), panel_input_rows(unknown),
        panel_input_rows(missing_trait), panel_input_rows(missing_source),
        ["standalone=binomial-logit-observation-x", "wrapped=survey:binomial-logit-observation-access",
         "control=survey:binomial-cloglog"]]
    return Dict(ids[i] => case_input_sha256(ids[i], specs[i], fixtures[i]) for i in eachindex(ids))
end
formula = :(value ~ 0 + trait + offset(log_support) + latent(0 + trait | unit, d = 1, unique = false))
function fam()
    isdm_sources(count=Poisson(), detect=(Binomial(), CLogLogLink()))
end
fit(data=panel(), family=fam()) = fit_isdm_gllvm(formula, data; family=family, unit=:unit)

base_fit = fit()
base_fit.table.admitted || error("valid mixed-source public positive control was not admitted")

diagnostic = diagnostic_fixture()
data = panel()
src = copy(data.isdm_source); src[1] = "unknown"
unknown = merge(data, (isdm_source=src,))
keep_trait = .!((data.trait .== "a") .& (data.isdm_source .== "detect"))
missing_trait = (; (k => getproperty(data, k)[keep_trait] for k in propertynames(data))...)
keep_source = data.isdm_source .== "count"
missing_source = (; (k => getproperty(data, k)[keep_source] for k in propertynames(data))...)
input_hashes = case_fixture_hashes(data, diagnostic, unknown, missing_trait, missing_source)
rows = [record(IDS[1]) do
    f = fit_gaussian_gllvm(diagnostic.Y; K=1, X=diagnostic.trait_X)
    GLLVModels.check_auto_residual(f)
end]
control = record("CORE070-FIRST7-CHECK-AUTO-RESIDUAL-ORDINAL-PROBIT-CONTROL") do
    fo = fit_ordinal_gllvm(diagnostic.ordinal; K=1, link=GLLVModels.ProbitLink())
    GLLVModels.check_auto_residual(fo)
end
control[2] == "RETURN" && control[4] == "false" || error(
    "check_auto_residual ordinal-probit negative control was not discriminating: $control")
rows[1] = (rows[1][1], rows[1][2], rows[1][3], rows[1][4],
    "ordinal-probit control flagged; " * rows[1][5])
push!(rows, record(IDS[2]) do
    ft = fit(panel(), isdm_sources(count=Poisson(), detect=Poisson()))
    (; count_admitted=ft.table.admitted)
end)
push!(rows, record(IDS[3]) do
    count(==("unknown"), unknown.isdm_source) == 1 && length(unknown.value) == length(data.value) || error("unknown-source mutation changed more than its target")
    fit(unknown)
end)
push!(rows, record(IDS[4]) do
    length(data.value) - length(missing_trait.value) == 2 || error("missing-in-trait mutation removed unexpected rows")
    fit(missing_trait)
end)
push!(rows, record(IDS[5]) do
    length(data.value) - length(missing_source.value) == 4 || error("missing-source mutation removed unexpected rows")
    fit(missing_source)
end)
push!(rows, record(IDS[6]) do
    logit_alone = isdm_source(Binomial(); observation=:(~ access))
    logit_alone isa IsdmSource || error("standalone logit isdm_source did not construct")
    isdm_sources(gbif=Poisson(), survey=(Binomial(), CLogLogLink()))
    try
        isdm_sources(gbif=Poisson(),
            survey=isdm_source(Binomial(); observation=:(~ access)))
        error("expected isdm_sources refusal for wrapped logit law")
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("REFUSED: isdm_sources: " * sprint(showerror, e)))
    end
end)

open(out, "w") do io
    println(io, "case_id\toutcome\tclass\tactual\tmessage\tcall\tengine\tpin\tjulia_version\thost\tglvmodels_commit\tjulia_threads\topenblas_threads\tomp_threads\trunner_sha256\tfixture_sha256\tpackage_source\tsrc_diff_sha256\tpublic_positive_control")
    for r in rows
        fields = replace.(string.(r), '\t' => ' ', '\n' => ' ')
        println(io, join((fields..., CALLS[r[1]], "Julia", "P1", string(VERSION), gethostname(),
            glvmodels_commit, get(ENV, "JULIA_NUM_THREADS", ""), string(LinearAlgebra.BLAS.get_num_threads()),
            get(ENV, "OMP_NUM_THREADS", ""), runner_sha256, input_hashes[r[1]],
            relpath(package_source, realpath(joinpath(@__DIR__, ".."))), src_diff_sha256, "mixed-source-accepted"), '\t'))
    end
end
println("CORE070_FIRST7_JULIA_RAW_WRITTEN")
