# Retained evidence runner, Julia side, for the "postfit-1" manifest-area
# batch's one EXECUTABLE_NOW case (CORE070-POSTFIT-COEF-MULTI-READBACK; see
# docs/dev-log/core070/postfit-1-batch-contract.json for the other 50 cases'
# NEEDS_NEW_JULIA_SURFACE bucketing). Fits GLLVModels.jl's closed-form Gaussian
# marginal on the contract's frozen fixture with an explicit per-species
# dummy design (matching R's `value ~ 0 + trait + latent(0 + trait | site,
# d = K)`), calls the public `coef(fit)` accessor (StatsAPI.coef, wired at
# src/postfit.jl:661), and writes a JSON receipt. No R, no RCall, no frozen
# library -- this half is fully runnable locally, unlike the paired R runner
# (tools/core070_postfit_1_batch.R), which needs the frozen oracle library
# and is Totoro-only.
#
# Usage: julia --project=<repo> tools/core070_postfit_1_batch.jl <destination>
#
# <destination> must not exist; it is created and populated with
# postfit-1-julia-results.json.

using GLLVModels
using LinearAlgebra

const ROOT = normpath(joinpath(@__DIR__, ".."))
# GLLVM_PARITY_PIN=P1 reads the P1-regenerated contract (cases verbatim); unset/P0 is unchanged.
_parity_pin = uppercase(strip(get(ENV, "GLLVM_PARITY_PIN", "P1")))
_parity_pin in ("P0", "P1") || error("GLLVM_PARITY_PIN must be P0 or P1, got $(repr(_parity_pin))")
const CONTRACT_PATH = _parity_pin == "P1" ?
    joinpath(ROOT, "docs", "dev-log", "core070", "true-parity-latest", "postfit-1-batch-contract-p1.json") :
    joinpath(ROOT, "docs", "dev-log", "core070", "postfit-1-batch-contract.json")
const REFERENCE_COMMIT = _parity_pin == "P1" ? "9539352f66f2db2cc26b1c393e67212a359b60c9" : "b4d5fee64def88bc768dda1f1f77c29b295edd86"

length(ARGS) == 1 || error("usage: julia tools/core070_postfit_1_batch.jl <destination>")
destination = ARGS[1]
isdir(destination) && error("destination already exists: $destination")
isfile(destination) && error("destination already exists: $destination")
mkpath(destination)

# ---------------------------------------------------------------------------
# Minimal JSON reader/writer (no external dependency; mirrors the existing
# repo convention in tools/core070_data_batch.jl / tools/core070_masks_known.jl).
# ---------------------------------------------------------------------------
function json_read(path::AbstractString)
    txt = read(path, String)
    pos = Ref(1)
    skip_ws!() = begin
        while pos[] <= lastindex(txt) && isspace(txt[pos[]])
            pos[] = nextind(txt, pos[])
        end
    end
    local parse_value
    function parse_string()
        pos[] += 1 # opening quote
        buf = IOBuffer()
        while true
            c = txt[pos[]]
            if c == '"'
                pos[] = nextind(txt, pos[])
                break
            elseif c == '\\'
                pos[] = nextind(txt, pos[])
                e = txt[pos[]]
                if e == 'n'; write(buf, '\n')
                elseif e == 't'; write(buf, '\t')
                elseif e == 'u'
                    hex = txt[nextind(txt, pos[]):nextind(txt, pos[], 4)]
                    write(buf, Char(parse(UInt32, hex; base = 16)))
                    pos[] = nextind(txt, pos[], 4)
                else write(buf, e)
                end
                pos[] = nextind(txt, pos[])
            else
                write(buf, c)
                pos[] = nextind(txt, pos[])
            end
        end
        String(take!(buf))
    end
    function parse_number()
        start = pos[]
        while pos[] <= lastindex(txt) && (isdigit(txt[pos[]]) || txt[pos[]] in ('-', '+', '.', 'e', 'E'))
            pos[] = nextind(txt, pos[])
        end
        s = txt[start:prevind(txt, pos[])]
        return occursin('.', s) || occursin('e', s) || occursin('E', s) ? parse(Float64, s) : parse(Int, s)
    end
    function parse_array()
        pos[] += 1
        out = Any[]
        skip_ws!()
        if txt[pos[]] == ']'
            pos[] = nextind(txt, pos[])
            return out
        end
        while true
            skip_ws!()
            push!(out, parse_value())
            skip_ws!()
            if txt[pos[]] == ','
                pos[] = nextind(txt, pos[])
            elseif txt[pos[]] == ']'
                pos[] = nextind(txt, pos[])
                break
            end
        end
        out
    end
    function parse_object()
        pos[] += 1
        out = Dict{String, Any}()
        skip_ws!()
        if txt[pos[]] == '}'
            pos[] = nextind(txt, pos[])
            return out
        end
        while true
            skip_ws!()
            k = parse_string()
            skip_ws!()
            @assert txt[pos[]] == ':'
            pos[] = nextind(txt, pos[])
            skip_ws!()
            out[k] = parse_value()
            skip_ws!()
            if txt[pos[]] == ','
                pos[] = nextind(txt, pos[])
            elseif txt[pos[]] == '}'
                pos[] = nextind(txt, pos[])
                break
            end
        end
        out
    end
    parse_value = function ()
        skip_ws!()
        c = txt[pos[]]
        if c == '"'; return parse_string()
        elseif c == '['; return parse_array()
        elseif c == '{'; return parse_object()
        elseif c == 't'; pos[] += 4; return true
        elseif c == 'f'; pos[] += 5; return false
        elseif c == 'n'; pos[] += 4; return nothing
        else; return parse_number()
        end
    end
    skip_ws!()
    parse_value()
end

json_escape(s::AbstractString) = replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n")
to_json(x::Bool) = x ? "true" : "false"
to_json(x::Nothing) = "null"
to_json(x::AbstractString) = "\"" * json_escape(x) * "\""
to_json(x::Integer) = string(x)
to_json(x::AbstractFloat) = repr(x)
to_json(x::AbstractVector) = "[" * join(to_json.(x), ",") * "]"
to_json(x::AbstractDict) = "{" * join(("\"$(json_escape(string(k)))\":" * to_json(v) for (k, v) in x), ",") * "}"

# ---------------------------------------------------------------------------
# Load the frozen fixture from the contract (single source of truth: no
# fixture values are duplicated by hand in this file).
# ---------------------------------------------------------------------------
contract = json_read(CONTRACT_PATH)
contract["reference_commit"] == REFERENCE_COMMIT || error("contract reference_commit drift")
eb = contract["executable_batch"]
eb["case_id"] == "CORE070-POSTFIT-COEF-MULTI-READBACK" || error("unexpected executable_batch case_id")
fixture = eb["fixture"]
p = Int(fixture["p"])
n = Int(fixture["n"])
K = Int(fixture["K"])
Y_rows = fixture["Y_rows_are_species_cols_are_sites"]
length(Y_rows) == p || error("fixture row count does not match p")
Y = Matrix{Float64}(undef, p, n)
for i in 1:p
    row = Y_rows[i]
    length(row) == n || error("fixture row $i does not have n columns")
    for j in 1:n
        Y[i, j] = Float64(row[j])
    end
end

# Per-species dummy design, matching R's `~0+trait` fixed part.
X = zeros(p, n, p)
for s in 1:p
    X[s, :, s] .= 1.0
end

fit = fit_gaussian_gllvm(Y; K = K, X = X)
julia_coef = coef(fit)

length(julia_coef) == p || error("coef(fit) length ($(length(julia_coef))) != p ($p)")

result = Dict{String, Any}(
    "schema" => "core070-postfit-1-julia-results/v1",
    "scope" => "CORE070_POSTFIT_1_BATCH",
    "case_id" => "CORE070-POSTFIT-COEF-MULTI-READBACK",
    "reference_commit" => REFERENCE_COMMIT,
    "julia_version" => string(VERSION),
    "p" => p, "n" => n, "K" => K,
    "converged" => fit.converged,
    "n_iter" => fit.n_iter,
    "logLik" => fit.logLik,
    "coef" => julia_coef,
    "all_checks" => fit.converged && length(julia_coef) == p && all(isfinite, julia_coef),
)

results_path = joinpath(destination, "postfit-1-julia-results.json")
open(results_path, "w") do io
    print(io, to_json(result))
end

println("CORE070_POSTFIT_1_BATCH_JULIA_RESULT coef=", julia_coef, " converged=", fit.converged)
println("CORE070_POSTFIT_1_BATCH_JULIA_", result["all_checks"] ? "PASS" : "FAIL")
exit(result["all_checks"] ? 0 : 1)
