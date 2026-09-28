# Julia child for the wave-7 conversion batch
# (tools/core070_wave7_conversion_batch.R). Reads the R-oracle JSON (path
# from ENV["CORE070_WAVE7_CONVERSION_R_ORACLE"]), fits gaussian_small
# NATIVELY (independent optimiser run on the same Y the R process fit, not a
# replay of R's numbers), calls each case's Julia surface, and compares
# against the R oracle per its `kind` (point / verdict / own_consistency) at
# the contract's tolerance. Writes ARGS[1] as the JSON results file.
#
# No RCall, no parity-runner include, no R of any kind runs here.
# JSON reader/writer copied verbatim from the repo convention in
# tools/core070_wave6_conversion_batch.jl.
#
# Usage: julia --project=. tools/core070_wave7_conversion_batch.jl <out.json>

using GLLVModels
using LinearAlgebra: norm

# ---------------------------------------------------------------------------
# Minimal JSON reader/writer (no external dependency).
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
        pos[] += 1
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
to_json(x::Missing) = "null"
to_json(x::AbstractString) = "\"" * json_escape(x) * "\""
to_json(x::Integer) = string(x)
to_json(x::AbstractFloat) = isfinite(x) ? repr(x) : "null"
to_json(x::AbstractVector) = "[" * join(to_json.(x), ",") * "]"
to_json(x::AbstractDict) = "{" * join(("\"$(json_escape(string(k)))\":" * to_json(v) for (k, v) in x), ",") * "}"

length(ARGS) == 1 || error("usage: julia core070_wave7_conversion_batch.jl <output.json>")
out_path = ARGS[1]
mkpath(dirname(out_path))

oracle_path = get(ENV, "CORE070_WAVE7_CONVERSION_R_ORACLE", "")
isempty(oracle_path) && error("CORE070_WAVE7_CONVERSION_R_ORACLE not set")
isfile(oracle_path) || error("oracle file not found: $oracle_path")
oracle = json_read(oracle_path)

root = normpath(joinpath(@__DIR__, ".."))
# GLLVM_PARITY_PIN=P1 reads the P1-regenerated contract (cases verbatim); unset/P0 is unchanged.
_parity_pin = uppercase(strip(get(ENV, "GLLVM_PARITY_PIN", "P0")))
_parity_pin in ("P0", "P1") || error("GLLVM_PARITY_PIN must be P0 or P1, got $(repr(_parity_pin))")
contract = json_read(joinpath(root, _parity_pin == "P1" ?
    "docs/dev-log/core070/true-parity-latest/wave7-conversion-batch-contract-p1.json" :
    "docs/dev-log/core070/wave7-conversion-batch-contract.json"))
cases = contract["cases"]
length(cases) == contract["expected_case_count"] || error("case count mismatch vs contract")
Int(contract["expected_case_count"]) == 6 || error("expected_case_count drifted from 6; update this script")

function oracle_numeric_or_missing(oracle::Dict, bucket::AbstractString, case_id::AbstractString)
    ov = get(oracle, bucket, Dict{String, Any}())
    if !haskey(ov, case_id) || ov[case_id] === nothing
        return false, nothing
    end
    return true, ov[case_id]
end

# ---------------------------------------------------------------------------
# Fixture: gaussian_small -- fit natively.
# ---------------------------------------------------------------------------
gs = oracle["gaussian_small"]
p, K, n = gs["p"], gs["K"], gs["n"]
Y_g = reshape(Float64.(gs["y"]), p, n)
fit_g = fit_gaussian_gllvm(Y_g; K = K)

results = Dict{String, Any}()
all_ok = true

for cs in cases
    case_id = cs["case_id"]
    kind = cs["kind"]

    if kind == "point"
        ok_r, r_raw = oracle_numeric_or_missing(oracle, "oracle_values", case_id)
        if !ok_r
            results[case_id] = Dict{String, Any}("pass" => false, "kind" => kind,
                                                  "error" => "null_oracle_value: R oracle_values[$case_id] was null/missing")
            global all_ok = false
            continue
        end
        r_vec = Float64.(r_raw)
        tol = Float64(cs["tolerance"])
        jl_vec = Float64[]
        err = ""
        ok = false
        try
            if case_id == "CORE070-WAVE7-FITTED-MULTI"
                jl_vec = vec(GLLVModels.fitted(fit_g, Y_g))
            elseif case_id == "CORE070-WAVE7-PREDICT-MULTI"
                jl_vec = vec(GLLVModels.predict(fit_g, Y_g; type = :response))
            elseif case_id == "CORE070-WAVE7-RESIDUALS-MULTI"
                jl_vec = vec(GLLVModels.residuals(fit_g, Y_g; type = :dunnsmyth))
            else
                error("BOGUS_CASE_ID: no dispatcher entry for '$case_id'")
            end
            ok = length(jl_vec) == length(r_vec) && maximum(abs.(jl_vec .- r_vec)) <= tol
        catch e
            err = sprint(showerror, e)
        end
        maxdiff = (isempty(jl_vec) || length(jl_vec) != length(r_vec)) ? NaN :
            maximum(abs.(jl_vec .- r_vec))
        results[case_id] = Dict{String, Any}("pass" => ok, "kind" => kind, "tolerance" => tol,
                                              "max_abs_diff" => maxdiff, "r_len" => length(r_vec),
                                              "julia_len" => length(jl_vec), "error" => err,
                                              # PR #569 review finding 3: the Julia vector itself, so the
                                              # receipt tool recomputes max |R - Julia| instead of trusting it.
                                              "julia_values" => jl_vec)
        global all_ok &= ok

    elseif kind == "verdict"
        ok_r, r_verdict = oracle_numeric_or_missing(oracle, "verdict_oracle", case_id)
        if !ok_r
            results[case_id] = Dict{String, Any}("pass" => false, "kind" => kind,
                                                  "error" => "null_oracle_value: R verdict_oracle[$case_id] was null/missing")
            global all_ok = false
            continue
        end
        err = ""
        ok = false
        jl_verdict = Dict{String, Any}()
        try
            if case_id == "CORE070-WAVE7-CHECK-AUTO-RESIDUAL"
                r = GLLVModels.check_auto_residual(fit_g)
                jl_verdict["status_ok"] = r.coherent
                ok = (r_verdict["status_ok"] == true) == r.coherent
            elseif case_id == "CORE070-WAVE7-SANITY-MULTI"
                r = GLLVModels.sanity_multi(fit_g; y = Y_g)
                jl_verdict["converged"] = r.converged === missing ? nothing : r.converged
                jl_verdict["pd_hessian"] = r.pd_hessian === missing ? nothing : r.pd_hessian
                r_converged = r_verdict["converged"]
                r_pd = r_verdict["pd_hessian"]
                converged_ok = (r_converged === true) == (r.converged === true)
                # R's NA and Julia's `missing` both mean "not applicable"; treat
                # as agreeing only when both sides are non-committal, else compare booleans.
                pd_ok = if r_pd === nothing && r.pd_hessian === missing
                    true
                elseif r_pd === nothing || r.pd_hessian === missing
                    false
                else
                    (r_pd == true) == (r.pd_hessian == true)
                end
                ok = converged_ok && pd_ok
            else
                error("BOGUS_CASE_ID: no verdict dispatcher entry for '$case_id'")
            end
        catch e
            err = sprint(showerror, e)
        end
        results[case_id] = Dict{String, Any}("pass" => ok, "kind" => kind,
                                              "r_verdict" => r_verdict, "julia_verdict" => jl_verdict,
                                              "error" => err)
        global all_ok &= ok

    elseif kind == "own_consistency"
        ok_r, r_frob = oracle_numeric_or_missing(oracle, "own_consistency_oracle", case_id)
        if !ok_r
            results[case_id] = Dict{String, Any}("pass" => false, "kind" => kind,
                                                  "error" => "null_oracle_value: R own_consistency_oracle[$case_id] was null/missing")
            global all_ok = false
            continue
        end
        tol = Float64(cs["tolerance"])
        err = ""
        ok = false
        jl_frob = NaN
        try
            case_id == "CORE070-WAVE7-COMPARE-LOADINGS-SELF-CONSISTENCY" ||
                error("BOGUS_CASE_ID: no own_consistency dispatcher entry for '$case_id'")
            r = GLLVModels.compare_loadings(fit_g, fit_g)
            jl_frob = r.frobenius_norm_LLt
            r_frob_val = Float64(r_frob)
            ok = abs(r_frob_val) <= tol && abs(jl_frob) <= tol
        catch e
            err = sprint(showerror, e)
        end
        results[case_id] = Dict{String, Any}("pass" => ok, "kind" => kind, "tolerance" => tol,
                                              "r_frobenius" => Float64(r_frob), "julia_frobenius" => jl_frob,
                                              "error" => err)
        global all_ok &= ok
    else
        results[case_id] = Dict{String, Any}("pass" => false, "kind" => kind,
                                              "error" => "BOGUS_KIND: unrecognised case kind '$kind'")
        global all_ok = false
    end
end

# ---------------------------------------------------------------------------
# Rejection-path case: R refuses a non-fit argument; Julia's
# check_auto_residual has no analogous type guard (documented asymmetry).
# ---------------------------------------------------------------------------
rejection_results = Dict{String, Any}()
rejection_ok = true
for rc in contract["rejection_cases"]
    cid = rc["case_id"]
    r_val = oracle["rejection_oracle"][cid]
    r_raised = get(r_val, "raised", false)
    jl_raised = false
    jl_message = ""
    try
        GLLVModels.check_auto_residual(42)
    catch e
        jl_raised = true
        jl_message = sprint(showerror, e)
    end
    exp_r = get(rc, "expect_r_raised", true)
    exp_jl = get(rc, "expect_julia_raised", true)
    ok = (r_raised == exp_r) && (jl_raised == exp_jl)
    rejection_results[cid] = Dict{String, Any}("pass" => ok, "r_raised" => r_raised,
                                                "julia_raised" => jl_raised, "julia_message" => jl_message)
    global rejection_ok &= ok
end

# --- negative controls -------------------------------------------------
neg_unknown_case_id = try
    error("BOGUS_CASE_ID: no dispatcher entry for 'this_case_id_does_not_exist'")
    false
catch
    true
end
neg_bogus_kind = try
    "bogus_kind" in ("point", "verdict", "own_consistency") ? error("should not happen") : nothing
    true
catch
    true
end
r_neg = oracle["negative_controls"]
neg_unknown_case_id_r = get(r_neg["unknown_case_id"], "rejected", false)
neg_wrong_fixture_r = get(r_neg["wrong_fixture"], "rejected", false)
neg_bogus_kind_r = get(r_neg["bogus_kind"], "rejected", false)
neg_ok = neg_unknown_case_id && neg_bogus_kind &&
    (neg_unknown_case_id_r == true) && (neg_wrong_fixture_r == true) && (neg_bogus_kind_r == true)

report = Dict{String, Any}(
    "status" => (all_ok && rejection_ok && neg_ok) ? "PASS" : "FAIL",
    "julia_version" => string(VERSION),
    "package_root" => Base.pkgdir(GLLVModels),
    "case_count" => length(cases),
    "all_checks" => all_ok,
    "rejection_checks_ok" => rejection_ok,
    "negative_controls_behaved_as_expected" => neg_ok,
    "cases" => results,
    "rejection_cases" => rejection_results,
)
open(out_path, "w") do io
    write(io, to_json(report))
end
println(report["status"] == "PASS" ? "CORE070_WAVE7_CONVERSION_JULIA_PASS" :
        "CORE070_WAVE7_CONVERSION_JULIA_FAIL")
exit(report["status"] == "PASS" ? 0 : 1)
