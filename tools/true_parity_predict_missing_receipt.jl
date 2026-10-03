#!/usr/bin/env julia
# Julia-side receipt for the postfit row POSTFIT-SURFACE-predict_missing (true-parity checker,
# tools/true_parity_check.mjs, clauses C1/C8).
#
# A standalone sibling of tools/true_parity_julia_receipts.jl, kept in its own file so that the
# two do not collide. It follows the same rules, and the helpers below (Fail, findline,
# test_tolerance, cite, mkcase, the JSON writer and reader, Receipt, receipt_object) are copied
# from that file unchanged:
#   * R values are copied from the tracked fixture test/fixtures/predict_missing_p1.toml; nothing
#     on the R side is recomputed here.
#   * The Julia side repeats the computation of test/test_predict_missing_p1.jl with the same
#     inputs and settings; the test file:line is recorded in `julia_source`.
#   * The tolerance is READ from the existing assertion in the test (file:line and the line's
#     text recorded). A difference above it aborts the run and nothing is written.
#   * src/, the test and the fixture are not modified.
# It writes docs/dev-log/core070/true-parity-latest/receipts/julia-twins/postfit-twins/predict_missing.json.
#
# Usage (from the repository root)
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_predict_missing_receipt.jl
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_predict_missing_receipt.jl --check
# --check recomputes and compares with the committed receipt: fixture and test sha256, case ids, R
# values and tolerances identical; each Julia value within 1e-3 x tolerance + 100 eps of the
# committed one; abs_diff still within tolerance.

using GLLVModels, TOML, SHA, Statistics, LinearAlgebra
const GMJ = GLLVModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins"
const P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const GENERATOR = "tools/true_parity_predict_missing_receipt.jl"
const CHECK_FRACTION = 1e-3        # of the case tolerance
const CHECK_EPS_MULT = 100         # rounding allowance, in eps(Float64) * max(1, |value|)

sha_file(rel) = bytes2hex(sha256(read(joinpath(ROOT, rel))))
filelines(rel) = readlines(joinpath(ROOT, rel))

struct Fail <: Exception
    msg::String
end
fail(msg) = throw(Fail(msg))

sha_file(rel) = bytes2hex(sha256(read(joinpath(ROOT, rel))))
filelines(rel) = readlines(joinpath(ROOT, rel))

"""Line number of the line of `rel` containing `frag`. Without `nth` the fragment must occur on
exactly one line (so a citation can never silently point at the wrong assertion)."""
function findline(rel, frag; nth = nothing)
    hits = [n for (n, l) in enumerate(filelines(rel)) if occursin(frag, l)]
    isempty(hits) && fail("$rel: fragment $(repr(frag)) not found")
    if nth === nothing
        length(hits) == 1 || fail("$rel: fragment $(repr(frag)) is on $(length(hits)) lines $(hits); make it unique or give nth")
        return hits[1]
    end
    nth <= length(hits) || fail("$rel: fragment $(repr(frag)) occurrence $nth not found")
    return hits[nth]
end

"""Tolerance literal of the assertion on the test line containing `frag` (`nth`-th such line).
The literal read is the first `<=`, `<` or `atol =` number AFTER the fragment on that line."""
function test_tolerance(rel, frag; nth = nothing, after = frag)
    n = findline(rel, frag; nth)
    text = filelines(rel)[n]
    i = findfirst(after, text)
    i === nothing && fail("$rel:$n: $(repr(after)) not on the line")
    rest = text[nextind(text, last(i)):end]
    m = match(r"(?:<=|<|atol\s*=)\s*([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)", rest)
    m === nothing && fail("$rel:$n: no tolerance literal after $(repr(after)) in $(strip(text))")
    return parse(Float64, m.captures[1]), "$rel:$n", String(strip(text))
end

cite(rel, frag; nth = nothing) = "$rel:$(findline(rel, frag; nth))"

asvec(x::Number) = Float64(x)
asvec(x::AbstractArray) = vec(Float64.(x))
asvec(x::AbstractVector) = Float64.(x)

absdiff(r::Number, j::Number) = abs(r - j)
function absdiff(r::AbstractVector, j::AbstractVector)
    (length(r) == length(j) && !isempty(r)) || fail("vector length mismatch $(length(r)) vs $(length(j))")
    return maximum(abs.(r .- j))
end

struct Case
    fields::Vector{Pair{String,Any}}
end

"""Build one comparison case. `tol` is (value, "file:line", line text) from `test_tolerance`
(or an explicit triple for the two derived statistical bounds of bootstrap_temporal)."""
function mkcase(id, quantity, r_source, julia_source, r, j, tol, note)
    rv, jv = asvec(r), asvec(j)
    d = absdiff(rv, jv)
    t, src, line = tol
    d <= t || fail("$id: abs diff $d > tolerance $t ($src); do not bind the row")
    return Case(Pair{String,Any}[
        "case_id" => id, "quantity" => quantity, "r_source" => r_source, "julia_source" => julia_source,
        "r_value" => rv, "julia_value" => jv, "abs_diff" => d, "tolerance" => t,
        "tolerance_source" => src, "tolerance_source_line" => line, "note" => note])
end

struct Receipt
    source_ids::Vector{String}
    origin_pr::String
    fixtures::Vector{String}
    tests::Vector{String}
    what_this_is_not::String
    cases::Vector{Case}
end

gllvmodels_commit() = strip(read(setenv(`git log -1 --format=%H -- src Project.toml`; dir = ROOT), String))

# ---------------------------------------------------------------------------------------------
# JSON: a writer that reproduces Python's json.dumps(obj, indent=2, ensure_ascii=False) byte for
# byte (so the receipts share the exact layout of the zi receipts), and a small reader for --check.
# ---------------------------------------------------------------------------------------------
function pyfloat(x::Float64)
    isfinite(x) || fail("non-finite number in receipt")
    x == 0 && return signbit(x) ? "-0.0" : "0.0"
    s = string(x)
    neg = startswith(s, "-")
    neg && (s = s[2:end])
    mant, ex = occursin('e', s) ? (split(s, 'e')...,) : (s, "0")
    ex10 = parse(Int, ex)
    ip, fp = occursin('.', mant) ? (split(mant, '.')...,) : (mant, "")
    digits = ip * fp
    pointpos = length(ip) + ex10                      # value = 0.DIGITS * 10^pointpos after strip
    lead = length(digits) - length(lstrip(==('0'), digits))
    digits = lstrip(==('0'), digits)
    pointpos -= lead
    digits = rstrip(==('0'), digits)
    isempty(digits) && (digits = "0")
    e = pointpos - 1                                  # scientific exponent
    body = if e < -4 || e >= 16
        m = length(digits) == 1 ? digits : digits[1] * "." * digits[2:end]
        m * "e" * (e < 0 ? "-" : "+") * lpad(string(abs(e)), 2, '0')
    elseif e >= 0
        intpart = length(digits) > e + 1 ? digits[1:e+1] : rpad(digits, e + 1, '0')
        frac = length(digits) > e + 1 ? digits[e+2:end] : "0"
        intpart * "." * frac
    else
        "0." * "0"^(-e - 1) * digits
    end
    return neg ? "-" * body : body
end

function jstring(s::AbstractString)
    io = IOBuffer()
    print(io, '"')
    for c in s
        c == '"' ? print(io, "\\\"") : c == '\\' ? print(io, "\\\\") :
        c == '\n' ? print(io, "\\n") : c == '\t' ? print(io, "\\t") :
        c == '\r' ? print(io, "\\r") : c < ' ' ? print(io, "\\u", lpad(string(Int(c), base = 16), 4, '0')) :
        print(io, c)
    end
    print(io, '"')
    return String(take!(io))
end

function jwrite(io::IO, x, ind::Int = 0)
    pad(n) = " "^(2n)
    if x isa AbstractString
        print(io, jstring(x))
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa Real
        print(io, pyfloat(Float64(x)))
    elseif x isa Case
        jwrite(io, x.fields, ind)
    elseif x isa AbstractVector{<:Pair}
        if isempty(x)
            print(io, "{}")
        else
            println(io, "{")
            for (i, (k, v)) in enumerate(x)
                print(io, pad(ind + 1), jstring(k), ": ")
                jwrite(io, v, ind + 1)
                println(io, i < length(x) ? "," : "")
            end
            print(io, pad(ind), "}")
        end
    elseif x isa AbstractVector
        if isempty(x)
            print(io, "[]")
        else
            println(io, "[")
            for (i, v) in enumerate(x)
                print(io, pad(ind + 1))
                jwrite(io, v, ind + 1)
                println(io, i < length(x) ? "," : "")
            end
            print(io, pad(ind), "]")
        end
    else
        fail("cannot serialise $(typeof(x))")
    end
end

function jrender(x)
    io = IOBuffer()
    jwrite(io, x)
    println(io)
    return String(take!(io))
end

# Minimal JSON reader (objects -> Dict, arrays -> Vector, numbers -> Float64/Int, strings, bool, null).
function jparse(s::AbstractString)
    i = Ref(1)
    peekc() = i[] <= lastindex(s) ? s[i[]] : '\0'
    function ws()
        while i[] <= lastindex(s) && isspace(s[i[]])
            i[] = nextind(s, i[])
        end
    end
    function val()
        ws()
        c = peekc()
        if c == '{'
            i[] += 1; d = Dict{String,Any}(); ws()
            if peekc() == '}'; i[] += 1; return d; end
            while true
                ws(); k = str(); ws(); peekc() == ':' || fail("json: ':' expected"); i[] += 1
                d[k] = val(); ws()
                if peekc() == ','; i[] += 1; continue; end
                peekc() == '}' || fail("json: '}' expected"); i[] += 1; return d
            end
        elseif c == '['
            i[] += 1; a = Any[]; ws()
            if peekc() == ']'; i[] += 1; return a; end
            while true
                push!(a, val()); ws()
                if peekc() == ','; i[] += 1; continue; end
                peekc() == ']' || fail("json: ']' expected"); i[] += 1; return a
            end
        elseif c == '"'
            return str()
        elseif startswith(SubString(s, i[]), "true"); i[] += 4; return true
        elseif startswith(SubString(s, i[]), "false"); i[] += 5; return false
        elseif startswith(SubString(s, i[]), "null"); i[] += 4; return nothing
        else
            m = match(r"^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?", SubString(s, i[]))
            m === nothing && fail("json: bad token at $(i[])")
            i[] += length(m.match)
            return (m.captures[1] === nothing && m.captures[2] === nothing) ? parse(Int, m.match) : parse(Float64, m.match)
        end
    end
    function str()
        peekc() == '"' || fail("json: string expected"); i[] += 1
        io = IOBuffer()
        while true
            c = peekc(); i[] = nextind(s, i[])
            if c == '"'; break
            elseif c == '\\'
                e = peekc(); i[] = nextind(s, i[])
                if e == 'u'
                    print(io, Char(parse(Int, SubString(s, i[], i[] + 3), base = 16))); i[] += 4
                else
                    print(io, e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e)
                end
            else
                print(io, c)
            end
        end
        return String(take!(io))
    end
    v = val(); ws()
    i[] > lastindex(s) || fail("json: trailing characters")
    return v
end

# ---------------------------------------------------------------------------------------------
# Receipt assembly
# ---------------------------------------------------------------------------------------------
function receipt_object(r::Receipt)
    return Pair{String,Any}[
        "schema" => "true-parity-julia-twin-receipt/v1",
        "source_ids" => r.source_ids,
        "verdict" => "PASS",
        "evidence_kind" => "julia_recomputed_vs_recorded_r",
        "pin" => "P1",
        "reference_commit" => P1_SHA,
        "origin_pr" => r.origin_pr,
        "generator" => GENERATOR,
        "julia_version" => string(VERSION),
        "gllvmodels_commit" => gllvmodels_commit(),
        "source_fixtures" => [Pair{String,Any}["path" => p, "sha256" => sha_file(p)] for p in r.fixtures],
        "source_tests" => [Pair{String,Any}["path" => p, "sha256" => sha_file(p)] for p in r.tests],
        "what_this_is_not" => r.what_this_is_not,
        "comparison" => Pair{String,Any}["pin" => "P1", "cases" => r.cases],
    ]
end

const NOT_A_FIXTURE_PAIR = "R values are copied from the tracked fixture. Julia values are computed " *
    "fresh by tools/true_parity_predict_missing_receipt.jl, which repeats the computation of the cited twin " *
    "test with the same inputs and settings; R was not run and is not recomputed here."

# ---------------------------------------------------------------------------------------------
# predict_missing   test/test_predict_missing_p1.jl
# ---------------------------------------------------------------------------------------------
function receipt_predict_missing()
    pmp = "test/fixtures/predict_missing_p1.toml"
    nsp = "test/fixtures/ns_numeric_p1.toml"
    tp = "test/test_predict_missing_p1.jl"
    pm = TOML.parsefile(joinpath(ROOT, pmp))
    ns = TOML.parsefile(joinpath(ROOT, nsp))
    pm["gllvmtmb_commit"] == P1_SHA || fail("predict_missing fixture is not pinned at P1")
    m = pm["main"]
    datap = "test/fixtures/" * m["data_file"]
    bytes2hex(sha256(read(joinpath(ROOT, datap)))) == m["data_sha256"] || fail("predict_missing data csv drifted")
    (m["converged"] === true && m["pd_hessian"] === true) || fail("predict_missing R fit not converged with a PD Hessian; not a valid twin")
    tn = String.(ns["trait_names"]); p, n = Int(ns["p"]), Int(ns["n_unit"])
    Y = zeros(Float64, p, n)
    open(joinpath(ROOT, datap)) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            a = split(line, ",")
            Y[findfirst(==(strip(a[2], '"')), tn), parse(Int, strip(a[1], '"'))] = parse(Float64, a[3])
        end
    end
    mu = Int.(m["masked_unit"]); mt = Int.(m["masked_trait"])
    mask = trues(p, n)
    for i in eachindex(mu)
        mask[mt[i], mu[i]] = false
    end
    count(!, mask) == Int(m["n_masked"]) || fail("mask does not have the recorded number of masked cells")
    X = zeros(Float64, p, n, p)
    for t in 1:p
        X[t, :, t] .= 1.0
    end
    Ym = copy(Y)
    Ym[.!mask] .= 0.0
    fit = GMJ.fit_gaussian_gllvm(Ym; K = 2, X = X, mask = mask)
    fit.converged || fail("masked Julia fit did not converge")
    abs(fit.logLik - Float64(m["loglik"])) <= 1e-6 || fail("masked logLik differs from R")
    outl = GMJ.predict_missing(fit, Ym; mask = mask, type = :link)
    outr = GMJ.predict_missing(fit, Ym; mask = mask, type = :response)
    (outl.col == mu && outl.row == mt) || fail("masked cells or their order differ from R's rows")
    fitc = cite(tp, "fit = fit_gaussian_gllvm(Ym; K = 2, X = X, mask = mask)")
    note = "Rank-2 Gaussian fit on the fixture data (sha256 checked) with " * string(count(!, mask)) *
        " masked cells (traits 2 and 5 in units u with u % 10 == 3, trait 1 in units with u % 7 == 4; the mask is recorded in the fixture); R: value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE) with miss_control(response = \"include\"), converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test). Cells and their order are asserted equal to R's predict_missing() rows. Julia takes the trait means through a one-hot X design and the caller re-supplies the mask (R reads it from the fit; see the predict_missing docstring). It replaces the postfit batch case CORE070-WAVE8-PREDICT-MISSING-ZERO-ROWS, which checked only that a complete-data fit returns zero rows. The R values are not one constant and not near zero. Limits: the mask pattern was chosen because many masks make fit_gaussian_gllvm(...; mask) throw PosDefException (Cholesky in src/families/aghq_gaussian_fit.jl ~205-212), a robustness bug outside this comparison; the mask that works is benign (light missingness), so heavy or irregular missingness is not covered."
    cs = [
        mkcase("P1-JULIA-POSTFIT-PREDICT-MISSING-LINK", "predict_missing(fit)\$est at the masked cells, link scale ($(length(mu)) values)",
            "$pmp [main.est_link]", "GLLVModels.predict_missing(fit, Ym; mask = mask, type = :link).est, as called at " * cite(tp, "out = predict_missing(fit, Ym; mask = mask, type = :link)") * "; fit as at $fitc",
            Float64.(m["est_link"]), outl.est, test_tolerance(tp, "@test isapprox(out.est, r_link;"), note),
        mkcase("P1-JULIA-POSTFIT-PREDICT-MISSING-RESPONSE", "predict_missing(fit, type = \"response\")\$est at the masked cells, response scale ($(length(mu)) values)",
            "$pmp [main.est_response]", "GLLVModels.predict_missing(fit, Ym; mask = mask, type = :response).est, as called at " * cite(tp, "outr = predict_missing(fit, Ym; mask = mask, type = :response)") * "; fit as at $fitc",
            Float64.(m["est_response"]), outr.est, test_tolerance(tp, "@test isapprox(outr.est,"), note * " Identity link, so the response scale equals the link scale in both engines: this case equals the link case bitwise, so it exercises the type = :response path but adds no independent numeric evidence."),
    ]
    return Pair{String,Receipt}["postfit-twins/predict_missing.json" =>
        Receipt(["postfit/POSTFIT-SURFACE-predict_missing"], "itchyshin/GLLVModels.jl#684", [pmp, datap], [tp], NOT_A_FIXTURE_PAIR, cs)]
end


function compare_receipt(rel, new::Receipt)
    path = joinpath(ROOT, OUT_DIR, rel)
    isfile(path) || return ["missing file $rel"]
    old = jparse(read(path, String))
    probs = String[]
    obj = Dict(receipt_object(new))
    for k in ("schema", "source_ids", "evidence_kind", "pin", "reference_commit", "origin_pr", "generator", "source_fixtures", "source_tests", "what_this_is_not")
        jparse(jrender(obj[k])) == old[k] || push!(probs, "$rel: field $k differs from the committed receipt")
    end
    oc = Dict(c["case_id"] => c for c in old["comparison"]["cases"])
    nc = Dict(String(c.fields[1].second) => Dict(c.fields) for c in new.cases)
    Set(keys(oc)) == Set(keys(nc)) || push!(probs, "$rel: case ids differ ($(sort(collect(symdiff(keys(oc), keys(nc))))))")
    for (id, n) in nc
        haskey(oc, id) || continue
        o = oc[id]
        ov, nv = asvec(o["julia_value"]), asvec(n["julia_value"])
        asvec(o["r_value"]) == asvec(n["r_value"]) || push!(probs, "$id: R value differs from the fixture")
        o["tolerance"] == n["tolerance"] || push!(probs, "$id: tolerance differs ($(o["tolerance"]) vs $(n["tolerance"]))")
        o["tolerance_source"] == n["tolerance_source"] || push!(probs, "$id: tolerance_source differs")
        length(ov) == length(nv) || (push!(probs, "$id: julia_value length differs"); continue)
        allow = CHECK_FRACTION * o["tolerance"] .+ CHECK_EPS_MULT * eps(Float64) .* max.(1.0, abs.(ov))
        worst = maximum(abs.(ov .- nv) ./ allow)
        worst <= 1 || push!(probs, "$id: recomputed Julia value moved by more than the strict allowance (max |d| / allowance = $(round(worst; sigdigits = 3)))")
        n["abs_diff"] <= n["tolerance"] || push!(probs, "$id: abs_diff now exceeds tolerance")
    end
    return probs
end

function main(args)
    check = "--check" in args
    receipts = try
        receipt_predict_missing()
    catch e
        e isa Fail || rethrow()
        println("FAIL ", e.msg)
        return 1
    end
    if check
        probs = String[]
        for (rel, r) in receipts
            append!(probs, compare_receipt(rel, r))
        end
        isempty(probs) && (println("OK predict_missing receipt reproduces within $(CHECK_FRACTION) x tolerance"); return 0)
        foreach(p -> println("STALE ", p), probs)
        return 1
    end
    for (rel, r) in receipts
        p = joinpath(ROOT, OUT_DIR, rel)
        mkpath(dirname(p))
        write(p, jrender(receipt_object(r)))
        for c in r.cases
            d = Dict(c.fields)
            println(d["case_id"], ": max abs diff ", d["abs_diff"], " tolerance ", d["tolerance"])
        end
    end
    println("wrote $(length(receipts)) receipt under $OUT_DIR (Julia $VERSION)")
    return 0
end

exit(main(ARGS))
