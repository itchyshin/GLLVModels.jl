#!/usr/bin/env julia
# Julia-side receipt for the namespace row export/confint_inspect (true-parity checker,
# tools/true_parity_check.mjs, clauses C1/C8).
#
# A standalone sibling of tools/true_parity_julia_receipts.jl, tools/true_parity_variance_decomp_receipt.jl
# and tools/true_parity_animal_scalar_receipt.jl, kept in its own file so that they do not collide.
# It follows the same rules, and the helpers below (Fail, findline, test_tolerance, cite, mkcase,
# the JSON writer and reader, Receipt, receipt_object, compare_receipt) are copied from
# tools/true_parity_animal_scalar_receipt.jl unchanged:
#   * R values are copied from the tracked fixture test/fixtures/confint_inspect_p1.toml; nothing
#     on the R side is recomputed here.
#   * The Julia side repeats the computation of test/test_confint_inspect_p1.jl with the same
#     inputs and settings; the test file:line is recorded in `julia_source`.
#   * The tolerance is READ from the existing assertion in the test (file:line and the line's
#     text recorded). A difference above it aborts the run and nothing is written.
#   * src/, the test and the fixture are not modified.
# It writes one receipt: docs/dev-log/core070/true-parity-latest/receipts/julia-twins/namespace-numeric/confint_inspect.json.
#
# Usage (from the repository root)
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_confint_inspect_receipt.jl
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_confint_inspect_receipt.jl --check
# --check recomputes and compares with the committed receipt: fixture and test sha256, case ids, R
# values and tolerances identical; each Julia value within 1e-3 x tolerance + 100 eps of the
# committed one; abs_diff still within tolerance.

using GLLVModels, TOML, SHA, Statistics, LinearAlgebra
using Distributions: Normal
const GMJ = GLLVModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins"
const P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const GENERATOR = "tools/true_parity_confint_inspect_receipt.jl"
const CHECK_FRACTION = 1e-3        # of the case tolerance
const CHECK_EPS_MULT = 100         # rounding allowance, in eps(Float64) * max(1, |value|)

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
    "fresh by tools/true_parity_confint_inspect_receipt.jl, which repeats the computation of the cited twin " *
    "test with the same inputs and settings; R was not run and is not recomputed here."

# R's write.csv long data (unit, trait, value) -> p x n response matrix; same parser as
# test/test_confint_inspect_p1.jl.
function load_data(path, trait_names, n)
    p = length(trait_names)
    Y = fill(NaN, p, n)
    open(joinpath(ROOT, path)) do io
        readline(io) == "\"unit\",\"trait\",\"value\"" || fail("unexpected header in $path")
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            t = findfirst(==(strip(parts[2], '"')), trait_names)
            t === nothing && fail("unrecognised trait in $path")
            Y[t, parse(Int, strip(parts[1], '"'))] = parse(Float64, parts[3])
        end
    end
    return Y
end

# ---------------------------------------------------------------------------------------------
# confint_inspect   test/test_confint_inspect_p1.jl
# ---------------------------------------------------------------------------------------------
function receipts_confint_inspect()
    fxp = "test/fixtures/confint_inspect_p1.toml"
    tp = "test/test_confint_inspect_p1.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["gllvmtmb_commit"] == P1_SHA || fail("confint_inspect fixture is not pinned at P1")
    m = fx["main"]
    (m["converged"] === true && m["pd_hessian"] === true) || fail("the R fit did not converge with a PD Hessian; not a valid twin")
    m["level"] == 0.95 || fail("fixture level is not 0.95")
    dp = "test/fixtures/" * m["data_file"]
    sha_file(dp) == m["data_sha256"] || fail("data csv drifted: $dp")
    Y = load_data(dp, String.(m["trait_names"]), Int(m["n_unit"]))
    all(isfinite, Y) || fail("data has missing cells")
    fit = GMJ.fit_gllvm(Y; family = Normal(), K = 2)
    (fit isa GMJ.GllvmFit && fit.converged) || fail("Julia fit did not converge")
    abs(fit.logLik - Float64(m["loglik"])) <= 1e-6 || fail("Julia optimum differs from R's (logLik)")
    lam = filter(startswith("Lambda_B["), GMJ.confint(fit, Y).term)
    (lam[2] == "Lambda_B[2,2]" && lam[3] == "Lambda_B[2,1]") || fail("Julia Lambda_B packing differs from R's theta_rr_B order")
    jterm = Dict("sigma_eps" => "sigma_eps", "b_fix[1]" => "beta[1]",
                 "Lambda_B_packed[2]" => lam[2], "Lambda_B_packed[3]" => lam[3])

    fitc = cite(tp, "fit = fit_gllvm(Y; family = Normal(), K = 2)")
    insc = cite(tp, "ci = confint_inspect(fit, Y; level = 0.95, parm = jp)")
    estc = cite(tp, "J[k] = (estimate = only(confint(fit, Y; level = 0.95, parm = jp).estimate),")
    note = "Gaussian fit of value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE), p = 6 traits, n = 200 units (the [main] data of the namespace numeric twins, ns_gauss_p1_data.csv, sha256 checked); R converged (nlminb code 0) with a positive-definite Hessian. R: confint_inspect(fit, parm, level = 0.95, ystep = 0.02)\$bounds on the natural scale (ystep 0.02 rather than the default 0.5 because the bound is read off the tmbprofile grid by linear interpolation; the default grid adds up to 2.2e-5 on these targets, recorded in the fixture and not compared). Julia: fit_gllvm(Y; family = Normal(), K = 2) (fit at $fitc); confint_inspect(fit, Y; level = 0.95, parm) for the Wald and profile bounds (at $insc) and confint(fit, Y; parm).estimate for the estimate (at $estc). Parameter map: R sigma_eps <-> Julia sigma_eps; R b_fix[1] <-> Julia beta[1]; R Lambda_B_packed[k] <-> Julia's k-th Lambda_B term (both engines pack the d diagonal entries first, then the strict lower triangle column by column; asserted on both sides). Not compared: R's \$curve, \$plot and \$diagnostics, and the Wald/profile disagreement flags (Julia's single flag uses the upper half-width, R's the full Wald half-width)."
    specs = [
        ("sigma_eps", "SIGMA-EPS", "sigma_eps", "residual SD sigma_eps (R exp of log_sigma_eps; Julia log-SD Wald, exp back-transformed)"),
        ("b_fix_1", "B-FIX-1", "b_fix_1", "trait-1 mean b_fix[1] (Julia beta[1])"),
        ("lambda_packed_2", "LAMBDA-PACKED-2", "lambda_packed_2", "loading Lambda_B_packed[2] = Lambda[2,2] (Julia Lambda_B[2,2])"),
        ("lambda_packed_3", "LAMBDA-PACKED-3", "lambda_packed_3", "loading Lambda_B_packed[3] = Lambda[2,1] (Julia Lambda_B[2,1])"),
    ]
    cs = Case[]
    for (k, tag, jk, what) in specs
        r = fx["inspect"][k]
        jp = jterm[r["parm"]]
        ci = GMJ.confint_inspect(fit, Y; level = 0.95, parm = jp)
        ci.term == [jp] || fail("$k: Julia returned terms $(ci.term)")
        est = only(GMJ.confint(fit, Y; level = 0.95, parm = jp).estimate)
        wald = [only(ci.wald_lower), only(ci.wald_upper)]
        prof = [only(ci.profile_lower), only(ci.profile_upper)]
        (all(isfinite, wald) && all(isfinite, prof)) || fail("$k: non-finite Julia bound")
        push!(cs, mkcase("P1-JULIA-CONFINT-INSPECT-$tag-ESTIMATE", "confint_inspect \$bounds estimate_natural: $what",
            "$fxp [inspect.$k.estimate]", "confint(fit, Y; parm = \"$jp\").estimate, as compared at " * cite(tp, "@test isapprox(J[\"$jk\"].estimate,") * "; fit as at $fitc",
            Float64(r["estimate"]), est, test_tolerance(tp, "@test isapprox(J[\"$jk\"].estimate,"), note))
        push!(cs, mkcase("P1-JULIA-CONFINT-INSPECT-$tag-WALD", "confint_inspect \$bounds [wald_lower_natural, wald_upper_natural], 95%: $what",
            "$fxp [inspect.$k.wald]", "confint_inspect(fit, Y; parm = \"$jp\") [wald_lower, wald_upper], as compared at " * cite(tp, "@test isapprox(J[\"$jk\"].wald,") * "; fit as at $fitc",
            Float64.(r["wald"]), wald, test_tolerance(tp, "@test isapprox(J[\"$jk\"].wald,"), note))
        push!(cs, mkcase("P1-JULIA-CONFINT-INSPECT-$tag-PROFILE", "confint_inspect \$bounds [lower_natural, upper_natural] (profile likelihood), 95%: $what",
            "$fxp [inspect.$k.profile]", "confint_inspect(fit, Y; parm = \"$jp\") [profile_lower, profile_upper], as compared at " * cite(tp, "@test isapprox(J[\"$jk\"].profile,") * "; fit as at $fitc",
            Float64.(r["profile"]), prof, test_tolerance(tp, "@test isapprox(J[\"$jk\"].profile,"), note))
    end
    return Pair{String,Receipt}[
        "namespace-numeric/confint_inspect.json" => Receipt(["namespace/export/confint_inspect"], "itchyshin/GLLVModels.jl#684", [fxp, dp], [tp], NOT_A_FIXTURE_PAIR, cs),
    ]
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
        receipts_confint_inspect()
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
        isempty(probs) && (println("OK confint_inspect receipt reproduces within $(CHECK_FRACTION) x tolerance"); return 0)
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
    println("wrote $(length(receipts)) receipts under $OUT_DIR (Julia $VERSION)")
    return 0
end

exit(main(ARGS))
