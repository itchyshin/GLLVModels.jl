#!/usr/bin/env julia
# Julia-side receipt for the postfit rows POSTFIT-SURFACE-sanity_multi and
# POSTFIT-SURFACE-compare_loadings and the namespace row export/sanity_multi (true-parity checker,
# tools/true_parity_check.mjs, clauses C1/C8).
#
# A standalone sibling of tools/true_parity_julia_receipts.jl, kept in its own file so that the
# two do not collide. It follows the same rules, and the helpers below (Fail, findline,
# test_tolerance, cite, mkcase, the JSON writer and reader, Receipt, receipt_object) are copied
# from tools/true_parity_variance_decomp_receipt.jl unchanged:
#   * R values are copied from the tracked fixture test/fixtures/diagnostics_p1.toml; nothing
#     on the R side is recomputed here.
#   * The Julia side repeats the computation of test/test_diagnostics_p1.jl with the same
#     inputs and settings; the test file:line is recorded in `julia_source`.
#   * The tolerance is READ from the existing assertion in the test (file:line and the line's
#     text recorded). A difference above it aborts the run and nothing is written.
#   * src/, the test and the fixture are not modified.
# It writes three receipts under docs/dev-log/core070/true-parity-latest/receipts/julia-twins/:
# postfit-twins/sanity_multi.json, namespace-numeric/sanity_multi.json and
# postfit-twins/compare_loadings.json.
#
# Usage (from the repository root)
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_diagnostics_receipt.jl
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_diagnostics_receipt.jl --check
# --check recomputes and compares with the committed receipt: fixture and test sha256, case ids, R
# values and tolerances identical; each Julia value within 1e-3 x tolerance + 100 eps of the
# committed one; abs_diff still within tolerance.

using GLLVModels, TOML, SHA, Statistics, LinearAlgebra
using Distributions: Normal   # root-project dependency; family marker for fit_gllvm
const GMJ = GLLVModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins"
const P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const GENERATOR = "tools/true_parity_diagnostics_receipt.jl"
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
    "fresh by tools/true_parity_diagnostics_receipt.jl, which repeats the computation of the cited twin " *
    "test with the same inputs and settings; R was not run and is not recomputed here."

mat(rows) = permutedims(reduce(hcat, [Float64.(r) for r in rows]))   # TOML array of rows -> Matrix

function load_Y(path, trait_names, n)
    Y = zeros(Float64, length(trait_names), n)
    open(joinpath(ROOT, path)) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            a = split(line, ",")
            Y[findfirst(==(strip(a[2], '"')), trait_names), parse(Int, strip(a[1], '"'))] = parse(Float64, a[3])
        end
    end
    return Y
end

# ---------------------------------------------------------------------------------------------
# sanity_multi / compare_loadings   test/test_diagnostics_p1.jl
# ---------------------------------------------------------------------------------------------
function receipts_diagnostics()
    fxp = "test/fixtures/diagnostics_p1.toml"
    tp = "test/test_diagnostics_p1.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["gllvmtmb_commit"] == P1_SHA || fail("diagnostics fixture is not pinned at P1")
    dp = "test/fixtures/" * fx["data_file"]
    bytes2hex(sha256(read(joinpath(ROOT, dp)))) == fx["data_sha256"] || fail("data csv drifted: $dp")
    Y = load_Y(dp, String.(fx["trait_names"]), Int(fx["n_unit"]))

    # sanity_multi on the d = 1 and d = 2 fits, in that order (as in the test)
    r_se = Float64[]; j_se = Float64[]; r_ld = Float64[]; j_ld = Float64[]
    for key in ("d1", "d2")
        s = fx["sanity"][key]
        (s["converged"] === true && s["pd_hessian"] === true) || fail("R fit $key did not converge with a PD Hessian; not a valid twin")
        fit = GMJ.fit_gllvm(Y; family = Normal(), K = Int(s["d"]))
        fit.converged || fail("Julia fit $key did not converge")
        abs(fit.logLik - Float64(s["loglik"])) <= 1e-6 || fail("Julia fit $key logLik differs from R")
        js = GMJ.sanity_multi(fit; y = Y)
        (js.converged === s["flag_converged"] && js.sdreport_ok === s["flag_sdreport_ok"] &&
         js.pd_hessian === s["flag_pd_hessian"]) || fail("sanity_multi boolean flags differ from R on $key")
        collect(keys(js))[1:length(s["flag_names"])] == Symbol.(String.(s["flag_names"])) || fail("sanity_multi flag names/order differ from R on $key")
        push!(r_se, Float64(s["max_se"])); push!(j_se, js.max_se)
        push!(r_ld, Float64(s["rr_B_min_loading"])); push!(j_ld, js.rr_B_min_loading)
    end
    fitd = cite(tp, "fit = fit_gllvm(Y; family = Normal(), K = Int(s[\"d\"]))")
    callsm = cite(tp, "js = sanity_multi(fit; y = Y, io = io)")
    note_sm = "Gaussian fits value ~ 0 + trait + latent(0 + trait | unit, d = D, unique = FALSE), D = 1 and 2, on ns_gauss_p1_data.csv (p = 6, 200 units, one observation per cell; sha256 checked); both R fits converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test); R's flags converged, sdreport_ok and pd_hessian equal Julia's (TRUE on both fits) and the flag names and order are R's (asserted in the test and in this generator, not receipt cases: booleans). Values in the order d = 1, d = 2. Not compared: max_gradient, the largest gradient component at each optimiser's stopping point (R nlminb about 1e-3, Julia about 1e-12), which measures how far the optimiser went rather than the model; both are below R's 1e-2 threshold (asserted in the test). Limits: Gaussian one-tier fits only, so rr_W_min_loading (two-level fits) is not twinned, and R's MSPL and ridge branches have no Julia estimator."
    cs_sm(prefix) = [
        mkcase("$prefix-MAX-SE", "sanity_multi(fit)\$max_se, the largest fixed-effect (trait intercept) standard error, d = 1 and d = 2 fits (2 values)",
            "$fxp [sanity.d1.max_se, sanity.d2.max_se]", "GLLVModels.sanity_multi(fit; y = Y).max_se, as called at $callsm; fit as at $fitd",
            r_se, j_se, test_tolerance(tp, "@test isapprox(j_se, r_se;"), note_sm * " The R values (0.0801 and 0.0770) are SEs of six distinct intercepts, not ~0."),
        mkcase("$prefix-RR-B-MIN-LOADING", "sanity_multi(fit)\$rr_B_min_loading, min |diag Lambda_B[1:d, 1:d]|, d = 1 and d = 2 fits (2 values)",
            "$fxp [sanity.d1.rr_B_min_loading, sanity.d2.rr_B_min_loading]", "GLLVModels.sanity_multi(fit; y = Y).rr_B_min_loading, as called at $callsm; fit as at $fitd",
            r_ld, j_ld, test_tolerance(tp, "@test isapprox(j_ld, r_ld;"), note_sm * " Both engines identify Lambda_B lower-triangular, so the diagonal is comparable up to sign and the absolute value removes the sign; the R values are 0.740 and 0.872."),
    ]

    # compare_loadings on the two recorded pairs
    cs_cl = Case[]
    callcl = cite(tp, "r = compare_loadings(A, B)")
    for (key, tag, what) in (("fit_vs_truth", "FIT-VS-TRUTH", "a 6 x 2 pair: the d = 2 fit's Lambda_B (Lambda_a) against the loadings the data were simulated from (Lambda_b), the use R's documentation gives; optimal transform a small rotation (about 0.5 degrees), residual Frobenius distance 0.120"),
                             ("reflected", "REFLECTED", "an 8 x 3 pair: Lambda_b random, Lambda_a = Lambda_b Q' + noise (sd 0.05) with Q a random orthogonal matrix with det(Q) = -1 (seed 20261004), so the optimal transform is a reflection (det(R) < 0, asserted by the R generator); residual Frobenius distance 0.176"))
        c = fx["compare_loadings"][key]
        A = mat(c["Lambda_a"]); B = mat(c["Lambda_b"])
        r = GMJ.compare_loadings(A, B)
        collect(keys(r)) == [:R, :Lambda_a_rot, :frobenius, :cor_per_factor] || fail("compare_loadings field names/order differ from R")
        note = "Inputs: $what. Both engines receive the same doubles: the inputs are recorded in the fixture at 17 significant digits and read back by Julia, so the comparison is of the Procrustes computation alone (R: svd(crossprod(Lambda_b, Lambda_a)), R = v u'; Julia: svd(Lambda_b' Lambda_a), R = V U'). Matrices are compared entrywise (column-major order)."
        push!(cs_cl,
            mkcase("P1-JULIA-POSTFIT-COMPARE-LOADINGS-$tag-R", "compare_loadings(Lambda_a, Lambda_b)\$R, the $(size(r.R, 1)) x $(size(r.R, 2)) orthogonal transform",
                "$fxp [compare_loadings.$key.R]", "GLLVModels.compare_loadings(A, B).R, as called at $callcl",
                mat(c["R"]), r.R, test_tolerance(tp, "@test maximum(abs, r.R .- rR)"), note),
            mkcase("P1-JULIA-POSTFIT-COMPARE-LOADINGS-$tag-LAMBDA-A-ROT", "compare_loadings(Lambda_a, Lambda_b)\$Lambda_a_rot, the aligned $(size(A, 1)) x $(size(A, 2)) Lambda_a",
                "$fxp [compare_loadings.$key.Lambda_a_rot]", "GLLVModels.compare_loadings(A, B).Lambda_a_rot, as called at $callcl",
                mat(c["Lambda_a_rot"]), r.Lambda_a_rot, test_tolerance(tp, "@test maximum(abs, r.Lambda_a_rot .- "), note),
            mkcase("P1-JULIA-POSTFIT-COMPARE-LOADINGS-$tag-FROBENIUS", "compare_loadings(Lambda_a, Lambda_b)\$frobenius, the Frobenius distance after alignment",
                "$fxp [compare_loadings.$key.frobenius]", "GLLVModels.compare_loadings(A, B).frobenius, as called at $callcl",
                Float64(c["frobenius"]), r.frobenius, test_tolerance(tp, "@test abs(r.frobenius - "), note),
            mkcase("P1-JULIA-POSTFIT-COMPARE-LOADINGS-$tag-COR-PER-FACTOR", "compare_loadings(Lambda_a, Lambda_b)\$cor_per_factor ($(length(r.cor_per_factor)) values)",
                "$fxp [compare_loadings.$key.cor_per_factor]", "GLLVModels.compare_loadings(A, B).cor_per_factor, as called at $callcl",
                Float64.(c["cor_per_factor"]), r.cor_per_factor, test_tolerance(tp, "@test maximum(abs, r.cor_per_factor .- "), note))
    end

    fixtures = [fxp, dp]
    pr_origin = "itchyshin/GLLVModels.jl#684"
    return Pair{String,Receipt}[
        "postfit-twins/sanity_multi.json" => Receipt(["postfit/POSTFIT-SURFACE-sanity_multi"], pr_origin, fixtures, [tp], NOT_A_FIXTURE_PAIR, cs_sm("P1-JULIA-POSTFIT-SANITY-MULTI")),
        "namespace-numeric/sanity_multi.json" => Receipt(["namespace/export/sanity_multi"], pr_origin, fixtures, [tp], NOT_A_FIXTURE_PAIR, cs_sm("P1-JULIA-SANITY-MULTI")),
        "postfit-twins/compare_loadings.json" => Receipt(["postfit/POSTFIT-SURFACE-compare_loadings"], pr_origin, [fxp], [tp], NOT_A_FIXTURE_PAIR, cs_cl),
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
        receipts_diagnostics()
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
        isempty(probs) && (println("OK diagnostics receipts reproduce within $(CHECK_FRACTION) x tolerance"); return 0)
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
