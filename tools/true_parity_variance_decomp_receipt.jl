#!/usr/bin/env julia
# Julia-side receipt for the postfit row POSTFIT-SURFACE-extract_proportions and namespace rows export/extract_proportions, export/extract_residual_split (true-parity checker,
# tools/true_parity_check.mjs, clauses C1/C8).
#
# A standalone sibling of tools/true_parity_julia_receipts.jl, kept in its own file so that the
# two do not collide. It follows the same rules, and the helpers below (Fail, findline,
# test_tolerance, cite, mkcase, the JSON writer and reader, Receipt, receipt_object) are copied
# from that file unchanged:
#   * R values are copied from the tracked fixture test/fixtures/variance_decomp_p1.toml; nothing
#     on the R side is recomputed here.
#   * The Julia side repeats the computation of test/test_variance_decomp_p1.jl with the same
#     inputs and settings; the test file:line is recorded in `julia_source`.
#   * The tolerance is READ from the existing assertion in the test (file:line and the line's
#     text recorded). A difference above it aborts the run and nothing is written.
#   * src/, the test and the fixture are not modified.
# It writes three receipts under docs/dev-log/core070/true-parity-latest/receipts/julia-twins/: postfit-twins/extract_proportions.json,
# namespace-numeric/extract_proportions.json and namespace-numeric/extract_residual_split.json.
#
# Usage (from the repository root)
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_variance_decomp_receipt.jl
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_variance_decomp_receipt.jl --check
# --check recomputes and compares with the committed receipt: fixture and test sha256, case ids, R
# values and tolerances identical; each Julia value within 1e-3 x tolerance + 100 eps of the
# committed one; abs_diff still within tolerance.

using GLLVModels, TOML, SHA, Statistics, LinearAlgebra
using Distributions: Normal   # root-project dependency; family marker for fit_gllvm
const GMJ = GLLVModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins"
const P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const GENERATOR = "tools/true_parity_variance_decomp_receipt.jl"
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
    "fresh by tools/true_parity_variance_decomp_receipt.jl, which repeats the computation of the cited twin " *
    "test with the same inputs and settings; R was not run and is not recomputed here."

# R's write.csv long data -> p x n_cols response matrix and the unit index per column.
function load_csv(path, trait_names, n_cols; obs_col = nothing)
    p = length(trait_names)
    Y = zeros(Float64, p, n_cols)
    ind = zeros(Int, n_cols)
    open(joinpath(ROOT, path)) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            unit = parse(Int, strip(parts[1], '"'))
            col = obs_col === nothing ? unit : parse(Int, strip(parts[obs_col], '"'))
            t = findfirst(==(strip(parts[obs_col === nothing ? 2 : 3], '"')), trait_names)
            Y[t, col] = parse(Float64, parts[obs_col === nothing ? 3 : 4])
            ind[col] = unit
        end
    end
    return Y, ind
end

# ---------------------------------------------------------------------------------------------
# extract_proportions / extract_residual_split   test/test_variance_decomp_p1.jl
# ---------------------------------------------------------------------------------------------
function receipts_variance_decomp()
    fxp = "test/fixtures/variance_decomp_p1.toml"
    tp = "test/test_variance_decomp_p1.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["gllvmtmb_commit"] == P1_SHA || fail("variance_decomp fixture is not pinned at P1")
    u, t = fx["unique"], fx["two"]
    for blk in (u, t)
        (blk["converged"] === true && blk["pd_hessian"] === true) || fail("an R fit did not converge with a PD Hessian; not a valid twin")
        bytes2hex(sha256(read(joinpath(ROOT, "test/fixtures/" * blk["data_file"])))) == blk["data_sha256"] || fail("data csv drifted: $(blk["data_file"])")
    end
    udp, tdp = "test/fixtures/" * u["data_file"], "test/fixtures/" * t["data_file"]
    # unique: has_diag GllvmFit
    Yu, _ = load_csv(udp, String.(u["trait_names"]), Int(u["n_unit"]))
    fu = GMJ.fit_gllvm(Yu; family = Normal(), K = 2, has_diag = true)
    fu.converged || fail("one-tier Julia fit did not converge")
    abs(fu.logLik - Float64(u["loglik"])) <= 1e-6 || fail("one-tier logLik differs from R")
    # two-level
    Yt, ind = load_csv(tdp, String.(t["trait_names"]), Int(t["n_obs"]); obs_col = 2)
    ft = GMJ.fit_twolevel_gaussian(Yt, ind; K_B = 1, K_W = 1)
    ft.converged || fail("two-level Julia fit did not converge")
    abs(ft.loglik - Float64(t["loglik"])) <= 1e-6 || fail("two-level loglik differs from R")
    p = Int(t["p"])
    comps = Symbol.(String.(t["components"]))
    pr = GMJ.extract_proportions(ft)
    pr.component == repeat(comps; inner = p) || fail("component order differs from R's rows")
    rs = GMJ.extract_residual_split(ft)
    ps = GMJ.extract_proportions(fu)

    fitu = cite(tp, "fit = fit_gllvm(Y; family = Normal(), K = 2, has_diag = true)")
    fitt = cite(tp, "fit = fit_twolevel_gaussian(Y, ind; K_B = 1, K_W = 1)")
    note_u = "One-tier Gaussian rank-2 fit with a unit-level diagonal, p = 6, 200 units, one observation per cell (sha256 checked); R: latent(d = 2) + unique at unit, converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test). R's shared_unit proportion is diag(L L') / (diag(L L') + Psi); the Julia :unit denominator on a has_diag fit with K_W == 0 is the identified total sigma_y_site(fit) (#701), which is the same number because R's Psi absorbs the whole residual. The six R values lie strictly inside (0, 1), between 0.66 and 0.82: not one constant. Replaces the postfit batch case CORE070-ESTIMAND-REBIND-EXTRACT-PROPORTIONS, whose unique = FALSE fixture returned 1 for every trait."
    note_t = "Two-level Gaussian fit, p = 5, 120 units x 4 observations (sha256 checked); R: latent(d = 1) + unique at unit and at unit_obs, converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test). R's extract_proportions() long frame has four components (shared_unit, unique_unit, shared_unit_obs, unique_unit_obs) x 5 traits = 20 rows; the Julia TwoLevelFit method returns the same rows in the same order (asserted in the test and again in the receipt generator). The R proportions are 20 distinct non-constant values between 0.02 and 0.45. Limits: Gaussian only, so R's link_residual component (zero for a Gaussian fit) is absent in both."
    cs_pro(prefix, rid) = [
        mkcase("$prefix-ONE-TIER-SHARED-UNIT", "extract_proportions(fit), shared_unit proportion of a latent + unique fit at one tier ($(length(ps)) values)",
            "$fxp [unique.shared_unit_proportion]", "GLLVModels.extract_proportions(fit), as called at " * cite(tp, "@test isapprox(extract_proportions(fit), r_sh;") * "; fit as at $fitu",
            Float64.(u["shared_unit_proportion"]), ps, test_tolerance(tp, "@test isapprox(extract_proportions(fit), r_sh;"), note_u),
        mkcase("$prefix-TWO-LEVEL-PROPORTION", "extract_proportions(fit)\$proportion, all four components of a two-level fit (long format, $(length(pr.proportion)) values)",
            "$fxp [two.proportion]", "GLLVModels.extract_proportions(fit).proportion, as called at " * cite(tp, "@test isapprox(pr.proportion, r_prop;") * "; fit as at $fitt",
            Float64.(t["proportion"]), pr.proportion, test_tolerance(tp, "@test isapprox(pr.proportion, r_prop;"), note_t),
        mkcase("$prefix-TWO-LEVEL-VARIANCE", "extract_proportions(fit)\$variance, all four components of a two-level fit (long format, $(length(pr.variance)) values)",
            "$fxp [two.variance]", "GLLVModels.extract_proportions(fit).variance, as called at " * cite(tp, "@test isapprox(pr.variance, r_var;") * "; fit as at $fitt",
            Float64.(t["variance"]), pr.variance, test_tolerance(tp, "@test isapprox(pr.variance, r_var;"), note_t),
    ]
    note_s = "Two-level Gaussian fit, p = 5, 120 units x 4 observations (sha256 checked); R: latent(d = 1) + unique at unit and at unit_obs, converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test). Every observation is its own cell at unit_obs, so R treats the unit_obs diagonal as a genuine observation-level term and reports it as sigma2_e; the Julia TwoLevelFit method reports sigma2_W. The five R sigma2_e values are distinct (0.32 to 0.51). Limits: Gaussian only, so R's sigma2_d is 0 for every trait (asserted in the test, not a receipt case since an all-zero comparison cannot discriminate); the non-Gaussian sigma2_d values are not reachable because the package has no non-Gaussian two-level fit, and there is no GllvmFit method. Because sigma2_d is 0, sigma2_total equals sigma2_e here, so the SIGMA2-TOTAL case repeats the SIGMA2-E measurement and adds no independent evidence; the same quantity also appears as unique_unit_obs in the proportions variance case."
    cs_split = [
        mkcase("P1-JULIA-RESIDUAL-SPLIT-SIGMA2-E", "extract_residual_split(fit)\$sigma2_e of a two-level fit ($(p) values)",
            "$fxp [two.sigma2_e]", "GLLVModels.extract_residual_split(fit).sigma2_e, as called at " * cite(tp, "@test isapprox(rs.sigma2_e, r_e;") * "; fit as at $fitt",
            Float64.(t["sigma2_e"]), rs.sigma2_e, test_tolerance(tp, "@test isapprox(rs.sigma2_e, r_e;"), note_s),
        mkcase("P1-JULIA-RESIDUAL-SPLIT-SIGMA2-TOTAL", "extract_residual_split(fit)\$sigma2_total of a two-level fit ($(p) values)",
            "$fxp [two.sigma2_total]", "GLLVModels.extract_residual_split(fit).sigma2_total, as called at " * cite(tp, "@test isapprox(rs.sigma2_total,") * "; fit as at $fitt",
            Float64.(t["sigma2_total"]), rs.sigma2_total, test_tolerance(tp, "@test isapprox(rs.sigma2_total,"), note_s),
    ]
    fixtures = [fxp, udp, tdp]
    pr_origin = "itchyshin/GLLVModels.jl#684"
    return Pair{String,Receipt}[
        "postfit-twins/extract_proportions.json" => Receipt(["postfit/POSTFIT-SURFACE-extract_proportions"], pr_origin, fixtures, [tp], NOT_A_FIXTURE_PAIR, cs_pro("P1-JULIA-POSTFIT-PROPORTIONS", 0)),
        "namespace-numeric/extract_proportions.json" => Receipt(["namespace/export/extract_proportions"], pr_origin, fixtures, [tp], NOT_A_FIXTURE_PAIR, cs_pro("P1-JULIA-PROPORTIONS", 0)),
        "namespace-numeric/extract_residual_split.json" => Receipt(["namespace/export/extract_residual_split"], pr_origin, [fxp, tdp], [tp], NOT_A_FIXTURE_PAIR, cs_split),
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
        receipts_variance_decomp()
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
        isempty(probs) && (println("OK variance_decomp receipts reproduce within $(CHECK_FRACTION) x tolerance"); return 0)
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
