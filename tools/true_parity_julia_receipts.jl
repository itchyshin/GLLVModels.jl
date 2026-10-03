#!/usr/bin/env julia
# Julia-side receipts for the true-parity checker (tools/true_parity_check.mjs, clauses C1/C8).
#
# Companion of tools/true_parity_fixture_receipts.py. That tool can only bind a row whose fixture
# stores BOTH the R value and the Julia value. The rows handled here have fixtures holding only
# R's side; their twin tests compute the Julia side when they run. This script runs the SAME Julia
# computation the twin test runs, against the same tracked R fixture, and writes the fresh Julia
# value next to the R value copied from the fixture.
#
# Rules this file keeps
#   * R values are copied from the tracked fixtures (parsed with the TOML stdlib / the same
#     helper the tests use). Nothing on the R side is recomputed, refitted or typed here. The only
#     transform is the one the test itself applies (a fixed +-1 sign alignment of a latent score,
#     and atanh/log on the profile bounds, both named in the case note).
#   * The Julia side is computed by calling the package functions with the inputs and settings
#     the test uses. The computation for each case is copied from the test, and the test file:line
#     is recorded in `julia_source` (line numbers are looked up in the test text at run time, so a
#     moved or edited test changes the citation rather than leaving a stale one).
#   * A tolerance is READ from the existing assertion in the test (file:line recorded with the
#     line's text; the line must contain the quoted fragment, so an edited assertion fails this
#     script). Quantities the test does not assert have no case.
#   * A case whose recomputed difference exceeds its tolerance aborts the run (exit 1); nothing
#     is written for it, and the row must not be bound.
#   * src/, the tests and the fixtures are not modified.
#
# Usage (from the repository root; see "Environment" below)
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_julia_receipts.jl
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_julia_receipts.jl --check
#
# --check re-runs every computation and compares with the committed receipts, STRICTLY:
#   * fixture and test sha256, case ids, R values and tolerances must be identical;
#   * every recomputed Julia value must equal the committed one within
#       1e-3 * tolerance + 100 * eps * max(1, |value|)
#     per element (a thousandth of the tolerance, never the tolerance itself). Bitwise identity is
#     not required, because optimiser-driven values can differ in the last digits across Julia or
#     BLAS builds; a change in a value larger than this allowance means the code or fixture moved
#     and the receipts must be regenerated and reviewed;
#   * each case must still satisfy abs_diff <= tolerance.
#   Julia version and GLLVModels source commit are reported if they differ from the receipt
#   (informational; the value check above is what fails).
#
# Environment: runs under the package's own project (`--project=.`). It uses only GLLVModels,
# Distributions-free stdlib pieces (TOML, SHA, Statistics, Random, LinearAlgebra) and, for the
# temporal rows, test/fixtures/temporal_p1/fixture_helpers.jl, which the temporal tests include.
# No package from test/Project.toml is needed. Runtime on a Mac Studio with
# OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4: a few minutes.

using GLLVModels, TOML, SHA, Statistics, Random, LinearAlgebra
using Distributions: Normal, NegativeBinomial   # root-project dependency; family marker for select_lv (section 7) and the namespace numeric twins (section 9)
const GMJ = GLLVModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins"
const P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const GENERATOR = "tools/true_parity_julia_receipts.jl"
const CHECK_FRACTION = 1e-3        # of the case tolerance
const CHECK_EPS_MULT = 100         # rounding allowance, in eps(Float64) * max(1, |value|)

include(joinpath(ROOT, "test", "fixtures", "temporal_p1", "fixture_helpers.jl"))
include(joinpath(ROOT, "test", "fixtures", "aghq_p1", "aghq_p1_helpers.jl"))   # the aghq twin test includes the same file

# ---------------------------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------------------------
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
    "fresh by tools/true_parity_julia_receipts.jl, which repeats the computation of the cited twin " *
    "test with the same inputs and settings; R was not run and is not recomputed here."

# =============================================================================================
# 1. boundary-inference: chibar2_pvalue, variance_lrt
#    test/test_chibar2_variance_lrt_p1_twin.jl (itchyshin/GLLVModels.jl#548)
# =============================================================================================
function receipts_chibar()
    fxp = "test/parity/fixtures/chibar2_p1_fixture.toml"
    tp = "test/test_chibar2_variance_lrt_p1_twin.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["gllvmtmb_pin_sha"] == P1_SHA || fail("chibar fixture is not pinned at P1")

    rows = fx["chibar2_pvalue"]
    r_p = [Float64(row["pvalue"]) for row in rows]
    j_p = [Float64(chibar2_pvalue(row["LRT"], row["q"])) for row in rows]          # test line cited below
    tol_p = test_tolerance(tp, "chibar2_pvalue(LRT, q) ≈ pvalue")
    c1 = mkcase("P1-JULIA-CHIBAR2-PVALUE", "chibar2_pvalue(LRT, q) over the fixture grid (q = 1,2,3; 17 LRT values each)",
        "$fxp [[chibar2_pvalue]].pvalue ($(length(rows)) rows, fixture order)",
        "GLLVModels.chibar2_pvalue(row.LRT, row.q), as called at $(cite(tp, "chibar2_pvalue(LRT, q) ≈ pvalue"))",
        r_p, j_p, tol_p,
        "Closed-form chi-bar-square survival values. One vector case: abs_diff is the maximum over all rows, each of which the test asserts separately at this tolerance.")
    r1 = Receipt(["boundary-inference/chibar2_pvalue"], "itchyshin/GLLVModels.jl#548", [fxp], [tp],
        NOT_A_FIXTURE_PAIR, [c1])

    vrows = fx["variance_lrt"]
    vr = [variance_lrt(row["ll_full"], row["ll_reduced"]; n_boundary = row["n_boundary"]) for row in vrows]  # test line ~61
    cl = mkcase("P1-JULIA-VARIANCE-LRT-LRT", "variance_lrt(...).LRT over the 5 fixture rows",
        "$fxp [[variance_lrt]].LRT", "GLLVModels.variance_lrt(ll_full, ll_reduced; n_boundary).LRT, as called at $(cite(tp, "variance_lrt(row["))",
        [Float64(row["LRT"]) for row in vrows], [Float64(r.LRT) for r in vr], test_tolerance(tp, "r.LRT ≈ row"),
        "The test also asserts n_boundary equality (exact, no tolerance), so it has no numeric case.")
    cp = mkcase("P1-JULIA-VARIANCE-LRT-PVALUE", "variance_lrt(...).pvalue over the 5 fixture rows",
        "$fxp [[variance_lrt]].pvalue", "GLLVModels.variance_lrt(ll_full, ll_reduced; n_boundary).pvalue, as called at $(cite(tp, "variance_lrt(row["))",
        [Float64(row["pvalue"]) for row in vrows], [Float64(r.pvalue) for r in vr], test_tolerance(tp, "r.pvalue ≈ row"),
        "Self-Liang chi-bar-square p-value through the variance-component wrapper.")
    r2 = Receipt(["boundary-inference/variance_lrt"], "itchyshin/GLLVModels.jl#548", [fxp], [tp],
        NOT_A_FIXTURE_PAIR, [cl, cp])
    return ["boundary-inference/chibar2_pvalue.json" => r1, "boundary-inference/variance_lrt.json" => r2]
end

# =============================================================================================
# 2. ordinal/ordinal_logit   test/test_ordinal_logit_twin.jl (itchyshin/GLLVModels.jl#528)
# =============================================================================================
# Verbatim from test/test_ordinal_logit_twin.jl (_load_ordinal_logit_csv).
function _load_ordinal_logit_csv(path::AbstractString, trait_names::Vector{String}, n_unit::Integer)
    Y = zeros(Int, length(trait_names), n_unit)
    open(path) do io
        readline(io)                          # header: "unit","trait","value"
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            unit = parse(Int, strip(parts[1], '"'))
            trait = strip(parts[2], '"')
            value = parse(Int, parts[3])
            t = findfirst(==(trait), trait_names)
            t === nothing && error("unrecognised trait \"$trait\" in $path")
            Y[t, unit] = value
        end
    end
    return Y
end

function receipts_ordinal()
    fxp = "test/fixtures/ordinal_logit_p1.toml"
    tp = "test/test_ordinal_logit_twin.jl"
    fixture = TOML.parsefile(joinpath(ROOT, fxp))
    fixture["gllvmtmb_commit"] == P1_SHA || fail("ordinal fixture is not pinned at P1")
    trait_names = String.(fixture["trait_names"])
    n_unit = Int(fixture["n_unit"])
    datap = "test/fixtures/" * fixture["data_file"]
    bytes2hex(sha256(read(joinpath(ROOT, datap)))) == fixture["data_sha256"] || fail("ordinal data csv drifted")

    Y = _load_ordinal_logit_csv(joinpath(ROOT, datap), trait_names, n_unit)
    fit = fit_gllvm(Y; family = ordinal_logit(), K = 1)                               # test line ~118
    fit.converged || fail("ordinal fit did not converge")
    r = fixture["r_reference"]

    # (2) Laplace marginal AT R's fitted (beta, Lambda, tau), exactly as in the test.
    β_r = Float64.(r["b_fix"])
    λ_r = Float64.(r["lambda_b"])
    Λ_r = reshape(λ_r, length(trait_names), 1)
    cp2 = Float64.(r["cutpoint_2"])
    cp3 = Float64.(r["cutpoint_3"])
    τ_r = zeros(length(trait_names), 3)
    for t in eachindex(trait_names)
        τ_r[t, 2] = cp2[t]
        τ_r[t, 3] = cp3[t]
    end
    C = fill(4, length(trait_names))
    ll_at_r = GLLVModels.ordinal_marginal_loglik_laplace_pertrait(Y, Λ_r, β_r, τ_r, C; link = LogitLink())

    LLt_julia = fit.Λ * fit.Λ'
    LLt_r = reduce(hcat, [Float64.(row) for row in r["lambda_lambda_t"]])'

    fitc = cite(tp, "fit = fit_gllvm(Y; family = ordinal_logit(), K = 1)")
    cases = [
        mkcase("P1-JULIA-ORDINAL-LOGIT-LOGLIK-OPTIMUM", "logLik at each side's own optimum",
            "$fxp [r_reference.loglik]", "GLLVModels.fit_gllvm(Y; family = ordinal_logit(), K = 1).loglik, as at $fitc",
            r["loglik"], fit.loglik, test_tolerance(tp, "isapprox(fit.loglik, r[\"loglik\"]"),
            "Y is read from test/fixtures/ordinal_logit_p1_data.csv (sha256 checked against the fixture), as in the test."),
        mkcase("P1-JULIA-ORDINAL-LOGIT-CROSS-OBJECTIVE-AT-R-PARAMETERS", "Julia Laplace marginal evaluated at R's fitted (beta, Lambda, tau)",
            "$fxp [r_reference.loglik]",
            "GLLVModels.ordinal_marginal_loglik_laplace_pertrait(Y, Lambda_r, beta_r, tau_r, C; link = LogitLink()), as at $(cite(tp, "ll_at_r = GLLVModels.ordinal_marginal_loglik_laplace_pertrait"))",
            r["loglik"], ll_at_r, test_tolerance(tp, "isapprox(ll_at_r, r[\"loglik\"]"),
            "Likelihood-function identity at R's parameters, independent of either optimiser's path."),
        mkcase("P1-JULIA-ORDINAL-LOGIT-CUTPOINT-2", "cutpoint_2 per trait (t1, t2, t3)",
            "$fxp [r_reference.cutpoint_2]", "GLLVModels fit.tau[:, 2] from the same fit, as compared at $(cite(tp, "fit.τ[t, 2], r[\"cutpoint_2\"][t]"))",
            cp2, [fit.τ[t, 2] for t in eachindex(trait_names)], test_tolerance(tp, "fit.τ[t, 2], r[\"cutpoint_2\"][t]"),
            "Per-trait assertion in the test; vector case takes the maximum."),
        mkcase("P1-JULIA-ORDINAL-LOGIT-CUTPOINT-3", "cutpoint_3 per trait (t1, t2, t3)",
            "$fxp [r_reference.cutpoint_3]", "GLLVModels fit.tau[:, 3] from the same fit, as compared at $(cite(tp, "fit.τ[t, 3], r[\"cutpoint_3\"][t]"))",
            cp3, [fit.τ[t, 3] for t in eachindex(trait_names)], test_tolerance(tp, "fit.τ[t, 3], r[\"cutpoint_3\"][t]"),
            "Per-trait assertion in the test; vector case takes the maximum."),
        mkcase("P1-JULIA-ORDINAL-LOGIT-LAMBDA-LAMBDA-T", "Lambda Lambda' (3 x 3, flattened column-major)",
            "$fxp [r_reference.lambda_lambda_t]", "GLLVModels fit.Λ * fit.Λ' from the same fit, as compared at $(cite(tp, "isapprox(LLt_julia, Matrix(LLt_r)"))",
            Matrix(LLt_r), LLt_julia, test_tolerance(tp, "isapprox(LLt_julia, Matrix(LLt_r)"),
            "Sign-free loading target (K = 1 loadings are identified only up to sign). R matrix is symmetric, so row/column-major flattening agree."),
    ]
    return ["ordinal/ordinal_logit.json" => Receipt(["ordinal/ordinal_logit"], "itchyshin/GLLVModels.jl#528",
        [fxp, datap], [tp], NOT_A_FIXTURE_PAIR, cases)]
end

# =============================================================================================
# 3. latent-scores/extract_latent_scores (+ .gllvmTMB_multi)
#    test/test_extract_latent_scores.jl (itchyshin/GLLVModels.jl#531)
#    The R z matrices in the fixture come from extract_latent_scores() called on a fit of class
#    gllvmTMB_multi (test header; generate_fixture.R), i.e. the .gllvmTMB_multi method through the
#    generic, so one receipt backs both rows. `.default` is not bound: see the PR text.
# =============================================================================================
_els_read_csv(path) = (lines = readlines(path); permutedims(reduce(hcat, [parse.(Float64, split(l, ',')) for l in lines[2:end]])))
_els_read_vector(path) = parse.(Float64, readlines(path))
_els_read_scalar(path) = parse(Float64, only(readlines(path)))

function receipts_latent_scores()
    dir = "test/fixtures/extract_latent_scores_p1"
    tp = "test/test_extract_latent_scores.jl"
    sums = Dict(split(l)[2] => split(l)[1] for l in readlines(joinpath(ROOT, dir, "SHA256SUMS.txt")) if !isempty(strip(l)))
    used = String[]
    function load(name, f)
        path = joinpath(ROOT, dir, name)
        bytes2hex(sha256(read(path))) == sums[name] || fail("$dir/$name sha256 differs from SHA256SUMS.txt")
        push!(used, "$dir/$name")
        return f(path)
    end
    Y_gauss_sites = load("Y_gauss.csv", _els_read_csv)
    Y_pois_sites = load("Y_pois.csv", _els_read_csv)
    z_r_gauss = load("z_gauss_unit.csv", _els_read_csv)
    z_r_pois = load("z_pois_unit.csv", _els_read_csv)
    Lambda_r_gauss = load("Lambda_gauss_hat.csv", _els_read_csv)
    Lambda_r_pois = load("Lambda_pois_hat.csv", _els_read_csv)
    beta_r_gauss = load("beta_gauss_hat.txt", _els_read_vector)
    beta_r_pois = load("beta_pois_hat.txt", _els_read_vector)
    sigma_eps_r = load("sigma_eps_hat.txt", _els_read_scalar)

    Y_gauss = permutedims(Y_gauss_sites)
    Y_pois = Int.(permutedims(Y_pois_sites))
    p, n = size(Y_gauss)
    K = size(z_r_gauss, 2)
    X = zeros(p, n, p)
    for i in 1:p, s in 1:n
        X[i, s, i] = 1.0
    end

    # Gaussian own optimum (test "Gaussian: own optimum").
    fit = fit_gaussian_gllvm(Y_gauss; K = K, X = X)
    z = extract_latent_scores(fit, Y_gauss; level = :unit, X = X)
    Lambda_j = getLoadings(fit; rotate = false)
    LZt_j_g = Lambda_j * z'
    LZt_r_g = Lambda_r_gauss * z_r_gauss'

    # Gaussian at R's fitted parameters (test "Gaussian: at R's fitted parameters").
    fit0 = fit_gaussian_gllvm(Y_gauss; K = K, X = X)
    fit_atR = GLLVModels.GllvmFit(fit0.model,
        merge(fit0.pars, (Λ = Lambda_r_gauss, β = beta_r_gauss, σ_eps = sigma_eps_r)),
        fit0.logLik, fit0.n_iter, fit0.converged, fit0.optim_result, fit0.cputime,
        fit0.integration)
    z_atR_g = extract_latent_scores(fit_atR, Y_gauss; level = :unit, X = X)

    # Poisson own optimum and at R's parameters.
    fitp = fit_poisson_gllvm(Y_pois; K = K)
    zp = extract_latent_scores(fitp, Y_pois; level = :unit)
    LZt_j_p = getLoadings(fitp; rotate = false) * zp'
    LZt_r_p = Lambda_r_pois * z_r_pois'
    fit0p = fit_poisson_gllvm(Y_pois; K = K)
    fit_atR_p = GLLVModels.PoissonFit(beta_r_pois, Lambda_r_pois, fit0p.link, fit0p.loglik,
        fit0p.converged, fit0p.iterations, fit0p.alpha_lv, fit0p.theta_packed, fit0p.hessian,
        fit0p.integration)
    z_atR_p = extract_latent_scores(fit_atR_p, Y_pois; level = :unit)

    # NB2 (NBGroupedFit, per-trait dispersion) at R's parameters.
    Y_nb2_sites = load("Y_nb2.csv", _els_read_csv)
    z_r_nb2 = load("z_nb2_unit.csv", _els_read_csv)
    Lambda_r_nb2 = load("Lambda_nb2_hat.csv", _els_read_csv)
    beta_r_nb2 = load("beta_nb2_hat.txt", _els_read_vector)
    phi_r_nb2 = load("phi_nb2_hat.txt", _els_read_vector)
    Y_nb2 = Int.(permutedims(Y_nb2_sites))
    fit_atR_nb = GLLVModels.NBGroupedFit(beta_r_nb2, Lambda_r_nb2, phi_r_nb2,
        collect(1:size(Y_nb2, 1)), LogLink(), NaN, true, 0)
    z_atR_nb = extract_latent_scores(fit_atR_nb, Y_nb2; level = :unit)

    calc = "GLLVModels.extract_latent_scores(fit, Y; level = :unit)"
    function c(id, q, rsrc, jsrc, r, j, tol, note)
        return mkcase(id, q, rsrc, jsrc, r, j, tol, note)
    end
    cases = [
        c("P1-JULIA-ELS-GAUSSIAN-Z-AT-R-PARAMETERS", "Gaussian K = 2 latent scores (15 x 2, column-major) at R's fitted parameters",
            "$dir/z_gauss_unit.csv (R extract_latent_scores(level = \"unit\"))",
            "$calc on a GllvmFit carrying R's Lambda, beta, sigma_eps, as at $(cite(tp, "z_atR = extract_latent_scores(fit_atR, Y_gauss"))",
            z_r_gauss, z_atR_g, test_tolerance(tp, "isapprox(z_atR, z_r_gauss"),
            "Isolates the definition of the score from optimiser differences, as the test does."),
        c("P1-JULIA-ELS-POISSON-Z-AT-R-PARAMETERS", "Poisson K = 2 latent scores (15 x 2, column-major) at R's fitted parameters",
            "$dir/z_pois_unit.csv (R extract_latent_scores(level = \"unit\"))",
            "$calc on a PoissonFit carrying R's beta and Lambda, as at $(cite(tp, "z_atR = extract_latent_scores(fit_atR, Y_pois"))",
            z_r_pois, z_atR_p, test_tolerance(tp, "isapprox(z_atR, z_r_pois"),
            "Laplace-Newton posterior mode at R's parameters."),
        c("P1-JULIA-ELS-NB2-Z-AT-R-PARAMETERS", "NB2 (per-trait dispersion) K = 2 latent scores (60 x 2, column-major) at R's fitted parameters",
            "$dir/z_nb2_unit.csv (R extract_latent_scores(level = \"unit\"))",
            "$calc on an NBGroupedFit carrying R's beta, Lambda, phi, as at $(cite(tp, "z_atR = extract_latent_scores(fit_atR, Y_nb2"))",
            z_r_nb2, z_atR_nb, test_tolerance(tp, "isapprox(z_atR, z_r_nb2"),
            "Own (larger) fixture dataset, see generate_fixture.R."),
        c("P1-JULIA-ELS-GAUSSIAN-LAMBDA-Z-OWN-OPTIMUM", "Gaussian Lambda z' (6 x 15, column-major), each side at its own optimum",
            "$dir/Lambda_gauss_hat.csv and z_gauss_unit.csv (Lambda_R z_R')",
            "GLLVModels getLoadings(fit; rotate = false) * extract_latent_scores(fit, Y; level = :unit, X)', as at $(cite(tp, "LZt_j = Lambda_j * z'", nth = 1))",
            LZt_r_g, LZt_j_g, test_tolerance(tp, "isapprox(LZt_j, LZt_r", nth = 1),
            "Rotation- and sign-invariant product: (Lambda, z) are identified only up to an orthogonal rotation."),
        c("P1-JULIA-ELS-POISSON-LAMBDA-Z-OWN-OPTIMUM", "Poisson Lambda z' (6 x 15, column-major), each side at its own optimum",
            "$dir/Lambda_pois_hat.csv and z_pois_unit.csv (Lambda_R z_R')",
            "GLLVModels getLoadings(fit; rotate = false) * extract_latent_scores(fit, Y; level = :unit)', as at $(cite(tp, "LZt_j = Lambda_j * z'", nth = 2))",
            LZt_r_p, LZt_j_p, test_tolerance(tp, "isapprox(LZt_j, LZt_r", nth = 2),
            "Rotation- and sign-invariant product."),
    ]
    note = NOT_A_FIXTURE_PAIR * " The R z matrices were produced by extract_latent_scores() on a fit of class gllvmTMB_multi."
    return ["latent-scores/extract_latent_scores.json" => Receipt(
        ["latent-scores/extract_latent_scores", "latent-scores/extract_latent_scores.gllvmTMB_multi"],
        "itchyshin/GLLVModels.jl#531", [dir * "/SHA256SUMS.txt"; unique(used)], [tp], note, cases)]
end

# =============================================================================================
# 4. temporal/*   test/test_temporal_fit_receipts.jl (#563) and test/test_temporal_helpers.jl (#543)
# =============================================================================================
const TFR = "test/test_temporal_fit_receipts.jl"
const TH = "test/test_temporal_helpers.jl"
const TDIR = "test/fixtures/temporal_p1"

# Verbatim from test/test_temporal_helpers.jl (theta_scale).
theta_scale(structure, v) = ismissing(v) ? missing :
    structure == "ar1" ? atanh(v / (1 - 1e-6)) : log(v)

struct TFit
    c::Dict{String,Any}
    at_r
    at_rt
    at_j
    cross::Dict{String,Any}
end

function temporal_fits()
    F = temporal_p1_load("fits.toml")
    F["gllvmTMB_commit"] == P1_SHA || fail("fits.toml not pinned at P1")
    temporal_p1_sha("fits.toml") == TEMPORAL_P1_SHA256["fits.toml"] || fail("fits.toml sha differs from the test's table")
    Xc = temporal_p1_load("cross_objective.toml")
    temporal_p1_sha("cross_objective.toml") == TEMPORAL_P1_SHA256["cross_objective.toml"] || fail("cross_objective.toml sha differs")
    cross = Dict(c["id"] => c for c in Xc["cells"])
    fits = TFit[]
    for c in F["fits"]
        rpar = temporal_p1_vec(c["par"]); rtight = temporal_p1_vec(c["par_tight"])
        at_r = temporal_p1_fit(F, c; start = rpar, iterations = 0)                    # TFR:~38
        at_rt = temporal_p1_fit(F, c; start = rtight, iterations = 0)
        xc = cross[c["id"]]
        at_j = temporal_p1_fit(F, c; start = temporal_p1_vec(xc["julia_par"]), iterations = 0)
        push!(fits, TFit(c, at_r, at_rt, at_j, xc))
    end
    return F, fits
end

fixture_paths(names...) = [TDIR * "/" * n for n in names]

function mode_row(F, fits, which::Function, source_id, stem)
    sel = filter(f -> which(f.c), fits)
    ids = [f.c["id"] for f in sel]
    fxs = "$TDIR/fits.toml"
    idnote = "Fits (fixture order): " * join(ids, ", ") * "."
    cases = Case[]
    catv(f) = vcat(f...)
    push!(cases, mkcase("P1-JULIA-$(stem)-NLL-AT-R-COORDINATES", "Julia NLL at R's opt\$par vs R's objective, per fit",
        "$fxs [fits.objective]", "-GLLVModels.fit_temporal_gllvm(...; start = R par, iterations = 0).loglik, as at $(cite(TFR, "@test abs(-at_r.loglik - c[\"objective\"])"))",
        [f.c["objective"] for f in sel], [-f.at_r.loglik for f in sel],
        test_tolerance(TFR, "@test abs(-at_r.loglik - c[\"objective\"])"), idnote))
    push!(cases, mkcase("P1-JULIA-$(stem)-NLL-AT-R-TIGHT-COORDINATES", "Julia NLL at R's tight-tolerance opt\$par vs R's tight objective, per fit",
        "$fxs [fits.objective_tight]", "-GLLVModels.fit_temporal_gllvm(...; start = R par_tight, iterations = 0).loglik, as at $(cite(TFR, "@test abs(-at_rt.loglik - c[\"objective_tight\"])"))",
        [f.c["objective_tight"] for f in sel], [-f.at_rt.loglik for f in sel],
        test_tolerance(TFR, "@test abs(-at_rt.loglik - c[\"objective_tight\"])"), idnote))
    push!(cases, mkcase("P1-JULIA-$(stem)-CROSS-OBJECTIVE-AT-JULIA-OPTIMUM", "R's objective at the recorded Julia optimum vs Julia NLL there, per fit",
        "$TDIR/cross_objective.toml [cells.r_fn_at_julia_par]", "-GLLVModels.fit_temporal_gllvm(...; start = recorded julia_par, iterations = 0).loglik, as at $(cite(TFR, "@test abs(-at_j.loglik - xc[\"r_fn_at_julia_par\"])"))",
        [f.cross["r_fn_at_julia_par"] for f in sel], [-f.at_j.loglik for f in sel],
        test_tolerance(TFR, "@test abs(-at_j.loglik - xc[\"r_fn_at_julia_par\"])"), idnote))
    push!(cases, mkcase("P1-JULIA-$(stem)-SIGMA-T-AT-R-COORDINATES", "Temporal covariance Sigma_T at R's coordinates (all fits, flattened)",
        "$fxs [fits.Sigma_T]", "GLLVModels fit.Sigma_T at R par, as compared at $(cite(TFR, "vec(at_r.Sigma_T)"))",
        catv([temporal_p1_vec(f.c["Sigma_T"]) for f in sel]), catv([vec(f.at_r.Sigma_T) for f in sel]),
        test_tolerance(TFR, "vec(at_r.Sigma_T)"), idnote))
    push!(cases, mkcase("P1-JULIA-$(stem)-AIC-AT-R-COORDINATES", "AIC at R's coordinates, per fit",
        "$fxs [fits.AIC]", "GLLVModels aic(fit) at R par, as at $(cite(TFR, "abs(aic(at_r) - c[\"AIC\"])"))",
        [f.c["AIC"] for f in sel], [aic(f.at_r) for f in sel], test_tolerance(TFR, "abs(aic(at_r) - c[\"AIC\"])"), idnote))
    push!(cases, mkcase("P1-JULIA-$(stem)-BIC-AT-R-COORDINATES", "BIC at R's coordinates, per fit",
        "$fxs [fits.BIC]", "GLLVModels bic(fit) at R par, as at $(cite(TFR, "abs(bic(at_r) - c[\"BIC\"])"))",
        [f.c["BIC"] for f in sel], [bic(f.at_r) for f in sel], test_tolerance(TFR, "abs(bic(at_r) - c[\"BIC\"])"), idnote))
    return sel, cases
end

function optimum_case(F, sel, stem)
    # test/test_temporal_fit_receipts.jl "(iii) between optima": a fit is held to |dll| <= 1e-6 when
    # Julia does not sit at a higher optimum than R's stopping point; cells where it does are
    # handled by the test through R's own objective (they are the cross-objective cases above) and
    # are listed here, not silently dropped.
    used, dropped = TFit[], String[]
    rv, jv = Float64[], Float64[]
    for f in sel
        jf = temporal_p1_fit(F, f.c; g_tol = TEMPORAL_P1_GTOL, iterations = TEMPORAL_P1_ITERATIONS)
        dll = jf.loglik - (-f.c["objective_tight"])
        dll >= -1e-8 || fail("$(f.c["id"]): Julia optimum is worse than R's by $dll")
        if dll > 1e-6
            push!(dropped, f.c["id"])
        else
            push!(used, f); push!(rv, -f.c["objective_tight"]); push!(jv, jf.loglik)
        end
    end
    isempty(used) && return nothing
    note = "Fits compared: " * join([f.c["id"] for f in used], ", ") * ". " *
        (isempty(dropped) ? "No fit was excluded." :
         "Excluded, because Julia reached a higher local optimum than R's stopping point (the test checks those through R's own objective at Julia's point instead, see the cross-objective case): " *
         join(dropped, ", ") * ".") * " Julia fit settings g_tol = $(TEMPORAL_P1_GTOL), iterations = $(TEMPORAL_P1_ITERATIONS), as the test."
    return mkcase("P1-JULIA-$(stem)-LOGLIK-AT-OPTIMA", "logLik at each side's own optimum, per fit",
        "$TDIR/fits.toml [fits.objective_tight] (negated)", "GLLVModels.fit_temporal_gllvm(...; g_tol, iterations).loglik, as at $(cite(TFR, "dll = jf.loglik - (-c[\"objective_tight\"])"))",
        rv, jv, test_tolerance(TFR, "@test abs(dll) <= 1e-6"; after = "abs(dll)"), note)
end

function receipts_temporal()
    F, fits = temporal_fits()
    fitf = "$TDIR/fits.toml"
    common_fx = fixture_paths("fits.toml", "cross_objective.toml")
    out = Pair{String,Receipt}[]
    not_pair = NOT_A_FIXTURE_PAIR

    # -- temporal_indep / temporal_dep / temporal_latent --------------------------------------
    for (mode_pred, sid, stem) in (
            (c -> c["mode"] == "indep", "temporal/temporal_indep", "TEMPORAL-INDEP"),
            (c -> c["mode"] == "dep", "temporal/temporal_dep", "TEMPORAL-DEP"),
            (c -> c["mode"] == "latent", "temporal/temporal_latent", "TEMPORAL-LATENT"))
        sel, cases = mode_row(F, fits, mode_pred, sid, stem)
        oc = optimum_case(F, sel, stem)
        oc === nothing || push!(cases, oc)
        if sid == "temporal/temporal_latent"
            # getLV vs R's reported conditional scores and loadings (TFR: getLV block).
            lvs = [(f, getLV(f.at_r)) for f in sel]
            zr = Float64[]; zj = Float64[]; lr = Float64[]; lj = Float64[]
            for (f, lv) in lvs
                z = temporal_p1_vec(f.c["report_z_temporal_state"])
                length(z) == size(lv.scores, 1) || fail("$(f.c["id"]): score length")
                append!(zr, lv.sign.multiplier .* z); append!(zj, lv.scores[:, 1])
                append!(lr, temporal_p1_vec(f.c["report_Lambda_temporal"])); append!(lj, vec(f.at_r.loadings))
            end
            push!(cases, mkcase("P1-JULIA-TEMPORAL-LATENT-GETLV-SCORES", "conditional latent state scores at R's coordinates, per fit (concatenated)",
                "$fitf [fits.report_z_temporal_state] times the +-1 sign multiplier of getLV's anchor", "GLLVModels.getLV(fit).scores[:, 1] at R par, as at $(cite(TFR, "lv.scores[:, 1] .- lv.sign.multiplier .* zR"))",
                zr, zj, test_tolerance(TFR, "@test d <= 1e-10"; after = "@test d"),
                "The R score is multiplied by getLV's sign multiplier (exactly +1 or -1), as the test does, because a rank-one loading is identified only up to sign."))
            push!(cases, mkcase("P1-JULIA-TEMPORAL-LATENT-LOADINGS-AT-R-COORDINATES", "temporal loadings at R's coordinates, per fit (concatenated)",
                "$fitf [fits.report_Lambda_temporal]", "GLLVModels fit.loadings at R par, as compared at $(cite(TFR, "vec(at_r.loadings)"))",
                lr, lj, test_tolerance(TFR, "vec(at_r.loadings)"), "Compared at R's own coordinates, so no sign alignment is needed."))
        end
        push!(out, "temporal/$(split(sid, '/')[2]).json" => Receipt([sid], "itchyshin/GLLVModels.jl#563",
            [fitf, "$TDIR/cross_objective.toml"], [TFR, "$TDIR/fixture_helpers.jl"], not_pair, cases))
    end

    # -- extract_temporal (all 21 fits) ---------------------------------------------------------
    tv_r = Float64[]; tv_j = Float64[]; va_r = Float64[]; va_j = Float64[]; ld_r = Float64[]; ld_j = Float64[]
    for f in fits
        e = extract_temporal(f.at_r)
        push!(tv_r, f.c["time_value"]); push!(tv_j, e.time.value)
        append!(va_r, temporal_p1_vec(f.c["variance"])); append!(va_j, e.variance.value)
        if haskey(f.c, "loadings")
            append!(ld_r, temporal_p1_vec(f.c["loadings"])); append!(ld_j, vec(e.loadings))
        end
    end
    xid = "Fits: all $(length(fits)) in fits.toml."
    xcases = [
        mkcase("P1-JULIA-EXTRACT-TEMPORAL-TIME-VALUE", "extract_temporal(fit).time.value at R's coordinates, per fit",
            "$fitf [fits.time_value]", "GLLVModels.extract_temporal(fit).time.value, as at $(cite(TFR, "abs(e.time.value - c[\"time_value\"])"))",
            tv_r, tv_j, test_tolerance(TFR, "abs(e.time.value - c[\"time_value\"])"), xid),
        mkcase("P1-JULIA-EXTRACT-TEMPORAL-VARIANCE", "extract_temporal(fit).variance.value at R's coordinates (all fits, concatenated)",
            "$fitf [fits.variance]", "GLLVModels.extract_temporal(fit).variance.value, as at $(cite(TFR, "e.variance.value ≈"))",
            va_r, va_j, test_tolerance(TFR, "e.variance.value ≈"), xid),
        mkcase("P1-JULIA-EXTRACT-TEMPORAL-LOADINGS", "extract_temporal(fit).loadings at R's coordinates (fits that have loadings, concatenated)",
            "$fitf [fits.loadings]", "GLLVModels.extract_temporal(fit).loadings, as at $(cite(TFR, "vec(e.loadings) ≈"))",
            ld_r, ld_j, test_tolerance(TFR, "vec(e.loadings) ≈"),
            "Fits with loadings: dep and latent modes. " * xid),
    ]
    push!(out, "temporal/extract_temporal.json" => Receipt(["temporal/extract_temporal"], "itchyshin/GLLVModels.jl#563",
        [fitf], [TFR, "$TDIR/fixture_helpers.jl"], not_pair, xcases))

    # -- forecast_temporal -----------------------------------------------------------------------
    R = temporal_p1_load("forecast.toml")
    R["gllvmTMB_commit"] == P1_SHA || fail("forecast.toml not pinned")
    byid = Dict(f.c["id"] => f for f in fits)
    fa = byid[R["ar1"]["fit_id"]].at_r
    future = (series = String.(R["ar1"]["future_series"]), occasion = temporal_p1_vec(R["ar1"]["future_occasion"]),
        trait = String.(R["ar1"]["future_trait"]))
    fc = forecast_temporal(fa, future; se_fit = true)
    cfit = byid[R["ar1"]["fit_id"]].c
    neg = temporal_p1_fit(F, cfit; start = temporal_p1_vec(R["ar1"]["negative_par"]), iterations = 0)
    fn = forecast_temporal(neg, future; se_fit = true)
    fo = byid[R["ou"]["fit_id"]].at_r
    future_ou = (series = String.(R["ou"]["future_series"]), elapsed = temporal_p1_vec(R["ou"]["future_elapsed"]),
        trait = String.(R["ou"]["future_trait"]))
    fco = forecast_temporal(fo, future_ou; se_fit = true)
    ffx = "$TDIR/forecast.toml"
    function fcase(id, q, key, field, fcobj, callnote, frag)
        return mkcase(id, q, "$ffx [$key.$field]", "GLLVModels.forecast_temporal(fit, future; se_fit = true).$(field == "est" || endswith(field, "_est") ? "est" : "se_fit"), as at $(cite(TH, frag))",
            temporal_p1_vec(R[key][field]), getproperty(fcobj, (field == "est" || endswith(field, "_est")) ? :est : :se_fit),
            test_tolerance(TH, frag), callnote)
    end
    fcases = [
        fcase("P1-JULIA-FORECAST-TEMPORAL-AR1-EST", "forecast est, AR1 temporal_indep fit at R's coordinates (12 future cells)", "ar1", "est", fc,
            "fit $(R["ar1"]["fit_id"]) at R par (program-forecast.R:47).", "fc.est .- temporal_p1_vec(R[\"ar1\"][\"est\"])"),
        fcase("P1-JULIA-FORECAST-TEMPORAL-AR1-SE-FIT", "forecast se.fit, AR1 temporal_indep fit at R's coordinates", "ar1", "se_fit", fc,
            "fit $(R["ar1"]["fit_id"]) at R par.", "fc.se_fit .- temporal_p1_vec(R[\"ar1\"][\"se_fit\"])"),
        fcase("P1-JULIA-FORECAST-TEMPORAL-AR1-NEGATIVE-EST", "forecast est, AR1 fit with negative persistence (phi = -0.6)", "ar1", "negative_est", fn,
            "negative_par from the fixture.", "fn.est .- temporal_p1_vec(R[\"ar1\"][\"negative_est\"])"),
        fcase("P1-JULIA-FORECAST-TEMPORAL-AR1-NEGATIVE-SE-FIT", "forecast se.fit, AR1 fit with negative persistence (phi = -0.6)", "ar1", "negative_se_fit", fn,
            "negative_par from the fixture.", "fn.se_fit .- temporal_p1_vec(R[\"ar1\"][\"negative_se_fit\"])"),
        fcase("P1-JULIA-FORECAST-TEMPORAL-OU-EST", "forecast est, OU temporal_indep fit at R's coordinates (6 future cells)", "ou", "est", fco,
            "fit $(R["ou"]["fit_id"]) at R par (program-forecast.R:104).", "fco.est .- temporal_p1_vec(R[\"ou\"][\"est\"])"),
        fcase("P1-JULIA-FORECAST-TEMPORAL-OU-SE-FIT", "forecast se.fit, OU temporal_indep fit at R's coordinates", "ou", "se_fit", fco,
            "fit $(R["ou"]["fit_id"]) at R par.", "fco.se_fit .- temporal_p1_vec(R[\"ou\"][\"se_fit\"])"),
    ]
    push!(out, "temporal/forecast_temporal.json" => Receipt(["temporal/forecast_temporal"], "itchyshin/GLLVModels.jl#543",
        [ffx, fitf], [TH, "$TDIR/fixture_helpers.jl"], not_pair, fcases))

    # -- compare_temporal -------------------------------------------------------------------------
    Cc = temporal_p1_load("compare.toml")
    Cc["gllvmTMB_commit"] == P1_SHA || fail("compare.toml not pinned")
    ll_r = Float64[]; ll_j = Float64[]; aic_r = Float64[]; aic_j = Float64[]; used = String[]
    for key in ("selection", "all_modes", "replicated")
        r = Cc[key]
        fs = [byid[id].at_r for id in r["fit_ids"]]
        res = compare_temporal(; (Symbol(m) => f for (m, f) in zip(r["model"], fs))...)
        append!(ll_r, temporal_p1_vec(r["logLik"])); append!(ll_j, res.logLik)
        append!(aic_r, temporal_p1_vec(r["AIC"])); append!(aic_j, res.AIC)
        push!(used, "$key ($(join(r["model"], ", ")))")
        res.df == Int.(r["df"]) || fail("compare_temporal df differs from R in $key")
    end
    cfx = "$TDIR/compare.toml"
    ccases = [
        mkcase("P1-JULIA-COMPARE-TEMPORAL-LOGLIK", "compare_temporal(...).logLik over the three fixture candidate sets (concatenated)",
            "$cfx [selection|all_modes|replicated].logLik", "GLLVModels.compare_temporal(; model = fit, ...).logLik on fits at R's coordinates, as at $(cite(TH, "out.logLik .- temporal_p1_vec(r[\"logLik\"])"))",
            ll_r, ll_j, test_tolerance(TH, "out.logLik .- temporal_p1_vec(r[\"logLik\"])"),
            "Sets: " * join(used, "; ") * ". df is asserted equal (exact) by the test and by this script, but has no tolerance, so it is not a case."),
        mkcase("P1-JULIA-COMPARE-TEMPORAL-AIC", "compare_temporal(...).AIC over the three fixture candidate sets (concatenated)",
            "$cfx [selection|all_modes|replicated].AIC", "GLLVModels.compare_temporal(...).AIC, as at $(cite(TH, "out.AIC .- temporal_p1_vec(r[\"AIC\"])"))",
            aic_r, aic_j, test_tolerance(TH, "out.AIC .- temporal_p1_vec(r[\"AIC\"])"),
            "Sets: " * join(used, "; ") * "."),
    ]
    push!(out, "temporal/compare_temporal.json" => Receipt(["temporal/compare_temporal"], "itchyshin/GLLVModels.jl#543",
        [cfx, fitf], [TH, "$TDIR/fixture_helpers.jl"], not_pair, ccases))

    # -- profile_temporal -------------------------------------------------------------------------
    P = temporal_p1_load("profile.toml")
    P["gllvmTMB_commit"] == P1_SHA || fail("profile.toml not pinned")
    est_r = Float64[]; est_j = Float64[]; th_r = Float64[]; th_j = Float64[]; tv_r2 = Float64[]; tv_j2 = Float64[]
    bd_r = Float64[]; bd_j = Float64[]; bd_ids = String[]; bd_skipped = String[]
    for key in ("ar1", "ar1_constrained", "ou", "ar1_default")
        r = P[key]
        c = byid[r["fit_id"]].c
        f = byid[r["fit_id"]].at_r
        idx = GMJ.TemporalLayout(size(f.X, 2), f.spec).time
        pr = isempty(r["parm_range_offset"]) ? (-Inf, Inf) : Tuple(f.parameters[idx] .+ temporal_p1_vec(r["parm_range_offset"]))
        o = profile_temporal(f; ystep = r["ystep"], ytol = r["ytol"], parm_range = pr)
        push!(est_r, r["estimate"]); push!(est_j, o.estimate)
        for side in (:lower, :upper)
            rv = temporal_p1_num(r[String(side)]); jv = getproperty(o, side)
            ismissing(rv) == ismissing(jv) || fail("profile $key $side: missing pattern differs from R")
            if !ismissing(rv) && !ismissing(jv) && isfinite(rv)
                isfinite(jv) || fail("profile $key $side: Julia bound infinite, R finite")
                push!(bd_r, theta_scale(c["structure"], rv)); push!(bd_j, theta_scale(c["structure"], jv))
                push!(bd_ids, "$key.$side")
            else
                push!(bd_skipped, "$key.$side (R: $(r[String(side)]); Julia agrees on missing/infinite)")
            end
        end
        tr = GMJ._temporal_tmbprofile(f; ystep = r["ystep"], ytol = r["ytol"], parm_range = pr)
        rt = temporal_p1_vec(r["trace_theta"])
        length(tr.theta) == length(rt) || fail("profile $key trace length differs")
        append!(th_r, rt); append!(th_j, tr.theta)
        rvv = [temporal_p1_num(v) for v in r["trace_value"]]
        all(ismissing.(rvv) .== ismissing.(tr.value)) || fail("profile $key trace missing pattern differs")
        for (a, b) in zip(rvv, tr.value)
            (ismissing(a) || ismissing(b)) && continue
            push!(tv_r2, a); push!(tv_j2, b)
        end
    end
    pfx = "$TDIR/profile.toml"
    pcases = [
        mkcase("P1-JULIA-PROFILE-TEMPORAL-ESTIMATE", "profile_temporal(fit).estimate over the four fixture profiles",
            "$pfx [ar1|ar1_constrained|ou|ar1_default].estimate", "GLLVModels.profile_temporal(fit; ystep, ytol, parm_range).estimate on the fit at R's coordinates, as at $(cite(TH, "abs(out.estimate - r[\"estimate\"])"))",
            est_r, est_j, test_tolerance(TH, "abs(out.estimate - r[\"estimate\"])"), "Profiles: ar1, ar1_constrained, ou, ar1_default."),
        mkcase("P1-JULIA-PROFILE-TEMPORAL-TRACE-THETA", "profile walk: visited theta displacements (concatenated over the four profiles)",
            "$pfx [*].trace_theta", "GLLVModels._temporal_tmbprofile(fit; ...).theta, as at $(cite(TH, "tr.theta .- rt"))",
            th_r, th_j, test_tolerance(TH, "tr.theta .- rt"), "The walk must visit the same displacements as TMB::tmbprofile."),
        mkcase("P1-JULIA-PROFILE-TEMPORAL-TRACE-VALUE", "profile walk: profiled objective values (concatenated; positions where both sides are non-missing)",
            "$pfx [*].trace_value", "GLLVModels._temporal_tmbprofile(fit; ...).value, as at $(cite(TH, "d = maximum(abs, skipmissing(rv .- tr.value))"))",
            tv_r2, tv_j2, test_tolerance(TH, "@test d <= 1e-5"; after = "@test d"),
            "The test asserts the missing pattern is identical (this script asserts it too) and compares the rest."),
    ]
    if !isempty(bd_r)
        push!(pcases, mkcase("P1-JULIA-PROFILE-TEMPORAL-FINITE-BOUNDS", "profile_temporal lower/upper bounds that are finite in R, on the theta scale",
            "$pfx [*].lower/upper, mapped to theta by atanh(v / (1 - 1e-6)) (AR1) or log(v) (OU), as the test does",
            "GLLVModels.profile_temporal(...).lower/upper mapped the same way, as at $(cite(TH, "theta_scale(c[\"structure\"], jv) - theta_scale(c[\"structure\"], rv)"))",
            bd_r, bd_j, test_tolerance(TH, "theta_scale(c[\"structure\"], jv) - theta_scale(c[\"structure\"], rv)"),
            "Bounds compared: " * join(bd_ids, ", ") * ". Not numeric cases (the test asserts agreement of the missing/infinite pattern, not a tolerance): " * join(bd_skipped, "; ") * "."))
    end
    push!(out, "temporal/profile_temporal.json" => Receipt(["temporal/profile_temporal"], "itchyshin/GLLVModels.jl#543",
        [pfx, fitf], [TH, "$TDIR/fixture_helpers.jl"], not_pair, pcases))

    # -- bootstrap_temporal ------------------------------------------------------------------------
    B = temporal_p1_load("bootstrap.toml")
    B["gllvmTMB_commit"] == P1_SHA || fail("bootstrap.toml not pinned")
    cb = only(filter(c -> c["id"] == B["fit_id"], F["fits"]))
    fi = temporal_p1_fit(F, cb)
    jb = bootstrap_temporal(fi; n_boot = 200, seed = 260931)                          # TH: jb = bootstrap_temporal(fi; n_boot=200, seed=260931)
    ok = isempty.(jb.error)
    (sum(ok) >= 190 && B["n_converged"] >= 190) || fail("fewer than 190 converged bootstrap replicates")
    te = Float64.(jb.time_estimate[ok])
    rmean = B["time_estimate_mean"]; rsd = B["time_estimate_sd"]
    se = sqrt(rsd^2 / B["n_converged"] + std(te)^2 / length(te))
    bfx = "$TDIR/bootstrap.toml"
    bcite_mean = cite(TH, "abs(mean(te) - rmean) <= 4 * se")
    bcite_sd = cite(TH, "0.75 <= std(te) / rsd <= 1 / 0.75")
    bnote = "bootstrap_temporal is stochastic, so the test's acceptance is statistical, and this case records exactly that band. n_boot = 200, seed = 260931, fit $(B["fit_id"]); $(sum(ok)) of 200 Julia replicates converged (R: $(B["n_converged"]))."
    bcases = [
        mkcase("P1-JULIA-BOOTSTRAP-TEMPORAL-TIME-ESTIMATE-MEAN", "mean of the bootstrap time_estimate distribution, Julia vs R (n_boot = 200)",
            "$bfx [time_estimate_mean]", "mean of GLLVModels.bootstrap_temporal(fit; n_boot = 200, seed = 260931).time_estimate over converged replicates, as at $bcite_mean",
            rmean, mean(te), (4 * se, bcite_mean, "@test abs(mean(te) - rmean) <= 4 * se"),
            bnote * " Tolerance is the test's 4 standard errors of the difference of means, se = sqrt(sd_R^2 / n_R + sd_J^2 / n_J), evaluated from this run ($(4 * se))."),
        mkcase("P1-JULIA-BOOTSTRAP-TEMPORAL-TIME-ESTIMATE-LOG-SD", "log sd of the bootstrap time_estimate distribution, Julia vs R (n_boot = 200)",
            "$bfx [time_estimate_sd] (log scale)", "log(sd of GLLVModels.bootstrap_temporal(...).time_estimate over converged replicates), as at $bcite_sd",
            log(rsd), log(std(te)), (log(1 / 0.75), bcite_sd, "@test 0.75 <= std(te) / rsd <= 1 / 0.75"),
            bnote * " The test asserts the ratio sd_J / sd_R lies in [0.75, 1/0.75]; on the log scale that is |log sd_J - log sd_R| <= log(1/0.75) = $(log(1 / 0.75)), the same band."),
    ]
    push!(out, "temporal/bootstrap_temporal.json" => Receipt(["temporal/bootstrap_temporal"], "itchyshin/GLLVModels.jl#543",
        [bfx, fitf], [TH, "$TDIR/fixture_helpers.jl"], not_pair, bcases))
    return out
end

# =============================================================================================
# 5. aghq/*   test/test_aghq_p1_twin.jl
#    Fourteen policy rows twinned against R at P1: Poisson / NB2 / ordinal-probit / Tweedie / delta-gamma / binomial / Gaussian fits with the same
#    aghq request R was given. Per row: the integration used (used flag and node count, vector
#    case, integers compared with the test's <= 0.5), logLik at each side's optimum, intercepts,
#    loadings (sign-aligned to R's, a fixed +-1 on the single axis) and, for Gaussian, the residual
#    SD. R values are copied from test/fixtures/aghq_p1/aghq_p1.toml; Julia values come from
#    aghq_p1_fit in test/fixtures/aghq_p1/aghq_p1_helpers.jl, the function the test calls.
#    AGHQ-POLICY-TRAITS20 is the same observation as AGHQ-POLICY-AUTO-ENFORCE-CUTOFF.
# =============================================================================================
function receipts_aghq()
    fxp = "test/fixtures/aghq_p1/aghq_p1.toml"
    hp = "test/fixtures/aghq_p1/aghq_p1_helpers.jl"
    tp = "test/test_aghq_p1_twin.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["gllvmtmb_commit"] == P1_SHA || fail("aghq fixture is not pinned at P1")
    rows = ["AGHQ-AUTO-K-POISSON", "AGHQ-AUTO-K-NB2", "AGHQ-AUTO-K-ORDINAL", "AGHQ-AUTO-K-TWEEDIE", "AGHQ-AUTO-K-DELTA", "AGHQ-AUTO-K-BINOMIAL", "AGHQ-AUTO-K-GAUSSIAN", "AGHQ-DEFAULT-OFF",
        "AGHQ-POLICY-OFF", "AGHQ-POLICY-EXPLICIT", "AGHQ-POLICY-EXPLICIT-BYPASS-CUTOFF",
        "AGHQ-POLICY-AUTO-ENFORCE-CUTOFF", "AGHQ-POLICY-TRAITS19", "AGHQ-POLICY-TRAITS20"]
    for (_, ds) in fx["dataset"]
        sha_file("test/fixtures/aghq_p1/" * ds["file"]) == ds["sha256"] || fail("aghq data csv $(ds["file"]) drifted")
    end
    fits = Dict{String,Any}()
    out = Pair{String,Receipt}[]
    callc = cite(tp, "aghq_p1_fit(fx, id)"; nth = 2)
    for row in rows
        id = aghq_p1_fit_id(row)
        r = fx["case"][id]["r"]
        r["converged"] === true || fail("$row: R did not converge; not a valid twin")
        j = get!(fits, id) do
            aghq_p1_fit(fx, id)
        end
        j.converged || fail("$row: Julia fit did not converge")
        j.used && j.reason !== :converged && fail("$row: Julia AGHQ reason is $(j.reason), not :converged")
        c = fx["case"][id]; ds = fx["dataset"][c["dataset"]]
        short = replace(row, "AGHQ-" => "")
        pre = "P1-JULIA-AGHQ-$short"
        desc = "$(ds["family"]) p = $(ds["p"]), n = $(ds["n_unit"]) (seed $(ds["seed"])), aghq request \"$(c["aghq_request"])\""
        rsrc(f) = "$fxp [case.$id.r].$f"
        jsrc(f) = "GLLVModels $(ds["family"] == "gaussian" ? "fit_gllvm(Normal())" : ds["family"] == "nb2" ? "fit_gllvm(NegativeBinomial(), disp_group = :species)" : ds["family"] == "ordinal" ? "fit_gllvm(Ordinal(), link = ProbitLink())" : ds["family"] == "tweedie" ? "fit_gllvm(TweedieED(1.0, 1.5), disp_group = :species, power_group = :species)" : ds["family"] == "delta_gamma" ? "fit_gllvm(DeltaGamma(), predictor = :shared, disp_group = :species)" : "fit_" * ds["family"] * "_gllvm")(...; K = 1, aghq = $(c["aghq_request"] == "default" ? "false" : repr(aghq_p1_request(c["aghq_request"])))) via $hp aghq_p1_fit, as called at $callc: $f"
        note = "Data read from test/fixtures/aghq_p1/$(ds["file"]) (sha256 checked); $desc. R and Julia both converged (R: " *
               (r["used"] ? "aghq\$converged" : "optimiser code 0") * "; Julia: fit.converged" * (j.used ? " and reason :converged" : "") * ")."
        cases = Any[
            mkcase("$pre-DECISION", "integration actually used: [AGHQ used (1/0), node count k] (Laplace = [0, 0])",
                rsrc("used, k"), jsrc("integration.actual, integration.k"),
                Float64[r["used"], r["k"]], Float64[j.used, j.nodes],
                test_tolerance(tp, "maximum(abs.(dec_j .- dec_r))"),
                note * " Integers, so the test's <= 0.5 is exact equality."),
            mkcase("$pre-LOGLIK", "logLik at each side's own optimum",
                rsrc("loglik"), jsrc("loglik"), r["loglik"], j.loglik,
                test_tolerance(tp, "isapprox(j.loglik, r[\"loglik\"]"), note),
            mkcase("$pre-BETA", "per-trait intercepts at each side's own optimum",
                rsrc("beta"), jsrc("beta"), Float64.(r["beta"]), j.beta,
                test_tolerance(tp, "isapprox(j.beta,"), note),
            mkcase("$pre-LAMBDA", "loadings at each side's own optimum, sign-aligned to R",
                rsrc("lambda"), jsrc("lambda (times +-1 so that sum(lambda_julia * lambda_R) >= 0)"), Float64.(r["lambda"]), j.lambda,
                test_tolerance(tp, "isapprox(j.lambda,"), note * " The single latent axis is identified up to sign; Julia's loadings are multiplied by the +-1 the test applies."),
        ]
        if ds["family"] == "gaussian"
            push!(cases, mkcase("$pre-SIGMA-EPS", "residual SD at each side's own optimum",
                rsrc("sigma_eps"), jsrc("pars.σ_eps"), r["sigma_eps"], j.sigma_eps,
                test_tolerance(tp, "isapprox(j.sigma_eps,"), note))
        end
        if ds["family"] == "nb2"
            push!(cases, mkcase("$pre-LOG-PHI", "per-trait NB2 dispersion log(phi) at each side's own optimum",
                rsrc("phi (log)"), jsrc("r_group (log)"), log.(Float64.(r["phi"])), log.(j.phi),
                test_tolerance(tp, "isapprox(log.(j.phi)"), note * " Compared on the log scale, the optimised scale; R's phi is Julia's r (Var = mu + mu^2/phi)."))
        end
        if ds["family"] == "delta_gamma"
            push!(cases, mkcase("$pre-LOG-PHI", "per-trait Gamma CV log(phi) at each side's own optimum",
                rsrc("phi (log)"), jsrc("1/sqrt(α) (log)"), log.(Float64.(r["phi"])), log.(j.phi),
                test_tolerance(tp, "isapprox(log.(j.phi)"), note * " Compared on the log scale, the optimised scale; R's phi is the Gamma coefficient of variation (shape = 1/phi^2), Julia's alpha is the shape, so Julia's phi is 1/sqrt(alpha). One latent drives both parts through one shared predictor on both sides."))
        end
        if ds["family"] == "tweedie"
            push!(cases, mkcase("$pre-LOG-PHI", "per-trait Tweedie dispersion log(phi) at each side's own optimum",
                rsrc("phi (log)"), jsrc("φ (log)"), log.(Float64.(r["phi"])), log.(j.phi),
                test_tolerance(tp, "isapprox(log.(j.phi)"), note * " Compared on the log scale, the optimised scale; same Var = phi mu^power on both sides."))
            push!(cases, mkcase("$pre-POWER", "per-trait Tweedie power (1, 2) at each side's own optimum",
                rsrc("power"), jsrc("power"), Float64.(r["power"]), j.power,
                test_tolerance(tp, "isapprox(j.power,"), note * " Compared on the natural scale both engines report (R: 1 + plogis(logit_p_tweedie); Julia: 1 + 1/(1+exp(-xi))); per-trait power and per-trait dispersion, gllvmTMB's tweedie()."))
        end
        if ds["family"] == "ordinal"
            push!(cases, mkcase("$pre-LOG-INCREMENTS", "free cutpoint log-increments (tau_1 = 0 fixed) at each side's own optimum",
                rsrc("log_incr"), jsrc("log diff of tau (per trait, trait by trait)"), Float64.(r["log_incr"]), j.log_incr,
                test_tolerance(tp, "isapprox(j.log_incr,"), note * " R's ordinal_log_increments and Julia's log cutpoint spacings, per trait in trait order; probit link, per-trait intercepts and cutpoints."))
        end
        push!(out, "aghq/$row.json" => Receipt(["aghq/$row"], "itchyshin/GLLVModels.jl#586",
            ["$fxp", "test/fixtures/aghq_p1/" * ds["file"], hp], [tp], NOT_A_FIXTURE_PAIR, cases))
    end
    return out
end

# 6. isdm: the integrated SDM through R's public door, R-at-P1 fits vs Julia
#    test/parity/isdm_cases.jl "iSDM P1 paired twins" (itchyshin/GLLVModels.jl#546)
#    Four paired fits (predict, ms3, srcform_pois, srcform_mixed). The R side is r_values_p1.toml;
#    the Julia side is the same table, marginal and fit the test builds.
# =============================================================================================
using Distributions: Poisson, Binomial
include(joinpath(ROOT, "test", "fixtures", "isdm", "isdm_fixture_io.jl"))

# Which isdm rows each paired fit is evidence for (decided in
# docs/dev-log/core070/true-parity-latest/audit-isdm-546-2026-10-01.md).
const ISDM_FIT_ROWS = Dict(
    "predict"       => ["isdm/ISDM-MIXED", "isdm/ISDM-SUPPORT", "isdm/ISDM-WITHIN-TRAIT-ADMIT"],
    "ms3"           => ["isdm/ISDM-SUPPORT", "isdm/ISDM-THREE", "isdm/ISDM-WITHIN-TRAIT-ADMIT"],
    "srcform_pois"  => ["isdm/ISDM-MASKED-COLUMNS", "isdm/ISDM-SUPPORT"],
    "srcform_mixed" => ["isdm/ISDM-MASKED-COLUMNS", "isdm/ISDM-MIXED", "isdm/ISDM-SUPPORT", "isdm/ISDM-WITHIN-TRAIT-ADMIT"],
)

function receipts_isdm()
    tp = "test/parity/isdm_cases.jl"
    rvp = "test/fixtures/isdm/r_values_p1.toml"
    jep = "test/fixtures/isdm/julia_estimates_p1.toml"
    RV = isdm_r_values()
    JE = TOML.parsefile(joinpath(ROOT, jep))
    RV["gllvmtmb_sha"] == P1_SHA || fail("isdm r_values_p1.toml is not pinned at P1")
    out = Pair{String,Receipt}[]
    function by_name_(names_from, vals, names_to)
        Set(names_from) == Set(names_to) || fail("isdm coefficient names do not pair: $names_from vs $names_to")
        return [vals[findfirst(==(n), names_from)] for n in names_to]
    end
    f_ll = "isdm_marginal_loglik_laplace(tab, RΛ, rb) - r[\"loglik\"]"
    f_x = "abs(RV[\"xobj\"][name] - jll)"
    f_b = "maximum(abs.(ft.b_fix .- rb))"
    f_e = "maximum(abs.(ft.eta .- Float64.(r[\"eta\"])))"
    tol_ll = test_tolerance(tp, f_ll)
    tol_x = test_tolerance(tp, f_x)
    tol_b = test_tolerance(tp, f_b)
    tol_e = test_tolerance(tp, f_e)
    for name in ISDM_CASES
        r = RV["cases"][name]; j = JE["julia"][name]
        c = isdm_case(name)
        csvp = "test/fixtures/isdm/" * c.csv
        bytes2hex(open(sha256, joinpath(ROOT, csvp))) == r["fixture_sha256"] || fail("isdm $name fixture sha256 drifted")
        dat = read_isdm_csv(c.csv)
        length(dat.value) == r["fixture_rows"] || fail("isdm $name fixture row count")
        tab = isdm_table(c.formula, dat; family = c.family)
        rb = by_name_(r["b_fix_names"], Float64.(r["b_fix"]), tab.X_names)
        K = Int(r["K"])
        RΛ = K == 0 ? zeros(2, 0) : reshape(Float64.(r["Lambda_B_colmajor"]), :, K)
        ll_at_r = isdm_marginal_loglik_laplace(tab, RΛ, rb)
        jb = by_name_(j["b_fix_names"], Float64.(j["b_fix"]), tab.X_names)
        JΛ = j["K"] == 0 ? zeros(2, 0) : GMJ.unpack_lambda(Float64.(j["theta_rr_B"]), 2, Int(j["K"]))
        jll = isdm_marginal_loglik_laplace(tab, JΛ, jb)
        ft = fit_isdm_gllvm(tab)
        (ft.converged && all(ft.cell_converged)) || fail("isdm $name fresh fit did not converge")
        r["convergence"] == 0 || fail("isdm $name: R did not converge")
        pre = "P1-JULIA-ISDM-" * uppercase(replace(name, "_" => "-"))
        cases = [
            mkcase("$pre-LOGLIK-AT-R-OPTIMUM", "Julia Laplace marginal at R's fitted (b_fix, Lambda) vs R's logLik",
                "$rvp [cases.$name].loglik (R nlminb optimum through gllvmTMB(family = isdm_sources(...)) at P1)",
                "GLLVModels.isdm_marginal_loglik_laplace(isdm_table(formula, data; family), Lambda_R, b_fix_R), as at $(cite(tp, f_ll))",
                r["loglik"], ll_at_r, tol_ll,
                "Same data (sha256 checked), same parameter vector, same Laplace objective: the Julia design and offset must reproduce R's. Coefficients are paired by name, as in the test."),
            mkcase("$pre-CROSS-OBJECTIVE-AT-JULIA-OPTIMUM", "R's objective at Julia's optimum vs Julia's logLik there",
                "$rvp [xobj].$name (R TMB objective evaluated at the recorded Julia estimate, sign flipped to logLik)",
                "GLLVModels.isdm_marginal_loglik_laplace(...) at the recorded Julia estimate in $jep [julia.$name], as at $(cite(tp, f_x))",
                RV["xobj"][name], jll, tol_x,
                "The recorded Julia estimate is the fit the test reproduces to 1e-8 on a fresh run; this receipt evaluates the objective at that recorded point, as the test does at the cited line."),
            mkcase("$pre-B-FIX", "fresh Julia fit b_fix vs R b_fix (paired by coefficient name; maximum absolute difference)",
                "$rvp [cases.$name].b_fix",
                "GLLVModels.fit_isdm_gllvm(tab).b_fix, as at $(cite(tp, f_b))",
                rb, ft.b_fix, tol_b,
                "R's nlminb stops with max abs gradient 3.9e-4 to 9.0e-4 on the K = 0 fits, so this is a loose absolute bound; the test records the same finding. The test also asserts Julia's logLik is not below R's (one-sided), which is not a case here."),
            mkcase("$pre-ETA", "fresh Julia fit linear predictor vs R's (maximum absolute difference over all rows)",
                "$rvp [cases.$name].eta",
                "GLLVModels.fit_isdm_gllvm(tab).eta, as at $(cite(tp, f_e))",
                Float64.(r["eta"]), ft.eta, tol_e,
                "Row order is the fixture's; R's eta and Julia's eta are both per long-table row."),
        ]
        push!(out, "isdm/$name.json" => Receipt(copy(ISDM_FIT_ROWS[name]), "itchyshin/GLLVModels.jl#546",
            [rvp, jep, csvp], [tp, "test/fixtures/isdm/isdm_fixture_io.jl"],
            NOT_A_FIXTURE_PAIR * " The check is a fit-level logLik and estimate comparison on one fitted case; it does not restate any admission predicate.",
            cases))
    end
    return out
end

# =============================================================================================
# 6b. isdm admission twins: five more R-at-P1 vs Julia fitted cases, each built to reach one positive
#     admission that the four fits above do not (audit-isdm-546-2026-10-01.md).
#     test/parity/isdm_admission_twins.jl (itchyshin/GLLVModels.jl#661)
# =============================================================================================
include(joinpath(ROOT, "test", "fixtures", "isdm", "isdm_admission_cases.jl"))

const ISDM_ADM_ROWS = Dict(
    "adm_aliased"    => "isdm/ISDM-ALIASED",
    "adm_align"      => "isdm/ISDM-ALIGN",
    "adm_nooffset"   => "isdm/ISDM-NO-OFFSET",
    "adm_zeroord"    => "isdm/ISDM-ZERO-ORDINARY",
    "adm_unbalanced" => "isdm/ISDM-UNBALANCED",
)
const ISDM_MASKED_ROW = "isdm/ISDM-MASKED-ARM"
const ISDM_ADM_PATH = Dict(
    "adm_aliased"    => "an aliased candidate column (an exact multiple of access) is dropped by the QR rank rule, in both engines, leaving the same coefficient list",
    "adm_align"      => "the family list is declared survey-first while the data's source levels put gbif first, so both engines re-order the list by name",
    "adm_nooffset"   => "the formula has no offset() term, so the offset evaluates to zeros on a mixed (cloglog-admitted) declaration",
    "adm_zeroord"    => "an all-count declaration (the mixed contract is not admitted, so the cloglog offset exception is off) with an identically zero offset column",
    "adm_unbalanced" => "a latent-variable fit on a cell x trait x source grid with 12 rows removed (every trait still carries every source)",
)

function receipts_isdm_admission()
    tp = "test/parity/isdm_admission_twins.jl"
    rvp = "test/fixtures/isdm/r_values_admission_p1.toml"
    jep = "test/fixtures/isdm/julia_estimates_admission_p1.toml"
    RV = isdm_admission_r_values()
    JE = TOML.parsefile(joinpath(ROOT, jep))
    RV["gllvmtmb_sha"] == P1_SHA || fail("isdm admission r_values_admission_p1.toml is not pinned at P1")
    out = Pair{String,Receipt}[]
    function by_name_(names_from, vals, names_to)
        Set(names_from) == Set(names_to) || fail("isdm admission coefficient names do not pair: $names_from vs $names_to")
        return [vals[findfirst(==(n), names_from)] for n in names_to]
    end
    f_ll = "isdm_marginal_loglik_laplace(tab, RΛ, rb) - r[\"loglik\"]"
    f_x = "abs(ARV[\"xobj\"][name] - jll)"
    f_b = "maximum(abs.(ft.b_fix .- rb))"
    f_e = "maximum(abs.(ft.eta .- Float64.(r[\"eta\"])))"
    tol_ll = test_tolerance(tp, f_ll)
    tol_x = test_tolerance(tp, f_x)
    tol_b = test_tolerance(tp, f_b)
    tol_e = test_tolerance(tp, f_e)
    for name in ISDM_ADMISSION_CASES
        r = RV["cases"][name]; j = JE["julia"][name]
        c = isdm_admission_case(name)
        csvp = "test/fixtures/isdm/" * c.csv
        bytes2hex(open(sha256, joinpath(ROOT, csvp))) == r["fixture_sha256"] || fail("isdm admission $name fixture sha256 drifted")
        dat = read_isdm_csv(c.csv)
        length(dat.value) == r["fixture_rows"] || fail("isdm admission $name fixture row count")
        tab = isdm_table(c.formula, dat; family = c.family)
        rb = by_name_(r["b_fix_names"], Float64.(r["b_fix"]), tab.X_names)
        K = Int(r["K"])
        RΛ = K == 0 ? zeros(2, 0) : reshape(Float64.(r["Lambda_B_colmajor"]), :, K)
        ll_at_r = isdm_marginal_loglik_laplace(tab, RΛ, rb)
        jb = by_name_(j["b_fix_names"], Float64.(j["b_fix"]), tab.X_names)
        JΛ = j["K"] == 0 ? zeros(2, 0) : GMJ.unpack_lambda(Float64.(j["theta_rr_B"]), 2, Int(j["K"]))
        jll = isdm_marginal_loglik_laplace(tab, JΛ, jb)
        ft = fit_isdm_gllvm(tab)
        (ft.converged && all(ft.cell_converged)) || fail("isdm admission $name fresh fit did not converge")
        r["convergence"] == 0 || fail("isdm admission $name: R did not converge")
        pre = "P1-JULIA-ISDM-" * uppercase(replace(name, "_" => "-"))
        path = ISDM_ADM_PATH[name]
        cases = [
            mkcase("$pre-LOGLIK-AT-R-OPTIMUM", "Julia Laplace marginal at R's fitted (b_fix, Lambda) vs R's logLik",
                "$rvp [cases.$name].loglik (R nlminb optimum through gllvmTMB(family = isdm_sources(...)) at P1)",
                "GLLVModels.isdm_marginal_loglik_laplace(isdm_table(formula, data; family), Lambda_R, b_fix_R), as at $(cite(tp, f_ll))",
                r["loglik"], ll_at_r, tol_ll,
                "Path exercised: $path. Same data (sha256 checked), same parameter vector, same Laplace objective. Coefficients are paired by name, as in the test."),
            mkcase("$pre-CROSS-OBJECTIVE-AT-JULIA-OPTIMUM", "R's objective at Julia's optimum vs Julia's logLik there",
                "$rvp [xobj].$name (R TMB objective evaluated at the recorded Julia estimate, sign flipped to logLik)",
                "GLLVModels.isdm_marginal_loglik_laplace(...) at the recorded Julia estimate in $jep [julia.$name], as at $(cite(tp, f_x))",
                RV["xobj"][name], jll, tol_x,
                "Path exercised: $path. The recorded Julia estimate is the fit the test reproduces to 1e-8 on a fresh run."),
            mkcase("$pre-B-FIX", "fresh Julia fit b_fix vs R b_fix (paired by coefficient name; maximum absolute difference)",
                "$rvp [cases.$name].b_fix",
                "GLLVModels.fit_isdm_gllvm(tab).b_fix, as at $(cite(tp, f_b))",
                rb, ft.b_fix, tol_b,
                "Path exercised: $path. The test also asserts Julia's logLik is not below R's (one-sided), which is not a case here."),
            mkcase("$pre-ETA", "fresh Julia fit linear predictor vs R's (maximum absolute difference over all rows)",
                "$rvp [cases.$name].eta",
                "GLLVModels.fit_isdm_gllvm(tab).eta, as at $(cite(tp, f_e))",
                Float64.(r["eta"]), ft.eta, tol_e,
                "Path exercised: $path. Row order is the fixture's."),
        ]
        push!(out, "isdm/$name.json" => Receipt([ISDM_ADM_ROWS[name]], "itchyshin/GLLVModels.jl#661",
            [rvp, jep, csvp], [tp, "test/fixtures/isdm/isdm_fixture_io.jl", "test/fixtures/isdm/isdm_admission_cases.jl"],
            NOT_A_FIXTURE_PAIR * " The check is a fit-level logLik and estimate comparison on one fitted case built to reach this row's path; it does not restate any admission predicate.",
            cases))
    end
    # ISDM-MASKED-ARM twin: both engines drop rows whose response is NA and fit the rest.
    rp = RV["reproducers"]["adm_maskedna"]
    mcsv = "test/fixtures/isdm/" * rp["fixture"]
    bytes2hex(open(sha256, joinpath(ROOT, mcsv))) == rp["fixture_sha256"] || fail("isdm adm_maskedna fixture sha256 drifted")
    mdat = read_isdm_csv(rp["fixture"])
    count(ismissing, mdat.value) == rp["na_rows"] || fail("isdm adm_maskedna NA row count")
    mform = :(value ~ 0 + trait + trait & env + trait & src_gbif + offset(log_support))
    mfam = isdm_sources(gbif = Poisson(), survey = (Binomial(), CLogLogLink()))
    mtab = isdm_table(mform, mdat; family = mfam)   # warns: dropped 10 row(s)
    length(mtab.y) == rp["rows"] - rp["na_rows"] || fail("isdm adm_maskedna kept-row count")
    rbn = by_name_(rp["b_fix_names"], Float64.(rp["b_fix"]), mtab.X_names)
    ll_at_r = isdm_marginal_loglik_laplace(mtab, zeros(2, 0), rbn)
    mft = fit_isdm_gllvm(mtab)
    (mft.converged && all(mft.cell_converged)) || fail("isdm adm_maskedna fresh fit did not converge")
    rp["convergence"] == 0 || fail("isdm adm_maskedna: R did not converge")
    g_ll = "isdm_marginal_loglik_laplace(tab, zeros(2, 0), rbn) - rp[\"loglik\"]"
    g_fl = "abs(ft.loglik - rp[\"loglik\"])"
    g_b = "maximum(abs.(ft.b_fix .- rbn))"
    mpath = "10 of 160 rows have an NA response; both engines drop them before fitting (R: drop_missing_response_rows, response = \"drop\"; Julia: isdm_table warns and drops), leaving 150 rows"
    mpre = "P1-JULIA-ISDM-ADM-MASKEDNA"
    mcases = [
        mkcase("$mpre-LOGLIK-AT-R-OPTIMUM", "Julia Laplace marginal at R's fitted b_fix on the kept rows vs R's logLik",
            "$rvp [reproducers.adm_maskedna].loglik (R nlminb optimum through gllvmTMB(family = isdm_sources(...)) at P1, NA rows dropped by R)",
            "GLLVModels.isdm_marginal_loglik_laplace(isdm_table(formula, data_with_NA; family), zeros(2, 0), b_fix_R), as at $(cite(tp, g_ll))",
            rp["loglik"], ll_at_r, test_tolerance(tp, g_ll),
            "Path exercised: $mpath. Same file (sha256 checked), same parameter vector, same Laplace objective. Coefficients are paired by name."),
        mkcase("$mpre-FIT-LOGLIK", "fresh Julia fit logLik on the kept rows vs R's logLik",
            "$rvp [reproducers.adm_maskedna].loglik",
            "GLLVModels.fit_isdm_gllvm(isdm_table(formula, data_with_NA; family)).loglik, as at $(cite(tp, g_fl))",
            rp["loglik"], mft.loglik, test_tolerance(tp, g_fl),
            "Path exercised: $mpath. Each side's own optimum."),
        mkcase("$mpre-B-FIX", "fresh Julia fit b_fix vs R b_fix (paired by coefficient name; maximum absolute difference)",
            "$rvp [reproducers.adm_maskedna].b_fix",
            "GLLVModels.fit_isdm_gllvm(tab).b_fix, as at $(cite(tp, g_b))",
            rbn, mft.b_fix, test_tolerance(tp, g_b),
            "Path exercised: $mpath. R's linear predictor was not recorded for this fixture, so there is no ETA case; with no latent term eta = X b, which the test checks against X at R's b_fix."),
    ]
    push!(out, "isdm/adm_maskedna.json" => Receipt([ISDM_MASKED_ROW], "itchyshin/GLLVModels.jl#661",
        [rvp, mcsv], [tp, "test/fixtures/isdm/isdm_fixture_io.jl"],
        NOT_A_FIXTURE_PAIR * " The check is a fit-level logLik and estimate comparison on one fitted case built to reach this row's path; it does not restate any admission predicate.",
        mcases))
    return out
end

# =============================================================================================
# 7. select-lv/select_lv   test/test_select_lv_p1_twin.jl
#    Gaussian rank sweep K = 1:3 vs R's select_lv() at P1 (criterion = bic, d_max = 3). The test also
#    asserts npar and the selected rank EXACTLY (integers, no tolerance literal), so those have no
#    numeric case here. print.gllvmTMB_select_lv has nothing numeric to compare and is not bound.
# =============================================================================================
function _load_select_lv_csv(path::AbstractString, trait_names::Vector{String}, n_unit::Integer)
    Y = zeros(Float64, length(trait_names), n_unit)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            unit = parse(Int, strip(parts[1], '"'))
            t = findfirst(==(strip(parts[2], '"')), trait_names)
            t === nothing && error("unrecognised trait in $path")
            Y[t, unit] = parse(Float64, parts[3])
        end
    end
    return Y
end

function receipts_select_lv()
    fxp = "test/fixtures/select_lv_p1.toml"
    tp = "test/test_select_lv_p1_twin.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["gllvmtmb_commit"] == P1_SHA || fail("select_lv fixture is not pinned at P1")
    datap = "test/fixtures/" * fx["data_file"]
    bytes2hex(sha256(read(joinpath(ROOT, datap)))) == fx["data_sha256"] || fail("select_lv data csv drifted")
    r = fx["r_reference"]
    (all(r["converged"]) && all(r["pd_hessian"])) || fail("select_lv fixture has a non-converged or non-PD-Hessian rank; not a valid twin (fence)")
    Y = _load_select_lv_csv(joinpath(ROOT, datap), String.(fx["trait_names"]), Int(fx["n_unit"]))
    Ks = Int.(r["d"])
    sel = select_lv(Y; family = Normal(), Kmax = maximum(Ks), criterion = :bic)       # test line cited below
    sel.K == Ks || fail("select_lv accepted ranks $(sel.K), expected $Ks")
    sel.nparams == Int.(r["npar"]) || fail("select_lv npar differs from R")
    sel.best_k == Int(r["selected_d"]) || fail("select_lv selected rank differs from R")
    callc = cite(tp, "sel = select_lv(Y; family = Normal()")
    note = "Rank sweep 1:3 on the fixture data (sha256 checked); vector case, abs_diff is the maximum over ranks. The test asserts npar and the selected rank (2, the true rank) exactly; both match R here. R reports converged and a positive-definite Hessian at every rank, so the sweep stays clear of the signed Hessian fence."
    cases = [
        mkcase("P1-JULIA-SELECT-LV-LOGLIK", "per-rank logLik, K = 1:3 (each side's own optimum)",
            "$fxp [r_reference.loglik]", "GLLVModels.select_lv(Y; family = Normal(), Kmax = 3, criterion = :bic).loglik, as called at $callc",
            Float64.(r["loglik"]), sel.loglik, test_tolerance(tp, "isapprox(sel.loglik, Float64.(r[\"loglik\"])"), note),
        mkcase("P1-JULIA-SELECT-LV-AIC", "per-rank AIC, K = 1:3",
            "$fxp [r_reference.aic]", "select_lv(...).aic, as called at $callc",
            Float64.(r["aic"]), sel.aic, test_tolerance(tp, "isapprox(sel.aic,"), note),
        mkcase("P1-JULIA-SELECT-LV-BIC", "per-rank BIC, penalty log(p*n) = log(900) (R's default criterion), K = 1:3",
            "$fxp [r_reference.bic]", "select_lv(...).bic, as called at $callc",
            Float64.(r["bic"]), sel.bic, test_tolerance(tp, "isapprox(sel.bic,"), note),
    ]
    return ["select-lv/select_lv.json" => Receipt(["select-lv/select_lv"], "itchyshin/GLLVModels.jl#518",
        [fxp, datap], [tp], NOT_A_FIXTURE_PAIR, cases)]
end

# =============================================================================================
# 8. model-comparison: AIC / BIC / anova (case-map.json) and logLik (namespace map)
#    test/test_model_comparison.jl (itchyshin/GLLVModels.jl#556), gllvmTMB P1 anova twin fixture
# =============================================================================================
function receipts_model_comparison()
    fxp = "test/fixtures/gllvmtmb_anova_fixture.toml"
    tp = "test/test_model_comparison.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["meta"]["gllvmtmb_pin"] == P1_SHA || fail("anova fixture is not pinned at P1")
    per_model = fx["per_model"]
    length(per_model) == 3 || fail("anova fixture does not hold 3 models")
    npar_r = Int[m["npar"] for m in per_model]
    loglik_r = Float64[m["loglik"] for m in per_model]
    d_r = Union{Missing,Int}[m["d"] for m in per_model]
    aic_r = Float64[m["aic"] for m in per_model]
    bic_r = Float64[m["bic"] for m in per_model]
    p_r = Int(fx["meta"]["p_trait"])

    # End-to-end fits, exactly as the test's "end-to-end" testset builds them.
    Yw = fx["Y_wide"]
    p = length(Yw); n = length(Yw[1])
    Y = Matrix{Float64}(undef, p, n)
    for i in 1:p, j in 1:n
        Y[i, j] = Yw[i][j]
    end
    X = zeros(p, n, p)
    for t in 1:p, s in 1:n
        X[t, s, t] = 1.0
    end
    fits = [fit_gaussian_gllvm(Y; X = X, K = K) for K in 1:3]
    fitc = cite(tp, "fits = [fit_gaussian_gllvm(Y; X = X, K = K) for K in 1:3]")
    e2e_note = "Julia fits the fixture's own response matrix (Y_wide) with fit_gaussian_gllvm(Y; X = per-species intercepts, K), K = 1, 2, 3, as in the test; R's values are the recorded P1 gllvmTMB fits of the same data and nested ranks. The test uses this looser tolerance because the two optimisers are independent."
    ref_chibar = fx["anova_chibar"]; ref_chisq = fx["anova_chisq"]
    ORIGIN = "itchyshin/GLLVModels.jl#556"

    c_aic = mkcase("P1-JULIA-MODEL-COMPARISON-AIC-E2E", "aic(fit) at each side's own optimum, ranks d = 1, 2, 3",
        "$fxp [[per_model]].aic", "GLLVModels.aic(fit_gaussian_gllvm(Y; X, K)), as at $(cite(tp, "aic(fits[i]) ≈ aic_r[i]")); fits as at $fitc",
        aic_r, [Float64(aic(f)) for f in fits], test_tolerance(tp, "aic(fits[i]) ≈ aic_r[i]"),
        e2e_note * " The test's separate formula-twin assertion on R's own logLik and df (aic_formula) is not used: it evaluates an inline expression, not a package function.")
    c_bic = mkcase("P1-JULIA-MODEL-COMPARISON-BIC-E2E", "bic(fit, nobs(fit, Y)) at each side's own optimum, ranks d = 1, 2, 3",
        "$fxp [[per_model]].bic", "GLLVModels.bic(fit, nobs(fit, Y)), as at $(cite(tp, "bic(fits[i], nobs(fits[i], Y)) ≈ bic_r[i]")); fits as at $fitc",
        bic_r, [Float64(bic(f, nobs(f, Y))) for f in fits], test_tolerance(tp, "bic(fits[i], nobs(fits[i], Y)) ≈ bic_r[i]"),
        e2e_note * " The test also asserts nobs equality and dof equality exactly (integers, no tolerance), so those have no numeric case.")
    c_ll = mkcase("P1-JULIA-MODEL-COMPARISON-LOGLIK-E2E", "loglikelihood(fit) at each side's own optimum, ranks d = 1, 2, 3",
        "$fxp [[per_model]].loglik", "GLLVModels.loglikelihood(fit_gaussian_gllvm(Y; X, K)), as at $(cite(tp, "loglikelihood(fits[i]) ≈ loglik_r[i]")); fits as at $fitc",
        loglik_r, [Float64(loglikelihood(f)) for f in fits], test_tolerance(tp, "loglikelihood(fits[i]) ≈ loglik_r[i]"),
        e2e_note)

    # anova: definition twin (_anova_core on R's recorded npar / logLik / d) and end-to-end twin.
    tabc = GLLVModels._anova_core(npar_r, loglik_r, d_r, p_r; test = :chibar)
    cc = cite(tp, "GLLVModels._anova_core(npar_r, loglik_r, d_r, p_r; test = :chibar)")
    tabq = GLLVModels._anova_core(npar_r, loglik_r, d_r, p_r; test = :chisq)
    cq = cite(tp, "GLLVModels._anova_core(npar_r, loglik_r, d_r, p_r; test = :chisq)")
    core_note(kind) = "GLLVModels._anova_core is called on R's recorded npar, logLik and rank d (no Julia fitting), so this isolates the LRT / $kind arithmetic. Rows 2 and 3 are compared (row 1 has no test statistic). The test also asserts the df column and the test-label text exactly, so those have no numeric case."
    rl(ref, k) = [Float64(ref[k][i]) for i in 2:3]
    lrt6 = "tab.LRT[i] ≈ Float64(ref[\"LRT\"][i]) atol=1e-6"
    pv6 = "tab.pvalue[i] ≈ Float64(ref[\"pvalue\"][i]) atol=1e-6"
    cases = [
        mkcase("P1-JULIA-ANOVA-CHIBAR-LRT", "anova LRT statistic, test = :chibar (definition twin)",
            "$fxp [anova_chibar].LRT[2:3]", "GLLVModels._anova_core(...; test = :chibar).LRT[2:3], as called at $cc",
            rl(ref_chibar, "LRT"), [Float64(tabc.LRT[i]) for i in 2:3], test_tolerance(tp, lrt6; nth = 1, after = "≈ Float64"), core_note("chi-bar-square p-value")),
        mkcase("P1-JULIA-ANOVA-CHIBAR-PVALUE", "anova p-value, test = :chibar (definition twin)",
            "$fxp [anova_chibar].pvalue[2:3]", "GLLVModels._anova_core(...; test = :chibar).pvalue[2:3], as called at $cc",
            rl(ref_chibar, "pvalue"), [Float64(tabc.pvalue[i]) for i in 2:3], test_tolerance(tp, pv6; nth = 1, after = "≈ Float64"), core_note("chi-bar-square p-value")),
        mkcase("P1-JULIA-ANOVA-CHISQ-LRT", "anova LRT statistic, test = :chisq (definition twin)",
            "$fxp [anova_chisq].LRT[2:3]", "GLLVModels._anova_core(...; test = :chisq).LRT[2:3], as called at $cq",
            rl(ref_chisq, "LRT"), [Float64(tabq.LRT[i]) for i in 2:3], test_tolerance(tp, lrt6; nth = 2, after = "≈ Float64"), core_note("chi-square p-value")),
        mkcase("P1-JULIA-ANOVA-CHISQ-PVALUE", "anova p-value, test = :chisq (definition twin)",
            "$fxp [anova_chisq].pvalue[2:3]", "GLLVModels._anova_core(...; test = :chisq).pvalue[2:3], as called at $cq",
            rl(ref_chisq, "pvalue"), [Float64(tabq.pvalue[i]) for i in 2:3], test_tolerance(tp, pv6; nth = 2, after = "≈ Float64"), core_note("chi-square p-value")),
    ]
    taba = gllvm_anova(fits...; test = :chibar)
    ca = cite(tp, "tab = gllvm_anova(fits...; test = :chibar)")
    push!(cases,
        mkcase("P1-JULIA-ANOVA-E2E-LRT", "gllvm_anova(fits...; test = :chibar).LRT[2:3] on the Julia fits",
            "$fxp [anova_chibar].LRT[2:3]", "GLLVModels.gllvm_anova(fits...; test = :chibar).LRT[2:3], as called at $ca; fits as at $fitc",
            rl(ref_chibar, "LRT"), [Float64(taba.LRT[i]) for i in 2:3], test_tolerance(tp, "tab.LRT[i] ≈ Float64(ref[\"LRT\"][i]) atol=1e-2"; after = "≈ Float64"), e2e_note),
        mkcase("P1-JULIA-ANOVA-E2E-PVALUE", "gllvm_anova(fits...; test = :chibar).pvalue[2:3] on the Julia fits",
            "$fxp [anova_chibar].pvalue[2:3]", "GLLVModels.gllvm_anova(fits...; test = :chibar).pvalue[2:3], as called at $ca; fits as at $fitc",
            rl(ref_chibar, "pvalue"), [Float64(taba.pvalue[i]) for i in 2:3], test_tolerance(tp, "tab.pvalue[i] ≈ Float64(ref[\"pvalue\"][i]) atol=1e-2"; after = "≈ Float64"), e2e_note))

    mk(ids, cs) = Receipt(ids, ORIGIN, [fxp], [tp], NOT_A_FIXTURE_PAIR, cs)
    return [
        "model-comparison/AIC.json" => mk(["model-comparison/AIC.gllvmTMB_multi"], [c_aic]),
        "model-comparison/BIC.json" => mk(["model-comparison/BIC.gllvmTMB_multi"], [c_bic]),
        "model-comparison/anova.json" => mk(["model-comparison/anova.gllvmTMB_multi"], cases),
        "model-comparison/logLik.json" => mk(["namespace/S3method/logLik,gllvmTMB_multi"], [c_ll]),
    ]
end


# =============================================================================================
# 9. namespace numeric twins   test/test_namespace_numeric_p1_twin.jl (itchyshin/GLLVModels.jl#652)
#    extract_loadings, extract_rotated_loadings_table (one rank-2 Gaussian fit), extract_lv_effects
#    (predictor-informed fit), extract_communality, extract_Sigma_B, extract_Sigma_W (two-level fit),
#    each against R at P1 from test/fixtures/ns_numeric_p1.toml, plus vcov (the trait-mean block of the
#    full covariance, itchyshin/GLLVModels.jl#656). The single-level communality split is NOT bound;
#    see test/fixtures/repro_namespace_twin_gaps_p1.jl.
# =============================================================================================
function _ns_load_csv(path::AbstractString, trait_names::Vector{String}, n_cols::Integer;
                      x_cols::Vector{Int} = Int[], obs_col::Union{Nothing,Int} = nothing)
    p = length(trait_names)
    Y = zeros(Float64, p, n_cols)
    X = zeros(Float64, n_cols, length(x_cols))
    ind = zeros(Int, n_cols)
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            unit = parse(Int, strip(parts[1], '"'))
            col = obs_col === nothing ? unit : parse(Int, strip(parts[obs_col], '"'))
            t = findfirst(==(strip(parts[obs_col === nothing ? 2 : 3], '"')), trait_names)
            t === nothing && error("unrecognised trait in $path")
            Y[t, col] = parse(Float64, parts[obs_col === nothing ? 3 : 4])
            ind[col] = unit
            for (j, c) in enumerate(x_cols)
                X[col, j] = parse(Float64, parts[c])
            end
        end
    end
    return Y, X, ind
end
_ns_mat(v, nrow, ncol) = permutedims(reshape(Float64.(v), ncol, nrow))

function receipts_namespace_numeric()
    fxp = "test/fixtures/ns_numeric_p1.toml"
    tp = "test/test_namespace_numeric_p1_twin.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["gllvmtmb_commit"] == P1_SHA || fail("namespace numeric fixture is not pinned at P1")
    dir = "test/fixtures/"
    tn = String.(fx["trait_names"]); p, n = Int(fx["p"]), Int(fx["n_unit"])
    ORIGIN = "itchyshin/GLLVModels.jl#652"
    function chk(sec)
        d = fx[sec]
        datap = dir * d["data_file"]
        bytes2hex(sha256(read(joinpath(ROOT, datap)))) == d["data_sha256"] || fail("$sec data csv drifted")
        (d["converged"] === true && d["pd_hessian"] === true) || fail("$sec R fit not converged with a PD Hessian; not a valid twin")
        return d, datap
    end
    out = Pair{String,Receipt}[]
    mk(rel, sid, fixs, cs) = push!(out, "namespace-numeric/$rel.json" => Receipt([sid], ORIGIN, fixs, [tp], NOT_A_FIXTURE_PAIR, cs))

    # ---- main fit: extract_loadings, extract_rotated_loadings_table ----
    m, datap1 = chk("main")
    Y, _, _ = _ns_load_csv(joinpath(ROOT, datap1), tn, n)
    fit = fit_gllvm(Y; family = Normal(), K = 2)
    fit.converged || fail("main Julia fit did not converge")
    abs(fit.logLik - Float64(m["loglik"])) <= 1e-6 || fail("main logLik differs from R")
    note1 = "Rank-2 Gaussian fit on the fixture data (sha256 checked), R: latent(0 + trait | unit, d = 2, unique = FALSE) converged with a positive-definite Hessian; both fits reach the same optimum (logLik within 1e-6, asserted in the test). Matrix cases are compared elementwise (maximum absolute difference)."
    fitc = cite(tp, "fit = fit_gllvm(Y; family = Normal(), K = 2)")
    L = extract_loadings(fit; rotate = false)
    mk("extract_loadings", "namespace/export/extract_loadings", [fxp, datap1], [
        mkcase("P1-JULIA-EXTRACT-LOADINGS-RAW", "raw (rotate = none) lower-triangular loading matrix, 6 x 2",
            "$fxp [main.loadings]", "GLLVModels.extract_loadings(fit; rotate = false), as called at " * cite(tp, "L = extract_loadings(fit; rotate = false)") * "; fit as at $fitc",
            _ns_mat(m["loadings"], p, 2), L, test_tolerance(tp, "_ns_mat(m[\"loadings\"], p, 2)"),
            note1 * " Both engines use the same lower-triangular, native-sign convention, so no rotation or sign alignment is applied.")])
    trr = extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :raw)
    trs = extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :standardized)
    ctab = cite(tp, "tr = extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :raw)")
    ctabs = cite(tp, "ts = extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :standardized)")
    note2 = note1 * " Varimax, order_axes = true and sign_anchor = auto on both sides; the table is axis-major (6 traits for axis 1, then axis 2) and the test asserts the axis column exactly."
    mk("extract_rotated_loadings_table", "namespace/export/extract_rotated_loadings_table", [fxp, datap1], [
        mkcase("P1-JULIA-ROTATED-LOADINGS-TABLE-RAW", "varimax-rotated raw loadings, table column `loading` (12 values)",
            "$fxp [main.rot_raw_loading]", "GLLVModels.extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :raw).loading, as called at $ctab",
            Float64.(m["rot_raw_loading"]), trr.loading, test_tolerance(tp, "Float64.(m[\"rot_raw_loading\"])"), note2),
        mkcase("P1-JULIA-ROTATED-LOADINGS-TABLE-AXIS-VARIANCE", "table column `axis_variance`, one value per axis (2 values)",
            "$fxp [main.rot_raw_axis_variance]", "same call as $ctab, .axis_variance at the first row of each axis",
            Float64.(m["rot_raw_axis_variance"]), trr.axis_variance[[1, p + 1]], test_tolerance(tp, "Float64.(m[\"rot_raw_axis_variance\"])"), note2),
        mkcase("P1-JULIA-ROTATED-LOADINGS-TABLE-AXIS-SHARE", "table column `axis_share`, one value per axis (2 values)",
            "$fxp [main.rot_raw_axis_share]", "same call as $ctab, .axis_share at the first row of each axis",
            Float64.(m["rot_raw_axis_share"]), trr.axis_share[[1, p + 1]], test_tolerance(tp, "Float64.(m[\"rot_raw_axis_share\"])"), note2),
        mkcase("P1-JULIA-ROTATED-LOADINGS-TABLE-STANDARDIZED", "varimax-rotated standardized loadings, `loading` with loading_scale = standardized (12 values)",
            "$fxp [main.rot_std_loading]", "GLLVModels.extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :standardized).loading, as called at $ctabs",
            Float64.(m["rot_std_loading"]), trs.loading, test_tolerance(tp, "Float64.(m[\"rot_std_loading\"])"), note2)])

    # ---- main fit: vcov.gllvmTMB_multi (the full-covariance fix, itchyshin/GLLVModels.jl#656) ----
    Vj = vcov(fit, Y)
    Vjb = Vj[1:p, 1:p]
    tpv = "test/test_namespace_numeric_p1_twin.jl"
    (Vj isa Matrix{Float64} && issymmetric(Vj)) || fail("vcov is not a symmetric Matrix{Float64}")
    maximum(abs, (Vjb - Diagonal(diag(Vjb)))) > 1e-3 || fail("Julia vcov trait-mean block is diagonal")
    tolv = test_tolerance(tpv, "@test isapprox(diag(Vb), diag(Vr)")
    tolv[1] == test_tolerance(tpv, "@test isapprox(offd(Vb), offd(Vr)")[1] || fail("vcov diagonal and off-diagonal tolerances differ")
    push!(out, "namespace-numeric/vcov.json" => Receipt(["namespace/S3method/vcov,gllvmTMB_multi"], "itchyshin/GLLVModels.jl#656",
        [fxp, datap1], [tp], NOT_A_FIXTURE_PAIR, [
        mkcase("P1-JULIA-VCOV-TRAIT-MEAN", "vcov(fit): covariance of the six trait means (fixed-effect block of the inverse joint Hessian), 6 x 6 row-major, off-diagonals included",
            "$fxp [main.vcov]", "GLLVModels.vcov(fit, Y)[1:p, 1:p], as called at " * cite(tp, "V = vcov(fit, Y)") * "; fit as at $fitc",
            _ns_mat(m["vcov"], p, p), Vjb, tolv,
            note1 * " Both blocks are at the joint optimum on the natural scale of the trait means (trait order t1..t6), so no reparameterisation map is needed; the test asserts the diagonal and the off-diagonal parts separately at the same tolerance and that R's block is not diagonal (largest off-diagonal above 1e-3). Julia's vcov returned only a diagonal before itchyshin/GLLVModels.jl#656.")]))

    # ---- lv fit: extract_lv_effects ----
    l, datap2 = chk("lv")
    Y2, X2, _ = _ns_load_csv(joinpath(ROOT, datap2), tn, n; x_cols = [4, 5])
    fit2 = fit_gllvm(Y2; family = Normal(), K = 2, X_lv = X2)
    fit2.converged || fail("lv Julia fit did not converge")
    abs(fit2.logLik - Float64(l["loglik"])) <= 1e-6 || fail("lv logLik differs from R")
    fit2c = cite(tp, "fit = fit_gllvm(Y; family = Normal(), K = 2, X_lv = X)")
    note3 = "Rank-2 Gaussian fit with two unit-level predictors (x1, x2) on the fixture data (sha256 checked); R: latent(0 + trait | unit, d = 2, lv = ~ x1 + x2, unique = FALSE) converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test). Point estimates only; standard errors are not compared."
    mk("extract_lv_effects", "namespace/export/extract_lv_effects", [fxp, datap2], [
        mkcase("P1-JULIA-LV-EFFECTS-TRAIT", "type = trait_effect: B_lv = Lambda alpha', 6 x 2 (trait x predictor), rotation-stable",
            "$fxp [lv.trait_effect]", "GLLVModels.extract_lv_effects(fit; type = :trait_effect), as called at " * cite(tp, "Bj = extract_lv_effects(fit; type = :trait_effect)") * "; fit as at $fit2c",
            _ns_mat(l["trait_effect"], p, 2), extract_lv_effects(fit2; type = :trait_effect), test_tolerance(tp, "_ns_mat(l[\"trait_effect\"], p, 2)"), note3),
        mkcase("P1-JULIA-LV-EFFECTS-AXIS", "type = axis_effect: alpha, 2 x 2 (predictor x latent axis)",
            "$fxp [lv.axis_effect]", "GLLVModels.extract_lv_effects(fit; type = :axis_effect), as called at " * cite(tp, "Aj = extract_lv_effects(fit; type = :axis_effect)") * "; fit as at $fit2c",
            _ns_mat(l["axis_effect"], 2, 2), extract_lv_effects(fit2; type = :axis_effect), test_tolerance(tp, "_ns_mat(l[\"axis_effect\"], 2, 2)"),
            note3 * " Axis effects are rotation dependent in general; both engines use the same lower-triangular Lambda convention here, so they are compared without alignment.")])

    # ---- two-level fit: extract_communality, extract_Sigma_B, extract_Sigma_W ----
    t, datap3 = chk("two")
    p2 = Int(t["p"])
    Y3, _, ind = _ns_load_csv(joinpath(ROOT, datap3), String.(t["trait_names"]), Int(t["n_obs"]); obs_col = 2)
    fit3 = fit_twolevel_gaussian(Y3, ind; K_B = 1, K_W = 1)
    fit3.converged || fail("two-level Julia fit did not converge")
    abs(fit3.loglik - Float64(t["loglik"])) <= 1e-6 || fail("two-level logLik differs from R")
    fit3c = cite(tp, "fit = fit_twolevel_gaussian(Y, ind; K_B = 1, K_W = 1)")
    note4 = "Two-level Gaussian fit, p = 5, 120 units x 4 observations (sha256 checked); R: latent(d = 1) + unique at unit and at unit_obs, converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test). R's observation-level residual sigma_eps is pinned near 0 by gllvmTMB when unique() is present, matching the Julia two-level model, which has no separate sigma_eps."
    mk("extract_communality", "namespace/export/extract_communality", [fxp, datap3], [
        mkcase("P1-JULIA-COMMUNALITY-UNIT", "per-trait communality at level = unit (between tier), 5 values",
            "$fxp [two.communality_unit]", "GLLVModels.extract_communality(fit; level = :unit), as called at " * cite(tp, "extract_communality(fit; level = :unit)") * "; fit as at $fit3c",
            Float64.(t["communality_unit"]), extract_communality(fit3; level = :unit), test_tolerance(tp, "Float64.(t[\"communality_unit\"])"), note4),
        mkcase("P1-JULIA-COMMUNALITY-UNIT-OBS", "per-trait communality at level = unit_obs (within tier), 5 values",
            "$fxp [two.communality_unit_obs]", "GLLVModels.extract_communality(fit; level = :unit_obs), as called at " * cite(tp, "extract_communality(fit; level = :unit_obs)") * "; fit as at $fit3c",
            Float64.(t["communality_unit_obs"]), extract_communality(fit3; level = :unit_obs), test_tolerance(tp, "Float64.(t[\"communality_unit_obs\"])"), note4)])
    sigB = extract_Sigma(fit3; level = :unit).Sigma
    sigW = extract_Sigma(fit3; level = :unit_obs).Sigma
    noteS = note4 * " The Julia call is extract_Sigma(fit; level = ...) on the two-level fit, the accessor that carries the legacy extract_Sigma_B / extract_Sigma_W payload (R: extract_Sigma_B(fit)\$Sigma_B, soft-deprecated at 0.7.0 in favour of extract_Sigma(level = ...)); the registered Julia symbol for these rows is the TwoLevelFit type, not a function."
    mk("extract_Sigma_B", "namespace/export/extract_Sigma_B", [fxp, datap3], [
        mkcase("P1-JULIA-SIGMA-B-TWOLEVEL", "between-tier total covariance Sigma_B, 5 x 5",
            "$fxp [two.sigma_unit]", "GLLVModels.extract_Sigma(fit; level = :unit).Sigma, as called at " * cite(tp, "extract_Sigma(fit; level = :unit).Sigma") * "; fit as at $fit3c",
            _ns_mat(t["sigma_unit"], p2, p2), sigB, test_tolerance(tp, "_ns_mat(t[\"sigma_unit\"], p2, p2)"), noteS)])
    mk("extract_Sigma_W", "namespace/export/extract_Sigma_W", [fxp, datap3], [
        mkcase("P1-JULIA-SIGMA-W-TWOLEVEL", "within-tier total covariance Sigma_W, 5 x 5",
            "$fxp [two.sigma_unit_obs]", "GLLVModels.extract_Sigma(fit; level = :unit_obs).Sigma, as called at " * cite(tp, "extract_Sigma(fit; level = :unit_obs).Sigma") * "; fit as at $fit3c",
            _ns_mat(t["sigma_unit_obs"], p2, p2), sigW, test_tolerance(tp, "_ns_mat(t[\"sigma_unit_obs\"], p2, p2)"), noteS)])
    return out
end

# =============================================================================================
# 10. postfit twins   test/test_postfit_twins_p1.jl and test/test_namespace_numeric_p1_twin.jl
#     Five postfit rows whose batch cases were NON-DISCRIMINATING (R values constant or ~1e-14),
#     PARTIAL (table-shape check only) or never executed (POST-DEVIANCE), bound to twins on
#     non-degenerate fixtures: extract_communality (two-level fit) and
#     extract_rotated_loadings_table (rank-2 fit) reuse the R values of ns_numeric_p1.toml;
#     tidy (fixed effects), coef and deviance read test/fixtures/postfit_twins_p1.toml.
# =============================================================================================
function receipts_postfit_twins()
    ORIGIN = "itchyshin/GLLVModels.jl#660"
    out = Pair{String,Receipt}[]
    dir = "test/fixtures/"
    mk(rel, sid, fixs, tests, cs) = push!(out, "postfit-twins/$rel.json" => Receipt([sid], ORIGIN, fixs, tests, NOT_A_FIXTURE_PAIR, cs))

    # ---- shared with section 9: ns_numeric_p1.toml ----
    nsp = "test/fixtures/ns_numeric_p1.toml"
    tpn = "test/test_namespace_numeric_p1_twin.jl"
    ns = TOML.parsefile(joinpath(ROOT, nsp))
    ns["gllvmtmb_commit"] == P1_SHA || fail("namespace numeric fixture is not pinned at P1")
    tn = String.(ns["trait_names"]); p, n = Int(ns["p"]), Int(ns["n_unit"])
    function chk(sec)
        d = ns[sec]
        datap = dir * d["data_file"]
        bytes2hex(sha256(read(joinpath(ROOT, datap)))) == d["data_sha256"] || fail("$sec data csv drifted")
        (d["converged"] === true && d["pd_hessian"] === true) || fail("$sec R fit not converged with a PD Hessian; not a valid twin")
        return d, datap
    end

    # ---- two-level fit: extract_communality (unit, unit_obs) ----
    t, datap3 = chk("two")
    Y3, _, ind = _ns_load_csv(joinpath(ROOT, datap3), String.(t["trait_names"]), Int(t["n_obs"]); obs_col = 2)
    fit3 = fit_twolevel_gaussian(Y3, ind; K_B = 1, K_W = 1)
    fit3.converged || fail("two-level Julia fit did not converge")
    abs(fit3.loglik - Float64(t["loglik"])) <= 1e-6 || fail("two-level logLik differs from R")
    fit3c = cite(tpn, "fit = fit_twolevel_gaussian(Y, ind; K_B = 1, K_W = 1)")
    noteC = "Two-level Gaussian fit, p = 5, 120 units x 4 observations (sha256 checked); R: latent(d = 1) + unique at unit and at unit_obs, converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test). Replaces the postfit batch case CORE070-ESTIMAND-REBIND-EXTRACT-COMMUNALITY, whose unique = FALSE fixture returned 1 for every trait (a constant 1.0 implementation passed): here the R communalities are five distinct values between 0.08 and 0.61 at the unit tier and between 0.27 and 0.61 at the unit_obs tier."
    mk("extract_communality", "postfit/POSTFIT-SURFACE-extract_communality", [nsp, datap3], [tpn], [
        mkcase("P1-JULIA-POSTFIT-COMMUNALITY-UNIT", "per-trait communality at level = unit (between tier), 5 non-constant values",
            "$nsp [two.communality_unit]", "GLLVModels.extract_communality(fit; level = :unit), as called at " * cite(tpn, "extract_communality(fit; level = :unit)") * "; fit as at $fit3c",
            Float64.(t["communality_unit"]), extract_communality(fit3; level = :unit), test_tolerance(tpn, "Float64.(t[\"communality_unit\"])"), noteC),
        mkcase("P1-JULIA-POSTFIT-COMMUNALITY-UNIT-OBS", "per-trait communality at level = unit_obs (within tier), 5 non-constant values",
            "$nsp [two.communality_unit_obs]", "GLLVModels.extract_communality(fit; level = :unit_obs), as called at " * cite(tpn, "extract_communality(fit; level = :unit_obs)") * "; fit as at $fit3c",
            Float64.(t["communality_unit_obs"]), extract_communality(fit3; level = :unit_obs), test_tolerance(tpn, "Float64.(t[\"communality_unit_obs\"])"), noteC)])

    # ---- main fit: extract_rotated_loadings_table ----
    m, datap1 = chk("main")
    Y, _, _ = _ns_load_csv(joinpath(ROOT, datap1), tn, n)
    fit = fit_gllvm(Y; family = Normal(), K = 2)
    fit.converged || fail("main Julia fit did not converge")
    abs(fit.logLik - Float64(m["loglik"])) <= 1e-6 || fail("main logLik differs from R")
    fitc = cite(tpn, "fit = fit_gllvm(Y; family = Normal(), K = 2)")
    trr = extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :raw)
    trs = extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :standardized)
    ctab = cite(tpn, "tr = extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :raw)")
    ctabs = cite(tpn, "ts = extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :standardized)")
    noteR = "Rank-2 Gaussian fit on the fixture data (sha256 checked), R: latent(0 + trait | unit, d = 2, unique = FALSE) converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test). Varimax, order_axes = true and sign_anchor = auto on both sides; the table is axis-major (6 traits for axis 1, then axis 2). Replaces the postfit batch case CORE070-WAVE8-EXTRACT-ROTATED-LOADINGS-TABLE-SHAPE, which compared table shape only."
    mk("extract_rotated_loadings_table", "postfit/POSTFIT-SURFACE-extract_rotated_loadings_table", [nsp, datap1], [tpn], [
        mkcase("P1-JULIA-POSTFIT-ROTATED-LOADINGS-TABLE-RAW", "varimax-rotated raw loadings, table column `loading` (12 values)",
            "$nsp [main.rot_raw_loading]", "GLLVModels.extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :raw).loading, as called at $ctab; fit as at $fitc",
            Float64.(m["rot_raw_loading"]), trr.loading, test_tolerance(tpn, "Float64.(m[\"rot_raw_loading\"])"), noteR),
        mkcase("P1-JULIA-POSTFIT-ROTATED-LOADINGS-TABLE-AXIS-VARIANCE", "table column `axis_variance`, one value per axis (2 values)",
            "$nsp [main.rot_raw_axis_variance]", "same call as $ctab, .axis_variance at the first row of each axis",
            Float64.(m["rot_raw_axis_variance"]), trr.axis_variance[[1, p + 1]], test_tolerance(tpn, "Float64.(m[\"rot_raw_axis_variance\"])"), noteR),
        mkcase("P1-JULIA-POSTFIT-ROTATED-LOADINGS-TABLE-AXIS-SHARE", "table column `axis_share`, one value per axis (2 values)",
            "$nsp [main.rot_raw_axis_share]", "same call as $ctab, .axis_share at the first row of each axis",
            Float64.(m["rot_raw_axis_share"]), trr.axis_share[[1, p + 1]], test_tolerance(tpn, "Float64.(m[\"rot_raw_axis_share\"])"), noteR),
        mkcase("P1-JULIA-POSTFIT-ROTATED-LOADINGS-TABLE-STANDARDIZED", "varimax-rotated standardized loadings, `loading` with loading_scale = standardized (12 values)",
            "$nsp [main.rot_std_loading]", "GLLVModels.extract_rotated_loadings_table(fit, Y; method = :varimax, loading_scale = :standardized).loading, as called at $ctabs; fit as at $fitc",
            Float64.(m["rot_std_loading"]), trs.loading, test_tolerance(tpn, "Float64.(m[\"rot_std_loading\"])"), noteR)])

    # ---- uncentred main fit: tidy (fixed), coef, deviance ----
    pfp = "test/fixtures/postfit_twins_p1.toml"
    tpp = "test/test_postfit_twins_p1.jl"
    pf = TOML.parsefile(joinpath(ROOT, pfp))
    pf["gllvmtmb_commit"] == P1_SHA || fail("postfit twin fixture is not pinned at P1")
    q = pf["main"]
    datapq = dir * q["data_file"]
    bytes2hex(sha256(read(joinpath(ROOT, datapq)))) == q["data_sha256"] || fail("postfit twin data csv drifted")
    (q["converged"] === true && q["pd_hessian"] === true) || fail("postfit twin R fit not converged with a PD Hessian; not a valid twin")
    abs(fit.logLik - Float64(q["loglik"])) <= 1e-6 || fail("postfit twin logLik differs from R")
    fitq = cite(tpp, "fit = fit_gllvm(Y; family = Normal(), K = 2)")
    noteU = "Rank-2 Gaussian fit on UNCENTRED data (trait means 0.5, -0.3, 0.2, 0.8, -0.6, 0.1; sha256 checked); R: value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE) converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test). The R values are far from zero (smallest |value| about 0.058) and pairwise distinct, so a constant or zero implementation fails. The postfit batch fixture these rows replace was row-centred (R values ~1e-14)."
    Rcoef = Float64.(q["coef"])
    rows = tidy(fit, Y)
    mk("tidy", "postfit/POSTFIT-SURFACE-tidy.gllvmTMB_multi", [pfp, datapq], [tpp], [
        mkcase("P1-JULIA-POSTFIT-TIDY-FIXED", "tidy(fit)\$estimate, fixed effects (6 trait means)",
            "$pfp [main.tidy_estimate]", "[r.estimate for r in GLLVModels.tidy(fit, Y)], as called at " * cite(tpp, "rows = tidy(fit, Y)") * "; fit as at $fitq",
            Float64.(q["tidy_estimate"]), [r.estimate for r in rows], test_tolerance(tpp, "[r.estimate for r in rows]"), noteU)])
    mk("coef", "postfit-policy/POST-COEF-NAMED", [pfp, datapq], [tpp], [
        mkcase("P1-JULIA-POSTFIT-COEF-NAMED", "coef(fit), 6 fixed-effect coefficients (trait means)",
            "$pfp [main.coef]", "StatsAPI.coef(fit), as called at " * cite(tpp, "@test isapprox(coef(fit), Float64.(m[\"coef\"])") * "; fit as at $fitq",
            Rcoef, coef(fit), test_tolerance(tpp, "@test isapprox(coef(fit), Float64.(m[\"coef\"])"), noteU * " Compared elementwise by value; the R names (traitt1, ..., traitt6) are asserted equal to \"trait\" * trait name in the test, Julia's coef is unnamed.")])
    mk("deviance", "postfit-policy/POST-DEVIANCE", [pfp, datapq], [tpp], [
        mkcase("P1-JULIA-POSTFIT-DEVIANCE", "deviance(fit), scalar",
            "$pfp [main.deviance]", "StatsAPI.deviance(fit), as called at " * cite(tpp, "@test isapprox(deviance(fit)") * "; fit as at $fitq",
            Float64(q["deviance"]), deviance(fit), test_tolerance(tpp, "@test isapprox(deviance(fit)"), noteU * " Julia's deviance is -2 logLik; R's is read from deviance(fit), so the comparison is R's own deviance against that identity, not a copy of the log-likelihood guard.")])
    return out
end

# =============================================================================================
# 11. namespace numeric twins (b)   test/test_namespace_numeric_p1_twin_b.jl
#    tidy.gllvmTMB_multi (fixed effects of the rank-2 Gaussian fit), and the Beta and nbinom2 family
#    exports (one-axis latent fits), each against R at P1 from test/fixtures/ns_numeric_p1_b.toml.
# =============================================================================================
function receipts_namespace_numeric_b()
    fxp = "test/fixtures/ns_numeric_p1_b.toml"
    tp = "test/test_namespace_numeric_p1_twin_b.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["gllvmtmb_commit"] == P1_SHA || fail("namespace numeric (b) fixture is not pinned at P1")
    dir = "test/fixtures/"
    p, n = Int(fx["p"]), Int(fx["n_unit"])
    ORIGIN = "itchyshin/GLLVModels.jl#663"
    function chk(sec)
        d = fx[sec]
        datap = dir * d["data_file"]
        bytes2hex(sha256(read(joinpath(ROOT, datap)))) == d["data_sha256"] || fail("$sec data csv drifted")
        (d["converged"] === true && d["pd_hessian"] === true) || fail("$sec R fit not converged with a PD Hessian; not a valid twin")
        return d, datap
    end
    out = Pair{String,Receipt}[]
    mk(rel, sid, fixs, cs) = push!(out, "namespace-numeric/$rel.json" => Receipt([sid], ORIGIN, fixs, [tp], NOT_A_FIXTURE_PAIR, cs))

    # ---- tidy ----
    t, datap = chk("tidy")
    Y, _, _ = _ns_load_csv(joinpath(ROOT, datap), String.(fx["trait_names"]), n)
    fit = fit_gllvm(Y; family = Normal(), K = 2)
    fit.converged || fail("tidy Julia fit did not converge")
    abs(fit.logLik - Float64(t["loglik"])) <= 1e-6 || fail("tidy logLik differs from R")
    tb = tidy(fit, Y)
    length(tb) == p == length(t["terms"]) || fail("tidy row count differs from R")
    fitc = cite(tp, "fit = fit_gllvm(Y; family = Normal(), K = 2)")
    tcall = cite(tp, "tb = tidy(fit, Y)")
    noteT = "Rank-2 Gaussian fit on the fixture data (sha256 checked; the same data as the main fit of the extract_loadings twin), R: latent(0 + trait | unit, d = 2, unique = FALSE) converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test). Rows are the six trait intercepts in trait order (R terms traitt1..traitt6, Julia beta[1..6]); the test asserts the row count and the identity link on both sides, and the numbers are compared elementwise."
    mk("tidy", "namespace/S3method/tidy,gllvmTMB_multi", [fxp, datap], [
        mkcase("P1-JULIA-TIDY-ESTIMATE", "tidy(fit) fixed rows, column `estimate`, 6 trait intercepts",
            "$fxp [tidy.estimate]", "GLLVModels.tidy(fit, Y), field estimate, as called at $tcall; fit as at $fitc",
            Float64.(t["estimate"]), [r.estimate for r in tb], test_tolerance(tp, "[r.estimate for r in tb]"), noteT),
        mkcase("P1-JULIA-TIDY-STD-ERROR", "tidy(fit) fixed rows, column `std.error`, 6 Wald standard errors",
            "$fxp [tidy.std_error]", "GLLVModels.tidy(fit, Y), field std_error, as called at $tcall; fit as at $fitc",
            Float64.(t["std_error"]), [r.std_error for r in tb], test_tolerance(tp, "[r.std_error for r in tb]"), noteT)])

    # ---- Beta, nbinom2 ----
    for (k, (sec, sid, fam, fld, famtxt, rfam)) in enumerate((
            ("beta", "namespace/export/Beta", GLLVModels.Beta(), :φ, "fit_gllvm(Y; family = GLLVModels.Beta(), K = 1)", "Beta()"),
            ("nb2", "namespace/export/nbinom2", NegativeBinomial(1.0, 0.5), :r_group, "fit_gllvm(Y; family = NegativeBinomial(1.0, 0.5), disp_group = :species, K = 1)", "nbinom2()")))
        b, datab = chk(sec)
        Yb, _, _ = _ns_load_csv(joinpath(ROOT, datab), String.(fx["trait_names"]), n)
        f = sec == "nb2" ? fit_gllvm(Yb; family = fam, disp_group = :species, K = 1) : fit_gllvm(Yb; family = fam, K = 1)
        f.converged || fail("$sec Julia fit did not converge")
        f.group == collect(1:p) || fail("$sec dispersion is not per trait")
        abs(f.loglik - Float64(b["loglik"])) <= 1e-6 || fail("$sec logLik differs from R")
        fc = cite(tp, "fit = fit_gllvm(Y; family = " * (k == 1 ? "GLLVModels.Beta()" : "NegativeBinomial(1.0, 0.5), disp_group = :species") * ", K = 1)")
        note = "One-axis latent fit, p = 6, n = 200 (sha256 checked); R: value ~ 0 + trait + latent(0 + trait | unit, d = 1, unique = FALSE), family = $rfam, converged with a positive-definite Hessian; same optimum (logLik within 1e-6, asserted in the test). Dispersion is per trait on both sides. The sign of a one-axis loading is not identified, so loadings are compared through Lambda Lambda' (6 x 6)."
        L = f.Λ * f.Λ'
        mk(sec == "beta" ? "Beta" : "nbinom2", sid, [fxp, datab], [
            mkcase("P1-JULIA-$(uppercase(sec))-INTERCEPTS", "trait intercepts (6 values) of a family = $rfam latent fit",
                "$fxp [$sec.beta]", "GLLVModels.fit_gllvm(...).β, fit as at $fc",
                Float64.(b["beta"]), f.β, test_tolerance(tp, "@test isapprox(bj, Float64.(b[\"beta\"])"; nth = k), note),
            mkcase("P1-JULIA-$(uppercase(sec))-LAMBDA-LAMBDAT", "Lambda Lambda' (6 x 6) of a family = $rfam latent fit",
                "$fxp [$sec.lambda_lambdat]", "GLLVModels.fit_gllvm(...).Λ * Λ', fit as at $fc",
                _ns_mat(b["lambda_lambdat"], p, p), L, test_tolerance(tp, "@test isapprox(LLt, _nsb_mat(b[\"lambda_lambdat\"], p, p)"; nth = k), note),
            mkcase("P1-JULIA-$(uppercase(sec))-DISPERSION", "per-trait dispersion phi (6 values) of a family = $rfam latent fit",
                "$fxp [$sec.phi]", "GLLVModels.fit_gllvm(...).$(fld), fit as at $fc",
                Float64.(b["phi"]), getproperty(f, fld), test_tolerance(tp, "@test isapprox(phij, Float64.(b[\"phi\"])"; nth = k),
                note * (sec == "beta" ? " R reports phi_beta; Julia the Beta precision phi." : " R reports exp(log_phi_nbinom2) with variance mu + mu^2/phi; Julia the NB size r (the same parameter)."))])
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
function build()
    out = Pair{String,Receipt}[]
    for (name, f) in (("chibar", receipts_chibar), ("ordinal", receipts_ordinal),
            ("latent-scores", receipts_latent_scores), ("temporal", receipts_temporal),
            ("aghq", receipts_aghq), ("model-comparison", receipts_model_comparison),
            ("select-lv", receipts_select_lv), ("isdm", receipts_isdm),
            ("isdm-admission", receipts_isdm_admission),
            ("namespace-numeric", receipts_namespace_numeric),
            ("postfit-twins", receipts_postfit_twins),
            ("namespace-numeric-b", receipts_namespace_numeric_b))
        t0 = time()
        append!(out, f())
        @info "built $name receipts" seconds = round(time() - t0; digits = 1)
    end
    return out
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
    old["julia_version"] == string(VERSION) || @info "$rel: receipt was generated on Julia $(old["julia_version"]), this run is $VERSION (informational)"
    old["gllvmodels_commit"] == gllvmodels_commit() || @info "$rel: receipt names src commit $(old["gllvmodels_commit"][1:9]), current src commit is $(gllvmodels_commit()[1:9]) (informational)"
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
        worst <= 1 || push!(probs, "$id: recomputed Julia value moved by more than the strict allowance (max |Δ| / allowance = $(round(worst; sigdigits = 3)))")
        n["abs_diff"] <= n["tolerance"] || push!(probs, "$id: abs_diff now exceeds tolerance")
    end
    return probs
end

function main(args)
    check = "--check" in args
    t0 = time()
    receipts = try
        build()
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
        if isempty(probs)
            println("OK $(length(receipts)) Julia receipts reproduce within $(CHECK_FRACTION) x tolerance ($(round(time() - t0; digits = 1)) s, Julia $VERSION)")
            return 0
        end
        foreach(p -> println("STALE ", p), probs)
        return 1
    end
    for (rel, r) in receipts
        p = joinpath(ROOT, OUT_DIR, rel)
        mkpath(dirname(p))
        write(p, jrender(receipt_object(r)))
    end
    println("wrote $(length(receipts)) receipts under $OUT_DIR ($(round(time() - t0; digits = 1)) s, Julia $VERSION)")
    return 0
end

exit(main(ARGS))
