#!/usr/bin/env julia
# Julia-side receipts for the namespace rows export/gllvmTMB_wide, S3method/ordiplot,gllvmTMB_multi
# and export/flag_unreliable_loadings (true-parity checker, tools/true_parity_check.mjs, clauses
# C1/C8).
#
# A standalone sibling of tools/true_parity_animal_scalar_receipt.jl, kept in its own file so that
# they do not collide. It follows the same rules, and the helpers below (Fail, findline,
# test_tolerance, cite, mkcase, the JSON writer and reader, Receipt, receipt_object,
# compare_receipt) are copied from that file unchanged; mkcase_int is added for the flag columns,
# which are compared as exact integer codes (kind integer_equality, tolerance 0.5):
#   * R values are copied from the tracked fixture test/fixtures/ns_gauss_w1_p1.toml; nothing on
#     the R side is recomputed here.
#   * The Julia side repeats the computation of test/test_namespace_gaussian_w1_p1.jl with the
#     same inputs and settings; the test file:line is recorded in `julia_source`.
#   * The tolerance is READ from the existing assertion in the test (file:line and the line's
#     text recorded). A difference above it aborts the run and nothing is written.
#   * src/, the test and the fixture are not modified.
# It writes three receipts under docs/dev-log/core070/true-parity-latest/receipts/julia-twins/
# namespace-numeric/: gllvmTMB_wide.json, ordiplot.json and flag_unreliable_loadings.json.
#
# Usage (from the repository root)
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_namespace_gaussian_w1_receipt.jl
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_namespace_gaussian_w1_receipt.jl --check
# --check recomputes and compares with the committed receipts: fixture and test sha256, case ids, R
# values and tolerances identical; each Julia value within 1e-3 x tolerance + 100 eps of the
# committed one; abs_diff still within tolerance.

using GLLVModels, TOML, SHA, Statistics, LinearAlgebra
using Distributions: Normal
using Statistics: std
const GMJ = GLLVModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins"
const P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const GENERATOR = "tools/true_parity_namespace_gaussian_w1_receipt.jl"
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

"""Integer-equality case (checker kind integer_equality): r and j are integer codes and the
tolerance, read from the test like any other, must be 0.5, so within tolerance means equal."""
function mkcase_int(id, quantity, r_source, julia_source, r, j, tol, note)
    rv, jv = Int.(collect(r)), Int.(collect(j))
    length(rv) == length(jv) && !isempty(rv) || fail("$id: integer vectors differ in length")
    d = Float64(maximum(abs.(rv .- jv)))
    t, src, line = tol
    t == 0.5 || fail("$id: integer_equality needs tolerance 0.5, got $t ($src)")
    d <= t || fail("$id: integer codes differ ($src); do not bind the row")
    return Case(Pair{String,Any}[
        "case_id" => id, "quantity" => quantity, "kind" => "integer_equality",
        "r_source" => r_source, "julia_source" => julia_source,
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
    "fresh by tools/true_parity_namespace_gaussian_w1_receipt.jl, which repeats the computation of the cited twin " *
    "test with the same inputs and settings; R was not run and is not recomputed here."

# R's write.csv long data (unit, trait, value) -> p x n response matrix; same parser as
# test/test_namespace_gaussian_w1_p1.jl.
function load_data(path, trait_names, n)
    Y = fill(NaN, length(trait_names), n)
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
    all(isfinite, Y) || fail("incomplete data in $path")
    return Y
end

rowmajor(v, nrow, ncol) = permutedims(reshape(Float64.(v), ncol, nrow))
flagcode(x) = x === missing ? -1 : Int(x)
rflagcode(s) = s == "NA" ? -1 : s == "TRUE" ? 1 : s == "FALSE" ? 0 : fail("bad flag $s")

# ---------------------------------------------------------------------------------------------
# gllvmTMB_wide, ordiplot, flag_unreliable_loadings   test/test_namespace_gaussian_w1_p1.jl
# ---------------------------------------------------------------------------------------------
function receipts_namespace_gaussian_w1()
    fxp = "test/fixtures/ns_gauss_w1_p1.toml"
    tp = "test/test_namespace_gaussian_w1_p1.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["gllvmtmb_commit"] == P1_SHA || fail("fixture is not pinned at P1")
    dp = "test/fixtures/" * fx["data_file"]
    sha_file(dp) == fx["data_sha256"] || fail("data csv drifted: $dp")
    p, n = Int(fx["p"]), Int(fx["n"])
    Y = load_data(dp, String.(fx["trait_names"]), n)
    data_note = "Shared Gaussian data ns_gauss_p1_data.csv (written by gen_namespace_numeric_p1.R; p = 6 traits, n = 200 units, one observation per trait and unit, sha256 checked), simulated from two latent axes plus residual noise. R values from gen_namespace_gaussian_w1_p1.R at the P1 pin; every R fit converged (nlminb code 0, rel.tol / sing.tol / x.tol 1e-12) with a positive-definite Hessian."

    # ---- gllvmTMB_wide ----
    w = fx["wide"]
    (w["converged"] === true && w["pd_hessian"] === true) || fail("wide: R fit did not converge with a PD Hessian")
    r_L = rowmajor(w["Lambda"], p, 2)
    r_beta, r_sdB = Float64.(w["beta"]), Float64.(w["sd_B"])
    c = max(1e-3 * std(vec(Y)), 1e-6)   # R's Q7 rule (R/fit-multi.R), as in the test
    abs(c - Float64(w["sigma_eps_fixed"])) <= 1e-15 || fail("wide: Q7 residual SD differs from the recorded R value")
    fw = GMJ.fit_gaussian_pervar_gllvm(Y; K = 2, fixed_residual_sd = c)
    fw.converged || fail("wide: Julia fit did not converge")
    X = zeros(p, n, p)
    for t in 1:p, s in 1:n
        X[t, s, t] = 1.0
    end
    ll_at_r = GMJ.gaussian_pervar_marginal_loglik(Y, r_L, r_sdB .^ 2 .+ c^2; X = X, β = r_beta)
    wfit = cite(tp, "fit = fit_gaussian_pervar_gllvm(Y; K = 2, fixed_residual_sd = c)")
    wnote = "$data_note gllvmTMB_wide(Y, d = 2) on the wide 200 x 6 matrix builds value ~ 0 + trait + latent(0 + trait | site, d = 2) with latent()'s default unique = TRUE, so R fits Sigma = Lambda Lambda' + diag(sd_B^2) + sigma_eps^2 I with sigma_eps mapped off and fixed by R's Q7 rule (1e-3 x sd(y) over all responses, $(c)). Julia recomputes c with the same rule (checked against the recorded R value) and fits the same covariance with that fixed residual as fit_gaussian_pervar_gllvm(Y; K = 2, fixed_residual_sd = c) (at $wfit); trait intercepts are the profiled row means. Loadings are compared as Lambda Lambda', which does not depend on the rotation. Limits: Gaussian only; the default call (no X, weights, phylo_vcv or formula_extra); the Julia side is the native fitter, not a wide-matrix wrapper."
    cw = [
        mkcase("P1-JULIA-GLLVMTMB-WIDE-LOGLIK", "log-likelihood at the optimum",
            "$fxp [wide.loglik]", "fit.loglik, as compared at " * cite(tp, "@test isapprox(fit.loglik, Float64(w[\"loglik\"]);") * "; fit as at $wfit",
            Float64(w["loglik"]), fw.loglik, test_tolerance(tp, "@test isapprox(fit.loglik, Float64(w[\"loglik\"]);"), wnote),
        mkcase("P1-JULIA-GLLVMTMB-WIDE-BETA", "trait intercepts b_fix ($(p) values)",
            "$fxp [wide.beta]", "fit.β, as compared at " * cite(tp, "@test isapprox(fit.β, r_beta;") * "; fit as at $wfit",
            r_beta, fw.β, test_tolerance(tp, "@test isapprox(fit.β, r_beta;"), wnote),
        mkcase("P1-JULIA-GLLVMTMB-WIDE-LAMBDA-LAMBDAT", "Lambda Lambda' from report\$Lambda_B ($(p*p) values, column-major)",
            "$fxp [wide.Lambda]", "fit.Λ * fit.Λ', as compared at " * cite(tp, "@test isapprox(fit.Λ * fit.Λ', r_L * r_L';") * "; fit as at $wfit",
            r_L * r_L', fw.Λ * fw.Λ', test_tolerance(tp, "@test isapprox(fit.Λ * fit.Λ', r_L * r_L';"), wnote),
        mkcase("P1-JULIA-GLLVMTMB-WIDE-SD-B", "per-trait unique SDs report\$sd_B ($(p) values)",
            "$fxp [wide.sd_B]", "sqrt.(fit.ψ²), as compared at " * cite(tp, "@test isapprox(sqrt.(fit.ψ²), r_sdB;") * "; fit as at $wfit",
            r_sdB, sqrt.(fw.ψ²), test_tolerance(tp, "@test isapprox(sqrt.(fit.ψ²), r_sdB;"), wnote),
        mkcase("P1-JULIA-GLLVMTMB-WIDE-OBJECTIVE-AT-R", "Julia log-likelihood evaluated at R's estimates (b_fix, Lambda_B, sd_B^2 + sigma_eps^2) against R's log-likelihood",
            "$fxp [wide.loglik]", "gaussian_pervar_marginal_loglik(Y, r_L, r_sdB .^ 2 .+ c^2; X, β = r_beta), as compared at " * cite(tp, "@test isapprox(ll_at_r, Float64(w[\"loglik\"]);"),
            Float64(w["loglik"]), ll_at_r, test_tolerance(tp, "@test isapprox(ll_at_r, Float64(w[\"loglik\"]);"), wnote),
    ]

    # ---- ordiplot ----
    o = fx["ordiplot"]
    (o["converged"] === true && o["pd_hessian"] === true) || fail("ordiplot: R fit did not converge with a PD Hessian")
    r_S, r_Lo = rowmajor(o["scores"], n, 2), rowmajor(o["loadings"], p, 2)
    fo = GMJ.fit_gllvm(Y; family = Normal(), K = 2)
    fo.converged || fail("ordiplot: Julia fit did not converge")
    od = GMJ.ordiplot(fo, Y)
    raw = GMJ.ordiplot(fo, Y; rotate = false)
    ofit = cite(tp, "fit = fit_gllvm(Y; family = Normal(), K = 2)")
    ocall = cite(tp, "od = ordiplot(fit, Y)")
    onote = "$data_note R: ordiplot(fit) on value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE) (plot sent to a null device); its invisible return list(scores, loadings) is recorded, unrotated (rotate = \"none\", the default). Julia: ordiplot(fit_gllvm(Y; family = Normal(), K = 2), Y) (fit at $ofit, call at $ocall), the same model, which returns the principal-rotated sites and species by default. The two are compared through rotation-invariant products: scores * loadings' (the latent part of the linear predictor) and loadings * loadings', and ordiplot(fit, Y; rotate = false) (at $(cite(tp, "raw = ordiplot(fit, Y; rotate = false)"))) is compared with R's raw scores and loadings directly (both use the lower-triangular loading convention). Limits: Gaussian only; the returned data, not the drawing; the axes, biplot and ellipse arguments are not exercised."
    co = [
        mkcase("P1-JULIA-ORDIPLOT-SCORES-LOADINGS", "ordiplot scores * loadings' ($(n*p) values, column-major)",
            "$fxp [ordiplot.scores, ordiplot.loadings]", "od.sites * od.species', as compared at " * cite(tp, "@test isapprox(od.sites * od.species', r_S * r_L';"),
            r_S * r_Lo', od.sites * od.species', test_tolerance(tp, "@test isapprox(od.sites * od.species', r_S * r_L';"), onote),
        mkcase("P1-JULIA-ORDIPLOT-LOADINGS-GRAM", "ordiplot loadings * loadings' ($(p*p) values, column-major)",
            "$fxp [ordiplot.loadings]", "od.species * od.species', as compared at " * cite(tp, "@test isapprox(od.species * od.species', r_L * r_L';"),
            r_Lo * r_Lo', od.species * od.species', test_tolerance(tp, "@test isapprox(od.species * od.species', r_L * r_L';"), onote),
        mkcase("P1-JULIA-ORDIPLOT-RAW-SCORES", "ordiplot raw scores, rotate = false in Julia and R's default rotate = \"none\" ($(2n) values, column-major)",
            "$fxp [ordiplot.scores]", "raw.sites, as compared at " * cite(tp, "@test isapprox(raw.sites, r_S;"),
            r_S, raw.sites, test_tolerance(tp, "@test isapprox(raw.sites, r_S;"), onote),
        mkcase("P1-JULIA-ORDIPLOT-RAW-LOADINGS", "ordiplot raw loadings, rotate = false in Julia and R's default rotate = \"none\" ($(2p) values, column-major)",
            "$fxp [ordiplot.loadings]", "raw.species, as compared at " * cite(tp, "@test isapprox(raw.species, r_L;"),
            r_Lo, raw.species, test_tolerance(tp, "@test isapprox(raw.species, r_L;"), onote),
        mkcase("P1-JULIA-ORDIPLOT-LOGLIK", "log-likelihood of the fit ordiplot is drawn from",
            "$fxp [ordiplot.loglik]", "fit.logLik, as compared at " * cite(tp, "@test isapprox(fit.logLik, Float64(o[\"loglik\"]);") * "; fit as at $ofit",
            Float64(o["loglik"]), fo.logLik, test_tolerance(tp, "@test isapprox(fit.logLik, Float64(o[\"loglik\"]);"), onote),
    ]

    # ---- flag_unreliable_loadings ----
    f = fx["flag"]
    (f["converged"] === true && f["pd_hessian"] === true) || fail("flag: R fit did not converge with a PD Hessian")
    Yc = Y .- sum(Y; dims = 2) ./ n
    M = fill(NaN, p, 2)
    for pin in f["pins"]
        M[pin[1], pin[2]] = 0.0
    end
    fc = GMJ.fit_gaussian_gllvm(Yc; K = 2, lambda_constraint = M)
    fc.converged || fail("flag: Julia confirmatory fit did not converge")
    rows = GMJ.flag_unreliable_loadings(fc, Yc)
    rows_w = GMJ.flag_unreliable_loadings(fc, Yc; null_region = (-0.5, 0.5))
    all(r -> r.pd_hessian, rows) || fail("flag: Julia reduced information not positive definite")
    [r.pinned for r in rows] == Bool.(f["pinned"]) || fail("flag: pinned entries differ from R")
    ffit = cite(tp, "fit = fit_gaussian_gllvm(Yc; K = 2, lambda_constraint = M)")
    fcall = cite(tp, "rows = flag_unreliable_loadings(fit, Yc)")
    fcallw = cite(tp, "rows_w = flag_unreliable_loadings(fit, Yc; null_region = (-0.5, 0.5))")
    fnote = "$data_note R: a confirmatory fit value ~ 0 + trait + latent(0 + trait | unit, d = 2, unique = FALSE) with lambda_constraint = list(unit = M), M pinning Lambda[1,2] = 0 (the structural zero, stated) and Lambda[2,1] = 0, then flag_unreliable_loadings(fit) (default null_region c(-0.1, 0.1); its loading_ci() route is raw Wald from sd_report\$cov.fixed) and flag_unreliable_loadings(fit, null_region = c(-0.5, 0.5)). Julia: fit_gaussian_gllvm(Yc; K = 2, lambda_constraint = M) on Y centred by trait means (fit at $ffit), then flag_unreliable_loadings(fit, Yc) (at $fcall) and with null_region = (-0.5, 0.5) (at $fcallw); the intervals come from the observed information with the pinned loading removed. Julia's confirmatory fit is zero-mean: with complete balanced Gaussian data the trait-mean MLE is the sample mean and the observed information is block diagonal between means and covariance parameters at the optimum, so the log-likelihood and loading SEs equal those of R's fit with intercepts. Flags are coded 1 TRUE, 0 FALSE, -1 NA (pinned) and compared exactly. Julia's confirmatory refit stops at g_tol 1e-4, which sets the agreement of the estimates (about 7e-5); the closest non-pinned bound is 0.012 from a null-region edge. Limits: Gaussian only; raw Wald route only (R's default), not wald_asym, profile or the standardized scale."
    est, se = [r.estimate for r in rows], [r.se for r in rows]
    lo, hi = [r.lower for r in rows], [r.upper for r in rows]
    cf = [
        mkcase("P1-JULIA-FLAG-UNRELIABLE-LOADINGS-ESTIMATE", "loading estimates in flag_unreliable_loadings(fit) ($(2p) rows, trait fastest)",
            "$fxp [flag.estimate]", "row estimate, as compared at " * cite(tp, "@test isapprox(est, Float64.(f[\"estimate\"]);"),
            Float64.(f["estimate"]), est, test_tolerance(tp, "@test isapprox(est, Float64.(f[\"estimate\"]);"), fnote),
        mkcase("P1-JULIA-FLAG-UNRELIABLE-LOADINGS-SE", "raw Wald standard errors ($(2p) rows; 0 for pinned)",
            "$fxp [flag.se]", "row se, as compared at " * cite(tp, "@test isapprox(se, Float64.(f[\"se\"]);"),
            Float64.(f["se"]), se, test_tolerance(tp, "@test isapprox(se, Float64.(f[\"se\"]);"), fnote),
        mkcase("P1-JULIA-FLAG-UNRELIABLE-LOADINGS-LOWER", "lower 95% Wald bounds ($(2p) rows)",
            "$fxp [flag.lower]", "row lower, as compared at " * cite(tp, "@test isapprox(lo, Float64.(f[\"lower\"]);"),
            Float64.(f["lower"]), lo, test_tolerance(tp, "@test isapprox(lo, Float64.(f[\"lower\"]);"), fnote),
        mkcase("P1-JULIA-FLAG-UNRELIABLE-LOADINGS-UPPER", "upper 95% Wald bounds ($(2p) rows)",
            "$fxp [flag.upper]", "row upper, as compared at " * cite(tp, "@test isapprox(hi, Float64.(f[\"upper\"]);"),
            Float64.(f["upper"]), hi, test_tolerance(tp, "@test isapprox(hi, Float64.(f[\"upper\"]);"), fnote),
        mkcase_int("P1-JULIA-FLAG-UNRELIABLE-LOADINGS-FLAGS-DEFAULT", "unreliable flags at the default null region (-0.1, 0.1), codes 1 TRUE / 0 FALSE / -1 NA",
            "$fxp [flag.unreliable_default]", "row unreliable, as compared at " * cite(tp, "@test maximum(abs.(code_d .- r_code_d))"),
            rflagcode.(f["unreliable_default"]), flagcode.([r.unreliable for r in rows]),
            test_tolerance(tp, "@test maximum(abs.(code_d .- r_code_d))"), fnote),
        mkcase_int("P1-JULIA-FLAG-UNRELIABLE-LOADINGS-FLAGS-WIDE", "unreliable flags at null region (-0.5, 0.5), codes 1 TRUE / 0 FALSE / -1 NA",
            "$fxp [flag.unreliable_wide]", "row unreliable, as compared at " * cite(tp, "@test maximum(abs.(code_w .- r_code_w))"),
            rflagcode.(f["unreliable_wide"]), flagcode.([r.unreliable for r in rows_w]),
            test_tolerance(tp, "@test maximum(abs.(code_w .- r_code_w))"), fnote),
        mkcase("P1-JULIA-FLAG-UNRELIABLE-LOADINGS-LOGLIK", "log-likelihood of the confirmatory fit",
            "$fxp [flag.loglik]", "fit.logLik, as compared at " * cite(tp, "@test isapprox(fit.logLik, Float64(f[\"loglik\"]);") * "; fit as at $ffit",
            Float64(f["loglik"]), fc.logLik, test_tolerance(tp, "@test isapprox(fit.logLik, Float64(f[\"loglik\"]);"), fnote),
    ]
    fixtures = [fxp, dp]
    return Pair{String,Receipt}[
        "namespace-numeric/gllvmTMB_wide.json" => Receipt(["namespace/export/gllvmTMB_wide"], "itchyshin/GLLVModels.jl#684", fixtures, [tp], NOT_A_FIXTURE_PAIR, cw),
        "namespace-numeric/ordiplot.json" => Receipt(["namespace/S3method/ordiplot,gllvmTMB_multi"], "itchyshin/GLLVModels.jl#684", fixtures, [tp], NOT_A_FIXTURE_PAIR, co),
        "namespace-numeric/flag_unreliable_loadings.json" => Receipt(["namespace/export/flag_unreliable_loadings"], "itchyshin/GLLVModels.jl#684", fixtures, [tp], NOT_A_FIXTURE_PAIR, cf),
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
        receipts_namespace_gaussian_w1()
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
        isempty(probs) && (println("OK namespace Gaussian W1 receipts reproduce within $(CHECK_FRACTION) x tolerance"); return 0)
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
