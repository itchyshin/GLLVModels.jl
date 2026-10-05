#!/usr/bin/env julia
# Julia-side receipts for the covariance rows COV-ORD-LATENT-BARE, COV-ORD-LATENT-DEFAULT and
# COV-ORD-LATENT-COMMON (true-parity checker, tools/true_parity_check.mjs, clauses C1/C8).
#
# A standalone sibling of tools/true_parity_animal_scalar_receipt.jl; the helpers (Fail, findline,
# test_tolerance, cite, mkcase, the JSON writer and reader, Receipt, receipt_object,
# compare_receipt) are copied from it unchanged. Same rules:
#   * R values are copied from the tracked fixture test/fixtures/cov_ord_latent_p1.toml; nothing on
#     the R side is recomputed here.
#   * The Julia side repeats the computation of test/test_cov_ord_latent_p1.jl with the same inputs
#     and settings; the test file:line is recorded in `julia_source`.
#   * The tolerance is READ from the existing assertion in the test (file:line and the line's text
#     recorded). A difference above it aborts the run and nothing is written.
#   * src/, the test and the fixture are not modified.
# It writes three receipts, one per row, under
# docs/dev-log/core070/true-parity-latest/receipts/julia-twins/covariance-twins/. They are cited by
# the twin overlay of tools/core070_covariance_p1_receipts.py (--apply-twins).
#
# Usage (from the repository root)
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=. tools/true_parity_cov_ord_latent_receipt.jl
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=. tools/true_parity_cov_ord_latent_receipt.jl --check
# --check recomputes and compares with the committed receipts: fixture and test sha256, case ids, R
# values and tolerances identical; each Julia value within 1e-3 x tolerance + 100 eps of the
# committed one; abs_diff still within tolerance.

using GLLVModels, TOML, SHA, Statistics, LinearAlgebra
const GMJ = GLLVModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins"
const P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const GENERATOR = "tools/true_parity_cov_ord_latent_receipt.jl"
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
    "fresh by tools/true_parity_cov_ord_latent_receipt.jl, which repeats the computation of the cited twin " *
    "test with the same inputs and settings; R was not run and is not recomputed here."

# R's write.csv long data (site, trait, value) -> p x n_site response matrix; same parser as
# test/test_cov_ord_latent_p1.jl.
function load_data(path, trait_names, n_site)
    Y = fill(NaN, length(trait_names), n_site)
    open(joinpath(ROOT, path)) do io
        readline(io) == "\"site\",\"trait\",\"value\"" || fail("unexpected header in $path")
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            t = findfirst(==(strip(parts[2], '"')), trait_names)
            t === nothing && fail("unrecognised trait in $path")
            Y[t, parse(Int, parts[1])] = parse(Float64, parts[3])
        end
    end
    return Y
end

rowmatrix(rows) = reduce(vcat, [permutedims(Float64.(r)) for r in rows])

# R parameter vector (par_names order) -> Julia's order; same as test/test_cov_ord_latent_p1.jl.
function julia_point(r_par, names)
    pick(nm) = r_par[names .== nm]
    return vcat(pick("b_fix"), pick("theta_rr_B"), pick("theta_diag_B"), pick("log_sigma_eps"))
end

const ROWS = (
    ("bare", "COV-ORD-LATENT-BARE", false, false,
     "latent(0 + trait | site, unique = FALSE): Sigma_B = L L' (rank 1), residual SD estimated"),
    ("default", "COV-ORD-LATENT-DEFAULT", true, false,
     "latent(0 + trait | site): Sigma_B = L L' + diag(psi_1..psi_p) (R's auto per-trait unique diagonal), residual SD fixed by R at max(0.001 sd(y), 1e-6)"),
    ("common", "COV-ORD-LATENT-COMMON", true, true,
     "latent(0 + trait | site, common = TRUE): Sigma_B = L L' + psi I_p (R's common unique diagonal, one theta_diag_B), residual SD fixed by R at max(0.001 sd(y), 1e-6)"),
)

# ---------------------------------------------------------------------------------------------
# COV-ORD-LATENT-*   test/test_cov_ord_latent_p1.jl
# ---------------------------------------------------------------------------------------------
function receipts_cov_ord_latent()
    fxp = "test/fixtures/cov_ord_latent_p1.toml"
    tp = "test/test_cov_ord_latent_p1.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["gllvmtmb_commit"] == P1_SHA || fail("cov_ord_latent fixture is not pinned at P1")
    dp = "test/fixtures/" * fx["data_file"]
    sha_file(dp) == fx["data_sha256"] || fail("data csv drifted: $dp")
    p, n = Int(fx["p"]), Int(fx["n_site"])
    Y = load_data(dp, String.(fx["trait_names"]), n)
    all(isfinite, Y) || fail("data has missing cells")
    fitc = cite(tp, "fit = fit_gaussian_sources(Y; sources = [source], sigma_eps_fixed = sigma_fixed, g_tol = 1e-8, iterations = 2000)")
    srcc = cite(tp, "source = SourceCovariance(Matrix(1.0I, n, n); groups = 1:n, name = :site, mode = :latent, rank = 1, unique = uniq, common = common)")
    out = Pair{String,Receipt}[]
    for (key, short, uniq, common, model) in ROWS
        s = fx[key]
        s["case"] == short || fail("fixture section [$key] is not $short")
        (s["converged"] === true && s["pd_hessian"] === true) || fail("$short: the R fit did not converge with a PD Hessian; not a valid twin")
        fixed = s["sigma_eps_fixed"]
        fixed == uniq || fail("$short: R's residual-SD rule is not the one the twin assumes")
        sigma_fixed = fixed ? Float64(s["sigma_eps"]) : nothing
        source = GMJ.SourceCovariance(Matrix(1.0I, n, n); groups = 1:n, name = :site, mode = :latent, rank = 1, unique = uniq, common = common)
        fit = GMJ.fit_gaussian_sources(Y; sources = [source], sigma_eps_fixed = sigma_fixed, g_tol = 1e-8, iterations = 2000)
        (fit.converged && fit.hessian_positive_definite) || fail("$short: Julia fit did not converge with a PD Hessian")
        GMJ.dof(fit) == s["r_df"] || fail("$short: Julia free-parameter count $(GMJ.dof(fit)) differs from R's df $(s["r_df"])")
        r_beta = Float64.(s["beta"])
        U = only(fit.trait_covariances)
        r_Sigma = rowmatrix(s["trait_covariance"])
        r_point = vcat(r_beta, Float64.(s["lambda"]), Float64.(s["diag_logsd"]), fixed ? Float64[] : [log(Float64(s["sigma_eps"]))])
        nll_at_r = GMJ._gaussian_sources_nll(Y, [source], r_point; sigma_eps_fixed = sigma_fixed)
        probe = julia_point(Float64.(s["probe_par"]), String.(s["par_names"]))
        nll_at_probe = GMJ._gaussian_sources_nll(Y, [source], probe; sigma_eps_fixed = sigma_fixed)
        note = "Gaussian fit of value ~ 0 + trait + $(model), unit = site, on one simulated data set (n = $n sites, p = $p traits, one observation per trait and site; data CSV sha256 checked; R fitted the data read back from that CSV). R converged (nlminb code 0, rel.tol / sing.tol / x.tol 1e-12, max |gradient| $(round(s["r_gradient_max"]; sigdigits = 2))) with a positive-definite Hessian; R's logLik df = $(s["r_df"]) = Julia's free-parameter count. Julia fits the same covariance, I_n kron Sigma_B + sigma_eps^2 I, as fit_gaussian_sources with SourceCovariance(I_n; mode = :latent, rank = 1, unique = $uniq, common = $common) (source at $srcc), g_tol 1e-8$(fixed ? ", sigma_eps_fixed = R's fixed value" : ""). Model identity is checked directly: the Julia objective at R's optimum and at R's perturbed probe point (opt\$par + 0.05 sin(1:k), R's -obj\$fn there) reproduces R's value. Limits: Gaussian only; rank 1; one observation per site and trait; the Julia side is the native source fitter, not formula sugar for latent()."
        id(q) = "P1-JULIA-$short-$q"
        cs = Case[
            mkcase(id("LOGLIK"), "log-likelihood at the optimum",
                "$fxp [$key.loglik]", "fit.loglik, as compared at " * cite(tp, "@test isapprox(fit.loglik, Float64(s[\"loglik\"]);") * "; fit as at $fitc",
                Float64(s["loglik"]), fit.loglik, test_tolerance(tp, "@test isapprox(fit.loglik, Float64(s[\"loglik\"]);"), note),
            mkcase(id("BETA"), "trait intercepts b_fix ($(p) values)",
                "$fxp [$key.beta]", "fit.beta, as compared at " * cite(tp, "@test isapprox(fit.beta, r_beta;") * "; fit as at $fitc",
                r_beta, fit.beta, test_tolerance(tp, "@test isapprox(fit.beta, r_beta;"), note),
            mkcase(id("TRAIT-COVARIANCE"), "site-level trait covariance Sigma_B = Lambda Lambda' + diag(exp(2 theta_diag_B)) ($(p)x$(p), row-major)",
                "$fxp [$key.trait_covariance]", "only(fit.trait_covariances), as compared at " * cite(tp, "@test isapprox(U, r_Sigma;") * "; fit as at $fitc",
                vec(permutedims(r_Sigma)), vec(permutedims(U)), test_tolerance(tp, "@test isapprox(U, r_Sigma;"), note),
        ]
        fixed || push!(cs, mkcase(id("SIGMA-EPS"), "residual SD sigma_eps = exp(log_sigma_eps)",
                "$fxp [$key.sigma_eps]", "fit.sigma_eps, as compared at " * cite(tp, "@test isapprox(fit.sigma_eps, Float64(s[\"sigma_eps\"]);") * "; fit as at $fitc",
                Float64(s["sigma_eps"]), fit.sigma_eps, test_tolerance(tp, "@test isapprox(fit.sigma_eps, Float64(s[\"sigma_eps\"]);"), note))
        push!(cs, mkcase(id("OBJECTIVE-AT-R"), "Julia log-likelihood evaluated at R's optimum against R's log-likelihood",
                "$fxp [$key.loglik]", "-GLLVModels._gaussian_sources_nll(Y, [source], r_point; sigma_eps_fixed), as compared at " * cite(tp, "@test isapprox(-nll_at_r, Float64(s[\"loglik\"]);"),
                Float64(s["loglik"]), -nll_at_r, test_tolerance(tp, "@test isapprox(-nll_at_r, Float64(s[\"loglik\"]);"), note))
        push!(cs, mkcase(id("OBJECTIVE-AT-PROBE"), "Julia log-likelihood at R's off-optimum probe point against R's -obj\$fn there",
                "$fxp [$key.probe_loglik]", "-GLLVModels._gaussian_sources_nll(Y, [source], probe; sigma_eps_fixed), as compared at " * cite(tp, "@test isapprox(-nll_at_probe, Float64(s[\"probe_loglik\"]);"),
                Float64(s["probe_loglik"]), -nll_at_probe, test_tolerance(tp, "@test isapprox(-nll_at_probe, Float64(s[\"probe_loglik\"]);"), note))
        push!(out, "covariance-twins/$short.json" => Receipt(["covariance/$short"], "itchyshin/GLLVModels.jl#810", [fxp, dp], [tp], NOT_A_FIXTURE_PAIR, cs))
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
        receipts_cov_ord_latent()
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
        isempty(probs) && (println("OK cov_ord_latent receipts reproduces within $(CHECK_FRACTION) x tolerance"); return 0)
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
