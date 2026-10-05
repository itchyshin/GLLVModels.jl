#!/usr/bin/env julia
# Julia-side receipt for the namespace rows export/extract_Gamma and export/extract_coevolution_modules (true-parity checker,
# tools/true_parity_check.mjs, clauses C1/C8).
#
# A standalone sibling of tools/true_parity_julia_receipts.jl and
# tools/true_parity_animal_scalar_receipt.jl, kept in its own file so that they do not collide.
# It follows the same rules, and the helpers below (Fail, findline, test_tolerance, cite, mkcase,
# the JSON writer and reader, Receipt, receipt_object, compare_receipt) are copied from
# tools/true_parity_animal_scalar_receipt.jl unchanged:
#   * R values are copied from the tracked fixture test/fixtures/cross_lineage_p1.toml; nothing
#     on the R side is recomputed here.
#   * The Julia side repeats the computation of test/test_cross_lineage_p1.jl with the same
#     inputs and settings; the test file:line is recorded in `julia_source`.
#   * The tolerance is READ from the existing assertion in the test (file:line and the line's
#     text recorded). A difference above it aborts the run and nothing is written.
#   * src/, the test and the fixture are not modified.
# It writes two receipts under docs/dev-log/core070/true-parity-latest/receipts/julia-twins/namespace-numeric/:
# extract_Gamma.json and extract_coevolution_modules.json.
#
# Usage (from the repository root)
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_cross_lineage_receipt.jl
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. tools/true_parity_cross_lineage_receipt.jl --check
# --check recomputes and compares with the committed receipt: fixture and test sha256, case ids, R
# values and tolerances identical; each Julia value within 1e-3 x tolerance + 100 eps of the
# committed one; abs_diff still within tolerance.

using GLLVModels, TOML, SHA, Statistics, LinearAlgebra
const GMJ = GLLVModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins"
const P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const GENERATOR = "tools/true_parity_cross_lineage_receipt.jl"
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
    "fresh by tools/true_parity_cross_lineage_receipt.jl, which repeats the computation of the cited twin " *
    "test with the same inputs and settings; R was not run and is not recomputed here."

# R's write.csv long data (site, trait, species, value) -> p x n_site response matrix and the
# species index of each site; same parser as test/test_cross_lineage_p1.jl.
function load_data(path, trait_names, species, n_site)
    Y = fill(NaN, length(trait_names), n_site)
    grp = zeros(Int, n_site)
    open(joinpath(ROOT, path)) do io
        readline(io) == "\"site\",\"trait\",\"species\",\"value\"" || fail("unexpected header in $path")
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            site = parse(Int, parts[1])
            t = findfirst(==(strip(parts[2], '"')), trait_names)
            s = findfirst(==(strip(parts[3], '"')), species)
            (t === nothing || s === nothing) && fail("unrecognised trait or species in $path")
            Y[t, site] = parse(Float64, parts[4])
            grp[site] = s
        end
    end
    return Y, grp
end

# R's write.csv of the named n x n kernel (header = species names, in order).
function load_K(path, species)
    lines = filter(!isempty, readlines(joinpath(ROOT, path)))
    String.(strip.(split(lines[1], ","), '"')) == species || fail("unexpected header in $path")
    length(lines) == length(species) + 1 || fail("unexpected row count in $path")
    return reduce(vcat, [permutedims(parse.(Float64, split(l, ","))) for l in lines[2:end]])
end

# ---------------------------------------------------------------------------------------------
# extract_Gamma / extract_coevolution_modules   test/test_cross_lineage_p1.jl
# ---------------------------------------------------------------------------------------------
function receipts_cross_lineage()
    fxp = "test/fixtures/cross_lineage_p1.toml"
    tp = "test/test_cross_lineage_p1.jl"
    fx = TOML.parsefile(joinpath(ROOT, fxp))
    fx["gllvmtmb_commit"] == P1_SHA || fail("cross_lineage fixture is not pinned at P1")
    f, g, m = fx["fit"], fx["gamma"], fx["modules"]
    (f["converged"] === true && f["pd_hessian"] === true) || fail("the R fit did not converge with a PD Hessian; not a valid twin")
    dp, kp = "test/fixtures/" * f["data_file"], "test/fixtures/" * f["K_file"]
    sha_file(dp) == f["data_sha256"] || fail("data csv drifted: $dp")
    sha_file(kp) == f["K_sha256"] || fail("K csv drifted: $kp")
    tr, sp = String.(f["trait_names"]), String.(f["species"])
    Y, grp = load_data(dp, tr, sp, Int(f["n_site"]))
    K = load_K(kp, sp)
    K_eff = K + Float64(f["K_jitter"]) * I
    fit = GMJ.fit_kernel_latent_gllvm(Y, K_eff, grp, Int(f["d"]); name = :cross, g_tol = 1e-8, iterations = 2000)
    (fit.converged && fit.hessian_positive_definite) || fail("Julia fit did not converge with a PD Hessian")
    host, partner = tr[1:2], tr[3:4]
    (String.(g["row_traits"]) == host && String.(g["col_traits"]) == partner) || fail("fixture trait blocks are not host x partner")
    Gam = GMJ.extract_Gamma(fit; level = :cross, row_traits = host, col_traits = partner, trait_names = tr)
    rp, cp = String.(g["row_perm"]), String.(g["col_perm"])
    Gp = GMJ.extract_Gamma(fit; level = :cross, row_traits = rp, col_traits = cp, trait_names = tr)
    mo = GMJ.extract_coevolution_modules(fit; level = :cross, row_traits = host, col_traits = partner, trait_names = tr)
    mo1 = GMJ.extract_coevolution_modules(fit; level = :cross, row_traits = host, col_traits = partner, trait_names = tr, n_modules = 1)
    (mo.modules.module == String.(m["module"]) && mo.row_axes.trait == String.(m["row_axes_trait"]) &&
        mo.col_axes.trait == String.(m["col_axes_trait"]) && mo.row_axes.module == String.(m["row_axes_module"]) &&
        mo.col_axes.module == String.(m["col_axes_module"])) || fail("module or axis labels differ from R's tables")
    U_r = reshape(Float64.(m["row_axes_loading"]), 2, 2)
    U_j = reshape(mo.row_axes.loading, 2, 2)
    V_j = reshape(mo.col_axes.loading, 2, 2)
    sgn = [sign(dot(U_j[:, k], U_r[:, k])) for k in 1:2]
    all(s -> s != 0, sgn) || fail("an axis is orthogonal to R's; cannot align the sign")

    fitc = cite(tp, "fit = fit_kernel_latent_gllvm(Y, K_eff, grp, Int(f[\"d\"]); name = :cross, g_tol = 1e-8,")
    base = "Gaussian fit of value ~ 0 + trait + kernel_latent(species, K = K, d = 3, name = \"cross\", unique = FALSE) (R, unit = site) with K = make_cross_kernel(A_H, A_P, W, rho = 0.6) over 8 host and 8 partner species, 4 sites per species, 4 traits (h_size, h_defence, p_size, p_attack), 256 cells, complete data (data and K CSVs sha256 checked; R fitted the data and K read back from those CSVs). R converged (nlminb code 0, rel.tol / sing.tol / x.tol 1e-12, max |gradient| 4.4e-5) with a positive-definite Hessian, and fits K + 1e-8 I (jitter measured from tmb_data\$Ainv_phy_rr); Julia fits the same model, (P K_eff P') kron Lambda Lambda' + sigma_eps^2 I with Lambda 4 x 3, as fit_kernel_latent_gllvm(Y, K_eff, groups, 3; name = :cross) at $fitc, g_tol 1e-8; same optimum (log-likelihoods within 1.3e-11). Estimand orientation as R: both functions slice the 4 x 4 TRAIT covariance Lambda Lambda' of the named tier (R's extract_Sigma(fit, \"cross\", part = \"shared\")) by trait name, rows = row_traits and columns = col_traits in the order given; the Julia methods are on GaussianSourcesFit (src/cross_lineage_extract.jl). The older positional extract_Gamma(::GllvmFit), whose rows are the stacked species of the Hadamard fit, is a different estimand and is not used. Limits: Gaussian only; scale = \"effect\" not twinned (a GaussianSourcesFit does not record rho; the Julia method refuses :effect); complete data, not R's block-NA layout (fit_gaussian_sources does not accept missing responses); the Julia side is the native source fitter, not formula sugar for kernel_latent()."
    note_g = base * " Gamma entries are four distinct values (0.50, -0.23, -0.20, 0.34); the reordered call (rows h_defence, h_size; columns p_attack, h_size, p_size) mixes the lineages and pins name-based order and orientation."
    note_m = base * " With d = 3 and two traits per lineage the first singular value is 1 by construction (two planes in R^3 share a line) and the second, 0.297, is a genuine canonical correlation, so the axes are identified up to one joint sign per pair (U[:, k], V[:, k]); the Julia loadings recorded here are multiplied by that sign, chosen from the row axes (sign = $(Int.(sgn)), so the alignment is the identity on this fit), and the same sign is applied to the column axes."
    ll_case(id) = mkcase(id, "log-likelihood at the optimum (same-optimum check for the fit both estimands are read from)",
        "$fxp [fit.loglik]", "fit.loglik, as compared at " * cite(tp, "@test isapprox(fit.loglik, Float64(f[\"loglik\"]);") * "; fit as at $fitc",
        Float64(f["loglik"]), fit.loglik, test_tolerance(tp, "@test isapprox(fit.loglik, Float64(f[\"loglik\"]);"), base)
    cs_g = [
        ll_case("P1-JULIA-GAMMA-LOGLIK"),
        mkcase("P1-JULIA-GAMMA-HOST-PARTNER", "extract_Gamma(fit, level = \"cross\", row_traits = c(\"h_size\", \"h_defence\"), col_traits = c(\"p_size\", \"p_attack\")), 2 x 2, column-major",
            "$fxp [gamma.Gamma]", "GLLVModels.extract_Gamma(fit; level = :cross, row_traits, col_traits, trait_names), as compared at " * cite(tp, "@test isapprox(Gam, r_gam;") * "; fit as at $fitc",
            Float64.(g["Gamma"]), vec(Gam), test_tolerance(tp, "@test isapprox(Gam, r_gam;"), note_g),
        mkcase("P1-JULIA-GAMMA-REORDERED", "extract_Gamma(fit, level = \"cross\", row_traits = c(\"h_defence\", \"h_size\"), col_traits = c(\"p_attack\", \"h_size\", \"p_size\")), 2 x 3, column-major",
            "$fxp [gamma.Gamma_perm]", "GLLVModels.extract_Gamma(fit; level = :cross, row_traits = row_perm, col_traits = col_perm, trait_names), as compared at " * cite(tp, "@test isapprox(Gp, r_perm;") * "; fit as at $fitc",
            Float64.(g["Gamma_perm"]), vec(Gp), test_tolerance(tp, "@test isapprox(Gp, r_perm;"), note_g),
    ]
    callm = "GLLVModels.extract_coevolution_modules(fit; level = :cross, row_traits = host, col_traits = partner, trait_names)"
    callm1 = "GLLVModels.extract_coevolution_modules(fit; ..., n_modules = 1)"
    cs_m = [
        ll_case("P1-JULIA-COEVOLUTION-LOGLIK"),
        mkcase("P1-JULIA-COEVOLUTION-R", "extract_coevolution_modules(fit, \"cross\", host, partner)\$R, Sigma_row^(-1/2) Gamma Sigma_col^(-1/2), 2 x 2, column-major",
            "$fxp [modules.R]", "$callm.R, as compared at " * cite(tp, "@test isapprox(mo.R, r_R;") * "; fit as at $fitc",
            Float64.(m["R"]), vec(mo.R), test_tolerance(tp, "@test isapprox(mo.R, r_R;"), note_m),
        mkcase("P1-JULIA-COEVOLUTION-SINGULAR-VALUE", "\$modules\$singular_value (2 values)",
            "$fxp [modules.singular_value]", "$callm.modules.singular_value, as compared at " * cite(tp, "@test isapprox(mo.modules.singular_value, r_sv;"),
            Float64.(m["singular_value"]), mo.modules.singular_value, test_tolerance(tp, "@test isapprox(mo.modules.singular_value, r_sv;"), note_m),
        mkcase("P1-JULIA-COEVOLUTION-SQUARED-SHARE", "\$modules\$squared_share (2 values)",
            "$fxp [modules.squared_share]", "$callm.modules.squared_share, as compared at " * cite(tp, "@test isapprox(mo.modules.squared_share,"),
            Float64.(m["squared_share"]), mo.modules.squared_share, test_tolerance(tp, "@test isapprox(mo.modules.squared_share,"), note_m),
        mkcase("P1-JULIA-COEVOLUTION-ROW-AXES", "\$row_axes\$loading (trait fastest, module slowest; 4 values), sign-aligned per module",
            "$fxp [modules.row_axes_loading]", "$callm.row_axes.loading times the per-module sign, as compared at " * cite(tp, "@test isapprox(U_j .* sgn', U_r;"),
            Float64.(m["row_axes_loading"]), vec(U_j .* sgn'), test_tolerance(tp, "@test isapprox(U_j .* sgn', U_r;"), note_m),
        mkcase("P1-JULIA-COEVOLUTION-COL-AXES", "\$col_axes\$loading (trait fastest, module slowest; 4 values), with the row-axis signs",
            "$fxp [modules.col_axes_loading]", "$callm.col_axes.loading times the same per-module sign, as compared at " * cite(tp, "@test isapprox(V_j .* sgn', V_r;"),
            Float64.(m["col_axes_loading"]), vec(V_j .* sgn'), test_tolerance(tp, "@test isapprox(V_j .* sgn', V_r;"), note_m),
        mkcase("P1-JULIA-COEVOLUTION-N1-SQUARED-SHARE", "n_modules = 1: \$modules\$squared_share (1 value; the share of the full sum of squares)",
            "$fxp [modules.n1_squared_share]", "$callm1.modules.squared_share, as compared at " * cite(tp, "@test isapprox(mo1.modules.squared_share,"),
            Float64.(m["n1_squared_share"]), mo1.modules.squared_share, test_tolerance(tp, "@test isapprox(mo1.modules.squared_share,"), note_m),
        mkcase("P1-JULIA-COEVOLUTION-N1-ROW-AXES", "n_modules = 1: \$row_axes\$loading (2 values), sign-aligned",
            "$fxp [modules.n1_row_axes_loading]", "$callm1.row_axes.loading times the module-1 sign, as compared at " * cite(tp, "@test isapprox(mo1.row_axes.loading"),
            Float64.(m["n1_row_axes_loading"]), mo1.row_axes.loading .* sgn[1], test_tolerance(tp, "@test isapprox(mo1.row_axes.loading"), note_m),
        mkcase("P1-JULIA-COEVOLUTION-N1-COL-AXES", "n_modules = 1: \$col_axes\$loading (2 values), with the module-1 sign",
            "$fxp [modules.n1_col_axes_loading]", "$callm1.col_axes.loading times the module-1 sign, as compared at " * cite(tp, "@test isapprox(mo1.col_axes.loading"),
            Float64.(m["n1_col_axes_loading"]), mo1.col_axes.loading .* sgn[1], test_tolerance(tp, "@test isapprox(mo1.col_axes.loading"), note_m),
    ]
    fixtures = [fxp, dp, kp]
    return Pair{String,Receipt}[
        "namespace-numeric/extract_Gamma.json" => Receipt(["namespace/export/extract_Gamma"], "itchyshin/GLLVModels.jl#684", fixtures, [tp], NOT_A_FIXTURE_PAIR, cs_g),
        "namespace-numeric/extract_coevolution_modules.json" => Receipt(["namespace/export/extract_coevolution_modules"], "itchyshin/GLLVModels.jl#684", fixtures, [tp], NOT_A_FIXTURE_PAIR, cs_m),
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
        receipts_cross_lineage()
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
        isempty(probs) && (println("OK cross_lineage receipts reproduce within $(CHECK_FRACTION) x tolerance"); return 0)
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
