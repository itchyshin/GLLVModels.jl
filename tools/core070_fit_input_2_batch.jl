# Inner executor for the "fit-input-2" (M2) batch (13 EXECUTABLE_NOW cases
# across 7 of the 9 fit-input rows; GAUSS-COMMON and POISSON-DEFAULT, plus
# GAUSS-LOADINGS' formula-interface case, are NEEDS_NEW_JULIA_SURFACE and are
# not attempted here -- see docs/dev-log/core070/fit-input-2-batch-contract.json).
#
# Pure-Julia consumer, mirroring tools/core070_namespace_2_batch.jl: reads a
# JSON oracle file the paired R runner (tools/core070_fit_input_2_batch.R)
# writes BEFORE invoking this script (that R process already has the frozen
# gllvmTMB library loaded and does 100% of the live R-side fitting itself),
# refits the identical data natively via direct `using GLLVModels` module calls
# AND via the `GLLVModels.gllvm(@formula(...), ...)` front door, for each row's
# NATIVE-MODEL and FORMULA-INTERFACE case. No RCall, no parity-runner
# include, no R of any kind runs in this process.
#
# argv:
#   ARGS[1]  destination path for the JSON results file (parent dir need not
#            exist yet; this script mkpath()s it). Must not already exist.
#
# Env vars required (set by the outer R runner before invoking this script):
#   CORE070_FIT_INPUT_2_R_ORACLE = <path to the R-written oracle JSON>
#
# Invocation:
#   julia --project=. tools/core070_fit_input_2_batch.jl <out.json>

using GLLVModels
using StatsModels
using Tables
using Distributions: Normal, Binomial

# ---------------------------------------------------------------------------
# Minimal JSON reader/writer (no external dependency; mirrors the existing
# repo convention in tools/core070_namespace_2_batch.jl).
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
to_json(x::AbstractString) = "\"" * json_escape(x) * "\""
to_json(x::Integer) = string(x)
to_json(x::AbstractFloat) = isfinite(x) ? repr(x) : "null"
to_json(x::AbstractVector) = "[" * join(to_json.(x), ",") * "]"
to_json(x::AbstractDict) = "{" * join(("\"$(json_escape(string(k)))\":" * to_json(v) for (k, v) in x), ",") * "}"

length(ARGS) == 1 || error("usage: julia tools/core070_fit_input_2_batch.jl <out.json>")
out_path = ARGS[1]
isfile(out_path) && error("destination already exists: $out_path")
mkpath(dirname(out_path))

@assert realpath(Base.pkgdir(GLLVModels)) == realpath(pwd()) "must run from the GLLVModels.jl package root"

oracle_path = get(ENV, "CORE070_FIT_INPUT_2_R_ORACLE", "")
isempty(oracle_path) &&
    error("CORE070_FIT_INPUT_2_R_ORACLE is required; refusing to silently default it")
isfile(oracle_path) ||
    error("CORE070_FIT_INPUT_2_R_ORACLE points to a nonexistent file: $oracle_path")

oracle = json_read(oracle_path)

# Tolerance calibration: see docs/dev-log/core070/fit-input-2-batch-contract.json
# `tolerances.justification` -- these compare INDEPENDENT optimizations (fresh
# R optim() vs fresh Julia Optim.LBFGS() runs), the established paired-fit
# precedent is 1e-4 absolute logLik/coef, not the tighter 1e-6..1e-8 band used
# for same-point identity checks elsewhere in core070.
tol_loglik = 1e-4
tol_coef = 1e-4
cases = Dict{String, Any}()

# jsonlite writes an R matrix as a row-major nested array of rows (one inner
# array per row, dimnames dropped); rebuild the Float64 matrix accordingly.
function jl_matrix(rows)
    nr = length(rows)
    nc = length(rows[1])
    M = Matrix{Float64}(undef, nr, nc)
    for i in 1:nr, j in 1:nc
        M[i, j] = Float64(rows[i][j])
    end
    M
end

coef_delta(a, b) = length(a) == length(b) ? maximum(abs.(a .- b)) : Inf

# ---------------------------------------------------------------------------
# Row 1: INPUT-GAUSS-DEFAULT (native + formula)
# ---------------------------------------------------------------------------
gd = oracle["gauss_default"]
p_g = Int(gd["p"]); n_g = Int(gd["n"]); K_g = Int(gd["K"])
Y_g = reshape(Float64.(gd["y"]), p_g, n_g)
c_default = Float64(gd["fixed_residual_sd"])
r_loglik_default = Float64(gd["loglik"]); r_coef_default = Float64.(gd["coef"])

fit_gd_native = GLLVModels.fit_gaussian_pervar_gllvm(Y_g; K = K_g, fixed_residual_sd = c_default)
d1 = abs(fit_gd_native.loglik - r_loglik_default)
d2 = coef_delta(fit_gd_native.β, r_coef_default)
cases["CORE070-FIT-INPUT-GAUSS-DEFAULT-NATIVE-MODEL"] = Dict(
    "pass" => d1 <= tol_loglik && d2 <= tol_coef, "loglik_delta" => d1, "coef_delta" => d2,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_gd_native.loglik, "julia_coef" => collect(Float64, fit_gd_native.β))

data_g = (dummy = zeros(n_g),)
fit_gd_formula = GLLVModels.gllvm(@formula(y ~ 1), Y_g, data_g;
    family = Normal(), K = K_g, pervar = true, fixed_residual_sd = c_default)
d3 = abs(fit_gd_formula.loglik - r_loglik_default)
d4 = coef_delta(fit_gd_formula.β, r_coef_default)
cases["CORE070-FIT-INPUT-GAUSS-DEFAULT-FORMULA-INTERFACE"] = Dict(
    "pass" => d3 <= tol_loglik && d4 <= tol_coef, "loglik_delta" => d3, "coef_delta" => d4,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_gd_formula.loglik, "julia_coef" => collect(Float64, fit_gd_formula.β))

# ---------------------------------------------------------------------------
# Row 2: INPUT-GAUSS-LOADINGS (native only; formula is NEEDS_NEW_JULIA_SURFACE)
# ---------------------------------------------------------------------------
gl = oracle["gauss_loadings"]
r_loglik_loadings = Float64(gl["loglik"]); r_coef_loadings = Float64.(gl["coef"])
X_g = zeros(p_g, n_g, p_g)
for j in 1:p_g
    X_g[j, :, j] .= 1
end
fit_gl_native = GLLVModels.fit_gaussian_gllvm(Y_g; K = K_g, X = X_g)
d5 = abs(fit_gl_native.logLik - r_loglik_loadings)
d6 = coef_delta(fit_gl_native.pars.β, r_coef_loadings)
cases["CORE070-FIT-INPUT-GAUSS-LOADINGS-NATIVE-MODEL"] = Dict(
    "pass" => d5 <= tol_loglik && d6 <= tol_coef, "loglik_delta" => d5, "coef_delta" => d6,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_gl_native.logLik, "julia_coef" => collect(Float64, fit_gl_native.pars.β))

# ---------------------------------------------------------------------------
# Row 3: INPUT-BINOMIAL-DEFAULT (native + formula)
# ---------------------------------------------------------------------------
bd = oracle["binomial_default"]
p_b = Int(bd["p"]); n_b = Int(bd["n"]); K_b = Int(bd["K"])
Y_b = reshape(Int.(bd["y"]), p_b, n_b)
r_loglik_binom = Float64(bd["loglik"]); r_coef_binom = Float64.(bd["coef"])

fit_bd_native = GLLVModels.fit_binomial_gllvm(Y_b; K = K_b)
d7 = abs(fit_bd_native.loglik - r_loglik_binom)
d8 = coef_delta(fit_bd_native.β, r_coef_binom)
cases["CORE070-FIT-INPUT-BINOMIAL-DEFAULT-NATIVE-MODEL"] = Dict(
    "pass" => d7 <= tol_loglik && d8 <= tol_coef, "loglik_delta" => d7, "coef_delta" => d8,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_bd_native.loglik, "julia_coef" => collect(Float64, fit_bd_native.β))

data_b = (dummy = zeros(n_b),)
fit_bd_formula = GLLVModels.gllvm(@formula(y ~ 1), Y_b, data_b; family = Binomial(), K = K_b)
d9 = abs(fit_bd_formula.loglik - r_loglik_binom)
d10 = coef_delta(fit_bd_formula.β, r_coef_binom)
cases["CORE070-FIT-INPUT-BINOMIAL-DEFAULT-FORMULA-INTERFACE"] = Dict(
    "pass" => d9 <= tol_loglik && d10 <= tol_coef, "loglik_delta" => d9, "coef_delta" => d10,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_bd_formula.loglik, "julia_coef" => collect(Float64, fit_bd_formula.β))

# ---------------------------------------------------------------------------
# Row 4: INPUT-ANIMAL-LATENT (native + formula)
# ---------------------------------------------------------------------------
al = oracle["animal_latent"]
p_a = Int(al["p"]); n_ids = Int(al["n_ids"])
A_animal = jl_matrix(al["A"])
Y_a = reshape(Float64.(al["y"]), p_a, n_ids)
r_loglik_animal = Float64(al["loglik"]); r_coef_animal = Float64.(al["coef"])

src_animal = GLLVModels.SourceCovariance(A_animal; groups = 1:n_ids, name = :animal)
fit_al_native = GLLVModels.fit_gaussian_sources(Y_a; sources = [src_animal])
d11 = abs(fit_al_native.loglik - r_loglik_animal)
d12 = coef_delta(fit_al_native.beta, r_coef_animal)
cases["CORE070-FIT-INPUT-ANIMAL-LATENT-NATIVE-MODEL"] = Dict(
    "pass" => d11 <= tol_loglik && d12 <= tol_coef, "loglik_delta" => d11, "coef_delta" => d12,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_al_native.loglik, "julia_coef" => collect(Float64, fit_al_native.beta))

data_a = (dummy = zeros(n_ids),)
fit_al_formula = GLLVModels.gllvm(@formula(y ~ 1), Y_a, data_a; family = Normal(), sources = [src_animal])
d13 = abs(fit_al_formula.loglik - r_loglik_animal)
d14 = coef_delta(fit_al_formula.beta, r_coef_animal)
cases["CORE070-FIT-INPUT-ANIMAL-LATENT-FORMULA-INTERFACE"] = Dict(
    "pass" => d13 <= tol_loglik && d14 <= tol_coef, "loglik_delta" => d13, "coef_delta" => d14,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_al_formula.loglik, "julia_coef" => collect(Float64, fit_al_formula.beta))

# ---------------------------------------------------------------------------
# Row 5: INPUT-KERNEL-ONE (native + formula)
# ---------------------------------------------------------------------------
k1 = oracle["kernel_one"]
p_k = Int(k1["p"]); n_units = Int(k1["n_units"])
A_kernel = jl_matrix(k1["A"])
Y_k1 = reshape(Float64.(k1["y"]), p_k, n_units)
r_loglik_k1 = Float64(k1["loglik"]); r_coef_k1 = Float64.(k1["coef"])

src_a = GLLVModels.SourceCovariance(A_kernel; groups = 1:n_units, name = :a)
fit_k1_native = GLLVModels.fit_gaussian_sources(Y_k1; sources = [src_a])
d15 = abs(fit_k1_native.loglik - r_loglik_k1)
d16 = coef_delta(fit_k1_native.beta, r_coef_k1)
cases["CORE070-FIT-INPUT-KERNEL-ONE-NATIVE-MODEL"] = Dict(
    "pass" => d15 <= tol_loglik && d16 <= tol_coef, "loglik_delta" => d15, "coef_delta" => d16,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_k1_native.loglik, "julia_coef" => collect(Float64, fit_k1_native.beta))

data_k1 = (dummy = zeros(n_units),)
fit_k1_formula = GLLVModels.gllvm(@formula(y ~ 1), Y_k1, data_k1; family = Normal(), sources = [src_a])
d17 = abs(fit_k1_formula.loglik - r_loglik_k1)
d18 = coef_delta(fit_k1_formula.beta, r_coef_k1)
cases["CORE070-FIT-INPUT-KERNEL-ONE-FORMULA-INTERFACE"] = Dict(
    "pass" => d17 <= tol_loglik && d18 <= tol_coef, "loglik_delta" => d17, "coef_delta" => d18,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_k1_formula.loglik, "julia_coef" => collect(Float64, fit_k1_formula.beta))

# ---------------------------------------------------------------------------
# Row 6/7: INPUT-KERNEL-TWO and INPUT-KERNEL-TWO-AUTO (native + formula each).
# KERNEL-TWO-AUTO uses the SAME unique=false sources as KERNEL-TWO -- see
# the contract's critical_finding_kernel_two_auto: gllvmTMB silently drops
# unique=TRUE in the multi-kernel combination, so the correct Julia
# comparand for the "auto" row is the plain two-source model, not
# SourceCovariance(...; unique=true).
# ---------------------------------------------------------------------------
k2 = oracle["kernel_two"]
B_kernel = jl_matrix(k2["B"])
Y_k2 = reshape(Float64.(k2["y"]), p_k, n_units)
r_loglik_k2 = Float64(k2["loglik"]); r_coef_k2 = Float64.(k2["coef"])

src_a2 = GLLVModels.SourceCovariance(A_kernel; groups = 1:n_units, name = :a)
src_b2 = GLLVModels.SourceCovariance(B_kernel; groups = 1:n_units, name = :b)
fit_k2_native = GLLVModels.fit_gaussian_sources(Y_k2; sources = [src_a2, src_b2])
d19 = abs(fit_k2_native.loglik - r_loglik_k2)
d20 = coef_delta(fit_k2_native.beta, r_coef_k2)
cases["CORE070-FIT-INPUT-KERNEL-TWO-NATIVE-MODEL"] = Dict(
    "pass" => d19 <= tol_loglik && d20 <= tol_coef, "loglik_delta" => d19, "coef_delta" => d20,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_k2_native.loglik, "julia_coef" => collect(Float64, fit_k2_native.beta))

data_k2 = (dummy = zeros(n_units),)
fit_k2_formula = GLLVModels.gllvm(@formula(y ~ 1), Y_k2, data_k2; family = Normal(), sources = [src_a2, src_b2])
d21 = abs(fit_k2_formula.loglik - r_loglik_k2)
d22 = coef_delta(fit_k2_formula.beta, r_coef_k2)
cases["CORE070-FIT-INPUT-KERNEL-TWO-FORMULA-INTERFACE"] = Dict(
    "pass" => d21 <= tol_loglik && d22 <= tol_coef, "loglik_delta" => d21, "coef_delta" => d22,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_k2_formula.loglik, "julia_coef" => collect(Float64, fit_k2_formula.beta))

k2a = oracle["kernel_two_auto"]
r_loglik_k2a = Float64(k2a["loglik"]); r_coef_k2a = Float64.(k2a["coef"])
r_matches_k2 = k2a["matches_kernel_two"] === true

src_a3 = GLLVModels.SourceCovariance(A_kernel; groups = 1:n_units, name = :a)
src_b3 = GLLVModels.SourceCovariance(B_kernel; groups = 1:n_units, name = :b)
fit_k2a_native = GLLVModels.fit_gaussian_sources(Y_k2; sources = [src_a3, src_b3])
d23 = abs(fit_k2a_native.loglik - r_loglik_k2a)
d24 = coef_delta(fit_k2a_native.beta, r_coef_k2a)
cases["CORE070-FIT-INPUT-KERNEL-TWO-AUTO-NATIVE-MODEL"] = Dict(
    "pass" => r_matches_k2 && d23 <= tol_loglik && d24 <= tol_coef,
    "loglik_delta" => d23, "coef_delta" => d24,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_k2a_native.loglik, "julia_coef" => collect(Float64, fit_k2a_native.beta), "r_matches_kernel_two" => r_matches_k2)

fit_k2a_formula = GLLVModels.gllvm(@formula(y ~ 1), Y_k2, data_k2; family = Normal(), sources = [src_a3, src_b3])
d25 = abs(fit_k2a_formula.loglik - r_loglik_k2a)
d26 = coef_delta(fit_k2a_formula.beta, r_coef_k2a)
cases["CORE070-FIT-INPUT-KERNEL-TWO-AUTO-FORMULA-INTERFACE"] = Dict(
    "pass" => r_matches_k2 && d25 <= tol_loglik && d26 <= tol_coef,
    "loglik_delta" => d25, "coef_delta" => d26,
    # The Julia values themselves, so the receipt tool recomputes |R - Julia| instead of trusting it.
    "julia_loglik" => fit_k2a_formula.loglik, "julia_coef" => collect(Float64, fit_k2a_formula.beta), "r_matches_kernel_two" => r_matches_k2)

# ---------------------------------------------------------------------------
# Negative controls: deliberately-wrong comparisons that MUST fail.
# ---------------------------------------------------------------------------
neg_coef_delta = coef_delta(fit_gd_native.β .+ 1.0, r_coef_default)
neg_loglik_delta = abs((fit_gd_native.loglik + 1.0) - r_loglik_default)

# NEG-KERNEL-TWO-AUTO-WRONG-MODEL: deliberately fit with unique=true (the
# wrong-model mapping the contract explicitly rejects) and confirm it does
# NOT match the KERNEL-TWO-AUTO R oracle within tolerance.
src_a_wrong = GLLVModels.SourceCovariance(A_kernel; groups = 1:n_units, name = :a, unique = true)
src_b_wrong = GLLVModels.SourceCovariance(B_kernel; groups = 1:n_units, name = :b, unique = true)
fit_k2a_wrong = GLLVModels.fit_gaussian_sources(Y_k2; sources = [src_a_wrong, src_b_wrong])
neg_kernel_wrong_delta = abs(fit_k2a_wrong.loglik - r_loglik_k2a)

# NEG-BINOMIAL-SWAPPED-Y: compare the binomial fit's logLik against the
# mismatched GAUSS-DEFAULT R oracle.
neg_swapped_delta = abs(fit_bd_native.loglik - r_loglik_default)

negative_controls = Dict(
    "NEG-COEF-SHIFTED" => Dict(
        "behaved" => neg_coef_delta > tol_coef, "delta" => neg_coef_delta),
    "NEG-LOGLIK-SHIFTED" => Dict(
        "behaved" => neg_loglik_delta > tol_loglik, "delta" => neg_loglik_delta),
    "NEG-KERNEL-TWO-AUTO-WRONG-MODEL" => Dict(
        "behaved" => neg_kernel_wrong_delta > tol_loglik, "delta" => neg_kernel_wrong_delta),
    "NEG-BINOMIAL-SWAPPED-Y" => Dict(
        "behaved" => neg_swapped_delta > tol_loglik, "delta" => neg_swapped_delta),
)

all_positive_pass = all(v["pass"] for v in values(cases))
negatives_behaved = all(v["behaved"] for v in values(negative_controls))
overall_ok = all_positive_pass && negatives_behaved && length(cases) == 13

report = Dict(
    "status" => overall_ok ? "PASS" : "FAIL",
    "area" => "fit-input-2",
    "scope" => "CORE070_FIT_INPUT_2_BATCH",
    "case_count" => length(cases),
    "negative_control_count" => length(negative_controls),
    "all_positive_pass" => all_positive_pass,
    "negative_controls_behaved_as_expected" => negatives_behaved,
    "all_checks" => overall_ok,
    "cases" => cases,
    "negative_controls" => negative_controls,
    "julia_version" => string(VERSION),
)
open(out_path, "w") do io
    print(io, to_json(report))
end
println("CORE070_FIT_INPUT_2_BATCH_RESULT ",
        overall_ok ? "PASS" : "FAIL",
        " positive=", all_positive_pass, " negatives=", negatives_behaved,
        " case_count=", length(cases))
overall_ok || exit(1)
