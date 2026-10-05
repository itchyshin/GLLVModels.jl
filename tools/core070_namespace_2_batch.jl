# Inner executor for the "namespace-2" manifest-area batch (9 planned
# EXECUTABLE_NOW cases; 6 of them decided here, the remaining 3
# -- gllvmTMB_wide native-fit consistency and the two lognormal/
# truncated_poisson rejection-path cases -- are decided entirely R-side by
# the paired outer runner and merged into the same receipt there).
#
# Pure-Julia consumer, mirroring tools/core070_postfit_policy_batch.jl: reads
# a JSON oracle file the paired R runner (tools/core070_namespace_2_batch.R)
# writes BEFORE invoking this script (that R process already has the frozen
# gllvmTMB library loaded and does 100% of the live R-side fitting itself),
# refits/dispatches the identical data natively via direct `using GLLVModels`
# module calls only. No RCall, no parity-runner include, no R of any kind
# runs in this process.
#
# argv:
#   ARGS[1]  destination path for the JSON results file (parent dir need not
#            exist yet; this script mkpath()s it). Must not already exist.
#
# Env vars required (set by the outer R runner before invoking this script;
# never defaulted silently here so a misconfigured environment fails loudly
# rather than quietly running against stale or missing oracle data):
#   CORE070_NAMESPACE_2_R_ORACLE = <path to the R-written oracle JSON>
#
# Invocation:
#   julia --project=. tools/core070_namespace_2_batch.jl <out.json>

using GLLVModels

# ---------------------------------------------------------------------------
# Minimal JSON reader/writer (no external dependency; mirrors the existing
# repo convention in tools/core070_postfit_policy_batch.jl).
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
        pos[] += 1 # opening quote
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

length(ARGS) == 1 || error("usage: julia tools/core070_namespace_2_batch.jl <out.json>")
out_path = ARGS[1]
isfile(out_path) && error("destination already exists: $out_path")
mkpath(dirname(out_path))

@assert realpath(Base.pkgdir(GLLVModels)) == realpath(pwd()) "must run from the GLLVModels.jl package root"

oracle_path = get(ENV, "CORE070_NAMESPACE_2_R_ORACLE", "")
isempty(oracle_path) &&
    error("CORE070_NAMESPACE_2_R_ORACLE is required; refusing to silently default it")
isfile(oracle_path) ||
    error("CORE070_NAMESPACE_2_R_ORACLE points to a nonexistent file: $oracle_path")

oracle = json_read(oracle_path)

# Tolerance calibration (2026-09-01): these cases compare INDEPENDENT
# optimizations (R fit vs Julia bridge fit from their own starts), not a
# same-point objective evaluation. The established paired-fit precedent in
# tests/testthat/test-julia-bridge.R (gllvmTMB lane) is 1e-4 absolute logLik
# (lognormal/truncated_poisson) — same-point identity checks keep 1e-6..1e-8
# elsewhere. 1e-6 here was a miscalibration of a not-yet-accepted contract,
# observed failing at delta 4.52e-6 with all semantic checks true.
tol = Dict("loglik_delta" => 1e-4, "coef_delta" => 1e-4)
cases = Dict{String, Any}()

# ---------------------------------------------------------------------------
# Case 1: CORE070-NAMESPACE2-GLLVMTMB-NATIVE-FIT
# ---------------------------------------------------------------------------
g = oracle["gaussian"]
p_g = Int(g["p"]); n_g = Int(g["n"]); K_g = Int(g["K"])
y_g_flat = Float64.(g["y"])
length(y_g_flat) == p_g * n_g || error("gaussian y length mismatch")
Y_g = reshape(y_g_flat, p_g, n_g)
X_g = zeros(p_g, n_g, p_g)
for j in 1:p_g
    X_g[j, :, j] .= 1
end

fit_g = fit_gaussian_gllvm(Y_g; K = K_g, X = X_g)
j_coef_g = GLLVModels.StatsAPI.coef(fit_g)
j_loglik_g = fit_g.logLik
r_coef_g = Float64.(g["coef"])
r_loglik_g = Float64(g["loglik"])

coef_delta = length(j_coef_g) == length(r_coef_g) ? maximum(abs.(j_coef_g .- r_coef_g)) : Inf
loglik_delta_g = abs(j_loglik_g - r_loglik_g)
cases["CORE070-NAMESPACE2-GLLVMTMB-NATIVE-FIT"] = Dict(
    "pass" => coef_delta <= tol["coef_delta"] && loglik_delta_g <= tol["loglik_delta"],
    "coef_delta" => coef_delta, "loglik_delta" => loglik_delta_g,
    # Raw values (additive, P1 re-measure): lets a receipt writer recompute both diffs.
    "julia_coef" => collect(Float64, j_coef_g), "julia_loglik" => Float64(j_loglik_g),
    "r_coef" => r_coef_g, "r_loglik" => r_loglik_g,
)

# ---------------------------------------------------------------------------
# Case 2: CORE070-NAMESPACE2-GLLVM-JULIA-SETUP-BRIDGE-ADMISSION
# ---------------------------------------------------------------------------
pkgdir_matches_pwd = realpath(Base.pkgdir(GLLVModels)) == realpath(pwd())
julia_version_string = string(VERSION)
setup_precondition_r = oracle["setup_precondition_r"] === true
cases["CORE070-NAMESPACE2-GLLVM-JULIA-SETUP-BRIDGE-ADMISSION"] = Dict(
    "pass" => pkgdir_matches_pwd && setup_precondition_r && !isempty(julia_version_string),
    "pkgdir_matches_pwd" => pkgdir_matches_pwd,
    "setup_precondition_r" => setup_precondition_r,
    "julia_version" => julia_version_string,
)

# ---------------------------------------------------------------------------
# Case 3: CORE070-NAMESPACE2-GLLVM-JULIA-FIT-BRIDGE-ADMISSION
# ---------------------------------------------------------------------------
gate = oracle["gate"]
gate_gaussian_ok = gate["gaussian"]["ok"] === true && gate["gaussian"]["key"] == "gaussian"
bridge_key_gaussian = GLLVModels._bridge_family_key("gaussian") == "gaussian"

br_g = GLLVModels.bridge_fit(; y = Y_g, family = "gaussian", d = K_g, X = X_g)
bridge_loglik_delta = abs(br_g.loglik - r_loglik_g)
cases["CORE070-NAMESPACE2-GLLVM-JULIA-FIT-BRIDGE-ADMISSION"] = Dict(
    "pass" => gate_gaussian_ok && bridge_key_gaussian && bridge_loglik_delta <= tol["loglik_delta"],
    "gate_gaussian_ok" => gate_gaussian_ok, "bridge_key_gaussian" => bridge_key_gaussian,
    "bridge_loglik_delta" => bridge_loglik_delta,
)

# ---------------------------------------------------------------------------
# Cases 4-5: negative-binomial family bridge (nbinom1 -> "nb1", nbinom2 -> "negbinomial")
# ---------------------------------------------------------------------------
nb = oracle["nb"]
p_nb = Int(nb["p"]); n_nb = Int(nb["n"]); K_nb = Int(nb["K"])
y_nb_flat = Float64.(nb["y"])
length(y_nb_flat) == p_nb * n_nb || error("nb y length mismatch")
Y_nb = reshape(y_nb_flat, p_nb, n_nb)
X_nb = zeros(p_nb, n_nb, p_nb)
for j in 1:p_nb
    X_nb[j, :, j] .= 1
end
r_loglik_nb1 = Float64(nb["loglik_nbinom1"])
r_loglik_nb2 = Float64(nb["loglik_nbinom2"])

gate_nb1_ok = gate["nbinom1"]["ok"] === true && gate["nbinom1"]["key"] == "nb1"
bridge_key_nb1 = GLLVModels._bridge_family_key("nb1") == "nb1"
br_nb1 = GLLVModels.bridge_fit(; y = Y_nb, family = "nb1", d = K_nb, X = X_nb)
nb1_loglik_delta = abs(br_nb1.loglik - r_loglik_nb1)
cases["CORE070-NAMESPACE2-NBINOM1-FAMILY-BRIDGE"] = Dict(
    "pass" => gate_nb1_ok && bridge_key_nb1 && nb1_loglik_delta <= tol["loglik_delta"],
    "gate_nb1_ok" => gate_nb1_ok, "bridge_key_nb1" => bridge_key_nb1,
    "loglik_delta" => nb1_loglik_delta,
)

gate_nb2_ok = gate["nbinom2"]["ok"] === true && gate["nbinom2"]["key"] == "negbinomial"
bridge_key_nb2 = GLLVModels._bridge_family_key("nbinom2") == "negbinomial"
br_nb2 = GLLVModels.bridge_fit(; y = Y_nb, family = "negbinomial", d = K_nb, X = X_nb)
nb2_loglik_delta = abs(br_nb2.loglik - r_loglik_nb2)
cases["CORE070-NAMESPACE2-NBINOM2-FAMILY-BRIDGE"] = Dict(
    "pass" => gate_nb2_ok && bridge_key_nb2 && nb2_loglik_delta <= tol["loglik_delta"],
    "gate_nb2_ok" => gate_nb2_ok, "bridge_key_nb2" => bridge_key_nb2,
    "loglik_delta" => nb2_loglik_delta,
)

# ---------------------------------------------------------------------------
# Case 6: CORE070-NAMESPACE2-ORDINAL-PROBIT-FAMILY-BRIDGE
# ---------------------------------------------------------------------------
ord = oracle["ordinal"]
p_ord = Int(ord["p"]); n_ord = Int(ord["n"]); K_ord = Int(ord["K"])
y_ord_flat = Float64.(ord["y"])
length(y_ord_flat) == p_ord * n_ord || error("ordinal y length mismatch")
Y_ord = reshape(y_ord_flat, p_ord, n_ord)
r_loglik_ord = Float64(ord["loglik"])

gate_ord_ok = gate["ordinal_probit"]["ok"] === true && gate["ordinal_probit"]["key"] == "ordinal_probit"
bridge_key_ord = GLLVModels._bridge_family_key("ordinal_probit") == "ordinal_probit"
br_ord = GLLVModels.bridge_fit(; y = Y_ord, family = "ordinal_probit", d = K_ord)
ord_loglik_delta = abs(br_ord.loglik - r_loglik_ord)
cases["CORE070-NAMESPACE2-ORDINAL-PROBIT-FAMILY-BRIDGE"] = Dict(
    "pass" => gate_ord_ok && bridge_key_ord && ord_loglik_delta <= tol["loglik_delta"],
    "gate_ord_ok" => gate_ord_ok, "bridge_key_ord" => bridge_key_ord,
    "loglik_delta" => ord_loglik_delta,
)

# ---------------------------------------------------------------------------
# Negative controls: deliberately-wrong comparisons that MUST fail.
# ---------------------------------------------------------------------------
neg_coef_delta = length(j_coef_g) == length(r_coef_g) ?
    maximum(abs.(j_coef_g .- (r_coef_g .+ 1.0))) : Inf
neg_loglik_delta = abs(j_loglik_g - (r_loglik_g + 1.0))
neg_nb_swapped_delta = abs(r_loglik_nb1 - br_nb2.loglik)
neg_lognormal_accepted = gate["lognormal"]["ok"] === true  # must be false (rejection expected)

negative_controls = Dict(
    "NEG-COEF-SHIFTED" => Dict(
        "behaved" => neg_coef_delta > tol["coef_delta"], "delta" => neg_coef_delta),
    "NEG-LOGLIK-SHIFTED" => Dict(
        "behaved" => neg_loglik_delta > tol["loglik_delta"], "delta" => neg_loglik_delta),
    "NEG-LOGNORMAL-ACCEPTED" => Dict(
        "behaved" => !neg_lognormal_accepted, "gate_ok" => neg_lognormal_accepted),
    "NEG-NB-FAMILY-SWAPPED" => Dict(
        "behaved" => neg_nb_swapped_delta > tol["loglik_delta"], "delta" => neg_nb_swapped_delta),
)

all_positive_pass = all(v["pass"] for v in values(cases))
negatives_behaved = all(v["behaved"] for v in values(negative_controls))
overall_ok = all_positive_pass && negatives_behaved

report = Dict(
    "status" => overall_ok ? "PASS" : "FAIL",
    "area" => "namespace-2",
    "scope" => "CORE070_NAMESPACE_2_BATCH",
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
println("CORE070_NAMESPACE_2_BATCH_RESULT ",
        overall_ok ? "PASS" : "FAIL",
        " positive=", all_positive_pass, " negatives=", negatives_behaved)
overall_ok || exit(1)
