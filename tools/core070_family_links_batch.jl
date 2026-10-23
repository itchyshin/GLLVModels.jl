# Inner executor for the "family-links" micro-batch (2 planned
# EXECUTABLE_NOW cases: binomial-probit and binomial-cloglog native-model
# fits; see docs/dev-log/core070/family-links-batch-contract.json for the
# full case list, binding family/FAMILY-01-PROBIT and family/FAMILY-01-CLOGLOG
# in docs/dev-log/core070/required-source-case-map.json).
#
# Pure-Julia consumer, mirroring tools/core070_namespace_2_batch.jl: reads a
# JSON oracle file the paired R runner (tools/core070_family_links_batch.R)
# writes BEFORE invoking this script (that R process already has the frozen
# gllvmTMB library loaded and does 100% of the live R-side fitting itself),
# and fits the identical data natively via direct `using GLLVModels` module calls
# only -- fit_binomial_gllvm(Y; K, link), the native family-dispatch fit path
# GLLVModels._bridge_family_key("binomial_probit"/"binomial_cloglog") route to
# (src/bridge.jl:147-148). No RCall, no parity-runner include, no R of any
# kind runs in this process.
#
# argv:
#   ARGS[1]  destination path for the JSON results file (parent dir need not
#            exist yet; this script mkpath()s it). Must not already exist.
#
# Env vars required (set by the outer R runner before invoking this script;
# never defaulted silently here so a misconfigured environment fails loudly
# rather than quietly running against stale or missing oracle data):
#   CORE070_FAMILY_LINKS_R_ORACLE = <path to the R-written oracle JSON>
#
# Invocation:
#   julia --project=. tools/core070_family_links_batch.jl <out.json>

using GLLVModels

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

length(ARGS) == 1 || error("usage: julia tools/core070_family_links_batch.jl <out.json>")
out_path = ARGS[1]
isfile(out_path) && error("destination already exists: $out_path")
mkpath(dirname(out_path))

@assert realpath(Base.pkgdir(GLLVModels)) == realpath(pwd()) "must run from the GLLVModels.jl package root"

oracle_path = get(ENV, "CORE070_FAMILY_LINKS_R_ORACLE", "")
isempty(oracle_path) &&
    error("CORE070_FAMILY_LINKS_R_ORACLE is required; refusing to silently default it")
isfile(oracle_path) ||
    error("CORE070_FAMILY_LINKS_R_ORACLE points to a nonexistent file: $oracle_path")

oracle = json_read(oracle_path)

# Tolerance calibration: these cases compare INDEPENDENT optimizations (R
# fit vs Julia fit from their own warm starts), not a same-point objective
# evaluation, and both engines use the ordinary Laplace approximation (see
# the contract's approximation_note and the header of the paired R runner).
# The established paired-fit precedent in tests/testthat/test-julia-bridge.R
# (gllvmTMB lane) and tools/core070_namespace_2_batch.jl is 1e-4 absolute;
# same-point identity checks elsewhere in this repo keep 1e-6..1e-8.
tol = Dict("loglik_delta" => 1e-4, "coef_delta" => 1e-4)
cases = Dict{String, Any}()

# ---------------------------------------------------------------------------
# Family-key admission gate (mirrors the R-side .gllvm_julia_family_scalar
# check): both bridge dispatch keys must resolve on the Julia side too.
# ---------------------------------------------------------------------------
gate = oracle["gate"]
bridge_key_probit  = GLLVModels._bridge_family_key("binomial_probit")  == "binomial_probit"
bridge_key_cloglog = GLLVModels._bridge_family_key("binomial_cloglog") == "binomial_cloglog"
gate_probit_ok  = gate["probit"]["ok"]  === true && gate["probit"]["key"]  == "binomial_probit"
gate_cloglog_ok = gate["cloglog"]["ok"] === true && gate["cloglog"]["key"] == "binomial_cloglog"
@assert bridge_key_probit && bridge_key_cloglog && gate_probit_ok && gate_cloglog_ok "family-key admission gate mismatch between R and Julia"

# ---------------------------------------------------------------------------
# Cross-objective identity check (panel 2026-09-01 pattern), reused here to
# diagnose a loglik-delta miss: is it a likelihood-function disagreement, or
# just a different attained local optimum on the SAME objective? Read-only
# `include` of the shared tool; nothing here mutates it.
# ---------------------------------------------------------------------------
include(joinpath(@__DIR__, "core070_cross_objective.jl"))

using Random

# ---------------------------------------------------------------------------
# Multi-start wrapper (2026-09-01 repair, defect 2): a single default warm
# start can miss the optimum, particularly for cloglog's steeper link. Start
# 0 is the existing default (empirical link-scale intercepts + PPCA-style
# loadings); starts 1..n_starts-1 are small random perturbations from a
# FIXED seed (reproducible), and the best (highest finite loglik, converged)
# fit wins. Cheap at this fixture's size (p<=5, n<=120).
# ---------------------------------------------------------------------------
function _best_of_multistart_binomial(Y::AbstractMatrix, K::Integer, link;
        n_starts::Int = 6, seed::Int = 20260901)
    p = size(Y, 1)
    best = nothing
    for s in 0:(n_starts - 1)
        fit = try
            if s == 0
                fit_binomial_gllvm(Y; K = K, link = link)
            else
                rng = Random.MersenneTwister(seed + s)
                β0 = randn(rng, p) .* 0.5
                Λ0 = randn(rng, p, K) .* 0.4
                fit_binomial_gllvm(Y; K = K, link = link, β_init = β0, Λ_init = Λ0)
            end
        catch
            nothing
        end
        if fit !== nothing && fit.converged && isfinite(fit.loglik) &&
           (best === nothing || fit.loglik > best.loglik)
            best = fit
        end
    end
    return best
end

# ---------------------------------------------------------------------------
# Shared case evaluator: fixes defect 1 (coef comparand) and adds defect 2's
# cross-objective diagnosis + asymmetric-optimum verdict.
#
# DEFECT 1 FIX: coef.gllvmTMB_multi() (R/vcov-coef.R) returns ONLY the p
# trait intercepts (b_fix / X_fix_names) -- never the loadings. Comparing
# against vcat(fit.beta, vec(fit.Lambda)) (length p+p*K) against R's length-p
# vector produced a permanent length mismatch -> Inf delta -> silently
# serialized as JSON null by `to_json`'s isfinite guard, with no indication
# WHY. Fixed: compare fit.beta alone, and attach an explicit non-empty
# `fail_reason` string whenever the comparison itself could not be made (not
# just when it failed the tolerance), so a length mismatch is loud, not null.
#
# DEFECT 2 DIAGNOSIS: when the coef/loglik deltas miss tolerance and the fit
# is NOT saturated, evaluate Julia's own objective (cross_objective_at) at
# R's fitted (beta, loadings) coordinates. If that value matches R's
# reported loglik (small identity_delta) -- confirming SAME likelihood
# function, not a bug in the Julia kernel -- and Julia's own best-of-
# multistart optimum is >= that value, the case is an ASYMMETRIC OPTIMUM: R
# stopped at a lower point on the identical objective. That is recorded
# explicitly (both engines' values, the cross-objective evaluation, and the
# identity delta) and the case PASSES on that evidence, not a naive delta
# gate. If Julia's own optimum is instead WORSE than the value at R's
# coordinates (even after multistart), that is a genuine unresolved
# optimizer gap and the case FAILS with a diagnostic fail_reason -- no
# tolerance widening substitutes for closing it.
# ---------------------------------------------------------------------------
function _evaluate_binomial_case(Y::AbstractMatrix, K::Integer, link,
        r_coef::Vector{Float64}, r_loglik::Float64, r_loadings_flat::Vector{Float64},
        tol::Dict{String, Float64})
    p = size(Y, 1)
    fit = _best_of_multistart_binomial(Y, K, link)
    fit === nothing && return (fit = nothing, result = Dict(
        "pass" => false, "fail_reason" => "no multistart candidate converged to a finite loglik",
        "coef_delta" => nothing, "loglik_delta" => nothing, "saturated" => nothing,
    ))

    j_coef = collect(Float64, fit.β)  # DEFECT 1 FIX: beta only, matches R's coef()
    j_loglik = fit.loglik
    sat = fit.saturation
    saturated = sat !== nothing && (sat.n_clamp > 0 || sat.n_wcollapse > 0)
    n_clamp = sat === nothing ? -1 : sat.n_clamp
    n_wcollapse = sat === nothing ? -1 : sat.n_wcollapse

    coef_ok = length(j_coef) == length(r_coef)
    coef_delta = coef_ok ? maximum(abs.(j_coef .- r_coef)) : nothing
    loglik_delta = abs(j_loglik - r_loglik)

    base = Dict{String, Any}(
        "coef_delta" => coef_delta, "loglik_delta" => loglik_delta,
        "saturated" => saturated, "n_clamp" => n_clamp, "n_wcollapse" => n_wcollapse,
        "julia_loglik" => j_loglik, "r_loglik" => r_loglik,
        # The coefficient vectors themselves, so the receipt tool recomputes max|R - Julia|.
        "julia_coef" => j_coef, "r_coef" => r_coef,
    )

    if !coef_ok
        base["pass"] = false
        base["fail_reason"] = "coef length mismatch: julia beta has $(length(j_coef)) " *
            "entries, R oracle coef has $(length(r_coef)) -- extraction bug, not a " *
            "tolerance miss (R's coef() returns b_fix trait intercepts only)"
        return (fit = fit, result = base)
    end

    if saturated
        base["pass"] = false
        base["fail_reason"] = "Laplace saturation ($(n_clamp) clamped / $(n_wcollapse) " *
            "collapsed-weight cells): a saturated fit is a FAIL regardless of the deltas"
        return (fit = fit, result = base)
    end

    if coef_delta <= tol["coef_delta"] && loglik_delta <= tol["loglik_delta"]
        base["pass"] = true
        base["fail_reason"] = ""
        return (fit = fit, result = base)
    end

    # Delta gate missed and not saturated: run the cross-objective identity
    # check before concluding anything about the likelihood implementation.
    r_Lambda = reshape(r_loadings_flat, p, K)
    julia_obj_at_r_coords = cross_objective_at(:binomial, Float64.(Y);
        beta = r_coef, crossprod_or_loadings = r_Lambda, rank = K, link = link)
    identity_delta = abs(julia_obj_at_r_coords - r_loglik)
    base["julia_obj_at_r_coords"] = julia_obj_at_r_coords
    base["cross_objective_identity_delta"] = identity_delta

    if identity_delta <= 1e-4 && j_loglik >= julia_obj_at_r_coords - 1e-6
        # Same objective (identity confirmed), and Julia's own best-of-
        # multistart optimum is at least as good as R's coordinates evaluated
        # under that SAME objective -> R stopped at a lower point. Record the
        # asymmetry as evidence, not a naive delta-gate failure.
        base["pass"] = true
        base["fail_reason"] = ""
        base["verdict"] = "asymmetric_optimum_julia_better"
        return (fit = fit, result = base)
    end

    base["pass"] = false
    base["fail_reason"] = if identity_delta > 1e-4
        "cross-objective identity check itself missed tolerance " *
        "(|julia_obj_at_r_coords - r_loglik| = $(identity_delta)); this points at a " *
        "likelihood-implementation disagreement, not merely an optimizer miss"
    else
        "Julia's own best-of-multistart optimum ($(j_loglik)) is still below Julia's " *
        "objective evaluated at R's fitted coordinates ($(julia_obj_at_r_coords)): a " *
        "genuine unresolved optimizer/warm-start gap, not closed by multistart"
    end
    return (fit = fit, result = base)
end

# ---------------------------------------------------------------------------
# Case 1: CORE070-FAMILY-01-PROBIT-NATIVE-MODEL
# ---------------------------------------------------------------------------
pr = oracle["probit"]
p_pr = Int(pr["p"]); n_pr = Int(pr["n"]); K_pr = Int(pr["K"])
y_pr_flat = Float64.(pr["y"])
length(y_pr_flat) == p_pr * n_pr || error("probit y length mismatch")
Y_pr = Int.(reshape(y_pr_flat, p_pr, n_pr))
r_coef_pr = Float64.(pr["coef"])
r_loglik_pr = Float64(pr["loglik"])
r_loadings_pr = Float64.(pr["loadings"])

eval_pr = _evaluate_binomial_case(Y_pr, K_pr, ProbitLink(), r_coef_pr, r_loglik_pr, r_loadings_pr, tol)
fit_pr = eval_pr.fit
cases["CORE070-FAMILY-01-PROBIT-NATIVE-MODEL"] = eval_pr.result
j_coef_pr = fit_pr === nothing ? Float64[] : collect(Float64, fit_pr.β)
j_loglik_pr = fit_pr === nothing ? NaN : fit_pr.loglik

# ---------------------------------------------------------------------------
# Case 2: CORE070-FAMILY-01-CLOGLOG-NATIVE-MODEL
# ---------------------------------------------------------------------------
cl = oracle["cloglog"]
p_cl = Int(cl["p"]); n_cl = Int(cl["n"]); K_cl = Int(cl["K"])
y_cl_flat = Float64.(cl["y"])
length(y_cl_flat) == p_cl * n_cl || error("cloglog y length mismatch")
Y_cl = Int.(reshape(y_cl_flat, p_cl, n_cl))
r_coef_cl = Float64.(cl["coef"])
r_loglik_cl = Float64(cl["loglik"])
r_loadings_cl = Float64.(cl["loadings"])

eval_cl = _evaluate_binomial_case(Y_cl, K_cl, CLogLogLink(), r_coef_cl, r_loglik_cl, r_loadings_cl, tol)
fit_cl = eval_cl.fit
cases["CORE070-FAMILY-01-CLOGLOG-NATIVE-MODEL"] = eval_cl.result
j_coef_cl = fit_cl === nothing ? Float64[] : collect(Float64, fit_cl.β)
j_loglik_cl = fit_cl === nothing ? NaN : fit_cl.loglik

# ---------------------------------------------------------------------------
# Negative controls: deliberately-wrong comparisons / adversarial fixtures
# that MUST fail / MUST be flagged.
# ---------------------------------------------------------------------------
neg_coef_delta_pr = length(j_coef_pr) == length(r_coef_pr) ?
    maximum(abs.(j_coef_pr .- (r_coef_pr .+ 1.0))) : Inf
neg_loglik_delta_cl = abs(j_loglik_cl - (r_loglik_cl + 1.0))
neg_link_swapped_delta = abs(r_loglik_pr - j_loglik_cl)

# Synthetic near-perfectly-separated Bernoulli-cloglog cell (Julia-only, no
# R oracle needed): exercises the LaplaceSaturationHealth mechanism itself
# so this batch is not vacuously trusting a diagnostic that never fires.
# Deliberately extreme, well past the |eta| <~ 1.5 bound used for the two
# positive cases above (see check-log 2026-08-28, the diagnosed cloglog
# saturation pathology this control guards against).
p_sat = 4; n_sat = 20; K_sat = 1
Y_sat = ones(Int, p_sat, n_sat)  # constant-1 response: near-complete separation
fit_sat = fit_binomial_gllvm(Y_sat; K = K_sat, link = CLogLogLink())
sat_sat = fit_sat.saturation
is_flagged_saturated = sat_sat !== nothing && (sat_sat.n_clamp > 0 || sat_sat.n_wcollapse > 0)

negative_controls = Dict(
    "NEG-COEF-SHIFTED-PROBIT" => Dict(
        "behaved" => neg_coef_delta_pr > tol["coef_delta"], "delta" => neg_coef_delta_pr),
    "NEG-LOGLIK-SHIFTED-CLOGLOG" => Dict(
        "behaved" => neg_loglik_delta_cl > tol["loglik_delta"], "delta" => neg_loglik_delta_cl),
    "NEG-LINK-SWAPPED" => Dict(
        "behaved" => neg_link_swapped_delta > tol["loglik_delta"], "delta" => neg_link_swapped_delta),
    "NEG-SATURATED-FIT-MUST-FAIL" => Dict(
        "behaved" => is_flagged_saturated,
        "n_clamp" => sat_sat === nothing ? -1 : sat_sat.n_clamp,
        "n_wcollapse" => sat_sat === nothing ? -1 : sat_sat.n_wcollapse),
)

all_positive_pass = all(v["pass"] for v in values(cases))
negatives_behaved = all(v["behaved"] for v in values(negative_controls))
overall_ok = all_positive_pass && negatives_behaved

report = Dict(
    "status" => overall_ok ? "PASS" : "FAIL",
    "area" => "family-links",
    "scope" => "CORE070_FAMILY_LINKS_BATCH",
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
println("CORE070_FAMILY_LINKS_BATCH_RESULT ",
        overall_ok ? "PASS" : "FAIL",
        " positive=", all_positive_pass, " negatives=", negatives_behaved)
overall_ok || exit(1)
