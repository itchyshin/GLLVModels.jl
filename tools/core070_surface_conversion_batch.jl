# Julia child for the surface-conversion batch
# (tools/core070_surface_conversion_batch.R). Reads the R-oracle JSON (path
# from ENV["CORE070_SURFACE_CONVERSION_R_ORACLE"], matching the env
# pass-through convention of tools/core070_inference_remainder_batch.jl),
# fits the three canonical fixtures NATIVELY (independent optimiser run on
# the same simulated Y the R process fit, not a replay of R's numbers),
# calls each case's NEW Julia surface, and compares against the R oracle at
# the contract's per-case tolerance. Writes ARGS[1] as the JSON results
# file.
#
# No RCall, no parity-runner include, no R of any kind runs here.
# No external JSON dependency (this project's Project.toml carries none);
# the minimal hand-rolled reader/writer below is copied verbatim from the
# repo convention in tools/core070_inference_remainder_batch.jl /
# tools/core070_namespace_2_batch.jl.
#
# Usage: julia --project=. tools/core070_surface_conversion_batch.jl <out.json>

using GLLVModels

# ---------------------------------------------------------------------------
# Minimal JSON reader/writer (no external dependency; mirrors the existing
# repo convention in tools/core070_inference_remainder_batch.jl).
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

length(ARGS) == 1 || error("usage: julia core070_surface_conversion_batch.jl <output.json>")
out_path = ARGS[1]
mkpath(dirname(out_path))

oracle_path = get(ENV, "CORE070_SURFACE_CONVERSION_R_ORACLE", "")
isempty(oracle_path) && error("CORE070_SURFACE_CONVERSION_R_ORACLE not set")
isfile(oracle_path) || error("oracle file not found: $oracle_path")
oracle = json_read(oracle_path)

root = normpath(joinpath(@__DIR__, ".."))
# GLLVM_PARITY_PIN=P1 reads the P1-regenerated contract (cases verbatim); unset/P0 is unchanged.
_parity_pin = uppercase(strip(get(ENV, "GLLVM_PARITY_PIN", "P1")))
_parity_pin in ("P0", "P1") || error("GLLVM_PARITY_PIN must be P0 or P1, got $(repr(_parity_pin))")
contract = json_read(joinpath(root, _parity_pin == "P1" ?
    "docs/dev-log/core070/true-parity-latest/surface-conversion-batch-contract-p1.json" :
    "docs/dev-log/core070/surface-conversion-batch-contract.json"))
cases = contract["cases"]
length(cases) == contract["expected_case_count"] || error("case count mismatch vs contract")
Int(contract["expected_case_count"]) == 20 || error("expected_case_count drifted from 20; update this script")

# ---------------------------------------------------------------------------
# Fixture 1: gaussian_small -- fit natively on the R oracle's simulated Y.
# ---------------------------------------------------------------------------
gs = oracle["gaussian_small"]
p, K, n = gs["p"], gs["K"], gs["n"]
Y_g = reshape(Float64.(gs["y"]), p, n)                 # R's as.numeric(Y_g) on a p x n matrix
                                                        # is column-major, matching Julia's
                                                        # default reshape(p, n) exactly.
fit_g = fit_gaussian_gllvm(Y_g; K = K)

# ---------------------------------------------------------------------------
# Fixture 2: twolevel_small.
# ---------------------------------------------------------------------------
tl = oracle["twolevel_small"]
p_tl, n_ind, reps = tl["p"], tl["n_individual"], tl["reps_per_individual"]
n_tl = n_ind * reps
Y_tl = reshape(Float64.(tl["y"]), p_tl, n_tl)
individual = Int.(tl["individual"])
fit_tl = fit_twolevel_gaussian(Y_tl, individual; K_B = 1, K_W = 1)

# ---------------------------------------------------------------------------
# Fixture 3: ordinal_small.
# ---------------------------------------------------------------------------
os = oracle["ordinal_small"]
p_o, K_o, n_o = os["p"], os["K"], os["n"]
Y_o = reshape(Int.(os["y"]), p_o, n_o)
fit_o = fit_ordinal_gllvm_pertrait(Y_o; K = K_o, link = ProbitLink(),
                                    g_tol = 1e-7, iterations = 1_000)

# ---------------------------------------------------------------------------
# Per-quantity Julia computation. Mirrors the R dispatcher in
# core070_surface_conversion_batch.R one-for-one by `quantity` key.
# ---------------------------------------------------------------------------
function julia_quantity(quantity::AbstractString)
    if quantity == "loadings_crossprod"
        # tcrossprod(L) = L*L' (p x p) is the rotation-invariant Gram
        # matrix; L'*L (d x d) is NOT rotation-invariant in general. The
        # first attempt used L'*L here, comparing a basis-dependent
        # quantity across two independently-rotated fits. REPAIR
        # (2026-09-01, wave5-conversion4): L*L', matching the R-side
        # tcrossprod() fix.
        L = GLLVModels.getLoadings(fit_g)
        return vec(L * L')
    elseif quantity == "lv_predictor"
        # rotate=false to match fit_g.pars.Λ, which is the model's RAW
        # (unrotated) internal Λ -- GLLVModels.getLV's own default is
        # rotate=true, which the first attempt left implicit, mixing a
        # rotated Z with a raw Λ and breaking the Λ*Z reconstruction
        # invariance internally (a self-inconsistency bug, not merely a
        # tolerance issue). REPAIR (2026-09-01, wave5-conversion4): both
        # sides now raw, matching R's own getLoadings/getLV default
        # rotate="none".
        # GLLVModels.getLV returns an n x K matrix (postfit.jl: `Zt =
        # permutedims(Z)` from an internal K x n solve) -- Λ (p x K) needs
        # Z TRANSPOSED (K x n) to reconstruct the p x n predictor. The
        # first attempt multiplied Λ * Z directly (p x K times n x K,
        # non-conformable for this fixture's K=2, n=80), which is why the
        # case failed outright rather than merely missing tolerance.
        Z = GLLVModels.getLV(fit_g, Y_g; rotate = false)
        return vec(fit_g.pars.Λ * Z')
    elseif quantity == "sigma_unit_total"
        return vec(GLLVModels.extract_Sigma(fit_g; level = :unit, part = :total).Sigma)
    elseif quantity == "sigma_table"
        t = GLLVModels.extract_Sigma_table(fit_g; level = :unit, part = :total)
        ts = sort(collect(t); by = r -> (r.trait_i, r.trait_j))
        return Float64[r.value for r in ts]
    # communality (postfit/POSTFIT-SURFACE-extract_communality) is NOT
    # computed here -- deferred (see contract.deferred): GLLVModels.jl's
    # communality(fit) denominator is the FULL total variance incl.
    # sigma_eps^2, while R's extract_communality() is single-tier-scoped
    # and never includes sigma_eps^2 for a Gaussian fit -- a genuine
    # cross-engine estimand mismatch, not a call-shape bug.
    # correlations (postfit/POSTFIT-SURFACE-extract_correlations) is NOT
    # computed here -- deferred (see contract.deferred, REPAIR 2026-09-01
    # wave5-conversion5): same tier-scoped-vs-total-variance estimand
    # mismatch as extract_communality (GLLVModels.jl's extract_correlations(fit)
    # = correlation(fit) standardises by sigma_y_site(fit), the FULL total
    # variance incl. sigma_eps^2; R's route standardises by a single tier
    # that never includes sigma_eps^2 for a Gaussian fit).
    elseif quantity == "cross_correlations"
        return vec(GLLVModels.extract_cross_correlations(fit_g; level = :unit,
                                                      traits_i = [1, 2], traits_j = [3, 4, 5]))
    elseif quantity == "residual_cov"
        # REPAIR (2026-09-01, wave5-conversion4: r_len=0). gaussian_small
        # has no unit_obs/W-tier block (K_W=0, unique=FALSE); R's
        # getResidualCov/getResidualCor default (and now this batch's) tier
        # is "unit", matching the R-side fix.
        return vec(GLLVModels.extract_residual_cov(fit_g; level = :unit))
    elseif quantity == "residual_cor"
        return vec(GLLVModels.extract_residual_cor(fit_g; level = :unit))
    elseif quantity == "ordination_sites"
        sites, _, _ = GLLVModels.extract_ordination(fit_g, Y_g)
        S = Matrix(sites)
        return size(S, 1) == n ? vec(sum(abs2, S; dims = 2)) : vec(sum(abs2, S; dims = 1))
    # proportions (postfit/POSTFIT-SURFACE-extract_proportions) is NOT
    # computed here -- deferred (see contract.deferred, REPAIR 2026-09-01
    # wave5-conversion5): SAME estimand mismatch as extract_communality --
    # extract_proportions(fit; component=:shared) IS communality(fit)
    # mathematically (both ΛΛ^T[t,t] / sigma_y_site(fit)[t,t]); confirmed
    # by the observed max_abs_diff (0.8713) exactly matching
    # extract_communality's own failure.
    # omega (postfit/POSTFIT-SURFACE-extract_Omega) is NOT computed here --
    # deferred (see contract.deferred, REPAIR 2026-09-01 wave5-conversion6):
    # GLLVModels.jl's extract_Omega(fit::GllvmFit) unconditionally sums
    # extract_Sigma(level=:unit,...) .+ extract_Sigma(level=:unit_obs,...),
    # and the :unit_obs total ALWAYS adds sigma_eps^2*I as a baseline term
    # regardless of whether a genuine W-tier exists; R's extract_Omega()
    # only sums tiers the fit actually carries (gaussian_small has none at
    # unit_obs), so sigma_eps^2 never enters R's sum. Diagonal-scale diff
    # (0.467, close to sigma_true^2=0.49) confirms this exactly. No
    # tiers/level kwarg exists on extract_Omega(fit::GllvmFit) to align the
    # call with R's narrower composition -- see the matching R-side trace
    # for the full source citation.
    elseif quantity == "icc_site"
        # REPAIR (2026-09-01, wave5-conversion5): R's extract_ICC_site()
        # needs BOTH a "unit" and a "unit_obs" tier; gaussian_small
        # (single-tier) has no unit_obs block, so R returned empty. Fixture
        # reassignment (not a call bug): now computed on fit_tl
        # (twolevel_small), matching the R-side fix. GLLVModels's
        # extract_ICC_site(fit::TwoLevelFit) is itself defined as
        # `= extract_repeatability(fit)` (src/extractors.jl), so this is
        # numerically the same quantity as repeatability_point on the same
        # fixture -- an intentional consequence of the two engines' own
        # internal equivalence, not redundant test design.
        return vec(GLLVModels.extract_ICC_site(fit_tl))
    elseif quantity == "loading_ci_wald_asym"
        tbl = GLLVModels.loading_ci(fit_g, Y_g; method = :wald_asym, conf_level = 0.95)
        return vcat(Float64[r.lower for r in tbl], Float64[r.upper for r in tbl])
    elseif quantity == "loading_profile"
        r = GLLVModels.loading_profile_exploratory(fit_g, 1, 1; level = 0.95)
        return Float64[r.lower, r.upper]
    elseif quantity == "profile_ci_total_variance"
        r = GLLVModels.profile_ci_total_variance(fit_g, 1; level = 0.95)
        return Float64[r.lower, r.upper]
    elseif quantity == "standard_errors"
        return vec(GLLVModels.standard_errors(fit_g, Y_g).se)
    elseif quantity == "repeatability_point"
        return vec(GLLVModels.extract_repeatability(fit_tl))
    elseif quantity in ("icc_ci_default", "icc_ci_wald")
        ci = GLLVModels.repeatability_ci(fit_tl, Y_tl, individual; method = :wald)
        return vcat(Float64[r.lower for r in ci], Float64[r.upper for r in ci])
    # icc_ci_bootstrap (CI-ROUTE-011) is NOT computed here -- it is a
    # `kind = "bootstrap_structural"` case, handled in the main case loop
    # below, not via julia_quantity()/oracle numeric comparison at all (see
    # the matching comment in tools/core070_surface_conversion_batch.R).
    elseif quantity == "cutpoints"
        # R's extract_cutpoints()$tau_estimate reports only the K-2 FREE
        # cutpoints per trait (Hadfield 2015: tau_1 = 0 fixed, never
        # reported). GLLVModels.jl's OrdinalPerTraitFit.τ is p x (C-1) with
        # column 1 the FIXED tau_1 = 0.0 (src/families/ordinal.jl:
        # `τ[t, 1] = 0.0`). REPAIR (2026-09-01, wave5-conversion4: r_len=5
        # vs julia_len=10): drop the fixed column so both sides report only
        # the free cutpoints, matching R's convention exactly.
        τ = GLLVModels.extract_cutpoints(fit_o).τ
        return vec(Matrix(τ)[:, 2:end])
    else
        error("BOGUS_QUANTITY: no dispatcher entry for '$(quantity)'")
    end
end

results = Dict{String, Any}()
all_ok = true

for cs in cases
    case_id = cs["case_id"]
    kind = cs["kind"]
    if kind == "refusal_pair"
        r_val = oracle["oracle_values"][case_id]
        r_raised = get(r_val, "raised", false)
        jl_raised = false
        jl_message = ""
        try
            GLLVModels.repeatability_ci(fit_tl, Y_tl, individual; method = :profile)
        catch e
            jl_raised = true
            jl_message = sprint(showerror, e)
        end
        ok = r_raised == true && jl_raised == true
        results[case_id] = Dict{String, Any}("pass" => ok, "kind" => kind,
                                              "r_raised" => r_raised, "julia_raised" => jl_raised,
                                              "julia_message" => jl_message)
        global all_ok &= ok
        continue
    end

    if kind == "bootstrap_structural"
        # REPAIR (2026-09-01, wave5-conversion4): n_boot=200 percentile
        # bootstrap endpoints from two INDEPENDENT stochastic simulate-
        # refit procedures carry too much Monte Carlo error for a numeric
        # endpoint-distance comparison. Each engine's own bootstrap CI is
        # checked structurally against ITS OWN wald-route point estimate --
        # finite, ordered, brackets the point -- with no cross-engine
        # numeric distance at all (see the matching R-side comment).
        r_val = oracle["oracle_values"][case_id]
        r_ok = get(r_val, "finite", false) == true && get(r_val, "ordered", false) == true &&
               get(r_val, "brackets_point", false) == true
        jl_ok, jl_facts, jl_err = try
            ci = GLLVModels.repeatability_ci(fit_tl, Y_tl, individual; method = :bootstrap,
                                         nsim = 200, seed = 11)
            lower = Float64[r.lower for r in ci]; upper = Float64[r.upper for r in ci]
            point = GLLVModels.extract_repeatability(fit_tl)
            finite = all(isfinite, lower) && all(isfinite, upper)
            ordered = all(lower .<= upper)
            brackets = all(lower .<= point .<= upper)
            (finite && ordered && brackets,
             Dict{String, Any}("lower" => lower, "upper" => upper, "point" => point,
                                "finite" => finite, "ordered" => ordered, "brackets_point" => brackets),
             "")
        catch e
            (false, Dict{String, Any}(), sprint(showerror, e))
        end
        ok = r_ok && jl_ok
        results[case_id] = Dict{String, Any}("pass" => ok, "kind" => kind,
                                              "r_structural" => r_val, "julia_structural" => jl_facts,
                                              "julia_error" => jl_err)
        global all_ok &= ok
        continue
    end

    quantity = cs["quantity"]
    tol = Float64(cs["tolerance"])
    # REPAIR (2026-09-01, wave5-conversion3): a case_id absent from
    # oracle_values (the R side recorded an oracle_errors entry for it
    # instead -- see tools/core070_surface_conversion_batch.R's loud
    # coverage check) must NOT abort the whole batch. Record a loud,
    # forensic per-case FAIL and move on so every other case still gets a
    # real verdict in one receipt.
    if !haskey(oracle["oracle_values"], case_id)
        r_err = get(get(oracle, "oracle_errors", Dict{String, Any}()), case_id, "(no oracle_errors entry either)")
        results[case_id] = Dict{String, Any}("pass" => false, "kind" => kind, "quantity" => quantity,
                                              "tolerance" => tol, "max_abs_diff" => NaN,
                                              "r_len" => 0, "julia_len" => 0,
                                              "error" => "missing_oracle_value: R oracle_errors[$case_id] = $(r_err)")
        global all_ok = false
        continue
    end
    r_vec = Float64.(oracle["oracle_values"][case_id])

    jl_vec = Float64[]
    err = ""
    ok = false
    try
        jl_vec = julia_quantity(quantity)
        ok = length(jl_vec) == length(r_vec) && maximum(abs.(jl_vec .- r_vec)) <= tol
    catch e
        err = sprint(showerror, e)
    end
    maxdiff = (isempty(jl_vec) || length(jl_vec) != length(r_vec)) ? NaN :
        maximum(abs.(jl_vec .- r_vec))

    results[case_id] = Dict{String, Any}("pass" => ok, "kind" => kind, "quantity" => quantity,
                                          "tolerance" => tol, "max_abs_diff" => maxdiff,
                                          "r_len" => length(r_vec), "julia_len" => length(jl_vec),
                                          "error" => err,
                                          # REPAIR (2026-09-01, wave5-conversion6 coordinator ask):
                                          # store the actual Julia-side vector alongside R's (already
                                          # in r-oracle.json's oracle_values) so future forensics on a
                                          # failing case can read both sides' numbers straight out of
                                          # the retained julia-results.json, without needing a refit.
                                          "julia_values" => jl_vec)
    global all_ok &= ok
end

# --- negative controls: mirror the R side's rejections ---------------------
neg_bogus_quantity = try
    julia_quantity("this_quantity_does_not_exist")
    false
catch
    true
end
r_neg = oracle["negative_controls"]
neg_bogus_quantity_r = get(r_neg["bogus_quantity"], "rejected", false)
neg_wrong_fixture_r = get(r_neg["wrong_fixture"], "rejected", false)
neg_ok = neg_bogus_quantity && (neg_bogus_quantity_r == true) && (neg_wrong_fixture_r == true)

report = Dict{String, Any}(
    "status" => (all_ok && neg_ok) ? "PASS" : "FAIL",
    "julia_version" => string(VERSION),
    "package_root" => Base.pkgdir(GLLVModels),
    "case_count" => length(cases),
    "all_checks" => all_ok,
    "negative_controls_behaved_as_expected" => neg_ok,
    "cases" => results,
)
open(out_path, "w") do io
    write(io, to_json(report))
end
println(report["status"] == "PASS" ? "CORE070_SURFACE_CONVERSION_JULIA_PASS" :
        "CORE070_SURFACE_CONVERSION_JULIA_FAIL")
exit(report["status"] == "PASS" ? 0 : 1)
