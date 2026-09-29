# Julia child for the wave-6 conversion batch
# (tools/core070_wave6_conversion_batch.R). Reads the R-oracle JSON (path
# from ENV["CORE070_WAVE6_CONVERSION_R_ORACLE"]), fits the structured-term
# fixture and the gaussian_small fixture NATIVELY (independent optimiser
# run on the same Y the R process fit, not a replay of R's numbers), calls
# each case's Julia surface, and compares against the R oracle at the
# contract's per-case tolerance. Writes ARGS[1] as the JSON results file.
#
# No RCall, no parity-runner include, no R of any kind runs here.
# JSON reader/writer copied verbatim from the repo convention in
# tools/core070_surface_conversion_batch.jl / tools/core070_inference_remainder_batch.jl.
#
# Usage: julia --project=. tools/core070_wave6_conversion_batch.jl <out.json>

using GLLVModels

# ---------------------------------------------------------------------------
# Minimal JSON reader/writer (no external dependency).
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

length(ARGS) == 1 || error("usage: julia core070_wave6_conversion_batch.jl <output.json>")
out_path = ARGS[1]
mkpath(dirname(out_path))

oracle_path = get(ENV, "CORE070_WAVE6_CONVERSION_R_ORACLE", "")
isempty(oracle_path) && error("CORE070_WAVE6_CONVERSION_R_ORACLE not set")
isfile(oracle_path) || error("oracle file not found: $oracle_path")
oracle = json_read(oracle_path)

root = normpath(joinpath(@__DIR__, ".."))
# GLLVM_PARITY_PIN=P1 reads the P1-regenerated contract (cases verbatim); unset/P0 is unchanged.
_wave6_pin = uppercase(strip(get(ENV, "GLLVM_PARITY_PIN", "P0")))
_wave6_pin in ("P0", "P1") || error("GLLVM_PARITY_PIN must be P0 or P1")
contract = json_read(joinpath(root, _wave6_pin == "P1" ?
    "docs/dev-log/core070/true-parity-latest/wave6-conversion-batch-contract-p1.json" :
    "docs/dev-log/core070/wave6-conversion-batch-contract.json"))
cases = contract["cases"]
length(cases) == contract["expected_case_count"] || error("case count mismatch vs contract")
Int(contract["expected_case_count"]) == 10 || error("expected_case_count drifted from 10; update this script")

# REPAIR (2026-09-01, wave6-conversion1 forensics item 2): a null/missing R
# oracle value must become a recorded per-case FAIL, never a crash. R's
# jsonlite writer (null="null") turns an R NULL into JSON `null`, which
# json_read() above parses as Julia `nothing` -- Float64.(::Nothing) throws
# a MethodError that previously crashed the whole Julia process before any
# report was written. `oracle_numeric_or_missing` centralises the guard: it
# returns (ok::Bool, vec::Vector{Float64}) and NEVER throws on a null/absent
# oracle entry, mirroring wave-5's soft-fail-per-case pattern
# (tools/core070_surface_conversion_batch.jl's missing-oracle-value branch).
function oracle_numeric_or_missing(oracle::Dict, case_id::AbstractString)
    ov = get(oracle, "oracle_values", Dict{String, Any}())
    if !haskey(ov, case_id) || ov[case_id] === nothing
        return false, Float64[]
    end
    raw = ov[case_id]
    try
        return true, Float64.(raw)
    catch
        return false, Float64[]
    end
end

# ---------------------------------------------------------------------------
# Fixture 1: structured_kernel_small -- fit natively on the R oracle's
# deterministic Y. Reconstruct df's columns (species grouping, C, K2)
# exactly as tools/core070_wave6_conversion_batch.R built them.
# ---------------------------------------------------------------------------
sk = oracle["structured_kernel_small"]
p_sk, n_site, n_species = sk["p"], sk["n_site"], sk["n_species"]
species_of_site = Int.(sk["species_of_site"])       # length n_site, 1-based species index per site
y_flat = Float64.(sk["y"])                           # length p_sk*n_site, R column-major over
                                                       # expand.grid(site, trait): site fastest
C = reshape(Float64.(sk["C"]), n_species, n_species)
K2 = reshape(Float64.(sk["K2"]), n_species, n_species)

# Build the "data" NamedTuple-of-columns Tables.jl-compatible object the
# recognizer expects: one row per (site, trait) is NOT what
# _fit_gaussian_structured_sources wants -- it wants Y (p x n_site) plus a
# `data` table carrying the GROUPING column at one row per unit (site).
# _source_term_covariance looks up `data`'s `spec.group` column and takes
# `sort(unique(...))` of it, then maps each row to its level index -- so
# `data` must have one row per SITE (the fit's units), not per (site,trait).
# R's expand.grid(site, trait) puts SITE fastest, so y_flat is ordered
# (site1..siteN) per trait block: that is column-major n_site x p. The fit
# wants p x n_site (trait rows), so reshape to n_site x p and transpose.
# (Attempt-4 root cause: the old reshape(y_flat, p_sk, n_site) scrambled Y,
# collapsing every structured fit to the null model with sigma_source -> 0.)
Y_sk = Matrix(transpose(reshape(y_flat, n_site, p_sk)))
species_symbols = [Symbol('a' + (species_of_site[s] - 1)) for s in 1:n_site]
data_sk = (site = collect(1:n_site), species = species_symbols)
kernel_env_sk = (C = C, K2 = K2)

function fit_structured(term_exprs::Vector{Expr})
    GLLVModels._fit_gaussian_structured_sources(Y_sk, data_sk, term_exprs; kernel_env = kernel_env_sk)
end

_structured_cache = Dict{String, Any}()
function structured_fit_cached(key::String, term_exprs::Vector{Expr})
    get!(_structured_cache, key) do
        fit_structured(term_exprs)
    end
end

function structured_quantity(fit, level_name::Symbol)
    ll = hasproperty(fit, :logLik) ? fit.logLik : fit.loglik
    # R records the requested tier's own p x p covariance
    # (extract_Sigma(level = <term name>, part = "total")). For the native
    # GaussianSourcesFit the paired comparand is that source's fitted trait
    # covariance; this batch's fixtures put the compared source (the lone
    # term, or the kernel named "k1") FIRST in the term list, so index 1.
    Σ = if fit isa GLLVModels.GaussianSourcesFit
        Matrix(fit.trait_covariances[1])
    else
        try
            Matrix(GLLVModels.extract_Sigma(fit; level = level_name, part = :total).Sigma)
        catch
            fill(NaN, n_species, n_species)
        end
    end
    vcat([ll], vec(Σ))
end

# ---------------------------------------------------------------------------
# Fixture 2: gaussian_small -- fit natively.
# ---------------------------------------------------------------------------
gs = oracle["gaussian_small"]
p, K, n = gs["p"], gs["K"], gs["n"]
Y_g = reshape(Float64.(gs["y"]), p, n)
fit_g = fit_gaussian_gllvm(Y_g; K = K)

results = Dict{String, Any}()
all_ok = true

term_expr_map = Dict{String, Vector{Expr}}(
    # CORE070-WAVE6-INDEP-FIT / -SCALAR-FIT REPAIR (2026-09-01,
    # wave6-conversion3 forensics): moved to contract.deferred[] -- the
    # frozen 0.7.0 gllvmTMB engine rejects a lone indep()/scalar() term at
    # ANY grouping ("Custom level=\"source\" is not yet supported",
    # confirmed at both |species and |site), so no oracle_values entry is
    # ever produced for them and they no longer appear in contract.cases.
    "CORE070-WAVE6-KERNEL-INDEP-FIT" => [:(kernel_indep(species, K = C, name = "k1"))],
    "CORE070-WAVE6-KERNEL-DEP-FIT" => [:(kernel_dep(species, K = C, name = "k1"))],
    "CORE070-WAVE6-KERNEL-SCALAR-FIT" => [:(kernel_scalar(species, K = C, name = "k1"))],
    "CORE070-WAVE6-KERNEL-LATENT-SINGLE-PSI-NAMESPACE" =>
        [:(kernel_latent(species, K = C, d = 1, name = "k1", unique = true))],
    "CORE070-WAVE6-KERNEL-LATENT-SINGLE-PSI-COVARIANCE" =>
        [:(kernel_latent(species, K = C, d = 1, name = "k1", unique = true))],
    "CORE070-WAVE6-KERNEL-LATENT-MULTI-NAMESPACE" =>
        [:(kernel_latent(species, K = C, d = 1, name = "k1", unique = false)),
         :(kernel_latent(species, K = K2, d = 1, name = "k2", unique = false))],
    "CORE070-WAVE6-KERNEL-LATENT-MULTI-COVARIANCE-EXPORT" =>
        [:(kernel_latent(species, K = C, d = 1, name = "k1", unique = true))],
)
level_name_map = Dict{String, Symbol}(
    "CORE070-WAVE6-KERNEL-INDEP-FIT" => :k1,
    "CORE070-WAVE6-KERNEL-DEP-FIT" => :k1,
    "CORE070-WAVE6-KERNEL-SCALAR-FIT" => :k1,
    "CORE070-WAVE6-KERNEL-LATENT-SINGLE-PSI-NAMESPACE" => :k1,
    "CORE070-WAVE6-KERNEL-LATENT-SINGLE-PSI-COVARIANCE" => :k1,
    "CORE070-WAVE6-KERNEL-LATENT-MULTI-NAMESPACE" => :k1,
    "CORE070-WAVE6-KERNEL-LATENT-MULTI-COVARIANCE-EXPORT" => :k1,
)

for cs in cases
    case_id = cs["case_id"]
    kind = cs["kind"]

    if haskey(term_expr_map, case_id)
        r_ok, r_vec = oracle_numeric_or_missing(oracle, case_id)
        if !r_ok
            results[case_id] = Dict{String, Any}("pass" => false, "kind" => kind,
                                                  "tolerance" => cs["tolerance"], "max_abs_diff" => NaN,
                                                  "r_len" => 0, "julia_len" => 0,
                                                  "error" => "null_oracle_value: R oracle_values[$case_id] was null/missing")
            global all_ok = false
            continue
        end
        jl_vec = Float64[]
        err = ""
        ok = false
        try
            fit = structured_fit_cached(case_id, term_expr_map[case_id])
            jl_vec = structured_quantity(fit, level_name_map[case_id])
            ok = length(jl_vec) == length(r_vec) && maximum(abs.(jl_vec .- r_vec)) <= Float64(cs["tolerance"])
        catch e
            err = sprint(showerror, e)
        end
        maxdiff = (isempty(jl_vec) || length(jl_vec) != length(r_vec)) ? NaN :
            maximum(abs.(jl_vec .- r_vec))
        results[case_id] = Dict{String, Any}("pass" => ok, "kind" => kind, "tolerance" => cs["tolerance"],
                                              "max_abs_diff" => maxdiff, "r_len" => length(r_vec),
                                              "julia_len" => length(jl_vec), "error" => err,
                                              "julia_values" => jl_vec)
        global all_ok &= ok
        continue
    end

    if kind == "own_receipt_defect"
        r_val = get(get(oracle, "oracle_values", Dict{String, Any}()), case_id, nothing)
        if r_val === nothing
            results[case_id] = Dict{String, Any}("pass" => false, "kind" => kind,
                                                  "error" => "null_oracle_value: R oracle_values[$case_id] was null/missing",
                                                  "known_defect_pending_decision" => true)
            global all_ok = false
            continue
        end
        r_ok = get(r_val, "matches_own_formula", false) == true
        jl_ok, jl_nobs, jl_err = try
            jn = Float64(GLLVModels.nobs(fit_g, Y_g))
            (jn == Float64(n), jn, "")
        catch e
            (false, NaN, sprint(showerror, e))
        end
        ok = r_ok && jl_ok
        results[case_id] = Dict{String, Any}("pass" => ok, "kind" => kind,
                                              "r_nobs" => get(r_val, "nobs", NaN),
                                              "r_expected_p_times_n" => get(r_val, "expected", NaN),
                                              "julia_nobs" => jl_nobs, "julia_expected_n" => Float64(n),
                                              "error" => jl_err,
                                              "known_defect_pending_decision" => true)
        global all_ok &= ok
        continue
    end

    # Remaining "point" postfit cases, keyed by `quantity`.
    quantity = cs["quantity"]
    tol = Float64(cs["tolerance"])
    r_ok, r_vec = oracle_numeric_or_missing(oracle, case_id)
    if !r_ok
        r_err = get(get(oracle, "oracle_errors", Dict{String, Any}()), case_id, "(no oracle_errors entry either; value was null)")
        results[case_id] = Dict{String, Any}("pass" => false, "kind" => kind, "quantity" => quantity,
                                              "tolerance" => tol, "max_abs_diff" => NaN,
                                              "r_len" => 0, "julia_len" => 0,
                                              "error" => "null_oracle_value: R oracle_values[$case_id] was null/missing; oracle_errors said: $(r_err)")
        global all_ok = false
        continue
    end

    jl_vec = Float64[]
    err = ""
    ok = false
    try
        if quantity == "loglik_scalar"
            jl_vec = [GLLVModels.loglikelihood(fit_g)]
        elseif quantity == "confint_sigma_eps_bounds"
            ci = GLLVModels.confint(fit_g, Y_g; parm = "sigma_eps", level = 0.95)
            # confint returns a NamedTuple of columns (term/estimate/lower/upper)
            jl_vec = vcat(Float64.(ci.lower), Float64.(ci.upper))
        else
            error("BOGUS_QUANTITY: no dispatcher entry for '$(quantity)'")
        end
        ok = length(jl_vec) == length(r_vec) && maximum(abs.(jl_vec .- r_vec)) <= tol
    catch e
        err = sprint(showerror, e)
    end
    maxdiff = (isempty(jl_vec) || length(jl_vec) != length(r_vec)) ? NaN :
        maximum(abs.(jl_vec .- r_vec))
    results[case_id] = Dict{String, Any}("pass" => ok, "kind" => kind, "quantity" => quantity,
                                          "tolerance" => tol, "max_abs_diff" => maxdiff,
                                          "r_len" => length(r_vec), "julia_len" => length(jl_vec),
                                          "error" => err, "julia_values" => jl_vec)
    global all_ok &= ok
end

# ---------------------------------------------------------------------------
# Rejection-path cases: both engines must refuse.
# ---------------------------------------------------------------------------
rejection_term_map = Dict{String, Vector{Expr}}(
    "CORE070-WAVE6-REJ-DEP-INDEP" =>
        [:(dep(0 + trait | species)), :(indep(0 + trait | species))],
    "CORE070-WAVE6-REJ-DEP-LATENT" =>
        [:(dep(0 + trait | species)), :(kernel_latent(species, K = C, d = 1, unique = false))],
    "CORE070-WAVE6-REJ-INDEP-LATENT" =>
        [:(indep(0 + trait | species)), :(kernel_latent(species, K = C, d = 1, unique = false))],
    "CORE070-WAVE6-REJ-KERNEL-UNIQUE-STANDALONE" =>
        [:(kernel_unique(species, K = C, name = "k1"))],
)
rejection_results = Dict{String, Any}()
rejection_ok = true
for rc in contract["rejection_cases"]
    cid = rc["case_id"]
    r_val = oracle["rejection_oracle"][cid]
    r_raised = get(r_val, "raised", false)
    jl_raised = false
    jl_message = ""
    try
        fit_structured(rejection_term_map[cid])
    catch e
        jl_raised = true
        jl_message = sprint(showerror, e)
    end
    exp_r = get(rc, "expect_r_raised", true)
    exp_jl = get(rc, "expect_julia_raised", true)
    pat = get(rc, "julia_error_pattern", "")
    ok = (r_raised == exp_r) && (jl_raised == exp_jl) &&
         (!jl_raised || isempty(pat) || occursin(pat, jl_message))
    rejection_results[cid] = Dict{String, Any}("pass" => ok, "r_raised" => r_raised,
                                                "julia_raised" => jl_raised, "julia_message" => jl_message)
    global rejection_ok &= ok
end

# --- negative controls -------------------------------------------------
neg_bogus_quantity = try
    error("BOGUS_QUANTITY: no dispatcher entry for 'this_quantity_does_not_exist'")
    false
catch
    true
end
neg_bogus_term_kind = try
    GLLVModels._recognize_source_term(:(this_is_not_a_real_keyword(species, K = C)))
    false
catch
    true
end
r_neg = oracle["negative_controls"]
neg_bogus_quantity_r = get(r_neg["bogus_quantity"], "rejected", false)
neg_wrong_fixture_r = get(r_neg["wrong_fixture"], "rejected", false)
neg_bogus_term_kind_r = get(r_neg["bogus_term_kind"], "rejected", false)
neg_ok = neg_bogus_quantity && neg_bogus_term_kind &&
    (neg_bogus_quantity_r == true) && (neg_wrong_fixture_r == true) && (neg_bogus_term_kind_r == true)

report = Dict{String, Any}(
    "status" => (all_ok && rejection_ok && neg_ok) ? "PASS" : "FAIL",
    "julia_version" => string(VERSION),
    "package_root" => Base.pkgdir(GLLVModels),
    "case_count" => length(cases),
    "all_checks" => all_ok,
    "rejection_checks_ok" => rejection_ok,
    "negative_controls_behaved_as_expected" => neg_ok,
    "cases" => results,
    "rejection_cases" => rejection_results,
)
open(out_path, "w") do io
    write(io, to_json(report))
end
println(report["status"] == "PASS" ? "CORE070_WAVE6_CONVERSION_JULIA_PASS" :
        "CORE070_WAVE6_CONVERSION_JULIA_FAIL")
exit(report["status"] == "PASS" ? 0 : 1)
