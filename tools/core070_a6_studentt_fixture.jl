# A6 Student-t interior-nu paired fixture -- Julia stage + verifier
# (maintainer round2-3 #11; docs/dev-log/decisions/2026-09-01-maintainer-decisions-round2-3.md #11).
#
# Companion to tools/core070_a6_studentt_fixture.R. Design rationale and the
# full "why this fixture, why these conventions" account live in
# docs/dev-log/core070/a6-studentt-notes.md; the frozen numeric contract lives
# in docs/dev-log/core070/a6-studentt-contract.json (reference_commit
# b4d5fee64def88bc768dda1f1f77c29b295edd86, paired tolerance 1e-4).
#
# REVISION (post-Totoro run suite-run-01/a6-run): the ORIGINAL single free-nu
# case (Julia disp_group=:shared vs R's default per-trait df) came back
# HONESTLY FAILED with a structural finding, not a numeric near-miss: R's
# student() with df=NULL estimates ONE degrees-of-freedom PER TRAIT
# (documented behaviour -- R/gllvmTMB.R:167-168, "The student() family fits
# one log-sigma and one log(df-1) per trait"; confirmed in
# R/fit-multi.R:5317-5349's per-trait `dispersion_trait_map`/pin machinery),
# while GLLVModels.jl's default `disp_group = :shared` fits ONE shared
# degrees-of-freedom across all traits. Same dispersion-structure class as
# the NB2 benchmark's shared-r-vs-per-trait-phi lesson: comparing two
# genuinely different models numerically was never going to converge, and
# the ~0.05-scale loading divergence measured on Totoro is model-mismatch
# noise, not an engine defect.
#
# Two cases now run in ONE invocation, using the frozen R `student()`
# constructor's documented `df` argument to control this directly
# (`student(link = "identity", df = <n>)` PINS that trait's degrees of
# freedom at exactly `<n>` for every trait using the family --
# R/fit-multi.R:5325-5342 -- so a single scalar `nu` fix on the Julia side
# is model-matched to R with a scalar `df`, unlike the free-nu case above):
#
#   "fixed"  (PRIMARY, GATING): R fits student(link="identity", df=6);
#            Julia fits fit_studentt_gllvm(Y; K=1, nu=6.0) -- the
#            FAMILY-09-FIXED-SHAPE path (the `nu::Real` branch of
#            fit_studentt_gllvm, src/families/studentt.jl). Matched models;
#            paired numeric comparison at PAIRED_TOL, exactly as before.
#            This case alone determines the exit code.
#   "free"   (SECONDARY, NON-GATING): R fits student(link="identity")
#            (df=NULL, per-trait); Julia fits fit_studentt_gllvm(Y; K=1)
#            (disp_group=:shared, one shared nu). RECORDED STRUCTURAL
#            FINDING ONLY -- values from both sides are printed for the
#            record, but no numeric tolerance is asserted and this case
#            never affects the exit code (dispersion-structure divergence,
#            expected-mismatch class; see contract JSON
#            "dispersion_structure_finding").
#
# This single script plays THREE house-convention roles because the owned
# file set for this leaf is {R, jl, contract.json, notes.md} -- no separate
# tools/core070_..._verify.py, so the "verifier" (argv 2, coverage guard,
# soft-fail, --self-test >=4 mutations) lives here rather than in a third
# file, mirroring e.g. tools/core070_wave6_conversion_batch.jl's
# soft-fail-per-case pattern and tools/core070_verify_wave6_conversion_batch.py's
# --self-test mutation battery, folded into one process:
#
#   julia tools/core070_a6_studentt_fixture.jl <data_path> <r_output_path>
#
#   1. `<data_path>` missing        -> GENERATE the fixture (fixed seed),
#      write Y, fit both Julia cases, print instructions for the R stage,
#      exit 0. This is the coverage guard for phase 1: nothing is claimed
#      verified yet.
#   2. `<data_path>` present, `<r_output_path>` missing -> re-fit both cases
#      on the existing data, print that the R stage has not run yet, exit 1
#      (NOT a pass -- a missing oracle side is never silently treated as
#      agreement; the soft-fail convention this repo uses for a
#      missing/null oracle value, e.g. tools/core070_wave6_conversion_batch.jl:158).
#   3. Both present -> "fixed" case: PAIRED COMPARISON at 1e-4 (loglik, beta,
#      sigma, nu, loading up to the K=1 sign indeterminacy) and BOTH-ENGINE
#      health (fit.converged / R's inline health check) -- gates the exit
#      code. "free" case: printed as a recorded structural finding, never
#      gates. An interior-nu fixture where EITHER engine failing health on
#      the GATING case is a recorded finding, never fudged away (round2-3
#      #11's explicit instruction).
#
#   julia tools/core070_a6_studentt_fixture.jl --self-test
#
#   Builds a SYNTHETIC agreeing (julia_result, r_result) pair for the
#   "fixed" case, with no data file, no fit, no R -- confirms `_a6_compare`
#   accepts it, then mutates it >=4 independent ways and confirms every
#   mutation is rejected. Runs with `julia --project=.` alone; never
#   substitutes for the real two-stage run.

using GLLVModels
using Random, Distributions, LinearAlgebra, DelimitedFiles, Statistics
using SHA, TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
# Pin switch (D-294/D-295): GLLVM_PARITY_PIN unset means P1; "P0" keeps the P0
# contract; "P1" names the twin written by tools/core070_family_p1_contract.py.
# The reference commit comes from tools/core070_oracle_pins.toml and must equal
# the contract's own. Any other value stops.
const PARITY_PIN = uppercase(strip(get(ENV, "GLLVM_PARITY_PIN", "P1")))
PARITY_PIN in ("P0", "P1") || error("GLLVM_PARITY_PIN must be P0 or P1, got $(repr(PARITY_PIN))")
const CONTRACT_PATH = joinpath(ROOT, PARITY_PIN == "P1" ?
    "docs/dev-log/core070/true-parity-latest/a6-studentt-contract-p1.json" :
    "docs/dev-log/core070/a6-studentt-contract.json")
const REFERENCE_COMMIT = TOML.parsefile(joinpath(ROOT, "tools/core070_oracle_pins.toml"))[PARITY_PIN]["reference_commit"]
let m = match(r"\"reference_commit\": \"([0-9a-f]{40})\"", read(CONTRACT_PATH, String))
    m !== nothing && m.captures[1] == REFERENCE_COMMIT ||
        error("$(CONTRACT_PATH) is not pinned at $(PARITY_PIN) $(REFERENCE_COMMIT)")
end
const PAIRED_TOL = 1e-4

# ---------------------------------------------------------------------------
# Fixture design constants (frozen; mirrored in the R stage and in the
# contract JSON -- K is a fixed design constant of THIS fixture, not a
# runtime argument, matching the argv-2 house convention).
# ---------------------------------------------------------------------------
const FIXTURE_SEED = 20260901
const FIXTURE_P = 5
const FIXTURE_K = 1
const FIXTURE_N = 250
const FIXTURE_BETA_TRUE = [0.5, -0.3, 0.2, 0.8, -0.6]
const FIXTURE_LOADING_TRUE = [0.9, -0.7, 0.5, 0.6, -0.4]   # p x K, K = 1
const FIXTURE_SIGMA_TRUE = 0.7
const FIXTURE_NU_TRUE = 6.0   # moderate, genuinely INTERIOR -- both engines expected to land away from any boundary; also the PINNED df for the "fixed" primary case

"""
    generate_fixture(rng) -> Matrix{Float64}

Simulate the p x n Student-t GLLVModels response matrix at the frozen fixture
design constants above: `y_tj = beta_t + Lambda_t * z_j + sigma * eps_tj`,
`z_j ~ N(0,1)` iid across sites, `eps_tj ~ t_(nu_true)` (standard, scale 1).
Deliberately genuinely heavy-tailed (nu_true = 6, not the pure-Gaussian
construction of test_studentt_boundary_honesty.jl / test_studentt_boundary.jl)
so the estimated-nu MLE is expected to land INTERIOR on both engines. The
SAME data feeds both the "fixed" and "free" cases below.
"""
function generate_fixture(rng)
    p, K, n = FIXTURE_P, FIXTURE_K, FIXTURE_N
    Λ = reshape(FIXTURE_LOADING_TRUE, p, K)
    Y = zeros(p, n)
    tdist = TDist(FIXTURE_NU_TRUE)
    for j in 1:n
        z = randn(rng, K)
        for t in 1:p
            Y[t, j] = FIXTURE_BETA_TRUE[t] + dot(view(Λ, t, :), z) + FIXTURE_SIGMA_TRUE * rand(rng, tdist)
        end
    end
    return Y
end

# ---------------------------------------------------------------------------
# R-output TSV readback: `key\tvalue` (scalar) or `key\tindex\tvalue` (vector),
# sprintf("%.17g", .) on the R side -- mirrors
# tools/core070_student_warmstart_readback.R's readback shape. Keys are
# prefixed `fixed_` / `free_` by the R stage to carry both cases in one file.
# ---------------------------------------------------------------------------
function read_r_output(path::AbstractString)
    scalars = Dict{String, Float64}()
    vectors = Dict{String, Vector{Float64}}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        parts = split(line, '\t')
        if length(parts) == 2
            scalars[parts[1]] = parse(Float64, parts[2])
        elseif length(parts) == 3
            key = parts[1]
            idx = parse(Int, parts[2])
            v = get!(vectors, key) do
                Float64[]
            end
            length(v) < idx && resize!(v, idx)
            v[idx] = parse(Float64, parts[3])
        else
            error("read_r_output: unparseable line in $path: $(repr(line))")
        end
    end
    return (scalars = scalars, vectors = vectors)
end

# ---------------------------------------------------------------------------
# The GATING comparison, used ONLY for the "fixed" (matched-model) case.
# Operates on plain Dicts so --self-test can exercise it against a
# SYNTHETIC pair with no fit and no R. Returns (pass, messages).
# ---------------------------------------------------------------------------
function _a6_compare(julia_result::Dict, r_result::Dict; tol::Real = PAIRED_TOL)
    messages = String[]
    ok = true
    fail!(msg) = (ok = false; push!(messages, msg))

    # Coverage guard: every quantity this comparison claims to check must
    # actually be present on both sides -- a missing key is a soft-fail, not
    # a silent skip.
    required = ("loglik", "converged", "nu_boundary", "beta", "sigma", "nu", "loading")
    for k in required
        haskey(julia_result, k) || fail!("julia_result missing required key $(repr(k))")
    end
    required_r = ("loglik", "healthy", "nu_at_boundary", "beta", "sigma_student", "df_student", "loading")
    for k in required_r
        haskey(r_result, k) || fail!("r_result missing required key $(repr(k))")
    end
    ok || return (false, messages)

    # Both-engine health at an INTERIOR, MATCHED-MODEL fixture: a failure
    # here is a recorded finding, never fudged away (round2-3 #11).
    julia_result["converged"] === true || fail!("Julia fit reports converged=false at the interior-nu fixed-shape fixture")
    julia_result["nu_boundary"] === false || fail!("Julia fit reports nu_boundary=true at the interior-nu fixed-shape fixture")
    r_result["healthy"] == 1.0 || fail!("R fit reports healthy=FALSE at the interior-nu fixed-shape fixture")
    r_result["nu_at_boundary"] == 0.0 || fail!("R fit reports nu_at_boundary=TRUE at the interior-nu fixed-shape fixture")

    # Paired numeric agreement at PAIRED_TOL. Sign-align the loading vector
    # first: K = 1 latent-variable GLLVMs are identified only up to a sign
    # flip of (Lambda, z), so the two engines' MLE loadings may differ by an
    # overall sign even when they are the SAME point in likelihood terms.
    jl_loading = Vector{Float64}(julia_result["loading"])
    r_loading = Vector{Float64}(r_result["loading"])
    if length(jl_loading) == length(r_loading) && length(jl_loading) > 0
        if dot(jl_loading, r_loading) < 0
            jl_loading = -jl_loading
        end
    else
        fail!("loading length mismatch: julia=$(length(jl_loading)) r=$(length(r_loading))")
    end

    close_enough(a, b) = isfinite(a) && isfinite(b) && abs(a - b) <= tol

    close_enough(julia_result["loglik"], r_result["loglik"]) ||
        fail!("loglik disagreement: julia=$(julia_result["loglik"]) r=$(r_result["loglik"]) " *
              "delta=$(abs(julia_result["loglik"] - r_result["loglik"]))")

    jl_beta = Vector{Float64}(julia_result["beta"])
    r_beta = Vector{Float64}(r_result["beta"])
    length(jl_beta) == length(r_beta) || fail!("beta length mismatch: julia=$(length(jl_beta)) r=$(length(r_beta))")
    if length(jl_beta) == length(r_beta)
        for i in eachindex(jl_beta)
            close_enough(jl_beta[i], r_beta[i]) ||
                fail!("beta[$i] disagreement: julia=$(jl_beta[i]) r=$(r_beta[i])")
        end
    end

    jl_sigma = Vector{Float64}(julia_result["sigma"])
    r_sigma = Vector{Float64}(r_result["sigma_student"])
    length(jl_sigma) == length(r_sigma) || fail!("sigma length mismatch: julia=$(length(jl_sigma)) r=$(length(r_sigma))")
    if length(jl_sigma) == length(r_sigma)
        for i in eachindex(jl_sigma)
            close_enough(jl_sigma[i], r_sigma[i]) ||
                fail!("sigma[$i] disagreement: julia=$(jl_sigma[i]) r=$(r_sigma[i])")
        end
    end

    # nu: R always reports one df_student entry PER TRAIT (even when pinned,
    # every entry is identical -- R/fit-multi.R:5325-5342). A Julia fit with
    # a scalar fixed `nu` (disp_group irrelevant to nu when nu is a Real) is
    # ONE number, not a length-p vector -- broadcast it against every R
    # per-trait entry rather than requiring equal lengths (a reporting-shape
    # difference, not a real model mismatch, given nu was pinned identically
    # on both sides).
    jl_nu = Vector{Float64}(julia_result["nu"])
    r_nu = Vector{Float64}(r_result["df_student"])
    if length(jl_nu) == 1 && length(r_nu) > 1
        for i in eachindex(r_nu)
            close_enough(jl_nu[1], r_nu[i]) ||
                fail!("nu[$i] disagreement (julia scalar broadcast): julia=$(jl_nu[1]) r=$(r_nu[i])")
        end
    elseif length(jl_nu) == length(r_nu)
        for i in eachindex(jl_nu)
            close_enough(jl_nu[i], r_nu[i]) ||
                fail!("nu[$i] disagreement: julia=$(jl_nu[i]) r=$(r_nu[i])")
        end
    else
        fail!("nu length mismatch: julia=$(length(jl_nu)) r=$(length(r_nu))")
    end

    if length(jl_loading) == length(r_loading)
        for i in eachindex(jl_loading)
            close_enough(jl_loading[i], r_loading[i]) ||
                fail!("loading[$i] disagreement (sign-aligned): julia=$(jl_loading[i]) r=$(r_loading[i])")
        end
    end

    return (ok, messages)
end

# ---------------------------------------------------------------------------
# The "free" case is NEVER gating: dispersion-structure divergence is the
# EXPECTED outcome (R per-trait nu vs Julia shared nu), not a defect to
# tolerance-check. This just formats both sides' values for the record.
# ---------------------------------------------------------------------------
function _a6_report_free(julia_free::Dict, r_free::Dict)
    lines = String[]
    push!(lines, "CORE070_A6_STUDENTT_FREE_CASE_STRUCTURAL_FINDING (non-gating; dispersion-structure divergence, expected-mismatch class)")
    push!(lines, "  julia (disp_group=:shared, ONE shared nu): converged=$(get(julia_free, "converged", missing)) " *
                 "nu_boundary=$(get(julia_free, "nu_boundary", missing)) " *
                 "nu=$(get(julia_free, "nu", missing)) sigma=$(get(julia_free, "sigma", missing)) " *
                 "loglik=$(get(julia_free, "loglik", missing))")
    push!(lines, "  r (student(df=NULL), PER-TRAIT nu -- R/gllvmTMB.R:167-168): healthy=$(get(r_free, "healthy", missing)) " *
                 "nu_at_boundary=$(get(r_free, "nu_at_boundary", missing)) " *
                 "df_student=$(get(r_free, "df_student", missing)) sigma_student=$(get(r_free, "sigma_student", missing)) " *
                 "loglik=$(get(r_free, "loglik", missing))")
    if haskey(julia_free, "loglik") && haskey(r_free, "loglik") &&
       isfinite(julia_free["loglik"]) && isfinite(r_free["loglik"])
        push!(lines, "  loglik delta (informational only, NOT gated): $(abs(julia_free["loglik"] - r_free["loglik"]))")
    end
    return lines
end

# ---------------------------------------------------------------------------
# --self-test: synthetic pair for the GATING "fixed" case only, no data/R
# required. The "free" case has no tolerance gate to self-test.
# ---------------------------------------------------------------------------
function _a6_synthetic_pair()
    # Realistic shapes for the "fixed" case: Julia's disp_group = :species fit
    # has PER-TRAIT sigma (length p) and a SCALAR pinned nu (length 1); R
    # always reports both per trait (length p), with df_student's p entries
    # all identical (the pin) -- exercised via the nu-broadcast path in
    # `_a6_compare`.
    julia_result = Dict{String, Any}(
        "loglik" => -812.345, "converged" => true, "nu_boundary" => false,
        "beta" => copy(FIXTURE_BETA_TRUE), "sigma" => fill(FIXTURE_SIGMA_TRUE, FIXTURE_P),
        "nu" => [FIXTURE_NU_TRUE], "loading" => copy(FIXTURE_LOADING_TRUE),
    )
    r_result = Dict{String, Any}(
        "loglik" => -812.345 + 3e-5, "healthy" => 1.0, "nu_at_boundary" => 0.0,
        "beta" => copy(FIXTURE_BETA_TRUE), "sigma_student" => fill(FIXTURE_SIGMA_TRUE, FIXTURE_P),
        "df_student" => fill(FIXTURE_NU_TRUE, FIXTURE_P), "loading" => -copy(FIXTURE_LOADING_TRUE),  # sign flip: must still pass
    )
    return julia_result, r_result
end

function run_self_test()
    julia_result, r_result = _a6_synthetic_pair()
    ok, messages = _a6_compare(julia_result, r_result)
    ok || error("synthetic agreeing pair rejected: $(join(messages, "; "))")

    rejected = String[]
    function expect_rejected(mutate!, label)
        jl2 = deepcopy(julia_result)
        r2 = deepcopy(r_result)
        mutate!(jl2, r2)
        ok2, _ = _a6_compare(jl2, r2)
        ok2 && error("REJECTED MUTATION FAILED TO BE CAUGHT: $label")
        push!(rejected, label)
    end

    expect_rejected("loglik moved past tolerance") do jl2, r2
        jl2["loglik"] = jl2["loglik"] + 10 * PAIRED_TOL
    end
    expect_rejected("beta[1] moved past tolerance") do jl2, r2
        jl2["beta"][1] = jl2["beta"][1] + 10 * PAIRED_TOL
    end
    expect_rejected("sigma[1] moved past tolerance (per-trait)") do jl2, r2
        jl2["sigma"][1] = jl2["sigma"][1] + 10 * PAIRED_TOL
    end
    expect_rejected("julia scalar nu moved past tolerance") do jl2, r2
        jl2["nu"][1] = jl2["nu"][1] + 10 * PAIRED_TOL
    end
    expect_rejected("one R per-trait df_student entry moved past tolerance (broadcast path)") do jl2, r2
        r2["df_student"][3] = r2["df_student"][3] + 10 * PAIRED_TOL
    end
    expect_rejected("Julia side reports converged=false") do jl2, r2
        jl2["converged"] = false
    end
    expect_rejected("Julia side reports nu_boundary=true") do jl2, r2
        jl2["nu_boundary"] = true
    end
    expect_rejected("R side reports healthy=FALSE") do jl2, r2
        r2["healthy"] = 0.0
    end
    expect_rejected("R side reports nu_at_boundary=TRUE") do jl2, r2
        r2["nu_at_boundary"] = 1.0
    end
    expect_rejected("required key missing from julia_result") do jl2, r2
        delete!(jl2, "sigma")
    end

    length(rejected) >= 4 || error("only $(length(rejected)) rejected mutations ran, need >=4")
    println("CORE070_A6_STUDENTT_SELF_TEST_OK rejected_mutations=$(length(rejected))")
end

# ---------------------------------------------------------------------------
# Fit both Julia cases and package each into the comparison Dict shape.
#   "fixed": nu pinned at FIXTURE_NU_TRUE, disp_group = :species so sigma is
#            estimated PER TRAIT -- R's student() always reports sigma_student
#            per trait regardless of whether df is pinned (R/gllvmTMB.R:167-168:
#            "one log-sigma ... per trait" is unconditional; only df pinning
#            is optional). Model-matched to R's student(df = FIXTURE_NU_TRUE):
#            per-trait sigma on both sides, and a single scalar df/nu pinned
#            identically for every trait on both sides (R re-reports that
#            scalar once per trait; see the nu broadcast in `_a6_compare`).
#   "free":  Julia default (disp_group = :shared, nu estimated -- ONE
#            shared value across traits; NOT model-matched to R's
#            per-trait-sigma-and-per-trait-nu default, by design -- see the
#            file header).
# ---------------------------------------------------------------------------
function fit_julia_side(Y::AbstractMatrix)
    fixed_fit = fit_studentt_gllvm(Y; K = FIXTURE_K, nu = FIXTURE_NU_TRUE, disp_group = :species)
    free_fit = fit_studentt_gllvm(Y; K = FIXTURE_K)
    to_dict(fit) = Dict{String, Any}(
        "loglik" => fit.loglik,
        "converged" => fit.converged,
        "nu_boundary" => fit.nu_boundary,
        "beta" => copy(fit.β),
        "sigma" => fit.σ isa Real ? [fit.σ] : copy(fit.σ),
        "nu" => fit.ν isa Real ? [fit.ν] : copy(fit.ν),
        "loading" => vec(fit.Λ),
    )
    return to_dict(fixed_fit), to_dict(free_fit), fixed_fit, free_fit
end

function read_r_result(path::AbstractString, case_prefix::AbstractString)
    scalars, vectors = read_r_output(path)
    key(name) = case_prefix * "_" * name
    return Dict{String, Any}(
        "loglik" => scalars[key("loglik")],
        "healthy" => scalars[key("healthy")],
        "nu_at_boundary" => scalars[key("nu_at_boundary")],
        "beta" => vectors[key("beta")],
        "sigma_student" => vectors[key("sigma_student")],
        "df_student" => vectors[key("df_student")],
        "loading" => vectors[key("loading")],
    )
end

# ---------------------------------------------------------------------------
# Optional results file (CORE070_A6_RESULTS_TOML): both sides' values for both
# cases, the gating verdict and its messages, the pin, and the hashes of the
# inputs, so a receipt can recompute |R - Julia| from saved values. The R
# stage's <r_output>.source-pin.tsv, when present, is carried in verbatim.
# ---------------------------------------------------------------------------
function write_results(r_output_path, data_path, julia_fixed, r_fixed, julia_free, r_free, ok, messages)
    path = get(ENV, "CORE070_A6_RESULTS_TOML", "")
    isempty(path) && return nothing
    pinfile = r_output_path * ".source-pin.tsv"
    pin = isfile(pinfile) ? Dict{String, Any}(String(a) => String(b) for (a, b) in
                                              (split(l, '\t'; limit = 2) for l in eachline(pinfile) if !isempty(l))) :
          Dict{String, Any}()
    sha(f) = bytes2hex(open(sha256, f))
    report = Dict{String, Any}(
        "parity_pin" => PARITY_PIN, "reference_commit" => REFERENCE_COMMIT,
        "contract" => relpath(CONTRACT_PATH, ROOT), "contract_sha256" => sha(CONTRACT_PATH),
        "paired_tol" => PAIRED_TOL, "verdict" => ok ? "PASS" : "FAIL", "messages" => messages,
        "data_sha256" => sha(data_path), "r_output_sha256" => sha(r_output_path),
        "r_source_pin" => pin,
        "julia_fixed" => julia_fixed, "r_fixed" => r_fixed, "julia_free" => julia_free, "r_free" => r_free)
    open(io -> TOML.print(io, report), path, "w")
    return nothing
end

# ---------------------------------------------------------------------------
# Two-stage generate-or-verify driver.
# ---------------------------------------------------------------------------
function main(argv)
    if length(argv) == 1 && argv[1] == "--self-test"
        run_self_test()
        return 0
    end
    length(argv) == 2 || error(
        "usage: julia tools/core070_a6_studentt_fixture.jl <data_path> <r_output_path>" *
        "  (or: julia tools/core070_a6_studentt_fixture.jl --self-test)")
    data_path, r_output_path = argv

    if !isfile(data_path)
        rng = MersenneTwister(FIXTURE_SEED)
        Y = generate_fixture(rng)
        writedlm(data_path, Y)
        println("CORE070_A6_STUDENTT_FIXTURE_GENERATED $(data_path) p=$(FIXTURE_P) K=$(FIXTURE_K) n=$(FIXTURE_N) " *
                "seed=$(FIXTURE_SEED) nu_true=$(FIXTURE_NU_TRUE) sigma_true=$(FIXTURE_SIGMA_TRUE)")
        println("Next: Rscript --vanilla tools/core070_a6_studentt_fixture.R $(data_path) $(r_output_path)")
        return 0   # coverage guard phase 1: generation only, nothing verified yet
    end

    Y = readdlm(data_path)
    size(Y, 1) == FIXTURE_P || error("data_path row count $(size(Y,1)) != fixed design p=$(FIXTURE_P)")
    julia_fixed, julia_free, fixed_fit, free_fit = fit_julia_side(Y)
    println("CORE070_A6_STUDENTT_JULIA_FIT_FIXED loglik=$(fixed_fit.loglik) converged=$(fixed_fit.converged) " *
            "nu_boundary=$(fixed_fit.nu_boundary) nu=$(fixed_fit.ν) sigma=$(fixed_fit.σ)")
    println("CORE070_A6_STUDENTT_JULIA_FIT_FREE loglik=$(free_fit.loglik) converged=$(free_fit.converged) " *
            "nu_boundary=$(free_fit.nu_boundary) nu=$(free_fit.ν) sigma=$(free_fit.σ)")

    if !isfile(r_output_path)
        println("CORE070_A6_STUDENTT_AWAITING_R $(r_output_path) does not exist yet -- Julia fits done and " *
                "printed above; run the R stage next, then re-run this exact command to verify.")
        return 1   # soft-fail: a missing oracle side is never silently a pass
    end

    r_fixed = read_r_result(r_output_path, "fixed")
    r_free = read_r_result(r_output_path, "free")

    for line in _a6_report_free(julia_free, r_free)
        println(line)
    end

    ok, messages = _a6_compare(julia_fixed, r_fixed)
    write_results(r_output_path, data_path, julia_fixed, r_fixed, julia_free, r_free, ok, messages)
    if ok
        println("CORE070_A6_STUDENTT_PAIRED_VERIFIED case=fixed tol=$(PAIRED_TOL) reference_commit=$(REFERENCE_COMMIT)")
        return 0
    else
        println("CORE070_A6_STUDENTT_PAIRED_FAILED case=fixed tol=$(PAIRED_TOL)")
        for m in messages
            println("  - ", m)
        end
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
