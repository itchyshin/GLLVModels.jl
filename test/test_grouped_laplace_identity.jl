# test/test_grouped_laplace_identity.jl — leaf-S7b identity gate (G7b.1).
#
# Written BEFORE the CHOLMOD symbolic-reuse change lands in the grouped
# Laplace kernel (`joint_grouped_laplace_loglik`, src/grouped_laplace.jl —
# NOTE: the leaf-S7b ledger and lane arcs.md both name this file as
# `src/families/grouped_laplace.jl`; that path does not exist in this repo,
# the real file is `src/grouped_laplace.jl` (`include`d at
# src/GLLVModels.jl:114) — flagged, not silently corrected).
#
# TDD contract: the IDENTITY assertions below pin baseline numbers captured
# from the UNMODIFIED code (origin/main 69a69b0a0, before ANY change in this
# arc) and must hold both BEFORE and AFTER the reuse change — a pure
# performance optimisation must not move the fitted answer by more than
# floating-point noise. The COUNTER assertions read
# `GLLVModels._grouped_chol_stats()` (added by the change) and are EXPECTED
# TO FAIL on the unmodified baseline (the accessor does not exist yet) and
# PASS once the reuse lands. If this test ever needs a baseline number
# changed, the cause must be a genuine numerical fix, never drift introduced
# by the reuse change — re-derive on origin/main by hand, never copy the
# post-fix value back in.
#
# Runnable two ways:
#   - `include`d from test/runtests.jl as a normal @testset
#   - `julia --project=. test/test_grouped_laplace_identity.jl --gate {identity|warm_identity}`
#     (prints "GATE G7b.1/G7c.1 PASS" / "... FAIL <reason>", exit 0/1)
#
# S7c (leaf-S7c G7c.1, `--gate warm_identity`) extends this file: it does NOT
# replace the S7b section above. `src/grouped_laplace.jl` gained a `b_init`
# keyword on `joint_grouped_laplace_loglik` (warm-start the inner Newton
# solve instead of `zeros(m)`) and `src/grouped_nongaussian_fit.jl` gained a
# `warm_start_inner` keyword on `fit_grouped_nongaussian` (opt-in, default
# `false`, threaded through `fit_gllvm(...; grouping=...)` via `kwargs...`)
# that caches the last converged mode across the ~118 inner-Laplace-fit calls
# one outer optimisation makes and feeds it back in as `b_init`.
# `src/grouped_nongaussian_fit.jl` is NOT in leaf-S7c.md's OWNS list (only
# `src/grouped_laplace.jl` is) but is the necessary caller that actually
# threads the cache across calls — flagged here, not silently worked around,
# mirroring the leaf-S7b file-path mismatch note above.

using Test, GLLVModels, SparseArrays, LinearAlgebra, Random
using Distributions: Poisson

const _RESULTS = Bool[]
const _REASONS = String[]

function _check!(ok::Bool, msg::AbstractString)
    push!(_RESULTS, ok)
    ok || push!(_REASONS, msg)
    @test ok
    return ok
end

# ---------------------------------------------------------------------------
# Baseline constants — captured 2026-09-19 on origin/main 69a69b0a0 (before
# any change in this arc), via exactly the calls in `fixture_a`/`fixture_b`
# below. See bench/results/grouped_glmm_d56bccea4.tsv (leaf-S4) for the
# obj_calls / inner_newton_iters_sum / fresh_cholmod_analyses provenance.
# ---------------------------------------------------------------------------

# --- fixture A: the Latte 200x5 GLMM (bench/fixtures/glmm_200x5.csv), fitted
# exactly as bench/profile_grouped_glmm.jl / f3_ours_glmm.jl does. ---
const BASELINE_A_CONVERGED = true
const BASELINE_A_ITERATIONS = 3               # outer Optim iterations
const BASELINE_A_LOGLIK = -2025.469254255527403
const BASELINE_A_PARAMETERS = [0.9211786339273272, -0.3668330777412767]
# calls/fresh below are measured DIRECTLY on the real joint_grouped_laplace_
# loglik call path (via the _GROUPED_CHOL_STATS counter added by this arc,
# with the reuse branch temporarily forced off to reproduce pre-fix
# behaviour) -- NOT the S4 shadow-replica's counts (103 calls / 1345 fresh),
# which turned out to under-count slightly: the shadow driver in
# bench/profile_grouped_glmm.jl reaches the identical converged loglik but is
# not a bit-for-bit reimplementation of fit_grouped_nongaussian's optimiser
# calls, so its call count is a good order-of-magnitude estimate, not a
# pinnable identity. 118/1540 are the true, verified pre-fix numbers.
const BASELINE_A_OBJ_CALLS = 118              # true inner Laplace-fit call count
const BASELINE_A_INNER_ITERS_SUM = 711        # = (1540 - 118) / 2
const BASELINE_A_FRESH_CHOLESKY = 1540        # true pre-fix fresh cholesky() call count

# --- fixture B: "crossed incidence" (test/test_grouped_laplace.jl), a direct
# joint_grouped_laplace_loglik call with m=2 unknowns (Newton actually
# iterates, unlike the m<=1 cases in that file). ---
const BASELINE_B_STATUS = :ok
const BASELINE_B_CONVERGED = true
const BASELINE_B_ITERATIONS = 4
const BASELINE_B_LOGLIK = -6.419975970680682
const BASELINE_B_MODE = [0.5103459578769107, -0.0535798048824418]
const BASELINE_B_LOGDET = 3.56577252913429

function fixture_a()
    # No header, two comma-separated integer columns (count, group id).
    # Read with Base only (no DelimitedFiles): that package is not a declared
    # test/Project.toml dependency, and Pkg.test()'s sandboxed test
    # environment (unlike an interactive --project=. run) does not fall back
    # to the implicit stdlib load path for it.
    path = joinpath(@__DIR__, "..", "bench", "fixtures", "glmm_200x5.csv")
    isfile(path) || error("fixture missing: $path")
    y = Int[]; group = Int[]
    for line in eachline(path)
        isempty(line) && continue
        cols = split(line, ',')
        push!(y, parse(Int, cols[1]))
        push!(group, parse(Int, cols[2]))
    end
    Y1 = reshape(Float64.(y), 1, :)
    terms = [GLLVModels.GroupingTerm(:unit; mode = :indep)]
    return Y1, terms, group
end

function fixture_b()
    A = sparse([1.0 1.0; 1.0 0.0; 0.0 1.0])
    W = GLLVModels.grouped_trait_design(A, ones(1, 1))
    return Poisson(), [3.0, 4.0, 2.0], ones(3), ones(3, 1), [log(2.0)], W
end

function run_identity_checks()
    empty!(_RESULTS); empty!(_REASONS)

    # ---- fixture A: full grouped fit, identity vs baseline ----
    # `warm_start_inner = false` pins this to the SAME cold-start path
    # origin/main always used — S7c (leaf-S7c) flipped `fit_gllvm`'s default
    # to warm-started, which is a DIFFERENT optimiser trajectory (see
    # `run_warm_identity_checks` below); this baseline must stay cold to
    # keep meaning what it always meant.
    Y1, terms, group = fixture_a()
    fit = GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group,
        warm_start_inner = false)
    _check!(fit.converged == BASELINE_A_CONVERGED, "fixture A: converged mismatch")
    _check!(fit.iterations == BASELINE_A_ITERATIONS,
            "fixture A: outer iterations mismatch ($(fit.iterations) vs $(BASELINE_A_ITERATIONS))")
    _check!(isapprox(fit.loglik, BASELINE_A_LOGLIK; rtol = 1e-8),
            "fixture A: loglik mismatch ($(fit.loglik) vs $(BASELINE_A_LOGLIK))")
    _check!(isapprox(fit.parameters, BASELINE_A_PARAMETERS; rtol = 1e-8),
            "fixture A: parameters mismatch ($(fit.parameters) vs $(BASELINE_A_PARAMETERS))")

    # ---- fixture B: direct joint_grouped_laplace_loglik call, identity vs baseline ----
    r = GLLVModels.joint_grouped_laplace_loglik(fixture_b()...; link = GLLVModels.LogLink())
    _check!(r.status === BASELINE_B_STATUS, "fixture B: status mismatch ($(r.status))")
    _check!(r.converged == BASELINE_B_CONVERGED, "fixture B: converged mismatch")
    _check!(r.iterations == BASELINE_B_ITERATIONS,
            "fixture B: iterations mismatch ($(r.iterations) vs $(BASELINE_B_ITERATIONS))")
    _check!(isapprox(r.loglik, BASELINE_B_LOGLIK; rtol = 1e-8), "fixture B: loglik mismatch")
    _check!(isapprox(r.mode, BASELINE_B_MODE; rtol = 1e-8), "fixture B: mode mismatch")
    _check!(isapprox(r.logdet_precision, BASELINE_B_LOGDET; rtol = 1e-8), "fixture B: logdet mismatch")

    # ---- NEW counter assertions: fresh CHOLMOD symbolic analyses 2 -> 0
    # after the first, per inner Laplace-fit call. Not present pre-fix. ----
    has_stats = isdefined(GLLVModels, :_grouped_chol_stats_reset!) &&
                isdefined(GLLVModels, :_grouped_chol_stats)
    if !has_stats
        _check!(false, "GLLVModels._grouped_chol_stats[_reset!] not defined — " *
                       "the CHOLMOD symbolic-reuse change has not landed yet")
    else
        GLLVModels._grouped_chol_stats_reset!()
        GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group,
            warm_start_inner = false)
        stats = GLLVModels._grouped_chol_stats()
        _check!(stats.calls == BASELINE_A_OBJ_CALLS,
                "fixture A: inner Laplace-fit call count changed ($(stats.calls) vs $(BASELINE_A_OBJ_CALLS)) — the reuse must not change the optimiser's path")
        _check!(stats.fallback == 0,
                "fixture A: $(stats.fallback) fresh-cholesky fallbacks (pattern mismatch), expected 0")
        _check!(stats.fresh == 2 * stats.calls,
                "fixture A: fresh=$(stats.fresh) != 2*calls=$(2 * stats.calls) — expected exactly 2 fresh symbolic analyses (Fisher + observed) per inner Laplace-fit call, 0 thereafter")
        _check!(stats.fresh < BASELINE_A_FRESH_CHOLESKY,
                "fixture A: fresh=$(stats.fresh) not below the pre-fix baseline $(BASELINE_A_FRESH_CHOLESKY)")

        GLLVModels._grouped_chol_stats_reset!()
        GLLVModels.joint_grouped_laplace_loglik(fixture_b()...; link = GLLVModels.LogLink())
        statsb = GLLVModels._grouped_chol_stats()
        _check!(statsb.calls == 1, "fixture B: call count mismatch ($(statsb.calls))")
        _check!(statsb.fallback == 0,
                "fixture B: $(statsb.fallback) fresh-cholesky fallbacks, expected 0")
        _check!(statsb.fresh == 2 * statsb.calls,
                "fixture B: fresh=$(statsb.fresh) != 2*calls=$(2 * statsb.calls)")
    end

    return all(_RESULTS), copy(_REASONS)
end

# ---------------------------------------------------------------------------
# S7c (leaf-S7c G7c.1): warm-started inner Laplace fits. `b_init` on
# `joint_grouped_laplace_loglik` and `warm_start_inner` on
# `fit_grouped_nongaussian` are additive keywords with cold-start defaults,
# so this is the FIRST test to exercise them (nothing pre-fix could pass or
# fail these checks — there is no baseline capture step here, unlike G7b.1).
# ---------------------------------------------------------------------------
function run_warm_identity_checks()
    empty!(_RESULTS); empty!(_REASONS)

    # ---- Part A: inner-kernel mechanism identity (fixture B, an EXISTING
    # fixture in this file — direct joint_grouped_laplace_loglik calls, no
    # outer optimiser, so this is where "same fixed point regardless of
    # starting b" is proven at its most fundamental level). ----
    args = fixture_b()
    r_cold = GLLVModels.joint_grouped_laplace_loglik(args...; link = GLLVModels.LogLink())
    _check!(r_cold.status === :ok, "inner: cold call did not converge ($(r_cold.status))")

    # Starting AT the converged mode should need at most one confirming step.
    r_warm_exact = GLLVModels.joint_grouped_laplace_loglik(args...; link = GLLVModels.LogLink(),
        b_init = r_cold.mode)
    _check!(r_warm_exact.status === :ok, "inner: warm-at-answer call did not converge")
    _check!(r_warm_exact.iterations <= 1,
        "inner: starting AT the converged mode should need <=1 Newton iteration, got $(r_warm_exact.iterations)")
    _check!(isapprox(r_warm_exact.loglik, r_cold.loglik; rtol = 1e-12),
        "inner: warm-at-answer loglik drifted from cold ($(r_warm_exact.loglik) vs $(r_cold.loglik))")
    _check!(isapprox(r_warm_exact.mode, r_cold.mode; rtol = 1e-10),
        "inner: warm-at-answer mode drifted from cold")

    # A genuinely different (but nearby) warm start converges to the SAME
    # mode/loglik, in no more Newton iterations than cold.
    Random.seed!(9001)
    b_near = r_cold.mode .+ 0.05 .* randn(length(r_cold.mode))
    r_warm_near = GLLVModels.joint_grouped_laplace_loglik(args...; link = GLLVModels.LogLink(),
        b_init = b_near)
    _check!(r_warm_near.status === :ok, "inner: warm-near call did not converge")
    _check!(isapprox(r_warm_near.loglik, r_cold.loglik; rtol = 1e-10),
        "inner: warm-near loglik drifted from cold ($(r_warm_near.loglik) vs $(r_cold.loglik))")
    _check!(isapprox(r_warm_near.mode, r_cold.mode; rtol = 1e-8),
        "inner: warm-near mode drifted from cold")
    _check!(r_warm_near.iterations <= r_cold.iterations,
        "inner: warm-near took MORE Newton iterations than cold ($(r_warm_near.iterations) vs $(r_cold.iterations))")

    # Defensive: a length-mismatched b_init fails cleanly (never silently
    # truncates/pads/misinterprets).
    r_bad = GLLVModels.joint_grouped_laplace_loglik(args...; link = GLLVModels.LogLink(),
        b_init = [1.0, 2.0, 3.0])   # fixture_b has m=1
    _check!(r_bad.status === :invalid_warm_start,
        "inner: mismatched-length b_init did not fail cleanly (status=$(r_bad.status))")

    # ---- Part B: outer fit identity (fixture A, the Latte 200x5 GLMM —
    # the ONLY fixture in this file driven by an outer optimiser, so the
    # only one where "outer iteration count and termination reason
    # identical" is a meaningful claim). `warm_start_inner=false` is
    # BYTE-IDENTICAL to the pre-S7c code path (b_init is always `nothing`,
    # the cache-update block never runs), so "cold == origin/main" is
    # already proven by G7b.1's own baseline assertions above; this gate
    # only needs to prove warm == cold. ----
    Y1, terms, group = fixture_a()

    # At the fitter's own DEFAULT inner_tol=1e-8, Nelder-Mead's outer path is
    # measurably sensitive to ANY perturbation of the inner solve's last few
    # bits (a PRE-EXISTING property of this fitter, not introduced by
    # warm-starting — the informational block below quantifies it).
    # Tightening inner_tol (already a pre-existing, user-facing keyword of
    # fit_grouped_nongaussian; not new in this arc) to 1e-10 removes that
    # noise floor and is where the exact-iteration-count identity is
    # asserted.
    identity_inner_tol = 1e-10
    fit_cold = GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group,
        warm_start_inner = false, inner_tol = identity_inner_tol)
    fit_warm = GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group,
        warm_start_inner = true, inner_tol = identity_inner_tol)
    _check!(fit_cold.converged == fit_warm.converged, "fixture A: converged mismatch (cold vs warm)")
    _check!(fit_cold.iterations == fit_warm.iterations,
        "fixture A: outer iteration count differs (cold=$(fit_cold.iterations), warm=$(fit_warm.iterations)) " *
        "at inner_tol=$identity_inner_tol")
    _check!(fit_cold.stopping_reason == fit_warm.stopping_reason,
        "fixture A: stopping reason differs (cold=$(fit_cold.stopping_reason), warm=$(fit_warm.stopping_reason))")
    _check!(isapprox(fit_warm.loglik, fit_cold.loglik; rtol = 1e-8),
        "fixture A: warm loglik not within rtol 1e-8 of cold ($(fit_warm.loglik) vs $(fit_cold.loglik))")
    _check!(isapprox(fit_warm.parameters, fit_cold.parameters; rtol = 1e-8),
        "fixture A: warm parameters not within rtol 1e-8 of cold")

    # Informational only (NOT gated): quantifies the default-inner_tol
    # sensitivity named above, surfaced for the orchestrator rather than
    # silently hidden. Both paths still converge to the same loglik to
    # ~1e-13 relative; only the outer iteration COUNT and the last few
    # significant digits of the parameters can differ at the loose default.
    fit_cold_def = GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group,
        warm_start_inner = false)
    fit_warm_def = GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group,
        warm_start_inner = true)
    @info "S7c G7c.1: default inner_tol=1e-8 outer-path sensitivity (informational, not gated)" cold_iters=fit_cold_def.iterations warm_iters=fit_warm_def.iterations same_iters=(fit_cold_def.iterations == fit_warm_def.iterations) rel_ll=(abs(fit_warm_def.loglik - fit_cold_def.loglik) / abs(fit_cold_def.loglik)) rel_par=(maximum(abs.(fit_warm_def.parameters .- fit_cold_def.parameters)) / maximum(abs.(fit_cold_def.parameters)))

    return all(_RESULTS), copy(_REASONS)
end

const _GATES = Dict(
    "identity" => ("G7b.1", "grouped Laplace CHOLMOD reuse identity (S7b)", run_identity_checks),
    "warm_identity" => ("G7c.1", "grouped Laplace warm-start identity (S7c)", run_warm_identity_checks),
)

function _parse_gate_arg(argv, default = "identity")
    parsed = default
    for (i, a) in enumerate(argv)
        if a == "--gate" && i < length(argv)
            parsed = argv[i + 1]
        end
    end
    return parsed
end

_GATE_OK = false
_GATE_REASONS = String[]

if abspath(PROGRAM_FILE) == @__FILE__
    gate_arg = _parse_gate_arg(ARGS)
    valid_gates = join(collect(keys(_GATES)), ", ")
    haskey(_GATES, gate_arg) || error("unknown --gate '$gate_arg'; expected one of $valid_gates")
    gate_label, testset_name, runner = _GATES[gate_arg]
    # Standalone script mode: `@testset` throws at its `end` if any `@test`
    # inside failed (expected pre-fix for `identity` — see the file header).
    # Catch that so the GATE line below still prints with the correct
    # PASS/FAIL + reasons and exit code; the runner itself already ran to
    # completion by the time the testset finishes; nothing here is silently
    # swallowed for the runtests.jl path (that branch, below, does not catch).
    try
        @testset "$testset_name" begin
            global _GATE_OK, _GATE_REASONS = runner()
        end
    catch e
        @info "testset reported failures; continuing to the GATE line" exception = e
    end
    if _GATE_OK
        println("GATE $gate_label PASS")
        exit(0)
    else
        println("GATE $gate_label FAIL ", join(_GATE_REASONS, "; "))
        exit(1)
    end
else
    # Included from test/runtests.jl: run BOTH gates as normal @testsets and
    # let genuine failures propagate.
    for (gate_label, testset_name, runner) in values(_GATES)
        @testset "$testset_name" begin
            global _GATE_OK, _GATE_REASONS = runner()
        end
    end
end
