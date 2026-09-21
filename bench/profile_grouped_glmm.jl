# bench/profile_grouped_glmm.jl — leaf-S4 gate G4.3.
#
# Profiles the grouped-Poisson route `fit_gllvm(Y1; family = Poisson(),
# grouping = [GroupingTerm(:unit; mode = :indep)], unit = group)`, which
# dispatches to `fit_grouped_nongaussian` (src/grouped_nongaussian_fit.jl:272)
# and its joint Laplace kernel `joint_grouped_laplace_loglik`
# (src/grouped_laplace.jl:165-292) — the route Latte.jl beat 12x on this exact
# 200x5 fixture (0.192 s ours vs 0.015 s Latte).
#
# WALL-CLOCK MEASUREMENT: calls the real, unmodified `GLLVModels.fit_gllvm`
# exactly as ~/local-scratch/latte-prerun-20260918/runs/f3_ours_glmm.jl does
# (one untimed warm-up, then >=3 timed reps via `@elapsed`, median reported).
#
# COUNT MEASUREMENT (outer gradient evaluations / inner Newton iterations /
# fresh CHOLMOD symbolic analyses): `fit_grouped_nongaussian` does not expose
# these counts on its returned `GroupedNonGaussianFit`, and `Optim`'s own
# f/g-call counters are internal to a private optimize call we cannot reach
# from outside. So this script builds a byte-for-byte SHADOW of
# `fit_grouped_nongaussian`'s procedure for THIS exact call pattern (Poisson,
# one :indep GroupingTerm, unit=group, no X, no N, default dispersion=:trait,
# default g_tol/iterations/inner_maxiter/inner_tol) — same internal, unexported
# functions (`GLLVModels._grouped_nongaussian_kind`, `_grouped_labels`,
# `_grouped_incidence`, `_trait_mean_design`, `_grouped_nongaussian_initial_
# parameters`, `_grouped_term_unpack`, `_grouped_nongaussian_family`,
# `_grouped_laplace_design`, `joint_grouped_laplace_loglik`,
# `_grouped_fd_gradient`), same Optim calls (NelderMead then BFGS-with-FD-
# gradient refinement), just with a counting objective wrapped around the
# REAL `joint_grouped_laplace_loglik` call. No src/ edit; no re-derivation of
# its ~90-line Newton/line-search body — we call it, we don't reimplement it.
#
# The shadow's own converged loglik is compared to the real `fit_gllvm` call's
# loglik as a faithfulness check (printed; large disagreement invalidates the
# counts, not just the timing).
#
# "Fresh CHOLMOD symbolic analyses" per `joint_grouped_laplace_loglik` call is
# DERIVED from its returned `.iterations` field, not independently instrumented:
# reading src/grouped_laplace.jl:208-291, each of the `.iterations` NON-
# terminal loop passes performs exactly 2 `cholesky(Symmetric(sparse))` calls
# (Ff at :230-236, Fn at :241-245 — a fresh symbolic+numeric CHOLMOD
# factorization each time, since neither reuses a prior symbolic factor), and
# the terminal (converged) pass performs exactly 1 more (Fo at :218-224). So
# for a `status === :ok` result: chol_count = 2*iterations + 1; otherwise
# (non-convergence / failure before the terminal branch): chol_count =
# 2*iterations (AGENT-INFERRED: exact for the dominant :nonconvergence and
# in-loop-failure branches; a `_joint_grouped_failure` returned from inside the
# state/logpost/curvature checks before any cholesky call in that pass would
# overcount by at most 2 for that single pass — negligible against the total).
#
# USAGE
#   env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. \
#       bench/profile_grouped_glmm.jl --gate

using GLLVModels
using DelimitedFiles, Statistics, LinearAlgebra, Printf, Profile, Random
using Distributions: Poisson
using Optim

const REPS = 5

# ---------------------------------------------------------------------------
# Header / provenance helpers (duplicated per leaf-S4's OWNS list).
# ---------------------------------------------------------------------------
function _git_sha()
    try
        strip(read(`git rev-parse --short HEAD`, String))
    catch
        "unknown"
    end
end

function _cpu_brand()
    try
        strip(read(`sysctl -n machdep.cpu.brand_string`, String))
    catch
        string(Sys.MACHINE)
    end
end

function _peak_rss_bytes()
    try
        buf = zeros(UInt8, 160)
        rc = GC.@preserve buf ccall(:getrusage, Cint, (Cint, Ptr{Cvoid}), 0, pointer(buf))
        rc == 0 || return missing
        GC.@preserve buf unsafe_load(Ptr{Int64}(pointer(buf) + 32))
    catch
        missing
    end
end

function header_lines()
    blas = try
        replace(sprint(show, LinearAlgebra.BLAS.get_config()), '\n' => ' ', '\r' => ' ')
    catch
        "unknown"
    end
    return [
        "# git_sha=$(_git_sha())",
        "# julia_version=$(VERSION)",
        "# blas_config=$(blas)",
        "# threads=$(Threads.nthreads())",
        "# os=$(Sys.KERNEL)-$(Sys.MACHINE)",
        "# cpu=$(_cpu_brand())",
        "# peak_rss_bytes=$(_peak_rss_bytes())",
    ]
end

# ---------------------------------------------------------------------------
# CLI (this gate takes no --p; the fixture is fixed at 200x5).
# ---------------------------------------------------------------------------
function parse_args(argv)
    gate = "run"
    i = 1
    while i <= length(argv)
        if argv[i] == "--gate"
            gate = (i + 1 <= length(argv) && !startswith(argv[i + 1], "--")) ? argv[i + 1] : "run"
            i += (gate == "run" ? 1 : 2)
        else
            i += 1
        end
    end
    return gate
end

# ---------------------------------------------------------------------------
# Fixture — bench/fixtures/glmm_200x5.csv (copied verbatim from the READ-ONLY
# ~/local-scratch/latte-prerun-20260918/data/glmm_200x5.csv: no header, col 1
# = Poisson count, col 2 = group id 1..200).
# ---------------------------------------------------------------------------
function load_fixture()
    path = joinpath(@__DIR__, "fixtures", "glmm_200x5.csv")
    isfile(path) || error("fixture missing: $path (expected a copy of " *
                           "~/local-scratch/latte-prerun-20260918/data/glmm_200x5.csv)")
    M = readdlm(path, ',', Int)
    y = M[:, 1]
    group = M[:, 2]
    G = maximum(group)
    N = length(y)
    return y, group, G, N
end

median_s(f; reps::Int = REPS) = begin
    f()
    ts = [(@elapsed f()) for _ in 1:reps]
    median(ts)
end

# ---------------------------------------------------------------------------
# Shadow-counted replica of `fit_grouped_nongaussian`'s procedure, specialised
# to (family = Poisson(), terms = [GroupingTerm(:unit; mode=:indep)], no X, no
# N, dispersion = :trait [normalises to :none for Poisson], default
# g_tol/iterations/inner_maxiter/inner_tol). Every function called below is
# the real, unexported GLLVModels internal — see src/grouped_nongaussian_fit.jl
# for the original (lines 272-376) this mirrors.
# ---------------------------------------------------------------------------
function shadow_fit_counts(Y1::Matrix{Float64}, group::Vector{Int}; warm::Bool = false)
    p, n = size(Y1)
    termvec = GLLVModels.GroupingTerm[GLLVModels.GroupingTerm(:unit; mode = :indep)]
    kind = GLLVModels._grouped_nongaussian_kind(Poisson())          # :poisson
    mode = GLLVModels._grouped_nongaussian_dispersion_mode(kind, :trait)  # :none for Poisson

    labels = GLLVModels._grouped_labels(n, termvec; unit = group, unit_obs = nothing,
                                         cluster = nothing, cluster2 = nothing)
    incidences = [GLLVModels._grouped_incidence(values, n) for values in labels]

    data = Matrix{Float64}(Y1)
    trials = ones(Float64, size(data))   # Poisson: _grouped_nongaussian_trials returns ones
    D = GLLVModels._trait_mean_design(p, n)
    q = size(D, 2)
    source_coordinates = sum(term -> GLLVModels._grouping_term_nparams(term, p), termvec; init = 0)
    dispersion_indices = GLLVModels._grouped_nongaussian_dispersion_indices(q, source_coordinates,
        kind, mode, p)
    expected_len = q + source_coordinates + length(dispersion_indices)

    theta0 = GLLVModels._grouped_nongaussian_initial_parameters(data, trials, D, termvec,
        kind, Poisson(), mode)

    obj_calls = Ref(0)
    inner_iters_sum = Ref(0)
    chol_count = Ref(0)
    ok_calls = Ref(0)
    failed_calls = Ref(0)
    final_inner_status = Ref(:never_ran)
    # S7c fix (mirrors src/grouped_nongaussian_fit.jl exactly, see that
    # file's `objective_cold`/`objective_warm` split and its comment for the
    # mechanism): warm-starting is confined to the Nelder-Mead VALUE-ONLY
    # search; every FD-differenced quantity (candidate_gradient, the BFGS
    # gradient!, the final value/gradient/Hessian) always uses `use_warm =
    # false`. `b_cache` is shared across BOTH so `warm calls` in the
    # printed distribution below means "warm-started during Nelder-Mead",
    # not "warm-started anywhere".
    b_cache = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    cold_iters = Int[]
    warm_iters = Int[]

    function counting_objective(value, use_warm::Bool)
        obj_calls[] += 1
        (length(value) == expected_len && all(isfinite, value)) || return GLLVModels._NLL_SENTINEL
        try
            gamma = collect(view(value, 1:q))
            loads, uniques, _, used = GLLVModels._grouped_term_unpack(
                view(value, (q + 1):(q + source_coordinates)), p, termvec)
            used == source_coordinates || return GLLVModels._NLL_SENTINEL
            family = GLLVModels._grouped_nongaussian_family(kind, value, dispersion_indices, p, n, mode)
            family === nothing && return GLLVModels._NLL_SENTINEL
            W = GLLVModels._grouped_laplace_design(incidences, loads; uniques = uniques)
            b_init = (warm && use_warm) ? b_cache[] : nothing
            was_cold = b_init === nothing
            result = GLLVModels.joint_grouped_laplace_loglik(family, vec(data), vec(trials), D,
                gamma, W; link = GLLVModels._grouped_nongaussian_link(Val(kind)),
                maxiter = 100, tol = 1e-8, b_init = b_init)
            inner_iters_sum[] += result.iterations
            if result.status === :ok
                chol_count[] += 2 * result.iterations + 1
                ok_calls[] += 1
                was_cold ? push!(cold_iters, result.iterations) : push!(warm_iters, result.iterations)
                (warm && use_warm) && (b_cache[] = copy(result.mode))
            else
                chol_count[] += 2 * result.iterations
                failed_calls[] += 1
            end
            final_inner_status[] = result.status
            return (result.status === :ok && result.converged && isfinite(result.loglik)) ?
                -result.loglik : GLLVModels._NLL_SENTINEL
        catch
            failed_calls[] += 1
            return GLLVModels._NLL_SENTINEL
        end
    end
    warm_objective(value) = counting_objective(value, true)
    cold_objective(value) = counting_objective(value, false)

    grad_calls = Ref(0)
    fd_gradient = (obj, val) -> begin
        grad_calls[] += 1
        GLLVModels._grouped_fd_gradient(obj, val)
    end

    result1 = Optim.optimize(warm_objective, theta0, Optim.NelderMead(),
        Optim.Options(g_tol = 1e-4, iterations = 100))
    candidate = collect(Optim.minimizer(result1))
    candidate_gradient = fd_gradient(cold_objective, candidate)
    final_result = result1
    if all(isfinite, candidate_gradient)
        gradient! = (storage, value) -> (storage .= fd_gradient(cold_objective, value))
        refined = try
            Optim.optimize(cold_objective, gradient!, candidate, Optim.BFGS(),
                Optim.Options(g_tol = 1e-4, iterations = 100))
        catch
            nothing
        end
        if refined !== nothing
            refined_estimate = collect(Optim.minimizer(refined))
            refined_value = cold_objective(refined_estimate)
            if isfinite(refined_value) && !(refined_value >= 1e12) &&
                    refined_value <= cold_objective(candidate)
                final_result = refined
            end
        end
    end
    estimate = collect(Optim.minimizer(final_result))
    shadow_loglik = -cold_objective(estimate)

    return (; shadow_loglik, obj_calls = obj_calls[], grad_calls = grad_calls[],
             inner_iters_sum = inner_iters_sum[], chol_count = chol_count[],
             ok_calls = ok_calls[], failed_calls = failed_calls[],
             final_inner_status = final_inner_status[],
             cold_iters = cold_iters, warm_iters = warm_iters)
end

# ---------------------------------------------------------------------------
# --gate after (leaf-S7b G7b.2): before/after the CHOLMOD symbolic-reuse
# change in src/grouped_laplace.jl (S7b). "Before" numbers are the pinned,
# independently-verified pre-fix baseline from
# test/test_grouped_laplace_identity.jl (118 inner Laplace-fit calls, 1540
# fresh CHOLMOD symbolic analyses, 0.192s banked warm wall — see that file's
# header for how 118/1540 were measured: the real fit_gllvm call path with
# the reuse branch temporarily forced off, NOT the shadow-driver's 103/1345
# estimate this same script's `--gate` mode still reports for obj_calls /
# outer_gradient_evals, which the S7b fix does not change). "After" numbers
# for inner-call/CHOLMOD counts come from the real GLLVModels._grouped_chol_
# stats() counter (exact, no shadow needed); outer_gradient_evals still needs
# the shadow driver (the reuse fix does not touch the outer FD gradient).
# ---------------------------------------------------------------------------
function main_after()
    println("Julia ", VERSION, "  threads=", Threads.nthreads())
    sha = _git_sha()

    y, group, G, N = load_fixture()
    Y1 = reshape(Float64.(y), 1, :)
    terms = [GLLVModels.GroupingTerm(:unit; mode = :indep)]

    before_wall_s = 0.192
    before_calls = 118
    before_fresh = 1540
    before_outer_grad_evals = 8

    warm_s_after = median_s(() -> GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group, warm_start_inner = false))
    after_fit = GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group, warm_start_inner = false)

    has_stats = isdefined(GLLVModels, :_grouped_chol_stats_reset!) && isdefined(GLLVModels, :_grouped_chol_stats)
    stats = if has_stats
        GLLVModels._grouped_chol_stats_reset!()
        GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group, warm_start_inner = false)
        GLLVModels._grouped_chol_stats()
    else
        (calls = -1, fresh = -1, reused = -1, fallback = -1)
    end

    counts = shadow_fit_counts(Matrix{Float64}(Y1), group)  # for outer_gradient_evals only, post-fix

    println("--- BEFORE (pinned, test/test_grouped_laplace_identity.jl baseline) ---")
    @printf("warm_wall=%.3fs  inner_calls=%d  fresh_cholmod=%d  outer_gradient_evals=%d\n",
            before_wall_s, before_calls, before_fresh, before_outer_grad_evals)
    println("--- AFTER (measured now, this commit) ---")
    @printf("warm_wall=%.4fs  inner_calls=%d  fresh_cholmod=%d  reused_cholmod=%d  fallback=%d  outer_gradient_evals=%d\n",
            warm_s_after, stats.calls, stats.fresh, stats.reused, stats.fallback, counts.grad_calls)
    println("--- reference ---")
    @printf("ours banked=0.192s  Latte.jl=0.015s  after/Latte=%.1fx\n", warm_s_after / 0.015)

    ok = has_stats && after_fit.converged && stats.calls > 0 && warm_s_after > 0
    reasons = String[]
    ok || push!(reasons, "measurement incomplete (has_stats=$has_stats converged=$(after_fit.converged) calls=$(stats.calls))")

    mkpath(joinpath(@__DIR__, "results"))
    out = joinpath(@__DIR__, "results", "grouped_after_$(sha).tsv")
    open(out, "w") do io
        for l in header_lines()
            println(io, l)
        end
        println(io, "N\tG\treps\tbefore_warm_wall_s\tbefore_inner_calls\tbefore_fresh_cholmod\t",
                     "before_outer_gradient_evals\tafter_warm_wall_s\tafter_inner_calls\t",
                     "after_fresh_cholmod\tafter_reused_cholmod\tafter_fallback\tafter_outer_gradient_evals\t",
                     "latte_wall_s\tafter_over_latte")
        println(io, N, "\t", G, "\t", REPS, "\t", before_wall_s, "\t", before_calls, "\t", before_fresh, "\t",
                     before_outer_grad_evals, "\t", warm_s_after, "\t", stats.calls, "\t", stats.fresh, "\t",
                     stats.reused, "\t", stats.fallback, "\t", counts.grad_calls, "\t", 0.015, "\t",
                     warm_s_after / 0.015)
    end
    println("TSV written: ", out)

    if ok
        println("GATE G7b.2 PASS")
    else
        println("GATE G7b.2 FAIL ", join(reasons, "; "))
    end
    exit(ok ? 0 : 1)
end

# ---------------------------------------------------------------------------
# --gate warm (leaf-S7c G7c.2): inner Newton iterations summed over one fit,
# cold (S7b baseline behaviour, `warm=false`) vs warm-started (`warm=true`,
# the S7c change), via the shadow driver's now warm-start-aware
# `shadow_fit_counts` (the real `fit_grouped_nongaussian` still does not
# expose per-call counts, so the byte-for-byte shadow is still how this is
# measured — same rationale as the file header's `--gate`/`--gate after`).
# The count is reported whatever it is; no pass/fail threshold on the value.
# ---------------------------------------------------------------------------
function main_warm()
    println("Julia ", VERSION, "  threads=", Threads.nthreads())
    y, group, G, N = load_fixture()
    Y1 = Matrix{Float64}(reshape(Float64.(y), 1, :))

    cold = shadow_fit_counts(Y1, group; warm = false)
    warm = shadow_fit_counts(Y1, group; warm = true)

    println("--- BEFORE (cold start every inner call, S7b baseline behaviour) ---")
    @printf("obj_calls=%d  inner_newton_iters_sum=%d  ok_calls=%d  failed_calls=%d  final_status=%s  loglik=%.6f\n",
            cold.obj_calls, cold.inner_iters_sum, cold.ok_calls, cold.failed_calls,
            cold.final_inner_status, cold.shadow_loglik)

    println("--- AFTER (warm-started from the previous call's converged mode, S7c) ---")
    @printf("obj_calls=%d  inner_newton_iters_sum=%d  ok_calls=%d  failed_calls=%d  final_status=%s  loglik=%.6f\n",
            warm.obj_calls, warm.inner_iters_sum, warm.ok_calls, warm.failed_calls,
            warm.final_inner_status, warm.shadow_loglik)

    # Per-call distribution (S7c fix: warm-starting is confined to the
    # Nelder-Mead value-only search): "cold calls" are every call OUTSIDE
    # that search (the very first call, plus every FD-differenced/BFGS
    # call, which always starts cold by design); "warm calls" are the
    # Nelder-Mead-phase calls that actually reused the cache.
    wc, ww = warm.cold_iters, warm.warm_iters
    println("--- per-call distribution (warm=true run) ---")
    @printf("cold calls: n=%d  sum=%d  values=%s\n", length(wc), sum(wc; init = 0), wc)
    if isempty(ww)
        println("warm calls: n=0 (no calls after the first — nothing to distribute)")
    else
        @printf("warm calls: n=%d  sum=%d  min=%d  median=%.1f  max=%.1f\n",
                length(ww), sum(ww), minimum(ww), median(ww), maximum(ww))
    end

    println("--- reference (S4/S7b corrected counter, banked) ---")
    println("before_inner_newton_iters_sum=711 (118 inner Laplace-fit calls)")

    loglik_gap_rel = abs(warm.shadow_loglik - cold.shadow_loglik) / max(abs(cold.shadow_loglik), 1.0)
    @printf("cold-vs-warm shadow loglik gap: rel=%.3e\n", loglik_gap_rel)

    ok = cold.obj_calls > 0 && warm.obj_calls > 0 && cold.ok_calls > 0 && warm.ok_calls > 0
    reasons = String[]
    ok || push!(reasons, "measurement incomplete (cold.obj_calls=$(cold.obj_calls) warm.obj_calls=$(warm.obj_calls) " *
                          "cold.ok_calls=$(cold.ok_calls) warm.ok_calls=$(warm.ok_calls))")
    if ok
        println("GATE G7c.2 PASS")
    else
        println("GATE G7c.2 FAIL ", join(reasons, "; "))
    end
    exit(ok ? 0 : 1)
end

# ---------------------------------------------------------------------------
# --gate after_warm (leaf-S7c G7c.3): warm median wall on the 200x5 fixture,
# before (warm_start_inner=false) vs after (warm_start_inner=true, now the
# default), against the banked 0.192s (ours, pre-warm-start) and 0.015s
# (Latte.jl); plus a sampling-`Profile` split naming the per-call GLM-state
# (`_joint_grouped_state`/`_joint_grouped_components`) share of one fit's
# wall, so the NEXT lever (named in leaf-S7b's checkpoint as the likely
# O(n=1000) per-Newton-iteration GLM state/score/curvature cost) is a number.
# ---------------------------------------------------------------------------
function main_after_warm()
    println("Julia ", VERSION, "  threads=", Threads.nthreads())
    sha = _git_sha()
    y, group, G, N = load_fixture()
    Y1 = reshape(Float64.(y), 1, :)
    terms = [GLLVModels.GroupingTerm(:unit; mode = :indep)]

    before_wall_s = median_s(() -> GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms,
        unit = group, warm_start_inner = false))
    after_wall_s = median_s(() -> GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms,
        unit = group, warm_start_inner = true))
    after_fit = GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group,
        warm_start_inner = true)

    println("--- BEFORE (warm_start_inner=false, 5-rep median wall) ---")
    @printf("warm_median_wall=%.4fs  (banked 0.192s)\n", before_wall_s)
    println("--- AFTER (warm_start_inner=true, now the default, 5-rep median wall) ---")
    @printf("warm_median_wall=%.4fs  converged=%s  iterations=%d\n",
            after_wall_s, after_fit.converged, after_fit.iterations)
    println("--- reference ---")
    @printf("ours before/Latte=%.1fx  ours after/Latte=%.1fx  Latte.jl=0.015s\n",
            before_wall_s / 0.015, after_wall_s / 0.015)

    # ---- Profile split: per-call GLM-state share of one warm fit's wall ----
    # `include_meta=false` strips the thread/task/cpu-cycle metadata blocks
    # Julia's raw profile buffer otherwise interleaves with instruction
    # pointers (without this, small metadata integers can collide with real
    # `ip` values and throw `KeyError` on `lookup[ip]` — found and fixed
    # while building this gate). A sample counts as "GLM-state" if
    # `_joint_grouped_state` or `_joint_grouped_components` appears ANYWHERE
    # in its stack (INCLUSIVE of whatever they call — score/curvature/link
    # arithmetic — not just their own self-time), since that pair is the
    # per-Newton-iteration cost this gate is naming a share for.
    Profile.clear()
    nreps_profile = 20
    Profile.@profile for _ in 1:nreps_profile
        GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group,
            warm_start_inner = true)
    end
    data = Profile.fetch(include_meta = false)
    lookup = Profile.getdict(data)
    glm_state_samples = 0
    total_samples = 0
    in_target = false
    for ip in data
        if ip == 0   # end-of-stack sentinel: tally the sample just finished
            total_samples += 1
            in_target && (glm_state_samples += 1)
            in_target = false
            continue
        end
        frames = lookup[ip]
        frames_vec = frames isa AbstractVector ? frames : [frames]
        if any(sf -> occursin("_joint_grouped_state", string(sf.func)) ||
                     occursin("_joint_grouped_components", string(sf.func)), frames_vec)
            in_target = true
        end
    end
    glm_state_share = total_samples > 0 ? glm_state_samples / total_samples : NaN
    @printf("Profile split (%d fits, %d total samples): GLM-state (_joint_grouped_state + _joint_grouped_components) share = %.1f%% (%d/%d samples)\n",
            nreps_profile, total_samples, 100 * glm_state_share, glm_state_samples, total_samples)

    mkpath(joinpath(@__DIR__, "results"))
    out = joinpath(@__DIR__, "results", "grouped_warm_$(sha).tsv")
    open(out, "w") do io
        for l in header_lines()
            println(io, l)
        end
        println(io, "N\tG\treps\tbefore_warm_wall_s\tafter_warm_wall_s\tbanked_wall_s\tlatte_wall_s\t",
                     "before_over_latte\tafter_over_latte\tglm_state_share\ttotal_profile_samples")
        println(io, N, "\t", G, "\t", REPS, "\t", before_wall_s, "\t", after_wall_s, "\t", 0.192, "\t", 0.015, "\t",
                     before_wall_s / 0.015, "\t", after_wall_s / 0.015, "\t", glm_state_share, "\t", total_samples)
    end
    println("TSV written: ", out)

    ok = after_fit.converged && before_wall_s > 0 && after_wall_s > 0 && total_samples > 0
    reasons = String[]
    ok || push!(reasons, "measurement incomplete (converged=$(after_fit.converged) before=$before_wall_s " *
                          "after=$after_wall_s total_samples=$total_samples)")
    if ok
        println("GATE G7c.3 PASS")
    else
        println("GATE G7c.3 FAIL ", join(reasons, "; "))
    end
    exit(ok ? 0 : 1)
end

# ---------------------------------------------------------------------------
# --gate sections (leaf-S8 GA.1/GA.2). Second, larger fixture: 3 traits, one
# :indep grouping term, Poisson, n=5000 observations, G=500 groups -> nθ = 3
# trait intercepts + 3 per-trait log-sd = 6 (meets the gate's nθ>=6 floor).
# Generated deterministically in code (Random.Xoshiro(seed)); no large CSV
# committed. Section timing is measured by wrapping the REAL, unmodified
# `_grouped_nongaussian_objective` closure (src/grouped_nongaussian_fit.jl:
# 232-268) and the REAL `_grouped_fd_gradient`/`_grouped_fd_hessian`
# (src/grouped_fit.jl:213-247) with bench-side counters/timers, then driving
# `Optim.optimize` with the EXACT same call sequence and options as
# `fit_grouped_nongaussian` (src/grouped_nongaussian_fit.jl:388-420, copied
# verbatim below). This is NOT the S4/S7b shadow pattern: that shadow
# hand-reimplemented the OBJECTIVE BODY itself (labels/incidence/family/W
# reconstruction inlined into a `counting_objective` closure), and that
# reimplementation silently drifted from the real one (103/1345 vs the real
# 118/1540, see this file's header). Here the objective body is never
# rewritten -- we call the real closure factory and only wrap its RETURN
# VALUE with a stopwatch, so every objective evaluation the outer optimizer
# sees is bit-identical to what `fit_grouped_nongaussian` itself would
# produce. A loglik-agreement check against a real `fit_gllvm` call on the
# same data is still printed and gates GA.1, as a faithfulness backstop.
#
# Inner Newton iteration COUNT is not re-derived by us: it is DERIVED from
# the existing, already-validated `_grouped_chol_stats()` counter (src/
# grouped_laplace.jl:59-100, added in S7b) via the documented relationship
# at this file's header (lines ~35-47): a `status===:ok` inner call costs
# 2*iterations+1 cholesky calls, any other status costs 2*iterations. We do
# not add new instrumentation to src/ for this.
#
# GLM-state/CHOLMOD/log-det SECONDS (a split of the inner-Newton-solve time
# already counted inside the four objective-call buckets above) come from a
# sampling `Profile` pass over real, unmodified `fit_gllvm` calls -- the same
# technique `main_after_warm` already uses for its GLM-state share, extended
# to three mutually-exclusive categories. This is a measured proportion of
# real execution (stack sampling), not a reimplementation and not a source
# edit; it is reported as a SUBDIVISION of already-counted time, not summed
# again into the top-level wall-partition check.
# ---------------------------------------------------------------------------
const _S8_LARGE_SEED = 20260920
const _S8_LARGE_P = 3
const _S8_LARGE_N = 5000
const _S8_LARGE_G = 500

function _S8_make_large_fixture()
    rng = Random.Xoshiro(_S8_LARGE_SEED)
    beta_true = [log(4.0), log(6.0), log(2.5)]
    sd_true = [0.4, 0.3, 0.5]
    group = rand(rng, 1:_S8_LARGE_G, _S8_LARGE_N)
    b = [sd_true[t] .* randn(rng, _S8_LARGE_G) for t in 1:_S8_LARGE_P]
    Y = zeros(Float64, _S8_LARGE_P, _S8_LARGE_N)
    for i in 1:_S8_LARGE_N, t in 1:_S8_LARGE_P
        Y[t, i] = rand(rng, Poisson(exp(beta_true[t] + b[t][group[i]])))
    end
    return Y, group, _S8_LARGE_G, _S8_LARGE_N, _S8_LARGE_P
end

struct _S8Fixture
    name::String
    Y::Matrix{Float64}
    group::Vector{Int}
    G::Int
    N::Int
    p::Int
    terms::Vector{GLLVModels.GroupingTerm}
end

function _S8_fixtures()
    y200, group200, G200, N200 = load_fixture()
    small = _S8Fixture("glmm_200x5", reshape(Float64.(y200), 1, :), group200, G200, N200, 1,
        [GLLVModels.GroupingTerm(:unit; mode = :indep)])
    Ylarge, grouplarge, Glarge, Nlarge, plarge = _S8_make_large_fixture()
    large = _S8Fixture("glmm_5000x3_g500", Ylarge, grouplarge, Glarge, Nlarge, plarge,
        [GLLVModels.GroupingTerm(:unit; mode = :indep)])
    return small, large
end

# `analytic_gradient` DEFAULTS TO FALSE deliberately, and that is not the
# package's default (src/grouped_nongaussian_fit.jl:579 defaults it true).
# `--gate sections` is GA.1's ticked CHECK and its banked EVIDENCE was measured
# on the all-FD path; flipping this default would silently make that gate
# measure something else and stop reproducing its own numbers. `--gate
# sections_after` passes both settings explicitly instead.
function _S8_measure_driver(fx::_S8Fixture; g_tol = 1e-4, iterations = 100,
        inner_maxiter = 100, inner_tol = 1e-8, warm_start_inner = true,
        analytic_gradient::Bool = false)
    p, n = fx.p, fx.N
    family = Poisson()
    kind = GLLVModels._grouped_nongaussian_kind(family)
    pubmode = GLLVModels._grouped_nongaussian_dispersion_mode(kind, :trait)

    labels = GLLVModels._grouped_labels(n, fx.terms; unit = fx.group, unit_obs = nothing,
                                         cluster = nothing, cluster2 = nothing)
    incidences = [GLLVModels._grouped_incidence(v, n) for v in labels]
    data = Matrix{Float64}(fx.Y)
    trials = GLLVModels._grouped_nongaussian_trials(data, nothing, kind)
    D = GLLVModels._trait_mean_design(p, n)
    theta0 = GLLVModels._grouped_nongaussian_initial_parameters(data, trials, D, fx.terms,
        kind, family, pubmode)

    # The REAL objective closures -- byte-identical to fit_grouped_nongaussian's
    # own objective_cold/objective_warm (src/grouped_nongaussian_fit.jl:371-378).
    objective_cold = GLLVModels._grouped_nongaussian_objective(data, trials, D, fx.terms,
        incidences, kind; dispersion_mode = pubmode, inner_maxiter = Int(inner_maxiter),
        inner_tol = Float64(inner_tol), warm_start_inner = false)
    objective_warm = warm_start_inner ?
        GLLVModels._grouped_nongaussian_objective(data, trials, D, fx.terms, incidences, kind;
            dispersion_mode = pubmode, inner_maxiter = Int(inner_maxiter),
            inner_tol = Float64(inner_tol), warm_start_inner = true) :
        objective_cold

    # `phase[]` tags which OUTER call site is currently invoking `cold_objective`:
    # :bfgs_obj covers BFGS's own line-search value evaluations AND the handful
    # of direct finalization checks fit_grouped_nongaussian also makes outside
    # any FD stencil (refined_value, the two `cold_objective(candidate)`-style
    # comparisons, and the final `value = objective(estimate)`) -- all of these
    # are single calls, negligible in count next to the line search itself.
    phase = Ref(:bfgs_obj)
    nm_calls = Ref(0); nm_seconds = Ref(0.0)
    bfgs_calls = Ref(0); bfgs_seconds = Ref(0.0)
    fdgrad_obj_calls = Ref(0); fdgrad_obj_seconds = Ref(0.0)
    fdhess_obj_calls = Ref(0); fdhess_obj_seconds = Ref(0.0)
    fd_gradient_invocations = Ref(0); fd_gradient_seconds = Ref(0.0)
    fd_hessian_invocations = Ref(0); fd_hessian_seconds = Ref(0.0)
    an_gradient_invocations = Ref(0); an_gradient_seconds = Ref(0.0)
    an_gradient_fallbacks = Ref(0)

    nm_objective = value -> begin
        t0 = time_ns()
        v = objective_warm(value)
        nm_seconds[] += (time_ns() - t0) / 1e9
        nm_calls[] += 1
        v
    end
    cold_objective = value -> begin
        t0 = time_ns()
        v = objective_cold(value)
        dt = (time_ns() - t0) / 1e9
        if phase[] === :bfgs_obj
            bfgs_calls[] += 1; bfgs_seconds[] += dt
        elseif phase[] === :fd_gradient
            fdgrad_obj_calls[] += 1; fdgrad_obj_seconds[] += dt
        else
            fdhess_obj_calls[] += 1; fdhess_obj_seconds[] += dt
        end
        v
    end
    counted_fd_gradient = (obj, x) -> begin
        fd_gradient_invocations[] += 1
        prev = phase[]; phase[] = :fd_gradient
        t0 = time_ns()
        g = GLLVModels._grouped_fd_gradient(obj, x)
        fd_gradient_seconds[] += (time_ns() - t0) / 1e9
        phase[] = prev
        g
    end
    counted_fd_hessian = (obj, x) -> begin
        fd_hessian_invocations[] += 1
        prev = phase[]; phase[] = :fd_hessian
        t0 = time_ns()
        H = GLLVModels._grouped_fd_hessian(obj, x)
        fd_hessian_seconds[] += (time_ns() - t0) / 1e9
        phase[] = prev
        H
    end

    # Mirrors the real `grad_fn` closure at src/grouped_nongaussian_fit.jl:
    # 672-676 exactly: analytic first, and on `nothing` or any non-finite
    # coordinate the SAME `_grouped_fd_gradient(objective_cold, ...)` fallback
    # the pre-S8 code always used. The analytic stopwatch is stopped BEFORE any
    # fallback call, so the analytic and FD buckets stay disjoint and the
    # section sum is a partition rather than a double count.
    # Two SEPARATE closures selected up front, rather than one closure that
    # branches on `analytic_gradient` inside its body. That is a measurement
    # requirement, not a style choice: a single branching closure is inferred
    # as a whole on its first call, which drags `_grouped_analytic_gradient`
    # through compilation even on the FD path. That compilation lands inside
    # `driver_wall` but in none of the section buckets, and on the 0.17 s small
    # fixture it pushed GA.1's sections-sum gap from 8.0% to 12.7%, past its
    # 10% bound. Caught by re-running GA.1 after patching this function.
    counted_grad_fn = analytic_gradient ? (x -> begin
        an_gradient_invocations[] += 1
        t0 = time_ns()
        g = GLLVModels._grouped_analytic_gradient(x, data, trials, D, fx.terms,
            incidences, kind; dispersion_mode = pubmode,
            inner_maxiter = Int(inner_maxiter), inner_tol = Float64(inner_tol))
        an_gradient_seconds[] += (time_ns() - t0) / 1e9
        if g === nothing || !all(isfinite, g)
            an_gradient_fallbacks[] += 1
            return counted_fd_gradient(cold_objective, x)
        end
        g
    end) : (x -> counted_fd_gradient(cold_objective, x))

    # Inner-Laplace-fit calls and summed inner Newton iterations come from the
    # existing `_grouped_chol_stats` counter (src/grouped_laplace.jl:76-85) via
    # this file's documented 2n(+1) relationship, scoped to the driver run
    # rather than to a separate `fit_gllvm` call -- which matters here because
    # `analytic_gradient` is not threaded through the public `fit_gllvm`.
    has_chol_stats = isdefined(GLLVModels, :_grouped_chol_stats_reset!) &&
                     isdefined(GLLVModels, :_grouped_chol_stats)
    has_chol_stats && GLLVModels._grouped_chol_stats_reset!()

    wall_t0 = time_ns()
    # ---- verbatim sequence of src/grouped_nongaussian_fit.jl:388-420 ------
    result = Optim.optimize(nm_objective, theta0, Optim.NelderMead(),
        Optim.Options(g_tol = Float64(g_tol), iterations = Int(iterations)))
    candidate = collect(Optim.minimizer(result))
    candidate_gradient = counted_grad_fn(candidate)
    if all(isfinite, candidate_gradient)
        gradient! = (storage, value) -> (storage .= counted_grad_fn(value))
        refined = try
            Optim.optimize(cold_objective, gradient!, candidate, Optim.BFGS(),
                Optim.Options(g_tol = Float64(g_tol), iterations = Int(iterations)))
        catch
            nothing
        end
        if refined !== nothing
            refined_estimate = collect(Optim.minimizer(refined))
            refined_value = cold_objective(refined_estimate)
            if isfinite(refined_value) && !GLLVModels._nll_failed(refined_value) &&
                    refined_value <= cold_objective(candidate)
                result = refined
            end
        end
    end
    estimate = collect(Optim.minimizer(result))
    value = cold_objective(estimate)
    valid = isfinite(value) && !GLLVModels._nll_failed(value)
    gradient = valid ? counted_fd_gradient(cold_objective, estimate) : fill(Inf, length(estimate))
    gradient_norm = all(isfinite, gradient) ? maximum(abs, gradient) : Inf
    H = valid ? counted_fd_hessian(cold_objective, estimate) : fill(NaN, length(estimate), length(estimate))
    driver_wall = (time_ns() - wall_t0) / 1e9
    # -------------------------------------------------------------------
    cstats = has_chol_stats ? GLLVModels._grouped_chol_stats() :
        (calls = -1, fresh = -1, reused = -1, fallback = -1)
    inner_calls = cstats.calls
    chol_total = cstats.fresh + cstats.reused + cstats.fallback
    inner_iters_sum = inner_calls > 0 ? (chol_total - inner_calls) / 2 : NaN

    converged = Optim.converged(result) && valid && gradient_norm <= g_tol
    driver_loglik = -value

    total_obj_calls = nm_calls[] + bfgs_calls[] + fdgrad_obj_calls[] + fdhess_obj_calls[]
    total_obj_seconds = nm_seconds[] + bfgs_seconds[] + fdgrad_obj_seconds[] + fdhess_obj_seconds[]
    remainder = driver_wall - total_obj_seconds

    return (; nθ = length(theta0), driver_wall, driver_loglik, converged, estimate,
             nm_calls = nm_calls[], nm_seconds = nm_seconds[],
             bfgs_calls = bfgs_calls[], bfgs_seconds = bfgs_seconds[],
             fd_gradient_invocations = fd_gradient_invocations[], fd_gradient_obj_calls = fdgrad_obj_calls[],
             fd_gradient_seconds = fdgrad_obj_seconds[],
             fd_hessian_invocations = fd_hessian_invocations[], fd_hessian_obj_calls = fdhess_obj_calls[],
             fd_hessian_seconds = fdhess_obj_seconds[],
             analytic_gradient, an_gradient_invocations = an_gradient_invocations[],
             an_gradient_seconds = an_gradient_seconds[],
             an_gradient_fallbacks = an_gradient_fallbacks[],
             inner_calls, chol_total, inner_iters_sum,
             total_obj_calls, total_obj_seconds, remainder)
end

function _S8_chol_derived_iterations(fx::_S8Fixture)
    has_stats = isdefined(GLLVModels, :_grouped_chol_stats_reset!) && isdefined(GLLVModels, :_grouped_chol_stats)
    has_stats || return (; calls = -1, chol_total = -1, iters_sum = NaN)
    GLLVModels._grouped_chol_stats_reset!()
    GLLVModels.fit_gllvm(fx.Y; family = Poisson(), grouping = fx.terms, unit = fx.group)
    stats = GLLVModels._grouped_chol_stats()
    chol_total = stats.fresh + stats.reused + stats.fallback
    iters_sum = (chol_total - stats.calls) / 2   # documented 2n(+1) relationship; assumes near-universal :ok
    return (; calls = stats.calls, chol_total, iters_sum)
end

function _S8_profile_split(fx::_S8Fixture; nreps::Int)
    Profile.clear()
    Profile.@profile for _ in 1:nreps
        GLLVModels.fit_gllvm(fx.Y; family = Poisson(), grouping = fx.terms, unit = fx.group)
    end
    data = Profile.fetch(include_meta = false)
    lookup = Profile.getdict(data)
    glm_samples = 0; chol_samples = 0; logdet_samples = 0; total_samples = 0
    in_glm = false; in_chol = false; in_logdet = false
    for ip in data
        if ip == 0
            total_samples += 1
            in_glm && (glm_samples += 1)
            in_chol && (chol_samples += 1)
            in_logdet && (logdet_samples += 1)
            in_glm = in_chol = in_logdet = false
            continue
        end
        frames = lookup[ip]
        frames_vec = frames isa AbstractVector ? frames : [frames]
        for sf in frames_vec
            fname = string(sf.func)
            if occursin("_joint_grouped_state", fname) || occursin("_joint_grouped_components", fname) ||
                    occursin("_joint_grouped_logpost", fname)
                in_glm = true
            elseif occursin("_grouped_cached_cholesky", fname) || fname == "cholesky" || fname == "cholesky!"
                in_chol = true
            elseif fname == "logdet"
                in_logdet = true
            end
        end
    end
    glm_share = total_samples > 0 ? glm_samples / total_samples : NaN
    chol_share = total_samples > 0 ? chol_samples / total_samples : NaN
    logdet_share = total_samples > 0 ? logdet_samples / total_samples : NaN
    return (; glm_share, chol_share, logdet_share, total_samples)
end

function main_sections()
    println("Julia ", VERSION, "  threads=", Threads.nthreads())
    sha = _git_sha()
    small, large = _S8_fixtures()

    rows = NamedTuple[]
    for (fx, reps, profreps) in ((small, 5, 20), (large, 3, 5))
        println("=== fixture: ", fx.name, "  N=", fx.N, " G=", fx.G, " p=", fx.p, " ===")
        real_wall = median_s(() -> GLLVModels.fit_gllvm(fx.Y; family = Poisson(),
            grouping = fx.terms, unit = fx.group); reps = reps)
        real_fit = GLLVModels.fit_gllvm(fx.Y; family = Poisson(), grouping = fx.terms, unit = fx.group)

        m = _S8_measure_driver(fx)
        loglik_gap_rel = abs(m.driver_loglik - real_fit.loglik) / max(abs(real_fit.loglik), 1.0)

        chol = _S8_chol_derived_iterations(fx)
        prof = _S8_profile_split(fx; nreps = profreps)
        glm_seconds = prof.glm_share * m.total_obj_seconds
        chol_seconds = prof.chol_share * m.total_obj_seconds
        logdet_seconds = prof.logdet_share * m.total_obj_seconds

        section_sum = m.nm_seconds + m.bfgs_seconds + m.fd_gradient_seconds + m.fd_hessian_seconds
        sum_gap_rel = abs(section_sum - m.driver_wall) / m.driver_wall
        fd_share = (m.fd_gradient_seconds + m.fd_hessian_seconds + m.nm_seconds) / m.driver_wall

        @printf("real fit: wall_median(reps=%d)=%.4fs converged=%s loglik=%.4f iterations=%d\n",
                reps, real_wall, real_fit.converged, real_fit.loglik, real_fit.iterations)
        @printf("driver:   wall=%.4fs converged=%s loglik=%.4f loglik_gap_rel=%.3e nθ=%d\n",
                m.driver_wall, m.converged, m.driver_loglik, loglik_gap_rel, m.nθ)
        @printf("  Nelder-Mead:       calls=%-5d seconds=%.4f\n", m.nm_calls, m.nm_seconds)
        @printf("  BFGS line search:  calls=%-5d seconds=%.4f\n", m.bfgs_calls, m.bfgs_seconds)
        @printf("  FD gradient:       invocations=%d  implied_obj_calls=%d  seconds=%.4f\n",
                m.fd_gradient_invocations, m.fd_gradient_obj_calls, m.fd_gradient_seconds)
        @printf("  FD Hessian:        invocations=%d  implied_obj_calls=%d  seconds=%.4f\n",
                m.fd_hessian_invocations, m.fd_hessian_obj_calls, m.fd_hessian_seconds)
        @printf("  remainder (driver_wall - sections): %.4fs\n", m.remainder)
        @printf("  section_sum=%.4fs vs driver_wall=%.4fs (gap_rel=%.3f)\n", section_sum, m.driver_wall, sum_gap_rel)
        @printf("  inner Newton iters (derived from chol stats): calls=%d chol_total=%d iters_sum=%.1f\n",
                chol.calls, chol.chol_total, chol.iters_sum)
        @printf("  inner-solve time split (profiled, %d fits, %d samples): GLM-state=%.1f%% CHOLMOD=%.1f%% logdet=%.1f%% -> seconds %.4f / %.4f / %.4f\n",
                profreps, prof.total_samples, 100 * prof.glm_share, 100 * prof.chol_share, 100 * prof.logdet_share,
                glm_seconds, chol_seconds, logdet_seconds)
        @printf("  FD-attributable share (NM+FDgrad+FDhess)/driver_wall = %.3f\n", fd_share)

        push!(rows, (; fixture = fx.name, N = fx.N, G = fx.G, p = fx.p, ntheta = m.nθ,
            real_wall_median_s = real_wall, driver_wall_s = m.driver_wall, loglik_gap_rel,
            nm_calls = m.nm_calls, nm_seconds = m.nm_seconds,
            bfgs_calls = m.bfgs_calls, bfgs_seconds = m.bfgs_seconds,
            fd_gradient_invocations = m.fd_gradient_invocations, fd_gradient_obj_calls = m.fd_gradient_obj_calls,
            fd_gradient_seconds = m.fd_gradient_seconds,
            fd_hessian_invocations = m.fd_hessian_invocations, fd_hessian_obj_calls = m.fd_hessian_obj_calls,
            fd_hessian_seconds = m.fd_hessian_seconds,
            remainder_s = m.remainder, section_sum_s = section_sum, sum_gap_rel = sum_gap_rel,
            chol_calls = chol.calls, chol_total = chol.chol_total, inner_iters_sum = chol.iters_sum,
            profile_reps = profreps, profile_samples = prof.total_samples,
            glm_state_share = prof.glm_share, cholmod_share = prof.chol_share, logdet_share = prof.logdet_share,
            glm_state_seconds = glm_seconds, cholmod_seconds = chol_seconds, logdet_seconds = logdet_seconds,
            fd_attributable_share = fd_share))
    end

    mkpath(joinpath(@__DIR__, "results"))
    out = joinpath(@__DIR__, "results", "grouped_sections_$(sha).tsv")
    open(out, "w") do io
        for l in header_lines()
            println(io, l)
        end
        names = fieldnames(typeof(rows[1]))
        println(io, "# columns: ", join(names, " "))
        println(io, join(names, "\t"))
        for r in rows
            println(io, join(getfield.(Ref(r), names), "\t"))
        end
    end
    println("TSV written: ", out)

    ok = all(r -> r.sum_gap_rel <= 0.10, rows) && all(r -> r.loglik_gap_rel <= 1e-4, rows)
    reasons = String[]
    for r in rows
        r.sum_gap_rel <= 0.10 || push!(reasons, "$(r.fixture): sections sum gap $(round(r.sum_gap_rel, digits=3)) exceeds 10%")
        r.loglik_gap_rel <= 1e-4 || push!(reasons, "$(r.fixture): driver loglik diverges from real fit by rel=$(round(r.loglik_gap_rel, sigdigits=3))")
    end
    if ok
        println("GATE GA.1 PASS")
    else
        println("GATE GA.1 FAIL ", join(reasons, "; "))
    end

    large_row = rows[end]
    small_row = rows[1]
    @printf("GA.2: FD-attributable share -- large fixture (%s): %.3f  small fixture (%s): %.3f\n",
            large_row.fixture, large_row.fd_attributable_share, small_row.fixture, small_row.fd_attributable_share)
    println(large_row.fd_attributable_share >= 0.25 ? "GA.2 VERDICT: PROCEED" : "GA.2 VERDICT: STOP")

    exit(ok ? 0 : 1)
end

# ---------------------------------------------------------------------------
# --gate sections_after (leaf-S8 GB.5). The AFTER half of GA.1's partition:
# the same driver, the same fixtures, the same counters, run TWICE inside ONE
# process -- once with `analytic_gradient = false` (the pre-S8 all-FD path
# GA.1 measured) and once with it `true` (what the branch now does by
# default). Both settings in one run is deliberate: the machine state that
# contaminates an absolute number is shared by both halves, so the RATIO
# survives it even when the seconds do not.
#
# What this gate reports, per the ledger: objective calls and summed inner
# Newton iterations before and after (118 and 711 banked at fixture A);
# fixture A's wall against the S7c-banked 0.150383 s and Latte's 0.015 s; the
# larger fixture against its own GA.1 baseline (10.8879 s). Numbers are
# reported whatever they are -- the PASS condition is the integrity of the
# measurement (the driver still lands on the real `fit_gllvm` loglik, the
# sections still sum to the wall, before and after still agree at rtol 1e-8),
# never the direction or the size of the speedup.
#
# Read every speedup here as a FLOOR. Two of the three things GA.2 measured
# are still paid in full on the after path: the S7c warm start is still
# confined to Nelder-Mead (GB.4 not done) and the final O(ntheta^2) FD Hessian
# is still computed for the diagnostics (GA.2 put that alone at 16.0% of the
# large-fixture wall). This gate measures what the gradient change bought on
# its own.
# ---------------------------------------------------------------------------
const _S8_BANKED_A_WALL_S7C = 0.150383     # bench/results/grouped_warm_68c2f067c.tsv, post-S7c
const _S8_BANKED_A_WALL_PRE_S7C = 0.184721 # same TSV, pre-S7c
const _S8_BANKED_A_OBJ_CALLS = 118         # leaf-S8 BASELINE line
const _S8_BANKED_A_INNER_ITERS = 711       # leaf-S8 BASELINE line
const _S8_LATTE_A_WALL = 0.015             # Latte.jl on this same fixture
const _S8_GA1_LARGE_DRIVER_WALL = 10.8879  # GA.1 EVIDENCE, commit f59757a4f

# Median driver wall over `reps` timed runs after one untimed warm-up, with the
# counts taken from the first timed run and every later run CHECKED against it
# rather than assumed identical.
function _S8_repeat_driver(fx::_S8Fixture; analytic_gradient::Bool, reps::Int)
    _S8_measure_driver(fx; analytic_gradient = analytic_gradient)  # untimed warm-up
    ms = [_S8_measure_driver(fx; analytic_gradient = analytic_gradient) for _ in 1:reps]
    walls = [m.driver_wall for m in ms]
    counts_stable = all(m -> m.total_obj_calls == ms[1].total_obj_calls &&
                             m.inner_calls == ms[1].inner_calls, ms)
    return (; m = ms[1], wall_median = median(walls), wall_min = minimum(walls),
              wall_max = maximum(walls), reps, counts_stable)
end

function main_sections_after()
    println("Julia ", VERSION, "  threads=", Threads.nthreads())
    sha = _git_sha()
    small, large = _S8_fixtures()

    rows = NamedTuple[]
    for (fx, reps) in ((small, 5), (large, 3))
        println("=== fixture: ", fx.name, "  N=", fx.N, " G=", fx.G, " p=", fx.p, " ===")
        # The public route, which on this branch already defaults to the
        # analytic gradient -- the loglik faithfulness backstop for both halves.
        real_fit = GLLVModels.fit_gllvm(fx.Y; family = Poisson(), grouping = fx.terms, unit = fx.group)

        bef = _S8_repeat_driver(fx; analytic_gradient = false, reps = reps)
        aft = _S8_repeat_driver(fx; analytic_gradient = true, reps = reps)
        b, a = bef.m, aft.m

        gap_before = abs(b.driver_loglik - real_fit.loglik) / max(abs(real_fit.loglik), 1.0)
        gap_after = abs(a.driver_loglik - real_fit.loglik) / max(abs(real_fit.loglik), 1.0)
        loglik_rel_ba = abs(a.driver_loglik - b.driver_loglik) / max(abs(b.driver_loglik), 1.0)

        sum_before = b.nm_seconds + b.bfgs_seconds + b.fd_gradient_seconds + b.fd_hessian_seconds +
                     b.an_gradient_seconds
        sum_after = a.nm_seconds + a.bfgs_seconds + a.fd_gradient_seconds + a.fd_hessian_seconds +
                    a.an_gradient_seconds
        gap_sum_before = abs(sum_before - b.driver_wall) / b.driver_wall
        gap_sum_after = abs(sum_after - a.driver_wall) / a.driver_wall
        speedup = bef.wall_median / aft.wall_median

        @printf("real fit_gllvm (branch default): loglik=%.10f converged=%s iterations=%d\n",
                real_fit.loglik, real_fit.converged, real_fit.iterations)
        @printf("BEFORE (analytic_gradient=false): wall median=%.6fs  min=%.6f max=%.6f (reps=%d)\n",
                bef.wall_median, bef.wall_min, bef.wall_max, bef.reps)
        @printf("AFTER  (analytic_gradient=true):  wall median=%.6fs  min=%.6f max=%.6f (reps=%d)\n",
                aft.wall_median, aft.wall_min, aft.wall_max, aft.reps)
        @printf("SPEEDUP (before/after medians) = %.3fx\n", speedup)
        @printf("  counts stable across reps: before=%s after=%s\n", bef.counts_stable, aft.counts_stable)
        @printf("  objective calls:            before=%-6d after=%-6d (delta %+d)\n",
                b.total_obj_calls, a.total_obj_calls, a.total_obj_calls - b.total_obj_calls)
        @printf("  inner Laplace-fit calls:    before=%-6d after=%-6d (delta %+d)\n",
                b.inner_calls, a.inner_calls, a.inner_calls - b.inner_calls)
        @printf("  inner Newton iters summed:  before=%-8.1f after=%-8.1f\n",
                b.inner_iters_sum, a.inner_iters_sum)
        @printf("  Nelder-Mead:      before calls=%-5d %.4fs | after calls=%-5d %.4fs\n",
                b.nm_calls, b.nm_seconds, a.nm_calls, a.nm_seconds)
        @printf("  BFGS line search: before calls=%-5d %.4fs | after calls=%-5d %.4fs\n",
                b.bfgs_calls, b.bfgs_seconds, a.bfgs_calls, a.bfgs_seconds)
        @printf("  FD gradient:      before inv=%-4d objcalls=%-5d %.4fs | after inv=%-4d objcalls=%-5d %.4fs\n",
                b.fd_gradient_invocations, b.fd_gradient_obj_calls, b.fd_gradient_seconds,
                a.fd_gradient_invocations, a.fd_gradient_obj_calls, a.fd_gradient_seconds)
        @printf("  analytic gradient: before inv=%-4d %.4fs | after inv=%-4d %.4fs  fallbacks_to_FD before=%d after=%d\n",
                b.an_gradient_invocations, b.an_gradient_seconds,
                a.an_gradient_invocations, a.an_gradient_seconds,
                b.an_gradient_fallbacks, a.an_gradient_fallbacks)
        @printf("  FD Hessian:       before inv=%-4d objcalls=%-5d %.4fs | after inv=%-4d objcalls=%-5d %.4fs\n",
                b.fd_hessian_invocations, b.fd_hessian_obj_calls, b.fd_hessian_seconds,
                a.fd_hessian_invocations, a.fd_hessian_obj_calls, a.fd_hessian_seconds)
        @printf("  section sum vs wall: before %.4f/%.4f gap=%.3f | after %.4f/%.4f gap=%.3f\n",
                sum_before, b.driver_wall, gap_sum_before, sum_after, a.driver_wall, gap_sum_after)
        @printf("  loglik: before=%.12f after=%.12f rel(after,before)=%.3e | vs real fit rel before=%.3e after=%.3e\n",
                b.driver_loglik, a.driver_loglik, loglik_rel_ba, gap_before, gap_after)
        @printf("  converged: before=%s after=%s\n", b.converged, a.converged)
        if fx.name == "glmm_200x5"
            @printf("  fixture A against the banked walls: pre-S7c %.6fs, post-S7c %.6fs, this AFTER %.6fs; Latte %.6fs -> gap now %.2fx (was %.2fx against post-S7c)\n",
                    _S8_BANKED_A_WALL_PRE_S7C, _S8_BANKED_A_WALL_S7C, aft.wall_median,
                    _S8_LATTE_A_WALL, aft.wall_median / _S8_LATTE_A_WALL,
                    _S8_BANKED_A_WALL_S7C / _S8_LATTE_A_WALL)
            @printf("  fixture A against the banked counts: objective calls %d banked vs %d after; inner Newton iters %d banked vs %.1f after (the banked pair predates S7c's counter, so this is a restatement, not a like-for-like delta)\n",
                    _S8_BANKED_A_OBJ_CALLS, a.total_obj_calls,
                    _S8_BANKED_A_INNER_ITERS, a.inner_iters_sum)
        else
            @printf("  large fixture against GA.1's banked driver wall %.4fs: this BEFORE %.4fs, this AFTER %.4fs\n",
                    _S8_GA1_LARGE_DRIVER_WALL, bef.wall_median, aft.wall_median)
        end

        push!(rows, (; fixture = fx.name, N = fx.N, G = fx.G, p = fx.p, ntheta = a.nθ, reps,
            wall_before_median_s = bef.wall_median, wall_before_min_s = bef.wall_min,
            wall_before_max_s = bef.wall_max,
            wall_after_median_s = aft.wall_median, wall_after_min_s = aft.wall_min,
            wall_after_max_s = aft.wall_max, speedup,
            obj_calls_before = b.total_obj_calls, obj_calls_after = a.total_obj_calls,
            inner_laplace_calls_before = b.inner_calls, inner_laplace_calls_after = a.inner_calls,
            inner_newton_iters_before = b.inner_iters_sum, inner_newton_iters_after = a.inner_iters_sum,
            nm_calls_before = b.nm_calls, nm_seconds_before = b.nm_seconds,
            nm_calls_after = a.nm_calls, nm_seconds_after = a.nm_seconds,
            bfgs_calls_before = b.bfgs_calls, bfgs_seconds_before = b.bfgs_seconds,
            bfgs_calls_after = a.bfgs_calls, bfgs_seconds_after = a.bfgs_seconds,
            fd_grad_inv_before = b.fd_gradient_invocations, fd_grad_seconds_before = b.fd_gradient_seconds,
            fd_grad_inv_after = a.fd_gradient_invocations, fd_grad_seconds_after = a.fd_gradient_seconds,
            an_grad_inv_after = a.an_gradient_invocations, an_grad_seconds_after = a.an_gradient_seconds,
            an_grad_fallbacks_after = a.an_gradient_fallbacks,
            fd_hess_inv_before = b.fd_hessian_invocations, fd_hess_seconds_before = b.fd_hessian_seconds,
            fd_hess_inv_after = a.fd_hessian_invocations, fd_hess_seconds_after = a.fd_hessian_seconds,
            section_sum_before_s = sum_before, section_sum_after_s = sum_after,
            sum_gap_before = gap_sum_before, sum_gap_after = gap_sum_after,
            loglik_before = b.driver_loglik, loglik_after = a.driver_loglik,
            loglik_rel_after_vs_before = loglik_rel_ba,
            loglik_gap_before_vs_real = gap_before, loglik_gap_after_vs_real = gap_after,
            converged_before = b.converged, converged_after = a.converged,
            counts_stable_before = bef.counts_stable, counts_stable_after = aft.counts_stable,
            banked_wall_post_s7c = fx.name == "glmm_200x5" ? _S8_BANKED_A_WALL_S7C : _S8_GA1_LARGE_DRIVER_WALL))
    end

    mkpath(joinpath(@__DIR__, "results"))
    out = joinpath(@__DIR__, "results", "grouped_sections_after_$(sha).tsv")
    open(out, "w") do io
        for l in header_lines()
            println(io, l)
        end
        names = fieldnames(typeof(rows[1]))
        println(io, "# columns: ", join(names, " "))
        println(io, join(names, "\t"))
        for r in rows
            println(io, join(getfield.(Ref(r), names), "\t"))
        end
    end
    println("TSV written: ", out)

    # PASS is about the integrity of the measurement, not its direction. The
    # ledger's words are "numbers reported whatever they are, no claim beyond
    # them", so a small or absent speedup is reported, not failed.
    reasons = String[]
    for r in rows
        r.sum_gap_before <= 0.10 ||
            push!(reasons, "$(r.fixture): BEFORE sections sum gap $(round(r.sum_gap_before, digits = 3)) exceeds 10%")
        r.sum_gap_after <= 0.10 ||
            push!(reasons, "$(r.fixture): AFTER sections sum gap $(round(r.sum_gap_after, digits = 3)) exceeds 10%")
        r.loglik_gap_before_vs_real <= 1e-4 ||
            push!(reasons, "$(r.fixture): BEFORE driver loglik diverges from the real fit by rel=$(round(r.loglik_gap_before_vs_real, sigdigits = 3))")
        r.loglik_gap_after_vs_real <= 1e-4 ||
            push!(reasons, "$(r.fixture): AFTER driver loglik diverges from the real fit by rel=$(round(r.loglik_gap_after_vs_real, sigdigits = 3))")
        r.loglik_rel_after_vs_before <= 1e-8 ||
            push!(reasons, "$(r.fixture): after-vs-before loglik rel=$(round(r.loglik_rel_after_vs_before, sigdigits = 3)) exceeds rtol 1e-8")
        (r.converged_before && r.converged_after) ||
            push!(reasons, "$(r.fixture): converged before=$(r.converged_before) after=$(r.converged_after)")
        r.an_grad_fallbacks_after == 0 ||
            push!(reasons, "$(r.fixture): the AFTER path fell back to the FD gradient $(r.an_grad_fallbacks_after) time(s), so its seconds are not purely analytic")
        (r.counts_stable_before && r.counts_stable_after) ||
            push!(reasons, "$(r.fixture): call counts moved between reps, so the reported counts are not deterministic")
    end
    ok = isempty(reasons)

    println()
    for r in rows
        @printf("GB.5 SUMMARY %s: wall %.6fs -> %.6fs (%.3fx), objective calls %d -> %d, inner Laplace fits %d -> %d, inner Newton iters %.1f -> %.1f\n",
                r.fixture, r.wall_before_median_s, r.wall_after_median_s, r.speedup,
                r.obj_calls_before, r.obj_calls_after,
                r.inner_laplace_calls_before, r.inner_laplace_calls_after,
                r.inner_newton_iters_before, r.inner_newton_iters_after)
    end
    if ok
        println("GATE GB.5 PASS")
    else
        println("GATE GB.5 FAIL ", join(reasons, "; "))
    end
    exit(ok ? 0 : 1)
end

function main()
    gate = parse_args(ARGS)
    gate == "after" && return main_after()
    gate == "warm" && return main_warm()
    gate == "after_warm" && return main_after_warm()
    gate == "sections" && return main_sections()
    gate == "sections_after" && return main_sections_after()
    println("Julia ", VERSION, "  threads=", Threads.nthreads())
    sha = _git_sha()

    y, group, G, N = load_fixture()
    Y1 = reshape(Float64.(y), 1, :)
    terms = [GLLVModels.GroupingTerm(:unit; mode = :indep)]

    # --- (1) real, unmodified wall-clock, exactly as f3_ours_glmm.jl does ---
    warm_s = median_s(() -> GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group, warm_start_inner = false))
    real_fit = GLLVModels.fit_gllvm(Y1; family = Poisson(), grouping = terms, unit = group, warm_start_inner = false)
    println("real fit: converged=", real_fit.converged, " loglik=", real_fit.loglik,
            " iterations=", real_fit.iterations, " stopping_reason=", real_fit.stopping_reason)
    @printf("warm median wall (%d reps): %.4f s\n", REPS, warm_s)

    # --- (2) shadow-counted replica for outer/inner/CHOLMOD counts ---
    counts = shadow_fit_counts(Matrix{Float64}(Y1), group)
    @printf("shadow: loglik=%.6f  obj_calls=%d  outer_gradient_evals=%d  inner_newton_iters_sum=%d  fresh_cholmod_analyses=%d  ok_inner_calls=%d  failed_inner_calls=%d  final_inner_status=%s\n",
            counts.shadow_loglik, counts.obj_calls, counts.grad_calls, counts.inner_iters_sum,
            counts.chol_count, counts.ok_calls, counts.failed_calls, counts.final_inner_status)

    loglik_gap = abs(counts.shadow_loglik - real_fit.loglik)
    loglik_gap_rel = loglik_gap / max(abs(real_fit.loglik), 1.0)
    @printf("shadow-vs-real loglik gap: abs=%.3e  rel=%.3e\n", loglik_gap, loglik_gap_rel)

    banked = 0.192
    band_lo, band_hi = 0.15, 0.25
    reasons = String[]
    ok = true
    if !(band_lo <= warm_s <= band_hi)
        ok = false
        push!(reasons, "warm_median_wall=$(round(warm_s, digits=4))s outside [$(band_lo),$(band_hi)]s (banked $(banked)s)")
    end
    if loglik_gap_rel > 1e-4
        ok = false
        push!(reasons, "shadow replica loglik diverges from real fit_gllvm by rel=$(round(loglik_gap_rel, sigdigits=3)) " *
                       "(shadow may not faithfully reproduce fit_grouped_nongaussian; counts are then unreliable)")
    end
    if counts.obj_calls == 0 || counts.grad_calls == 0
        ok = false
        push!(reasons, "zero objective/gradient calls recorded — instrumentation did not engage")
    end

    mkpath(joinpath(@__DIR__, "results"))
    out = joinpath(@__DIR__, "results", "grouped_glmm_$(sha).tsv")
    open(out, "w") do io
        for l in header_lines()
            println(io, l)
        end
        println(io, "N\tG\treps\twarm_median_wall_s\treal_loglik\treal_iterations\t",
                     "shadow_loglik\tobj_calls\touter_gradient_evals\tinner_newton_iters_sum\t",
                     "fresh_cholmod_analyses\tok_inner_calls\tfailed_inner_calls\tloglik_gap_rel")
        println(io, N, "\t", G, "\t", REPS, "\t", warm_s, "\t", real_fit.loglik, "\t", real_fit.iterations, "\t",
                     counts.shadow_loglik, "\t", counts.obj_calls, "\t", counts.grad_calls, "\t",
                     counts.inner_iters_sum, "\t", counts.chol_count, "\t", counts.ok_calls, "\t",
                     counts.failed_calls, "\t", loglik_gap_rel)
    end
    println("TSV written: ", out)

    if ok
        println("GATE G4.3 PASS")
    else
        println("GATE G4.3 FAIL ", join(reasons, "; "))
    end
    exit(ok ? 0 : 1)
end

main()
