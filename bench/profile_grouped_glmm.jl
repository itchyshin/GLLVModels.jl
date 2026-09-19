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
using DelimitedFiles, Statistics, LinearAlgebra, Printf, Profile
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
    # S7c: per-call inner-iteration record, split cold (b_cache empty — the
    # first successful call only) vs warm (every call thereafter), mirroring
    # src/grouped_nongaussian_fit.jl's `_grouped_nongaussian_objective` cache
    # exactly (same b_cache Ref pattern) so this shadow measures what the
    # real fitter does when `warm=true`.
    b_cache = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    cold_iters = Int[]
    warm_iters = Int[]

    function counting_objective(value)
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
            b_init = warm ? b_cache[] : nothing
            was_cold = b_init === nothing
            result = GLLVModels.joint_grouped_laplace_loglik(family, vec(data), vec(trials), D,
                gamma, W; link = GLLVModels._grouped_nongaussian_link(Val(kind)),
                maxiter = 100, tol = 1e-8, b_init = b_init)
            inner_iters_sum[] += result.iterations
            if result.status === :ok
                chol_count[] += 2 * result.iterations + 1
                ok_calls[] += 1
                was_cold ? push!(cold_iters, result.iterations) : push!(warm_iters, result.iterations)
                warm && (b_cache[] = copy(result.mode))
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

    grad_calls = Ref(0)
    fd_gradient = (obj, val) -> begin
        grad_calls[] += 1
        GLLVModels._grouped_fd_gradient(obj, val)
    end

    result1 = Optim.optimize(counting_objective, theta0, Optim.NelderMead(),
        Optim.Options(g_tol = 1e-4, iterations = 100))
    candidate = collect(Optim.minimizer(result1))
    candidate_gradient = fd_gradient(counting_objective, candidate)
    final_result = result1
    if all(isfinite, candidate_gradient)
        gradient! = (storage, value) -> (storage .= fd_gradient(counting_objective, value))
        refined = try
            Optim.optimize(counting_objective, gradient!, candidate, Optim.BFGS(),
                Optim.Options(g_tol = 1e-4, iterations = 100))
        catch
            nothing
        end
        if refined !== nothing
            refined_estimate = collect(Optim.minimizer(refined))
            refined_value = counting_objective(refined_estimate)
            if isfinite(refined_value) && !(refined_value >= 1e12) &&
                    refined_value <= counting_objective(candidate)
                final_result = refined
            end
        end
    end
    estimate = collect(Optim.minimizer(final_result))
    shadow_loglik = -counting_objective(estimate)

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

    # Per-call distribution: the FIRST successful call is necessarily cold
    # (b_cache starts empty) even when warm=true; every call after it is
    # warm-started from the previous call's mode.
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

function main()
    gate = parse_args(ARGS)
    gate == "after" && return main_after()
    gate == "warm" && return main_warm()
    gate == "after_warm" && return main_after_warm()
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
