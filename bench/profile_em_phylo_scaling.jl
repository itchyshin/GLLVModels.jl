# bench/profile_em_phylo_scaling.jl — leaf-S4 gate G4.4.
#
# Profiles `em_fit_phylo`'s DEFAULT sparse E-step path (src/em_phylo.jl:710-808,
# `_estep_sparse` at :322-419 selected whenever a `phy::AugmentedPhy` is
# supplied) at p = 200/1000/5000, to settle the step-8 dispute: is the
# per-iteration wall O(p) (the sparse E-step's advertised cost) or does it
# stay O(p^3) because something else in the loop still pays a dense p x p
# price?
#
# `em_phylo.jl` is NOT wired into the GLLVModels module (same convention as
# bench/em_phylo_bench.jl); it is `include`d directly.
#
# WHAT IS TIMED, AND WHY THREE BUCKETS (not the "SQUAREM map" named in the
# Fable plan): reading src/em_phylo.jl:760-787, `em_fit_phylo`'s loop body is
# (a) a per-iteration monotonicity CHECK — one call to
# `GLLVModels.gaussian_marginal_loglik` (:764-765) — then (b) the E-step
# (`_estep_sparse` by default), then (c) the M-step (`_mstep_dense`). There is
# NO SQUAREM call inside `em_fit_phylo` itself — SQUAREM acceleration is a
# SEPARATE function, `em_fit_phylo_squarem` (src/em_squarem.jl), not invoked
# by the default driver this gate profiles. Read src/likelihood.jl:212-244:
# the "loglik check" (`gaussian_marginal_loglik`'s J3 phylogenetic path) forms
# a DENSE p x p `A` and does TWO `cholesky(Symmetric(p x p))` factorizations
# (`cA_sym` at :232, `cAnB` at :234) EVERY call — i.e. the monotonicity check
# itself is O(p^3), independent of which E-step is selected. This is the
# concrete mechanism behind the Fable plan's "two p x p Choleskys per
# iteration" claim; it is real, but it is the LOGLIK CHECK's cost, not a
# SQUAREM map (em_fit_phylo does not run one). `_estep_sparse` ALSO hides a
# dense p x p Cholesky at src/em_phylo.jl:399-403 (used only to get the K_B x
# K_B `ImβΛ` term) — a second, independent O(p^3) contributor inside the
# nominally-sparse E-step.
#
# So each EM iteration of `em_fit_phylo`, even on the sparse-E-step path, pays
# (at least) THREE dense p x p Choleskys per iteration (2 in the loglik check,
# 1 inside `_estep_sparse`), all timed separately below.
#
# TIME BUDGET: each p-cell's iterated work is capped so it stays under 5 min
# (D-139 / the leaf-S4 ledger); if a single-iteration PROBE at p=5000 already
# threatens the per-cell or whole-script budget, the p=5000 cell is stopped
# after the probe and the exponent is fit from p=200/1000 only (reported
# explicitly, never silently).
#
# USAGE
#   env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. \
#       bench/profile_em_phylo_scaling.jl --gate --p 200,1000,5000
#
# leaf-S7 G7.5 re-measurement after landing a change (writes
# bench/results/em_phylo_after_<sha>.tsv instead of em_phylo_scaling_<sha>.tsv):
#   env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. \
#       bench/profile_em_phylo_scaling.jl --gate after --p 200,1000,5000

using Random, LinearAlgebra, SparseArrays, Statistics, Printf
using GLLVModels
# `em_phylo.jl` is included directly (see header) into THIS module (Main), so
# any GLLVModels-internal helper it calls UNQUALIFIED (not `GLLVModels.foo`)
# must be brought into Main's namespace explicitly here if it is not exported
# — `takahashi_diag` (src/takahashi_selinv.jl) is one such helper, used bare
# inside `_estep_sparse` (src/em_phylo.jl:386).
import GLLVModels: takahashi_diag

include(joinpath(@__DIR__, "..", "src", "em_phylo.jl"))

const REPS_COMPONENT = 3
const EM_ITERS_CAP = 5
const PER_CELL_BUDGET_S = 5 * 60.0
const PROBE_ABORT_S = 90.0          # a single loglik-check/E-step probe above this aborts the cell
const SCRIPT_BUDGET_S = 13 * 60.0   # leave margin under the 15-min whole-script ceiling

const SEED = 30

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
# CLI
# ---------------------------------------------------------------------------
function parse_args(argv)
    plist = [200, 1000, 5000]
    mode = "scaling"
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--p"
            plist = parse.(Int, split(argv[i + 1], ","))
            i += 2
        elseif a == "--gate"
            # `--gate` alone (leaf-S4 usage) takes no value; `--gate after`
            # (leaf-S7 G7.5) selects the post-change measurement, writing
            # bench/results/em_phylo_after_<sha>.tsv instead of the leaf-S4
            # em_phylo_scaling_<sha>.tsv name.
            if i + 1 <= length(argv) && argv[i + 1] == "after"
                mode = "after"
                i += 2
            else
                i += 1
            end
        else
            i += 1
        end
    end
    return plist, mode
end

# ---------------------------------------------------------------------------
# Fixture — same shape as bench/em_phylo_bench.jl's `make_fixture` (K_B=1,
# n=200, sigma_phy_scale=0.9, sigma_eps=0.5), reused verbatim.
# ---------------------------------------------------------------------------
function make_fixture(p; K_B = 1, n = 200, σ_phy_scale = 0.9, σ_eps = 0.5)
    Random.seed!(SEED)
    phy   = GLLVModels.random_balanced_tree(p; branch_length = 0.1)
    Σ_phy = GLLVModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    Λ_B   = randn(p, K_B)
    for k in 1:K_B, i in 1:(k - 1)
        Λ_B[i, k] = 0.0
    end
    σ_phy = fill(σ_phy_scale, p)
    η_B   = randn(K_B, n)
    φ     = cholesky(Symmetric(Σ_phy)).L * randn(p)
    z     = σ_phy .* φ
    y     = Λ_B * η_B .+ reshape(z, p, 1) .+ σ_eps .* randn(p, n)
    return (; phy, Σ_phy, Λ_B, σ_phy, σ_eps, y, n)
end

median_s(f; reps::Int = REPS_COMPONENT) = begin
    f()
    ts = [(@elapsed f()) for _ in 1:reps]
    median(ts)
end

function main()
    plist, mode = parse_args(ARGS)
    println("Julia ", VERSION, "  threads=", Threads.nthreads(), "  p=", plist, "  mode=", mode)
    sha = _git_sha()
    t_script_start = time()

    rows = NamedTuple[]
    skipped = String[]

    for p in plist
        if time() - t_script_start > SCRIPT_BUDGET_S
            push!(skipped, "p=$p: skipped before starting — cumulative script wall already " *
                            "$(round(time() - t_script_start, digits=1))s, over the $(SCRIPT_BUDGET_S)s budget")
            println(last(skipped))
            continue
        end

        t_cell_start = time()
        fx = make_fixture(p)

        # Warm-start state (same as em_fit_phylo's own warm start, src/em_phylo.jl:738-751),
        # obtained WITHOUT running the EM loop, for isolated component timing at a
        # realistic (not degenerate) iterate.
        Λ0, σ0 = GLLVModels.ppca_init(Matrix{Float64}(fx.y), 1)
        Λ_B0 = Matrix{Float64}(Λ0)
        σ_eps0 = float(σ0)
        σ_phy0 = fill(0.1 * sqrt(mean(abs2, fx.y)), p)

        # ---- PROBE: one untimed-context loglik-check + one E-step, single rep ----
        # S7 item 2 landed: `em_fit_phylo`'s internal monotonicity check now
        # routes through the O(p) sparse path (`gaussian_marginal_loglik_sparse_phy`)
        # whenever a `phy::AugmentedPhy` is supplied (this fixture always
        # supplies one), matching what the real driver below now does — so
        # this standalone component measurement calls the SAME function the
        # driver calls, not the dense J3 path it replaced.
        t_probe = @elapsed loglik_probe = GLLVModels.gaussian_marginal_loglik_sparse_phy(fx.y, Λ_B0, σ_eps0;
            σ_phy = σ_phy0, phy = fx.phy, σ²_phy = 1.0)
        t_loglik_probe = @elapsed GLLVModels.gaussian_marginal_loglik_sparse_phy(fx.y, Λ_B0, σ_eps0;
            σ_phy = σ_phy0, phy = fx.phy, σ²_phy = 1.0)
        t_estep_probe = @elapsed ss_probe = _estep_sparse(fx.y, Λ_B0, σ_eps0, σ_phy0, fx.phy; σ²_phy = 1.0)
        @printf("p=%-5d probe: loglik_check=%.4fs  estep=%.4fs\n", p, t_loglik_probe, t_estep_probe)

        if t_loglik_probe + t_estep_probe > PROBE_ABORT_S
            push!(skipped, "p=$p: aborted after probe — loglik_check=$(round(t_loglik_probe, digits=1))s + " *
                            "estep=$(round(t_estep_probe, digits=1))s exceeds the $(PROBE_ABORT_S)s probe-abort " *
                            "threshold; would risk the 5 min per-cell / whole-script budget")
            println(last(skipped))
            continue
        end

        # ---- component medians (independent, standalone timing) ----
        reps_here = (t_loglik_probe + t_estep_probe) * REPS_COMPONENT * 4 > PER_CELL_BUDGET_S ? 1 : REPS_COMPONENT
        loglik_ms = 1000 * median_s(() -> GLLVModels.gaussian_marginal_loglik_sparse_phy(fx.y, Λ_B0, σ_eps0;
            σ_phy = σ_phy0, phy = fx.phy, σ²_phy = 1.0); reps = reps_here)
        estep_ms = 1000 * median_s(() -> _estep_sparse(fx.y, Λ_B0, σ_eps0, σ_phy0, fx.phy; σ²_phy = 1.0);
            reps = reps_here)
        ss = _estep_sparse(fx.y, Λ_B0, σ_eps0, σ_phy0, fx.phy; σ²_phy = 1.0)
        mstep_ms = 1000 * median_s(() -> _mstep_dense(fx.y, ss); reps = reps_here)

        # ---- real driver, capped iterations, for the actual per-iteration wall ----
        # `λ_init`/`σ_eps_init`/`σ_phy_init` reuse the SAME warm start already
        # computed above (identical to what `em_fit_phylo` would compute
        # itself via `ppca_init`), so the timed call does NOT re-run PPCA's
        # O(p^3) dense eigendecomposition of the p x p sample covariance
        # inside the timed region — at p=5000 that one-time cost is tens of
        # seconds and, once amortised over only `EM_ITERS_CAP` iterations,
        # swamped the actual per-iteration EM cost this gate measures (this
        # was already true pre-fix but invisible: the O(p^3) per-iteration
        # Cholesky cost this arc removes was itself large enough to hide it).
        remaining_budget = PER_CELL_BUDGET_S - (time() - t_cell_start)
        est_iter_cost = (loglik_ms + estep_ms + mstep_ms) / 1000
        iters_cap = remaining_budget > 0 && est_iter_cost > 0 ?
            max(1, min(EM_ITERS_CAP, floor(Int, remaining_budget / est_iter_cost / 2))) : 1
        # Warm-up (JIT) on the same shapes, 1 iteration, untimed for compute purposes.
        em_fit_phylo(fx.y, 1, fx.Σ_phy; phy = fx.phy, tol = 1e-9, max_iter = 1, assert_monotone = true,
            λ_init = Λ_B0, σ_eps_init = σ_eps0, σ_phy_init = σ_phy0)
        t_driver = @elapsed emf = em_fit_phylo(fx.y, 1, fx.Σ_phy; phy = fx.phy, tol = 1e-9,
            max_iter = iters_cap, assert_monotone = true,
            λ_init = Λ_B0, σ_eps_init = σ_eps0, σ_phy_init = σ_phy0)
        driver_ms_per_iter = 1000 * t_driver / max(emf.n_iter, 1)

        cell_wall = time() - t_cell_start
        @printf("p=%-5d loglik_check=%8.3fms  estep=%8.3fms  mstep=%8.3fms  sum=%8.3fms  driver(%d iters, converged=%s)=%8.3fms/iter  cell_wall=%.1fs\n",
                p, loglik_ms, estep_ms, mstep_ms, loglik_ms + estep_ms + mstep_ms,
                emf.n_iter, emf.converged, driver_ms_per_iter, cell_wall)

        push!(rows, (; p, reps = reps_here, loglik_ms, estep_ms, mstep_ms,
                     sum_ms = loglik_ms + estep_ms + mstep_ms,
                     driver_iters = emf.n_iter, driver_converged = emf.converged,
                     driver_ms_per_iter, cell_wall_s = cell_wall))

        if cell_wall > PER_CELL_BUDGET_S
            push!(skipped, "p=$p: cell took $(round(cell_wall, digits=1))s, OVER the $(PER_CELL_BUDGET_S)s " *
                            "per-cell budget (measurement still recorded above; no further p attempted after this)")
            println(last(skipped))
            break
        end
    end

    # ---------------------------------------------------------------------
    # Scaling exponent fit on the REAL driver's per-iteration wall (the
    # metric the gate names: "wall per EM iteration ... with the default
    # sparse E-step"), using whichever p-cells actually completed.
    # ---------------------------------------------------------------------
    exponent = NaN
    verdict = "EM_SCALING undetermined (fewer than 2 completed cells)"
    if length(rows) >= 2
        logp = log.([r.p for r in rows])
        logw = log.([r.driver_ms_per_iter for r in rows])
        if length(rows) == 2
            exponent = (logw[2] - logw[1]) / (logp[2] - logp[1])
        else
            # simple least-squares slope for >=3 points
            xb, yb = mean(logp), mean(logw)
            exponent = sum((logp .- xb) .* (logw .- yb)) / sum((logp .- xb) .^ 2)
        end
        verdict = if exponent >= 2.5
            "EM_SCALING p^3"
        elseif exponent <= 1.5
            "EM_SCALING p^1"
        else
            "EM_SCALING p^$(round(exponent, digits=2)) ambiguous"
        end
    end
    println(verdict, "  (fitted exponent = ", round(exponent, digits = 3), ", from p = ",
            [r.p for r in rows], ")")

    mkpath(joinpath(@__DIR__, "results"))
    out_name = mode == "after" ? "em_phylo_after_$(sha).tsv" : "em_phylo_scaling_$(sha).tsv"
    out = joinpath(@__DIR__, "results", out_name)
    open(out, "w") do io
        for l in header_lines()
            println(io, l)
        end
        println(io, "p\treps\tloglik_check_ms\testep_ms\tmstep_ms\tsum_ms\tdriver_iters\t",
                     "driver_converged\tdriver_ms_per_iter\tcell_wall_s")
        for r in rows
            println(io, r.p, "\t", r.reps, "\t", r.loglik_ms, "\t", r.estep_ms, "\t", r.mstep_ms, "\t",
                        r.sum_ms, "\t", r.driver_iters, "\t", r.driver_converged, "\t",
                        r.driver_ms_per_iter, "\t", r.cell_wall_s)
        end
        println(io, "# fitted_scaling_exponent=", exponent)
        println(io, "# verdict=", verdict)
        for s in skipped
            println(io, "# skipped: ", s)
        end
    end
    println("TSV written: ", out)

    ok = length(rows) >= 2 && !isnan(exponent)
    gate_name = mode == "after" ? "G7.5" : "G4.4"
    if ok
        println("GATE $(gate_name) PASS")
    else
        println("GATE $(gate_name) FAIL fewer than 2 usable p-cells completed: ", join(skipped, " | "))
    end
    exit(ok ? 0 : 1)
end

main()
