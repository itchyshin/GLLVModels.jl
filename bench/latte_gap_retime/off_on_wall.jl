# Paired wall: diag_precision_kernel OFF vs ON (measure-only; default stays OFF).
#
# Timing scope (identical for both arms):
#   real GLLVModels.fit_gllvm(...; family=Poisson(), grouping=[GroupingTerm(:unit; mode=:indep)],
#        unit=group, warm_start_inner=true, diag_precision_kernel=<false|true>)
#   one untimed warm-up per (fixture, arm), then REPS timed @elapsed; report median.
#   JULIA_NUM_THREADS / OPENBLAS_NUM_THREADS from the environment (Totoro certify: 4 / 1).
#
# Fixtures: glmm_200x5 (banked CSV; md5 60a293d6…) and glmm_5000x3_g500 (same seed/DGP as
# bench/profile_grouped_glmm.jl _S8_make_large_fixture).
#
# Does NOT flip defaults. Does NOT compare to Latte.jl wall.

using GLLVModels
using DelimitedFiles, Statistics, Random, Printf, Dates
using Distributions: Poisson

const REPS = parse(Int, get(ENV, "LATTE_WALL_REPS", "5"))
const OUT_DIR = get(ENV, "LATTE_WALL_OUT",
    joinpath(@__DIR__, "..", "..", "docs", "dev-log", "evidence", "2026-09-23-latte-off-on-wall"))

function _git_sha()
    env = get(ENV, "LATTE_WALL_SHA", "")
    !isempty(env) && return env
    try
        strip(read(`git rev-parse --short HEAD`, String))
    catch
        "unknown"
    end
end

function _host()
    try
        strip(read(`hostname -s`, String))
    catch
        Sys.MACHINE
    end
end

function load_small()
    path = joinpath(@__DIR__, "..", "fixtures", "glmm_200x5.csv")
    isfile(path) || error("missing fixture $path")
    M = readdlm(path, ',', Int)
    y = Float64.(M[:, 1])
    group = Int.(M[:, 2])
    Y = reshape(y, 1, :)
    return "glmm_200x5", Y, group
end

function load_large()
    rng = Random.Xoshiro(20260920)
    beta_true = [log(4.0), log(6.0), log(2.5)]
    sd_true = [0.4, 0.3, 0.5]
    G, N, P = 500, 5000, 3
    group = rand(rng, 1:G, N)
    b = [sd_true[t] .* randn(rng, G) for t in 1:P]
    Y = zeros(Float64, P, N)
    for i in 1:N, t in 1:P
        Y[t, i] = rand(rng, Poisson(exp(beta_true[t] + b[t][group[i]])))
    end
    return "glmm_5000x3_g500", Y, group
end

function median_wall(f; reps::Int = REPS)
    f()  # untimed warm-up
    ts = [(@elapsed f()) for _ in 1:reps]
    return median(ts), ts
end

function fit_once(Y, group; kernel::Bool)
    terms = [GLLVModels.GroupingTerm(:unit; mode = :indep)]
    GLLVModels.fit_gllvm(Y; family = Poisson(), grouping = terms, unit = group,
        warm_start_inner = true, diag_precision_kernel = kernel)
end

function run_fixture(name, Y, group; sha, host, threads, blas)
    terms_note = "fit_gllvm Poisson GroupingTerm(:unit;mode=:indep) warm_start_inner=true"
    println("="^72)
    @printf("fixture=%s  size=%dx%d  REPS=%d  threads=%s blas=%s\n",
        name, size(Y, 1), size(Y, 2), REPS, threads, blas)
    flush(stdout)

    fit_off = fit_once(Y, group; kernel = false)
    fit_on = fit_once(Y, group; kernel = true)
    dll = abs(fit_on.loglik - fit_off.loglik)
    dll_rel = dll / max(1.0, abs(fit_off.loglik))
    @printf("identity: ll_off=%.12f ll_on=%.12f |Δll|=%.3e |Δll|/max=%.3e\n",
        fit_off.loglik, fit_on.loglik, dll, dll_rel)
    flush(stdout)

    wall_off, reps_off = median_wall(() -> fit_once(Y, group; kernel = false))
    wall_on, reps_on = median_wall(() -> fit_once(Y, group; kernel = true))
    speedup = wall_off / wall_on
    @printf("wall_off_median=%.6f  wall_on_median=%.6f  speedup_off/on=%.3f×\n",
        wall_off, wall_on, speedup)
    @printf("reps_off=%s\nreps_on=%s\n", join(round.(reps_off; digits=6), ","),
        join(round.(reps_on; digits=6), ","))
    flush(stdout)

    return (
        cell_id = name,
        sha = sha,
        host = host,
        threads = threads,
        blas = blas,
        reps = REPS,
        timing_scope = terms_note,
        warm_start_inner = true,
        wall_off_s = wall_off,
        wall_on_s = wall_on,
        speedup = speedup,
        ll_off = fit_off.loglik,
        ll_on = fit_on.loglik,
        abs_dll = dll,
        rel_dll = dll_rel,
        reps_off = join(string.(round.(reps_off; digits = 6)), ";"),
        reps_on = join(string.(round.(reps_on; digits = 6)), ";"),
    )
end

function main()
    mkpath(OUT_DIR)
    sha = _git_sha()
    host = _host()
    threads = get(ENV, "JULIA_NUM_THREADS", "unset")
    blas = get(ENV, "OPENBLAS_NUM_THREADS", "unset")
    started = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"

    rows = Any[]
    for loader in (load_small, load_large)
        name, Y, group = loader()
        push!(rows, run_fixture(name, Y, group; sha, host, threads, blas))
    end

    tsv = joinpath(OUT_DIR, "latte_off_on_wall_$(sha).tsv")
    open(tsv, "w") do io
        println(io, "# timing_scope=real fit_gllvm Poisson GroupingTerm(:unit;mode=:indep) warm_start_inner=true")
        println(io, "# warm=1 untimed + REPS timed @elapsed; median reported; arms differ only by diag_precision_kernel")
        println(io, "# identical_path_flags=diag_precision+reuse_identical_hf_ho (both driven by diag_precision_kernel)")
        println(io, "# fixture_glmm_200x5_md5=60a293d6c45e36cc458adc5df2458b30")
        println(io, "# started_utc=$started host=$host sha=$sha JULIA_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$blas REPS=$REPS")
        println(io, "# default_flip=NO (measure-only; diag_precision_kernel default remains false)")
        println(io, "cell_id\tsha\thost\tthreads\tblas\treps\twarm_start_inner\twall_off_s\twall_on_s\tspeedup_off_over_on\tll_off\tll_on\tabs_dll\trel_dll\treps_off\treps_on\ttiming_scope")
        for r in rows
            @printf(io,
                "%s\t%s\t%s\t%s\t%s\t%d\t%s\t%.6f\t%.6f\t%.4f\t%.12f\t%.12f\t%.6e\t%.6e\t%s\t%s\t%s\n",
                r.cell_id, r.sha, r.host, r.threads, r.blas, r.reps, string(r.warm_start_inner),
                r.wall_off_s, r.wall_on_s, r.speedup, r.ll_off, r.ll_on, r.abs_dll, r.rel_dll,
                r.reps_off, r.reps_on, r.timing_scope)
        end
    end
    println("WROTE $tsv")
    for r in rows
        @printf("SUMMARY %s: off=%.4fs on=%.4fs speedup=%.3f× |Δll|/max=%.3e\n",
            r.cell_id, r.wall_off_s, r.wall_on_s, r.speedup, r.rel_dll)
    end
end

main()
