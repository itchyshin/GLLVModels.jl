# speed_board_firstwave.jl — Phase B first-wave GLLVM cells (Julia abs walls)
#
# Plan §4.1 cell_ids. No Latte default flip — latte-gap uses banked Latte wall.
# Array / single-cell:
#   SPEED_CELL_ID=gllvm-binom-glmm-200x5 julia --project=. bench/speed_board_firstwave.jl
#   SPEED_ARRAY_INDEX=1 julia --project=. bench/speed_board_firstwave.jl
#
# Env: GLLVM_FIRSTWAVE_REPS (default 3), GLLVM_FIRSTWAVE_OUT (TSV path)

using GLLVModels
using Random
using Statistics
using Printf
using LinearAlgebra
using DelimitedFiles
using Distributions: Poisson, Binomial, NegativeBinomial, Bernoulli

const REPS = parse(Int, get(ENV, "GLLVM_FIRSTWAVE_REPS", "3"))
const LATTE_BANKED_WALL_S = 0.015  # Latte.jl on glmm_200x5 (labelled comparator only)
const CELLS = [
    "gllvm-binom-glmm-200x5",
    "gllvm-gauss-lv-t4-p20n500",
    "gllvm-pois-lv-t4-p20n500",
    "gllvm-nb2-lv-t4-p20n500",
    "gllvm-gauss-unstruct-p50n2k",
    "gllvm-gauss-phylo-fit-p200",
    "gllvm-profile-ci-glmm",
    "gllvm-latte-gap-200x5",
    "gllvm-pois-phylo-small",
    "gllvm-spatial-gauss-small",
]

function timed_median(f; reps::Int = REPS)
    f()
    secs = Float64[]
    val = nothing
    for _ in 1:reps
        t = Base.@timed f()
        push!(secs, t.time)
        val = t.value
    end
    return (median(secs), val)
end

_loglik(fit) = hasproperty(fit, :loglik) ? Float64(fit.loglik) :
               hasproperty(fit, :logLik) ? Float64(fit.logLik) :
               hasproperty(fit, :nll) ? -Float64(fit.nll) : NaN

function _sha()
    try
        strip(read(`git rev-parse --short HEAD`, String))
    catch
        get(ENV, "GLLVM_BOARD_SHA", "unknown")
    end
end

function _pick_cell()
    if haskey(ENV, "SPEED_CELL_ID") && !isempty(ENV["SPEED_CELL_ID"])
        return ENV["SPEED_CELL_ID"]
    end
    idx = parse(Int, get(ENV, "SPEED_ARRAY_INDEX", get(ENV, "SLURM_ARRAY_TASK_ID", "1")))
    1 <= idx <= length(CELLS) || error("SPEED_ARRAY_INDEX=$idx out of 1:$(length(CELLS))")
    return CELLS[idx]
end

function _write_row(path, cell_id, kind, detail, wall_s, loglik, note, status)
    mkpath(dirname(path))
    host = gethostname()
    nt = Threads.nthreads()
    sha = _sha()
    open(path, "w") do io
        println(io, "cell_id\tkind\tdetail\twall_s\tloglik\tnote\tstatus\tsha\thost\tthreads")
        @printf(io, "%s\t%s\t%s\t%.6f\t%.10g\t%s\t%s\t%s\t%s\t%d\n",
                cell_id, kind, detail,
                isnan(wall_s) ? NaN : wall_s,
                isnan(loglik) ? NaN : loglik,
                replace(note, '\t' => ' ', '\n' => ' '),
                status, sha, host, nt)
    end
    @printf("%-36s  %10.4f s  status=%s  -> %s\n", cell_id, wall_s, status, path)
end

function load_glmm_200x5()
    path = joinpath(@__DIR__, "fixtures", "glmm_200x5.csv")
    raw = readdlm(path, ',')
    y = Float64.(raw[:, 1])
    group = Int.(raw[:, 2])
    return reshape(y, 1, :), group
end

function run_binom_glmm()
    cell = "gllvm-binom-glmm-200x5"
    kind = "G-binom-glmm; grouped Binomial glmm_200x5-class"
    _, group = load_glmm_200x5()
    Random.seed!(9301)
    G = maximum(group)
    n = length(group)
    u = 0.35 .* randn(G)
    η = 0.2 .+ u[group]
    p_i = 1 ./ (1 .+ exp.(-η))
    Y = reshape(Float64.([rand(Bernoulli(clamp(p_i[i], 1e-4, 1 - 1e-4))) for i in 1:n]), 1, n)
    N = ones(Int, 1, n)
    terms = [GroupingTerm(:unit; mode = :indep)]
    wall, fit = timed_median(() -> fit_gllvm(Y; family = Binomial(), grouping = terms,
                                             unit = group, N = N, warm_start_inner = false))
    return cell, kind, "fit_gllvm Binomial grouped", wall, _loglik(fit), "julia_abs_only"
end

function run_gauss_t4()
    cell = "gllvm-gauss-lv-t4-p20n500"
    kind = "G-gauss-lv-large; T4 Gaussian (p,n,K)=(20,500,2)"
    Random.seed!(9302)
    p, n, K = 20, 500, 2
    Λ = 0.55 .* randn(p, K)
    Y = Λ * randn(K, n) .+ 0.7 .* randn(p, n)
    wall, fit = timed_median(() -> fit_gaussian_gllvm(Y; K = K))
    return cell, kind, "fit_gaussian_gllvm", wall, _loglik(fit), "julia_abs_only"
end

function run_pois_t4()
    cell = "gllvm-pois-lv-t4-p20n500"
    kind = "G-nb-binom-lv; T4 Poisson (p,n,K)=(20,500,2)"
    Random.seed!(9303)
    p, n, K = 20, 500, 2
    Λ = 0.4 .* randn(p, K)
    β = 0.15 .* randn(p)
    η = β .+ Λ * randn(K, n)
    Y = [rand(Poisson(exp(clamp(η[t, s], -6, 6)))) for t in 1:p, s in 1:n]
    wall, fit = timed_median(() -> fit_poisson_gllvm(Y; K = K, iterations = 120))
    return cell, kind, "fit_poisson_gllvm", wall, _loglik(fit), "julia_abs_only"
end

function run_nb2_t4()
    cell = "gllvm-nb2-lv-t4-p20n500"
    kind = "G-nb2-lv-scale; T4 NB2 (p,n,K)=(20,500,2)"
    Random.seed!(9304)
    p, n, K = 20, 500, 2
    r = 3.0
    Λ = 0.35 .* randn(p, K)
    β = 0.2 .* randn(p)
    η = β .+ Λ * randn(K, n)
    Y = Matrix{Int}(undef, p, n)
    for t in 1:p, s in 1:n
        μ = exp(clamp(η[t, s], -5, 5))
        Y[t, s] = rand(NegativeBinomial(r, r / (r + μ)))
    end
    wall, fit = timed_median(() -> fit_nb_gllvm(Y; K = K, iterations = 120))
    return cell, kind, "fit_nb_gllvm", wall, _loglik(fit), "julia_abs_only"
end

function run_gauss_large()
    cell = "gllvm-gauss-unstruct-p50n2k"
    kind = "G-gauss-lv-large; (p,n,K)=(50,2000,2)"
    Random.seed!(9305)
    p, n, K = 50, 2000, 2
    Λ = 0.5 .* randn(p, K)
    Y = Λ * randn(K, n) .+ 0.8 .* randn(p, n)
    wall, fit = timed_median(() -> fit_gaussian_gllvm(Y; K = K))
    return cell, kind, "fit_gaussian_gllvm", wall, _loglik(fit), "julia_abs_only"
end

function run_gauss_phylo_fit()
    # Non-EM sparse phylo Gaussian fit wall (label ≠ EM ms/iter cells 4–6).
    cell = "gllvm-gauss-phylo-fit-p200"
    kind = "G-gauss-phylo-em adjacent; non-EM fit_phylo_gaussian p=200"
    Random.seed!(9306)
    p = 200
    phy = GLLVModels.random_balanced_tree(p; branch_length = 0.15)
    # Univariate tip response under mild phylo signal.
    y = 0.4 .* randn(p) .+ 0.8 .* randn(p)
    wall, fit = timed_median(() -> fit_phylo_gaussian(phy, y; iterations = 200))
    return cell, kind, "fit_phylo_gaussian AugmentedPhy", wall, _loglik(fit),
           "julia_abs_only; label≠EM"
end

function run_profile_ci_glmm()
    # Profile on a small grouped Poisson (not unstructured LV).
    cell = "gllvm-profile-ci-glmm"
    kind = "G-profile-ci; profile on small grouped Poisson"
    Random.seed!(9307)
    G, m = 40, 5
    n = G * m
    group = repeat(1:G, inner = m)
    u = 0.4 .* randn(G)
    η = 0.3 .+ u[group]
    Y = reshape(Float64.([rand(Poisson(exp(clamp(η[i], -4, 4)))) for i in 1:n]), 1, n)
    terms = [GroupingTerm(:unit; mode = :indep)]
    fit = fit_gllvm(Y; family = Poisson(), grouping = terms, unit = group,
                    warm_start_inner = false)
    # Prefer true profile if admitted; else bank Wald via grouped_nongaussian_intervals.
    wall = NaN
    detail = "confint profile beta[1] grouped Poisson"
    note = "julia_abs_only"
    try
        wall, _ = timed_median(() -> confint(fit, Y; method = :profile, parm = "beta[1]"))
    catch
        wall, _ = timed_median(() -> grouped_nongaussian_intervals(Y, fit; unit = group))
        detail = "grouped_nongaussian_intervals Wald (profile not admitted)"
        note = "julia_abs_only; Wald_fallback_not_profile"
    end
    return cell, kind, detail, wall, _loglik(fit), note
end

function run_latte_gap()
    # Labelled ratio vs banked Latte wall — no Latte.jl call, no default flip.
    cell = "gllvm-latte-gap-200x5"
    kind = "G-latte-gap; tip HESS-class wall / banked Latte 0.015s"
    Y, group = load_glmm_200x5()
    terms = [GroupingTerm(:unit; mode = :indep)]
    wall, fit = timed_median(() -> fit_gllvm(Y; family = Poisson(), grouping = terms,
                                             unit = group, warm_start_inner = false))
    ratio = wall / LATTE_BANKED_WALL_S
    note = @sprintf("julia_abs_only; labelled_ratio_vs_latte_banked=%.2fx; latte_wall_s=%.3f; no_default_flip",
                    ratio, LATTE_BANKED_WALL_S)
    return cell, kind, "fit_gllvm Poisson grouped / Latte banked", wall, _loglik(fit), note
end

function run_pois_phylo_small()
    cell = "gllvm-pois-phylo-small"
    kind = "G-phylo-nongauss; Poisson phylo tip smoke 6×12"
    Random.seed!(9308)
    phy = GLLVModels.augmented_phy(
        "(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
    p = phy.n_leaves
    n = 12
    Y = rand(0:5, p, n)
    wall, fit = timed_median(() -> fit_phylo_glm(Y, phy; family = Poisson(),
                                                 iterations = 60))
    return cell, kind, "fit_phylo_glm Poisson", wall, _loglik(fit), "julia_abs_only"
end

function run_spatial_gauss_small()
    cell = "gllvm-spatial-gauss-small"
    kind = "G-spatial; SPDE Matérn Gaussian 8×8 mesh M=16"
    Random.seed!(9309)
    m = 8
    L = 7.0
    xs = range(0.0, L; length = m)
    N = m * m
    nodes = Matrix{Float64}(undef, N, 2)
    nodeid(i, j) = (j - 1) * m + i
    for j in 1:m, i in 1:m
        nodes[nodeid(i, j), 1] = xs[i]
        nodes[nodeid(i, j), 2] = xs[j]
    end
    tris = Matrix{Int}(undef, 2 * (m - 1) * (m - 1), 3)
    t = 0
    for j in 1:(m - 1), i in 1:(m - 1)
        a = nodeid(i, j); b = nodeid(i + 1, j)
        c = nodeid(i, j + 1); d = nodeid(i + 1, j + 1)
        t += 1; tris[t, :] = [a, b, d]
        t += 1; tris[t, :] = [a, d, c]
    end
    Cdiag, G = spde_fem(nodes, tris)
    Q = spde_precision(Cdiag, G, 1.0, 1.0; α = 2)
    M = 16
    locs = 0.5 .+ (L - 1.0) .* rand(M, 2)
    A = spde_projector(nodes, tris, locs)
    Σu = inv(Matrix(Symmetric(Q)))
    Lu = cholesky(Symmetric(Σu)).L
    u = Lu * randn(N)
    y = 1.2 .+ A * u .+ 0.35 .* randn(M)
    wall, fit = timed_median(() -> fit_spde_gaussian(y, nodes, tris, locs))
    return cell, kind, "fit_spde_gaussian", wall, _loglik(fit), "julia_abs_only"
end

const DISPATCH = Dict(
    "gllvm-binom-glmm-200x5" => run_binom_glmm,
    "gllvm-gauss-lv-t4-p20n500" => run_gauss_t4,
    "gllvm-pois-lv-t4-p20n500" => run_pois_t4,
    "gllvm-nb2-lv-t4-p20n500" => run_nb2_t4,
    "gllvm-gauss-unstruct-p50n2k" => run_gauss_large,
    "gllvm-gauss-phylo-fit-p200" => run_gauss_phylo_fit,
    "gllvm-profile-ci-glmm" => run_profile_ci_glmm,
    "gllvm-latte-gap-200x5" => run_latte_gap,
    "gllvm-pois-phylo-small" => run_pois_phylo_small,
    "gllvm-spatial-gauss-small" => run_spatial_gauss_small,
)

function main()
    cell = _pick_cell()
    out = get(ENV, "GLLVM_FIRSTWAVE_OUT",
              joinpath(@__DIR__, "results", "firstwave_$(cell)_$(_sha()).tsv"))
    println("=== GLLVM first-wave @ $(_sha()) cell=$cell ===")
    println("threads=$(Threads.nthreads()) OPENBLAS=$(get(ENV, "OPENBLAS_NUM_THREADS", "?")) reps=$REPS host=$(gethostname())")
    haskey(DISPATCH, cell) || error("unknown cell $cell")
    try
        cid, kind, detail, wall, ll, note = DISPATCH[cell]()
        _write_row(out, cid, kind, detail, wall, ll, note, "ok")
    catch e
        msg = sprint(showerror, e)
        _write_row(out, cell, "ERROR", "ERROR", NaN, NaN, msg, "error")
        @error "cell failed" cell exception = e
        # Do not rethrow — array jobs should leave an error receipt.
    end
end

main()
