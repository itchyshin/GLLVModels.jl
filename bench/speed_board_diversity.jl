# speed_board_diversity.jl — tip-abs walls for NEW GLLVM speed-board kinds
#
# Julia absolute wall only (no R H2H unless a cheap paired script exists — none
# for these kinds). Totoro: JULIA_NUM_THREADS≤16, OPENBLAS_NUM_THREADS=1.
# Not CI. No Latte default-ON. Soft claim fence: tip abs, not before/after.
#
#   JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
#     julia --project=. bench/speed_board_diversity.jl
#
# Optional: GLLVM_DIVERSITY_REPS=3 (default), GLLVM_DIVERSITY_OUT=path.tsv

using GLLVModels
using Random
using Statistics
using Printf
using LinearAlgebra
using SparseArrays
using Distributions: Poisson, Categorical, NegativeBinomial

const REPS = parse(Int, get(ENV, "GLLVM_DIVERSITY_REPS", "3"))
const OUT = get(ENV, "GLLVM_DIVERSITY_OUT",
                joinpath(@__DIR__, "results",
                         "board_gllvm_diversity_$(get(ENV, "GLLVM_BOARD_SHA", "tip")).tsv"))

function timed_median(f; reps::Int = REPS)
    f()  # warm / compile
    secs = Float64[]
    val = nothing
    for _ in 1:reps
        t = Base.@timed f()
        push!(secs, t.time)
        val = t.value
    end
    return (median(secs), val)
end

function _loglik(fit)
    hasproperty(fit, :loglik) && return Float64(fit.loglik)
    hasproperty(fit, :logLik) && return Float64(fit.logLik)
    return NaN
end

# TSV rows accumulate here
const ROWS = Vector{NamedTuple}()

function record!(cell_id::String, kind::String, detail::String, wall_s::Float64,
                 loglik::Float64; note::String = "julia_abs_only")
    push!(ROWS, (; cell_id, kind, detail, wall_s, loglik, note, status = "ok"))
    @printf("%-36s  %10.4f s  ll=% .6g  %s\n", cell_id, wall_s, loglik, detail)
end

function record_err!(cell_id::String, kind::String, err)
    msg = sprint(showerror, err)
    push!(ROWS, (; cell_id, kind, detail = "ERROR", wall_s = NaN,
                 loglik = NaN, note = msg, status = "error"))
    @printf("%-36s  ERROR  %s\n", cell_id, msg)
end

# ---------------------------------------------------------------------------
# 1. Ordinal unstructured Laplace (distinct from NB/Binom unstruct)
# ---------------------------------------------------------------------------
function cell_ordinal()
    cell = "gllvm-ordinal-unstruct-small"
    kind = "Ordinal unstructured Laplace; (p,n,K,C)=(6,80,1,3)"
    try
        Random.seed!(9201)
        p, n, K, C = 6, 80, 1, 3
        τ = [-0.6, 0.7]
        Λ = 0.55 .* randn(p, K)
        η = Λ * randn(K, n)
        Y = Matrix{Int}(undef, p, n)
        for i in 1:n, t in 1:p
            pr = [GLLVModels._ord_prob(c, η[t, i], τ) for c in 1:C]
            Y[t, i] = rand(Categorical(pr))
        end
        wall, fit = timed_median(() -> fit_ordinal_gllvm(Y; K = K, iterations = 120))
        record!(cell, kind, "fit_ordinal_gllvm", wall, _loglik(fit))
    catch e
        record_err!(cell, kind, e)
    end
end

# ---------------------------------------------------------------------------
# 2–3. ZIP / ZINB unstructured (two-part; distinct from NB unstruct)
# ---------------------------------------------------------------------------
function _zi_counts(p, n, K; seed, nb::Bool = false)
    Random.seed!(seed)
    βz = 0.3 .* randn(p) .- 0.5
    βc = 0.3 .* randn(p) .+ 0.8
    Λc = 0.4 .* randn(p, K)
    π = inv.(1 .+ exp.(-βz))
    Z = randn(K, n)
    ηc = βc .+ Λc * Z
    Y = zeros(Int, p, n)
    r = 3.0
    for t in 1:p, s in 1:n
        if rand() < π[t]
            Y[t, s] = 0
        elseif nb
            μ = exp(clamp(ηc[t, s], -4, 4))
            Y[t, s] = rand(NegativeBinomial(r, r / (r + μ)))
        else
            Y[t, s] = rand(Poisson(exp(clamp(ηc[t, s], -4, 4))))
        end
    end
    return Y
end

function cell_zip()
    cell = "gllvm-zip-unstruct-small"
    kind = "ZIP unstructured Laplace; (p,n,K)=(5,40,1)"
    try
        Y = _zi_counts(5, 40, 1; seed = 9202, nb = false)
        wall, fit = timed_median(() -> fit_zip_gllvm(Y; K = 1, iterations = 80))
        record!(cell, kind, "fit_zip_gllvm", wall, _loglik(fit))
    catch e
        record_err!(cell, kind, e)
    end
end

function cell_zinb()
    cell = "gllvm-zinb-unstruct-small"
    kind = "ZINB unstructured Laplace; (p,n,K)=(5,40,1)"
    try
        Y = _zi_counts(5, 40, 1; seed = 9203, nb = true)
        wall, fit = timed_median(() -> fit_zinb_gllvm(Y; K = 1, iterations = 80))
        record!(cell, kind, "fit_zinb_gllvm", wall, _loglik(fit))
    catch e
        record_err!(cell, kind, e)
    end
end

# ---------------------------------------------------------------------------
# 4. Phylo Poisson GLM (joint Laplace; not Gaussian EM phylo)
# ---------------------------------------------------------------------------
function cell_phylo_pois_glm()
    cell = "gllvm-phylo-pois-glm-smoke"
    kind = "Phylo Poisson GLM augmented-state Laplace; 6 tips × n=12"
    try
        Random.seed!(9204)
        phy = GLLVModels.augmented_phy(
            "(((A:0.3,B:0.3):0.2,(C:0.3,D:0.3):0.2):0.2,(E:0.4,F:0.4):0.2);")
        p = phy.n_leaves
        n = 12
        Y = rand(0:5, p, n)
        wall, fit = timed_median(() -> fit_phylo_glm(Y, phy; family = Poisson(),
                                                     iterations = 60))
        record!(cell, kind, "fit_phylo_glm Poisson", wall, _loglik(fit))
    catch e
        record_err!(cell, kind, e)
    end
end

# ---------------------------------------------------------------------------
# 5. Spatial SPDE Gaussian smoke (not GLLVM LV; spatial path)
# ---------------------------------------------------------------------------
function cell_spatial_spde()
    cell = "gllvm-spatial-spde-gauss-smoke"
    kind = "SPDE Matérn Gaussian spatial; 8×8 mesh, M=16 sites"
    try
        Random.seed!(9205)
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
        record!(cell, kind, "fit_spde_gaussian", wall, _loglik(fit))
    catch e
        record_err!(cell, kind, e)
    end
end

# ---------------------------------------------------------------------------
# 6. Concurrent / constrained LV (Poisson + X)
# ---------------------------------------------------------------------------
function cell_concurrent()
    cell = "gllvm-concurrent-pois-small"
    kind = "Concurrent (constrained) Poisson LV; (p,n,q,K)=(5,40,2,1)"
    try
        rng = MersenneTwister(9206)
        p, n, q, K = 5, 40, 2, 1
        β = 0.2 .* randn(rng, p)
        Λ = zeros(p, K)
        for k in 1:K, t in k:p
            Λ[t, k] = (t == k ? 0.7 : 0.2 * randn(rng))
        end
        B = 0.4 .* randn(rng, q, K)
        X = randn(rng, n, q)
        Y = Matrix{Int}(undef, p, n)
        for s in 1:n
            z = B' * X[s, :] .+ randn(rng, K)
            η = β .+ Λ * z
            for t in 1:p
                Y[t, s] = rand(rng, Poisson(exp(clamp(η[t], -8, 8))))
            end
        end
        wall, fit = timed_median(() -> fit_concurrent_gllvm(Y; family = Poisson(),
                                                            X = X, K = K,
                                                            iterations = 80))
        record!(cell, kind, "fit_concurrent_gllvm Poisson", wall, _loglik(fit))
    catch e
        record_err!(cell, kind, e)
    end
end

# ---------------------------------------------------------------------------
# 7. Missing-data Gaussian (mask path)
# ---------------------------------------------------------------------------
function cell_missing_gauss()
    cell = "gllvm-missing-gauss-mask"
    kind = "Gaussian GLLVM with ~10% missing mask; (p,n,K)=(8,60,2)"
    try
        Random.seed!(9207)
        p, n, K = 8, 60, 2
        Λ = 0.55 .* randn(p, K)
        # Keep observed residual scale healthy so Σ stays PD under mask.
        Y = Λ * randn(K, n) .+ 0.8 .* randn(p, n)
        mask = trues(p, n)
        n_miss = max(1, round(Int, 0.10 * p * n))
        for idx in randperm(p * n)[1:n_miss]
            r = ((idx - 1) % p) + 1
            c = ((idx - 1) ÷ p) + 1
            mask[r, c] = false
            # Leave Y[r,c] as simulated; mask drops it from the likelihood
            # (zeroing can make empirical Σ singular on small grids).
        end
        wall, fit = timed_median(() -> fit_gaussian_gllvm(Y; K = K, mask = mask))
        record!(cell, kind, "fit_gaussian_gllvm mask", wall, _loglik(fit))
    catch e
        record_err!(cell, kind, e)
    end
end

# ---------------------------------------------------------------------------
# 8. Larger profile CI (medium grid vs board small 8×40×1)
# ---------------------------------------------------------------------------
function cell_profile_ci_medium()
    cell = "gllvm-profile-ci-medium"
    kind = "Profile CI beta[1] Poisson; (p,n,K)=(12,50,1)"
    try
        Random.seed!(9208)
        p, n, K = 12, 50, 1
        Λ = 0.5 .* randn(p, K)
        β = 0.2 .* randn(p)
        η = β .+ Λ * randn(K, n)
        Y = [rand(Poisson(exp(clamp(η[t, s], -5, 5)))) for t in 1:p, s in 1:n]
        fit = fit_poisson_gllvm(Y; K = K, iterations = 100)
        wall, _ = timed_median(() -> confint(fit, Y; method = :profile,
                                             parm = "beta[1]"))
        record!(cell, kind, "confint profile beta[1]", wall, _loglik(fit);
                note = "julia_abs_only; larger than gllvm-profile-ci-small")
    catch e
        record_err!(cell, kind, e)
    end
end

# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------
function main()
    mkpath(dirname(OUT))
    sha = try
        strip(read(`git rev-parse --short HEAD`, String))
    catch
        get(ENV, "GLLVM_BOARD_SHA", "unknown")
    end
    only = lowercase(strip(get(ENV, "GLLVM_DIVERSITY_ONLY", "all")))
    println("=== GLLVM speed-board diversity @ $sha ===")
    println("threads=$(Threads.nthreads())  OPENBLAS=$(get(ENV, "OPENBLAS_NUM_THREADS", "?"))  reps=$REPS  only=$only")
    println("host=$(gethostname())")
    run(name, f) = (only == "all" || only == name) && f()
    run("ordinal", cell_ordinal)
    run("zip", cell_zip)
    run("zinb", cell_zinb)
    run("phylo", cell_phylo_pois_glm)
    run("spatial", cell_spatial_spde)
    run("concurrent", cell_concurrent)
    run("missing", cell_missing_gauss)
    run("profile", cell_profile_ci_medium)

    open(OUT, "w") do io
        println(io, "cell_id\tkind\tdetail\twall_s\tloglik\tnote\tstatus\tsha\thost\tthreads")
        host = gethostname()
        nt = Threads.nthreads()
        for r in ROWS
            @printf(io, "%s\t%s\t%s\t%.6f\t%.10g\t%s\t%s\t%s\t%s\t%d\n",
                    r.cell_id, r.kind, r.detail,
                    isnan(r.wall_s) ? NaN : r.wall_s,
                    isnan(r.loglik) ? NaN : r.loglik,
                    replace(r.note, '\t' => ' ', '\n' => ' '),
                    r.status, sha, host, nt)
        end
    end
    println("Wrote $OUT")
    n_ok = count(r -> r.status == "ok", ROWS)
    println("ok=$n_ok / $(length(ROWS))")
end

main()
