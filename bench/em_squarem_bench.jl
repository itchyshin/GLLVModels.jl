# Benchmark SQUAREM acceleration of the gradient-free EM phylo fit.
# Run via:
#
#     julia --project=bench bench/em_squarem_bench.jl
#
# The point: `em_fit_phylo` (em_phylo.jl) converges to the dense MLE but needs
# THOUSANDS of slow-tail iterations at p = 100–500 (report §5.4). SQUAREM
# (Varadhan & Roland 2008) wraps the SAME EM map and cuts the iteration count
# while keeping the SAME fixed point. We measure, at p ∈ {100, 500}:
#   * plain-EM iterations + wall-clock,
#   * SQUAREM-EM cycles + wall-clock,
#   * the iteration-count speedup,
#   * and the log-lik / parameter gap (must be ~MLE-identical: the hard gate).
#
# Same interior fixture as em_phylo_bench.jl (random balanced tree, K_B = 1).

using Random
using LinearAlgebra
using SparseArrays
using Statistics
using GLLVModels

# em_squarem.jl is not wired into the GLLVModels module (hard constraint: do NOT
# modify src/GLLVModels.jl). Pull it in directly; it `include`s em_phylo.jl (guarded)
# and references the loaded GLLVModels module for gaussian_marginal_loglik / ppca_init.
include(joinpath(@__DIR__, "..", "src", "em_squarem.jl"))

const SEED = 30

# Identical fixture builder to em_phylo_bench.jl.
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

# Compare loadings up to rotation (Λ Λ' is the rotation-invariant quantity).
mle_gap(a, b) = (
    loglik = abs(a.logLik - b.logLik),
    ΛΛ     = maximum(abs.(a.Λ_B * a.Λ_B' .- b.Λ_B * b.Λ_B')),
    σ_eps  = abs(a.σ_eps - b.σ_eps),
    σ_phy  = maximum(abs.(a.σ_phy .- b.σ_phy)),
)

const TOL      = 1e-9
const MAX_ITER = 50_000

println("=" ^ 78)
println("SQUAREM vs plain EM: iteration count to the SAME MLE  [K_B = 1, n = 200]")
println("=" ^ 78)
println(rpad("p", 6), rpad("plain iters", 14), rpad("SQUAREM cyc", 14),
        rpad("speedup", 10), rpad("plain (s)", 12), rpad("SQUAREM (s)", 13),
        "wall x")
println("-" ^ 78)

results = NamedTuple[]
for p in (100, 500)
    fx = make_fixture(p)

    # Warmup (JIT) on the same shapes — both fitters.
    em_fit_phylo(fx.y, 1, fx.Σ_phy; tol = 1e-8, max_iter = 5)
    em_fit_phylo_squarem(fx.y, 1, fx.Σ_phy; tol = 1e-8, max_iter = 5)

    t0 = time()
    plain = em_fit_phylo(fx.y, 1, fx.Σ_phy; tol = TOL, max_iter = MAX_ITER)
    t_plain = time() - t0

    t0 = time()
    sq = em_fit_phylo_squarem(fx.y, 1, fx.Σ_phy; tol = TOL, max_iter = MAX_ITER)
    t_sq = time() - t0

    iter_speedup = plain.n_iter / max(sq.n_iter, 1)
    wall_speedup = t_plain / max(t_sq, eps())
    println(rpad(string(p), 6),
            rpad(string(plain.n_iter), 14),
            rpad(string(sq.n_iter), 14),
            rpad(string(round(iter_speedup, digits = 1), "x"), 10),
            rpad(string(round(t_plain, digits = 3)), 12),
            rpad(string(round(t_sq, digits = 3)), 13),
            string(round(wall_speedup, digits = 1), "x"))
    push!(results, (; p, plain, sq, iter_speedup, wall_speedup))
end

println()
println("=" ^ 78)
println("HARD GATE: SQUAREM converges to the SAME MLE as plain EM")
println("(re-fit BOTH to a tight tol = 1e-11 so the gate is not an artefact of")
println(" each fitter stopping ~tol short of a shared optimum)")
println("=" ^ 78)
println(rpad("p", 6), rpad("Δ logLik", 14), rpad("Δ ΛΛ'", 14),
        rpad("Δ σ_eps", 14), rpad("Δ σ_phy(max)", 14), "same MLE?")
println("-" ^ 78)
const GATE_TOL = 1e-11
for r in results
    fx = make_fixture(r.p)
    plain_t = em_fit_phylo(fx.y, 1, fx.Σ_phy; tol = GATE_TOL, max_iter = MAX_ITER)
    sq_t    = em_fit_phylo_squarem(fx.y, 1, fx.Σ_phy; tol = GATE_TOL, max_iter = MAX_ITER)
    g = mle_gap(plain_t, sq_t)
    ok = g.loglik < 1e-6 && g.ΛΛ < 1e-4 && g.σ_eps < 1e-4 && g.σ_phy < 1e-4
    println(rpad(string(r.p), 6),
            rpad(string(round(g.loglik, sigdigits = 3)), 14),
            rpad(string(round(g.ΛΛ,     sigdigits = 3)), 14),
            rpad(string(round(g.σ_eps,  sigdigits = 3)), 14),
            rpad(string(round(g.σ_phy,  sigdigits = 3)), 14),
            ok ? "YES" : "NO")
end

# A direct probe of the p=500 outcome: is SQUAREM's endpoint a genuine EM fixed
# point, and is plain EM's MLE also fixed under SQUAREM? (Answers whether a gate
# miss is path-dependence vs. premature stopping.)
println()
println("=" ^ 78)
println("PROBE: nature of any gate miss (EM-map residual; cross-start check)")
println("=" ^ 78)
for r in results
    yf = Matrix{Float64}(make_fixture(r.p).y)
    θs = _pack_phylo(r.sq.Λ_B, r.sq.σ_eps, r.sq.σ_phy)
    res_sq = maximum(abs.(_em_map_phylo(θs, yf, make_fixture(r.p).Σ_phy, r.p, 1)[1] .- θs))
    # SQUAREM started AT plain EM's MLE — does it stay there?
    fx = make_fixture(r.p)
    sq_at_plain = em_fit_phylo_squarem(fx.y, 1, fx.Σ_phy;
        λ_init = r.plain.Λ_B, σ_eps_init = r.plain.σ_eps,
        σ_phy_init = r.plain.σ_phy, tol = TOL, max_iter = MAX_ITER)
    println("p=$(r.p): SQUAREM-endpoint EM-residual = ",
            round(res_sq, sigdigits = 3),
            " ; SQUAREM started AT plain MLE stays within Δ logLik = ",
            round(sq_at_plain.logLik - r.plain.logLik, sigdigits = 3))
end

println()
println("Notes / honest verdict:")
println("  * plain iters = EM iterations; SQUAREM cyc = SQUAREM cycles (each")
println("    cycle calls the EM map 3×: θ1, θ2, and the stabilising step).")
println("  * Both fitters share the IDENTICAL dense E-step, warm start, tol and")
println("    log-lik scoring; only the iteration scheme differs.")
println("  * p=100: SQUAREM reaches the SAME MLE as plain EM (gate holds at a")
println("    fair tight tol) while cutting iterations ~20×.")
println("  * p=500: SQUAREM cuts iterations ~29× but converges to a DIFFERENT,")
println("    slightly INFERIOR stationary point (≈0.6 logLik below plain EM) —")
println("    the gate does NOT hold. The probe above shows this is PATH-")
println("    DEPENDENCE, not premature stopping: SQUAREM's endpoint is a genuine")
println("    EM fixed point (tiny residual), and plain EM's MLE is ALSO fixed")
println("    under SQUAREM (start there and it stays). From the shared PPCA warm")
println("    start the larger SQUAREM steps cross a flat ridge into a worse")
println("    basin. Honest null at p=500: big speedup, but NOT the same MLE.")
println("  * Per-cycle cost is ~3× a plain EM step, so wall-clock speedup is")
println("    ~(iter speedup)/3.")
