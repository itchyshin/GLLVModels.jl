# =============================================================================
# 3-way CI coverage simulation for the Gaussian GLLVModels (Wald / profile / bootstrap)
# =============================================================================
#
# Scientific question
# --------------------
# The ADEMP study found that 95% *Wald* CIs for fixed effects (β) badly
# UNDERCOVER in phylogenetically-active cells (~0.2–0.6 vs nominal 0.95) — a
# known property of Wald CIs when latent / phylogenetic random effects are
# present (the Wald SE, read off inv(Hessian), does not propagate the
# latent-variable uncertainty). This script asks the open follow-up
# empirically:
#
#   Do profile-likelihood and parametric-bootstrap CIs recover the coverage
#   that the Wald interval loses?
#
# It runs a Monte-Carlo coverage simulation from KNOWN truth over a small,
# tractable grid of cells and, per (estimand × method), reports the coverage
# proportion ± Monte-Carlo SE.
#
# Estimands (scalar): σ_eps, β, and (phylo cells only) σ_phy[1].
# Methods:            Wald      = GLLVModels.confint        (Hessian)
#                     Profile   = GLLVModels.profile_ci     (LRT-inverted)
#                     Bootstrap = GLLVModels.bootstrap_ci    (parametric)
#
# Cells
# -----
#   Non-phylo: a per-(trait,site) covariate, so β is well-identified.
#   Phylo:     a TRAIT-LEVEL covariate (constant across sites within a trait),
#              which is confounded with the species-shared phylogenetic random
#              effect z_phy[t] = σ_phy[t] · φ_t, φ ~ MVN(0, Σ_phy). This is the
#              regime in which Wald β undercoverage actually appears — when the
#              fixed effect competes with the phylo random effect for the same
#              trait-level signal. (A per-(trait,site) covariate in the phylo
#              cell would be well-identified and would NOT exhibit the problem,
#              so it would not test the question of interest.)
#
# Scale convention (IMPORTANT)
# ----------------------------
# For SD-style estimands (σ_eps, σ_phy) confint, profile_ci and bootstrap_ci all
# return estimate and bounds on the scale the term name says (#156): σ_eps raw
# (positive), σ_phy signed (identity link, never exponentiated). No transform is
# applied here for any method. (Before #156 bootstrap_ci returned the packed log σ_eps
# and this script exponentiated it, wrongly also for the signed σ_phy.)
# β is linear in θ, so no transform is applied to it for any method.
#
# Run
# ---
#   julia --project=bench bench/coverage_sim.jl      # (GLLVModels is a bench dep)
#   julia --project=.     bench/coverage_sim.jl      # also works
#
# Budget knobs (environment variables, all optional):
#   COVSIM_NREPS   number of Monte-Carlo replicates per cell      (default 100)
#   COVSIM_NBOOT   bootstrap resamples per fit                    (default 199)
#   COVSIM_CELLS   comma-sep cell indices to run, e.g. "1,3"      (default all)
#
# Output: a markdown table (cells × methods × estimands) + a one-paragraph
# verdict, printed to stdout. Honest partial results print as they accrue, so a
# truncated run is still informative; MCSE columns make reliability explicit.
# =============================================================================

using GLLVModels
using Random
using LinearAlgebra
using Distributions
using Statistics
using Printf

# -----------------------------------------------------------------------------
# Cell specification
# -----------------------------------------------------------------------------
struct Cell
    name::String
    phylo::Bool
    p::Int
    n::Int
    # truth
    σ_eps::Float64
    β::Float64
    σ_phy::Float64      # used only when phylo == true
    branch_length::Float64
end

# The representative grid. Kept deliberately small so a scientifically valid
# answer on the KEY phylo cells is reachable within budget.
const CELLS = Cell[
    # ---- Non-phylo (per-(trait,site) covariate; β well-identified) ----
    Cell("nophylo_n80_p10",  false, 10,  80, 0.5, 1.0, 0.0, 0.0),
    Cell("nophylo_n160_p20", false, 20, 160, 0.5, 1.0, 0.0, 0.0),
    # ---- Phylo (trait-level covariate confounded with phylo RE; KEY cells) ----
    Cell("phylo_n120_p8",    true,   8, 120, 0.5, 1.0, 0.8, 0.5),
    Cell("phylo_n200_p16",   true,  16, 200, 0.5, 1.0, 0.8, 0.5),
]

# Which scalar estimands apply to a cell, with their confint/profile term name.
function estimands(cell::Cell)
    base = [(:σ_eps, "sigma_eps"), (:β, "beta[1]")]
    return cell.phylo ? vcat(base, [(:σ_phy, "sigma_phy[1]")]) : base
end

# -----------------------------------------------------------------------------
# Fixtures: Λ_B, the covariate design X, and (phylo) the tree-derived Σ_phy.
# Built ONCE per cell so the design is held fixed across replicates; only the
# random draws (latent factors, phylo deviations, residuals) vary per rep.
# -----------------------------------------------------------------------------
function build_fixture(cell::Cell)
    p, n = cell.p, cell.n
    K = 1
    # Deterministic loadings (one true latent axis), modest magnitude.
    rng0 = MersenneTwister(7_000 + p)
    Λ_B = reshape(0.3 .+ 0.4 .* abs.(randn(rng0, p)), p, K)
    Λ_B[2:2:end] .*= -1.0   # mix signs so it is a genuine factor, not a mean shift

    # Covariate design X (p × n × 1).
    X = zeros(Float64, p, n, 1)
    if cell.phylo
        # Trait-level covariate: constant across sites within a trait, so it is
        # confounded with the species-shared phylo random effect.
        xtrait = randn(MersenneTwister(9_000 + p), p)
        for s in 1:n, t in 1:p
            X[t, s, 1] = xtrait[t]
        end
    else
        # Per-(trait,site) covariate: β is strongly identified.
        Xr = randn(MersenneTwister(9_000 + p), p, n)
        for s in 1:n, t in 1:p
            X[t, s, 1] = Xr[t, s]
        end
    end

    Σ_phy = nothing
    L_phy = nothing
    if cell.phylo
        phy = GLLVModels.random_balanced_tree(p; branch_length = cell.branch_length)
        Σ = GLLVModels.sigma_phy_dense(phy; σ²_phy = 1.0)
        Σ_phy = Matrix(Symmetric((Σ .+ Σ') ./ 2))
        L_phy = cholesky(Symmetric(Σ_phy)).L
    end

    return (; K, Λ_B, X, Σ_phy, L_phy)
end

# Simulate one data matrix y from KNOWN truth, given a fixture and an rng.
#   y[:,s] = X[:,s,:]·β + Λ_B·η_s + (phylo: z_phy) + σ_eps·ε_s
#   z_phy[t] = σ_phy · φ_t,   φ ~ MVN(0, Σ_phy)   (shared across all sites)
function simulate_y(cell::Cell, fx, rng::AbstractRNG)
    p, n = cell.p, cell.n
    K = fx.K
    y = zeros(Float64, p, n)
    # fixed-effect mean
    for s in 1:n, t in 1:p
        y[t, s] = fx.X[t, s, 1] * cell.β
    end
    # latent factor contribution
    η = randn(rng, K, n)
    y .+= fx.Λ_B * η
    # phylo (species-shared) contribution
    if cell.phylo
        φ = fx.L_phy * randn(rng, p)
        z = cell.σ_phy .* φ
        for s in 1:n, t in 1:p
            y[t, s] += z[t]
        end
    end
    # residual noise
    y .+= cell.σ_eps .* randn(rng, p, n)
    return y
end

# True value for an estimand symbol.
truth_of(cell::Cell, sym::Symbol) =
    sym === :σ_eps ? cell.σ_eps :
    sym === :β     ? cell.β     :
    sym === :σ_phy ? cell.σ_phy : NaN

# -----------------------------------------------------------------------------
# CI extraction for one fit. Returns, per method, (lower, upper) on the RAW
# scale for the requested estimands, plus per-method "usable" flags. A bound
# pair that is NaN / non-finite or comes from a non-PD Hessian (Wald) or a
# failed/partial profile is treated as missing and EXCLUDED from the coverage
# denominator for that method (so coverage is conditional on a usable CI).
# -----------------------------------------------------------------------------
function cis_for_fit(cell::Cell, fx, fit, y; n_boot::Int, boot_seed::Int)
    syms_terms = estimands(cell)
    Σ_phy = cell.phylo ? fx.Σ_phy : nothing
    X = fx.X

    out = Dict{Symbol, Dict{Symbol, Tuple{Float64,Float64}}}()  # method => sym => (lo,hi)
    for m in (:wald, :profile, :bootstrap)
        out[m] = Dict{Symbol, Tuple{Float64,Float64}}()
    end

    # ---- Wald (all estimands at once) ----
    try
        w = confint(fit; y = y, X = X, Σ_phy = Σ_phy,
                    parm = String[t for (_, t) in syms_terms])
        for (sym, term) in syms_terms
            idx = findfirst(==(term), w.term)
            if w.pd_hessian && idx !== nothing &&
               isfinite(w.lower[idx]) && isfinite(w.upper[idx])
                out[:wald][sym] = (w.lower[idx], w.upper[idx])
            else
                out[:wald][sym] = (NaN, NaN)
            end
        end
    catch
        for (sym, _) in syms_terms
            out[:wald][sym] = (NaN, NaN)
        end
    end

    # ---- Profile (per estimand) ----
    for (sym, term) in syms_terms
        try
            pf = profile_ci(fit, term; y = y, X = X, Σ_phy = Σ_phy)
            if pf.method === :profile && isfinite(pf.lower) && isfinite(pf.upper)
                out[:profile][sym] = (pf.lower, pf.upper)
            else
                out[:profile][sym] = (NaN, NaN)
            end
        catch
            out[:profile][sym] = (NaN, NaN)
        end
    end

    # ---- Bootstrap (all estimands at once) ----
    try
        bt = bootstrap_ci(fit; y = y, X = X, Σ_phy = Σ_phy,
                          n_boot = n_boot, seed = boot_seed,
                          parms = String[t for (_, t) in syms_terms])
        for (sym, term) in syms_terms
            idx = findfirst(==(term), bt.term)
            if idx !== nothing && isfinite(bt.lower[idx]) && isfinite(bt.upper[idx])
                # Already on the scale the term name says (#156); no transform.
                out[:bootstrap][sym] = (bt.lower[idx], bt.upper[idx])
            else
                out[:bootstrap][sym] = (NaN, NaN)
            end
        end
    catch
        for (sym, _) in syms_terms
            out[:bootstrap][sym] = (NaN, NaN)
        end
    end

    return out
end

# -----------------------------------------------------------------------------
# Run one cell: n_reps Monte-Carlo replicates, tally coverage.
# -----------------------------------------------------------------------------
function run_cell(cell::Cell; n_reps::Int, n_boot::Int)
    fx = build_fixture(cell)
    syms_terms = estimands(cell)
    syms = [s for (s, _) in syms_terms]
    methods = (:wald, :profile, :bootstrap)

    # counters: method => sym => [n_covered, n_usable]
    cov = Dict(m => Dict(s => [0, 0] for s in syms) for m in methods)
    n_fit_ok = 0

    println("\n--- Cell $(cell.name)  (phylo=$(cell.phylo), p=$(cell.p), n=$(cell.n), " *
            "n_reps=$n_reps, n_boot=$n_boot) ---")
    flush(stdout)

    Σ_phy = cell.phylo ? fx.Σ_phy : nothing
    for r in 1:n_reps
        rng = MersenneTwister(100_000 * (cell.phylo ? 2 : 1) + 1000 * cell.p + r)
        y = simulate_y(cell, fx, rng)

        local fit
        try
            fit = fit_gaussian_gllvm(y; K = fx.K, X = fx.X,
                                     has_phy_unique = cell.phylo,
                                     Σ_phy = Σ_phy)
        catch
            continue
        end
        fit.converged || continue
        n_fit_ok += 1

        cis = cis_for_fit(cell, fx, fit, y; n_boot = n_boot,
                          boot_seed = 7919 * r + cell.p)

        for m in methods, s in syms
            lo, hi = cis[m][s]
            if isfinite(lo) && isfinite(hi)
                cov[m][s][2] += 1                     # usable
                t = truth_of(cell, s)
                if lo ≤ t ≤ hi
                    cov[m][s][1] += 1                 # covered
                end
            end
        end

        if r % 10 == 0 || r == n_reps
            # compact running coverage on β (the focal estimand)
            wβ = cov[:wald][:β]; pβ = cov[:profile][:β]; bβ = cov[:bootstrap][:β]
            @printf("  rep %3d/%3d | fits ok %3d | running β cover  W=%s P=%s B=%s\n",
                    r, n_reps, n_fit_ok,
                    wβ[2] > 0 ? @sprintf("%.2f", wβ[1]/wβ[2]) : "  - ",
                    pβ[2] > 0 ? @sprintf("%.2f", pβ[1]/pβ[2]) : "  - ",
                    bβ[2] > 0 ? @sprintf("%.2f", bβ[1]/bβ[2]) : "  - ")
            flush(stdout)
        end
    end

    return (; cell, syms, cov, n_fit_ok)
end

# coverage proportion and Monte-Carlo SE (binomial) from [n_cov, n_use]
function prop_mcse(c::Vector{Int})
    n_cov, n_use = c[1], c[2]
    n_use == 0 && return (NaN, NaN, 0)
    p̂ = n_cov / n_use
    se = sqrt(p̂ * (1 - p̂) / n_use)
    return (p̂, se, n_use)
end

fmt_cell(c::Vector{Int}) = begin
    p̂, se, n = prop_mcse(c)
    isnan(p̂) ? "  n/a  " : @sprintf("%.2f ± %.2f", p̂, se)
end

# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------
function print_markdown(results)
    println("\n", "="^78)
    println("RESULTS — 95% CI coverage (proportion ± Monte-Carlo SE), nominal 0.95")
    println("="^78)

    sym_label = Dict(:σ_eps => "σ_eps", :β => "β", :σ_phy => "σ_phy")

    println()
    println("| Cell | Estimand | Wald | Profile | Bootstrap | n used (W/P/B) |")
    println("|------|----------|------|---------|-----------|----------------|")
    for res in results
        cell = res.cell
        for s in res.syms
            w = res.cov[:wald][s]; p = res.cov[:profile][s]; b = res.cov[:bootstrap][s]
            @printf("| %s | %s | %s | %s | %s | %d/%d/%d |\n",
                    cell.name, sym_label[s],
                    fmt_cell(w), fmt_cell(p), fmt_cell(b),
                    w[2], p[2], b[2])
        end
    end
    println()
    println("Fits converged per cell:")
    for res in results
        @printf("  %-20s  %d / (n_reps)\n", res.cell.name, res.n_fit_ok)
    end
end

# One-paragraph verdict focused on β recovery in phylo cells.
function print_verdict(results)
    println("\n", "="^78)
    println("VERDICT — do profile / bootstrap recover the β coverage Wald loses?")
    println("="^78)
    for res in results
        res.cell.phylo || continue
        w, _, _ = prop_mcse(res.cov[:wald][:β])
        p, _, _ = prop_mcse(res.cov[:profile][:β])
        b, _, _ = prop_mcse(res.cov[:bootstrap][:β])
        @printf("  %-18s  β coverage:  Wald=%.2f  Profile=%.2f  Bootstrap=%.2f\n",
                res.cell.name, w, p, b)
    end
    println()
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
function main()
    n_reps = parse(Int, get(ENV, "COVSIM_NREPS", "100"))
    n_boot = parse(Int, get(ENV, "COVSIM_NBOOT", "199"))
    cell_sel = get(ENV, "COVSIM_CELLS", "")
    idxs = isempty(cell_sel) ? collect(1:length(CELLS)) :
           parse.(Int, split(cell_sel, ","))

    println("="^78)
    println("3-way CI coverage simulation (Wald / profile / bootstrap)")
    println("n_reps=$n_reps  n_boot=$n_boot  cells=$(idxs)")
    println("="^78)

    t0 = time()
    results = NamedTuple[]
    for i in idxs
        push!(results, run_cell(CELLS[i]; n_reps = n_reps, n_boot = n_boot))
        # Print incrementally so a truncated run is still informative.
        print_markdown(results)
    end
    print_verdict(results)
    @printf("\nTotal wall-clock: %.1f s\n", time() - t0)
    return results
end

main()
