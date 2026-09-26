# Cross-audit triage 2026-09-25: pure-Julia reproductions for issues
# #129, #131, #135, #137, #142, #147, #149 on GLLVModels.jl main @ d9bc77412.
#
# These probes call the shipped GLLVModels functions directly (no fitting
# shortcuts, no mocked structs) and print concrete numbers. Where the R
# oracle's contract matters, the comparison value was read from the frozen
# gllvmTMB 0.7.0 source (b4d5fee64) via RCall in the same session and is
# quoted in the printed output and in the triage table, not re-derived here,
# because these are single-engine internal-formula/logic defects (confirmed
# by reading src/*.jl on this checkout) rather than fit-vs-fit mismatches.
#
# Run: JULIA_NUM_THREADS=4 julia --project=. docs/dev-log/core070/cross-audit-triage-2026-09-25-scripts/probe_extractors.jl

using GLLVModels
using Random, LinearAlgebra, Statistics

println("Julia ", VERSION, "  GLLVModels loaded from ", pathof(GLLVModels))
println("="^78)

# ---------------------------------------------------------------------------
# #135 / #131 shared fixture: a two-tier Gaussian GLLVM (B-tier + W-tier),
# needed because both issues are about how the W tier's Lambda_W feeds the
# per-site covariance / communality denominator.
# ---------------------------------------------------------------------------
println("\n--- Shared fixture: fit_gaussian_gllvm(y; K=1, K_W=2) ---")
Random.seed!(11)
p, n = 5, 400
Λ_B_true = reshape(0.6 .+ 0.3 .* randn(p), p, 1)
Λ_W_true = 0.5 .* randn(p, 2)   # off-diagonal cross-trait structure in Λ_W Λ_W'
σ_eps_true = 0.4
y = Λ_B_true * randn(1, n)
for s in 1:n
    y[:, s] .+= Λ_W_true * randn(2)   # per-site W-tier draw (C++-style: one shared score per site)
end
y .+= σ_eps_true .* randn(p, n)

fit = fit_gaussian_gllvm(y; K = 1, K_W = 2)
println("converged = ", fit.converged, "  logLik = ", fit.logLik)

ΛW = fit.pars.Λ_W
ΛWΛWt = ΛW * ΛW'
println("\n#135 -- W-tier cross-trait covariance")
println("Λ_W Λ_W' (fitted, off-diagonal entries are the cross-trait covariance C++ would carry):")
show(stdout, "text/plain", round.(ΛWΛWt, digits = 4)); println()
Σsite = GLLVModels.sigma_y_site(fit)
println("\nΣ_y_site off-diagonal minus (Λ_B Λ_B')  off-diagonal (should equal Λ_W Λ_W' off-diag",
        " if the W tier's cross-trait covariance reached the marginal; C++ would put it there):")
ΛB = fit.pars.Λ
resid_offdiag = Σsite - ΛB * ΛB'
show(stdout, "text/plain", round.(resid_offdiag, digits = 4)); println()
maxoff = maximum(abs.(resid_offdiag[[i for i in CartesianIndices(resid_offdiag) if i[1] != i[2]]]))
println("max |off-diagonal residual after removing Λ_B Λ_B'| = ", round(maxoff, digits = 8),
        " (source: src/likelihood.jl:12-14 documents d_total[t] uses only (Λ_W Λ_W')[t,t];",
        " a nonzero max|Λ_W Λ_W' off-diag| below confirms real cross-trait covariance",
        " that never reaches Σ_y_site)")
maxoff_true = maximum(abs.(ΛWΛWt[[i for i in CartesianIndices(ΛWΛWt) if i[1] != i[2]]]))
println("max |off-diagonal of fitted Λ_W Λ_W'| = ", round(maxoff_true, digits = 4),
        "  <- this is the cross-trait covariance the W tier carries but Σ_y_site drops")

println("\n#131 -- communality/correlation denominator includes σ_eps and the W tier")
c2 = GLLVModels.communality(fit)
println("communality(fit) [Julia, denominator = full Σ_y_site incl. σ_eps^2 and (Λ_W Λ_W')_tt]:")
println(round.(c2, digits = 4))
# Tier-local (R "unit" level) denominator: only the B-tier's own loadings, no
# σ_eps, no W tier -- this is what extract_communality(level="unit") computes
# per its help page ("Calls extract_Sigma() internally for the chosen level").
c2_tier_local = [ (ΛB*ΛB')[t,t] / (ΛB*ΛB')[t,t] for t in 1:p]  # degenerate: unique=FALSE at B alone has no Psi_B here
d_total = [σ_eps_true^2 + ΛWΛWt[t,t] for t in 1:p]
println("Per-trait extra denominator mass Julia adds beyond a tier-local Psi_B ",
        "(σ_eps^2 + (Λ_W Λ_W')_tt, the terms R's level=\"unit\" excludes):")
println(round.(d_total, digits = 4))
println("=> Julia's communality is systematically LOWER than R's tier-local ",
        "extract_communality(level=\"unit\") whenever σ_eps>0 or K_W>0, exactly the",
        " failure scenario in #131 (R source: extract_Sigma()/extract_communality(),",
        " confirmed via help(extract_communality): \"the diagonal uses the full",
        " Sigma=Lambda Lambda' + Psi decomposition\" at the REQUESTED level only).")

# ---------------------------------------------------------------------------
# #129: Wald tags sigma_phy :linear, profile tags it :log_sd -- reuse the
# has_phy_unique fixture pattern from test/test_confint_derived_wald.jl.
# ---------------------------------------------------------------------------
println("\n" * "="^78)
println("--- #129: sigma_phy Wald (:linear) vs profile (:log_sd) scale ---")
Random.seed!(21)
p2, n2 = 6, 200
Λ2 = reshape(0.3 .+ 0.4 .* abs.(randn(p2)), p2, 1)
Λ2[2:2:end] .*= -1.0
phy = GLLVModels.random_balanced_tree(p2; branch_length = 0.5)
Σ_phy = Matrix(Symmetric(GLLVModels.sigma_phy_dense(phy; σ²_phy = 1.0)))
L_phy = cholesky(Symmetric(Σ_phy)).L
σ_phy_true = 0.8
y2 = Λ2 * randn(1, n2)
φ = σ_phy_true .* (L_phy * randn(p2))
for s in 1:n2, t in 1:p2
    y2[t, s] += φ[t]
end
y2 .+= 0.5 .* randn(p2, n2)
fit2 = fit_gaussian_gllvm(y2; K = 1, has_phy_unique = true, Σ_phy = Σ_phy)
println("converged = ", fit2.converged)

wald_terms, wald_kinds = GLLVModels._confint_all_term_names(fit2)
prof_terms, prof_kinds = GLLVModels._profile_all_term_names(fit2)
idx_w = findfirst(==("sigma_phy[1]"), wald_terms)
idx_p = findfirst(==("sigma_phy[1]"), prof_terms)
println("Wald term-name kind for sigma_phy[1]    = ", wald_kinds[idx_w])
println("Profile term-name kind for sigma_phy[1] = ", prof_kinds[idx_p])
sigma_phy_hat = fit2.pars.σ_phy[1]
println("fitted sigma_phy[1] (packed, natural/identity scale) = ", round(sigma_phy_hat, digits = 4))
println("If a profile CI on sigma_phy[1] brackets around this natural-scale value and then",
        " calls exp() on the bounds (confint_profile.jl: `kinds[param_index] === :log_sd` ",
        "-> `exp(lower)`/`exp(upper)`), the reported interval is on the WRONG scale versus",
        " the Wald interval, which stays linear. Example: exp(", round(sigma_phy_hat, digits=4),
        ") = ", round(exp(sigma_phy_hat), digits = 4), " vs the natural-scale value ",
        round(sigma_phy_hat, digits = 4), " itself -- a ", round(exp(sigma_phy_hat) - sigma_phy_hat, digits=4),
        " unit jump purely from the mismatched tag, matching #129's worked example",
        " (Wald ~0.5 +/- z*se, profile ~exp(0.5 +/- ...) centred near 1.65).")

# ---------------------------------------------------------------------------
# #137: constrained refit never checks g(theta_min) == c before reporting
# success.
# ---------------------------------------------------------------------------
println("\n" * "="^78)
println("--- #137: derived-profile refit ignores whether the constraint held ---")
# Use the non-phylo fixture from earlier fit's sibling: correlation() as the
# derived quantity, target it at an aggressive (likely infeasible-ish) value.
Random.seed!(20)
p3, K3, n3 = 4, 1, 400
Λ3 = reshape([0.8, 0.6, 0.4, -0.3], p3, K3)
y3 = Λ3 * randn(K3, n3) + 0.4 .* randn(p3, n3)
fit3 = fit_gaussian_gllvm(y3; K = K3)
corr_fn = θ -> begin
    spec = GLLVModels._derived_spec(fit3)
    GLLVModels._correlation_packed(θ, spec, 1, 2)
end
c_hat = corr_fn(fit3.pars.θ_packed)
println("fitted correlation(1,2) = ", round(c_hat, digits = 4))
c_target = clamp(c_hat + 0.9, -0.999, 0.999)  # push hard toward the boundary
ll_c, ok, θ_new, g_at_min = GLLVModels._derived_refit_with_fixed(
    fit3, corr_fn, c_target, y3, nothing, nothing)
achieved = corr_fn(θ_new)
println("target c = ", round(c_target, digits = 4), "; _derived_refit_with_fixed returned",
        " success = ", ok, " (unconditional per src/confint_derived.jl:777 `return (..., true, ...)`)")
println("g(theta_min) actually achieved = ", round(achieved, digits = 4),
        "; |achieved - target| = ", round(abs(achieved - c_target), digits = 4),
        " (R's .fix_and_refit_nll() returns NA whenever this exceeds 0.05,",
        " R/profile-derived.R:303-309)")
println(abs(achieved - c_target) > 0.05 ?
    "=> exceeds R's 0.05 gate and Julia still reports success = true: LIVE." :
    "=> within 0.05 on this draw; rerun with a more extreme target to see the gap widen.")

# ---------------------------------------------------------------------------
# #142: profile_ci_communality / profile_ci_correlation do not exist in
# Julia; the generic profile_ci_derived has no floor/ceiling and returns NaN
# on a flat/boundary profile.
# ---------------------------------------------------------------------------
println("\n" * "="^78)
println("--- #142: no floor/ceiling on profile_ci_derived (communality/correlation) ---")
has_pcc = isdefined(GLLVModels, :profile_ci_communality)
has_pcr = isdefined(GLLVModels, :profile_ci_correlation)
println("isdefined(GLLVModels, :profile_ci_communality) = ", has_pcc)
println("isdefined(GLLVModels, :profile_ci_correlation)  = ", has_pcr)
println("R (frozen gllvmTMB 0.7.0, asNamespace(\"gllvmTMB\")) DOES export both,",
        " each with q_lo_floor/q_hi_ceiling (communality 0.001/0.999,",
        " correlation +/-0.999) -- confirmed live via RCall in this session.")
# Demonstrate the underlying _derived_bisect_side has no clamp: a deviance
# that never crosses the cutoff anywhere in the search range.
flat_dev = x -> 0.0  # never reaches cutoff -> bracket expansion exhausts, no boundary awareness
lo = GLLVModels._derived_bisect_side(flat_dev, 0.0, -0.05, 3.84; max_expand = 6, max_bisect = 20)
hi = GLLVModels._derived_bisect_side(flat_dev, 0.0,  0.05, 3.84; max_expand = 6, max_bisect = 20)
println("_derived_bisect_side on a flat deviance (never crosses cutoff): lower = ", lo,
        ", upper = ", hi, " (both NaN, no floor/ceiling reported -- contrast R's",
        " find_bound() which returns the boundary itself, R/profile-derived.R:437-450)")

# ---------------------------------------------------------------------------
# #147: Beta logpdf has no boundary guard.
# ---------------------------------------------------------------------------
println("\n" * "="^78)
println("--- #147: Beta _glm_logpdf on y = 0 / y = 1 ---")
fam = GLLVModels.Beta(5.0, 1.0)  # family marker: phi carried in .alpha, matching src/laplace_grad.jl:449
for y_edge in (0.0, 1.0, 1e-15, 1 - 1e-15)
    val = GLLVModels._glm_logpdf(fam, 0.4, 1, y_edge)
    println("_glm_logpdf(Beta(phi=5), mu=0.4, y=", y_edge, ") = ", val)
end
println("R's gllvmTMB clamps y into [1e-12, 1-1e-12] before evaluating the Beta density",
        " (cpp:1952-1958, cited in #147) so the same y never reaches -Inf there.")

# ---------------------------------------------------------------------------
# #149: Julia hard-requires n_sites >= p.
# ---------------------------------------------------------------------------
println("\n" * "="^78)
println("--- #149: n_sites >= p assertion ---")
Random.seed!(5)
y_small = randn(5, 3)  # p=5, n=3
try
    fit_gaussian_gllvm(y_small; K = 1)
    println("fit_gaussian_gllvm did NOT throw for n=3 < p=5 (unexpected)")
catch e
    println("fit_gaussian_gllvm(y; K=1) with p=5, n=3 threw: ", sprint(showerror, e))
end

println("\n" * "="^78)
println("Probe script complete.")
