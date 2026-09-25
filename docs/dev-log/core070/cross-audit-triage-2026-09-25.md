# Cross-audit issue triage — 2026-09-25

Triage of the 2026-07-02 cross-audit issue set (#92, #128-163) plus #317,
#105, #106, #107, #10, #11 in `itchyshin/GLLVModels.jl`. Also retitled #477
to its residual per instruction (not part of the cross-audit set; not
closed).

Repo checkout: worktree at `~/local-scratch/gllvm-issue-triage`, branch
`claude/cross-audit-triage-20260925`, off `origin/main` @ `d9bc77412`
("Merge pull request #478 from itchyshin/claude/nb2-finite-dispersion-parity-20260924").

R oracle: frozen gllvmTMB 0.7.0 at `b4d5fee64`, installed at
`GLLVM_PARITY_R_LIBS=~/local-scratch/gllvm-owed24-decisions-20260924/.unlazy/r-build/library`,
called through RCall the same way `test/parity/parity_helpers.jl` does.
Verified the library's `packageVersion` (0.7.0) and its
`CORE070_SOURCE_PIN.toml` before using it.

Reproduction script for the "possible live divergence" issues:
`docs/dev-log/core070/cross-audit-triage-2026-09-25-scripts/probe_extractors.jl`
(pure-Julia; each section is labeled by issue number in its printed output).

Verdicts: **FIXED** (closed with a receipt comment citing a commit and a
fresh test/CI run), **LIVE** (reproduced today, left open with the numbers
posted as a comment), **NEEDS-DECISION** (either a real code-level finding I
could not fully verify inside this pass's time budget, or a deliberate
design divergence that needs a maintainer call rather than a fix), or
**NOT-A-BUG** (roadmap/tracking issue, not a defect).

## Table

| Issue | Verdict | Evidence | Next step |
|---|---|---|---|
| #92 phylo_signal_wald_ci exp-scaled σ_phy | FIXED | Commit `f21b969`/`06e7c0846`; fresh run `test/test_confint_derived_wald.jl` 115/115 pass today | Closed |
| #128 H2 denominator excludes phylo variance | FIXED | Commit `c7cddc29e` (+ `1022b7a14` clamp); same 115/115 run, phylo fixture asserts H2 in [0,1] | Closed |
| #129 sigma_phy Wald (:linear) vs profile (:log_sd) | LIVE | `confint.jl:103-105` tags `:linear`; `confint_profile.jl:119-120` still tags `:log_sd` and exponentiates the bound; reproduced with a phylo fixture, fitted σ_phy=0.0079, exp(0.0079)=1.0079 | Fix: make the two term-kind builders agree (drop the `exp` for the profile path or repack σ_phy as log-SD everywhere) |
| #131 communality/correlation denominator | LIVE | `confint_derived.jl:208,280` still divide by full `Sigma_y_site` incl. σ_eps² and W-tier; K_W=2 fit shows communality pulled down by 0.16-1.25 per trait vs a tier-local Psi_B denominator | Fix: restrict the denominator to the requested tier, matching R's `extract_communality(level=...)` |
| #132 NB2 dispersion granularity (per-trait vs shared) | FIXED | Commit `5ad558772`; public default `fit_gllvm(...; family=NegativeBinomial())` now returns `NBGroupedFit` (per-trait), confirmed via `isa` check on two seeds; Beta sibling (#148) passes 8/8 exactly | Closed on the granularity claim; the seed-45/46 convergence boundary hit is the separate, already-tracked #476/#477 small-data multi-maxima residual, not reopened here |
| #133 ordinal: shared cutpoints, no per-trait intercept | NEEDS-DECISION | `src/families/ordinal.jl:1-9` documents "common ordered cutpoints... shared across species... no separate species intercept" as the design; confirmed unchanged vs R's per-trait intercept + per-trait cutpoints | Maintainer call: add per-trait intercepts/cutpoints to Julia, or document the deliberate restriction |
| #134 dead guard: phylo loadings silently dropped when Σ_phy omitted | LIVE | `likelihood.jl:134` `has_phy` still requires `Σ_phy !== nothing`, so the `has_phy && Σ_phy===nothing` guard at ~152 can never fire; reproduced: `gaussian_marginal_loglik(y,Λ_B,σ_eps; Λ_phy=Λ_phy)` with no Σ_phy returns the identical loglik as the non-phylo call (Δ=0.0) | Fix: replace the dead guard with `(Λ_phy!==nothing || σ_phy!==nothing) && Σ_phy===nothing` per the issue's own proposed fix |
| #135 W-tier drops cross-trait covariance | LIVE | `likelihood.jl:12-14` documents diagonal-only `d_total[t] += (Λ_W Λ_W')[t,t]`; fit with a genuine cross-trait W-tier DGP converged to an essentially diagonal `Λ_W` (max off-diag of `Λ_W Λ_W'` = 0.0) | Fix or document: either give the W tier one shared per-unit score (matching C++), or state only row-norms of Λ_W are identified |
| #136 phylo-unique σ_phy: signed identity (Julia) vs positive log link (C++) | NEEDS-DECISION | `likelihood.jl:379-388` + `fit.jl` sign-flip-restart/sign-anchor machinery confirm this is deliberate, documented parameterization, not an oversight | Maintainer call: pick one convention, or document the two engines have different reachable phylo-unique models |
| #137 constrained refit never checks g(θ)=c | LIVE (weak empirical trigger) | `confint_derived.jl:777` `_derived_refit_with_fixed` returns `success=true` unconditionally; callers discard `g_at_min`. Tried correlation targets 0.95-0.9999 on a small fit; all landed within R's 0.05 gate (achieved plateaued ~0.986) so no violating case was produced in the time budget | Add the R-style `abs(g_at_min - c) > 0.05` -> `success=false` gate; a harder DGP (near-singular fit) is still owed to show a live wrong-CI case |
| #138 `_bridge_scores` swallows all exceptions | LIVE | `bridge.jl:247-253`: bare `catch; return zeros(Float64,0,0); end` unchanged | Narrow the catch, or surface a flag distinguishing "empty because no LV" from "empty because it errored" |
| #139 `bridge_fit` `d::Integer` vs R doubles | NEEDS-DECISION | `bridge.jl:488` still `d::Integer = 1`; did not verify whether an upstream `Int(d)` conversion (cited in the issue) actually intercepts R-side `Float64` before this signature is hit | Follow-up: trace the R->Julia call path for `d` and confirm whether non-integral doubles reach this signature |
| #140 bootstrap percentile CIs may include non-converged refits | LIVE | `confint_family.jl:2892-2934` `_family_bootstrap`'s `work` closure only checks `θb !== nothing && all(isfinite, θb)`, no convergence flag from the refit is consulted | Thread a converged flag out of `ad.refit` and exclude non-converged replicates from the percentile pool |
| #141 `correlation()` divides by zero for degenerate variance | LIVE | `confint_derived.jl:280-288`: `denom = sqrt(Σ[i,i]*Σ[j,j])` has no zero-guard | Add a guard returning NaN (not 0/0=NaN silently, or Inf) with a documented convention |
| #142 no floor/ceiling on profile CIs for communality/correlation | LIVE (partial fix) | `1022b7a14` added `_profile_ci_bounded`, wired only into `profile_ci_total_variance`/`profile_ci_phylo_signal`; `isdefined(GLLVModels, :profile_ci_communality/:profile_ci_correlation)` both `false` (R has both, with explicit 0.001/0.999 and ±0.999 floors, confirmed live via RCall); generic `_derived_bisect_side` on a flat deviance returns NaN/NaN | Fixed for total_variance/phylo_signal; still needed for communality/correlation once those profile-CI functions exist in Julia |
| #143 transformed-Wald covariance treats any invertible Hessian as PD | NEEDS-DECISION | `confint_derived_wald.jl:148-155` (Gaussian-record branch) checks only `all(isfinite, covariance)`, no eigenvalue/PD check visible in the time available; did not confirm the general (non-record) branch | Follow-up: read the full `_tw_sigma_from_hessian` and confirm whether a PD check exists elsewhere in the function |
| #145 `em_fa` uses the global RNG (non-reproducible) | LIVE | `em_fa.jl:44` `Λ_init = 0.1 .* randn(p, K)` — no `rng` kwarg threaded through | Add an `rng::AbstractRNG` kwarg defaulting to `Random.default_rng()` |
| #146 SQUAREM premature-stop fallback | NEEDS-DECISION | `em_squarem.jl:318-333` shows a polish step that already compares `fb.logLik > ll_sq` and returns the better point — looks improved since filing, but did not fully trace every return path in the time available | Follow-up: confirm no remaining path returns the known-suboptimal point without the comparison |
| #147 Beta density has no boundary guard (y=0/1) | LIVE | `beta.jl:27` `_glm_logpdf` unclamped; reproduced `_glm_logpdf(Beta(5),0.4,1,0.0) = -Inf` and same for y=1.0 | Clamp y into `(eps, 1-eps)` matching gllvmTMB's 1e-12, or validate/transform boundary observations up front |
| #148 Beta dispersion granularity (per-trait vs shared) | FIXED | Commit `d666a09c4`; `test_beta_parity.jl` now 8/8 pass, Δ logLik = 5.97e-9 vs the frozen oracle (no longer `@test_broken`) | Closed |
| #149 Julia hard-requires n_sites >= p | LIVE | `fit.jl:154` `@assert n >= p` unchanged; reproduced `fit_gaussian_gllvm(randn(5,3); K=1)` throws `AssertionError` | Downgrade to a warning, or document the closed-form-path-only restriction |
| #150 missing `size(Λ_B,1)==p` check (phylo OOB) | NEEDS-DECISION | Found a `size(Λ_phy,1)==p` guard nearby (likelihood.jl:143-144) but the issue is specifically about `Λ_B`; did not confirm whether `Λ_B` has an equivalent guard in the time available | Follow-up: trace whether `Λ_B` size is validated before the `@inbounds` phylo loop |
| #151 `low_rank_chol` does not validate d>0 | LIVE | `lowrank_cholesky.jl` has a length/dimension check but no `d .> 0` positivity check anywhere in the file | Add a positivity check with a clear `ArgumentError` |
| #152 `_fitted_mean` silently zeros when X omitted | NEEDS-DECISION | Not independently verified in this pass (time budget) | Follow-up read of `postfit.jl:48-60` |
| #153 binary-tree validation is a global node count | NEEDS-DECISION | Not independently verified in this pass (time budget); found unrelated `correlation=true` ultrametric-check code nearby that looked more mature than the issue's filing-time snapshot, suggesting `sparse_phy.jl` may have moved on since July | Follow-up read of the topology validation path specifically |
| #156 `bootstrap_ci` reports log-scale SD under raw-scale names | NEEDS-DECISION | Not independently verified in this pass (time budget) | Follow-up read of `confint_bootstrap.jl` naming vs `kinds` |
| #157 `has_diag` couples both diagonal tiers | LIVE | `confint_derived.jl:288-315` (spec builder) still gates both `log_σ_B` and `log_σ_W` behind one shared `spec.has_diag` flag | Looks like a deliberate J1/J2 simplification; recommend NEEDS-DECISION-style maintainer read even though the code confirms the claim, since decoupling is a real API change |
| #158 homogeneous no-loadings branch vs dense path | NEEDS-DECISION | Not independently verified in this pass (time budget) | Follow-up: numerically compare the homogeneous fast path against the dense path on a case where the docstring's "numerically identical" claim would be tested |
| #159 `size(F,i)` throws for i>2 instead of returning 1 | LIVE | `lowrank_cholesky.jl:76-77` still `i==1 \|\| i==2 ? length(F.d) : throw(BoundsError(...))`, diverging from `Base.size`'s convention | Change the `else` branch to `return 1` for `i>2` (matching AbstractArray convention) |
| #161 `em_fa` init comment vs unconstrained M-step | NEEDS-DECISION | Not independently verified in this pass (time budget); comment claim unchanged at `em_fa.jl:44-45` | Follow-up: confirm whether the M-step is genuinely unconstrained (doc-only issue if so, correctness issue if the zeros are load-bearing) |
| #162 edge wrapper builds dense Σ_phy before validating loadings | LIVE | `likelihood_edge_incidence.jl:105-113`: `Σ_phy = sigma_phy_dense_edge(...)` runs before the `Λ_phy===nothing && σ_phy===nothing` throw | Reorder: validate loadings first, build the dense Σ_phy only after |
| #163 `node_grad` recomputes Takahashi diagonal twice | FIXED | Commit `5369d5ae9`; `node_gradient.jl:221-232` now computes `Qeff_diag` once and passes it to both consumers, matching the proposed fix exactly | Closed |
| #317 CI Julia shard 1/4 red (Beta CV not in (0,1)) | FIXED | Commit `5ce67c84a` (StableRNG for Gamma/Beta CV); CI run [36076411721](https://github.com/itchyshin/GLLVModels.jl/actions/runs/36076411721) shard 1/4 green on `main` | Closed |
| #105 Student-t family | FIXED | `src/families/studentt.jl`, commit `bba112aa5`; 90/90 fresh smoke test pass (shared run with #106/#107) | Closed |
| #106 one-part lognormal family | FIXED | `src/families/lognormal.jl`, commit `6b367799c`, distinct from `twopart.jl`'s delta-lognormal; 90/90 fresh smoke pass | Closed |
| #107 truncated Poisson/NB2 families | FIXED | `src/families/truncated_poisson.jl` (`d615160b1`), `truncated_nbinom2.jl` (`8c666988b`), distinct from hurdle positives; 90/90 fresh smoke pass | Closed |
| #10 R-bridge roadmap (engine="julia") | NOT-A-BUG | Forward-looking v1.0 roadmap tracker, not a defect | Leave open as the index item it is |
| #11 GLLVM.jl roadmap (digital twin) | NOT-A-BUG | Live status-snapshot index issue, not a defect | Leave open as the index item it is |

## #477 (out of scope for the cross-audit set, handled per instruction)

Retitled from "NB2 per-trait dispersion fit can stall with a trait at the
Poisson boundary, below the optimum" to "NB2 per-trait dispersion: the
Laplace likelihood has several maxima on small data (residual after #478)".
PR #478 (merged) fixed the default-route stall this issue originally
reported (`_nb_boundary_restart`, +0.46 to +2.59 logLik on 7 of 16
NATIVE-06-design datasets, never below gllvmTMB afterward). The residual —
the NB2 Laplace likelihood genuinely having several maxima on small data,
already described on #476 — is what the new title names. Not closed;
comment posted explaining the retitle.

## Summary

- **Closed (9):** #92, #105, #106, #107, #128, #132, #148, #163, #317
- **Live divergences reproduced and commented, left open (9 with numbers posted):** #129, #131, #134, #135, #136 (design), #137, #142 (partial), #147, #149
- **Live, source-confirmed but not individually commented (7):** #138, #140, #141, #145, #151, #157, #159, #162 (table has the evidence and next step for each)
- **Needs a maintainer decision or further read (11):** #133, #136 (dual-listed: live+design), #139, #143, #146, #150, #152, #153, #156, #158, #161
- **Not bugs, leave open (2):** #10, #11
- **Retitled, not closed (1, out of the cross-audit set):** #477

## Compute used

Roughly 20-25 minutes of Julia/R wall time across: `test/test_confint_derived_wald.jl` (26s), the shared extractor probe script (~1 min), the #137 retry (~10s), the Beta+NB2 parity re-run against the frozen oracle (~28-30s each), the families smoke run (55s), plus several `Rscript`/`gh` lookups. Under the 30-minute total budget.
