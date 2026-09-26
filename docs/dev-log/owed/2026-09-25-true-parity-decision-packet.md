# True-parity decision packet (2026-09-25)

Every decision in the true-parity programme that only the maintainer can make, in one place. Each
row has the question, a recommendation, the cost of each answer, and a reply you can paste. Nothing
here is signed yet.

Base: GLLVModels.jl `main` at d9bc77412. Frozen oracle: gllvmTMB 0.7.0 at b4d5fee64. Source of the
gap ids (A-, B-, C-, D-, X-): the read-only gap inventory of 2026-09-24, summarised in the gate-tier
scoreboard (see the scoreboard PR).

## Where parity stands, in one paragraph

True parity is defined by seven clauses (C1 to C7, `docs/dev-log/core070/true-parity-decision-map.md`)
and 32 gate-tier rows (`true-parity-gate-tier-2026-09-05.md`). Today 0 of the 32 rows are promoted.
Three things stop the programme finishing soon, whatever engineering does:

- Real-data workflows (C4) wait on the R bridge, gllvmTMB #1236, and need 18 to 28 agent-days.
- The phylogenetic rows A14 and A15 need a 5 to 25 day build.
- About 20 decisions below are yours.

The engineering that needs no decision is running as separate PRs this session.

## The four answers that unlock the most

| # | Question | Recommendation | Cost | Reply to paste |
|---|---|---|---|---|
| 1 | B-06. Which R version does "true parity" mean? | Frozen 0.7.0 (b4d5fee64), as built. Put 0.7.1 and later (the zero-inflated families, ordinal_logit, the censored_poisson engine, the temporal and slope rows, 20 untracked exports) on a separate catch-up ledger. | Frozen: no rework. Re-freeze at R main: 2 to 5 days plus Totoro, and every receipt re-run. | "B-06: frozen 0.7.0. Open a 0.7.1+ catch-up ledger." |
| 2 | D-01. Revise the frozen contract for the three failing cells NATIVE-06, 10 and 12? | Yes. Diagnosed this session: none of the three is a Julia defect. Each sits at a genuine boundary, and the R side has numerical noise at large dispersion or ν. Proposed v2: NATIVE-10 asserts the boundary flag, `converged == false` (your A6 #11 rule) and a one-sided logLik; NATIVE-12 moves to policy v2 (a Newton polish on R's own gradient, boundary dispersions held fixed); NATIVE-06 uses the finite-dispersion n = 200 data from #478 plus a boundary-agreement check. Stored data with hashes for the whole required cohort. | 1 to 3 agent-days, then a Track A re-run on Totoro (about 1 h, so a pre-run and your ack). | "D-01: approve contract v2 for NATIVE-06/10/12 as proposed." |
| 3 | B-01 and B-02. Sign the ledger dispositions? | Let an agent draft one T9 PR that proposes bind, excluded or needs-surface, with a reason, for each of the 47 open rows. The same PR states that the 122 needs-surface rows are signed compatibility dispositions and lists the Ada defaults for you to ratify or reverse. Evidence of sign-off: a dated maintainer block in the PR body, because agents merge under your account. | 1 to 2 agent-days to draft; about 1 h of your review. | "B-01/B-02: draft the T9 disposition PR. Sign-off evidence is a dated maintainer block in the PR body." |
| 4 | Merge the fixes from this session? | Yes, after you read them: #481 (Gamma mode search, #479) and #483 (Beta convergence, #480). They conflict on CHANGELOG and AGENTS lines, so the second one needs a rebase. The PRs opened later this session are listed in the handover. | Minutes. | "Merge #481, then rebase and merge #483, when green." |

## A new decision from this session's audit

| # | Question | Recommendation | Cost | Reply to paste |
|---|---|---|---|---|
| 5 | #485. Should `converged = true` require a small gradient? Today about 85 fitters pass Optim's flag through, and Optim counts a zero-length step as convergence. Measured: 1 of 3 ZIP fits and 3 of 3 NB1 fits report converged at non-stationary points; one ZIP fit is 14.7 log-likelihood units low. | Yes. Use one rule for every fitter: the gradient test, or the scale-aware test already used by `_tweedie_verdict` and #483. Add a restart when the optimizer stops on x or f alone. Some parity tests assert `converged`, so the PR must first list every flag that changes. | A draft PR is about 1 day. Your review, because reported results change. | "#485: yes, one gradient-checked rule for all fitters; show me the list of flags that flip before merge." |

## Scope decisions (each one sets how much work "true parity" means)

| # | Question | Recommendation | Reply to paste |
|---|---|---|---|
| 6 | B-03. Five second-order holdouts: the GP-1 comparator, Student-t free ν, raw Λ loadings, jointly fitted Tweedie power, BetaBinomial φ pairing. | Keep all five out of C2, and write section 7 of the second-order contract as "C2 is complete when the listed cells pass". GP-1 has no R twin; raw Λ is not identified (the rotation-invariant ΛΛᵀ is compared instead); free ν and joint Tweedie power sit on boundaries where the R side is rough (C-05 below). | "B-03: keep all five OUT; section 7 = the listed cells." |
| 7 | B-04. The grouping levels unit, unit_obs, cluster and cluster2 (C5) cannot "pair": your Q1 ruling closed their numerical gates. | Re-scope C5 to "exists under the same names on both engines". Pairing leaves the claim. Reopening the gates is 3 to 7 days. | "B-04: re-scope C5 to same names; pairing out." |
| 8 | B-05. Phylogenetic rows A14 and A15 need a 5 to 25 day build, and the phylo transport questions Q1 to Q4 still carry agent defaults. | Ratify the Q1 to Q4 defaults, and schedule the build as its own programme after decision 2 lands. | "B-05: ratify Q1-Q4 defaults; schedule A14/A15 as the next build." |
| 9 | B-07. 62 R 0.7.0 exports with no Julia twin (formula keywords, spatial, meta-analysis, iSDM and others). | Record them as signed exclusions: outside true parity, as the 2026-09-05 map already says ("not the destination"). Promote rows later by name if needed. | "B-07: all 62 are signed exclusions for now." |
| 10 | B-08. Six rows where the engines disagree, and three that agree in name only (Student-t ν per trait, ZINB shared r vs per-trait φ, kernel PSD vs PD). | Exclude the six with their fences written down. Student-t follows decision 6. ZINB follows decision 1 (R's zero-inflated families post-date 0.7.0). Kernel: record PD-only as a Julia limit. | "B-08: exclude the six; ZINB follows B-06; kernel PD-only is a recorded limit." |
| 11 | B-09. The ledger's CLOSURE PASS does not test parity: it never gates the R-only, R-NARROWER or DIFFER rows. | Sign a stricter rule: every R-only row has a twin or a written disposition; every DIFFER and R-NARROWER row has a signed fence; one shared status vocabulary. The R lane implements it (C-04). | "B-09: sign the stricter closure rule; R lane implements." |
| 12 | B-10. NB2 on small data has several likelihood maxima; Laplace can rank them wrongly by about 0.3. | Keep the #478 restart only. An AGHQ-backed NB2 fit (3 to 5 days) is not needed for parity. | "B-10: restart only." |
| 13 | B-11. No fallback if gllvmTMB #1283 (the frozen recorder fix) never lands. | Assign #1283 to the R lane with a date. If it has not landed by then, use the environment-level deviation (attach testthat through R_DEFAULT_PACKAGES, and isolate the embedded Julia), about 0.5 day. | "B-11: R lane owns #1283; deviation route after <date>." |
| 14 | B-12. The default `latent()` (with Ψ) silently loses Ψ when sent to Julia. | Refuse it through the Julia engine with a named alternative (`latent(unique = FALSE)`) until Julia supports Ψ. A silent model change is worse than a refusal. | "B-12: refuse with a named route until Julia has Ψ." |
| 15 | B-13. R-lane naming choices that change parameter maps (#1080 dispersion names, #897 and #1097 ordinal degeneracy detector). | Keep today's names and document the mapping; renaming now churns every receipt. | "B-13: keep names, document the map." |
| 16 | X-03. Which build of the frozen oracle is the authority when two builds disagree? | The build pinned by `GLLVM_PARITY_R_LIBS` on Totoro, with Julia 1.10 (the reference platform decided on 2026-09-24). Receipts without a recorded build are provenance-unknown (the R-library audit PR lists them). | "X-03: Totoro pinned build is the authority." |

## Writes to gllvmTMB (gated: this lane does not write there without your yes)

| # | Issue to file on gllvmTMB | Evidence | Reply to paste |
|---|---|---|---|
| 17 | C-05. The Student-t df profile interval reports df minus 1 (it back-transforms log(df - 1) with plain exp). Still on R main. | profile-targets code path; every ν interval compared against 0.7.0 inherits it. | "File C-05 on gllvmTMB." |
| 18 | C-06. The frozen oracle stops below Julia's NB2 optimum on 6 of 16 datasets, by 0.29 to 1.15 log-likelihood units. | `docs/dev-log/core070/nb2-boundary-screen-20260924/`, #477. | "File C-06 on gllvmTMB." |
| 19 | R-side roughness at large dispersion or ν (lgamma differences) makes R gradient gates machine-dependent (NATIVE-06, 10, 12). | This session's holdout diagnoses; decision 2. | "File the R precision issue on gllvmTMB." |

## Compute campaigns (each over 30 minutes, so each needs a pre-run and your ack)

| # | Campaign | Estimate | Reply to paste |
|---|---|---|---|
| 20 | D-02. Screen the Tweedie grouped fitter for the NB2, Gamma and Beta bug classes. | 4 to 73 min per fit on the Mac; 10 datasets on Totoro at 4 cores each is about 1 to 3 h wall. A pre-run of 2 datasets comes first. | "D-02: run the pre-run, then the screen if it matches." |
| 21 | D-01 re-run. Track A on Totoro after the contract revision. | About 1 h. | Covered by decision 2's reply. |
| 22 | D-03. The D3 loading_profile Stage 1 heavy grid. | Over 30 min; not estimated yet, and blocked on A-06. | "D-03: later, after A-06." |
| 23 | D-04. The full realistic-size grid for Binomial and Beta (p up to 50, n up to 2000). | 30 to 110 min per cell. C3 only needs one cell per family, which this session runs. | "D-04: not now." |

## New items (2026-09-26 update): PR #495's 11 NEEDS-DECISION issues, plus the live divergences #129 and #131

Added from `itchyshin/GLLVModels.jl` PR #495 (cross-audit issue triage, 2026-09-25), which triaged
the 2026-07-02 cross-audit set against `origin/main` @ `d9bc77412`. Two groups below: the 11 issues
PR #495 itself could not resolve without a maintainer decision or more read time, and the 2 live
divergences (#129, #131) that gate-tier extractors (Wald/profile CIs, communality/correlation)
depend on and that PR #495 reproduced with numbers but left open.

### PR #495's 11 NEEDS-DECISION issues

| # | Question | Recommendation | Reply to paste |
|---|---|---|---|
| 24 | #133. Ordinal families: Julia's cutpoints are shared across species with no per-trait intercept (`src/families/ordinal.jl:1-9`, confirmed deliberate design, unchanged vs R's per-trait intercept + per-trait cutpoints). Add per-trait intercepts/cutpoints to Julia, or keep the restriction and document it? | Document the restriction rather than build per-trait ordinal cutpoints now: it is a real capability gap, not a bug, and adding per-trait intercepts to a shared-cutpoint model is a design change with its own identifiability questions, not a quick fix. | "#133: document the shared-cutpoint restriction; no Julia change now." |
| 25 | #136. Phylo-unique sigma_phy uses a signed identity link in Julia's likelihood but a positive log link in R's C++ (`likelihood.jl:379-388`, confirmed deliberate via the sign-flip-restart/sign-anchor machinery). Pick one convention, or document the two engines reach different phylo-unique models? | Document the divergence rather than unify the link: the sign-flip-restart machinery is load-bearing for Julia's own optimizer, and switching to R's positive log link is a numerics change to a working phylo fitter, not a documentation fix. | "#136: document the link-convention divergence; no engine change now." |
| 26 | #139. `bridge_fit`'s `d::Integer = 1` (`bridge.jl:488`) vs R doubles; PR #495 did not verify whether an upstream `Int(d)` conversion actually intercepts non-integral R-side doubles before this signature is hit. | Trace the R-to-Julia call path for `d` before deciding anything: if the conversion already guards it, this is a non-issue; if not, the fix is a type relaxation or an explicit round/error, and which one depends on what R actually sends. | "#139: trace the R->Julia `d` call path first; decide the fix after." |
| 27 | #143. The transformed-Wald covariance's Gaussian-record branch (`confint_derived_wald.jl:148-155`) checks only `all(isfinite, covariance)`, no visible eigenvalue/PD check; PR #495 did not confirm the general (non-record) branch. | Read the full `_tw_sigma_from_hessian` function before deciding whether this is a gap: if a PD check exists elsewhere in the function it may already be covered; if not, an eigenvalue floor is the standard fix for a covariance built from a possibly-indefinite Hessian. | "#143: read the full `_tw_sigma_from_hessian` path; decide after." |
| 28 | #146. The SQUAREM premature-stop fallback (`em_squarem.jl:318-333`) already compares `fb.logLik > ll_sq` and returns the better point, which looks improved since filing; PR #495 did not trace every return path in the time available. | Trace the remaining return paths to confirm no path still returns the known-suboptimal point without the comparison; this reads as likely already fixed, so the remaining work is verification, not a new patch. | "#146: trace remaining return paths to confirm the fix is complete." |
| 29 | #150. A `size(Λ_phy,1)==p` guard exists nearby (`likelihood.jl:143-144`) but the issue is specifically about `Λ_B`; PR #495 did not confirm whether `Λ_B` has an equivalent guard. | Trace whether `Λ_B`'s size is validated before the `@inbounds` phylo loop; if it is unguarded, an out-of-bounds access under `@inbounds` is silent memory corruption, not a bounds error, so this is worth the follow-up read even though it is unconfirmed. | "#150: trace whether `Λ_B` has an equivalent size guard before the `@inbounds` loop." |
| 30 | #152. `_fitted_mean` silently zeros when X is omitted (`postfit.jl:48-60`); not independently verified in PR #495's pass. | Read `postfit.jl:48-60` directly against the issue's claim before deciding; a silent zero (vs. an error) on a missing design matrix is the kind of thing that should either be intentional and documented, or raise. | "#152: read `postfit.jl:48-60` against the issue; decide silent-zero vs. error after." |
| 31 | #153. Binary-tree validation is a global node count (issue's claim); not independently verified, but PR #495 found unrelated `correlation=true` ultrametric-check code nearby that looked more mature than the issue's July filing-time snapshot, suggesting `sparse_phy.jl` may have moved on. | Re-read the topology validation path specifically (not the nearby ultrametric check) before deciding; the code may already have superseded the issue. | "#153: re-read the topology validation path in `sparse_phy.jl`; may already be moot." |
| 32 | #156. `bootstrap_ci` reports log-scale SD under raw-scale names (issue's claim); not independently verified. | Read `confint_bootstrap.jl` naming vs. `kinds` directly; a naming/scale mismatch here would silently mislabel a CI's scale, which is worth confirming even though PR #495 could not fit it into its time budget. | "#156: read `confint_bootstrap.jl` naming vs. `kinds`; decide after." |
| 33 | #158. The homogeneous no-loadings branch's docstring claims it is numerically identical to the dense path; not independently verified. | Run the numerical comparison the issue calls for (homogeneous fast path vs. dense path on a case where the claim would be tested) before deciding whether the docstring is accurate. | "#158: run the numerical homogeneous-vs-dense comparison; decide after." |
| 34 | #161. `em_fa`'s init comment claims something the M-step's code may not enforce (comment unchanged at `em_fa.jl:44-45`); not independently verified. | Confirm whether the M-step is genuinely unconstrained; if so this is doc-only (fix the comment), but if the claimed constraint is load-bearing elsewhere, it is a correctness issue and needs the fuller read first. | "#161: confirm M-step constraint status; doc fix or correctness fix follows." |

### Live divergences #129 and #131 (gate-tier extractors depend on these)

| # | Question | Recommendation | Reply to paste |
|---|---|---|---|
| 35 | #129. Julia's own Wald and profile CI code tag the SAME packed `sigma_phy` term with different scale kinds -- `:linear` in `confint.jl:103-105` (Wald) vs `:log_sd` in `confint_profile.jl:119-120` (profile), so profile_ci applies `exp()` to an already-natural-scale value. This is an internal Julia inconsistency, not just a Julia-vs-R mismatch, and confirmed still live on `origin/main` @ `d9bc77412` (reproduced numerically in PR #495: fitted `sigma_phy[1] = 0.0079`, profile path would `exp()` it to `1.0079`). R's own convention (log-SD + `exp()` in both paths) is internally consistent. | Tag `sigma_phy` `:linear` in `_profile_all_term_names` too (drop the `exp()`), matching what `confint.jl` and `_derived_unpack` already assume; this is the smaller, more surgical fix of the two options in the issue, since #92 already confirmed the packed value is natural-scale. Any gate-tier row that reports a phylo-unique profile CI is wrong until this lands. | "#129: fix `confint_profile.jl` to tag `sigma_phy` `:linear`, matching the Wald path and the natural-scale packing." |
| 36 | #131. `communality()`/`correlation()` (`confint_derived.jl:208`, `:280`) divide by the FULL per-site `Sigma_y_site` (including `sigma_eps^2` and, when active, the W-tier), while R's `extract_communality()`/`profile_ci_communality()` divide by a tier-LOCAL total instead. Confirmed still live and reproduced numerically in PR #495: every trait's Julia communality is pulled down by at least `sigma_eps^2 = 0.16` relative to what a tier-local denominator would give, and a W-tier trait by 1.2547. Related to #142 (Julia has no `level=` tier-local machinery yet). | This needs a maintainer decision on which "total variance" convention true parity means, because it is not obviously a bug in either engine -- both conventions are defensible factor-analytic choices. Recommend picking R's tier-local convention as the target (it is what `extract_communality`'s `level=` argument already documents) and scoping the Julia fix together with #142's tier-local machinery, since a tier-local `communality()`/`correlation()` needs the same per-tier `Sigma` #142 would add. | "#131: target R's tier-local total-variance convention; scope the Julia fix together with #142's tier-local machinery." |

## Final sign-off (B-14)

Only after the rows above are closed or signed: a Rose audit panel, your merge of the T9 scoreboard
PR, the joint decision note covering C1 to C7 and the 32 rows, and the Project.toml version decision.
Project.toml stays at 0.3.0 until then.
