# bootstrap_Sigma checkpoint ledger (2026-11-01)

Lane: claude:bootstrap-sigma-p1 · branch codex/true-parity-p1-restored-20261101 · base 9974d0bb5 · origin/main aed938e1f

## Decisions of record
- Signature evidence for the estimand: Shinichi's chat words on 2026-11-01, "Sign, wait for #1036". I read this as signing the estimand and sequencing `src/confint_derived_wald.jl` behind PR #1036. This is chat evidence only. The plan packet header (`plans/2026-10-28-p1-bootstrap-sigma-estimand.md`) still reads PROPOSED / UNSIGNED, the gates below are unsigned drafts, and the Spec is a candidate. Nothing here has been countersigned in a repo file.
- The handover of 2026-11-01 approved the "full pinned-R behavior target"; that is not an independent sign-off of the narrowed v1 scope.
- The row stays required-core, REGISTRATION-ONLY and unbound. No row status, score, receipt, scope or tolerance changes in this checkpoint.
- Pinned R oracle: gllvmTMB 9539352f66f2db2cc26b1c393e67212a359b60c9, read-only.

## Lease
Lease as reported by `lane_lease.sh` output on 2026-11-01: lane `claude:bootstrap-sigma-p1`, 8 h, exact paths. I did not re-run the script to re-check it, and this note is not itself a lease claim. Paths: `src/bootstrap_sigma_r.jl`, `test/test_bootstrap_sigma_r.jl`, this ledger, `docs/dev-log/check-log.d/2026-11-01-bootstrap-sigma-impl.md`, `docs/dev-log/after-task/2026-11-01-bootstrap-sigma-impl.md`.
Deferred until PR #1036 is merged or sequenced: `src/confint_derived_wald.jl`, `src/GLLVModels.jl`, `test/runtests.jl`, `test/test_se_machinery.jl`, docs/CHANGELOG.

## Known gaps
- GATES.md for this estimand (was untracked under .unlazy/) is missing from the restored worktree; G1-G4 and G6 must be rebuilt from the plan packet and re-confirmed.
- No simulate-and-refit route was verified for TwoLevelFit, PrecisionMultivariateFit or JointPhyloGroupedGaussianFit. v1 scope: GllvmFit; TwoLevelFit stretch; all else refused.
- GllvmFit `extract_Sigma(:unit_obs)` and `extract_ICC_site` include sigma_eps^2, unlike R's tiers. The driver never calls them. For GllvmFit the W tier and ICC are not reportable in v1: they are DROPPED with an `@info`, and the call throws only when nothing reportable remains (Spec 3.2, 3.4, 3.5; see G6.a resolution below).
- R `link_residual` is a no-op for Gaussian; no mapping to Julia `residual_mode` exists.
- Any experiment over 3 h needs a plan, pre-run and approval (D-287).

## Gates draft

> DRAFT-NOT-SIGNED. Rebuilt by a drafter from the plan packet because the original GATES.md is lost; the G-number mapping is a RECONSTRUCTION (UNVERIFIED). Where this draft disagrees with the Spec below, the Spec wins (see 'Draft vs Spec disagreements'). Verbatim draft follows, headings demoted one level.

**G6.a resolution (applies to the verbatim draft below; gates remain UNSIGNED).** The G6.a text below is superseded by the Spec on two points. (1) GllvmFit with `has_diag` and `K_W == 0` is SUPPORTED, not refused: tier B is the identified total `sigma_y_site(fit)`. (2) GllvmFit W tier, `:unit_obs`, `:phy` and ICC are NOT refused: they are dropped with an `@info`, and `ArgumentError` is thrown only when no requested tier or no reportable summary remains. Genuine refusals remain: `K_W > 0`, retained Gaussian record, pins, phylo block, `PrecisionMultivariateFit`, `JointPhyloGroupedGaussianFit`, non-Gaussian fits, `n_cores > 1`. All of this needs Shinichi's confirmation when the gates are signed.

### bootstrap_Sigma GATES — DRAFT-NOT-SIGNED

Status: DRAFT-NOT-SIGNED. Rebuilt from the plan packet (docs/dev-log/plans/2026-10-28-p1-bootstrap-sigma-estimand.md, items 1-5), the after-task note of the same date, the checkpoint ledger, and the read-only prep JSON (result.design.plan, blockers). The original GATES.md (.unlazy/.../bootstrap-sigma-estimand/GATES.md) is lost. The G-number to content mapping below is a RECONSTRUCTION (UNVERIFIED): the packet lists five controls; I assigned G1-G4 to items 1-4 and G6 to item 5 (public-door/negative-control receipts). G5 is not reconstructed (no source for it). Shinichi must confirm or relabel.

Oracle: pinned R gllvmTMB 9539352f66f2db2cc26b1c393e67212a359b60c9, R/bootstrap-sigma.R (sha256 c30fbad6...4238), R/extract-sigma.R (sha256 dbd33782...4a14). Read-only.
Scope v1: GllvmFit supported; TwoLevelFit stretch; all else refused with explicit ArgumentError. No row status / receipt / scoreboard / scope / tolerance change. Row stays required-core, REGISTRATION-ONLY, unbound.
Env for every Julia check: ~/.juliaup/bin/julialauncher --project=. , JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1, tiny fixtures, <5 min each. Test file: test/test_bootstrap_sigma_r.jl (leased path; appended to runtests.jl only after PR #1036 sequencing).

#### G1 — Discriminating fixture (wrong-tier and residual contamination are observable)
Executable checks:
- G1.a Fixture A (GllvmFit): Gaussian p=4, K=1, n=150, sigma_eps=0.5, NO K_W, NO diag. Assert sigma_eps^2 > 0 and Lambda_B Lambda_B' off-diagonals nonzero. Independent expected value: diag(B_tier) = diag(Lambda Lambda'); diag(sigma_y_site) - diag(B_tier) == sigma_eps^2 (absolute value, computed independently of the implementation, not only a difference).
- G1.b Fixture B (TwoLevelFit): B and W both identified, Sigma_B != Sigma_W (max abs elementwise gap > 0.1).
- G1.c Two-fixture split is required because GllvmFit cannot identify B vs W when K_W>0 or has_diag (extractors.jl:116-121). NEEDS MAINTAINER APPROVAL (blocker 6).
- G1.d phy fixture: NOT defined; phy route is refused in v1 (open question 1).
Pass: all asserts true; fixture seeds recorded in the test file.

#### G2 — Tier-specific point estimates and table fields vs direct extraction
Executable checks (point estimate only, no bootstrap needed):
- G2.a GllvmFit, tier B: point_est[Sigma_B] == Lambda Lambda' (+ diag sigma2_B only if present) via a private _r_tier_total; NOT extract_Sigma(:unit_obs), NOT sigma_y_site. Control: differs from sigma_y_site diag by exactly sigma_eps^2.
- G2.b R, communality, cross_corr: equal direct extract_correlations / extract_communality on the same fit. CAVEAT (critique, OWED): extract_communality for has_diag fits uses a sigma_y_site denominator; for G2 compute the expected value independently from R's B-only definition, otherwise the test passes trivially.
- G2.c Aliases: level in (:unit, :B) and (:unit_obs, :W) dedupe to internal B / W; stored level is B/W/phy, never "unit"/"unit_obs".
- G2.d TwoLevelFit (stretch): B and W each equal the TwoLevelFit extractors; ICC = B/(B+W) diag, per twolevel.jl:260-263.
- G2.e Output struct fields: point_est, ci_lower, ci_upper, n_effective, ci_method="percentile", link_residual, conf, n_boot, coverage_ceiling, n_failed, level, what, draws (nothing unless keep_draws). Defaults: what = Sigma,R,communality,ICC,cross_corr; ICC dropped with message unless both B and W present; absent tiers dropped silently; error if no requested tier present; error if conf outside (0,1).
- G2.f link_residual in (:auto,:none) accepted and identical for Gaussian (R no-op, R/extract-sigma.R:613-617).
Pass: tolerance 1e-10 vs independently computed expectation; no tolerance widening.

#### G3 — Refit-count and seed contract
Executable checks (wrap the internal _refit seam with a counter; do not make the seam public):
- G3.a Exactly n_boot refits for any p and any `what` (e.g. n_boot=40, p=4, all summaries => 40). Negative control: legacy design = p(p+1)/2 * n_boot (=400).
- G3.b One shared simulated response batch: record the response passed to _refit; replicate b data is the same object for every summary.
- G3.c Same seed -> bit-identical output; different seed -> different output. Seed convention (single stream vs seed+b) is UNDECIDED (open question 4); gate fixes whichever Shinichi picks.
- G3.d Failed-refit accounting: cripple every k-th refit; n_failed == injected count; bounds use survivors only; attrition warning fires when any summary keeps <80% of replicates (R:465-486). Control: uncrippled run n_failed == 0.
- G3.e Percentile correctness: inject deterministic _refit with known sequence; bounds equal quantile(., alpha/2) and quantile(., 1-alpha/2); catches wrong alpha or swapped tails.
- G3.f Warnings: n_boot=10 at conf 0.95 warns (min_boot=39) and does not throw; n_boot<999 warns; coverage_ceiling == (n_boot-1)/(n_boot+1). (R roxygen says "refused" but R code only warns; copy the code, R:252-277.)
- G3.g keep_draws: nothing by default; n_boot entries when true.
Pass: all asserts; default n_boot (999 vs 500) UNDECIDED, tests pass n_boot explicitly.

#### G4 — Runtime estimate and bounded pre-run (D-287)
Executable checks:
- G4.a Before any bootstrap run, write estimate in the ledger. Currently NOTHING is measured (all figures inferred): one 4x150 refit tens to few hundred ms; n_boot=40 seconds; new test file ~1-3 min; three mutation runs ~3-9 min. All UNVERIFIED.
- G4.b Time the new test file once on a quiet machine; record wall time; any single step >5 min stops and re-reports.
- G4.c Hard line: any run estimated >3 h (e.g. p=80, K=2, 999 reps ~37-44 h from check-log.md:13609-13613) requires a simulation plan, a pre-run with results shown, and Shinichi's approval. No spider/beetle refits in this checkpoint.
Pass: estimate stated before run; actual <= estimate, else stop and re-report.

#### G6 — Public-door and negative-control receipts (independent; row stays unbound)
Executable checks:
- G6.a Must-refuse matrix throws ArgumentError with a clear message, each paired with a supported-case control that does not throw: GllvmFit W; GllvmFit ICC; GllvmFit phy; GllvmFit has_diag & K_W=0 (explicit guard BEFORE extraction, since _r_tier_total returns sigma_y_site there, extractors.jl:272-278); PrecisionMultivariateFit and JointPhyloGroupedGaussianFit (any tier); TwoLevelFit phy.
- G6.b Pinned fit (lambda_constraint) refused through bootstrap_Sigma via _check_unpinned_refit (confint.jl:552-561); currently untested for bootstrap_Sigma.
- G6.c Mutation controls on a scratch copy of src (never shipped): M1 wrong tier (sigma_y_site or extract_Sigma(:unit_obs)) must turn G2 RED; M2 wrong summary/quantile level (ICC from extract_Sigma, swapped tail) must turn G2/G3.e RED; M3 refit once per entry must turn G3.a RED. Baseline green afterwards. Record in checkpoint ledger.
- G6.d Rewrite test/test_se_machinery.jl:210-235 to the new contract, not loosened (DEFERRED behind PR #1036).
- G6.e Independent reviewer re-runs the exact public call and checks outputs, source pins and refit count. No row, receipt, scoreboard, scope or tolerance edit; no #990 action.
Pass: all throws/RED/green as stated; reviewer is not the author.

#### Unsigned items blocking signature
Estimand signed in chat ("Sign, wait for #1036", 2026-11-01) but the plan packet header still reads PROPOSED / UNSIGNED (blocker 1); G-mapping above; API shape (replace vs legacy table); phy scope; v1 TwoLevelFit inclusion; seed convention; n_boot default; #1036 sequencing for confint_derived_wald.jl.

### Draft vs Spec disagreements (integrator, 2026-11-01)

Resolved in favour of the Spec; each needs Shinichi's confirmation when the gates are signed.
1. G6.a says GllvmFit `has_diag && K_W == 0` must be refused. Spec 3.2 says SUPPORTED: tier B is the identified total `sigma_y_site(fit)`. Implemented per Spec; tested with an absolute hand-built value (Fixture A2). UNVERIFIED against R output: the claim that this total equals R's unit tier rests on a Julia source comment (`_r_unit_folds_diag` in `src/extractors.jl`, "R fits that whole sum as the single unit-tier unique part", #701). No R run or comparison was made.
2. G6.a says a GllvmFit W tier and ICC must throw. Spec 3.2/3.5: W is absent (R has no W unless rr_W/diag_W), so `:W`/`:unit_obs`/`:phy` are dropped with one `@info`, ICC is dropped with an `@info`, and the call throws only when NO requested tier survives (all-absent error). Implemented per Spec.
3. G6.a says TwoLevelFit phy is refused. Spec 3.3: an absent tier is dropped with an `@info`; alone it hits the all-absent error. Implemented per Spec.
4. G3.c leaves the seed convention open. Spec 5.3 fixes ONE MersenneTwister stream, whole batch before any refit. Implemented; tests include a seed-shift check that kills a `seed + b` design.
5. G2.b CAVEAT (has_diag communality uses a sigma_y_site denominator): resolved by computing the expected value by hand from `fit.pars` for Fixture A2, not through the extractor.
6. Code draft `bootstrap_sigma_r.jl` (n_boot 500, seed 0, `tiers`/`summaries`, long table, `n_used >= 10` minimum-survivor rule, beta fits refused, W/ICC/cross_corr refused) and the test draft built on it contradict the Spec on all of those. NOT used; the driver was written from the Spec. The TwoLevel/mutation draft (reference module `BSigTL`, ICC via `extract_repeatability`, `n_boot >= 2`, `[:B,:phy]` refused, cross_corr as an R upper triangle) likewise disagrees (Spec: cross_corr always skipped for Gaussian, phy dropped, n_boot >= 1). Only its probe ideas were reused.

## Spec

> CANDIDATE for signature (not a signature). Verbatim from the design agent; headings demoted two levels. Implementation deviations are listed after it.

##### bootstrap_Sigma_r: final design spec (pinned-R behavior target)

Status: CANDIDATE for signature. The estimand packet (plans/2026-10-28-p1-bootstrap-sigma-estimand.md:3) is PROPOSED, UNSIGNED. This spec is not a signature. The handover approval (handover :13) is of the "full pinned-R behavior target" and is not an independent signoff of a narrowed v1. See section 11.
Oracle: gllvmTMB 9539352f66f2db2cc26b1c393e67212a359b60c9 at /Users/z3437171/local-scratch/gllvmtmb-p1-pin-wt (HEAD re-read this session, worktree clean). Read-only, never edited.
Julia worktree HEAD at read time: 9974d0bb5.
Nothing was run. No Julia, no R, no fits. Every claim is tagged V(file:line) (read at the cited line this session) or UNVERIFIED.
R = R/bootstrap-sigma.R unless another R file is named. Julia paths are relative to the worktree.

#### 1. What was decided about each critic defect

| # | Defect | Resolution |
|---|---|---|
| C1 | B tier: "SUPPORTED" and "REFUSE" are one code path (has_diag && K_W==0) | Not a refusal. See 3.2. In that case `_r_tier_total(:unit)` returns sigma_y_site, which Julia documents as exactly R's unit tier (R fits the whole diagonal sum as the unit unique part) V(extractors.jl:270-276, 283-287). The W tier is absent in the same case V(extractors.jl:317). So B = identified total, W absent, ICC dropped. This is one consistent row, stated explicitly, with an absolute-value test. |
| C2 | Per-summary estimands unspecified | Section 4. |
| C3 | Simulator mis-specified | Section 5. Marginal draw is distributionally equal to R's random-effect redraw for GllvmFit; TwoLevelFit uses its own per-individual redraw. Test by empirical covariance. |
| C4 | Seeding unresolved | One MersenneTwister stream, the whole batch drawn BEFORE any refit; refit consumes none of it. Section 5.3. |
| C5 | Failure definition | Section 5.4. |
| C6 | n_boot warning / ceiling | Section 6. |
| C7 | Absent tier, all-absent, ICC | Section 3.3, 3.4. |
| C8 | Replacing old bootstrap_Sigma changes the estimand for callers | NOT replaced. New function `bootstrap_Sigma_r`; legacy `bootstrap_Sigma` and confint_derived_wald.jl are untouched V(confint_derived_wald.jl:964-995 is the legacy method). Also avoids the PR #1036 overlap. |
| C9 | Circular point-estimate test | Test 3 uses hand-computed values from `fit.pars`, an absolute check, not a difference or the extractor itself. |
| C10 | phy silently narrows the default | Section 3.3: a GllvmFit with a phylo block is refused whole. A requested but absent phy is dropped with an @info naming it. |
| C11 | Level naming Sigma_B vs Julia Sigma_unit | Keys use R's internal names (B, W, phy). Docs state Sigma_W is R's W, with no sigma_eps. GllvmFit never produces W in v1 (3.2), so there is no clash today. |
| C12 | Existing Julia warner is "more than half dropped" | Not reused. R's rule is below. |
| C13 | Gate 1 cannot hold on one GllvmFit | Two fixtures (section 9). Needs approval, because it reinterprets a gate. GATES.md not read: UNVERIFIED. |

#### 2. R contract, verified at the pin

Signature V(R:202-213):
`bootstrap_Sigma(fit, n_boot = 999, level = c("unit","unit_obs","phy","B","W"), what = c("Sigma","R","communality","ICC","cross_corr"), conf = 0.95, seed = NULL, n_cores = 1, progress = TRUE, keep_draws = FALSE, link_residual = c("auto","none"))`

Order of checks V(R:214-231):
1. `.temporal_assert_no_iid_inference`, `.structured_rho_refit_assert`, class check, mspl and unweighted asserts.
2. `level` via match.arg then `.normalise_level` per element (unit to B, unit_obs to W; B/W aliases emit a once-per-session deprecation, option key per level, V(R/normalise-level.R:59-76, :81-85)). Default vector contains "B" and "W", so by source the first default call emits that warning. Not run: UNVERIFIED by execution.
3. `what`, `link_residual` via match.arg.
4. `conf` must be numeric in (0,1), else abort V(R:225-227). `n_boot` must be numeric >= 1, else abort V(R:228-231). Then `as.integer`.
5. Ceiling warning block V(R:252-277), then tier availability V(R:280-296), then ICC drop V(R:299-306).

Tier availability V(R:280-286): B = rr_B or diag_B; W = rr_W or diag_W; phy = phylo_rr or phylo_diag. `level_kept <- unique(level[level_avail[level]])` V(R:289). Order is the user's order after normalisation, so the default gives B, W, phy. Empty gives an abort with "None of the requested level(s) are present" V(R:290-295).

Batch and refit V(R:333-336, 371-415): `set.seed(seed)` once, only if non-NULL; `Y_sim <- simulate(fit, nsim = n_boot)` once in the parent; the refit loop is unseeded. Each replicate refits with `control = gllvmTMBcontrol(se = FALSE)` and extracts summaries. Failure V(R:407-413): fit NULL (error caught), wrong class, or `opt$convergence != 0`; then `.na_summaries(point_est)` tagged `failed`.

simulate default V(R/methods-gllvmTMB.R:1457-1464): `condition_on_RE = FALSE` redraws random effects; an unhandled tier falls back to conditional simulation with a warning, giving too-narrow intervals.

Summaries V(R:518-602): per level, `extract_Sigma(level, part="total", link_residual, .skip_warn=TRUE)`, with errors caught to NULL (the level is skipped). Entries: `Sigma_<lvl>` = `$Sigma`; `R_<lvl>` = `$R`; `communality_<lvl>` only for B, W, phy (`extract_communality`, phy forced to link_residual "none"); `multiple_r_<lvl>` from `extract_cross_correlations(contrasts = FALSE)` with errors caught to NULL. After the loop: `ICC_site` = `extract_ICC_site` only if both B and W are in `level`. Note: ICC is computed whenever B and W are both in `level` AND "ICC" in `what`.

Aggregation V(R:625-735): alpha = 1 - conf; bounds are `quantile(probs = alpha/2, 1-alpha/2, na.rm = TRUE)` (R type 7). A draw missing a name contributes NA, which is not counted in n_failed. Matrix entries: bounds only, NO n_effective entry (the n_effective assignment sits in the vector branch only, V(R:728-730)). Vector entries: `n_effective` = `rowSums(is.finite(stacked))`. `multiple_r_*` entries also get NA bounds where n_eff/n_boot < 0.8 and a `boot_median`.

Attrition V(R:465-486): over names of `n_effective` (vector entries only), take min(n_eff)/n_boot; entries below 0.8 trigger ONE warning naming the worst. Because matrix entries have no n_effective, `Sigma_*` and `R_*` never trigger it. This is an R limitation; copy it, document it, do not "fix" it.

Return V(R:488-508): list with point_est, ci_lower, ci_upper, n_effective, boot_median, ci_method = "percentile", link_residual, conf, n_boot, coverage_ceiling, n_failed, level (internal names after dedup), what (after ICC drop), draws (NULL unless keep_draws); class c("bootstrap_Sigma","list").

Doc/code mismatch V(R:81-84 says "refused", R:254-265 warns only): copy the warning. Do not add a refusal.

#### 3. Julia scope and per-class verdicts

##### 3.1 Classes
- GllvmFit, non-record: supported (v1).
- TwoLevelFit: stretch item. Supported if the stretch is approved (section 8).
- GllvmFit with a retained Gaussian record (`_has_gaussian_record`, aghq_gaussian_fit.jl:260): refused in v1. Its refit route `_gaussian_record_refit` with `simulate(f, n; rng = MersenneTwister(seed+b))` V(aghq_gaussian_fit.jl:406) is per-replicate seeded, not an R-style batch. Equivalence to R: UNVERIFIED.
- PrecisionMultivariateFit, JointPhyloGroupedGaussianFit, any non-Gaussian fit: refused (ArgumentError naming the class). No simulate and refit route was verified: UNVERIFIED.
- GllvmFit with `lambda_constraint` pins: refused via `_check_unpinned_refit(fit, "bootstrap_Sigma_r")` V(confint.jl:552-560).
- GllvmFit with a phylogenetic block (`K_phy > 0 || has_phy_unique`, fit.jl:29-36): refused whole. No Julia phy tier extractor exists V(extractors.jl:125-127 docstring: no phy tier).

##### 3.2 GllvmFit tiers (v1)
Identification: one unit per column of y, so only the sum of B and W is identified V(extractors.jl:129-135).

| Fit structure | B tier | W tier | ICC | Verdict |
|---|---|---|---|---|
| K_W > 0 (any has_diag) | split unidentified | split unidentified | n/a | REFUSE the whole call (ArgumentError citing identification) V(extractors.jl:131-135) |
| K_W == 0, has_diag == false, K >= 1 | present, `ΛΛ'` only (no diagonal, no sigma_eps) | absent | dropped | SUPPORTED. V(extractors.jl:288-293, 316-321) |
| K_W == 0, has_diag == true | present = identified total `sigma_y_site(fit)` (ΛΛ' + diag(σ²_B + σ²_W + σ_eps²)); this IS R's unit tier for that model | absent | dropped | SUPPORTED, with the semantics stated in the docs. V(extractors.jl:270-276, 285-287, 317) |
| K == 0 and has_diag == false | absent | absent | n/a | tier dropped; all-absent error (3.4) |
| K == 0 and has_diag == true, K_W == 0 | present (diagonal only) | absent | dropped | SUPPORTED; communality omitted (R requires rr tier, V(R/extractors.R:245-247)) |

"W absent" is not a refusal. It matches R, where W exists only if rr_W or diag_W is fit V(R:282). Do NOT use `extract_Sigma(:unit_obs)` (adds σ_eps² unconditionally, V(extractors.jl:82)) or `extract_ICC_site(::GllvmFit)` (built from it, V(extractors.jl:706-711)) anywhere in this function.
The has_diag && K_W==0 simulation covariance contains the whole sum, and the refit recovers the same identified total, so the percentile interval is for an identified estimand. V(confint_derived.jl:426-447).

##### 3.3 Level argument
`level::Union{Symbol,Tuple/Vector{Symbol}}` default `(:unit, :unit_obs, :phy, :B, :W)`. Normalise :unit/:B to `:B`, :unit_obs/:W to `:W`, :phy to `:phy` (R mapping V(R/normalise-level.R:81-85)). Any other symbol: ArgumentError (R: match.arg abort). Deduplicate in first-seen order, then drop absent tiers silently except one @info naming dropped tiers ("not present in this fit"). No deprecation warning for :B/:W (R's is session-noise, decision, documented).

##### 3.4 All-absent
If no requested tier survives: `ArgumentError("None of the requested level(s) are present in the fit. Available: ...")` V(R:290-295).

##### 3.5 ICC
Dropped with an @info ("ICC requires both B and W tiers; dropping ICC from what") unless both :B and :W survive V(R:299-306). For GllvmFit v1 this means ICC is always dropped. For TwoLevelFit it is kept.

#### 4. Per-summary estimands (point estimate AND every replicate use the same function)

Define for a fit f and tier t in {B, W} an internal pair (Sigma_t, shared_t):

GllvmFit: `Sigma_B = _r_tier_total(f, :unit)`, `shared_B = _r_tier_shared(f, :unit)` V(extractors.jl:283-312). Never call `_r_tier_total(f, :unit_obs)` (W is absent for supported fits).
TwoLevelFit: `Sigma_B = f.Σ_B`, `Sigma_W = f.Σ_W` (no σ_eps; V(twolevel.jl:153-156, 243-244)); `shared_B = Λ_B Λ_B'`, `shared_W = Λ_W Λ_W'`.

| Summary | Key | Definition | Present when |
|---|---|---|---|
| Sigma | `Sigma_B`, `Sigma_W` | Sigma_t, a p x p Matrix | tier present |
| R | `R_B`, `R_W` | `_cov2cor(Sigma_t)` V(extractors.jl:88-96) (NaN on a non-positive diagonal; R uses its own cov2cor, NaN behaviour: UNVERIFIED) | tier present |
| communality | `communality_B`, `communality_W` | `diag(shared_t) ./ diag(Sigma_t)`, NaN where the denominator is not finite-positive (matches `extract_communality`, V(extractors.jl:360-366)) | rr tier present: GllvmFit K >= 1; TwoLevelFit always (K_B,K_W >= 1, twolevel.jl:201). Omitted otherwise, as R returns NULL when the rr tier is unused V(R/extractors.R:238-247). |
| ICC | `ICC_site` | `vB ./ (vB .+ vW)` with `vB = diag(Sigma_B)`, `vW = diag(Sigma_W)`, NaN where `vB+vW` is not finite or <= 0 (R `.safe_icc_ratio`, V(R/extractors.R:127-133); R names the vector by trait levels V(R/extractors.R:123)) | both :B and :W present (TwoLevelFit only in v1) |
| cross_corr | `multiple_r_B`, `multiple_r_W` | R needs a multinomial() trait; the abort is caught and the entry skipped V(R:577-591, doc R:155-158). Julia Gaussian fits have no nominal trait, so this is ALWAYS skipped silently. Do not call the Julia `extract_cross_correlations` (different, :unit-only function, V(extractors.jl:441)). `what` still stores `:cross_corr` as given. | never in v1 |

link_residual: accepted values `:auto`, `:none`; both give identical numbers for Gaussian. Reason: R's doc says "auto" is a no-op for Gaussian V(R/extract-sigma.R:612-617, documentation lines; the executable `link_residual_per_trait` at :115-130 was not traced to its Gaussian branch: UNVERIFIED in code). R's ICC applies link_residual to W only and forces "none" for B V(R/extractors.R:105-116); no effect for Gaussian. Test that :auto and :none are bit-identical. Retracted premise: R's link_residual does NOT map to Julia `residual_mode` V(precision_multivariate_fit.jl:19-22 cited by the plan, not re-read: UNVERIFIED).

Failed replicate placeholder: every entry the same shape as the point entry, filled with NaN, replicate flagged failed (R `.na_summaries`, V(R:609-617)). An extraction that yields no entry for a replicate (while the point estimate has one) gives NaN for that replicate and is NOT counted in n_failed V(R:643-645, 447-451).

#### 5. Simulation, refit and seeding semantics

##### 5.1 Batch
Draw all `n_boot` replicate datasets BEFORE any refit, in the parent, from ONE stream (R V(R:333-341)). Memory: n_boot x p x n floats. R holds the same, so this mirrors it.

##### 5.2 Draw distribution
GllvmFit: replicate data are iid columns `y_s ~ N(mu_hat_s, A)` with A the full site covariance `_derived_site_cov(fit)` V(confint_derived.jl:426-447), reusing `_derived_simulate!` V(confint_derived.jl:449-469). This equals R's "redraw random effects from the fitted covariance, then add residual" in distribution, because the marginal covariance is `ΛΛ' + Ψ + σ_eps²I`. Equivalence is argued from the structure, never run: UNVERIFIED. Test: with n large, empirical covariance of the batch is close to A.
TwoLevelFit: per individual `b_i = LB randn(p)`, observation `y_ij = b_i + LW randn(p)`, with LB/LW from `_psd_sqrt_factor` V(twolevel.jl:547-554, 484). That inline loop must be factored into a helper that takes an RNG (no behavior change to `repeatability_bootstrap_ci`; this edit to twolevel.jl needs a lease decision: section 11). The fixed-effect mean: GllvmFit with X forwards `X` via `_mean_X` / `_derived_mean_from_X` V(confint_derived.jl:580-607); TwoLevelFit has mean removed (`center = true`, twolevel.jl:204-207).
No tier falls back to a conditional simulation in v1. A fit class without a redraw is refused; there is no too-narrow fallback.

##### 5.3 Seeding
`seed::Union{Nothing,Integer} = nothing`. Not nothing: `rng = MersenneTwister(seed)`. Nothing: `Random.default_rng()` (R: no seed set V(R:333-335)). Only the batch uses rng; the refit loop draws nothing from it. Consequence, to be tested: a crippled or failing refit leaves every other replicate's data unchanged. Discriminating test: record the datasets passed to `_refit` under two runs that differ only in which refits fail; datasets must be identical. The legacy sequential stream V(confint_derived.jl:627-634) and the record path's `seed+b` V(aghq_gaussian_fit.jl:406) are NOT used. `n_cores`: accepted for signature parity; value 1 only. `n_cores > 1` throws ArgumentError "unsupported in v1" (R uses future, V(R:418-435); Julia parallelism not designed).

##### 5.4 Failure
A replicate FAILS if the refit throws, or returns a fit with `converged == false` (GllvmFit V(fit.jl:58); TwoLevelFit V(twolevel.jl:561-562)). R equivalent V(R:407-413). A fit with `converged == true` but non-finite summary entries is NOT failed; its entries are NaN and excluded from the quantile (R: na.rm, non-finite counted out of n_effective, V(R:707)). Refit seam: reuse `_refit::Function = fit_gaussian_gllvm` (GllvmFit, V(confint_derived.jl:540, 636-643)) and `fit_twolevel_gaussian` (V(twolevel.jl:194)); internal test seam, not public.

##### 5.5 Quantiles
For each entry element: collect the finite draws, `_derived_percentile(v, alpha/2)` and `(v, 1-alpha/2)`; this is linear interpolation at h = (n-1)p, identical to R type 7 V(confint_derived.jl:32-44). No minimum-survivor rule (the legacy `n_used >= 10` rule V(confint_derived.jl:664-668) is NOT in R and is not copied). Zero finite draws gives NaN bounds (R: NA).

#### 6. Warnings (all non-throwing)

1. `min_boot = ceil(2/(1-conf)) - 1` (39 at 0.95); `coverage_ceiling = (n_boot-1)/(n_boot+1)` V(R:252-253). If `n_boot < min_boot`: `@warn` stating the ceiling and min_boot V(R:254-265). Else if `n_boot < 999`: `@warn` about noisy bounds V(R:266-277). Never both. Neither throws.
2. Attrition `@warn` per section 2 rule V(R:465-486), over vector entries only.
3. A drop-level `@info` and the ICC `@info` (sections 3.3, 3.5).
Julia @warn text is Julia's own; only the trigger conditions and the returned fields must match R.
The ceiling is also returned in `coverage_ceiling` V(R:498-502).

#### 7. API and output

```julia
bootstrap_Sigma_r(fit::Union{GllvmFit,TwoLevelFit};
    n_boot::Integer = 999,
    level = (:unit, :unit_obs, :phy, :B, :W),
    what  = (:Sigma, :R, :communality, :ICC, :cross_corr),
    conf::Real = 0.95,
    seed::Union{Nothing,Integer} = nothing,
    n_cores::Integer = 1,
    progress::Bool = true,
    keep_draws::Bool = false,
    link_residual::Symbol = :auto,
    # data the fit does not retain (same meaning as bootstrap_ci_derived):
    y = nothing, n_sites = nothing, X = nothing, Σ_phy = nothing,   # GllvmFit
    individual = nothing, center::Bool = true)                       # TwoLevelFit
    -> BootstrapSigmaR
```
Defaults mirror R V(R:202-213). The extra keywords exist because Julia fits do not carry the data (GllvmFit non-record fits need `y` or `n_sites`, V(confint_derived.jl:567-578); TwoLevelFit stores only `nindiv`, V(twolevel.jl:158-169)). `Σ_phy` is accepted only to be refused when non-nothing (phylo fits refused in v1). `what` and `level` accept a single Symbol or a collection; `what` is validated against the five names (unknown name: ArgumentError, R match.arg). `link_residual` outside `(:auto,:none)`: ArgumentError. Validation order follows R: class, level, what, link_residual, conf (0 < conf < 1), n_boot (>= 1), then warnings, then availability, then ICC drop, then simulate, then refit.
`progress = true` (R default V(R:439-441) prints one message per replicate): implement as one `@info` per replicate. Tests pass `progress = false`. UNVERIFIED that the maintainer wants Julia's default noisy; recommend mirroring R and say so in docs.

`struct BootstrapSigmaR` (R list V(R:488-508)); all fields:
- `point_est`, `ci_lower`, `ci_upper`: `NamedTuple` or `Dict{Symbol,Any}` keyed `:Sigma_B`, `:R_B`, `:communality_B`, `:Sigma_W`, `:R_W`, `:communality_W`, `:multiple_r_*`, `:ICC_site`, in R's order: per level in `level` order (Sigma, R, communality, multiple_r), then `ICC_site` V(R:526-600). Matrices are `Matrix{Float64}`; vectors are `Vector{Float64}`.
- `n_effective`: same keys, VECTOR entries only (communality_*, ICC_site, multiple_r_*), `Vector{Int}`; NO entry for Sigma_* or R_* V(R:728-730).
- `boot_median`: `multiple_r_*` entries only (empty in v1) V(R:717-720).
- `ci_method::String = "percentile"`, `link_residual::Symbol`, `conf`, `n_boot::Int`, `coverage_ceiling::Float64`, `n_failed::Int`.
- `level::Vector{Symbol}`: internal names `:B`, `:W`, `:phy` after dedup and availability drop (NOT :unit / :unit_obs) V(R:296, 504).
- `what::Vector{Symbol}`: as given, minus `:ICC` if dropped V(R:304-305, 505).
- `draws`: `nothing` unless `keep_draws`; else `Vector` of per-replicate entries (failed ones NaN-shaped) V(R:506).
Column and field names verified against R:488-508. No tabular output: the legacy `(i, j, estimate, lower, upper, n_converged, n_valid)` table V(confint_derived_wald.jl:993-994) is not reused.

Not edited: `bootstrap_ci_derived`, the legacy `bootstrap_Sigma`, extractors.jl, the record path.

#### 8. v1 scope summary
Supported: GllvmFit (non-record, no pins, no phylo block, K_W == 0): B tier; Sigma, R, communality; ICC dropped; cross_corr skipped.
Stretch (needs your yes): TwoLevelFit with `individual`: B and W tiers; Sigma, R, communality; ICC.
Refused with explicit errors: K_W > 0; record fits; pinned fits; phylo block; PrecisionMultivariateFit; JointPhyloGroupedGaussianFit; non-Gaussian fits; n_cores > 1; unknown level/what/link_residual.

#### 9. Tests (RED first, each with a named control), new file test/test_bootstrap_sigma_r.jl
Fixture A: GllvmFit p=4, K=1, n=150, K_W=0, has_diag=false (control: sigma_eps = 0.5 so sigma_eps² is visible). Fixture A2: same with has_diag=true. Fixture T: TwoLevelFit, distinct Σ_B and Σ_W.
1. RED baseline: the new function does not exist / legacy table has no `what`.
2. Validation: defaults; `:B`/`:unit` and `:W`/`:unit_obs` dedup; all-absent ArgumentError; conf outside (0,1); n_boot < 1; unknown what; ICC dropped with @info on A.
3. Absolute point estimates. A: `Sigma_B == Λ*Λ'` computed from `fit.pars`. Control: diag differs from `sigma_y_site` by exactly `σ_eps² (+ σ²_W)` computed from pars. A2: `Sigma_B == sigma_y_site`. T: `Sigma_B != Sigma_W`, equal to hand-built `ΛΛ'+diag`. Communality and ICC likewise hand-computed.
4. Refit count == n_boot regardless of p and `what` (wrap `_refit`). Control: legacy method performs p(p+1)/2 x n_boot.
5. Batch semantics: the datasets handed to `_refit` are identical across summaries; identical for equal seeds; different for different seeds; identical across two runs that differ only in which refits fail (C4 test).
6. Failure accounting: cripple every k-th refit; `n_failed` equals injected; bounds from survivors only; attrition @warn fires for vector entries when below 80 percent; no attrition warning for matrix-only `what` (R behavior). Control: uncrippled gives n_failed == 0. Control 2: a converged refit with a NaN entry is not counted failed.
7. Percentile correctness: injected deterministic `_refit` returning a known sequence; bounds equal `quantile(draws, [alpha/2, 1-alpha/2])` (type 7); catches swapped tails or wrong alpha.
8. Warnings: n_boot = 10 at conf 0.95 warns and does not throw; n_boot = 100 warns the other message only; n_boot = 999 silent; `coverage_ceiling == (n_boot-1)/(n_boot+1)`; `min_boot` boundary at 39 / 40.
9. keep_draws: nothing by default; n_boot entries when true.
10. Refusals each with a supported control: K_W > 0; record fit; pinned fit (also closes the coverage gap V(plan: test_confint_method_routes.jl:414-440 has no bootstrap_Sigma_r call)); phylo block; PrecisionMultivariateFit; JointPhyloGroupedGaussianFit; non-Gaussian fit; n_cores = 2.
11. `:auto` vs `:none` bit-identical.
12. Simulator: with large n the empirical covariance of the batch is within MC tolerance of A (GllvmFit) and of Σ_B + Σ_W per individual structure (TwoLevelFit).
13. ICC edge: injected vB + vW <= 0 gives NaN, not Inf or an error.
Mutations (scratch copy of src OUTSIDE the worktree, in the scratchpad; record in the ledger; never shipped): M1 wrong tier (use sigma_y_site or extract_Sigma(:unit_obs)) must redden tests 3, 4; M2 wrong summary (communality replaced by a Sigma diagonal; ICC from extract_Sigma; wrong quantile level) redden 3, 7; M3 repeated refit per entry redden 4; M4 seed+b per replicate redden 5; M5 count failures by non-finite only redden 6.
test_se_machinery.jl:210-235 pins the legacy method and is NOT touched, because the legacy method is not changed.

#### 10. Estimates (D-287)
Nothing measured. Inference only: one p=4, n=150 refit about tens to a few hundred ms; the new test file about 600 refits, roughly 1 to 3 min plus compile; mutations 5 runs, roughly 5 to 15 min. All far under 3 h. Time one n_boot=10 pre-run first and put it in the ledger. No p=80 campaign is part of this work (about 37 to 44 h by recorded figures, needs a plan and approval).

#### 11. Not decided here; needs Shinichi or a gate
1. Whether the handover approval counts as the signed leaf contract for a narrowed v1.
2. Wiring: `include("bootstrap_sigma_r.jl")` and the export edit live in src/GLLVModels.jl, and the test shard entry in test/runtests.jl; both are deferred behind PR #1036 by the task rules. Until then, new code is exercised only by including the file into the module in a scratch run. That the interim include works without touching other files: UNVERIFIED.
3. Stretch scope for TwoLevelFit requires a behavior-preserving helper extraction in src/twolevel.jl (leasing it is D-87, not decided).
4. GATES.md (G1-G4, G6) unread; the two-fixture reading of Gate 1 needs sign-off.
5. Default n_boot 999 (R) rather than the legacy 500, and a noisy `progress = true` default.
6. Phylo tier and the phy simulate/refit routes (Precision, JointPhylo): out of scope, unverified.

#### 12. Claim register
VERIFIED (read this session): R signature, check order, warnings, tier availability, dedup, ICC drop, seeding, failure rule, quantile aggregation, n_effective only for vectors, attrition rule, return fields (R:202-735); R normalise-level; R extract_ICC_site and `.safe_icc_ratio`; R extract_communality NULL when rr unused; R simulate doc; Julia `_r_tier_*` and `_r_tier_present`, `extract_communality`, `extract_correlations`, `extract_ICC_site`; `_derived_site_cov`, `_derived_simulate!`, `bootstrap_ci_derived` (rng, converged, quantile); `_derived_percentile` (type 7); TwoLevelFit struct, fitter (K_B,K_W >= 1), extractors, inline redraw simulator; `_check_unpinned_refit`; legacy bootstrap_Sigma; record path seeding.
UNVERIFIED: any run (R or Julia), including the default-level deprecation warning; R `link_residual_per_trait` Gaussian branch in code; R cov2cor NaN behavior; whether `fit_gaussian_gllvm` consumes any global RNG; interim include works; GATES.md; PR #1036 and #990 content; unfinished Dropbox worktree scan; origin/docs/bootstrap-sigma-reader-wording; PrecisionMultivariateFit and JointPhylo routes; TwoLevelFit mapping from an R fit with both diag_W and sigma_eps.

### Implementation deviations from the Spec (integrator)

- TwoLevelFit simulator: Spec 5.2 says to factor the inline redraw loop of `repeatability_bootstrap_ci` into a helper in `src/twolevel.jl`. That file is outside the lease, so the loop is DUPLICATED inside `src/bootstrap_sigma_r.jl` (same factors `_psd_sqrt_factor`, same per-individual structure). Risk: the two loops can drift. A later extraction needs a lease (D-87).
- Added a refusal the Spec is silent on: if no requested summary is reportable for the surviving tiers (e.g. `what = [:cross_corr]`, always skipped for Gaussian), the call throws `ArgumentError` before simulating, rather than running `n_boot` refits to return an empty result.
- Floating point is copied, not corrected: `min_boot = ceil(2/(1-conf)) - 1` is 39 at 0.95 but 20 (not 19) at 0.9 because `2/(1-0.9) > 20` in IEEE doubles; R's expression behaves the same.
- Extra internal test seam `_refit` (not public), signature `_refit(y; K,K_W,has_diag,K_phy,has_phy_unique,Σ_phy,X)` for GllvmFit and `_refit(y, individual; K_B,K_W,center)` for TwoLevelFit.
- Structural refusals (K_W, record, phylo, pins) run before argument validation, not after. Only the order among refusals differs from R.
- A fit that carries intercept/fixed effects is supported via `X` (or the stored intercept design through `_mean_X`), as in `bootstrap_ci_derived`; this path has NO test beyond the shared code path (UNVERIFIED).

### Mutation record (integrator, 2026-11-01)

Scratch driver copies in `/private/tmp/claude-503/-Users-z3437171-Dropbox-Github-Local-GLLVM-jl/eb56923c-569d-427b-bdc5-d1f448428f52/scratchpad/bsig/mut/` (never in the worktree). Baseline unmutated: green (154/154). 17 of 17 mutants KILLED by the unchanged test file. Testsets that killed each (numbers refer to the testsets in `test/test_bootstrap_sigma_r.jl`): M1a tier=sigma_y_site: 2,4,5,6; M1b tier=extract_Sigma(:unit_obs): 2,4,5,6; M2a communality=diag(Sigma): 2,12; M2b alpha not halved: 4,5,6,12; M2c swapped tails: 4,5,6,12,13; M2d ICC complement: 12,13; M3 refit per entry: 4,5,6,8,10,12; M4 seed+b: 4; M5 failure by non-finite only: 5,6,8,12; M6 seed ignored: 4,9; M7 converged-NaN counted failed: 5; M8 TwoLevel W reads B: 12,13; M10 failed placeholder zero: 5,6,8; M11 min_boot off by one: 7; M12 attrition threshold: 5; M13 unit_obs aliased to B: 1,12; M14 ICC kept with one tier: 1,12. Raw: `/private/tmp/claude-503/-Users-z3437171-Dropbox-Github-Local-GLLVM-jl/eb56923c-569d-427b-bdc5-d1f448428f52/scratchpad/bsig/run/mutation-results.txt`.

### Status after implementation (integrator, 2026-11-01)

Driver and tests written and green in a scratch project; not staged, not committed, not wired. Superseded by the update below (211/211). Blocked: PR #1036 hook-in (`src/GLLVModels.jl`, `test/runtests.jl`, legacy rewrite in `test_se_machinery.jl`), signed GATES, signature of the Spec. Unsupported in v1: record, pinned, phylo, precision, joint-phylo and non-Gaussian fits, `n_cores > 1`, phy tier, cross_corr.

### Update after review (2026-11-01)

Three reviews led to additions: `src/bootstrap_sigma_r.jl` is now 403 lines and `test/test_bootstrap_sigma_r.jl` 570 lines, 17 testsets, 211 assertions, 211/211 passing in the scratch project (`bsig/envwt`; the worktree has no Manifest, `src` is symlinked). New testsets: 14 (R_B, communality_B and Sigma_B bounds on varying replicates for `has_diag`, plus a failed-replicate variant), 15 (rr-only `communality_B` is exactly 1 with a zero-width interval, flagged by an `@info`), 16 (`has_diag` real simulate-and-refit route; `n_sites` > `X` > `y` priority), 17 (`InterruptException` is rethrown; other refit errors count as failed replicates). One earlier test bug was fixed by renaming a shadowed variable `n` to `nb41`; no assertion was changed and no tolerance was loosened.
Mutation record above is from the 154-assertion version. It was not re-run on the 211-assertion file in this checkpoint.

Unverified: `has_diag` with `K_W == 0` as R's unit tier rests on a Julia source comment, not on R output. Nothing was compared with R output at all, so no parity-with-R claim is made.

Still blocked or open: PR #1036 hook-in (`src/GLLVModels.jl`, `test/runtests.jl`, legacy rewrite in `test/test_se_machinery.jl`); gates unsigned (plan packet still PROPOSED / UNSIGNED); unsupported classes (record, pinned, phylo, precision, joint-phylo, non-Gaussian, `n_cores > 1`, phy tier, cross_corr); no R comparison; no full or core suite run; no independent re-run of G6.e.

## Further deviations from pinned R (R-fidelity re-read, source-only; nothing compared with R output)
1. `n_cores != 1` throws (R runs sequentially for n_cores <= 1).
2. Seeding uses a private `MersenneTwister(seed)` and leaves the global RNG alone; R calls `set.seed`. Draws cannot match R's bit for bit.
3. Refit warnings and logs are not suppressed (R wraps each refit in suppressMessages/suppressWarnings, `se = FALSE`).
4. Strict argument types and exact matching: `n_boot::Integer` (R coerces 1e3 or 50.0), no partial matching of `level`, `what`, `link_residual`.
5. `what` is de-duplicated with `unique`; R keeps duplicates unless ICC is dropped.
6. Failed replicates in `draws` are NaN-shaped with no failed flag; an extraction error NaN-fills the whole replicate where R skips per level.
7. No trait names on returned matrices and vectors; R carries dimnames.
8. Result is a `BootstrapSigmaR` struct, not an R-style list; `$` and `[[ ]]` access do not carry over.
9. Check order: structural refusals and the `n_cores` check run before level/what/conf/n_boot validation.
10. Empty `what` after the ICC drop throws; R proceeds to refits and returns empty lists.

### Deviations added by PR #1059 hardening (2026-11-01; nothing compared with R output)
11. **Loss warning for every `what`.** The driver warns whenever ANY replicate is lost (refit failed or threw, or its summaries could not be extracted), naming `n_lost of n_boot`. R warns only when a vector summary keeps under 80% of replicates, and a matrix-only request got no warning in the Spec. This is a strict superset of R's warning (the 80% attrition warning is unchanged and can fire as well, so a user may see two). The returned structure is unchanged.
12. **Optimizer settings for `has_diag` (or `K_W > 0`) refits.** The built-in refit (never a user `_refit`, never a phylo fit) uses `iterations = 2000, f_tol = 1e-12` (R's nlminb stops on a relative objective change). This is a deviation in how convergence is judged, not in the model. Measured on the has_diag K=2 fixture (1998 refits, two seeds): stock lost 8 (0.4%) at the 500-iteration cap; with these settings 0 are lost, logLik is within 1e-4 of R, and it is not slower than stock (mean 60 vs 77 ms, Totoro under load). Bounds: stock survivors differed from R's by up to 3.2e-3; with the lost replicates rescued the difference is 6e-6 to 1.1e-5 (fixtures a and b move by about 1e-5 or less). So it also moves already-converged replicates slightly (about 1e-5), which is NOT "results unchanged". `f_tol = 1e-10` was tried first and rejected (stops early, logLik up to 1.3e-3 low). It is a separate commit and can be reverted alone. Source: the OPTIMUM investigation (scratchpad `investigate-optimum/OPTIMUM.md`); only 2 seeds and one DGP per fixture; the TwoLevel route is untouched (its ~0.6-0.9% worse optima are converged local optima the driver cannot flag).
13. **`R_*` bound clamp.** Correlation bounds are clamped to [-1, 1] only when they exceed it by at most 1e-8 (interpolation rounding); a larger excursion is left visible so a genuine upstream error is not hidden.
14. **Wrong-shaped replicate summary.** Becomes a NaN draw and is counted in the "lost" warning, like an extraction error. It is not in `n_failed` (which, as before, counts failed refits only).
15. **`n_sites` between 2 and p.** Accepted. A fit with fewer sites than traits usually fails every refit, so the result is all-NaN, announced only by the "lost" warning (no refusal). `n_sites < 2` is refused.
16. **Extra input refusals** (the Spec was silent): vector or wrong-rowed `y`, 2-d `X`, `X` whose site count differs from `n_sites`, too-short `individual`, negative `seed`.
17. **Test coverage of item 12:** one replicate (`test/data/bsig_c_s101_b65.csv`, seed 101 replicate 65) pins that the built-in refit converges where the stock fit stops at 500 iterations, and the seam sees no injected options. That the options are NOT applied to fits without a diagonal is untestable on well-posed data (the options change nothing there), so that guard rests on the code only.

### Deviations added by the phy tier (2026-11-02; items 18 to 22 are Stage 2, `JointPhyloGroupedGaussianFit`; nothing compared with R output)
18. **Failure rule beyond R's.** For a `JointPhyloGroupedGaussianFit`, a replicate fails (counted in `n_failed`, NaN-shaped, excluded from percentiles) when the refit throws, does not converge, OR its covariance components are not identified (`_joint_phylo_grouped_component_identification(refit).reason !== :identified`). R's rule is `nlminb` convergence only and has no identification check. The Julia `converged` flag also requires a fresh finite-difference gradient norm below `g_tol` (the same difference from R already applies to `PrecisionMultivariateFit`, added in #1069 and not listed above). Failure rates are not comparable with R's. The breakdown (`n_thrown`, `n_not_converged`, `n_non_identified`) is in `result.diagnostics`.
19. **B/W dropped.** R would report B/W tiers for a comparable multi-tier fit; the mapping of the ordinary terms of this class to R's B/W is unverified, so the tiers are dropped with an `@info` that says so (coverage difference from R).
20. **Non-PD Hessian kept and counted.** As in R, a converged replicate with a non-positive-definite Hessian is kept. Added: the count of such replicates among converged ones (`diagnostics.n_hessian_not_pd` of `n_converged`) is reported in an `@info`, and a `@warn` is issued when the original fit's Hessian is not PD. Pre-run (Totoro, six fixtures, one seed each, 200 replicates): non-PD among converged replicates 33.5% (tiny, 67/200), 9.5%, 9%, 0%, 33.7% (67/199) and 71.4% (140/196); non-converged or non-identified or thrown replicates 0% to 2.5% (worst: 5/200). With 200 replicates a count of 5 cannot exclude a true failure rate above 5%. Fixture refits ran at `g_tol` 1e-5 to 2e-4 and at most 200 iterations; no `:iteration_limit` failure occurred, so no iteration deviation was made.
21. **Fitter warning silenced.** The fitter's per-replicate "not separately identifiable" `@warn` is silenced inside the loop (status checked and counted instead); exceptions are not silenced. The original fit must be converged and identified or the call is refused (`ArgumentError`).
22. **Evidence level.** Structural plus Julia descriptive only. Untested in a real refit: more than one ordinary term, rank above 1, `:unit_obs`/`:cluster2` terms, covariates beyond trait intercepts, larger or unbalanced trees. No R comparison, no R parity claim, no row, score or tolerance change. The simulator draws ordinary terms per group through the stored incidences; group labels for the refit are recovered from those incidences, keyed by term name (for a `:unit_obs`-only fit the `unit_obs` partition stands in for the unit labels, which the fit does not store).

### Driver fixes after fuzzing (2026-11-02)
23. **PMV refit warning silenced.** The `PrecisionMultivariateFit` fitter warns about structural redundancy on every likelihood evaluation (d = 2 rank 1 `:explicitunique`, d = 2 rank 2, d = 5 rank 3: over a million warnings at `n_boot = 999`), so it is silenced inside the refit loop only (`src/grouped_fit.jl` untouched); the driver's own warnings stay and an exception is not silenced.
24. **String `level` / `what`.** A `String` or a vector of `String` is accepted (it was iterated as characters, giving "unknown level :p").
25. **`what` echo.** `result.what` echoes the request (after the ICC drop), as R does, so it can list `:cross_corr` although no such entry exists; `point_est` keys are the authoritative list. Documented rather than changed. The refusal for `what = :ICC` or `:cross_corr` alone now names the classes that support `:Sigma`, `:R`, `:communality`.

### Group-term guard (2026-11-02; finding S2-F1; nothing compared with R output)
26. **Group-term guard for `JointPhyloGroupedGaussianFit` (driver choices, not R's).** The upstream identification gate returns `:identified` for an ordinary group term with 1 or 2 groups or one aliased with fixed group dummies (group variance about 1e-13). The driver therefore (a) REFUSES (`ArgumentError`, naming the term and its group count) when any ordinary term has fewer than 3 groups (`_BSIG_MIN_GROUPS = 3`: 3 is the smallest count leaving a contrast after the intercept; a floor, not a guarantee of a well-estimated variance); (b) REFUSES when, for some trait, the term's group indicators add no rank to that trait's rows of the mean design (`rank([X_t Z_k]) == rank(X_t)`, exact for total aliasing; near-aliasing, such as a covariate almost proportional to a group contrast, is NOT detected); (c) WARNS, without refusing, when a term's estimated variance is below `1e-8` of that trait's total variance (phylogenetic + all ordinary + residual) so the user knows the group interval is on a boundary (`_BSIG_TINY_REL`; the fuzz runs showed 1e-13 to 1e-15 against O(1) totals). R has no such rule; the upstream fitter and `src/confint_derived_wald.jl` are untouched. Whether the `:phy` intervals are wrong in these cases was never assessed (only that the refusal guarantee failed); this guard is a conservative refusal, not a coverage result.

## Verification on Totoro (2026-11-01, clean instantiated `--project=.`)
Synced the worktree (no .git) to totoro:~/claude-bsig-p1/GLLVM.jl, `Pkg.instantiate()` on Julia 1.12.6, then `test/test_bootstrap_sigma_r.jl`: 211 pass / 211 total, 52 s, one thread. Scale timing of `bootstrap_Sigma_r` (n=200, n_boot=99, K=2, Sigma + communality, one thread): p=10 about 10 ms per replicate, p=20 about 1 ms, p=40 about 7 ms, zero failures. The p=20 fit time printed as 0.0 s, which is unexplained and not investigated. Extrapolation to n_boot=999 at p=40 is roughly 7 s; this is the Gaussian closed-form route only and says nothing about R's runtime.
Mutation re-run on the 211-assertion file: 29 of 30 mutants killed; the survivor N6 (attrition `maximum` instead of `minimum`) led to one added assertion in testset 5, after which the file is 212 assertions; re-run on Totoro (same clean instantiated project): 212 pass / 212 total, 52.8 s. The N6-killing assertion was checked against N6 only in a scratch copy of the test file, not by re-running all 30 mutants.
