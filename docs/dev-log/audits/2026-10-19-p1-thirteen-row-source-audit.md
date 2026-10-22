# Gauss: source audit of the exact thirteen remaining P1 rows

Date: 2026-10-19. Producer: Gauss bounded diagnostic lane. Version: v1.
Purpose: retain the source-backed producer findings so an independent reviewer can check coverage and claim support. This is a source/evidence audit, not a fresh final panel, signature, execution result, row admission, or closure claim.

## Scope and provenance

- Committed engine/scoreboard baseline: `d9afa090c90eb1f71299df6e25129d81f1606add` (`origin/main` observed during this audit). C6 tools worktree HEAD observed as `c62b877f06b22e30f2c16df38becca34d746f5c7`; those tools are not evidence that an engine row is closed.
- Primary repository root (M): `/Users/z3437171/.codex/worktrees/p1-c6-authorized/GLLVM.jl`.
- Retained older candidate root (C): `/private/tmp/gllvm-p1-current-main`, HEAD `1a230eb51b4d1845d5dc7938890c909aac05f85d`, dirty in other owners paths.
- Retained later extractor root (L): `/private/tmp/gllvm-p1-main-refresh`, HEAD observed `7d6f088a7eba8dbd2f6cccfe98e4a1011a67752f`; do not infer cleanliness or a new test pass from this reference.
- R reference inspected previously in this producer task: P1 `9539352f66f2db2cc26b1c393e67212a359b60c9`, read-only source through Git.
- No fits, simulations, jobs, remote-host contacts, private urban input, or per-observation urban outputs were inspected/run to produce this note. This note is the only authorized write.
- Numeric tables below are selected aggregate fields from committed receipt blobs. Historical results retain their recorded source/runtime; they are not runs at current main.
- Input inventory SHA-256: `7c217fcd90877ac505f1daf355c82404f3e468bb6a0d8d32936aa6c6bd0cba2b`.
- Inspected decision-packet SHA-256: `78f2a517e6da7438bc5559c5a12e73690a2460e5412b475b9ecc1fd2a6736f0e`.
- The inventory has 19 rows. The exact thirteen are the inventory minus the six first-checkpoint behavior rows listed below; the packet explicitly separates those six.

Excluded from this thirteen-row audit: `isdm/ISDM-COUNT`, `isdm/ISDM-EXTRA-SOURCE`, `isdm/ISDM-MISSING-IN-TRAIT`, `isdm/ISDM-MISSING-SOURCE`, `isdm/ISDM-WRAPPER-LAW`, `postfit/POSTFIT-SURFACE-check_auto_residual`.

## Exact row inventory and committed primary evidence

Paths in this table are relative to M and were read as `git show d9afa090...:<path>`. SHA-256 binds the complete committed aggregate receipt, not any private raw data.

| Exact source ID | Baseline status | Primary committed path | SHA-256 |
| --- | --- | --- | --- |
| `data/RD-URBANISATION-BINOMIAL` | FAIL | `docs/dev-log/core070/true-parity-latest/receipts/data/campaign/RD-URBANISATION-BINOMIAL.json` | `0225263bb9b82d1410a02d6f28d0b481a7ff6897d0ef629daea841b3990feacf` |
| `data/RD-SPIDER-NB2` | FAIL | `docs/dev-log/core070/true-parity-latest/receipts/data/campaign/RD-SPIDER-NB2.json` | `343e9f683a565f699c2814ada7112de88c99c9e31c105ef10e775aaf7e4fb877` |
| `data/RD-BEETLE-NB2` | FAIL | `docs/dev-log/core070/true-parity-latest/receipts/data/campaign/RD-BEETLE-NB2.json` | `89c1f28d53cc3c004fd24525639ba867801470b79c6c8dfc9dd6ebe4de1c1286` |
| `data/RD-FUNGI-BINOMIAL` | FAIL | `docs/dev-log/core070/true-parity-latest/receipts/data/campaign/RD-FUNGI-BINOMIAL.json` | `f32bec140c2c9cf9efd5daa0c6d294d834c12f67a876b3ae09b1a3c3d963524d` |
| `family/FAMILY-00-IDENTITY` | PARTIAL | `docs/dev-log/core070/true-parity-latest/receipts/family/cases/CORE070-FAMILY-00-IDENTITY-PUBLIC-R-BRIDGE.json` | `302d7730d061595b2a7b2aebac10af171cd81baf7a38eceeb7a4a2ef7ea4fdc3` |
| `family/ORDINAL-LOGIT-RSZ` | FAIL | `docs/dev-log/core070/true-parity-latest/receipts/family/campaign/ORDINAL-LOGIT-RSZ.json` | `60678b72a7409e244c9f4241560bdcd34fa9df3970a0f7dc24c32da15b06d5ae` |
| `isdm/ISDM-HEADLINE-RSZ` | FAIL | `docs/dev-log/core070/true-parity-latest/receipts/isdm/campaign/ISDM-HEADLINE-RSZ.json` | `e2e4654e1692f85155a7c90f61517dd3a4ee827b147cb9214f609509b5b00502` |
| `namespace/S3method/simulate,gllvmTMB_multi` | REGISTRATION-ONLY | `docs/dev-log/core070/true-parity-latest/receipts/namespace/cases/CORE070-NAMESPACE-SIMULATE-MULTI-NATIVE.json` | `93df3767cd1b12256d9f29b7c4e4b85514cbfd2c09e8905984140a979a867ef4` |
| `namespace/export/animal_latent` | NOT-MEASURED | `docs/dev-log/core070/true-parity-latest/case-map-namespace.json` | `641e8095db0ff16b479434e0780276dfc63cb341e7be36457cb7ba6749516dbb` |
| `namespace/export/bootstrap_Sigma` | REGISTRATION-ONLY | `docs/dev-log/core070/true-parity-latest/receipts/namespace/cases/CORE070-NAMESPACE-BOOTSTRAP-SIGMA-NATIVE.json` | `fac4d8c12f63ea57fa7e61c2e10386be3c23d1cb3a035e1b1d8ad00a4da85de6` |
| `namespace/export/extract_phylo_signal` | REGISTRATION-ONLY | `docs/dev-log/core070/true-parity-latest/receipts/namespace/cases/CORE070-NAMESPACE-EXTRACT-PHYLO-SIGNAL-NATIVE.json` | `1a380afcedb153f8358fefd4186f59551c4b279897a1aeea3a92c7e7f5a0f837` |
| `postfit-policy/POST-COEF-EMPTY` | NON-DISCRIMINATING | `docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/CORE070-POSTFIT-COEF-EMPTY-NATIVE.json` | `72fdb2739730d2ca763a8b5d291442bd5847403b3e30ac49c82b0ae1a31bab80` |
| `postfit-policy/POST-SIMULATE-DEFAULT` | PARTIAL | `docs/dev-log/core070/true-parity-latest/receipts/julia-twins/postfit-twins/simulate_default.json` | `b330d932797c44479faa6396b06ecccd83ff19d0d0843cb1021e4edbcb7f9563` |

For `animal_latent`, the case-map entry is the evidence of the absent receipt/cases; no invented receipt is supplied. For FAMILY-00 the native/formula case receipts and Gaussian boundary summary are additional evidence discussed below.

## Numeric rows: measured comparisons and health

All numbers below are retained measurements, not hypotheses. Null/absent health fields mean unmeasured, never passing. Existing receipt bounds are reproduced without changing them. Some old annotations label bounds PROPOSED; preserve that provenance and resolve it explicitly before any future admission.

### data/RD-URBANISATION-BINOMIAL

Measured: receipt verdict `FAIL`; does not bind: beta: abs difference 0.000231344 > tolerance 0.0001; LLt: abs difference 0.00164848 > tolerance 0.0001; linear predictor on the training data (link scale): abs difference 0.000821878 > tolerance 1e-06.

| Quantity | Absolute difference | Bound | Within bound |
| --- | ---: | ---: | --- |
| logLik | 1.18859134091e-07 | 1e-06 | True |
| beta | 0.000231343939676 | 0.0001 | False |
| LLt | 0.00164847963896 | 0.0001 | False |
| linear predictor on the training data (link scale) | 0.000821877816414 | 1e-06 | False |

Health/source metadata:
- R: convergence=0; pdHess=True; max_abs_gradient=0.000182821891849816.
- julia: converged=True; iterations=115; pd_hessian=None; gllvmodels_commit=f318d9d9a59888ed815d8d52cff1ef80129f358e.
- `convergence_parity` block: absent.

Limit/hold: private raw input and raw engine outputs are absent from the repository; the receipt records hashes only. Larger-vector maxima cannot be independently rederived from committed raw evidence. No raw dataset or observation-level output was read here.
Inference: close logLik does not establish matching predictions/covariance or optimum. Earlier stopping is a hypothesis, not a demonstrated cause.
Recommendation and gate: keep held. The maintainer must separately locate/authorize the private input/raw-output location and processing/output scope, followed by a cell-specific method contract and runner-owner release. Urban is expressly excluded from the approved four captures. R gradient above 1e-5 is diagnostic here; do not apply C3 N9 automatically to C4.

### data/RD-SPIDER-NB2

Measured: receipt verdict `FAIL`; does not bind: logLik: abs difference 1.45732e-06 > tolerance 1e-06; LLt: abs difference 0.000161728 > tolerance 0.0001; linear predictor on the training data (link scale): abs difference 6.73994e-05 > tolerance 1e-06.

| Quantity | Absolute difference | Bound | Within bound |
| --- | ---: | ---: | --- |
| logLik | 1.45732315104e-06 | 1e-06 | False |
| beta | 4.95112861956e-05 | 0.0001 | True |
| LLt | 0.000161727789775 | 0.0001 | False |
| linear predictor on the training data (link scale) | 6.73994424503e-05 | 1e-06 | False |

Health/source metadata:
- R: convergence=0; pdHess=True; max_abs_gradient=0.00294377547171853.
- julia: converged=True; iterations=127; pd_hessian=False; gllvmodels_commit=896a0a2286c224fe9979d3cf7e7e2c6be27f9357.
- `convergence_parity` block: absent.

Inference: endpoint/stopping differences could explain the numeric discrepancies but are not proven by these receipts. Spider also has a measured Julia `pd_hessian=false`; this is distinct from the absent Julia Hessian fields on beetle/fungi.
Recommendation and gate: this exact cell is one of the four approved returned-fit captures. Wait for shared runner-owner release, correct guards/tests to the signed four-cell contract, then bind source/input/output identity before any approved capture. Capture remains diagnostic; later admission is separate.

### data/RD-BEETLE-NB2

Measured: receipt verdict `FAIL`; does not bind: R's Hessian is not positive definite (pdHess is false); logLik: abs difference 9.08069e-06 > tolerance 1e-06; beta: abs difference 0.000287141 > tolerance 0.0001; LLt: abs difference 0.00463293 > tolerance 0.0001; linear predictor on the training data (link scale): abs difference 0.000869455 > tolerance 1e-06.

| Quantity | Absolute difference | Bound | Within bound |
| --- | ---: | ---: | --- |
| logLik | 9.08069159777e-06 | 1e-06 | False |
| beta | 0.000287141201289 | 0.0001 | False |
| LLt | 0.00463293429518 | 0.0001 | False |
| linear predictor on the training data (link scale) | 0.000869454564207 | 1e-06 | False |

Health/source metadata:
- R: convergence=0; pdHess=False; max_abs_gradient=0.00295313866816116.
- julia: converged=True; iterations=398; pd_hessian=None; gllvmodels_commit=896a0a2286c224fe9979d3cf7e7e2c6be27f9357.
- `convergence_parity` block: absent.

Hypothesis: flat Poisson-limit dispersion directions may contribute; historical aggregate warning/boundary metadata flags groups 13, 28, 30, 37. No eigenmode attribution establishes flatness, loading degeneracy, or negative-curvature mechanism. R pdHess=false is measured; Julia non-PD curvature is not established.
Recommendation and gate: use the approved returned-fit objective/gradient capture only after runner release and guard correction. That capture can check reconstruction and endpoint gradients; it cannot establish the curvature mechanism. A later Hessian/eigenmode/dispersion-slice diagnostic would need a separate approval. No polishing is authorized.

### data/RD-FUNGI-BINOMIAL

Measured: receipt verdict `FAIL`; does not bind: beta: abs difference 0.000123933 > tolerance 0.0001; LLt: abs difference 0.000474438 > tolerance 0.0001; linear predictor on the training data (link scale): abs difference 0.000328279 > tolerance 1e-06.

| Quantity | Absolute difference | Bound | Within bound |
| --- | ---: | ---: | --- |
| logLik | 6.15127646597e-08 | 1e-06 | True |
| beta | 0.000123932525228 | 0.0001 | False |
| LLt | 0.000474437526716 | 0.0001 | False |
| linear predictor on the training data (link scale) | 0.000328279012452 | 1e-06 | False |

Health/source metadata:
- R: convergence=0; pdHess=True; max_abs_gradient=0.00386035726088885.
- julia: converged=True; iterations=154; pd_hessian=None; gllvmodels_commit=896a0a2286c224fe9979d3cf7e7e2c6be27f9357.
- `convergence_parity` block: absent.

Inference: endpoint/stopping differences could explain the numeric discrepancies but are not proven by these receipts. Do not infer unrecorded Julia gradient health from a converged flag.
Recommendation and gate: this exact cell is one of the four approved returned-fit captures. Wait for shared runner-owner release, correct guards/tests to the signed four-cell contract, then bind source/input/output identity before any approved capture. Capture remains diagnostic; later admission is separate.

### family/ORDINAL-LOGIT-RSZ

Measured: receipt verdict `FAIL`; does not bind: LLt: abs difference 0.000129891 > tolerance 0.0001.

| Quantity | Absolute difference | Bound | Within bound |
| --- | ---: | ---: | --- |
| logLik | 2.10353391594e-07 | 1e-06 | True |
| LLt | 0.000129891277598 | 0.0001 | False |
| thresholds | 5.58508966209e-06 | 0.0001 | True |

Health/source metadata:
- R: convergence=0; pdHess=True; max_abs_gradient=0.000430188546080568.
- julia: converged=True; iterations=56; pd_hessian=True; gllvmodels_commit=896a0a2286c224fe9979d3cf7e7e2c6be27f9357.
- `convergence_parity` block: absent.

Inference: endpoint/stopping differences could explain the numeric discrepancies but are not proven by these receipts. Do not infer unrecorded Julia gradient health from a converged flag.
Recommendation and gate: this exact cell is one of the four approved returned-fit captures. Wait for shared runner-owner release, correct guards/tests to the signed four-cell contract, then bind source/input/output identity before any approved capture. Capture remains diagnostic; later admission is separate.

### isdm/ISDM-HEADLINE-RSZ

Measured: receipt verdict `FAIL`; does not bind: predict_link: abs difference 7.69188e-05 > tolerance 1e-06; predict_response: abs difference 0.000149001 > tolerance 1e-06; predict_newdata_offset0_link: abs difference 7.69188e-05 > tolerance 1e-06; predict_newdata_offset0_response: abs difference 9.9334e-05 > tolerance 1e-06.

| Quantity | Absolute difference | Bound | Within bound |
| --- | ---: | ---: | --- |
| logLik | 4.09054337069e-07 | 1e-06 | True |
| beta | 1.11815521821e-05 | 0.0001 | True |
| predict_link | 7.69188157492e-05 | 1e-06 | False |
| predict_response | 0.000149001010115 | 1e-06 | False |
| predict_newdata_offset0_link | 7.69188157492e-05 | 1e-06 | False |
| predict_newdata_offset0_response | 9.93340067552e-05 | 1e-06 | False |

Health/source metadata:
- R: convergence=0; pdHess=True; max_abs_gradient=0.0211623821882103.
- julia: converged=True; iterations=97; pd_hessian=None; gllvmodels_commit=896a0a2286c224fe9979d3cf7e7e2c6be27f9357.
- `convergence_parity` block: absent.

Limit: the synthetic input and compressed raw artifacts are committed (paths recorded in the receipt); only aggregate receipt fields and contract/source metadata were used here. Julia stationarity is unmeasured. R convergence code 0 and pdHess do not prove max-gradient stationarity.
Rule: main `GATES.md`, N9 section, says the 1e-5 rule applies only to C3 receipts carrying `convergence_parity`; this historical receipt has no block. A future block would require both finite max-absolute gradients at most 1e-5, and this historical R gradient would fail.
Recommendation and gate: prepare a separate bounded iSDM diagnostic contract for approval: exact input/source, returned-coordinate reconstruction, derivative settings, compared point, caps, and retained health. This is not an implicitly approved fifth capture. No Newton polish or tolerance change is authorized; shared runner paths still require owner release.

## family/FAMILY-00-IDENTITY

Measured: the P1 bridge case is `NOT_EXECUTED`; native and formula case receipts also retain FAIL runner verdicts. Their numeric objective/logLik differences are about 3.4e-13 to 6.8e-13, but those numbers do not erase the recorded runner failure. The retained Gaussian run failed the random-effect-name expectation before the source assertion was corrected to `["z_B"]`; see C `test/parity/test_gaussian_original_required.jl:50-57` and its retained run metadata.
Additional inspected evidence: main `docs/dev-log/core070/gaussian-bridge-boundary-evidence.json` records `VERIFIED_REFERENCE_MODEL_CHANGE_NOT_PARITY`, df=5 versus required df=8, zero mean, reduced logLik -581.668831823926 and common residual SD 0.697282597486069. `tools/core070_gaussian_bridge_boundary.R` and verifier test automatic unique handling against explicit unique=false; the frozen P1 `R/julia-bridge.R:3617-3643` removes the auto Psi companion and warns.
Limit: this boundary measurement is historical, not a fresh P1 run at d9. N1 accepts the specified R refusal receipt, not a successful return of a changed model. Main GATES contains loose old prose about setting the case aside; the actual receipt conditions and current held row govern.
Recommendation and gate: reuse/correctly remeasure the native/formula legs after exact source binding. Keep the bridge held until the maintainer chooses preservation of the required unique model or an explicit scope/disposition. A unique=false reduced-model twin is narrower evidence and cannot silently replace the required case. No R engine changes or new measurements are made here.

## namespace/S3method/simulate,gllvmTMB_multi

Measured: its committed case receipt is Tier 0 registration only: pinned R S3 registration/definition and Julia callable/exported symbol exist. Semantics and numeric output were not compared. No seed-sensitivity result has been measured for this namespace row. The 7/40 finding belongs to POST-SIMULATE-DEFAULT, not this row.
Source inspected: P1 `R/methods-gllvmTMB.R:1523-1643` defaults condition_on_RE=false and redraws supported RE tiers; C `src/simulate_fit.jl:95-125` redraws latent normal scores for Poisson fits. This source alignment is scoped and not a blanket all-family parity certificate.
Recommendation and gate: under N4, freeze an aligned model/mean/covariance/default-simulation contract, independent streams, all compared moments, sample denominators, multiplicity, failure accounting and a wrong-simulator control before the judged run. A narrow receipt needs an explicit scope; do not borrow the held POST default receipt to bind this export.

## namespace/export/bootstrap_Sigma

Measured: committed case receipt is registration only, mapping R bootstrap_Sigma to Julia bootstrap_ci. No bootstrap seed/endpoint numeric comparison was judged for this row.
Source inspected: P1 `R/bootstrap-sigma.R:202-320,334-448` simulates, refits and extracts named tier summaries. The historical registration maps the row to bootstrap_ci; C `src/confint_bootstrap.jl:295-329,419-504` returns parameter intervals/packed refit parameters. However d9 also already exports a covariance-bootstrap wrapper: `src/confint_derived_wald.jl:913-945` implements bootstrap_Sigma(level=:unit) by calling bootstrap_ci_derived(fb -> sigma_y_site(fb)[i,j]) separately per upper-triangle entry. Main `src/confint_derived.jl:178-198` defines sigma_y_site to include B, W where present and observation residual variance, excluding phylogenetic covariance. Thus an existing API is available, but the actual public R B-tier and Julia site-total diagonal targets differ. Do not diagnose a missing covariance-bootstrap API or presume wrapper equality.
Existing unsigned options: C `docs/dev-log/plans/2026-10-07-p1-namespace-mc-contracts.md`, Proposal B, gives three alternatives: public-wrapper off-diagonal-only comparison; full B-tier comparison with a Julia derived bootstrap and separate public-wrapper evidence; or full site-total comparison with a custom R refit summary plus separate R wrapper evidence. None is signed; the original mismatched wrapper calls must not be run as a numerical parity test.

Recommendation and gate: define the exact tier covariance from every retained Julia refit before percentile calculation. Specify failure denominators, interpolation, finite-draw/endpoint precision and wrong-tier controls. Preregister independent repeated bootstrap batches and a bounded MC rule under N4. R unit-tier covariance and total response covariance must not be presumed equal. The row remains registration-only until a scoped measured receipt and admission review.

## postfit-policy/POST-SIMULATE-DEFAULT

Measured: historical twin receipt `receipts/julia-twins/postfit-twins/simulate_default.json` passes at the recorded seeds: B=400, 25 moments, maximum difference/tolerance ratio 0.962, conditional-draw control failing 10/25 moments. The owner `tools/core070_postfit_p1_receipts.py:337-345` deliberately holds it after a reported 7/40 alternate Julia streams failed. Main GATES N4 preserves that hold.
Uncertainty: the inspected retained packet contains no seed-level raw audit for all 40 alternatives. They reused one fixed R realization, so rejection events share the R reference. 7/40 is sensitivity evidence, not an unconditional 17.5% false-rejection estimate or proof of a simulator defect.
Source/contract: C `test/mc_simulate_helpers_p1.jl:63-78`, `test/test_mc_simulate_p1.jl:28-40`, and `test/fixtures/gen_mc_simulate_p1.R:70-94` retain 20 mean/covariance statistics on B draws plus five pair-difference statistics on B/2 independent pairs. The stored fixture lacks R fitted intercepts/loadings and individual replicate statistics. C `docs/dev-log/plans/2026-10-07-p1-namespace-mc-contracts.md` proposes independent outer pairs; it is a proposal, not a signed result.
Recommendation and gate: recover the old per-seed evidence and compare fitted intercepts/LLt before diagnosing simulator behavior. If doing analytic Poisson-lognormal moment checks, account for implementation clamp effects. Retain individual statistics in independently refreshed R/Julia outer batches. A replacement twin needs a new preregistration and maintainer agreement (C handover section 5.8); preserve the old rule/results.

## namespace/export/animal_latent

Measured: main case map/scoreboard records no executable P1 cases or receipt. C `src/animal_latent.jl:36-74` is a Gaussian matrix wrapper with fixed relatedness, unique=false, K=1/K_phy=d. Its accepted 2026-09-14 identity excludes slopes, pedigree parsing and formula/bridge parity.
Source: P1 `R/brms-sugar.R:3407-3444` recognizes animal_latent(1+x|id); `tests/testthat/test-animal-latent-slope-gaussian.R:1-17,82-95` describes separate rank-d intercept/slope covariance blocks with zero cross-column covariance. Existing P0 IDs do not constitute P1 measurement.
Recommendation and gate: obtain the signed symbolic public representation/slope contract: relatedness axis/order, predictor basis, per-column rank, cross-column covariance, mean/residual model and unique semantics. Reuse the existing proposal for two positive random-slope cases and a same-route rank-4 refusal, plus dropped-slope/wrong-relatedness controls. An intercept-only matrix-wrapper receipt cannot close the random-slope scope. Agents do not sign or disposition it.

## namespace/export/extract_phylo_signal

Main fact: d9 lacks the joint extractor and shared-residual joint mode. Its existing namespace case receipt is registration-only. The C6 tools commit is not engine landing.
Approved scope/source: C `docs/dev-log/decisions/2026-10-07-joint-phylo-signal-contract.md` authorizes implementation only: JointPhyloGroupedGaussianFit, required species_level, matching observation partition, point output, ci=true refusal, shared Gaussian residual option. V_eta includes phylogenetic variance (including unique) plus selected ordinary species variance, excludes site/residual; C2_non=0 for this diagonal ordinary representation.
Retained older implementation: C commit 47b1a78ab in HEAD 1a230 has source `src/joint_phylo_grouped_postfit.jl:143-163`, shared packing/reconstruction, and `test/test_joint_phylo_grouped_postfit.jl:78-118`. C after-task `2026-10-07-joint-phylo-signal-shared-residual.md:163-172` supersedes its earlier iterations=0 note with 44/44 focused assertions, optimizer-backed shared fit (160 iterations), gradient 9.6731871861e-5 at g_tol=2e-4. That is retained focused Julia evidence, not a new run or paired R receipt.
Retained later implementation: L `test/test_joint_phylo_signal.jl` and `docs/dev-log/after-task/2026-10-16-joint-phylo-signal-extractor.md` retain 35/35 extractor and 105/105 combined assertions plus docs-build success. This variant repairs a shared-residual identifiability bug: a single I_p residual tangent, with a rank 5 / five-label control versus trait-mode rank 5 / six labels. Tests also include phylogenetic-unique extraction, missing/wrong selectors and CI refusal. It changes some representation details (trait labels), so do not treat variants as identical.
Limits: the later report retains broader core failure 31,158 pass/85 fail/0 error/25 broken followed by focused unrelated test repairs, not a subsequent full pass. Dedicated owner tree `/Users/z3437171/.codex/worktrees/joint-phylo-extractor/GLLVM.jl` HEAD 9074da8bd was dirty in related engine/tests when inspected; preserve it. `/private/tmp/p1-extractor-{postfit,joint-fit,joint-covariance,joint-alias}-shard.log` and `/private/tmp/p1-extractor-focused.GATES.md` show historical focused passes but do not by themselves bind a current dirty source tree.
Recommendation and gate: reconcile existing variants under owner release; retain the later identifiability correction rather than blindly replay old 47b. Bind one exact source/input/runtime before measurement. C plan `2026-10-07-p1-extract-phylo-signal-contract.md:23-49` retains two positive cases (bare+species diagonal; phylogenetic unique+crossed site), wrong denominator/source allocation/tip mapping/alias controls, common-point likelihood/extraction and matched independent optima/cross-evaluation. Those paired R requirements and separate admission remain open. Implementation approval is not row-admission, merge, interval or parity approval.

## postfit-policy/POST-COEF-EMPTY

Measured/source: committed case receipt records PASS for length 0 = 0 but discriminating=false. R batch `tools/core070_postfit_policy_batch.R:175-184` parses the pinned function and evaluates a synthetic object list(X_fix_names=character(0)); Julia batch `.jl:241-263` uses a real no-X fit. Main `src/postfit.jl:799` already returns Float64[] for absent beta. The current comparison does not establish public R dispatch, type or names.
Rule: D-319 N2 permits integer equality; it does not waive the shared degeneracy rule. A constant-empty/zero accessor still passes. C audit `docs/dev-log/audits/2026-10-07-p1-post-coef-empty-diagnostic.md:5-15` explains why the separate near-zero named-coefficient case does not challenge this branch.
Recommendation and gate: no engine fix is established. Keep current receipt nonbinding. Obtain approval for actual public no-X and positive-X fits, length pair [0,q], spurious-coefficient and always-empty controls, with type/names recorded separately; or obtain a signed disposition. New case scope/binding treatment is a maintainer decision, not an agent classification.

## Shared four-capture approval and owner hold

- C `docs/dev-log/decisions/2026-10-17-p1-c6-and-c3-c4-contract-approval.md:29-58` and `docs/dev-log/plans/2026-10-17-p1-c3-c4-capture-method-decision.md` approve ordinal/spider/beetle/fungi only: reconstruction_atol=0.0, ordinal centralFD h=1e-5 / inner maxiter=100 / tol=1e-9, analytic three C4 gradients, one Julia/BLAS thread, sequential fresh output, 30-minute stop per fit, no CI/prediction/point replacement/DONE. No polish, Hessian/slices, private urban, fifth iSDM capture, row admission or merge is included.
- C `.unlazy/true-parity-p1-next/gates/leaf-c3-c4-diagnostics.md:13-30` leaves P2 raw captures and P3 independent review unmet; latest inspected parent ledger entries on 2026-10-19 retain owner hold.
- C `tools/true_parity/campaign/run_J.jl:16-59` still permits urban and nonnegative variable tolerance; `pilot_capture.jl:104-129` starts four child threads and bounds capture at 300 s, not the entire fit at 30 min. Captures omit required convergence/iteration/input fields. `test/test_campaign_pilot_capture.jl` is structural/toy coverage, not real-fit evidence, and admits urban/nonzero tolerance. These are gaps to correct only after release.
- Owner `/Users/z3437171/.codex/worktrees/p1-c3-c4-next/GLLVM.jl` at 0b6e930573 remained dirty in run_J/polish_J/run_R and tests when inspected. Quiet preflight/no lease is not an explicit release.
- No exact four-cell capture records were located in the inspected local evidence directories. `/private/tmp/gllvm-p1-ordinal-capture-20261017` contains input CSV only; `/private/tmp/p1-ordinal-capture` is empty. The other ordinal log is an R fit on a different small fixture, not the approved returned-fit capture. Behavioral capture bundles/scripts are different work.
- Do not widen tolerances to overcome any failure. Future C3 N9 blocks require exactly 1e-5; do not impose that bound on C4 automatically. A converged flag is not a measured gradient/Hessian.

## Reconciliation findings for the inspected decision packet

The inspected packet digest above already incorporates corrections 1 and 2 and acknowledges the existing public bootstrap_Sigma wrapper and its target mismatch; they are retained here so the reviewer can trace why those statements changed.

1. Beetle recommendation must say objective/gradient capture first, then separately approved curvature attribution. The existing exact four-fit approval cannot itself distinguish flatness/loading degeneracy/negative curvature.
2. Spider has retained Julia pd_hessian=false. Fungi and beetle have absent Julia Hessian fields; these are different evidence states.
3. Earlier iterations=0/43-test extractor prose is superseded by the older variant addendum 44/44; later 35/105 counts belong to a different retained variant. Do not combine them into a single fresh pass.
4. Inventory19 minus six behavior rows gives the exact 13. All thirteen have an explicit section above; animal_latent has no receipt and is mapped to the absence in its committed case-map entry.
5. The note does not resolve annotations saying PROPOSED, owner release, private-data scope, unavailable seed-level evidence, exact future source binding or unsigned admission decisions.

## Local artifacts used: source fingerprints

These fingerprints identify the inspected local evidence and specifically named committed sources only. They do not sign a contract, prove runtime success, or pin an entire dirty worktree.

| Local path | SHA-256 |
| --- | --- |
| `/private/tmp/gllvm-p1-current-main/docs/dev-log/plans/2026-10-17-p1-c3-c4-capture-method-decision.md` | `ced5d5e18293640199be00e5f2204a14fd629e24d4996d635f32657f531cf501` |
| `/private/tmp/gllvm-p1-current-main/docs/dev-log/decisions/2026-10-17-p1-c6-and-c3-c4-contract-approval.md` | `63611f273af4a7233f7cade796c64a163edb37eeda839cd38323735d39752e48` |
| `/private/tmp/gllvm-p1-current-main/.unlazy/true-parity-p1-next/gates/leaf-c3-c4-diagnostics.md` | `34b126c38b805b6afdabb24ab7dc5dbfce1b31316d44567eeb0c71f1970945e7` |
| `/private/tmp/gllvm-p1-current-main/tools/true_parity/campaign/run_J.jl` | `dc7bfd92a52aaf8abd73d16d260de55801497c59de883c018e7d099394f2b51a` |
| `/private/tmp/gllvm-p1-current-main/tools/true_parity/campaign/pilot_capture.jl` | `b61d118d2c2bc2fac5ba47dd63f85b200d9ce6ec1651c836ef3726416ed7585b` |
| `/private/tmp/gllvm-p1-current-main/test/test_campaign_pilot_capture.jl` | `d620233a9b3caae7261e09d200c0ef6ec58759371aeb90384803a949b5231540` |
| `/private/tmp/gllvm-p1-current-main/docs/dev-log/plans/2026-10-07-p1-namespace-mc-contracts.md` | `5039cba6d4416741e6526fddef6abe1f2564a979c7833747e5a40f60e6accf14` |
| `/private/tmp/gllvm-p1-current-main/docs/dev-log/audits/2026-10-07-p1-post-coef-empty-diagnostic.md` | `24b69551466d84dcdbdfefc5fdc9059b3028f402057033ce5985843006c8918c` |
| `/private/tmp/gllvm-p1-current-main/docs/dev-log/plans/2026-10-07-p1-extract-phylo-signal-contract.md` | `e2807b1bbc7fe2497f8be56eadd03b87dfc7f26284a7a1957945a28234624488` |
| `/private/tmp/gllvm-p1-current-main/docs/dev-log/after-task/2026-10-07-joint-phylo-signal-shared-residual.md` | `381daf5286538aa2066e5bbd02fa1195c454e03e1637864e1e7619ee155b7f5b` |
| `/private/tmp/gllvm-p1-main-refresh/docs/dev-log/after-task/2026-10-16-joint-phylo-signal-extractor.md` | `d24bf2ca71c19c6175cf6f62e220d41e69d04dc03fab60bf17dee6cb1f260274` |
| `/private/tmp/gllvm-p1-main-refresh/src/joint_phylo_grouped_postfit.jl` | `fc7d9e8f1eaf5e6a959126057d224943c4769b231a50ebfcd4d606fea5bf4293` |
| `/private/tmp/gllvm-p1-main-refresh/src/joint_covariance_identification.jl` | `fe60bec07b4d60f4ac191ca1473fb222602c116a015a46cc6144842f6501e07a` |
| `/private/tmp/gllvm-p1-main-refresh/test/test_joint_phylo_signal.jl` | `40a67315b08e28e01ffda5031b96bed4c5cd9ee33715005c7b7c43206cd24e7f` |

| `M@d9afa090c90eb1f71299df6e25129d81f1606add:src/confint_derived_wald.jl` | `4c54fc94ad8dc26b48dda178a85961b0549028975f1df9593d658128189d5410` |
| `M@d9afa090c90eb1f71299df6e25129d81f1606add:src/confint_derived.jl` | `488754eb4f2772f48b1fc0c4ba4d06717fdbacef933f430504f46d90aabe5997` |

## Audit completion and limits

Packet binding refresh, 2026-10-19: reread the current packet at SHA-256 `78f2a517e6da7438bc5559c5a12e73690a2460e5412b475b9ecc1fd2a6736f0e`. Its thirteen-row map, numeric/health statements, approval boundaries, recommendations, privacy limits and owner holds remain substantively consistent with the previously inspected packet `16403800f5599db2f12bb6b626b01ed55609be541d5267034aa4b4facbb1ebeb`. The changed ending adds the link to this audit and updates the independent-recheck status; it supplies no new measurement, signature or row admission. This refresh changes only this audit note, not the packet or source evidence. The independently reported primary-fingerprint review is not represented as a fresh final panel.

Producer coverage: 13/13 exact IDs. A read-only Python set check compared the exact row-section headings with the supplied 19-row inventory minus the six declared behavior exclusions: 13 sections, missing=[], extra=[]. It also verified that the current packet SHA-256 matches the digest above and that the urban privacy, spider Hessian and beetle capture-limit statements are present. Source/aggregate evidence and recommendation limits are explicit per row. This is a producer self-audit, not independent review. No fresh final-panel verdict is claimed.
Editorial self-review: technical internal audit note, whole v1; perceived formulaic style 3/10, moderate confidence. Located evidence: repeated Measured/Recommendation labels aid review but make the note mechanical; retained intentionally for traceability. Scientific gate checked for scoped inference and gradient-rule limits; factual gate checked against committed aggregate fields/local inspected sources; reference gate checked for path/ref identity. Full saved-prose assessment validator was not run: NOTASSESSED by that mechanical workflow. No universal correctness or runtime certificate is asserted.
