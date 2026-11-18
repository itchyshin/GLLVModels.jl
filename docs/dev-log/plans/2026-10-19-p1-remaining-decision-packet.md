# Remaining P1 decision packet

Status: recommendations and retained evidence, not new signatures or row admissions.
Historical baseline: main `d9afa090c90eb1f71299df6e25129d81f1606add`, X2 298/317.
Candidate `7670e106b25e02a1ee2c1ff7dcdaf0e2a8927238` now passes the exact
seven-row checkpoint at X2 304/317 and C2 290/297, leaving these thirteen rows open.
GitHub main advanced to `8ca3eb76ecbea4edb78aed308bc3868fdbcc71e0`; the new
main has not been measured here. Candidate evidence remains bound to its own ref. Five F-capability gates also remain open; see the [local checkpoint report](<../after-task/2026-10-19-p1-c6-approved.md>).

## Decisions already received

The [C6 approval record](<../decisions/2026-10-17-p1-c6-and-c3-c4-contract-approval.md>)
and [TOML](<../core070/true-parity-latest/c6-approved-decisions-2026-10-17.toml>) preserve seventeen individually approved Julia
extras, eleven scoped names and nine evidence holds. The three optimizer exports
from #939 use the earlier documented-extra rule, separately from the seventeen.
The joint phylogenetic-signal contract has implementation approval only: explicit
species selector, point estimates, CI refusal, and shared Gaussian residual mode.
The four returned-fit diagnostics are approved only for ordinal, spider, beetle
and fungi, sequentially, with one Julia/BLAS thread and a 30-minute stop per fit.
Ordinal uses central finite differences with h=1e-5, inner maxiter=100 and
tol=1e-9; the three C4 fits use analytic gradients. Reconstruction tolerance is
exactly 0.0. No Newton polish, slices, urban input, row admission or merge is included.

## Exact thirteen-row map

| Row | Evidence and blocker | Recommended next action |
| --- | --- | --- |
| [`family/ORDINAL-LOGIT-RSZ`](<../core070/true-parity-latest/receipts/family/campaign/ORDINAL-LOGIT-RSZ.json>) | FAIL. Approved returned-fit capture has not run. The retained runner admits more cells/tolerances/threads than approved. | Obtain runner-owner release; repair guards and tests to the exact approved four-fit contract, then capture this cell. |
| [`data/RD-SPIDER-NB2`](<../core070/true-parity-latest/receipts/data/campaign/RD-SPIDER-NB2.json>) | FAIL. Julia `pd_hessian=false`; R `pdHess=true`. Same uncaptured four-fit contract. | Same released, bounded objective and gradient capture; no new Hessian calculation is approved. |
| [`data/RD-BEETLE-NB2`](<../core070/true-parity-latest/receipts/data/campaign/RD-BEETLE-NB2.json>) | FAIL. R `pdHess=false`, gradient 0.00295314; several comparisons fail. Julia gradient/Hessian not retained. Flat dispersion is a hypothesis. | Use the approved objective and gradient capture to check reconstruction and stationarity. Distinguishing flatness, loading degeneracy and negative curvature requires a separately approved diagnostic. |
| [`data/RD-FUNGI-BINOMIAL`](<../core070/true-parity-latest/receipts/data/campaign/RD-FUNGI-BINOMIAL.json>) | FAIL. Same uncaptured four-fit contract. | Same released, bounded diagnostic. No tolerance changes. |
| [`data/RD-URBANISATION-BINOMIAL`](<../core070/true-parity-latest/receipts/data/campaign/RD-URBANISATION-BINOMIAL.json>) | FAIL despite close logLik: fixed effects, loading covariance and link predictor exceed bounds. Raw input/outputs are private and absent from the repo. | Keep held. Require an exact authorized private location and processing/output scope, then a separate cell contract and owner release. The four-fit approval excludes urban. |
| [`isdm/ISDM-HEADLINE-RSZ`](<../core070/true-parity-latest/receipts/isdm/campaign/ISDM-HEADLINE-RSZ.json>) | FAIL: prediction differences exceed 1e-6. Historical R gradient 0.0211624; Julia gradient/Hessian absent. No convergence-parity block in the legacy receipt. | Prepare a separate diagnostic contract for approval. This is not an implicitly approved fifth capture. A future N9 convergence block requires both finite gradients at most 1e-5. |
| [`family/FAMILY-00-IDENTITY`](<../core070/gaussian-bridge-boundary-evidence.json>) | PARTIAL. Retained bridge returns a changed model, df 5 versus required df 8, dropping the unique-residual companion. It is not an R refusal. | Re-measure native/formula legs; keep the changed-model bridge held while presenting a precise boundary/model contract for signature. Do not claim full-unique parity from the reduced bridge model. |
| [`namespace/export/extract_phylo_signal`](<../core070/true-parity-latest/receipts/namespace/cases/CORE070-NAMESPACE-EXTRACT-PHYLO-SIGNAL-NATIVE.json>) | REGISTRATION-ONLY. Approved implementation exists in multiple unlanded variants; later variant repairs shared-residual identifiability. No paired R numerical receipt. | Reconcile under owner release, retain the later identifiability repair, bind one source tree, and qualify the existing two-case/control proposal before measurement and separate admission. |
| [`namespace/S3method/simulate,gllvmTMB_multi`](<../core070/true-parity-latest/receipts/namespace/cases/CORE070-NAMESPACE-SIMULATE-MULTI-NATIVE.json>) | REGISTRATION-ONLY. No judged Monte Carlo contract. The 7/40 result belongs to POST-SIMULATE-DEFAULT. | Preregister model, estimands, streams, multiplicity, failure denominator and wrong-simulator control before the judged run. |
| [`namespace/export/bootstrap_Sigma`](<../core070/true-parity-latest/receipts/namespace/cases/CORE070-NAMESPACE-BOOTSTRAP-SIGMA-NATIVE.json>) | REGISTRATION-ONLY receipt. Main already has public `bootstrap_Sigma`, but its site-total covariance includes residual variance, whereas R targets a named random-effect tier. | Agree the target covariance tier and derive it from each Julia refit; test wrong-tier controls and preregister repeated-batch Monte Carlo bounds before judging. |
| [`postfit-policy/POST-SIMULATE-DEFAULT`](<../core070/true-parity-latest/receipts/postfit/cases/CORE070-POSTFIT-SIMULATE-DEFAULT-NATIVE.json>) | PARTIAL. 7/40 alternate Julia trials failed against one shared R realization. This is not a 17.5% unconditional false-rejection estimate. | Retain failures. Separate fitted-model mismatch from simulation noise, then seek approval for independent R/Julia outer batches under a new preregistration. |
| [`postfit-policy/POST-COEF-EMPTY`](<../core070/true-parity-latest/receipts/postfit/cases/CORE070-POSTFIT-COEF-EMPTY-NATIVE.json>) | NON-DISCRIMINATING. Synthetic R object versus real Julia no-X fit; only length 0 equals 0. An always-empty accessor passes. | Approve actual public no-X and positive-X fits, length pair [0,q], spurious-coefficient and always-empty controls; record type/names separately. No engine fix is presently established. |
| `namespace/export/animal_latent` | NOT-MEASURED. Current Julia matrix wrapper is not the R random-slope contract. | Approve a symbolic slope contract, two positive random-slope cases and a rank-4 refusal control. An intercept-only receipt cannot close this row. |

## Evidence limits and ownership

The current main scoreboard and case receipts are the measured status source.
The numeric diagnoses above use committed aggregate receipts; no private urban
input or per-observation output was inspected for this packet. Existing bounds
remain unchanged. The C3 N9 gradient bound must not be imposed automatically on
C4. Some old tolerance annotations still say PROPOSED and need
explicit provenance resolution.

The retained four-fit ledger says execution and review are unmet, and its owner
has not released the runner. Dirty owner files are preserved. The joint extractor
also has an unreleased dirty owner tree. Older 44/44 and later 105/105 focused
results belong to different retained variants; they do not establish a current
main result or a full-suite pass. The later report records a failed broader core
run. Reconcile and reverify rather than replaying the older patch blindly.

This packet supplies a decision map. It does not authorize private data access,
a new fifth diagnostic, changed convergence rules, changed classifications,
row admission, merging, or a public parity claim.

Retained implementation sources are local and unlanded: joint extractor variants
at `1a230eb51` (older implementation) and `7d6f088a7` (later focused tests and
identifiability repair), plus the dirty dedicated owner tree at `9074da8bd`.
They need exact source binding before their logged test counts can support a new
candidate. The animal-latent row has no receipt in the main case map.

Review status: the independent review verified the exact thirteen-row set and
approval boundaries. The [retained thirteen-row source audit](<../audits/2026-10-19-p1-thirteen-row-source-audit.md>) binds the producer findings to receipts and local evidence.
A detached independent review verified the original packet and all thirteen
primary source fingerprints. This filed version updates links and checkpoint
status; its review record preserves the original hashes. Landing remains open.
