# True-parity gate-tier scoreboard (2026-09-25)

A per-row status table for the 32 rows of the signed gate-tier list
(`true-parity-gate-tier-2026-09-05.md`), built to answer the M5-A1 question of
whether each row is evidenced or dispositioned yet (`m2-slice-table-2026-09-05.md`).
Measured at `origin/main` commit `d9bc77412cc309339730e806c8192b7f6c7ca10f`
(2026-09-25), against the frozen oracle gllvmTMB 0.7.0
`b4d5fee64def88bc768dda1f1f77c29b295edd86`. It promotes nothing: no row here is
bound under T9, no maintainer sign-off is recorded by this table, and building
it is not itself the M5-A1 review. Every entry below cites a path that exists
on `origin/main` at that commit, a GitHub PR, or a GitHub issue. Where no such
citation was found, the status is OPEN.

Status legend:

- **EVIDENCED**: a retained receipt meets the row's own bar (tier definition
  in `true-parity-gate-tier-2026-09-05.md` "Tier key").
- **PARTIAL**: a receipt exists but misses part of the bar; the gap is named.
- **DISPOSITION-SIGNED**: a maintainer-signed disposition covers the row.
- **OPEN**: no retained receipt and no signed disposition were found.

## A: Paired second-order families (15 rows)

| Row id | Requires (short) | Status | Receipt / disposition | Blocking gap id(s) | Owner |
|---|---|---|---|---|---|
| A1 `family/GAUSSIAN-IDENTITY-1FO` | Gaussian + bare `latent()`, first-order paired logLik/estimates, NAT or BRG route | PARTIAL | Native route: `docs/dev-log/core070/second-order-prerun-2026-09-02.md`, `docs/dev-log/core070/t4-p6-gaussian-*-receipt-2026-09-05.json` (4 files). No bridge (BRG) receipt specific to bare Gaussian `latent()` was found distinct from A12's ordination model. | X-01 | This repo |
| A2 `family/GAUSSIAN-IDENTITY-2SO` | Gaussian SE/vcov/Wald CI, native route | PARTIAL | `docs/dev-log/core070/t4-p6-gaussian-p20-n500-K2-receipt-2026-09-05.json` and 3 sibling cells, all PASS. Each file's own `claim_boundary` field reads "NOT true-parity or gate-tier promotion"; never T9-bound to this row. | A-05 | This repo |
| A3 `family/GAUSSIAN-IDENTITY-RSZ` | p in {20,50}, n in {500,2000}, cond(H) recorded | PARTIAL | Same 4 T4 receipts; `native_condition_number` and `r_condition_number` both present in each JSON. Same non-promotion caveat as A2. | A-05 | This repo |
| A4 `family/POISSON-LOG-1FO` | Poisson + bare `latent()`, bridge route | OPEN | No bare-Poisson bridge receipt found. Current bridge logLik receipts on `main` cover lognormal and truncated-Poisson (PR #357) and a `binomial_probit` real-data scout only. | X-01 | This repo |
| A5 `family/POISSON-LOG-2SO` | Poisson SE/vcov/Wald CI, native route | PARTIAL | `docs/dev-log/core070/t4-p6-poisson-*-receipt-2026-09-05.json` (4 files), PASS with the same non-promotion `claim_boundary`. | A-05 | This repo |
| A6 `family/BINOMIAL-LOGIT-1FO` | Binomial-logit + bare `latent()`, bridge route | OPEN | No bare-Binomial-logit bridge receipt found (the urbanisation scout is `binomial_probit`, a different link). | X-01 | This repo |
| A7 `family/BINOMIAL-LOGIT-2SO` | Binomial-logit SE/vcov/Wald CI, native route | PARTIAL | Toy batch-1 each-own-optimum receipt (`docs/dev-log/core070/second-order-batch-out/`) and matched-coordinates pilot PASS (`docs/dev-log/core070/second-order-matched-pilot-batch1-20260905.md`, se_max_rel 4.2e-8). No realistic-size (RSZ) cell; Binomial is not in the T4 family set. | A-05 | This repo |
| A8 `family/BETA-LOGIT-1FO` | Beta-logit + bare `latent()`, bridge route | OPEN | No bare-Beta-logit bridge receipt found. | X-01 | This repo |
| A9 `family/BETA-LOGIT-2SO` | Beta-logit SE/vcov/Wald CI, native route | PARTIAL | Toy batch-1 each-own-optimum receipt exists; matched-coordinates tier explicitly BLOCKED (`docs/dev-log/core070/theta-map-disposition-2026-09-05.md`: R per-trait `log_phi_beta` vs Julia shared dispersion, no θ-map). Additionally, the default Beta grouped fitter this receipt used is under active correction: PR #483 (open, needs maintainer review) and issue #480/#482 found the fitter could report `converged = true` at a non-stationary point. Any existing Beta receipt may need re-measurement once #483 merges. | B-03, A-02 (see #480, #482, #483) | This repo, maintainer decision |
| A10 `family/NB2-LOG-1FO` | NB2-log + bare `latent()`, bridge route | OPEN | No bare-NB2-log bridge receipt found. | X-01 | This repo |
| A11 `family/NB2-LOG-2SO` | NB2-log SE/vcov/Wald CI, native route | PARTIAL | Toy batch-1 each-own-optimum PASS and T4 RSZ receipts PASS (`docs/dev-log/core070/t4-p6-nb2-*-receipt-2026-09-05.json`, 4 files, same non-promotion caveat). Matched-coordinates tier BLOCKED, same θ-map reason as A9. The underlying NB2 fitter changed under PR #478 (merged 2026-09-25, restart for fits stuck at the Poisson boundary) and issue #477 records that the small-data NB2 likelihood has several maxima that Laplace can rank up to about 0.3 off from AGHQ; existing receipts predate that fix and are not re-measured. | A-05, B-10 | This repo, compute campaign |
| A12 `covariance/COV-ORD-LATENT-BARE-1FO` | Ordinary bare-`latent()` ordination, first-order paired, bridge route | PARTIAL | `docs/dev-log/core070/latent-bare-model-evidence.json` (tracked on `origin/main`): `r_reference`, `native_julia`, `julia_formula` and `public_r_bridge` all converge and agree on logLik, beta, loading crossproduct and residual variance to 1e-7 or tighter. The four routes are compared at each route's own optimum; the tier key's stronger bar of "cross-objective identity in both directions" (each engine's fit evaluated under the other engine's objective) is not demonstrated here. The raw process/result files this summary cites (`process_receipt`, `result_path`) sit under `.unlazy/`, which is git-ignored and absent from `origin/main` (confirmed via `.gitignore:19`). | X-04 | This repo |
| A13 `covariance/COV-ORD-LATENT-BARE-RSZ` | Same model at realistic size (p>=20, n>=500) | OPEN | The ledger's `covariance/COV-ORD-LATENT-BARE` row (`required-source-case-map.json`) carries only one executable case, `CORE070-COV-ORD-LATENT-BARE-FORMULA`; no RSZ-shaped case exists. Never bound to the T4 receipts, which cover families rather than this covariance structure. | A-05 | This repo, compute campaign |
| A14 `covariance/COV-PHYLO-LATENT-1FO` | `phylo_latent()` bare, first-order paired, after phylo transport Q1-Q4 | OPEN | Ledger row `covariance/COV-PHYLO-LATENT` (`required-source-case-map.json`) has zero executable case ids; its three planned cases (`STRUCT-PHY-TREE-RR`, `STRUCT-PHY-DENSE-RR`, `STRUCT-PHY-TREE-PROPTO`) are all `PREPARED_REFERENCE_NUMERICS_UNPAID`. | B-05 | This repo, maintainer decision |
| A15 `covariance/COV-PHYLO-LATENT-RSZ` | Same, realistic size | OPEN | Same ledger row and same unpaid planned cases as A14. | B-05 | This repo, maintainer decision |

## B: Grouping levels (4 rows)

| Row id | Requires (short) | Status | Receipt / disposition | Blocking gap id(s) | Owner |
|---|---|---|---|---|---|
| B1 `grouping-levels/UNIT-KWARG-NAME-PARITY` | `unit` exists on both engines under that name and pairs | PARTIAL | `test/test_destination_b_b1_unit_paired_fit.jl` runs in CI (`test/runtests.jl:262`) against `docs/dev-log/core070/destination-b-b1/frozen-r070-unit-gaussian-paired-receipt-20260910.json`. Gaussian only; `docs/src/grouped-models.md` states in the same file that "agreement with R and recovery of known simulated parameters have not yet been established" for the surface generally. | B-04 | This repo, maintainer decision |
| B2 `grouping-levels/UNIT-OBS-NONGAUSSIAN-KWARG` | `unit_obs`, beyond Gaussian-only `TwoLevelFit` | PARTIAL | `test/test_destination_b_b1_unit_obs_paired_fit.jl` in CI against a frozen-R070 Gaussian paired receipt in the same directory. Non-Gaussian pairing not found. | B-04 | This repo, maintainer decision |
| B3 `grouping-levels/CLUSTER-THIRD-AXIS-KWARG` | `cluster`, non-species third grouping | PARTIAL | `test/test_destination_b_b1_cluster_paired_fit.jl` in CI against `frozen-r070-cluster-gaussian-paired-receipt-20260910.json`. Gaussian only. | B-04 | This repo, maintainer decision |
| B4 `grouping-levels/CLUSTER2-INDEP-KWARG` | `cluster2`, second independent diagonal grouping | PARTIAL | `test/test_destination_b_b1_cluster2_paired_fit.jl` in CI against `frozen-r070-cluster2-gaussian-paired-receipt-20260910.json`; read directly, the test checks logLik, beta, sigma_eps and the cluster2 covariance block against the receipt. Gaussian only. | B-04 | This repo, maintainer decision |

All four B rows share one open decision: the Q1 ruling closed the joint numerical
gates (`B1-JOINT-STATIONARY`, `B1-JOINT-PAIR`) as a permanent interface limit, so
clause C5's word "pair" cannot be met at the strength originally written. Binding
these four Gaussian receipts to B1-B4 as the intended reading of "pair" is a
maintainer decision, not engineering (B-04).

## C: Real-data workflows (5 rows)

| Row id | Requires (short) | Status | Receipt / disposition | Blocking gap id(s) | Owner |
|---|---|---|---|---|---|
| C1 `real-data/URBANISATION-MAP-RD` | `urbanisation_map`, full eight-class `engine="julia"` acceptance | PARTIAL | Thin scout PASS: `docs/dev-log/core070/acc-bridge-urbanisation-receipt-2026-09-05.json`, logLik agreement 1.6e-7. `max_gradient` is not surfaced on the bridge object, which is short of the ACC-BRIDGE-GRADIENT class, and the scout used `n_init=1` against production's `n_init=5`. Full eight-class run not attempted. | C-01 | gllvmTMB lane, compute campaign |
| C2 `real-data/AVIAN-TRAIT-SCALES-RD` | `avian_trait_scales`, Gaussian/mixed continuous | OPEN | Recon only: `docs/dev-log/core070/real-data-model-inventory.md` catalogues the repo's models; no fit attempted. | C-01 | gllvmTMB lane, compute campaign |
| C3 `real-data/NEST-MORPHO-GLLVM-RD` | `nest_morpho_gllvm`, count/morphometric | OPEN | Same inventory doc, recon only. | C-01 | gllvmTMB lane, compute campaign |
| C4 `real-data/BIRDBASE-PCM-RD` | `BIRDBASE_pcm`, phylo-structured | OPEN | Same inventory doc, recon only; additionally needs the phylo transport prerequisite. | C-01, B-05 | gllvmTMB lane, compute campaign |
| C5 `real-data/META-WORKFLOW-SMOKE` | >=1 repo per family class with an end-to-end `engine="julia"` path | OPEN | Depends on C1-C4; only C1 has a scout-depth receipt, and it is not full-class. | C-01 | gllvmTMB lane, compute campaign |

`real-data/URBANISATION-MAP-RD` and the other three repos' promotion to full RD
depend on gllvmTMB PR #1236 (open, draft, marked CONFLICTING as of this check,
0 checks configured) landing or being replaced; that PR is on the gllvmTMB
repository, which this work does not touch.

## D: Bridge spine + user-facing extractors (8 rows)

| Row id | Requires (short) | Status | Receipt / disposition | Blocking gap id(s) | Owner |
|---|---|---|---|---|---|
| D1 `bridge/ACC-CLASS-RECEIPT-TEMPLATE` | Bridge-eligible tag + ACC receipt template | OPEN | Design note landed (`docs/dev-log/core070/bridge-eligible-row-tag-design-2026-09-05.md`), but its own "Implementation deferral" table lists the `bridge_eligible` field in `tools/parity_ledger.py` as not yet built, and the row needs real-data ACC receipts that do not yet exist (C-01). | A-07, C-01 | This repo |
| D2 `extractor/PREDICT-INSAMPLE-LINK` | `predict()` in-sample, link scale | OPEN | Measured in the wave-7 conversion batch (`docs/dev-log/core070/wave7-conversion-notes.md`, max_abs_diff 5.7e-6), but the install used was a live local **0.7.1** R build rather than the pinned 0.7.0 oracle. The gate-tier doc's own note reads "Wave-7 measured, not parity gate, disposition only if demoted", and clause T3 (`true-parity-decision-map.md`) already excludes predict/fitted/residuals from the claim. No demote disposition has been written. | A-07 | Maintainer decision |
| D3 `extractor/SUMMARY-FIXED-EFFECTS` | `summary()` fixed-effects table, native | PARTIAL | No receipt through the actual `summary()` extractor was found. The fixed-effect point estimates it would report are implicitly paired inside the second-order cells (`tools/core070_second_order/cells.jl`, `beta_idx_jl` vs `r_beta_idx = 'b_fix'`), but no row-specific receipt binds this. | A-07 | This repo |
| D4 `extractor/VCov-FIXED-BLOCK` | `vcov()` fixed block, second-order | PARTIAL | Same situation as D3: T4 receipts carry `vcov_frobenius_relative_delta` (e.g. `t4-p6-gaussian-p20-n500-K2-receipt-2026-09-05.json`), which is the underlying quantity, but the row itself was never bound. | A-07 | This repo |
| D5 `extractor/CONFINT-WALD-LINK` | Wald CI endpoints, second-order | PARTIAL | Same pattern: T4 receipts carry `ci_endpoint_max_delta`, not bound to this row. | A-07 | This repo |
| D6 `fit-input/FIT-GAUSSIAN-CORE` | Gaussian fit-input contract | PARTIAL | `docs/dev-log/core070/fit-input-contract.md`: 14 of 14 prepared-input expectations pass, but the document states this explicitly: "prepared-input evidence, not fitted parity" and "Julia calls and numerical acceptance tolerances remain UNRESOLVED/UNPAID." | A-07 | This repo |
| D7 `fit-input/FIT-LAPLACE-CORE` | Non-Gaussian Laplace fit-input, six dense-Laplace families | PARTIAL | Same `fit-input-contract.md`, same prepared-input-only caveat. | A-07 | This repo |
| D8 `parity/REVERSE-GAP-DISPOSITION` | Reverse gap list, tool-produced, written decision per item | PARTIAL | `python3 tools/parity_ledger.py` runs and reports `FORWARD=62 REVERSE=91` on this checkout, confirming the tool exists and produces the list. Of the 91 Julia-ahead exports, `tools/parity_ledger.py` itself reports them as "genuinely ahead, unclassified"; no per-item written class exists yet, so clause C6's "each item a written decision" is not met. | A-09 | This repo |

## Count summary

| Status | Count |
|---|---|
| EVIDENCED | 0 |
| PARTIAL | 19 |
| DISPOSITION-SIGNED | 0 |
| OPEN | 13 |
| **Total** | **32** |

Zero rows are EVIDENCED and zero are DISPOSITION-SIGNED, so zero of 32 rows are
promoted, consistent with the programme's own accounting that no gate-tier row
has gone through a T9 draft PR since the list was signed on 2026-09-05.

## The 42-vs-32 count

`true-parity-gate-tier-2026-09-05.md` states a row count of 42 twice (its
"Row count" line near the top and its "Signed row count" field in the sign-off
block at the bottom), but its own tables enumerate exactly 32 rows: A1-A15
(15), B1-B4 (4), C1-C5 (5), D1-D8 (8). A companion decision file,
`docs/dev-log/decisions/destination-b-scope-reconciliation.md`, already
reconciles the gate tier to 32 rows, so the discrepancy is a stale count left
in the original document's header and sign-off block, not a missing set of 10
rows. Proposed correction, for the maintainer to apply directly to the signed
document (this table does not edit it): change "42 proposed gate-tier rows"
to "32 proposed gate-tier rows" on line 12, and "Signed row count | **42**"
to "Signed row count | **32**" on line 126, each with a dated note that the
scope reconciliation in `destination-b-scope-reconciliation.md` is the
authority for the correction.

## Recent evidence considered

- PR #474 (merged 2026-09-24): the second-order receipt tooling now refuses
  to load gllvmTMB from an unnamed library when `GLLVM_PARITY_R_LIBS` is
  set. Relevant to the provenance of every second-order receipt cited above
  (A2, A3, A5, A7, A9, A11).
- PR #475 (merged 2026-09-25): fixes the Julia reference version (1.10) and
  stores the NATIVE-06 NB2 fixture as data rather than regenerating it from a
  seed; also the source of the `r_gradient_max`-only recording for NATIVE-10.
  Neither NATIVE-06 nor NATIVE-10 nor NATIVE-12 is a gate-tier row; they are
  #323 frozen-contract cells, noted here because they share the NB2 and
  Student-t fitters that A9/A11 depend on.
- PR #478 (merged 2026-09-25): adds a restart for grouped NB2 fits stuck at
  the Poisson boundary, changing the fitter behind A11's existing receipts
  (see A11 above).
- PR #481 (open, needs maintainer review): fixes a Gamma grouped-fit inner
  mode search that could report `converged = true` 43 log-likelihood units
  below the optimum (issue #479). Gamma is not one of the five gate-tier
  batch-1 families (Gaussian, Poisson-log, Binomial-logit, Beta-logit,
  NB2-log), so it does not gate any of the 32 rows directly; it shares the
  same per-site mode-search class of defect as the Beta fix below.
- PR #483 (open, needs maintainer review): fixes the Beta grouped fitter's
  convergence flag (issue #480), which directly bears on A8/A9 (see A9
  above). Its own body names a follow-on, issue #482: the Beta inner mode
  search can still oscillate and leave cliffs in the objective, so a fit can
  honestly report `converged = false` even after #483 lands.
- The three #323 holdouts NATIVE-06, NATIVE-10 and NATIVE-12 were diagnosed
  in `docs/dev-log/decisions/2026-09-24-parity-reference-julia-and-fixture-pins.md`
  and `docs/dev-log/after-task/2026-09-24-totoro-323-track-a-receipt.md` as a
  genuine boundary case (NATIVE-06) plus R-side numerics that depend on the
  reference platform at large dispersion or nu (NATIVE-10, NATIVE-12), not
  Julia defects. They still fail as written and need a frozen-contract
  revision, which is maintainer-gated; they are not gate-tier rows and are
  listed here only because A9 and A11 share fitters, and A2/A3/A5/A11 share
  oracle-build concerns, with them.
- **Post-drafting update:** PR #481 and PR #483 (both listed above as "open,
  needs maintainer review") merged 2026-09-25, after this scoreboard was
  drafted. A8/A9's Beta receipts still await re-measurement against the
  landed #483 fix; no such re-measurement is recorded in this document or
  elsewhere as of this note. This does not change any row's status above.
