# A3 re-measure sizing: P0 to P1 parity evidence

Read-only sizing study. P0 = gllvmTMB 0.7.0 @ `b4d5fee64def88bc768dda1f1f77c29b295edd86`. P1 = `9539352f66f2db2cc26b1c393e67212a359b60c9`. Built on draft PR #527 (branch `claude/true-parity-p1-carry`, commit `f884c6f21`), which found 0 of 306 required P0 rows carry cleanly to P1. No fits, jobs, edits, or commits were made while preparing this note.

## 1. How P0 evidence was produced (harness map)

Each family in `docs/dev-log/core070/required-source-case-map.json` (769 rows, `origin/main`) is executed by a fixed three-script pattern under `tools/`:

- `core070_<family>_batch.R` — R driver. Reads a frozen contract JSON/TOML under `docs/dev-log/core070/`, asserts `contract$reference_commit == "b4d5fee64..."`, calls the installed gllvmTMB oracle, writes `r-facts.json`.
- `core070_<family>_batch.jl` — Julia driver, same contract, calls `GLLVModels`, writes `julia-facts.json`.
- `core070_verify_<family>_batch.py` — compares R vs Julia facts against a declared tolerance, writes `receipt.json`.

Confirmed directly by reading the namespace triple (`tools/core070_namespace_1_batch.R:1-50`, `tools/core070_namespace_1_batch.jl:1-27`): the contract is `docs/dev-log/core070/namespace-1-batch-contract.json`, 48 `EXECUTABLE_NOW` cases, and this batch needs **no installed R package and no RCall** — it is a text scan of the pinned P0 R source tree (existence/registration parity only). Not every family is this cheap; families that compare fitted log-likelihoods/gradients (aghq, covariance, data, family, fit-input, isdm, masks-known, postfit, postfit-policy) do call the built R oracle.

CI: `.github/workflows/CI.yml`, job `test-parity` (advisory, `continue-on-error: true`). It checks out gllvmTMB at the frozen ref into `.unlazy/r-source` (`CI.yml:131`), pins R 4.5.3, builds an isolated source-pinned oracle with `tools/core070_build_oracle.py prepare|build|verify`, stages receipts, then runs `test/parity/runparity.jl` with `GLLVM_PARITY_TESTS=1 CORE070_PARITY_REQUIRED=1`, single-threaded BLAS/OMP/Julia. `test/parity/runparity.jl:1-50` is the only entry point (not included by `runtests.jl`) and fails closed if RCall/R is unavailable in required mode.

The oracle builder, `tools/core070_build_oracle.py`, hardcodes the P0 pin and three companion byte-hashes at lines 19-22:
```
REFERENCE = 'b4d5fee64def88bc768dda1f1f77c29b295edd86'
NAMESPACE = '9094613610789faab69c43195d3cfdafb2c7dfef284e6646b10dababa4fa132c'
SOURCE_TREE = 'f83545faa6543dbb1f64d64bbf5a9498adcdf036cc3da5851f269912698b1cc7'
ARCHIVE = '0c2f4323eb9fb19acccf039b8d57b4dd6bda82e2aa8b4a7bb712f36a64b022bc'
```

Per-family required-row counts (from `carry-scan-p1.md`'s status table on `origin/claude/true-parity-p1-carry`, summing DANGLING + PARTIAL_STALE_AT_P1 + NO_R_PINS + RETIRED — the pool the P0 ledger counted before the carry question applies; totals to 308, close to but not identical to the task's stated 306, a small residual in the ledger's own accounting not chased further here):

| Family | Required rows |
| --- | --- |
| namespace | 71 |
| postfit | 36 |
| inference | 63 |
| data | 28 |
| covariance | 17 |
| family | 21 |
| aghq | 21 |
| isdm | 20 |
| postfit-policy | 16 |
| masks-known | 9 |
| fit-input | 6 |

## 2. What must change to run at P1

**The P0 SHA is not centralized.** `git grep -n "b4d5fee64" -- '*.py' '*.jl' '*.R' '*.yml' '*.toml'` (excluding `.unlazy/`) returns well over 130 hits, including: `.github/workflows/CI.yml:131`; `tools/core070_build_oracle.py:19-22`; `tools/parity_oracle.py:10` (`FROZEN_GLLVMTMB_ORACLE`); `test/parity/parity_helpers.jl:18` (`_CORE070_REFERENCE_COMMIT`); and a `stopifnot(identical(contract$reference_commit, "b4d5fee64..."))` or Python equivalent baked into essentially every per-family batch/verify script individually (e.g. `tools/core070_aghq_batch.R:65`, `tools/core070_namespace_1_batch.R:46-50`, `tools/core070_postfit_policy_batch.R:68`, `tools/core070_wave6_conversion_batch.R:79`, `tools/core070_verify_*_batch.py`). There is no single constant to flip; moving to P1 means regenerating each family's frozen contract JSON/TOML with the new `reference_commit`, then updating each script's literal-equality guard to match.

**R library build at P1.** `tools/core070_build_oracle.py`'s `prepare`/`build`/`verify` subcommands need to run again against a fresh `git checkout gllvmTMB@9539352f6...`, and the `NAMESPACE`/`SOURCE_TREE`/`ARCHIVE` hash constants (lines 20-22) must be recomputed once the new archive exists — they currently assert the exact P0 bytes.

**API changed between the pins.** Direct `git diff` on the local `gllvmTMB` checkout (`/Users/z3437171/Dropbox/Github Local/gllvmTMB`) between the two SHAs: 305 commits, 74 of 129 R/+src files changed (73 of 125 `R/` files, 1 of 4 `src/` files; `src/gllvmTMB.cpp` alone gained 866 lines). `R/fit-multi.R` and `R/gllvmTMB.R` — the two files cited by the most P0 receipts — both changed. `NAMESPACE` diff: new S3 methods `AIC.gllvmTMB_multi`, `BIC.gllvmTMB_multi`, `anova.gllvmTMB_multi`, `extract_latent_scores.*` (4 classes), `print.anova.gllvmTMB_multi`, `print.gllvmTMB_ordination_uncertainty`, `print.gllvmTMB_select_lv`, `update.gllvmTMB_multi`; new export `animal_coef`; two exports retired, `.proportions_bootstrap_ci` and `.proportions_wald_ci` — matching the carry-scan's 4 `RETIRED` rows exactly. The branch's own tool, `tools/true_parity_carry_scan.py` (added at `f884c6f21` on `origin/claude/true-parity-p1-carry`), independently re-hashed every gllvmTMB file reachable from a case-map receipt and found 62 of 117 *cited* files changed — a narrower, receipt-reachable count than the 74-of-129 whole-package figure above, and the reason 0 rows carried.

## 3. Runtime evidence found (after-task reports; all Totoro, single Julia/BLAS/OMP thread, per AGENTS.md convention)

| Batch | Evidence | Cases | Wall time |
| --- | --- | --- | --- |
| aghq | `docs/dev-log/after-task/2026-08-31-core070-aghq-poisson.md:35` | predeclared budget | estimate 1-3 min, hard cap 300 s; "Final 316 pass" |
| aghq | `docs/dev-log/after-task/2026-08-31-core070-aghq-binomial.md:34` | 78 assertions | PASS, no exact seconds given, under 30-min gate |
| covariance | `docs/dev-log/after-task/2026-08-31-core070-covariance-formulas.md:35` | 18 (9 native + 9 formula) | 64.04 s |
| covariance | `docs/dev-log/after-task/2026-08-31-core070-covariance-modes.md:37-39` | 9/9 then 18/18 | 1.067 s, 0.917 s |
| data | `docs/dev-log/after-task/2026-08-30-core070-data-controls.md:22` | — | no run occurred (Totoro socket absent) |
| family | `docs/dev-log/after-task/2026-08-31-core070-family-formulas.md:27` | 1 (failing registration check) | 7.08 s — not a representative fit |
| fit-input | none found | — | — |
| inference | `docs/dev-log/after-task/2026-08-30-core070-inference-routing.md:16` | 98 checks | all under 1 s each, no fits (static control-flow tier) |
| isdm | none found; all 20 P0 receipts are DANGLING even at P0 | — | — |
| masks-known | `docs/dev-log/after-task/2026-08-31-core070-masks-known.md:34-38` | 17/17, 16/16 | 1.167 s, 0.917 s |
| namespace | none dedicated; script itself (`core070_namespace_1_batch.R`) is a text scan, no R install/fit | 48 | not separately timed, expected sub-second |
| postfit | `docs/dev-log/after-task/2026-08-30-core070-postfit-contract.md:16` | 29/29 probes | 0.366 s |
| postfit | `docs/dev-log/after-task/2026-08-30-core070-gaussian-fitted.md` (related, not identically scoped) | 42 assertions | 39.26 s |
| postfit-policy | none found | — | — |

No DRAC run was found for any P0 core070 batch. Every recorded run is Totoro CPU, single-threaded; several after-task reports state explicitly that no DRAC job was needed (e.g. covariance-formulas), and DRAC is reserved for anything crossing the 30-minute approval line.

## 4. Estimate for the P1 re-run

Basis: every recorded per-case time is 1-4 seconds; the only multi-case batch total on record is 64.04 s for 18 covariance cases. Extrapolating conservatively:

- **namespace (71 rows)**: confirmed Julia/text-only, no R oracle needed — seconds to low tens of seconds on either kohaku or a laptop. Do not send this to DRAC; array overhead would exceed the work.
- **masks-known (9 rows)**: basis 0.9-1.2 s/case — well under 30 s total.
- **postfit (36 rows)**: basis 0.366 s/29 probes and 39.26 s/42 assertions on a related batch — estimate 1-5 min.
- **covariance (17 rows)**: basis 64.04 s/18 cases and 0.9-1.1 s/case on the lighter sub-batch — estimate 2-10 min.
- **aghq (21 rows)**: basis a 300 s per-case hard cap historically not hit — estimate 2-15 min.
- **inference (63 rows)**: the only recorded P0 evidence for this family is a static, no-fit control-flow tier (under 1 min total for 98 checks). If the P1 rows are the same tier, expect under 1 minute, no R oracle needed; if any of the 63 are numeric-fit rows this evidence does not cover, treat those as **cannot-estimate**.
- **data (28 rows), family (21 rows), fit-input (6 rows), postfit-policy (16 rows)**: **cannot estimate** — data has no recorded run at all, family's only number is a failing-case check, fit-input and postfit-policy have no after-task timing at all. Propose a **2-case pre-run** for each on kohaku before committing to the full batch (e.g. one native + one formula case from `core070_data_batch.R`/`.jl` for data; the analogous pair for the others).
- **isdm (20 rows)**: no usable P0 timing (all 20 receipts are DANGLING even at P0). The carry-scan notes the separate iSDM port (PR #525) re-measures these natively; recommend deferring isdm re-measurement to that port rather than re-deriving a parallel harness here. If it must be done in this harness, pre-run 2 cases first.

**Total estimate, excluding isdm (deferred to PR #525) and treating namespace as a near-zero-cost Julia-only pass**: roughly **10-45 minutes** of actual compute on **kohaku at 8 vCPU, CPU-only** — dominated by aghq, covariance, and postfit, contingent on the pre-runs for data/family/fit-input/postfit-policy not revealing an outlier. On **DRAC** as one job array per batch (one case per array task), each array's wall time is close to its slowest single case (seconds to low minutes per the evidence above) plus queue wait; queue wait, not compute, has historically dominated DRAC cost for jobs this small. Given every recorded P0 case is a sub-90-second single-threaded fit, **kohaku is the better default host** for this re-run; DRAC only earns its overhead if the pre-runs reveal a batch that is materially heavier than anything seen at P0. This sizing is a wall-clock estimate only — it does not address whether P1's API/NAMESPACE changes (Section 2) change convergence behavior for any case; that is a correctness risk, not a timing one, and this file-hash-based sizing cannot see it.

**Rows re-measurable without R at all**: namespace (71 rows, confirmed by direct code read: registration-existence check against the pinned R source tree, no RCall, no built oracle). Inference's static control-flow tier (evidenced) is also R-source-text-only if its P1 rows repeat that tier, but this needs per-row confirmation since the case map may include fit-adjacent inference rows not covered by the one after-task report found. **Everything else** (aghq, covariance, data, family, fit-input, isdm, masks-known, postfit, postfit-policy) compares fitted R and Julia log-likelihoods/gradients and needs the installed P1 R oracle built by `tools/core070_build_oracle.py`.

## 5. Harness change plan (ordered PRs)

1. Re-pin infrastructure: update `tools/core070_build_oracle.py:19-22` and `CI.yml:131` to the P1 SHA and recomputed hashes, and introduce one shared pin source that every per-family script reads, replacing the ~130-file literal duplication found in Section 2.
2. Regenerate contracts: re-derive each family's frozen contract JSON/TOML against the P1 checkout, re-anchoring `R/<file>:<lines>` citations that moved (`R/fit-multi.R`, `R/gllvmTMB.R` both changed), and add case-map rows for the API diff (new S3 methods, `animal_coef`, the two retired `.proportions_*_ci` exports already tracked as RETIRED).
3. Build and verify the P1 R oracle once, on Totoro or kohaku (not DRAC), via `tools/core070_build_oracle.py prepare|build|verify`, and commit its build/source receipts under `docs/dev-log/core070/true-parity-latest/receipts/` (tracked, never `.unlazy/`).
4. Re-run the already-evidenced cheap batches first (namespace, masks-known, postfit-contract, covariance-modes) to land tracked receipts recording R value, Julia value, both pins, host, and versions, proving the tracked-receipt convention before scaling up.
5. Pre-run 2 cases each for data, family, fit-input, postfit-policy (and any fit-adjacent inference rows), then size and run the remaining full batches on kohaku, deferring isdm to the native iSDM port (PR #525).
