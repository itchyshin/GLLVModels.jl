# Review: PR #587 — isdm rows re-measured at gllvmTMB P1 (draft)

Reviewed range: `0fb2b9411..d1eb87947` (branch `claude/true-parity-p1-isdm`, stacked on #584).
Reviewer worktree: detached at `d1eb87947` under `/Users/z3437171/local-scratch/lanes/GLLVM.jl-review-587` (removed after review).
Date: 2026-09-27. Nothing on the branch was edited, pushed, commented on, or merged.

## Verdict

**Acceptable as non-binding evidence; no blocking findings.** The 20 isdm receipts reproduce
byte-for-byte (modulo timestamp/elapsed) from a fresh run at P1, the contract and receipt
`--check`s are current, both checkers (main and #561's) report `bound=0 free=20` so no row can
bind, the P0 path still runs and verifies, the pin gates refuse every tamper I tried that touches a
checked field, and the diff stays inside `tools/` + `docs/dev-log/` with no host path or agent
handle. The PR body's claims are accurate; the one place its wording overreaches slightly is
"19 of these *exact* predicates" for the #546 Julia twins (finding 4). Findings 5–7 are minor
and none is introduced by this PR alone.

## Numbered findings (evidence in each)

1. **Batch reproduces at P1 (~1 s).** Extracted the P1 source tree from
   `/Users/z3437171/local-scratch/a3cov-oracle/source/gllvmTMB-core070.tar` (sha256
   `73b665b2…` = pins.toml `[P1].archive_sha256`) and ran
   `GLLVM_PARITY_PIN=P1 CORE070_P1_ORACLE_LIBRARY=<oracle lib> Rscript --vanilla tools/core070_isdm_batch.R <tree> <dest>`
   → `CORE070_ISDM_BATCH_PASS`, wall 1.2 s. `raw.tsv` identical to the tracked file; the 20
   `(case_id, actual, passed)` triples identical; the only differing keys are `generated_at` and
   per-case `elapsed_seconds`; `receipt.json` differs only in `results_sha256` (consequence of the
   timestamp). Verifier on the rerun: `NEGATIVES_PASS 10`, `SOURCE_PIN_NEGATIVES_PASS 5`,
   `CORE070_ISDM_BATCH_VERIFIED 20`, exit 0.
   Note: the tracked `isdm-p1/` cannot be fed to the batch verifier as-is ("missing retained
   run") because `diagnostics.log` is not tracked; `verify.txt` is the record. It was empty
   (sha `e3b0c442…` is the empty file), so anyone re-verifying must recreate an empty file. Minor.

2. **Contract and receipt tools current.**
   `GLLVMTMB_DIR=<clone> python3 tools/core070_isdm_p1_contract.py --check` → `CORE070_ISDM_P1_CONTRACT_CURRENT`.
   `python3 tools/core070_isdm_p1_receipts.py --check` → `CORE070_ISDM_P1_RECEIPTS_CURRENT 20 case receipts, 20 rows`.
   The twin's `regeneration_log` records exactly what the PR body says: `R/isdm-sources.R` and
   `R/fit-multi.R` changed, `R/offset.R` unchanged; five of nine function bodies changed
   (`isdm_source`, `isdm_sources`, `.gll_isdm_observation_design`,
   `.gllvmTMB_isdm_declared_core`, `.align_mixed_family_list`), four identical. Cases: 20,
   `expected_case_count` 20, ids match the P0 contract.

3. **No row can bind under either checker.** With `PARITY_REF=FS PARITY_CASEMAP=…/case-map-isdm.json`:
   main's `tools/true_parity_check.mjs`: `C1 required=20 bound=0 free=20 … C1_NOT_MET`, `C8 rows=20 failing=all 20 :NOT_TWINNED_NOT_SIGNED`.
   #561's checker (`git show origin/claude/true-parity-p1-namespace-v2:tools/true_parity_check.mjs`, temp
   copy, deleted): `C1 required=20 bound=0 bound_numeric=0 bound_registration_only=0 bound_signed=0 free=20 … numeric_label_without_numeric_receipt=none …`, same C8. Matches the PR body verbatim.
   `node tools/test_true_parity_check.mjs`: all negative controls pass.

4. **Option A claim, verified against #546's actual files — holds by id, "exact" is generous.**
   - `git show origin/claude/isdm-build:test/fixtures/isdm/admission_p1.toml`: 20 ids, all `"TRUE"`,
     `gllvmtmb_sha` = P1, `fixture_sha256 e47b6658…` = this PR's contract `fixture_sha256`. Its
     `contract_sha256 1288c8b5…` is the **P0** contract file (`docs/dev-log/core070/isdm-batch-contract.json`,
     identical bytes on both branches), i.e. #546's R replay evaluates the same 20 expressions this
     PR carries verbatim into the P1 twin, but sources the functions from the installed namespace
     (`env <- new.env(parent = asNamespace("gllvmTMB"))`, `export_p1_fixtures.R:432`) rather than
     parsing them from source. So the "same 20 predicates, different route, agree on all 20"
     cross-check is real.
   - `test/parity/isdm_cases.jl` `_ADMISSION_JULIA` has exactly 19 keys (all but `ISDM-LEGACY`);
     the testset asserts `keys == setdiff(cases, LEGACY)` and `=== true` for each. Case-for-case
     mapping by id holds.
   - But the Julia predicates are hand-translated analogues, not the R expressions: e.g.
     `ISDM-WITHIN-TRAIT-ADMIT` is `isTRUE(.gllvmTMB_validate_family_scale_by_trait(…))` in R vs
     `_isdm_assert_trait_scale(…) === nothing` in Julia; `ISDM-ALIGN` tests `.align_mixed_family_list`
     re-ordering in R vs `isdm_sources` keyword-order preservation in Julia; `MASKED-ARM` uses `NA` vs
     `NaN`. Same intent, different mechanism in a few rows. The PR body's caveat that this is a boolean
     pairing is correct; I would soften "exact predicates" to "id-matched analogues".
   - I did **not** run #546's Julia tests; the "41/41" figure is taken from #546's PR body, not re-measured.

5. **Option B fixtures do exercise the admission path, on both sides (code read, not run).**
   R at P1: `gllvmTMB(family = isdm_sources(...))` reaches `.gllvmTMB_integrated_sources_contract`
   (`fit-multi.R:1471`), `.gll_isdm_observation_design` (`:3335`) and `gll_prepare_offset` (`:3351`).
   Julia on #546: `fit_isdm_gllvm(formula, data)` → `isdm_table` → `_isdm_declared_core`,
   `_isdm_assert_trait_scale`, `_isdm_observation_design`, `_isdm_prepare_offset`
   (`isdm_table.jl` lines ~331–347). So a fit agreeing means the guarded path ran, but as the PR body
   already says, it does not restate any specific predicate (rank = 5 for ALIASED, etc.). All four
   R fixture fits use `unique = FALSE` (`export_p1_fixtures.R:181–199`); the admission predicates do
   not depend on `unique`, so #558 is indeed orthogonal to Option A and only bears on Option B.
   I did not check whether the `srcform_*` fixtures actually contain an aliased column.

6. **Pin strictness — every checked field refuses, and no destination is left behind.**
   Tampered P1 tree (`R/offset.R` + 1 line, path containing a space) → `identical(digest, source_pins[[rel]]) is not TRUE`, no dest.
   P1 without `CORE070_P1_ORACLE_LIBRARY` → halted, no dest. P0 pin against the P1 tree → halted, no dest.
   `GLLVM_PARITY_PIN=P2` → runner and verifier both refuse. Library copy with tampered marker
   `reference_commit` → halted, no dest.
   **Gap (pre-existing, shared with #569's `core070_source_pin.R`, not new here):** tampering the
   marker's `installed_tree_sha256` (`5d48…` → `0d48…`) still gives `CORE070_ISDM_BATCH_PASS`; that key is
   copied into the receipt but is not in `SOURCE_PIN_KEYS`, and neither the verifier nor receipt
   `--check` compares the recorded `marker_sha256` with anything. Low risk (the four checked hashes
   plus version pin the build), but worth a one-line fix in the shared tool at some point.

7. **Receipt-tool tamper tests.** `raw.tsv` line 2 `PASS→FAIL` → `--check` STALE
   ("tracked receipt with no re-derived case", because the sha in `read_from` and the verdict both
   move). Case-map row `evidence_tier` → `numeric_pass` → STALE ("case-map rows differ …
   isdm/ISDM-ALIASED"). Case receipt `glvmodels_commit` off by one hex digit → STALE with the
   run-commit mismatch named. All three restored from backups; worktree clean afterwards.

8. **P0 path unchanged.** `git archive b4d5fee…` of the P0 tree from the local gllvmTMB clone, run
   with the pin unset → `CORE070_ISDM_BATCH_PASS`; default-pin verifier `NEGATIVES_PASS 10`,
   `SOURCE_PIN_NEGATIVES_PASS 4`, `VERIFIED 20`. P0 contract, `required-source-case-map.json`,
   `case-map.json` not in the diff.

9. **Scope and hygiene.** `git diff --name-only` touches only `tools/` (4 files) and
   `docs/dev-log/` (28 files): no `src/`, `Project.toml`, `.github/`, `GATES.md`, P0 evidence.
   No `/Users/…` string in any tracked isdm receipt, contract twin or case map (the marker path is
   `gllvmTMB/CORE070_SOURCE_PIN.toml`, relative). No `@Pat`/`@Rose`/other agent handles in the diff or
   PR body. The PR body ends with the required generation line.

10. **PR body honesty.** Counts (20/0/0/0/0/0), the seven `!admitted` rejection cases, the
    five-changed/four-identical function record, checker lines, runtime, and the "what is not covered"
    list all match what I measured or read. The five maintainer decisions are genuinely open and
    nothing in the case map pre-empts them (`disposition: null` on every row, LEGACY's D-296 note
    kept out of the field).

## Nits (no action required for a draft)

- `tools/core070_verify_isdm_batch.py`: P1's reference commit comes from `R_REF_PINS`, P0's stays a
  string literal; harmless asymmetry.
- `tools/core070_isdm_batch.R`: at P1 the runner adds the oracle library to `.libPaths()` purely so
  `find.package`/`packageVersion` resolve; the namespace is never loaded, which the comment says. Fine.

## What I did not check

- Julia: did not run `test/parity/test_core070_pin.jl` (27/27 claimed) or full `Pkg.test()`; no
  `src/` or test change in range, so I accepted the PR body's figure.
- Did not run #546's `P1-ISDM-ADMISSION-20` testset or its fits; Option A/B facts above come from
  reading #546's files at `origin/claude/isdm-build`, not from execution.
- Did not audit the other five family contract/receipt `--check`s the check-log lists as current
  (family, data, covariance, postfit, inference); out of range.
- Did not read the 20 case receipts individually beyond one (LEGACY) and the tamper targets; relied
  on `--check` re-derivation for the rest.
- Did not verify #546's srcform fixtures contain an aliased column (Option B ALIASED candidate).
- The stack base (#584 at `0fb2b9411`) was taken as given.
