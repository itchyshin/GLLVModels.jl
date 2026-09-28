# Review: PR #571 — inference family re-measured at gllvmTMB P1

Reviewed range: `28d880acb..1f9a979f9` (commits `ec36c65bd`, `88d56d4c8`, `1f9a979f9`) on
`claude/true-parity-p1-inference`, stacked on #569 (`28d880acb`) on #567. Independent, adversarial;
worked in a detached worktree at `1f9a979f9` (removed afterwards). Nothing pushed, commented or edited
on the branch. Date: 2026-09-27.

## Verdict: NON-BLOCKING

The PR does what it says. The 63 case-map rows keep their P0 classification, case ids and disposition
byte-for-byte; 59 are correctly free (routing / error-class / partial tiers citing
`evidence.non_binding_receipts`); the 2 numeric rows carry a real, discriminating comparison block at the
harness's own tolerance; every new runner and verifier refuses a mistyped pin; the hash checks are real;
no engine, CI, gate or P0 evidence file is touched; no raw logs are committed; no agent handles. The
PR body is candid (it discloses the 011 collapse, the stale runner sha and the path-bound library).
The findings below are provenance and disclosure gaps, not evidence defects.

## Verified OK (with the evidence)

| Check | Result |
| --- | --- |
| `git diff --stat 28d880acb 1f9a979f9` | 45 files, all under `docs/dev-log/` and `tools/`. No `src/`, `Project.toml`, `.github/`, `GATES.md`, P0 contracts or P0 evidence. No `.log` files. |
| Tracked receipts vs raw runs (`/Users/z3437171/local-scratch/a3inf-runs/`) | `cmp` byte-identical for all 11 copied JSON/TSV files (wave2 Julia 3, wave2 R crosscheck 4, wave4 4). |
| Case map vs `docs/dev-log/core070/required-source-case-map.json` | 63 rows; classification (`compatibility_adapter` ×63), `executable_case_ids`, `disposition` identical for every row. Tiers: 45 `routing_control_flow`, 14 `reject_error_class`, 2 `partial_non_numeric_case`, 2 `numeric`. Only the 2 numeric rows cite `evidence.receipt`; the 61 free rows cite `evidence.non_binding_receipts`. |
| #561 checker (`92cf39571`, run from a temp copy) | `C1 required=63 bound=2 bound_numeric=2 ... free=61 ... numeric_recorded_diff_mismatch=none`, `C1_NOT_MET`; `C8 rows=63 failing=<61 rows>:NOT_TWINNED_NOT_SIGNED`, `C8_NOT_MET`. The 61 are all rows except CI-ROUTE-008/010. main's checker: `C1 required=63 bound=2 free=61`. Matches the PR body and check-log. `node tools/test_true_parity_check.mjs`: all negative controls pass. |
| 008/010 discriminating? (mutate a temp copy of the receipt, run #561 checker) | `julia_value[0] * 1.01` -> `bound=1`, `numeric_recorded_diff_mismatch=inference/CI-ROUTE-008`. All-constant `julia_value=[0.5]*8` -> bound=1. Stale `max_abs_diff=5e-4` with vectors intact -> bound=1. `verdict="FAIL"` -> `numeric_receipt_not_passed`. Yes, discriminating; the vectors are eight distinct interior values (0.15..0.84), not 1s or 1e-14s. |
| Generator discriminating? (`tools/core070_inference_p1_receipts.py` re-run in a scratch copy with #569's `julia-results.json` tampered) | +1% on one Julia value with harness `max_abs_diff` left stale -> `SystemExit: recomputed 0.00596 != harness 1.77e-06`. Same tamper with `max_abs_diff` made self-consistent -> 008 receipt `verdict: FAIL`, case-map row `numeric_fail`, counts `numeric_pass: 1, numeric_fail: 1`. |
| Tolerance 1e-3 origin | Per-case `tolerance: 0.001` on `CORE070-SURFCONV-INFERENCE-CI-ROUTE-008/010` in the **P0** `docs/dev-log/core070/surface-conversion-batch-contract.json`, carried verbatim into the P1 twin. Harness-declared, not builder-chosen. |
| `tools/core070_inference_routes_p1.R` vs `tools/core070_inference_routes.R` | `diff`: a 7-line header comment plus exactly one changed line (`'temporal.R'` appended to the parsed file list). "One-line change" is true. |
| Hash check of the 4 P1 R files real? | `p1_r_source_pins` match sha256 of `/Users/z3437171/local-scratch/a3cov-oracle/build/source/R/{mspl,fit-multi,z-confint-gllvmTMB,temporal}.R`. Tamper test: appended a comment to a copied `temporal.R`, ran `GLLVM_PARITY_PIN=P1 CORE070_P1_R_SOURCE_ROOT=<copy> Rscript --vanilla tools/core070_inference_batch.R . <dest>` -> `Error: identical(sha256_file(...), src_pins[[rel]]) is not TRUE`, halted. Untampered run: `64 rows; 64 R-side PASS`, `crosscheck_sha256 cdbdbc68...` identical to the tracked receipt. Same for the remainder runner (tampered `z-confint-gllvmTMB.R` -> `identical(digest, contract$source_pins[[rel]]) is not TRUE`, before any fit). |
| Mistyped pin refused? | `tools/core070_inference_batch.R` (P2): `GLLVM_PARITY_PIN must be P0 or P1, got 'P2'`. `tools/core070_inference_remainder_batch.R` (P2): same. `tools/core070_verify_inference_batch.py` (P2) and `..._remainder_batch.py` (p1x): parity_oracle message, exit 1. `GLLVM_PARITY_PIN=P1` without `CORE070_P1_R_SOURCE_ROOT`: refused. Default (unset) pin on the wave2 R runner falls through to the P0 retained-run path and fails loudly (`file.exists(results_path) is not TRUE`, since `.unlazy/` is absent), not silently. P0 code paths are unchanged in the diff apart from the pin switch. |
| Contract twins current and verbatim | `GLLVMTMB_DIR=<gllvmTMB clone> python3 tools/core070_inference_p1_contract.py --check` -> `CORE070_INFERENCE_P1_CONTRACTS_CURRENT`. Deep diff P0 vs P1: inference twin changes only `reference_commit`, `r_route_comparand.pins`; adds `p0_pins`, `p1_probe`, `p1_r_source_pins`, `p1_note`, `p0_contract*`, `p0_to_p1_r_function_diff`, `regeneration_log`; `rows` and `cases` identical. Remainder twin changes only `reference_commit`, `source_pins`; `cases` identical. |
| Verifiers at P1 on the raw runs | `GLLVM_PARITY_PIN=P1` batch verifier: `CORE070_INFERENCE_BATCH_FULLY_VERIFIED`; remainder verifier: `..._BATCH_VERIFIED 5 negative_controls 2`. Self-tests at P1 pass (5 + 5 negatives). Contract sha256 in both R receipts equals the tracked twins (`b41b4c43...`, `4f4540a6...`). |
| Wave2/wave4 evidence tiers | Wave2 Julia route tags vary across cases (profile 15, bootstrap 12, wald_packed 9, wald_derived 7, reject 2); R probe rows are `route_boundary` strings like `.confint_lambda:wald`. Wave4: R raises and message matches for all 14 (method, quantity) pairs; Julia `MethodError` for all. Neither carries a number; correctly free. |
| Agent handles | `git diff` scan for `@[A-Za-z]` in added lines: none. |

## Findings

### F1 (non-blocking, provenance). Wave5 receipts read #569's run but record no hash of what they read

`tools/core070_inference_p1_receipts.py` reads three tracked files from #569
(`receipts/postfit/surface-conversion-p1/{julia-results.json,r-oracle.json,receipt.json}`), checks
`receipt.json.reference_commit == P1`, recomputes the diff, and writes the vectors into
`receipts/inference/cases/CORE070-SURFCONV-INFERENCE-CI-ROUTE-008..011.json`. But the case receipts
contain **zero** `sha256` fields (`grep -c sha256` = 0) and do not cite #569's own
`julia_results_sha256` / `raw_sha256` / `contract_sha256`. The only tie is `glvmodels_commit_at_receipt_write`
(`ec36c65bd`, the tools commit that has #569's files as an ancestor). If #569 is rebased or its run
regenerated with different numbers before this PR merges, the inference receipts keep the old vectors and
nothing fails: the generator has no `--check` mode (unlike the contract generator) and nothing re-runs it.
The generator *would* refuse if re-run (demonstrated above), so this is a missing trigger, not a missing
check.

Suggested fix: in each wave5 case receipt add `read_from: {path: sha256}` for the three files and copy
#569's `julia_results_sha256`/`raw_sha256`; add `--check` to `core070_inference_p1_receipts.py`
(regenerate in memory, diff against tracked) so a stale receipt is caught the way a stale twin is.

### F2 (non-blocking, overstatement risk). "2 numeric pass" is one measurement counted twice

CI-ROUTE-008 (`confint(parm='icc')`, R default method) and CI-ROUTE-010 (`method='wald'`) have
byte-identical R vectors and byte-identical Julia vectors in #569's raw files (checked), because both
engines' default is Wald. The two rows are legitimately distinct *surface* rows (default routing vs
explicit method), but they pay with a single R-vs-Julia comparison. The PR body says "8 CI bounds each"
without saying they are the same 8 bounds. Suggested fix: one sentence in the PR body / check-log:
"008 and 010 are the same fit and the same 8 bounds; the numeric evidence is one comparison, the routing
evidence is two."

### F3 (non-blocking, disclosure). CI-ROUTE-011 passes while two Julia bootstrap lower bounds collapse

Receipt `measured.julia_structural.lower = [0.5615, 1.06e-07, 0.0804, 4.53e-40]` vs R
`[0.5713, 0.1262, 0.1269, 0.4144]`; uppers agree to ~0.03; points agree to 6.4e-7. The receipt's verdict is
PASS and its `why_not_numeric` says only "endpoints are not compared". The raw numbers are disclosed in
the receipt and the PR body flags it under "Finding worth a look", so this is not hidden. It is, however,
not flagged *in the receipt or the case-map row*, which is where a later reader will look. Inference (not
verified here, another lane owns it): a 2.5th-percentile of 4.5e-40 from 200 draws means at least 5
replicate fits returned ICC ~ 0 for that entry, which looks like a boundary-fit pattern in the Julia
bootstrap, not Monte Carlo noise; R's 0.41 says the sampling distribution is nowhere near 0.
Suggested fix: add an `anomaly` field to the 011 receipt naming the two entries and the R comparands,
and a one-line `note` on the case-map row, so the PASS is not read as "bootstraps agree".

### F4 (non-blocking, misleading field). Stale `julia_runner_sha256` carried into the P1 twin

`inference-batch-contract-p1.json` carries `julia_runner_sha256: d0884be6...` while
`tools/core070_inference_batch.jl` hashes to `5ba2d5cd...` (rename #423). No verifier or runner reads the
field (grep over `tools/` and `test/`: no hits), so it is inert. Leaving a field named `*_sha256` that does
not match the file is misleading: a reader takes it as a pin. Should a verifier check it? At P1, yes, it
is cheap and the field exists for that purpose; at P0 the contract is frozen and the mismatch is history.
Suggested fix: have `core070_inference_p1_contract.py` record `julia_runner_sha256_at_p1` (actual hash)
next to the carried value with a `regeneration_log` entry, and have
`core070_verify_inference_batch.py` check it when `SELECTED_PIN == "P1"`.

### F5 (non-blocking, carried from #567/#569 reviews). Remainder runner binds the library by path, not pin

`tools/core070_inference_remainder_batch.R` checks `find.package("gllvmTMB")` equals
`<frozen-library>/gllvmTMB` and writes `reference_commit = contract$reference_commit` into the receipt:
the receipt's P1 claim comes from the contract, not from the installed package. A P0-built library passed
with `GLLVM_PARITY_PIN=P1` would produce a receipt that says P1. The P1 DESCRIPTION carries no
`RemoteSha`, but `<oracle>/build/build.json` has `reference_commit` and `installed_tree_sha256`. The PR
body discloses this ("The runners check the library path, not that it is the P1 build"). Suggested fix
(can be a follow-up across the stack): at P1 require `CORE070_P1_ORACLE_BUILD_JSON`, check its
`reference_commit == P1` and recompute `installed_tree_sha256` over the library dir.

### F6 (minor). Destination directory created before the pin check

Both R runners `dir.create(destination)` before reading `GLLVM_PARITY_PIN`, so a refused run leaves an
empty directory (the `badpin/`, `badpin2/` dirs in `a3inf-runs/` are this). Harmless; move the pin check
above `dir.create` if touching the files again.

### F7 (info). Julia runners are pin-unaware

`tools/core070_inference_batch.jl` and `..._remainder_batch.jl` are unchanged and do not read
`GLLVM_PARITY_PIN`; the wave2 Julia batch receipt carries no `reference_commit`. That is coherent (the
Julia side never touches R; the R side carries the pin) but means "every new runner refuses a bad pin"
should be read as "every runner that reads the pin". Nothing to fix.

### F8 (info, pre-existing checker limit). Receipt tolerance is trusted

Mutating `tolerance` to 1.0 in a temp receipt copy still binds (bound=2). The checker takes the receipt's
tolerance at face value; tolerance provenance rests on the generator copying it from the contract (which
it does, and which `--check` on the contract side protects). Not this PR's defect; noted for #561's owner.

## What I did not check

- Did not re-run the Julia wave2 (57 s) or wave4 (14 s) batches or the Julia side of anything; I re-ran
  only the wave2 R crosscheck at P1 (identical `crosscheck_sha256`) and the two Python verifiers on the
  raw runs.
- Did not re-run #569's surface-conversion batch, and did not investigate the CI-ROUTE-011 Julia bootstrap
  collapse beyond reading the numbers (another lane owns it).
- Did not verify #567's oracle build beyond reading `build.json` (`reference_commit` P1,
  `original_source_unchanged: true`).
- Did not exercise the P0 wave2 path end to end (needs the `.unlazy/` retained run, absent here; it fails
  loudly, which is the right behaviour).
- Did not tamper the pinned P1 probe script or fixture (that hash loop is the unchanged P0 code path).
- Did not review the 34 excluded inference rows or CI-ROUTE-005.
