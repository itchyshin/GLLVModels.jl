# Review: PR #586 — aghq rows re-measured at gllvmTMB P1 (A3)

Reviewed range: `0fb2b9411..8f4427d4a` (branch `claude/true-parity-p1-aghq`, stacked on #584).
Reviewer worktree: detached at `8f4427d4a` under `local-scratch/lanes/GLLVM.jl-review-586` (removed after review).
Environment: local Mac, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=4`, Julia 1.10.0 via juliaup, R 4.6.0, installed P1 oracle library at `local-scratch/a3cov-oracle/build/library` (read-only), P1 R source tree extracted from `a3cov-oracle/source/gllvmTMB-core070.tar` into scratch for `CORE070_P1_R_SOURCE_ROOT`.

## Verdict

**Accept as a non-binding docs/tools PR.** No blocking findings. The tracked receipts reproduce from a clean re-run, no row is numeric, no row binds under either main's or #561's checker, the P0 code paths are preserved, the pin/marker/runner-hash guards refuse tampering, and the PR body's claims match what I measured. Two maintainer decisions the PR already asks for (stall assertions; Julia twins) are real, and my probe answers the second one: Julia twins are feasible for 10 of the 14 R-only policy rows today.

## Findings (numbered, with evidence)

1. **Both batches reproduce.**
   - Policy bind at P1 (`GLLVM_PARITY_PIN=P1 Rscript --vanilla tools/core070_aghq_public_policy_bind.R <lib> <dest>`): 10 s, `CORE070_AGHQ_PUBLIC_POLICY_BIND_PASS bound=14/14`. Receipt identical to the tracked `aghq-policy-p1/receipt.json` except `generated_at` and the `receipt_sha256` that covers it; all 14 `cases[*].observed` blocks (used, k, objective, convergence, reason) byte-equal.
   - Control batch at P1: 3 s, `CORE070_AGHQ_CONTROL_BATCH_PASS`. `results.tsv`, `julia-results.json`, `r-oracle.json` byte-identical to the tracked copies; `receipt.json` identical except `julia_elapsed_seconds`. `GLLVM_PARITY_PIN=P1 tools/core070_verify_aghq_batch.py --state <rerun> --self-test` prints the same three `..._PASS` lines and `CORE070_AGHQ_CONTROL_BATCH_VERIFIED 16 negative_controls 3` as the tracked `verify.txt`.
   - Note (not a defect): my first control run failed because the fresh worktree had no instantiated Julia environment (`Optim` not installed). The batch wrote a `status: FAIL` receipt with `julia_exit_code: 1` and `julia_results_sha256: null`, i.e. it fails closed rather than passing on a missing child. After `Pkg.instantiate()` the re-run matched.
   - `python3 tools/core070_aghq_p1_receipts.py --check` → `CORE070_AGHQ_P1_RECEIPTS_CURRENT 21 case receipts, 21 rows`. `GLLVMTMB_DIR=<gllvmTMB clone> python3 tools/core070_aghq_p1_contract.py --check` → `CORE070_AGHQ_P1_CONTRACTS_CURRENT` (without `GLLVMTMB_DIR` the tool exits 1 with a clear message).

2. **No row is numeric; nothing binds under either checker.**
   - `case-map-aghq.json`: `numeric_rows: []`; counts `paired_control_categorical_pass: 7`, `r_only_policy_pass: 14`, all other tiers 0; the only `evidence` keys are `non_binding_receipts` and `tier` (no row has `evidence.receipt`); no case receipt carries a `comparison` block (the receipt tool's `--check` rejects one).
   - main's `tools/true_parity_check.mjs` (identical to the branch copy) and #561's (`origin/claude/true-parity-p1-namespace-v2`, byte-equal to `92cf39571`, temp copy in the worktree, deleted after), both with `PARITY_REF=FS PARITY_CASEMAP=docs/dev-log/core070/true-parity-latest/case-map-aghq.json`:
     - main C1: `required=21 bound=0 free=21 unsigned_or_blocked=0 ... C1_NOT_MET`
     - #561 C1: `required=21 bound=0 bound_numeric=0 bound_registration_only=0 bound_signed=0 free=21 ... registration_only=none numeric_label_without_numeric_receipt=none numeric_receipt_not_passed=none numeric_recorded_diff_mismatch=none C1_NOT_MET`
     - C8 (both): all 21 rows `NOT_TWINNED_NOT_SIGNED`, `C8_NOT_MET`.
   These are the lines the PR body quotes.

3. **P0 paths preserved (read from the diff).**
   - `core070_aghq_public_policy_bind.R`: the P0 branch still does `devtools::load_all(pkg_root)`, writes `receipt_path` directly, records `r_engine.source_tree/git_head` and the original `oracle_note`. The P1 branch is a separate `if` block.
   - `core070_aghq_batch.R`: P0 still reads `docs/dev-log/core070/aghq-batch-contract.json`, expects `b4d5fee6...`, and checks pins against `.unlazy/core070-aghq/oracle-source/readback`.
   - Two P0-visible but benign changes: `dir.create(output_dir)` moved after the pin checks (a refused run leaves nothing behind), and the P0 receipt now carries a `source_pin` field (NULL when the P0 library has no marker; `core070_source_pin.R` checks a marker only if present at P0). `shQuote` in `sha256_file` is a hardening for paths with spaces. Neither alters what a valid P0 run measures. The P0 paths were not re-run (the readback tree and the P0 source tree are not on this host), as the PR states.
   - `core070_verify_aghq_batch.py` at the default pin still targets the P0 contract and P0 reference commit; run unpinned against my P1 run it raises `ValueError: wrong reference commit` (P0 verifier refuses a P1 run).

4. **Pin strictness and hash checks are real (tamper tests, all restored by copy afterwards; `git status` clean).**
   - `GLLVM_PARITY_PIN=P2` on either runner: `Error: GLLVM_PARITY_PIN must be P0 or P1, got 'P2'`; destination not created.
   - `GLLVM_PARITY_PIN=P1` without `CORE070_P1_R_SOURCE_ROOT` on the control batch: stops with the named error; destination not created.
   - Library copy with the marker's `reference_commit` altered: both runners stop inside `core070_source_pin()`; destination not created.
   - P1 source tree with one byte appended to `R/aghq-gate.R`: control batch stops at `identical(digest, contract$source_pins[[rel]]) is not TRUE`; destination not created.
   - Runner edit (a comment appended to `core070_aghq_public_policy_bind.R`): contract `--check` reports `STALE .../aghq-public-policy-contract-p1.json`. Receipt `--check` still reports CURRENT, because `policy_checks()` hashes the runner blob **at the recorded run commit** (`blob_sha_at(rc, POLICY_RUNNER)`), not the working tree. That is by design (receipts describe a past run), and a re-run from the dirty tree would be held; but note the working-tree guard for the runner lives in the contract check, not the receipt check. Minor.
   - Tracked case receipt with `julia_label` "1"→"2": receipt `--check` → `STALE ... receipt body differs from the re-derivation`.
   - Case-map tier `r_only_policy_pass`→`numeric_pass` on one row: receipt `--check` → `STALE ... case-map rows differ from the re-derivation: aghq/AGHQ-AUTO-K-BINOMIAL`.

5. **Health flags on passing R fits: confirmed, with one wording nit.** From the tracked policy receipt's `observed.reason`:
   - `AUTO-K-BINOMIAL` (k=5): "8 adaptation passes, stalled ... max |grad| = 2.73 (relative 0.0281)", convergence 0.
   - `POLICY-EXPLICIT` (binomial, k=3): "stalled ... max |grad| = 2.23 (relative 0.0232)", convergence 0.
   - `AUTO-K-ORDINAL` (k=9): "100 adaptation passes, stalled ... max |grad| = 0.000434 (relative 2.81e-06)", **convergence 1**.
   The runner's assertions (`tools/core070_aghq_public_policy_bind.R` lines 189–190, 236, 249, 275) check only `used`, `k` and `is.finite(objective)`, so these pass. The PR body sentence "stalled for AUTO-K-BINOMIAL (max |grad| 2.73), AUTO-K-ORDINAL and POLICY-EXPLICIT (max |grad| 2.23)" is correct as parsed, but a reader may attach 2.23 to ORDINAL; ORDINAL's gradient is 4.3e-4 (near tolerance) and its problem is the optimizer code, not the gradient. Suggest one clause per row. Maintainer decision 4 (add convergence to the P1 assertions, a contract edit) is the right question; the two binomial stalls at relative gradient ~0.02–0.03 are not near tolerance.
   - P0→P1 objective agreement: max `p0_to_p1_objective_abs_diff` over the 13 fit rows is 2.42e-6 (`POLICY-EXPLICIT`), matching the PR's "at most 2.4e-6". R against R, not a comparison, as labelled.

6. **Control batch contents are what the PR says.** `r-oracle.json` now records `r_value` for all 16 cases: `FALSE` for FALSE/NULL, `"auto"` for TRUE/"auto", `1L`/`2L`/`9L` for the integers, `NA` (errored, as expected) for the 9 INVALID cases. Julia calls are `_aghq_request(false|nothing|true|:auto|1|2|9)` with expected labels `off/off/auto/auto/1/2/9`. The Julia `_aghq_request` (src/families/aghq_fit_info.jl:37) accepts `false`, `nothing`, `true`, `:auto`, positive Int; it rejects the String `"auto"` (R's normalizer accepts the string). The contract uses `:auto` on the Julia side, so the paired control compares the two normalizers on their own idiomatic inputs; a Julia twin fed R-style inputs verbatim would need a String→Symbol adapter. Not a defect of this PR.

7. **Julia twins for the R-only policy rows: feasible for 10 of 14 (called, not inferred).** Probe in the worktree with `fit_poisson_gllvm`, `fit_binomial_gllvm`, `fit_gaussian_gllvm` (all take `Y` as species × sites, `K=`, `aghq=`), reading `fit.integration.{requested, actual, k, reason}`:
   - Poisson p=5 n=30 `aghq=:auto` → actual `aghq`, k=5, converged; `aghq=3` → k=3; `aghq=false` → plain `PoissonFit`, `_is_aghq_fit` false.
   - Binomial p=5 n=30 `aghq=:auto` → k=5 (reason `no_merit_descent`); `aghq=3` → k=3; p=20 n=40 `aghq=:auto` → actual `laplace`, reason `auto_trait_cutoff` (matches R's cutoff at 20 traits); p=20 n=40 `aghq=9` → k=9 used; p=19 n=35 `aghq=:auto` → k=5 used.
   - Gaussian p=5 n=30 `aghq=:auto` → k=5, converged.
   - `fit_gllvm(Y; family=Poisson(), K=1, aghq=5)` also routes through (family must be an instance; `family=:poisson` / `Poisson` type are rejected). The top-level `gllvm()` has no matrix method (formula only), so a twin must use the `fit_*` functions.
   So the AUTO-K-POISSON/-BINOMIAL/-GAUSSIAN, DEFAULT-OFF (default `aghq=false`), and the six binomial POLICY rows (OFF, EXPLICIT, EXPLICIT-BYPASS-CUTOFF, AUTO-ENFORCE-CUTOFF, TRAITS19, TRAITS20) have a same-model Julia AGHQ surface now. AUTO-K-NB2, -DELTA, -ORDINAL, -TWEEDIE do not: no `aghq=` keyword exists for those families (`src/families/aghq_grid.jl:6` "No public aghq= knob"). Note for the maintainer: on the binomial p=5 n=30 toy shape both engines report a non-converged adaptation (R "stalled", Julia `no_merit_descent`) on independent random data, so the small-binomial AGHQ stall looks shared, not an R-side artefact.

8. **Hygiene.** Diffstat touches only `tools/`, `docs/dev-log/check-log.md`, and `docs/dev-log/core070/true-parity-latest/`; no `src/`, `Project.toml`, `.github/`, `GATES.md`, P0 contract, or P0 receipt. No `/Users/`, `local-scratch`, `Dropbox` or `/private/tmp` strings added by the diff (the pre-existing host path in a `core070_oracle_pins.toml` comment is not from this PR). Commit messages carry only a `Co-Authored-By` trailer; no agent `@handle` in commits or PR body. The `check-log.md` entry matches the PR body and my measurements. Runtime claim ("under 15 min estimated; 4 s + 10 s actual") matches (3 s + 10 s here).

## What I did not check

- The P0 code paths of either runner were not executed (P0 readback tree and P0 bind source tree absent on this host); only read in the diff.
- `Pkg.test()` and `test/parity/test_core070_pin.jl` were not run (no `src/` or test change in range).
- The other families' contract/receipt `--check`s (family, data, covariance, postfit, inference) and `tools/test_true_parity_check.mjs` were not run.
- The "hold" path (writing receipts from a dirty tree so the policy rows go to `r_only_policy_held_batch_verifier_failed`) was not exercised; only read in `tools/core070_aghq_p1_receipts.py`.
- The carry-scan source (`origin/claude/true-parity-p1-carry:.../carry-scan-p1.json`) sha256 and the `ROW_TO_CASE` map from `core070_aghq_public_policy_bind_apply.py` were not independently re-derived; `--check` CURRENT is my evidence that the tracked case map agrees with them.
- The stacked base (#584 at `0fb2b9411`) and the P0 policy receipt's contents were taken as given.
- Julia twins were probed on my own random toys, not on the exact R fixtures (seeds/design of the R runner), so "feasible" means the surface exists and behaves as the rows require, not that a numeric comparison was made.
