# Review: draft PR #569 — postfit and postfit-policy re-measured at gllvmTMB P1

- Branch `claude/true-parity-p1-postfit`, head `28d880acb`, stacked on #567 (`508d297d0`).
- Reviewed diff: `508d297d0..28d880acb` (commits `9d2dd6cbf`, `5086c7589`, `28d880acb`).
- Reviewer worktree: detached at `28d880acb` under `/Users/z3437171/local-scratch/lanes/GLLVM.jl-review-569` (removed at the end). Nothing on the branch edited, nothing pushed, nothing posted on GitHub.
- Compute used: ~5 min wall (two full batch re-runs, one estimand-rebind batch run twice under mutation, pin-typo probes). Threads capped `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4`.

## Verdict: NON-BLOCKING

The receipts are what they claim to be: R at P1 versus Julia, reproduced independently to the last digit, and every "numeric" row binds under both main's checker and #561's rule, with stale or over-tolerance receipts caught. The pin switch is strict in all three languages, the P0 surface is untouched, and no classification was changed. Nothing here blocks a merge behind #567.

What should still be acted on (none by editing this PR's receipts): three of the 35 numeric passes are measured on a fixture where the quantity is constant by construction, so they do not discriminate a wrong implementation (Finding 1, demonstrated by mutation). That is a fixture defect to record and a maintainer call on whether such rows bind, not a receipt error.

## Findings

### 1. Three numeric passes cannot discriminate a wrong implementation (non-blocking; needs a maintainer call)

`extract_communality` and `extract_proportions` are measured on the `unique = FALSE` Gaussian fixture (`tools/core070_estimand_rebind_batch.R:84`), where communality is identically 1 for every trait. `tidy` fixed effects (and `POST-COEF-NAMED`) are measured on row-centred data, so both engines return ~1e-14.

Evidence (mutation, run in my worktree with `src/extractors.jl` temporarily edited, then restored byte-for-byte):

```
# Mutation A: Julia extract_communality returns a constant 1.0 (wrong formula)
$ GLLVM_PARITY_PIN=P1 Rscript --vanilla tools/core070_estimand_rebind_batch.R <P1 lib> $S/mutA
CORE070_ESTIMAND_REBIND_BATCH_PASS
postfit/POSTFIT-SURFACE-extract_communality   CORE070-ESTIMAND-REBIND-EXTRACT-COMMUNALITY   PASS

# Mutation B: 0.99 * s/t (wrong by 1 %)
CORE070_ESTIMAND_REBIND_BATCH_FAIL
postfit/POSTFIT-SURFACE-extract_communality   CORE070-ESTIMAND-REBIND-EXTRACT-COMMUNALITY   FAIL
```

So the harness bounds |R − Julia| correctly (B fails), but on this fixture a constant-1 implementation is indistinguishable from the right one (A passes). The tracked receipt records `r_value = [1,1,1,1,1]`, `julia_value = [1.0,...]`, `max_abs_diff = 0.0`. Same structure for proportions. For `tidy`, the R oracle is `[-1.36e-14, 1.29e-14, ...]` and the harness diff is `1.359e-14`, i.e. Julia returns ~0; a Julia `tidy` that returned `zeros(5)` would pass.

Under #561's rule as written these rows bind (tolerance > 0, diff within tolerance, pin P1). The PR body discloses this. Whether "binds but non-discriminating" is acceptable is the maintainer's call.

Suggested fix (tooling, separate PR): have `tools/core070_postfit_p1_receipts.py` add a `degenerate_fixture: true` flag (or `discriminating: false`) to a comparison entry when the R oracle vector is constant or has max |value| below, say, 1e-10, and let the checker report those rows in a separate count (`bound_numeric_degenerate`). A non-degenerate fixture (`unique = TRUE`, uncentred X) for these four accessors would be the real cure; it is a new case, which is a contract change and therefore the maintainer's.

### 2. Runners bind the library by path, not by pin (non-blocking; disclosed)

Only `tools/core070_postfit_policy_batch.R` checks source bytes against the pin (`source_pins` sha256 against `CORE070_P1_R_SOURCE_ROOT`). The other five runners assert only that `find.package("gllvmTMB")` lives under the library directory the operator passed. A P0 library passed with `GLLVM_PARITY_PIN=P1` would produce receipts labelled P1 whose only tell is `gllvmTMB_version: "0.7.0"`, which no Python verifier checks.

I verified independently that the library actually used is P1:

```
$ shasum -a 256 build/source/R/{vcov-coef,methods-gllvmTMB,z-confint-gllvmTMB,predictive-diagnostics,output-methods}.R
  == git show 9539352f6:<same paths>   (all five identical)
$ Rscript -e 'deparse(gllvmTMB::getLV)' | grep -c temporal_index    -> 4   (P1-only token; P0 getLV has none)
build.json: reference_commit = 9539352f6..., source_tree_sha256 = 86fa00f0... (== tools/core070_oracle_pins.toml [P1])
```

So the P1 runs exercised the P1 accessor bodies, including the 13 changed ones. The gap is future-proofing, not a defect in these receipts.

Suggested fix (tooling): each runner reads `tools/core070_oracle_pins.toml` and `stopifnot(packageVersion("gllvmTMB") == <pin version>)`; each verifier checks `receipt.gllvmTMB_version` against the pin table. Cheap, and it turns the PR body's "the operator passes the P1 library" into a checked fact.

`CORE070_P1_R_SOURCE_ROOT` handling is correct: missing -> `Error: GLLVM_PARITY_PIN=P1 needs CORE070_P1_R_SOURCE_ROOT`; a copy of the P1 tree with one byte appended to `R/methods-gllvmTMB.R` -> `Error: identical(digest, contract$source_pins[[rel]]) is not TRUE`. The builder's `env.sh` points it at `a3cov-oracle/build/source`, whose bytes match `git show P1:` for all four pinned files.

### 3. Harness-reported differences: acceptable, and regenerable (non-blocking)

Wave7, wave8 and all postfit-policy numeric receipts carry `diff_source: harness-reported` because those Julia children write only `max_abs_diff`/`delta`, not the Julia vector (`tools/core070_wave8_conversion_batch.jl:242`, `tools/core070_postfit_policy_batch.jl:267-321`). Two checks close the trust gap:

- Independent re-run at P1 (my worktree, same env): surface-conversion 114 s, postfit-policy 24 s, both `*_BATCH_PASS`, verifiers `STATE_OK`/`BATCH_VERIFIED`. `results.tsv` identical to tracked. R oracle values: max |mine − tracked| = 0.0 over 19 + 14 cases. Every Julia `max_abs_diff`/`delta` identical to tracked to the printed 4 s.f. (e.g. LOGLIK-VALUE 1.135e-09, COEF-NAMED 1.359e-14, RESIDUAL-* 8.644e-06).
- Regeneration from tracked inputs: copying the tracked `receipts/postfit/<batch>-p1/` directories into the generator's expected layout and running `tools/core070_postfit_p1_receipts.py` reproduces every comparison block and the case map byte-for-byte except three fields: `verifier_output` (empty, because `verify.txt` is not tracked), `glvmodels_commit_at_receipt_write`, `glvmodels_src_tree`. Tracked copies are byte-identical to the raw runs in `a3pf-runs/` (24 files checked).

Suggested fix (tooling, optional): track `verify.txt` per batch (a few lines each) so regeneration is exact; have the wave7/wave8/policy Julia children emit `julia_values` in a future batch version so the receipt generator's `vec_entry` cross-check applies everywhere.

### 4. Accessor-diff record misses the estimand-rebind accessors (nit)

The five contract twins record 13 changed bodies. The PR body says 14, adding `extract_proportions`. That fourteenth is true (`R/extract-omega.R`: P0 `ab67d7d4` != P1 `bcde90a0`) but lives only in the PR body, because estimand-rebind has no contract file and `ACCESSORS` in `tools/core070_postfit_p1_contract.py` has no entry for it. `extract_communality`, `extract_correlations`, `extract_Omega` are unchanged (checked).

Suggested fix: emit the estimand-rebind accessor diff into a small tracked JSON (or into the batch `receipt.json`) so the record is not PR-body-only.

### 5. Decision items: classification (maintainer) vs tooling

All five of the builder's decision items are correctly left undone. Sorting them:

| Item | Kind | Who |
| --- | --- | --- |
| nobs expectation (Julia n vs p*n; P0 expectation now false) | contract expectation + convention | maintainer |
| `tolerance: 0` for exact-integer cases (4 policy rows) | checker rule (extends #561's `tolerance > 0`) | maintainer, since it changes what binds |
| POST-DEVIANCE rebind to `CORE070-WAVE8-DEVIANCE-MULTI` | case-map `executable_case_ids` edit | maintainer |
| retire `.proportions_*_ci` rows (export gone at P1; confirmed `git show P1:NAMESPACE` has no `proportions_`) | classification | maintainer |
| a binding tier for verdict/policy rows (13 partial) | checker rule | maintainer |

Confirmed no field of the 52 rows differs from `required-source-case-map.json` in `classification`, `disposition` or `executable_case_ids`.

### 6. Pin switch strictness and P0 behaviour: verified

```
R      GLLVM_PARITY_PIN=P2  -> Error: GLLVM_PARITY_PIN must be P0 or P1, got 'P2'   (surface-conversion)
R      GLLVM_PARITY_PIN=p9  -> Error: ... got 'P9'                                  (postfit-policy)
Julia  GLLVM_PARITY_PIN=P2  -> ERROR: LoadError: GLLVM_PARITY_PIN must be P0 or P1, got "P2"   (postfit-1)
Julia  GLLVM_PARITY_PIN=" p3 " -> ... got "P3"                                      (surface-conversion child)
Python GLLVM_PARITY_PIN=P2  -> GLLVM_PARITY_PIN='P2' is not a recognized pin ...    (via parity_oracle)
```

With the variable unset, all six Python verifiers resolve to the P0 contract paths and P0 reference commit (module-load probe). `tools/test_parity_oracle_defaults.py`, `tools/test_core070_build_oracle_pin.py`: OK. `node tools/test_true_parity_check.mjs`: all negative controls pass. `GLLVMTMB_DIR=<clone> python3 tools/core070_postfit_p1_contract.py --check` -> `CORE070_POSTFIT_P1_CONTRACTS_CURRENT`.

### 7. #561 rule binding and stale-receipt detection: verified

```
main checker  C1 required=52 bound=35 free=15 unsigned_or_blocked=2 ... C1_NOT_MET
#561 checker  C1 required=52 bound=35 bound_numeric=35 bound_registration_only=0 bound_signed=0 ... numeric_label_without_numeric_receipt=none
probe: comparison.pin -> "P0" in one receipt   -> bound_numeric=34, numeric_label_without_numeric_receipt=postfit/POSTFIT-SURFACE-tidy...(comparison not pinned to P1 ...)
probe: max_abs_diff -> 0.5 in one receipt      -> bound_numeric=34, ...POST-LOGLIK-VALUE(case ...: abs_diff 0.5 > tolerance 0.000001)
```

Audit of all 35 numeric rows' receipts: every comparison entry has `pin = P1`, `tolerance > 0`, `max_abs_diff <= tolerance`; two-case rows (`extract_Sigma`) carry both receipts. The nobs row is `numeric_fail` and does not bind.

### 8. Diff hygiene: clean

- `git diff --name-only 508d297d0 28d880acb | grep -E '^(src/|Project.toml|\.github/|GATES.md)'` -> nothing.
- Modified (not added) files are only the 17 `tools/core070_*` scripts and `docs/dev-log/check-log.md`; every P0 contract, `required-source-case-map.json`, `case-map.json`, and `receipts/covariance/` untouched.
- No `.log`/`verify.txt`/untracked raw files under `receipts/postfit/` (76 files, 318 KB; largest `surface-conversion-p1/r-oracle.json` 39 KB).
- PR body and commit messages: no agent `@handles` (only the `Co-Authored-By` e-mail domain). Counts in the body match `case-map-postfit.json` (`{numeric_pass: 35, numeric_fail: 1, partial: 13, needs_surface: 1, retired: 2, not_measured: 0}`), and the body's "what is not covered" list is accurate, including the weak-pass and library-path caveats.

## What I did not check

- wave6, wave7, wave8, estimand-rebind (un-mutated) and postfit-1 were not re-run by me; their tracked receipts were checked for byte-identity with the raw runs and regenerated from tracked inputs, not re-measured.
- The wave6 FAIL (nobs expectation) was taken from the tracked receipt and #567's review, not reproduced.
- The R bridge (`engine = "julia"`) path, non-Gaussian or realistic-size fixtures: out of the PR's scope and not exercised.
- Whether the P0 batches still pass end-to-end with the pin unset (the P0 `.unlazy/...readback` tree is not present on this machine); P0 behaviour was checked by reading the code paths and the Python module-load probe only.
- I did not mutate `tidy`/`coef` on the Julia side; the non-discrimination claim for those two rests on the recorded oracle values (~1e-14) and the harness diff equalling max |R|.
