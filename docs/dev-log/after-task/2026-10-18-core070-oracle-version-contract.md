# After-task: Core070 oracle version-qualified Gaussian contract

## 1. Goal

Correct the Frozen R Gaussian contract failure by binding each expected random-effect set to the actual pinned gllvmTMB commit and installed package version.

## 2. Implemented

The Gaussian parity fixture derives its expected random-effect names from the selected commit and installed version. The 0.7.0 pin expects `z_B` and `s_B`; the 0.7.1 pin expects `z_B`. Unknown commits and mismatched versions are rejected. The oracle builder records the installed `gllvmTMB` version in its build receipt and installed marker. Required parity setup checks the marker version against both the selected pin and `utils::packageVersion()`. Build receipt and marker validation reject missing version metadata with a rebuild instruction. Build-failure and source-drift diagnostics take precedence over version mismatch diagnostics.

## 3a. Decisions and Rejected Alternatives

- Expected behavior is keyed to exact reference commit and package version. The test does not accept either random-effect set generically.
- Existing markers without `gllvmTMB_version` fail closed and require rebuilding the selected pinned oracle; no version is inferred from the CI job label or written into legacy evidence.
- No R oracle build was started in this repair slice. The root requested source and unit-test review before deciding whether a bounded build was needed.

## 4. Files Touched

- `test/parity/core070_pin.jl`
- `test/parity/parity_helpers.jl`
- `test/parity/test_core070_pin.jl`
- `test/parity/test_gaussian_original_required.jl`
- `tools/core070_build_oracle.py`
- `tools/test_core070_build_oracle_pin.py`
- `docs/dev-log/check-log.md`
- `docs/dev-log/after-task/2026-10-18-core070-oracle-version-contract.md`

## 5. Checks Run

- `JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=. test/parity/test_core070_pin.jl`: 69/69 passed in 29.8 seconds; `/tmp/cross-package-optimizers-20261018/gates/gllvmodels-version-contract-green.log`.
- Builder-focused Python tests, excluding the expensive fresh-archive hash neighbor: 9/9 passed in 10.2 seconds; `/tmp/cross-package-optimizers-20261018/gates/gllvmodels-version-builder-focused-final2.log`.
- `python3 -m py_compile core070_build_oracle.py test_core070_build_oracle_pin.py`: passed.
- Julia parse check for `test/parity/test_gaussian_original_required.jl`: passed; `/tmp/cross-package-optimizers-20261018/gates/gllvmodels-version-gaussian-parse.log`.
- `git diff --check`: passed.

## 6. Tests of the Tests

The pin test covers both supported commit/version mappings, unknown commits, both cross-version mismatches, missing and mismatched installed-marker versions, and missing or mismatched build-receipt versions. The builder test queries `packageVersion()` from temporary P0 and P1 library fixtures. Builder verification tests accept a matching receipt and marker, and reject wrong receipt or marker versions. A pure validation test confirms that an install failure is reported before absent version metadata.

## 7a. Issue Ledger

- Fixed: the Frozen R job encoded the 0.7.1 `z_B` contract while its named P0 source pin was 0.7.0, whose expected names include `s_B`.
- Fixed: old build receipts and source markers did not prove which installed package version they represented.
- Incomplete: the full Python pin test file was interrupted while the unchanged fresh-archive hash test was running. The isolated hash test also remained live past 120 seconds and was stopped. No pass is claimed for that test. Logs: `/tmp/cross-package-optimizers-20261018/gates/gllvmodels-version-builder-red.log` and `/tmp/cross-package-optimizers-20261018/gates/gllvmodels-version-builder-hash-diagnostic.log`.

## 8. Consistency Audit

The commit-to-version table is read from `tools/core070_oracle_pins.toml`; the Julia fixture, installed marker, build receipt, and live R package version are checked against that source. The failure path retains the raw build receipt and does not write a qualified marker for a version mismatch. No model, estimator, or statistical contract changed beyond selecting the correct version-specific frozen assertion.

## 9. What Did Not Go Smoothly

The full Python file includes a local archive re-derivation that did not finish within 120 seconds. The new installed-version probe itself passed independently in 1.9 seconds. I stopped the long archive diagnostic at the cap and kept its log; I did not retry the whole file or start an oracle installation.

## 10. Known Residuals

The old archive-hash neighbor remains unverified in this repair. The main advisory CI workflow is not rerun, no full Julia suite is claimed, and the R oracle build has not been run. The patch awaits parent review and any requested bounded build gate.

## 11. Team Learning

When a pin-specific evidence test combines a small metadata probe with a large archive re-derivation, run the metadata probe independently and keep the archive calculation bounded so the useful contract result is not obscured by unrelated hashing time.

## 12. Cross-Product Coverage

The repaired test reads the actual installed R package version from Julia/RCall and compares it with the Julia pin table before making the Gaussian parity assertion. No downstream package or other product was changed.
