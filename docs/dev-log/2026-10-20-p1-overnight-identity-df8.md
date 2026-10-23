# FAMILY-00-IDENTITY df 8 measurement

Date: 2026-10-20. Worktree `/Users/z3437171/.codex/worktrees/p1-isdm-20261019/GLLVM.jl` at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Local measurement only. No commit, push, merge, Totoro job, or full suite. `GATES.md` was not edited. Ordinal, grouped-NB2, and binomial covariate fitters were not edited. No tolerance was changed. The row is not admitted.

Verdict: **DF8_PASS**.

## What was required

Shinichi kept the unique model at df 8 for `family/FAMILY-00-IDENTITY`. The reduced bridge at df 5 does not close the row (`docs/dev-log/plans/2026-10-20-p1-thirteen-row-signature-drafts.md`). The boundary receipt stays `VERIFIED_REFERENCE_MODEL_CHANGE_NOT_PARITY`, with `actual_df` 5 and `required_unique_model_df` 8 (`docs/dev-log/core070/gaussian-bridge-boundary-evidence.json`). That bridge was not run, and it is not used as a pass.

The existing df 8 route is `test/parity/test_gaussian_original_required.jl`. R fits `value ~ 0 + latent(0 + trait | site, d=1)` on the original fixture (`test/parity/fixtures/core070_gaussian_original.toml`, p=4, n=120, K=1, seed 81031). The free parameter vector is four `theta_rr_B` loadings and four `theta_diag_B` unique terms. Julia fits the same fixed residual SD with `fit_gaussian_pervar_gllvm`, then the wide formula `y ~ 0` and the reordered long formula, both with `pervar=true`.

## This run

Threads: `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. Julia reported 1 thread and BLAS reported 1 thread.

R library: `/Users/z3437171/local-scratch/a3cov-oracle/build/library/gllvmTMB`, version 0.7.1. Its `CORE070_SOURCE_PIN.toml` names reference commit `9539352f66f2db2cc26b1c393e67212a359b60c9`. Before the path check, that marker matched the P1 pin on reference commit, namespace hash, source-tree hash, and archive hash. The required-mode loader then stopped because `find.package("gllvmTMB")` still pointed at the user library. The fit itself loaded the oracle library above. Receipt files were written under `/tmp/family00-identity-df8-20261020` and were not copied into the repo.

| Check | Result | Existing bound |
| --- | ---: | ---: |
| `native_dof` | 8 | `length(rtheta)` = 8 |
| formula `dof` | 8 | `length(rtheta)` = 8 |
| R names | 4 `theta_rr_B`, 4 `theta_diag_B` | unique diagonal present |
| `r_random` | `z_B` | P1 contract `z_B` |
| `point_value_delta` | 3.410605131648481e-13 | 1e-6 |
| `point_gradient_delta` | 1.2079226507921703e-13 | 1e-6 |
| `r_endpoint_value_delta` | 7.958078640513122e-13 | 1e-6 |
| `native_endpoint_cross_delta` | 4.547473508864641e-13 | 1e-6 |
| native `delta_loglik` | 7.958078640513122e-13 | 1e-3 |
| formula `delta_loglik` | 7.958078640513122e-13 | 1e-3 |
| `native_gradient_max` | 1.5424461707880255e-10 | 1e-4 |
| `r_gradient_max` | 2.1296294572574696e-6 | 1e-4 |
| long `native_parameter_delta` | 0 | 1e-7 |
| long `native_loglik_delta` | 0 | 1e-7 |

Test counts: native pair 12/12, wide formula pair 10/10, reordered long formula 9/9. Elapsed fit time in those three summaries was 0.5s, 0.1s, and 0.2s after compilation. Exit status 0.

## What this leaves in place

The stored case receipts stay as they were. `CORE070-FAMILY-00-IDENTITY-NATIVE-MODEL` and `CORE070-FAMILY-00-IDENTITY-FORMULA-INTERFACE` remain FAIL in `docs/dev-log/core070/true-parity-latest/receipts/family/cases/`. `CORE070-FAMILY-00-IDENTITY-PUBLIC-R-BRIDGE` remains `NOT_EXECUTED`. The scoreboard row stays `partial_case_not_executed`. This run does not bind the row, move a scoreboard count, or mark C2, C3, C4, or X2 MET.

## Pin check stop

Stop: NOT_PIN_CLEAN. No FAMILY-00-IDENTITY receipt was written, and no receipt file was changed. No official writer was run. The continuation that stuffed a reference commit into the source dict is not a pin-clean pass.

`_core070_source_pin!` in `test/parity/parity_helpers.jl` refused at the path comparison. `realpath` of `GLLVM_PARITY_R_SOURCE_PIN` must equal `CORE070_SOURCE_PIN.toml` inside the directory returned by `find.package("gllvmTMB")`. On the df 8 run, `find.package("gllvmTMB")` still pointed at the user library, so that equality failed with `R source-pin marker must be CORE070_SOURCE_PIN.toml in the installed gllvmTMB library`. The marker file does exist at `/Users/z3437171/local-scratch/a3cov-oracle/build/library/gllvmTMB/CORE070_SOURCE_PIN.toml`. Its sha256 is `5b667a7d5f1718364a2794642cec52eac1ead345ec39fff5051c2c1ca7a62218`, the same `marker_sha256` recorded in `docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build.json` and in `/Users/z3437171/local-scratch/a3cov-oracle/build/build.json`.

The marker content does not satisfy the checker. `reference_commit`, `namespace_sha256`, `source_tree_sha256`, and `archive_sha256` match the P1 entry in `tools/core070_oracle_pins.toml`. The installed `NAMESPACE` hash matches `c1e91cd75bf29966932bcaec4a2b15234b23c7f60e1e39b3ce0917c6537f0f0c`. The installed tree hash, with the marker file ignored, is `5d4877193c3727965412546ca49c77588ba3083eab87629080e610cb889516d4`, matching the marker and the build receipt. The marker has no `gllvmTMB_version` key. The next check in the same function, `_core070_check_installed_package_version`, requires that key to be a string equal to the selected pin version `0.7.1`, and throws `R source-pin marker lacks gllvmTMB_version` when the key is missing (`test/parity/core070_pin.jl`). Later in the same function, `_core070_oracle_receipt_paths` checks the tracked build receipt, and that receipt also has no `gllvmTMB_version` field, which `_core070_check_oracle_receipt` refuses. A corrected `find.package` path would still fail on this marker.

`test/parity/Project.toml` and `test/parity/Manifest.toml` are absent from `git status --short`. HEAD remains `5cfc6754212588222779ccfd66ad7000375cf8d5`.

## Official checker, 2026-10-21

One more run of `test/parity/test_gaussian_original_required.jl`, from this worktree at `5cfc6754212588222779ccfd66ad7000375cf8d5`. Threads stayed at 1 (`JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `julia --threads=1`). `CORE070_PARITY_REQUIRED=1` and `GLLVM_PARITY_TESTS=1`. `GLLVM_PARITY_R_SOURCE_PIN` was `/Users/z3437171/local-scratch/a3cov-oracle/build/library/gllvmTMB/CORE070_SOURCE_PIN.toml`. `GLLVM_PARITY_R_LIBS` was `/Users/z3437171/local-scratch/a3cov-oracle/build/library`. The required-mode helper derives that same library from the pin path and prepends it. The checker file was left unchanged. No tolerance was changed. `GATES.md`, the pin file, and the stored receipts were left unchanged.

The path comparison passed. `_core070_source_pin!` then called `_core070_check_installed_package_version`, and that check refused because the marker has no `gllvmTMB_version` key. The process stopped there. `/tmp/family00-identity-df8-pincheck-20261021` was not created. No pass receipt was written.

Exact refusal:

```
ArgumentError: R source-pin marker lacks gllvmTMB_version; rebuild the pinned oracle with tools/core070_build_oracle.py before using it
```

Verdict: NOT_PIN_CLEAN. HEAD remains `5cfc6754212588222779ccfd66ad7000375cf8d5`.
