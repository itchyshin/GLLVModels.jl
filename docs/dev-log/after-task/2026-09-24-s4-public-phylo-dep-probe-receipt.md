# S4 public phylo_dep probe receipt (paste-gated)

**Date:** 2026-09-24
**Paste ack:** Shinichi pasted `S4 probe yes` in chat on 2026-09-24, and chose option A (probe-only `GLLVM` shim) for the renamed-package blocker.
**GLLVModels.jl tip at the official run:** `943a3be49` (branch `claude/s4-probe-run-20260924`, from `origin/main` `3b19b2817`)
**gllvmTMB recorder:** PR #1283 @ `97214679c94cc4a6b9e02d3c2b03ccce516027d8` (branch `codex/destination-b-s4-phylo-dep-formula-20260910`), detached worktree, read-only
**Frozen oracle pin (unchanged):** `b4d5fee64def88bc768dda1f1f77c29b295edd86`

**Verdict in one sentence:** the probe ran, but the frozen recorder refused to write a receipt, because its own runner cannot call `test_that()` under `Rscript --vanilla`; no endpoint was produced, so there is no S4 pass.

```text
S4_ESTIMATE est_min=3 recorded=post_run pre_run_record=orchestrator_brief_2to6min
S4_RESULT pass=0 fail=2 julia_gap=0 recorder_drift=0 oracle_defect=2 wall_min=0.22
```

How the counts are derived. No endpoint pair was produced, so the unit here is the recorder's selected `test_that()` expression, not an endpoint. The recorder's own `FAILED.json` reports `selected_test_count: 1` and `test_counts: {failed: 0, skipped: 0, error: 1}`, with one `test_tab` row (`test: null`, `nb: 1`). That is because both `test_that()` calls errored before they registered a test name, so the reporter folded them into one row. The same file lists two entries in `source.selected_test_expressions` and two `expectation_error` entries in `reporter_details`, one per expression. `fail=2` and `oracle_defect=2` count those two selected expressions, each of which produced an `expectation_error`. Counted in the runner's own reported unit, the result is 1 error. Both failures share one root cause.

The `S4_ESTIMATE` line was written after the run, when the receipt was drafted; see Estimate.

## Scope boundary

- IN: one isolated public-formula probe against the recorder runner (DestB S4 cell), plus the shim it needed.
- OUT: Arc 0 promotion, capability row `covered`, honest-0.7 FINAL-REVIEW complete, any gllvmTMB edit.

This probe receipt is not Arc 0 promotion and does not make any capability row `covered`. It is not a parity claim beyond the numbers it records.

## Environment

| Field | Value |
|-------|--------|
| Host | local Mac, Darwin 25.6.0, arm64 (Mac-light, D-50) |
| Julia | 1.10.0 (`~/.julia/juliaup/julia-1.10.0+0.aarch64.apple.darwin14/bin/julia`, sha256 `fd739b38…4ead30`) |
| R | 4.6.0; JuliaCall 0.17.6, testthat 3.3.2, digest 0.6.39, jsonlite 2.0.0, ape 5.8.1 |
| GLLVModels.jl project (as `--julia-project`) | `tools/destination_b/probe_env/GLLVM` (the shim; see deviation) |
| Julia env (`GLLVM_S4_JULIA_ENV`) | `tools/destination_b/probe_env` |
| gllvmTMB root HEAD | `97214679c94cc4a6b9e02d3c2b03ccce516027d8`, clean before and after |
| Receipt JSON path | `docs/dev-log/core070/s4-public-phylo-dep-probe-receipt-2026-09-24.json`: **not written** by the recorder |
| Failed-attempt diagnostic | `docs/dev-log/core070/s4-public-phylo-dep-probe-receipt-2026-09-24.json.failed-attempt/FAILED.json` (sha256 `7379d79e…6def`, written by the recorder, unedited) |
| Wall clock | official run 13.2 s; diagnostic run 99.3 s |
| Threads | `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, through the lane semaphore; both Julia slots held (launcher plus the embedded Julia) |
| Log | lane log `.unlazy/true-parity-20260924/logs/s4-probe.log` (local, not committed) |

## Estimate

The `est_min=3` figure was recorded after the run, when this receipt was written. No estimate line was written to the lane log or elsewhere by the builder before the full run started (log, 16:59:43Z). The pre-run basis was:

- the orchestrator's written brief for this run, which estimated 2 to 6 min (well under the 30-minute line); this is the only estimate written before the run, and it sits in the workflow transcript, not in a repository or lane file;
- the define-only smoke at 16:59:11Z (source the runner with `GLLVM_S4_PUBLIC_PHYLO_DEP_DEFINE_ONLY=1`, call `s4_public_phylo_dep_clean_julia_probe()` on the shim and probe env): 1.58 s wall, returned the expected six-field list with `package_root` = the shim directory.

So gate X409 G2 ("pre-run estimate line") is met in substance by the orchestrator's brief, not by a pre-run artifact in this receipt. A checker reading `S4_ESTIMATE` here is not evidence that the estimate came first. The official run took 13.2 s. The diagnostic run, which reached both fits, took 1.7 min, measured after the fact.

## Commands (exact)

The launcher `run_s4_public_phylo_dep_probe.jl` could not be used as is: it derives the repo root by walking up from `--julia-project` to the first `Project.toml`, which is now the shim's, so the template lookup fails. The run called the same harness functions the launcher calls, with `gllvm_root` set to the repo root:

```text
# run from the GLLVModels.jl worktree root, under julia_slot.sh (both slots)
GLLVM_S4_PROBE_PASTE='S4 probe yes' \
~/.julia/juliaup/julia-1.10.0+0.aarch64.apple.darwin14/bin/julia --startup-file=no --project=. -e '
  include("tools/destination_b/s4_public_phylo_dep_probe_harness.jl");
  s4_public_phylo_dep_require_paste();
  r = s4_public_phylo_dep_preflight!(;
    gllvmtmb_root = "/Users/z3437171/local-scratch/gllvmtmb-s4-recorder-97214679c",
    julia_project = "tools/destination_b/probe_env/GLLVM",
    julia_executable = "/Users/z3437171/.julia/juliaup/julia-1.10.0+0.aarch64.apple.darwin14/bin/julia",
    julia_env = "tools/destination_b/probe_env",
    receipt_path = "docs/dev-log/core070/s4-public-phylo-dep-probe-receipt-2026-09-24.json",
    rscript_executable = "/usr/local/bin/Rscript",
    gllvm_root = pwd());
  print(s4_public_phylo_dep_preflight_summary(r.report));
  s4_public_phylo_dep_run!(r.cfg);
  println("S4_PUBLIC_PHYLO_DEP_PROBE_DONE receipt=", r.cfg.receipt_path)'
```

The first attempt with this command (tip `907c59b9f`) aborted inside the harness before R started: `Cmd(::Vector{String}; dir=)` has no method on Julia 1.10. Fixed in `943a3be49` (`s4_public_phylo_dep_r_command()`, with a unit test), then rerun. That abort wrote nothing and reserved no receipt path.

## Shim deviation (option A)

- **What.** `tools/destination_b/probe_env/GLLVM/` is a local package named `GLLVM` (fresh UUID `530e1681-f0ef-4870-b414-280ff3d1dd3b`). Its module body is `using GLLVModels: GLLVModels, bridge_fit; export bridge_fit`. It is developed into `tools/destination_b/probe_env` only, and its own directory has a gitignored Manifest that develops the repo root.
- **Why.** The package was renamed to GLLVModels (#423). The frozen recorder still runs `using GLLVM` (test file lines 209, 273, 368, 437; `R/julia-bridge.R` line 305; the runner's clean-Julia probe) and calls exactly one name, `GLLVM.bridge_fit` (`R/julia-bridge.R` lines 2753, 3263).
- **Second part of the same deviation.** The recorder asserts `Base.pkgdir(GLLVM) == GLLVM_DESTINATION_B_PROJECT` and that the embedded active project is `<that path>/Project.toml`. With a shim, the only path that satisfies both is the shim directory, so `--julia-project` is the shim, not the repo root. The recorder therefore attests the shim root. The GLLVModels code under test is still pinned: the recorder's source snapshot takes `git rev-parse HEAD` and a repo-wide clean `git status` from that path, which is this worktree (commit `943a3be49`, clean).
- **Containment evidence** (lane log, 2026-09-24T16:58:52Z):
  - repo root: `julia --project=. -e 'using GLLVM'` fails with `Package GLLVM not found in current path` (exit 1).
  - probe env: `pkgdir(GLLVM)` = `.../tools/destination_b/probe_env/GLLVM`; `pathof(GLLVModels)` = `<worktree>/src/GLLVModels.jl`; `GLLVM.bridge_fit === GLLVModels.bridge_fit` is `true`; `names(GLLVM)` = `[:GLLVM, :bridge_fit]`.
  - shim env: active project `.../probe_env/GLLVM/Project.toml`; `pathof(GLLVModels)` = `<worktree>/src/GLLVModels.jl`.
  - root `Project.toml` diff is empty; no version bump; the shim is not registered anywhere.

## Outcome (official run)

| Check | Pass/Fail | Notes |
|-------|-----------|-------|
| Preflight (recorder pin, paths, template) | Pass | recorder HEAD `97214679c`; remote-tip check skipped (see below) |
| Clean-Julia probe (runner stage 1) | Pass | `S4_JULIA_ENVIRONMENT_CLEAN`; `package_root` = shim |
| Sealed gllvmTMB load (runner stage 2) | Pass | loaded from the seal's isolated library |
| Test "generic engine remains closed" | Fail (error) | `could not find function "test_that"` |
| Test "paired transformed-Wald endpoints" | Fail (error) | same error; no fit ran |
| Recorder runner exit 0 | Fail | exit 1 |
| Receipt JSON present | Fail | recorder wrote `FAILED.json` instead, as designed |

### Endpoint table (official run)

| Target | Native lower/upper | Julia lower/upper | Max abs delta | Gate 1e-4 |
|--------|--------------------|-------------------|---------------|-----------|
| `beta[1]` | not produced | not produced | n/a | n/a |
| `beta[2]` | not produced | not produced | n/a | n/a |
| `phylo_cov[1,1]` | not produced | not produced | n/a | n/a |
| `phylo_cov[2,1]` | not produced | not produced | n/a | n/a |
| `phylo_cov[2,2]` | not produced | not produced | n/a | n/a |
| `residual_var_shared[1]` | not produced | not produced | n/a | n/a |
| `residual_var_shared[2]` | not produced | not produced | n/a | n/a |

## Failure classification

- [ ] Julia surface gap
- [ ] Recorder drift vs `97214679c` (none: the recorder is exactly the pin)
- [x] R-oracle / frozen-R defect: **the recorder's runner, not its numerical oracle.** It parses the test file and runs each selected `test_that(...)` call through `testthat:::test_code(expr, globalenv())`. It loads testthat with `requireNamespace()` but never attaches it, and `Rscript --vanilla` attaches nothing extra, so `test_that` is not found. This is in `tests/testthat/run-destination-b-s4-public-phylo-dep-isolated.R` at `97214679c` and cannot be fixed from this repo (D-220).

No tolerance was changed.

## Diagnostic run (not a receipt; shown only to size the next decision)

To see what sits behind the runner defect, one more run set `R_DEFAULT_PACKAGES=datasets,utils,grDevices,graphics,stats,methods,testthat`, so R attaches testthat at startup. The recorder, shim and tolerances were unchanged. The receipt path was a scratch directory outside the repo. This environment change was not part of the maintainer's option A. The run is therefore diagnostic only and is not counted in `S4_RESULT`. Its `FAILED.json` (sha256 `0850fccc…31d1`) is kept in the lane log directory, not committed.

Result: 2 tests selected; expectations 6 pass, 1 fail, 0 error.

| Expectation (in order) | Result |
|------------------------|--------|
| generic engine refuses `phylo_dep` with `GJL-GATE-STRUCTURED-TERMS` | pass |
| embedded active project = `<shim>/Project.toml` | pass |
| embedded `pkgdir(GLLVM)` = shim | pass |
| interval row names = the 7 targets in order | pass |
| `julia$phylo_covariance` equals native `Sigma_phy`, tolerance 5e-6 | **fail** |
| Julia lower endpoints equal native lower, tolerance 1e-4 | pass |
| Julia upper endpoints equal native upper, tolerance 1e-4 | pass |

**This run's embedded Julia did not meet the recorder's own qualification standard.** The embedded Julia logged six `Error during loading of extension LogExpFunctionsInverseFunctionsExt ... loglogistic not defined` blocks. That is exactly the pattern the runner's stage-1 clean-Julia check rejects (runner line 318 matches `loglogistic not defined`). Stage 1 passed only because it checks a separate fresh process, not the embedded one. Both probe Manifests pin LogExpFunctions 0.3.29, but the default `~/.julia/environments/v1.10` carries LogExpFunctions 0.3.26 and RCall, so the embedded process very likely loaded a mix of the two versions (JuliaCall loads RCall and its dependencies from `@v1.10` before the shim project is activated). The errors did not stop the run, but every number below comes from an environment the recorder would not qualify.

The failing gate: Julia `(0.2256317, 0.0969768, 0.0969768, 0.0416807)` against native `(0.2256295, 0.0969762, 0.0969762, 0.0416806)`. The largest absolute difference is 2.2e-6 and the mean relative difference is 7.6e-6, above testthat's 5e-6. The recorder's own comment records 9e-7 when it set that gate against the pre-rename Julia package. Two competing hypotheses, neither tested:

1. a change in where the Julia optimizer stops since the gate was set (a Julia-side cause);
2. environment contamination: mixed LogExpFunctions versions (0.3.26 from `@v1.10` against the pinned 0.3.29) in the embedded process changed the numerics.

The two 1e-4 "passes" are testthat `expect_equal()` checks, which compare a mean relative difference across all seven targets. They are not the per-target maximum absolute delta that the endpoint table above uses as its gate. The per-target endpoint numbers were not retained, because the runner stops before it reads them. So these passes are not endpoint evidence, and nothing in this section says the 1e-4 endpoint gates passed.

Correction: the message of commit `8471f1fa3` says the diagnostic run "passed the 1e-4 endpoint gates". That overstates it, for the reasons just given. The branch is pushed and is not rewritten; any squash or merge message for this work must not repeat that wording.

## Other findings

- The harness's remote-tip check runs `git ls-remote origin <recorder branch>` from the GLLVModels.jl worktree, so it asks the wrong repository and always reports "skipped". The recorder pin was still confirmed locally against `HEAD`.
- `run_s4_public_phylo_dep_probe.jl` cannot take the shim as `--julia-project` (see Commands). It was not changed here.

## Checks

- `test/test_destination_b_s4_public_phylo_dep_probe_harness.jl`: 22 of 22 pass (includes the new `s4_public_phylo_dep_r_command` test, which failed before the fix).
- Probe env setup, fresh clone of `8471f1fa3` (no Manifests), under the lane semaphore: the old one-step `Pkg.develop(path="../../.."); Pkg.instantiate()` from `tools/destination_b/probe_env` fails with `expected package GLLVM [530e1681] to be registered`; the two-step setup now in the runbook and `probe_env/Project.toml` instantiates both environments, `using GLLVM` resolves to the shim, and `git status --porcelain` stays empty.
- Recorder worktree after all runs: `git status --porcelain` empty, `HEAD` `97214679c94cc4a6b9e02d3c2b03ccce516027d8`.

## Follow-up (needs Shinichi)

- [ ] Choose a remedy for the runner defect. The recorder is frozen, so any remedy is either an environment-level deviation recorded in the receipt (for example `R_DEFAULT_PACKAGES` attaching testthat, as in the diagnostic) or a new recorder commit. Note that `R_DEFAULT_PACKAGES` alone would not give a qualified embedded environment: the diagnostic also showed the embedded Julia mixing LogExpFunctions versions from `@v1.10`, so any rerun also needs the embedded process kept off the default environment (for example a `JULIA_LOAD_PATH` or depot that excludes `@v1.10`, recorded as a deviation) and a check that the embedded log has no `loglogistic not defined` line.
- [ ] Then decide on the `phylo_covariance` 5e-6 gate result above. Do not widen it. First find out which optimizer is further from the optimum.
- [ ] Pending board / paste packet update (docs PR only).
- [ ] GOAL QS4 checkbox (maintainer only; stays open).
