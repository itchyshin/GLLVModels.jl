# After-task: second-order tools honour GLLVM_PARITY_R_LIBS (2026-09-24)

Lane: `claude/second-order-r-libs-guard-20260924`, from `origin/main` @ `4e6e7eef5`.
Closes OWED item 3 of `docs/dev-log/handover/2026-09-24-claude-handover-closeout.md`.

## 1. Goal

Make `tools/core070_second_order/common.jl` load gllvmTMB from `GLLVM_PARITY_R_LIBS`
when that variable is set, or stop with an error. Before this change the variable was
ignored, so a run that set only it silently used R's default gllvmTMB (found during the
Delta A D1 remeasure, `2026-09-24-delta-dispersion-a-d1-remeasure.md`).

## 2. Implemented

- New `tools/core070_second_order/r_lib.jl`: `second_order_r_lib(env = ENV)`, pure Julia,
  no RCall. Unset or blank variable: returns `nothing` (R's own `.libPaths()` decides, as
  before). Set: the library must contain `gllvmTMB`, otherwise `ArgumentError`; returns
  `realpath(lib)`.
- `_require_gllvmtmb!()` in `common.jl` now calls it. With a library named, the R block
  refuses if gllvmTMB is already loaded from somewhere else, puts the library first on
  `.libPaths()`, runs `library(gllvmTMB, lib.loc = lib)`, and checks the loaded namespace
  path afterwards. The block runs inside `local()`, so its helper names stay out of R's
  global environment. With no library named, it runs the same bare `library(gllvmTMB)` as before.
- New `test/test_second_order_r_libs.jl`, wired into `test/runtests.jl` after the other
  second-order files. One testset always runs (6 checks, no R). One runs only with
  `GLLVM_PARITY_TESTS=1` and `GLLVM_PARITY_R_LIBS` set (3 checks against live R).

## 3a. Decisions and Rejected Alternatives

- **Unset keeps the old behaviour.** Failing whenever the variable is unset would break
  every existing local and CI call of these tools, and the handover asked for "honour it
  or fail closed", not "require it". Setting the variable is now enough to pin the library.
- **The choice lives in its own file.** `common.jl` uses `R"..."` string macros, so the
  core suite cannot include it without RCall, and CI's core jobs cannot load RCall. Putting
  the choice in `r_lib.jl` lets the core suite test it on every run.
- **Rejected: copy `_parity_prepend_twin_lib!()` from `test/parity/parity_helpers.jl`.**
  That helper returns quietly when the library has no gllvmTMB, which is the same fall-back
  this task removes. The new code keeps its "already loaded elsewhere" refusal and adds a
  check after loading.
- **Rejected: fix the seven other tools that call a bare `library(gllvmTMB)` in this PR.**
  Out of the handover's scope; listed in section 8 for a separate decision.

## 4. Files Touched

- `tools/core070_second_order/r_lib.jl` (new)
- `tools/core070_second_order/common.jl` (modified: `_require_gllvmtmb!`)
- `test/test_second_order_r_libs.jl` (new)
- `test/runtests.jl` (one `_shard_include` line)
- `docs/dev-log/after-task/2026-09-24-second-order-r-libs-guard.md` (this file)
- `docs/dev-log/check-log.md` (entry at the top)
- `AGENTS.md` (Phase state snapshot line only)

## 5. Checks Run

All on the Mac Studio, Julia 1.10.0, with `JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1` and `R_LIBS` / `R_LIBS_USER` unset. Frozen oracle library:
`~/local-scratch/R-gllvmtmb-frozen-b4d5fee64` (gllvmTMB 0.7.0, built 2026-09-24). Default
library: `~/Library/R/arm64/4.6/library` (gllvmTMB 0.7.1, built 2026-09-14). Their
`gllvmTMB.so` SHA-1 prefixes differ (`e259de70494c` vs `98e39beaccaa`).

| Command | Result |
|---|---|
| `julia --startup-file=no test/test_second_order_r_libs.jl` (no parity variables) | 6 pass, 1 skipped (live-R testset) |
| same, with `GLLVM_PARITY_TESTS=1 GLLVM_PARITY_R_LIBS=<frozen>` | 6 pass + 3 pass |
| same, but with `origin/main`'s `common.jl` | 6 pass + **2 fail** (see section 6) |
| `smoke_delta_lognormal_eoo.jl`, only `GLLVM_PARITY_R_LIBS=<frozen>` set | PASS, 27.8 s |
| `smoke_delta_lognormal_eoo.jl`, no library variables (default gllvmTMB) | PASS, 29.7 s |

Time estimate written before the smoke runs: 5 to 15 minutes including a first precompile.
Actual: 15 s instantiate and precompile, then about 32 s per smoke run. After the final
`local()` edit, the live-R testset (6 + 3 pass, no overwrite warnings in the full log) and the
frozen-library smoke (PASS, identical to the receipt) were run again.

Rose (independent read-only review, Sonnet): **ROSE: OK**, no blockers. Two of its notes were
checked and not acted on: `.libPaths()` already drops duplicate entries, and loading
`r_lib.jl` twice printed no warning in the full log (Julia's default `--warn-overwrite=no`).
Its third note, that CI's parity job never reaches this code and CI's core job runs only the
pure testset, is recorded in section 10.

The full `Pkg.test()` suite was not run (about 50 minutes; see section 10).

## 6. Tests of the Tests

The live-R testset was run against `origin/main`'s `common.jl` with the frozen library
named. It failed on both checks, and the failure shows the bug:

- `@test_throws ArgumentError _require_gllvmtmb!()` with an empty library: no exception.
- loaded namespace path: `~/Library/R/arm64/4.6/library/gllvmTMB`, expected
  `~/local-scratch/R-gllvmtmb-frozen-b4d5fee64/gllvmTMB`.

The same testset passes with the fix. The third check (a library that differs from the one
already loaded must stop) was added after the red run, so it has no red run of its own.

**The smoke runs do not show which library loaded.** Both runs reproduced the committed D1
receipt `delta-lognormal-2so-d1-remeasure-receipt.json` to every printed digit (R logLik
−744.4500531152822, Δ logLik 1.8173750504502095e-08, SE max relative Δ
4.013227800865021e-05). The frozen 0.7.0 build and the default 0.7.1 build give the same
answer on this cell. They show that the real tool path still runs with only the variable
set; the namespace-path check above is the evidence for routing.

## 7a. Issue Ledger

- Fixed: the FINDING-OF-RECORD "`tools/core070_second_order` ignores `GLLVM_PARITY_R_LIBS`"
  (closeout handover). No GitHub issue was filed for it.
- Open, not touched: gllvmTMB#1283 (S4 recorder), which still has head `97214679c` and no reply
  from its owning lane as of this session.

## 8. Consistency Audit

Same class, not fixed here (bare `library(gllvmTMB)` that ignores `GLLVM_PARITY_R_LIBS`):

- `tools/core070_aghq_gaussian_pair_run.jl:36`
- `tools/core070_aghq_binomial_pair_run.jl:48`
- `tools/core070_aghq_poisson_pair_run.jl:46`
- `tools/core070_aghq_public_gaussian_pair_run.jl:38`
- `tools/core070_source_fixed_residual_pair.jl:8`
- `tools/core070_covariance_mode_fits.jl:83`
- `tools/d220_paired_gaussian_cell.jl:39`

A near relative: `_parity_prepend_twin_lib!()` (`test/parity/parity_helpers.jl:232`) reads the
variable but returns quietly when the named library has no gllvmTMB.
`test/parity/README.md` documents that fall-back for the historical `/tmp` default. Required
runs (`CORE070_PARITY_REQUIRED=1`) go through a source-pin marker instead and do not fall back.

Consumers of the variable checked: `.github/workflows/CI.yml:183` and
`docs/dev-log/core070/totoro-323-track-a-20260924/trackA.sh:25` set one directory that holds
gllvmTMB, so the new refusal does not fire for them.

**Side finding, the 2026-09-15 D1 FAIL.** The R side of both Delta cells has not moved.
`check-log.md` for 2026-08-28 records R logLik −744.4500531 (delta_lognormal) and
−725.9597544 (delta_gamma). Today's frozen build gives −744.4500531152822 and
−725.9597544173939, and today's default build gives the same lognormal value. The −1.923
logLik Δ reported on 2026-09-15 equals the 2026-08-28 shared-dispersion Julia result against
that same R value. So the move to D1 PASS comes from the Julia `disp_group` change, not from
which R library was loaded. Limit: the 2026-09-15 R logLik itself was not recorded. The
conclusion rests on the R value agreeing across two builds and across dates. That
agreement is measured; the link to the 2026-09-15 run is inferred.

## 9. What Did Not Go Smoothly

- The session opened in the gllvmTMB checkout. The handover lives in GLLVModels.jl, so the
  first step was finding it (Spotlight, then the merged closeout worktree).
- A first reading of the smoke match as proof of routing was wrong; the default-library run
  showed the two builds agree on this cell.

## 10. Known Residuals

- Full `Pkg.test()` (Aqua, JET, every shard) was not run locally. The change adds one
  test file and one tools file with no package code; CI's shards run it on push.
- The live-R testset needs R, RCall and a named library, so CI's core jobs only run the
  pure testset.
- The seven same-class tools in section 8 still ignore the variable.

## 11. Team Learning

When two R builds are candidates, a receipt that matches exactly does not tell you which
build ran, because different builds can give identical answers. Check the loaded namespace
path (`getNamespaceInfo("gllvmTMB", "path")`), or run the other build and see whether
the numbers move.

## 12. Cross-Product Coverage

The R library choice is cross-cutting across every tool that fits the R twin.

- Covers ✓: every caller of `_require_gllvmtmb!` in `tools/core070_second_order/` (the
  `r_fit_se*` helpers, the smoke drivers, `run_cell.jl`, `run_matched_batch1.jl`) and the
  `test/test_second_order_*` files that include `common.jl` under `GLLVM_PARITY_TESTS=1`.
- This change does NOT cover: the seven tools in section 8, `test/parity/parity_helpers.jl`'s
  quiet fall-back, the S4 recorder in gllvmTMB (read-only from here), or any Totoro or DRAC
  launcher that sets `R_LIBS` or `R_LIBS_USER` directly.

Memory receipt: `route.py` has no LOAD-FIRST manifest for this repo, so none was loaded.
Applied: this repo's `AGENTS.md` (no push without instruction, stage by name, check-log and
after-task on every change), the closeout handover, `lane_preflight.sh` and a lane lease on
the touched paths (D-88), and a time estimate before each run (D-139).

Golden Set: not in scope. No memory or retrieval class was touched.

Validator note: `check-after-task.R` prints "after-task structure check passed" for this file,
then halts on a tracked `.unlazy` acceptance ledger from another lane. The closeout handover
already lists those ledgers as PROTECTED foreign work.
