# After-task: closeout OWED 2 and 4 decided (A, A) and implemented (2026-09-24)

Lane: `claude/owed-2-4-decisions-20260924`, from `origin/main` @ `4e6e7eef5`.
Decision record: `docs/dev-log/decisions/2026-09-24-parity-reference-julia-and-fixture-pins.md`.

## 1. Goal

Record Shinichi's answers to OWED items 2 and 4 of
`docs/dev-log/handover/2026-09-24-claude-handover-closeout.md` ("OWED 2: A, OWED 4: A") and
carry them out: Julia 1.10 LTS is the reference for #323 evidence, the NB2 fixture is stored
as data, the Student cell records R's gradient, and the Stage 0 L11 sign stays with a comment.

## 2. Implemented

- **Decision record** (new), with the options, the Track A versus CI holdout table, and a
  measured hash table for the NB2 data on four Julia versions.
- **NB2 stored data.** New `test/parity/fixtures/nb2_original_data.toml` holds the draw that
  matches the existing pin `7abde273…`, drawn on Julia 1.13.0 by the unchanged seeded code.
  New `parity_nb2_original_Y()` in `test/parity/nb2_health.jl` reads it and refuses unless the
  data hash equals both the file's own `data_sha256` and the pin. `test_negbin_parity.jl` keeps
  the seeded loop, so the random-number state before the fit is what it was on every version,
  then sets `Y = parity_nb2_original_Y()`. `test_nb2_formula_parity.jl` imports the loader
  into its module, because it evaluates the same code text there.
- **Student Parity Cell 9** (`test_studentt_parity.jl`) reads R's gradient right after its own
  R fit and prints `gllvmTMB r_gradient_max = … (recorded, not a gate)`.
- **Stage 0 fixture comment** on `loading_profile_fixture_mask_b_pins()`, placed above the
  docstring so the docstring still attaches, naming R's +0.8 and where it is recorded.

## 3a. Decisions and Rejected Alternatives

- Shinichi chose A for both items; the record lists B for each.
- **Rejected: delete the seeded loop.** Keeping it keeps the code-text hashes meaningful as a
  record of how the data were drawn, and keeps the random-number state that the fit sees
  unchanged on each Julia version. Only `Y` changes.
- **Rejected: add the new data file to the execution-inventory lists**
  (`_core070_execution_paths` in `parity_helpers.jl`, `EXECUTION_STATIC` in
  `tools/core070_evidence.py`). The loader already pins the data content by hash, which is
  stronger than a file hash, and growing those lists would stop the readback verifiers from
  accepting older retained runs.
- **Rejected: add `r_gradient_max` to the shared Student helper.** Reading it in Cell 9 only
  keeps the other Student cells unchanged.

## 4. Files Touched

- `docs/dev-log/decisions/2026-09-24-parity-reference-julia-and-fixture-pins.md` (new)
- `test/parity/fixtures/nb2_original_data.toml` (new)
- `test/parity/nb2_health.jl`
- `test/parity/test_negbin_parity.jl`
- `test/parity/test_nb2_formula_parity.jl`
- `test/parity/test_studentt_parity.jl`
- `test/parity/fixtures/loading_profile_confirmatory_substrate.jl` (comment only)
- `docs/dev-log/after-task/2026-09-24-owed-2-4-decisions.md` (this file)
- `docs/dev-log/check-log.md`
- `AGENTS.md` (Phase state snapshot line only)

## 5. Checks Run

Mac Studio (arm64), single-threaded BLAS and Julia.

| Check | Result |
|---|---|
| NB2 seeded draw, hashed on Julia 1.10.0, 1.10.12, 1.12.6, 1.13.0 | `af68c91f…` on both 1.10s; `7abde273…` (the pin) on 1.12.6 and 1.13.0 |
| `parity_nb2_original_Y()` on Julia 1.10.12 and 1.13.0 | 5 × 80, `Int64`, sum 1071, hash equals the pin on both |
| Stage 0 fixture loads; `@doc` still attached | L11 = −0.8, docstring intact |
| Oracle `b4d5fee64` built with `tools/core070_build_oracle.py` | prepare, build (86 s), verify: all PASS |
| Required mode, Julia 1.10.12, `CORE070_PARITY_CASE_IDS=NATIVE-06-NB2,NATIVE-10-STUDENT` | 46 pass, 5 fail, 41 s wall (estimate was 10 to 25 min) |
| `tools/core070_verify_nb2_health.checks()` on the new `nb2-health.toml` | structural assertions pass; gates as in the table below |

Time estimate written before the runs: 10 to 25 minutes in total, including the environment
and the oracle. Actual: about 4 minutes.

**NATIVE-06 (NB2), 16 of 18.** The data-hash guard now passes on Julia 1.10.12, so the cell
reaches its R check for the first time on this version. Two checks fail:

| Quantity | Value | Gate |
|---|---|---|
| Δ logLik (Julia − R) | 1.21e-7 | passes |
| same-point density Δ | 1.36e-12 | passes |
| Julia gradient max / FD stability | 1.34e-6 / 6.2e-8 | pass |
| Julia `converged` | false | **fails** |
| R optimizer code / message | 0, "relative convergence (4)" | passes |
| R `r_gradient_max` | 4.85e-3 | **fails** (≤ 1e-4) |

**NATIVE-10 (Student), 30 of 33.** Parity Cell 9 passes (Δ logLik 1.98e-8, R optimizer code
0) and now records `gllvmTMB r_gradient_max = 9.15e-4`. The three failures are all in the
near-Gaussian estimated-ν diagnostic: R `converged` false with optimizer code 1, and Julia
`converged` false (Δ logLik −1.45e-5). On Totoro, Julia 1.10.12, only the Julia flag failed
in that diagnostic (Track A receipt, 32 of 33).

What this adds to the picture:

- **NB2's R-side failure is not a Julia-version effect.** R's fit does not involve Julia. On the
  same data, R stops at `r_gradient_max` 4.85e-3 on this Mac and 2.43e-3 on CI (Linux), and
  the CI workflow notes a retained Totoro receipt for the same data that passed 18 of 18. R's
  optimizer end point depends on the machine.
- **NB2's Julia flag does depend on the version.** On the same data the CI job (Julia 1.13.0)
  reports only the R-side failure, while Julia 1.10.12 here reports `converged = false` with a
  gradient of 1.3e-6.
- Student Cell 9's R gradient (9.15e-4) would not meet the 1e-4 bar other cells use; it is
  recorded only, as decided.

## 6. Tests of the Tests

- The loader's hash check is the same check that refused NATIVE-06 on Totoro. On Julia
  1.10.12 the seeded code gives `af68c91f…`, which that check refuses; the stored draw gives
  `7abde273…`, which it accepts. Both were run.
- A stored file edited by hand would fail the loader's own check before any fit runs.

## 7a. Issue Ledger

- Resolved: closeout OWED 2 and OWED 4 (decisions recorded and carried out).
- Resolved: Track A receipt's open item "decide how the NB2 fixture should be pinned across
  Julia versions, and re-run that cell" (Mac re-run; see section 5).
- Still open: OWED 1 (gllvmTMB#1283 has no new recorder commit).

## 8. Consistency Audit

- Other fixtures that draw data from a seed with Julia's default generator could split the
  same way across versions. Not audited here; the NB2 case was the only one that tripped a
  data-hash guard in Track A. `poisson_beta_health.jl` writes its drawn data to a receipt file
  but still redraws from the seed on each run.
- The committed `aghq-*-evidence.json` and `family-boundary-*.json` files pin an older hash of
  `test_negbin_parity.jl` (`db83f338…`) that the 2026-09-18 rename had already made stale.
  Nothing checks those pins against the live file; they are history.

## 9. What Did Not Go Smoothly

- The Mac's existing frozen library had no `CORE070_SOURCE_PIN.toml` marker, which the NB2
  health report requires. The oracle was rebuilt locally with `tools/core070_build_oracle.py`
  (prepare, build, verify all passed; archive hash `0c2f4323…` as pinned; 86 s build).

## 10. Known Residuals

- NATIVE-06 and NATIVE-10 still fail on Julia 1.10.12. The data fix removes the guard failure
  only; it does not make either cell pass.
- These results are from the Mac (arm64). The reference platform for #323 evidence is still
  Totoro on Julia 1.10.12, which has not been re-run with this change.
- `tools/core070_verify_student_refinement.py` pins `test_studentt_parity.jl` to `484aac83…`.
  That pin was already stale on `main` after the rename (`main` has `5da7a5c9…`), and nothing
  runs the script, so this change does not break a live check.
- The core suite does not include the parity files, so CI's core jobs do not exercise these
  edits; the advisory Frozen R job does.

## 11. Team Learning

When a seeded fixture has to give the same data on every Julia version, store the draw and
pin its hash. The seed alone is not enough: Julia 1.10 and 1.12+ draw different numbers from
`Random.seed!(45)` here.

## 12. Cross-Product Coverage

The reference-version decision is cross-cutting across all #323 holdouts.

- Covers ✓: NATIVE-06 (data now fixed across versions) and NATIVE-10 (gradient recorded).
- This change does NOT cover: NATIVE-12's R-side gradient on 1.10.12, a Totoro re-run, any
  other seeded fixture, or the CI advisory job's Julia version (it stays 1.13.0 and advisory).

Memory receipt: `route.py` has no LOAD-FIRST manifest for this repo. Applied: this repo's
`AGENTS.md` (no push without instruction, stage by name, check-log and after-task on every
change), the closeout handover, a lane lease on the touched paths (D-88), and a time estimate
before each run (D-139).

Golden Set: not in scope. No memory or retrieval class was touched.
