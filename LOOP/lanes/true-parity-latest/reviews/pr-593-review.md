# Review of PR #593 — P1 Gaussian grouping-level paired receipts (unit, unit_obs, cluster, cluster2)

- Branch `claude/true-parity-p1-grouping-receipts`, head `dcd50098489c2e8392278a7222e5925db8167ad8`, base `main` (merge-base `85b7a688d`).
- Reviewer worktree: detached at `dcd500984` under `local-scratch/lanes/GLLVM.jl-review-593` (removed after review). Threads capped `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4`.
- Oracle: `a3cov-oracle/build/library` (read-only), `build.json` names P1 `9539352f…`, exit 0, gllvmTMB 0.7.1, R 4.6.0.
- Reviewed 2026-09-28. Nothing edited, pushed, commented or merged.

## Verdict: NON-BLOCKING

The receipts are real, reproduce exactly, compare the same 5-parameter model on both engines, discriminate under tampering, and carry the comparison block the #561 checker requires. The two findings below are hygiene (host paths in tracked outputs; a `--check` exit code that does not encode the paired verdict), not correctness.

## Findings

### 1. Receipts reproduce (evidence: re-run in the review worktree)

`python3 tools/core070_grouping_p1_receipts.py run` at `dcd500984` (after `Pkg.instantiate()`; a fresh worktree needs it, the runner does not do it for you) re-ran both engines. `git diff` against the committed outputs shows changes only in: `fit_seconds`/`wall_seconds`/`finished_utc`, `glvmodels_commit` (`db5b9ed2e` -> `dcd500984`, expected: the run commit was the runner commit, HEAD adds only the receipts), the `read_from` hashes that follow from those, and two host-path strings (Finding 5). Every R and Julia numeric value (logLik, beta, Sigma_diag, sigma_eps, membership-control logLik, negative-control messages) is byte-identical. Batch verifier 22/22 PASS. Printed gaps identical to the PR table: unit `logLik 1.92e-10, beta 2.10e-06, Sigma_diag 9.40e-07, sigma_eps 4.15e-07`; unit_obs `1.34e-12 / 2.65e-08 / 9.59e-08 / 1.69e-08`; cluster `2.43e-12 / 6.50e-08 / 2.59e-07 / 1.29e-08`; cluster2 `5.14e-12 / 4.48e-07 / 6.24e-08 / 2.57e-08`.

`--check` and `fixtures.py --check` pass on the committed tree.

### 2. R values are R at P1, not Julia (evidence: runner code + oracle receipt)

`tools/core070_grouping_p1_r.R` refuses unless `build.json` names P1 with exit 0, loads with `lib.loc = <oracle>/library`, and aborts if `find.package("gllvmTMB")` is not the oracle path. `r-results.json` records `loaded_path` = oracle library, version 0.7.1, `oracle_installed_tree_sha256` matching `build.json`. The `run-commit.json` `oracle_build_json_sha256` (`0a648dc0…`) matches the current `build.json` on disk. The R log carries gllvmTMB's own extract_Sigma messages ("Cluster (third-slot) tier…", "cluster2 (second diagonal grouping) tier…"), so extraction went through the R package, not a Julia call.

### 3. Same model on both engines at every level, including cluster and cluster2 (evidence: R parameter vector, P1 source, Julia source)

I refit each level in R from the oracle library and printed `names(fit$opt$par)`:

| level | R fixed parameters (5) | R random slot | Julia |
|---|---|---|---|
| unit | `b_fix` x2, `log_sigma_eps`, `theta_diag_B` x2 | `s_B` (1..12) | `GroupingTerm(:unit; mode=:indep)` |
| unit_obs | `b_fix` x2, `log_sigma_eps`, `theta_diag_W` x2 | `s_W` (1..24) | `GroupingTerm(:unit_obs; mode=:indep)` |
| cluster | `b_fix` x2, `log_sigma_eps`, `theta_diag_species` x2 | `q_sp` (1..6) | `GroupingTerm(:cluster; mode=:indep)` |
| cluster2 | `b_fix` x2, `log_sigma_eps`, `theta_diag_cluster2` x2 | `r_c2` (1..6) | `GroupingTerm(:cluster2; mode=:indep)` |

`logLik` df = 5 at every level, matching Julia's 2 intercepts + 2 diagonal variances + sigma_eps. No hidden shared or extra term: `use_diag_B`/`use_diag_W`/`use_diag_species`/`use_diag_cluster2` are set by the single `indep()` term only, and the P1 roxygen (`R/gllvmTMB.R` lines 134-179) documents `cluster` as the third grouping slot (non-phylogenetic when no `phylo_vcv`/`phylo_tree`) and `cluster2` as a plain crossed/nested per-trait diagonal, "exactly as `cluster` does". Julia's `fit_grouped_gaussian` (`src/grouped_fit.jl:359`) takes labels for incidence only; one `GroupingTerm` -> one per-trait diagonal covariance; `cluster2` is restricted to `mode=:indep`.

One asymmetry worth naming: on `cluster`/`cluster2` R is passed `unit = "unit"` (required by gllvmTMB; the four unit labels recur across clusters), Julia is passed no `unit`. This does not change the model: R's `site_id` builds `n_sites = 4` but adds no parameter (5 free parameters, `s_B` unused), and the two engines' logLik agree to 2.4e-12 / 5.1e-12 with Julia never seeing the unit labels. The receipt `call` strings record the difference honestly.

`Sigma_offdiag` is 0 on both engines by construction; the receipt records it and does not compare it (correct: a structural zero would be a vacuous match). The compared quantities are all order 0.1-0.6, none near zero or constant.

### 4. The comparisons discriminate; tolerances are not papering over a gap (evidence: mutation + tight-tolerance R refit)

Mutations (in-memory `derive()` on the tracked raw outputs): +1% on Julia `Sigma_diag[1]` flips every level to FAIL (diffs 5.8e-3, 5.1e-3, 1.5e-3, 1.4e-3 vs tol 1e-5); +1% on `sigma_eps` -> FAIL; +2e-6 on logLik -> FAIL. Editing one Julia value in the tracked `julia-results.toml` makes `--check` exit 1 (receipt differs from re-derivation; `read_from` hash mismatch).

The unit `beta` gap (2.1e-6): I refit R at `unit` with `optArgs = list(control = list(rel.tol = 1e-14, abs.tol = 0, x.tol = 1e-14, iter.max = 5000, eval.max = 5000))`. R max |gradient| fell 1.58e-4 -> 1.41e-6 and the gap to Julia's committed values shrank: logLik `-73.391946319803225` vs Julia `-73.39194631980321` (1.5e-14); beta2 `-0.62001558590` vs `-0.62001556728` (1.9e-8); Sigma_diag gaps 8e-9 / 1.7e-8; sigma_eps 4e-9. So the 2.1e-6 is R's early stop, as the PR says, and the 1e-5 parameter tolerance is a convergence-scale allowance, not a cover.

### 5. Host paths in tracked outputs (hygiene; NON-BLOCKING)

`main`'s `true-parity-latest/` receipts contain no `/Users/…`, `local-scratch` or hostname strings (only `GATES.md` has a `~/local-scratch` lane pointer). This PR adds several:

- `run/run-commit.json`: `"host": "w-kw3k3y6229.psych.ualberta.ca"`, `"oracle_build_dir": "/Users/z3437171/local-scratch/a3cov-oracle/build"`.
- `run/r-results.json`: `"loaded_path": "/Users/z3437171/local-scratch/a3cov-oracle/build/library/gllvmTMB"`.
- `run/julia-results.toml`: `glvmodels_path = "/Users/z3437171/local-scratch/grouping-receipts-wt/src/GLLVModels.jl"`.
- The four receipts `{unit,unit_obs,cluster,cluster2}.json` and `julia-results.toml`: the negative-control `message` embeds `@ GLLVModels ~/local-scratch/grouping-receipts-wt/src/grouped_fit.jl:359` (Julia's MethodError text). These are also worktree-dependent, so a re-run from another checkout changes the receipt bytes and `read_from` hashes even when every number is identical (observed in my re-run).
- `tools/core070_grouping_p1_receipts.py:47`: `DEFAULT_ORACLE = "/Users/z3437171/local-scratch/a3cov-oracle/build"` hard-coded (overridable with `--oracle`).

Suggested fix if the maintainer wants this clean before any row is cut from it: record `rejected: true` plus the exception type only (or strip the `@ …` location line), drop `host`, and make the oracle dir an env/flag with no default. None of this affects the numbers.

### 6. `--check` exit code does not encode the paired verdict (hygiene; NON-BLOCKING)

`check()` re-derives receipts and exits 1 on drift, but prints `unit: PASS/FAIL` informationally: a run that had legitimately produced `FAIL` receipts would still make `--check` exit 0 as long as stored == re-derived. Not a problem for these four (all PASS, and the #561 checker reads `verdict` itself), but the docstring's "A tampered Julia logLik makes it fail" holds via hash/derivation drift, not via the verdict.

### 7. Receipt shape passes the #561 checker (evidence: C1 on a throwaway case map)

Temp copy of `tools/true_parity_check.mjs` from `origin/claude/true-parity-p1-namespace-v2` (deleted after), run in `PARITY_REF=FS` mode on a throwaway root holding only the four receipts and a four-row case map (`evidence_tier: "numeric"`, `executable_case_ids` = the five `case_id`s per receipt, `measured_against` = P1):

`C1 required=4 bound=4 bound_numeric=4 … numeric_recorded_diff_mismatch=none` -> `C1_MET`.

Negative: editing `julia_value[0]` of the cluster `Sigma_diag` case by 1% while leaving `max_abs_diff` stale -> `numeric_recorded_diff_mismatch=grouping/THROWAWAY-cluster(… recorded 2.59e-7 != recomputed 1.47e-3 …)` -> `C1_NOT_MET`. So the block has `pin: "P1"`, per-case `case_id`, `tolerance > 0`, `r_value`/`julia_value` vectors of equal length, a recomputable `max_abs_diff`, and a top-level `verdict: "PASS"`; the checker's recompute cross-check works on it.

### 8. Scope, body, handles

- `git diff --stat 85b7a688d..HEAD`: 19 files, all under `docs/dev-log/core070/true-parity-latest/receipts/grouping/` and `tools/core070_grouping_p1_*`. No `src/`, `Project.toml`, `.github/`, `GATES.md`, `case-map.json`, `scoreboard.md`, or P0 evidence touched.
- PR body: numbers match the receipts to the digit; the "honest note" on the unit beta gap is accurate and, per Finding 4, slightly conservative (it is R's stop, and it closes). "Not covered" matches `NOT_COVERED` in the tool. Draft, "not for merge", no row claimed.
- No agent `@handles` in the PR body or the three commit messages (the only `@` strings are `noreply@anthropic.com` co-author trailers).

## What I did not check

- Whether R's `unit_obs` default (`"site_species"`, absent from the data) or `cluster` default (`"species"`, absent) emits any warning that `r.log` swallowed; the log shows none and the fits converged.
- Julia's final gradient norm is not recorded in the receipt (default `g_tol = 1e-5` in `fit_grouped_gaussian`; the runner does not tighten it). I did not refit Julia with a tighter `g_tol`; the R side of the tight refit was enough to attribute the gap.
- The `extract_Sigma(level = "cluster2", part = "total")` mapping in the P1 R source beyond the roxygen and the log message; the diagonal values agree with Julia to 1e-7, which is the operative check.
- The fixture generator's statistical claims (LCG + Box-Muller) beyond `--check` bytes; the fixtures are literal and hashed, so their origin does not affect the receipts.
- Windows/Linux reproduction; macOS only.
- The full Julia test suite (not touched by this PR; nothing under `src/` or `test/` changed).
