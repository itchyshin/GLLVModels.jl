# PR #502 CI state and cross-check (2026-09-26)

Lens: CI and cross-check only — read-only on GitHub, no local pushes.
PR: itchyshin/GLLVModels.jl#502, `claude/fit-verdict-gradient-485`, head
`802f3965e5f0dd9bcdc32b6b5542eff1b2a25e64`. Fixes #485.

## 1. CI state on the PR's current head

Checked `gh api repos/itchyshin/GLLVModels.jl/commits/802f3965e/check-runs`
repeatedly between 15:36 and 15:51 UTC (~16 minutes from the CI.yml run's own
`run_started_at` of 15:34:39Z, within the 20-minute budget) — a background
`Monitor` poll (30s cadence) ran alongside for the same window and reported no
state change beyond the one recorded below.

**Last observed state (15:50:33Z), 9 required check-runs on this exact head sha:**

| check | status | conclusion |
|---|---|---|
| Documenter | completed | **success** |
| Julia 1 — ubuntu-latest shard 1/4 | in_progress | — |
| Julia 1 — ubuntu-latest shard 2/4 | in_progress | — |
| Julia 1 — ubuntu-latest shard 3/4 | in_progress | — |
| Julia 1 — ubuntu-latest shard 4/4 | in_progress | — |
| Julia 1.10 — ubuntu-latest shard 1/4 | in_progress | — |
| Julia 1.10 — ubuntu-latest shard 2/4 | in_progress | — |
| Julia 1.10 — ubuntu-latest shard 3/4 | in_progress | — |
| Julia 1.10 — ubuntu-latest shard 4/4 | in_progress | — |
| Frozen R 0.7.0 family smoke (advisory; rebuilt oracle) | in_progress | — (`continue-on-error: true`, confirmed in `.github/workflows/CI.yml:109` — will not block even on failure) |

**Reporting plainly, per the task's own instruction: all 8 Julia test shards and
the advisory Frozen-R leg were still `in_progress` at the 16-minute mark, with no
state change across the whole observation window** (only `Documenter` had
completed, at 15:38:29Z, 4 minutes after the run started). This is not a stall —
the repo's two most recent completed `CI.yml` runs took 21+ minutes (still
in-progress when last observed) and ~69 minutes (14:14:25Z→15:23:59Z, concluded
`failure`) respectively, so a sharded 8-way Julia matrix plus an R-toolchain
build commonly runs 45-70+ minutes here; 16 minutes in is not evidence of a
hang. `gh run view --log` refuses to stream a still-running job's log on this
`gh` version ("logs will be available when it is complete"), so no partial
per-shard test output could be inspected while waiting.

**CI is therefore NOT YET fully green (nor fully resolved) as of this report.**
No shard has failed and none has reported a converged-flag flip in a check-run
title/summary — but 8 of 9 required legs have not reported a result yet, so
"CI is fully green apart from the known advisory Frozen R cell" cannot be
confirmed at this time. Recommend re-running `gh api
repos/itchyshin/GLLVModels.jl/commits/802f3965e/check-runs` in ~30-45 minutes,
or watching the run directly: `https://github.com/itchyshin/GLLVModels.jl/actions/runs/36252419813`.

## 2. Cross-check with #501 (ordered beta) via the sibling-screen script

Script read: `.../sibling-screen-2026-09-26-raw/sibling_screen.jl`
(`route_ordered_beta` + `finish_row!`'s FD-gradient/`gscaled` diagnostic). Refit
locally (Mac Studio, Julia 1.10.12, `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`),
using a throwaway worktree at the PR head (`~/local-scratch/review-502-ci-crosscheck`)
and a second at `origin/main` (`~/local-scratch/review-502-main-crosscheck`), both via
`test/parity`'s standalone env (`Pkg.develop(path="../..")`; `SpecialFunctions` added
locally to that throwaway env only — the tracked `test/parity/Project.toml` was not
touched). Estimate before running: seconds-to-low-tens-of-seconds per fit (well under
the 20-90s/Totoro figure given); actual per-fit time was 0.3-3.8s, so the whole
cross-check (both worktrees, seeds 2001/2003/2005 individually plus a
`SCREEN_ONLY=ordered_beta SCREEN_N=10` full-range rerun on each, to check
determinism) completed in well under 3 minutes of Julia compute.

### Branch (#502 head, fix applied)

| seed | converged | loglik | max FD `\|g\|` | `gscaled` |
|---|---|---|---|---|
| 2001 | **true** | -169.0038 | 4.79e-06 | 1.80e-05 |
| 2003 | **false** | -175.1268 | 2.08e+06 | 2.92e+06 |
| 2005 | **true** | -182.8099 | 7.04e-06 | 1.44e-05 |

### Main (`origin/main`, unfixed)

| seed | converged | loglik | max FD `\|g\|` | `gscaled` |
|---|---|---|---|---|
| 2001 | true | -169.0038 | 4.79e-06 | 1.80e-05 |
| 2003 | **true** | -175.1268 | 2.08e+06 | 2.92e+06 |
| 2005 | true | -182.8099 | 7.04e-06 | 1.44e-05 |

Both worktrees produced bit-identical loglik/gradient numbers for each seed
(confirms the RNG is scoped per-seed and per-route — same data, same fit,
only `.converged` differs) — verified by re-running with the seed set widened
to 2001:2010 (`SCREEN_ONLY=ordered_beta SCREEN_N=10`, both worktrees) to rule out
any global-RNG-state dependence on execution order.

**Result vs. the stated expectation — partial match, one discrepancy found:**

- Seed 2001 (clean): `converged == true` on the branch. **Matches expectation.**
- Seed 2003: `converged == false` on the branch, `converged == true` on main, FD
  gradient 2.08e6 (in the stated 1e5-1e8 range). **Matches expectation exactly** —
  this is precisely the fix's intended behaviour: an honesty correction from a
  false "converged" to a correctly-flagged non-convergence at a large gradient.
- Seed 2005: `converged == true` on **both** branch and main, with a **tiny**
  FD gradient (7.04e-6, `gscaled` 1.44e-5) — not in the 1e5-1e8 range. **Does not
  match the stated expectation** ("converged == false for both, gradients 1e5 to
  1e8"). Seed 2005 is, on this evidence, a genuinely well-converged fit, not a
  large-gradient stall; the branch correctly leaves it `converged = true` (it never
  needed the new criterion) and this is consistent behaviour, not a defect in #502.

To characterise how isolated this is, the full `SCREEN_ONLY=ordered_beta SCREEN_N=10`
sweep (seeds 2001-2010) was run on both worktrees. On main, **all 10 seeds** report
`converged = true`, 7 of them (2002, 2003, 2004, 2006, 2007, 2008, 2010) with FD
gradients from 1.5e3 to 2.2e7 — i.e. 7 false "converged" verdicts, not just seed
2003. On the branch, exactly those same 7 seeds flip to `converged = false`; seeds
2001, 2005 and 2009 (all with FD gradients ~1e-5) stay `converged = true` on both.
So seed 2005 sits in the "genuinely converged" group together with 2001 and 2009,
not the "large-gradient, now-corrected" group together with 2003 — the pattern is
internally consistent and matches the PR's claimed mechanism exactly, it is just
that seed 2005 specifically was not one of the flagged cases. **This looks like a
labeling slip in the task brief** (2005 vs. one of 2004/2006, which do show
gradients of 6.1e6/5.2e6 and do flip main:true → branch:false) rather than a
problem with the fix: the fix's own diagnostic — large-FD-gradient seeds flip
false-positive → correct-negative, small-FD-gradient seeds are untouched — holds
for all 10 seeds tested, 9 of them exactly as the wider pattern predicts and the
10th (2005) as a true negative rather than the expected true positive.

Raw run logs: `ordbeta_screen.jl` (custom 3-seed script, copied verbatim from
`route_ordered_beta`/`finish_row!`) and the `SCREEN_ONLY=ordered_beta SCREEN_N=10`
full-script reruns, both worktrees — outputs captured in this session's transcript;
not separately filed since they reproduce on demand in under a minute.

## 3. `test/test_fit_verdict_gradient.jl` on main vs. branch

Run directly (`julia --project=test test/test_fit_verdict_gradient.jl`), each
worktree instantiated independently (`Pkg.develop(path=".")`, `Pkg.instantiate()`
against `test/Project.toml`); the test file itself was copied from the branch onto
the `origin/main` worktree (it does not exist there — it is new in this PR) so the
same assertions run against main's unfixed `_fit_verdict`.

- **`origin/main`: FAILS as expected.** 3/4 assertions pass; the one failure is
  exactly `@test !fit.converged` (NB1 grouped, seed 101) — `fit.converged` is
  `true` on main, matching the reported defect. The `loglik ≈ -1038.5503750549412`
  assertion in the same testset passes on both, confirming this is the honesty-flag
  defect specifically, not a different regression.
- **Branch (#502 head): PASSES.** 4/4 assertions pass, including the previously-
  failing `!fit.converged` case and the "genuinely stationary fit stays converged"
  guard (seed 703) — confirms the fix does not over-correct a real convergence.

This matches the task's expectation exactly (fail on main, pass on branch) and is a
direct, first-party reproduction of the builder's own claimed test behaviour — not
merely reading the after-task report.

## 4. Other evidence read (not independently re-run)

Read `docs/dev-log/after-task/2026-09-26-fit-verdict-gradient-485.md` on the branch
to cross-reference the builder's own claims against what CI/local reruns show here:

- 15-file/211-assertion targeted sweep: 3 internal verdict flips, all traced and
  classified (1 is this PR's own reported defect; 2 are already independently
  gated by other files' own gradient checks and were `.converged == false` before
  this fix too, for unrelated reasons) — consistent with "old=true→new=false,
  honesty corrections" and with what this session's own ordered-beta sweep found
  (7/10 flips, all old=true→new=false).
- 139-file sweep: reported as inconclusive, killed twice inside
  `test_zero_inflated.jl` (dense per-site Newton solve, ZIP/twopart family) — matches
  the task brief verbatim; not independently re-run here (out of this lens's scope,
  and the after-task report already documents the timing failure mode in detail,
  section 9).
- The R-parity suite (`test/parity/`, RCall-gated) was not run by the builder and is
  not run here either — it is exactly the "Frozen R" advisory CI leg (section 1);
  see that leg's own conclusion above.

## 5. Verdict inputs

- CI: **not resolved** — 8/9 legs still `in_progress` at last observation
  (16 minutes in; historically this repo's full matrix takes 45-70+ minutes),
  0 failures observed, `Documenter` green, `Frozen R` is confirmed
  `continue-on-error` (advisory, per the task's own framing). Cannot yet confirm
  "fully green apart from the known advisory cell."
- Cross-check: matches expectation for seeds 2001 and 2003; does **not** match the
  stated expectation for seed 2005 (converges cleanly instead of flagging
  non-convergence) — but the wider 10-seed sweep shows this is very likely a
  labeling slip in the brief (2005 belongs with the "genuinely converged" group,
  not the "large-gradient" group), not a defect in the fix. The core mechanism the
  PR claims — `Optim.converged` true + large FD gradient on main, correctly
  flipped to `converged = false` on the branch, small-gradient fits left untouched
  — is fully confirmed across all 7 flip cases and all 3 stable cases in the
  10-seed range, on top of the builder's own NB1 evidence and the direct
  `test_fit_verdict_gradient.jl` fail-on-main/pass-on-branch reproduction.
- Test file: fails on main (exactly the reported assertion), passes on branch —
  matches expectation exactly.
