# Handover: GLLVModels.jl evening session, 2026-09-27

From: Claude (GLLVM.jl folder session that took over the auto-d, #519/#520 and #521 lanes at 16:45Z).
To: the next Claude session on GLLVModels.jl. Nothing here is merged; every PR is a draft and needs
Shinichi's sign-off.

## Where things stand

| PR | branch | head | state |
|---|---|---|---|
| #518 auto-d (Julia) | `claude/lane-auto-d-20260926` | see `git log` | Ridge review fixes, NB figure withdrawn, ridge evidence, merged `main` (a CHANGELOG conflict had blocked CI). CI on `0d3283fdb`: 8/8 shards green, advisory smoke = `main`'s. |
| gllvmTMB #1324 auto-d (R) | `claude/lane-auto-d-r-20260926` | `edd90850f` | NB figures withdrawn; reader-surface IDs removed; ledger rows MS-03 to MS-06 mapped. CI re-running. |
| #519 Gaussian intercepts | `claude/gaussian-intercept-20260927` | `f8e06f97c` | 8/8 shards green; advisory smoke = `main`'s. Ready for sign-off. |
| #520 Gaussian formula slopes | stacked on #519 | `8c7f483e5` | After #519 merges: `gh pr edit 520 --base main`. |
| #521 NB2 grouped kernel | `claude/nb-grouped-init-v2` | `a2c5b5cb7` | Tests registered, CHANGELOG, review done (nothing blocking); 8/8 shards green. Documenter fail is inherited from `main` (fixed there in #530). |
| #529 getLV grouped modes | `claude/getlv-grouped-mode-20260927`, stacked on #521 | `2420bfa56` | After #521 merges: `gh pr edit 529 --base main`, then merge `main` into it. |
| #540 Beta grouped kernel | `claude/beta-grouped-mode-search-503` | `583acb6de` | Damped search + max(observed, Fisher) fallback; plateau-triggered #480 restart (Shinichi's choice). CI running. |

## Decisions made today (record)

- #529 is stacked on #521 (Shinichi, 2026-09-27).
- #540: the #480 restart also fires when a group precision exceeds 100x the median (Shinichi chose
  this over always-restart or parking).

## Next steps

1. Watch CI on #518, #1324, #540; fix anything new (last known: only inherited or advisory failures).
2. After merges: #520 and #529 retarget; rebase or merge `main` into #518.
3. NB grid re-run: plan, pre-run test and submission kit in
   `LOOP/lanes/auto-d-20260926/nb-rerun-plan.md`. Pre-run: the new kernel is 2.2x slower overall
   (3.15x on the heaviest cell) and on 300 x 20, K = 3 it chooses the true K = 3 where the old
   kernel chose 5. Estimate about 3,800 to 5,500 core-h; kit sized at 5,500 (943 tasks,
   `--time=11:00:00`). Needs #521 merged and Shinichi's approval (D-287) before any DRAC submission.
4. Gaussian grid cells re-run on uncentred data after #519 merges.
5. Follow-ups, not owned by this lane (other lanes are active on truncated NB2): the truncated-NB2
   per-trait objective (`truncated_nbinom2.jl:323`) and the root-cause fix inside
   `_grouped_laplace_mode`; Beta `getLV` dispatch after #529 and #540 both merge.

## Open questions for Shinichi

- Sign-off order suggestion: #519, #521, then #520, #529, #540, #518/#1324.
- NB re-run estimate approval (see the plan file).
- Julia vs R binary-ridge gap at n = 120, p = 10 (8/10 vs 1/10 under the same criterion, different
  random datasets): worth a matched-data check?

## Gotchas

- Pushing any commit to a PR restarts its full CI (paths-ignore is judged on the whole PR diff).
- `destructive_command_guard` blocks `git checkout --ours`; for a throwaway combined tree, copy the one
  changed file with `git show <ref>:<path> > <path>` instead of merging.
- `agent_mention_check.py --text <file>` takes a path.
- A kernel fix can move L-BFGS to a different stationary point; evaluate the old optimum under the new
  objective before calling it a regression (#521, #540 d05).
