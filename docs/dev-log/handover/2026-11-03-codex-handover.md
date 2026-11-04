# Codex handover, 2026-11-03: bootstrap_Sigma_r lane (written by Claude)

You are Codex, picking up the GLLVModels.jl `bootstrap_Sigma_r` true-parity lane from a Claude Code session. You cannot see that chat. This file is the whole handover. `AGENTS.md` is native to you: read it first, then this file, then the files linked below.

Date stamps are 2026-11-03. `origin/main` at write time: `6b35956f7` (merge of #1086) plus later merges; re-read it, do not trust this number.

## 0. The grand goal: true R-Julia parity (P1)

The lane you are in serves one durable aim: **GLLVModels.jl must reproduce R `gllvmTMB` (pinned, read-only, commit `9539352f66f2db2cc26b1c393e67212a359b60c9`) in behaviour, not only in API shape or ledger counts.** "Core070 ledger accounting is not true parity" (see `docs/src/gllvmtmb-parity.md`). Parity is claimed only from measured evidence, row by row, under signed maintainer rulings.

- **Finish line (P1 is NOT met):** C0 to C8 and X2 all MET on `origin/main`; F-capability coverage verified; required negative controls passing on main; an independent completion review; reconciliation; a validated after-task report; and a final handover on main. Do not call P1 done before every one of those holds.
- **Last recorded exact-main measurement (2026-10-28, from `docs/dev-log/handover/2026-11-01-claude-handover.md` on branch `codex/true-parity-p1-restored-20261101`):** C0, C1, C5, C6, C7, C8 MET; C2 290/297; C3 6/8; C4 4/8; X2 304/317; F-capability 1,146/1,146 across five selected suites only (not full-suite, not closure). Re-measure on current main before quoting any of these.
- **Open rows at that measurement:** C2 `FAMILY-00-IDENTITY`, `simulate`/`gllvmTMB_multi`, `animal_latent`, `bootstrap_Sigma`, `extract_phylo_signal`, `POST-COEF-EMPTY`, `POST-SIMULATE-DEFAULT`; C3 `ORDINAL-LOGIT-RSZ`, `ISDM-HEADLINE-RSZ`; C4 `RD-URBANISATION-BINOMIAL`, `RD-SPIDER-NB2`, `RD-BEETLE-NB2`, `RD-FUNGI-BINOMIAL`.
- **What this session moved toward it:** `bootstrap_Sigma_r` now implements the pinned-R `bootstrap_Sigma` behaviour (the maintainer approved the full-behaviour target). That is an implementation checkpoint only. The `bootstrap_Sigma` row stays open until measured evidence supports binding it, and no scoreboard, receipt, case-map or GATES file was changed. N1b (R vs Julia, phy tier) is Inconclusive, so it does not yet count as evidence for a bind.
- **Fitted-comparison rule that makes a row pass:** positive-definite Hessian and max|grad| below 1e-5 on both engines. Signed P1 FAIL rows stay FAIL; the 2026-10-21 limits in `docs/dev-log/decisions/2026-10-21-p1-optimizer-stopping-rule.md` stand. Usability is never traded away for an accuracy gain (D-139).
- **Where truth lives:** the pending board `docs/dev-log/2026-09-14-true-parity-pending-board.md`, the decision map `docs/dev-log/core070/true-parity-decision-map.md`, the scoreboard and receipts under `docs/dev-log/core070/true-parity-latest/`, and the P1 candidate branch `codex/true-parity-p1-current-main-20261024` (Codex-owned, protected; do not push to it without its owner). The `AGENTS.md` P1 snapshot is stale; many lanes are live, so verify PRs, branches and worktrees before acting.
- **Order of leverage toward the goal:** (1) unblock the receipt chain (#1046, #1055, #1063) via the `negative-controls` Julia fix; (2) the open C2 rulings (simulate, `extract_phylo_signal`, POST-COEF-EMPTY, POST-SIMULATE-DEFAULT) with the maintainer; (3) N1c on 1101 so `bootstrap_Sigma` can eventually be bound; (4) C3 and C4 rows only through new maintainer rulings, never by refits of spider or beetle; (5) fresh exact-main measurement, completion panel, reconciliation and final handover.

## 1. Critical context

- **Goal:** implement the pinned-R `bootstrap_Sigma` behaviour as `bootstrap_Sigma_r` and move other open true-parity rows only through maintainer rulings. Pinned R `gllvmTMB` is `9539352f66f2db2cc26b1c393e67212a359b60c9`, read-only: never modify it.
- **Standing constraints (all still binding):**
  - Never touch or merge PR #990. Never touch `src/confint_derived_wald.jl` (the maintainer said "sign, wait for #1036"). Never touch Codex or other lanes' branches without the owner's word.
  - No tolerance widening. Signed FAIL rows stay FAIL. Diagnostic evidence is never a pass. No row status, scoreboard, receipt, case-map or GATES change from this lane.
  - Never refit spider or beetle. No urbanisation raw data. C2 rows stay UNBOUND with packets.
  - Estimate before any run; over 3 h needs a plan and approval (D-287). Totoro at most 150 cores total (this lane used at most 14, usually 4), one thread per process, existing ControlMaster only (`ssh -o BatchMode=yes totoro`; julia at `/home/snakagaw/.juliaup/bin/julia`). Never compute on a DRAC login node. No GitHub Actions for campaigns.
  - Stage by name, never `git add -A`. Commits end `Co-Authored-By:` with the model that wrote them. PR bodies: no @handles, no fix/close/resolve words near issue or PR numbers (a wrong-close guard fails the PR).
  - No em dashes in shipped prose; run `python3 ~/shinichi-brain/tools/slop_check.py <absolute path>`.
  - Never run `git merge` or rebase in the main checkout; confirm `pwd` and `git branch --show-current` before any git write.
- **Maintainer rulings in force:** N9 covers only ordinal, iSDM, fungi, spider. Ordinal Newton polish is diagnostic only. Beetle, fungi, urbanisation guarded in SIGNED_FAIL_ROWS only. No FAMILY-00 oracle rebuild. Auto-merge on green is allowed; push your own clean branches cut from `origin/main`.

## 2. What was accomplished (all merged to main)

- `src/bootstrap_sigma_r.jl`: `bootstrap_Sigma_r(fit; n_boot=999, conf=0.95, seed, n_cores, ...)` returning `BootstrapSigmaR`. Supported: GllvmFit (tier B), TwoLevelFit (B, W, ICC_site), PrecisionMultivariateFit (`:phy` only), JointPhyloGroupedGaussianFit (`:phy` only). Everything else is refused with `ArgumentError`. PRs #1054, #1056, #1059, #1069, #1073, #1076, #1081, #1083.
- Hardening: group-term guard for JPG, threaded `n_cores` with bit-identical draws and Ctrl-C handling, warn-once lock in `src/fit.jl` (#1083), CHANGELOG and docs (#1077, #1085).
- #1091: opt-in sufficient-statistic refit (`_sufficient_stat`, default OFF, internal). Verified: default-OFF identical to main across 93 test files; objective matches a dense oracle to 1.1e-13.
- #1086: CI gives every shard 4 threads (`JULIA_NUM_THREADS`) so the threaded testset runs in CI; `test/runtests.jl` guard `GLLVM_TEST_REQUIRE_THREADS`. (Shard of the last include moves whenever a test file is added; that was the bug in the first version.)
- #1093: N7/N11/N12 threaded tests made robust (locked logger, interrupt waits for first refit, `GC.safepoint()` in the hammer loop).
- #1087 (joint-fit phylo-signal extractor, point estimates only) and #1088 (POST-COEF-EMPTY option B) merged. The extractor row stays REGISTRATION-ONLY / UNBOUND.
- N1b paired R-vs-Julia phy-tier comparison run and signed by delegation: see section 4.

## 3. Current working state

| Item | State |
|---|---|
| `bootstrap_Sigma_r` on main | working, merged |
| #1046 (zi receipt fingerprint refresh, ready, not merged) | blocked: `negative-controls` also fails because two tests in `tools/test_core070_behaviour_receipts.py` call `shutil.which("julia")` and the job has no Julia. Fingerprint fix itself is verified (`OK 3 receipts match`). Duplicate draft #1065 exists. |
| #1055 (N9 guard), #1063 (POST-COEF-EMPTY bookkeeping, draft) | blocked on #1046 / #1065 |
| #1078 (N1 prereg, draft, unsigned) and #1092 (N1 diagnosis + N1b prereg + N1b result, draft) | open drafts, evidence only |
| #1036 | open draft, hold; SIGINT patch on branch `claude/sigint-twolevel` is held |
| goal-meter plugin | not loaded in the Claude session; user must run `/reload-plugins` (irrelevant to Codex) |

## 4. N1b result (evidence only, no row moves)

Result note: `docs/dev-log/check-log.d/2026-11-03-bootstrap-sigma-phy-n1b-result.md` on branch `claude/n1b-draft-20261103` (commit `cef86b337`, in draft PR #1092). Prereg: `docs/dev-log/plans/2026-11-03-bootstrap-sigma-phy-n1b-prereg-DRAFT.md` (Section 10 records approval by the maintainer's chat delegation, not a personal signature).

Overall **Inconclusive**: 1104, 1205, 1206 Agree; 1101 Inconclusive because a communality absolute rule fails (0.088 against 0.05 at comm4 lower, R 0.138 vs Julia 0.050). Paired max z 0.029 to 0.339, SE ratios 0.954 to 1.082, nothing near Disagree. Simulator-moment check passed (worst 2.15 MC SE vs 4; within-row and within-species covariances only). About 0.9 core-hours. Caveats: seeds 1201/1202 were boundary fits and were replaced by 1205/1206; the wrapper sentence in the prereg was ambiguous and the agent's reading is stated in the note; the own-interval-bounds check was NOT RUN; on 1101 the largest logLik gap between engines is 13.77 (not investigated).

## 5. Key decisions and rationale

- Refit objective in the built-in GllvmFit path uses `iterations=2000, f_tol=1e-12` only when `has_diag`; a deliberate deviation from R, documented in the ledger `docs/dev-log/plans/2026-11-01-bootstrap-sigma-checkpoint-ledger.md` (items 18 to 28).
- Sufficient-statistic refit stays default OFF: draws move at rounding level, so ON would need a maintainer decision.
- Thresholds in the N1b prereg were fixed before the run; do not change them after reading results.
- A failing CI check is never bypassed; use `~/shinichi-brain/tools/pr_merge_when_green.sh itchyshin/GLLVModels.jl <PR>` (takes owner/repo first).

## 6. Files created or modified this session

Landed on main (see `git log origin/main --oneline`): `src/bootstrap_sigma_r.jl`, `src/fit.jl`, `src/profile.jl`, `src/GLLVModels.jl`, `src/joint_phylo_grouped_postfit.jl`, `test/test_bootstrap_sigma_r.jl`, `test/test_extract_phylo_signal_joint.jl`, `test/runtests.jl`, `.github/workflows/CI.yml`, `CHANGELOG.md`, `docs/make.jl`, `docs/src/bootstrap-sigma-r-example.md`, `docs/src/bootstrap-sigma-r-phy-example.md`, `docs/src/se-profile-machinery.md`, the ledger above, and the `docs/dev-log/check-log.d/2026-11-02-bootstrap-sigma-*.md` notes.

Not on main (declared below): `docs/dev-log/check-log.d/2026-11-03-bootstrap-sigma-phy-n1-diagnosis.md`, the N1b prereg and result, and `2026-11-03-bootstrap-sigma-phy-n1b-scripts/` (all on `claude/n1b-draft-20261103`, PR #1092).

This handover writes: `docs/dev-log/handover/2026-11-03-codex-handover.md` and one bullet in the `AGENTS.md` Phase state snapshot.

## 7. Next immediate steps (classify OWED / DONE / RETRACTED / PROTECTED on arrival)

1. **OWED, Codex or Claude, needs a maintainer yes:** add a Julia setup step (`julia-actions/setup-julia@v2`) to the `negative-controls` job in `.github/workflows/true-parity-check.yml`, in a small PR from `origin/main`. The workflow and the failing tests belong to the Codex true-parity lane, so the maintainer's yes is needed; the tests may also need a package install, so run them in CI first. Then #1046 (or #1065), #1055, #1063 can land in that order.
2. **OWED, maintainer decision:** N1c, a short pre-registered follow-up on dataset 1101 only (logLik gap 13.77, communality boundary rule). R refits are cheap (85 to 110 s per 999 batch). Do not start without a yes.
3. **OWED, maintainer:** sign or release #1078 (N1) and #1092 (N1b); decide #1091 default ON or OFF; C2 rulings (simulate options A/C wording, E2 rule and R seed, Poisson `condition_on_RE` branch push, which of #1046/#1065 merges first, POST-COEF-EMPTY signature); #1036 hold interpretation.
4. **PROTECTED:** #990, `src/confint_derived_wald.jl`, `claude/sigint-twolevel`, all Codex branches (including `codex/true-parity-p1-restored-20261101`), spider and beetle refits.
5. **Live-toolchain work for Codex** (Claude had no standing slot for it): full `Pkg.test()` on `origin/main` at `-t 4` and `-t 1`; a fresh full-suite check after #1086's thread change, because shard 3 `test_convergence_sweep.jl:108` (`maximum(abs, fita.Λ) > 8`) failed once on #1093's CI and was rerun; if it recurs, check whether 4 threads change that ordinal fit.

## 8. Blockers and open questions

- Maintainer yes/no on item 1 and item 2 above.
- Goal-meter display is a Claude-side plugin issue only.
- The vault `~/shinichi-brain/memory/DECISIONS.md` was emptied once by a subagent and restored from backup; another lane's uncommitted D-342 should be confirmed intact by its owner.

## 9. Gotchas and failed approaches

- A failed `cd` into a scratch worktree once ran `git merge` in the maintainer's main checkout; always chain `cd "$W" && pwd && git branch --show-current` and stop if it fails.
- `agent_mention_check.py --text` treats its argument as a file path; check PR bodies by eye.
- A stale failed guard run can block `pr_merge_when_green.sh`; fix the PR body, let the guard rerun.
- Classifier denied touching another lane's PR (`gh pr ready`, `gh pr update-branch`, pushes to #1046's branch) until the maintainer ran it; do not work around such denials.
- The scratchpad is purged between sessions; re-commit anything you want to keep.
- Julia 1.10.0's `Test.TestLogger` has no lock, so concurrent log capture loses messages; #1093 wraps it.

## 10. Landing state

| Branch / item | State | Why | Resume |
|---|---|---|---|
| main (merged PRs above) | LANDED | | |
| `claude/n1b-draft-20261103` (PR #1092, draft) | CARRIED-OVER | evidence notes awaiting maintainer sign-off | `git fetch origin claude/n1b-draft-20261103` |
| `claude/zi-receipt-fingerprints` (#1046, ready) | CARRIED-OVER | `negative-controls` needs a Julia setup (item 1) | gate: `~/shinichi-brain/tools/pr_merge_when_green.sh itchyshin/GLLVModels.jl 1046` |
| `claude/sigint-twolevel` | CARRIED-OVER | held pending #1036 | none |
| `handover/2026-11-03-codex` (this file) | pushed with PR, not merged | the maintainer merges | |

`handoff_gate.sh` exits non-zero in the main checkout because 33 acceptance ledgers under `.unlazy/` belong to other, older lanes (latte kernel, S9c coverage and similar); none belong to this lane. They are declared here, not fixed.

FINDINGS-OF-RECORD: N1b Inconclusive (3 of 4 Agree, 1101 fails a communality absolute rule, no Disagree); canonical source is branch `claude/n1b-draft-20261103` commit `cef86b337`. vault-note: none written yet; the repo note is the record.

Other lanes: see the coordination board's Active-Lane-Split (`docs/dev-log/coordination-board.md`); their rolling handovers are not carried here and are not superseded by this file.

## 11. How to resume

Live-toolchain exports (Codex runs these):

```sh
export PATH="$HOME/.juliaup/bin:$PATH"
cd "/Users/z3437171/Dropbox/Github Local/GLLVM.jl"
~/shinichi-brain/tools/lane_preflight.sh GLLVM.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -t 4 --project=. -e 'using Pkg; Pkg.test()'
```

Paste into Codex, from the repo root:

> Read AGENTS.md and docs/dev-log/handover/2026-11-03-codex-handover.md. Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps.

Mission-control summary:

| Repo | main | CI | Shipped | Plan by leverage |
|---|---|---|---|---|
| GLLVModels.jl | `6b35956f7` and later | green on merged PRs | `bootstrap_Sigma_r` for 4 fit classes, threaded CI, robust tests, N1b run | 1 Julia setup in negative-controls, 2 N1c on 1101, 3 maintainer sign-offs |

Claude plans, writes prose and gates merges; Codex runs the live Julia/R/Totoro toolchain.
