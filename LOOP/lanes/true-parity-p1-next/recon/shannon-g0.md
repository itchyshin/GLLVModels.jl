## Shannon’s G0 reconnaissance

**Result:** I confirmed the pinned source and lane status, ran the canonical gates sequentially against `origin/main`, and located the existing P1 oracle. No files were edited, no fits were run, and no new parity claim or signed classification was made.

### Lane and pin

- Branch: `codex/true-parity-p1-next`. HEAD is `17f98582b999698d6f35d56a68ef594ee6ae38bb`, the lane scaffold commit; `origin/main` is ahead-count baseline **not** the current HEAD. The plan records baseline `f220379d0937d0afffc6a030023c63c0f715e168` and P1 source `9539352f66f2db2cc26b1c393e67212a359b60c9`.
- Project version at `origin/main`: **0.3.0**.
- Pre-existing worktree changes include `.claude/settings.json`, `.cursor/hooks.json`, and the `LOOP/lanes/true-parity-p1-next/` goal, checkpoint, arcs, and plan files. I left them intact.
- I read `LOOP/lanes/true-parity-p1-next/GOAL.md`, `ultra-plan.md`, and `docs/dev-log/handover/2026-10-06-codex-handover.md`. The seven checkpoint IDs and scope are in the plan; the handover documents earlier main state and must not replace this live measurement.

### Canonical gate run

I ran:

```sh
for c in C0 C1 C2 C3 C4 C5 C6 C7 C8 X2; do
  PARITY_REF=origin/main node tools/true_parity_check.mjs "$c"
done
```

Results, in order:

- C0 **MET**
- C1 **MET**: 24 required rows, 20 numeric and 4 behavioural bound
- C2 **NOT MET**: 283/297; all seven checkpoint IDs are not done
- C3 **NOT MET**: 6/8
- C4 **NOT MET**: 4/8
- C5 **MET**: 4/4
- C6 **NOT MET**: 378 items; 37 undecided names remain, plus invalid null decisions
- C7 **MET**
- C8 **MET**: 38 rows, no failures
- X2 **NOT MET**: 297/317

All 10 commands completed. The checker’s C6 is reading tracked decisions; I did not run the Julia classifier. `gh pr view 715 --json ...` failed with “error connecting to api.github.com,” so PR metadata was unavailable.

### P1 oracle and bridge metadata

- Mac P1 R library exists at `$HOME/local-scratch/gllvmTMB-p1-scratch/Rlib/gllvmTMB`.
- The handover names the P1 oracle build selector `GLLVM_PARITY_ORACLE_BUILD=totoro`, receipt `docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build-totoro.json`, and bridge environment `~/hsq_work/gllvm-w2-3`.
- The existing Totoro ControlMaster socket is present at `~/.ssh/cm-snakagaw@totoro.biology.ualberta.ca:22`. I checked filesystem metadata only; no remote command/login was opened.
- Existing FAMILY-11 raw artifacts are under `docs/dev-log/core070/true-parity-latest/receipts/family/runparity-truncated-nb2/`. Existing iSDM batch artifacts are under `.../receipts/isdm/isdm-p1/`. Existing postfit surface evidence includes `.../receipts/postfit/surface-conversion-p1/` and the `CORE070-WAVE7-CHECK-AUTO-RESIDUAL` case receipt.

### First-seven runner pointers and risks

Receipt entrypoints are `tools/core070_family_p1_receipts.py`, `tools/core070_postfit_p1_receipts.py`, and `tools/core070_isdm_p1_receipts.py`. The family tool documents `test/parity/runparity.jl` for numeric family cases and the existing truncated-NB2 fixture; postfit’s `check_auto_residual` row is handled by the postfit receipt flow; iSDM has an existing R-side batch fixture plus Julia public-door work described in the handover. The receipt tools expect run artifacts tied to a clean commit, so the current dirty lane cannot be used directly to write receipts.

First-seven invocation path: run the owning receipt checks/runner for FAMILY-11, then `check_auto_residual`, then the five iSDM public-door cases **sequentially**, using the plan’s clean owned worktree and P1 bridge settings; only then have the parent perform shared case-map assembly and gate remeasurement. Do not launch the existing whole-batch generators as a shortcut: those tools may rewrite shared receipt directories and case maps.

**Runtime risk:** the handover reports the truncated-NB2 public bridge refusal must run on Totoro; JuliaCall requires the documented library preload and a clean `LD_LIBRARY_PATH` for Julia children. It also flags campaign work for realistic-size checks. This reconnaissance did not run those jobs or estimate their run times.

**Next action:** have the parent review this evidence and assign the first owned clean-worktree invocation.