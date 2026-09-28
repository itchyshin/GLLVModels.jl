# Morning report, 2026-09-28 (overnight run 00:10Z to about 11:00Z)

Goal: GOAL-2026-09-28-overnight.md. Written at the start and updated as arcs finish.

## Needs you
1. Review and merge #519 (green), then #521 (green), then #540 (green).
2. Approve the NB re-run estimate (about 5,500 core-h; nb-rerun-plan.md). Not submitted.
3. Julia vs R guard difference under the ridge (A1 below): which Hessian rule should both use?

## Arcs
- A0 (#518 CI): DONE. CI on `159631a5b`: all 8 test shards, Documenter and the P1 twin tests pass; only the advisory smoke job fails (as on `main`). Held commits pushed together with A1 (see A4).
- A1 (Julia vs R ridge on the same data): DONE. gllvmTMB fitted the 40 exact Julia datasets. With the
  ridge, recovery now agrees within one dataset in most cells (`bic`: Julia 1/8/8/4 vs R 1/7/8/3 out of
  10). **Last night's 8/10 vs 1/10 gap came from the two languages drawing different random datasets, not
  from the engines.** One real difference remains: at n = 120, p = 20, K = 3 (`bic_sites`, ridge) Julia
  recovers 9/10 and R 4/10, because R's guard rejects ridge fits whose Hessian is non-positive-definite
  ("unconverged") and Julia's guard does not check the Hessian. **Decision for you:** should the Julia
  guard adopt R's Hessian check, or R relax it under the ridge? Recorded in design/74 T7.
- A2 (Gaussian re-run for #519): DONE, and done in full rather than just planned, because Gaussian
  fits take seconds. On #518 + #519 in a throwaway tree, all 4,800 Gaussian datasets on uncentred data
  (trait means 3 + N(0, 1)): exact recovery **0.950** with `:bic_sites` (original grid 0.948) and 0.864
  with `:bic` (grid 0.865), 0 failures. So the Gaussian claim in #518 holds with trait intercepts; after
  #519 merges this only needs a confirming re-run on `main`. Recorded in design/74 (local commit
  `dd383ba22`).
- A3 (CI watch): RUNNING
- A4 (this report, checkpoint, final push): PENDING
