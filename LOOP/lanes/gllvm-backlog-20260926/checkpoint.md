GOAL: see GOAL.md.   STATE: IN PROGRESS, wave 1 running (workflow wf_2f0e7019-416, started 2026-09-26 ~13:05Z).
ARCS DONE (verified): S0 #494 MERGED d4da31544 (latest check per name green on 60105df5b; advisory 277/9 = main's own range).
ARC IN PROGRESS: S1 recon (Haiku) -> S2 Opus review #491/#493 + S3 Sonnet review #487-#490/#495 (reports in reviews/pr-N.md);
  S4 #484 builder (~/local-scratch/gllvm-twopart-484), S5 #485 builder (~/local-scratch/gllvm-verdict-485),
  S6 reverse-gap (~/local-scratch/gllvm-reverse-gap) + packet (~/local-scratch/gllvm-packet).
NEXT: read verdicts; apply fixes; merge train (S7) one PR at a time, head-pinned.
OPEN GATES (need human): #491 and #493 merges.
LEDGERS: run ledger .unlazy/gllvm-backlog/GATES.md (1/14 met after --reverify; copy backlog-GATES.md); true-parity ledger .unlazy/true-parity/GATES.md (0/10; copy true-parity-ledger/).
NEW PRs from wave 1 so far: #496 reverse-gap classes (draft), #497 decision packet update (draft).
TRUTH LIVES IN: branch claude/lane-gllvm-backlog-20260926 (this kit); reviews/; PRs on itchyshin/GLLVModels.jl.
RESUME: read GOAL.md -> checkpoint.md -> ultra-plan.md; if wf_2f0e7019-416 finished, read its journal.jsonl; continue from NEXT.
