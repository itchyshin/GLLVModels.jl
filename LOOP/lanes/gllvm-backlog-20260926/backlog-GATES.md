# GATES — gllvm-backlog-20260926

SCOPE: GOAL.md of lane gllvm-backlog-20260926. A PR is resolved when MERGED, or open with a PR comment starting 'RETURNED:' that gives the written reason.
OWNS: .unlazy/gllvm-backlog/**

- [x] G0: #494 merged
  CHECK: sh prstate.sh 494
  EXPECT: PR_494_MERGED
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/gllvm-backlog; path=8b01951b15dd/30 entries; output=PR_494_MERGED

- [x] P487: #487 reviewed and resolved (merged, or open with a written RETURNED reason)
  CHECK: sh prstate.sh 487
  EXPECT: /PR_487_(MERGED|OPEN_WITH_REASON)/
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/gllvm-backlog; path=8b01951b15dd/30 entries; output=PR_487_MERGED

- [x] P488: #488 reviewed and resolved (merged, or open with a written RETURNED reason)
  CHECK: sh prstate.sh 488
  EXPECT: /PR_488_(MERGED|OPEN_WITH_REASON)/
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/gllvm-backlog; path=8b01951b15dd/30 entries; output=PR_488_MERGED

- [x] P489: #489 reviewed and resolved (merged, or open with a written RETURNED reason)
  CHECK: sh prstate.sh 489
  EXPECT: /PR_489_(MERGED|OPEN_WITH_REASON)/
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/gllvm-backlog; path=8b01951b15dd/30 entries; output=PR_489_MERGED

- [x] P490: #490 reviewed and resolved (merged, or open with a written RETURNED reason)
  CHECK: sh prstate.sh 490
  EXPECT: /PR_490_(MERGED|OPEN_WITH_REASON)/
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/gllvm-backlog; path=8b01951b15dd/30 entries; output=PR_490_MERGED

- [ ] P491: #491 reviewed and resolved (merged, or open with a written RETURNED reason)
  CHECK: sh prstate.sh 491
  EXPECT: /PR_491_(MERGED|OPEN_WITH_REASON)/
  EVIDENCE: pending

- [ ] P493: #493 reviewed and resolved (merged, or open with a written RETURNED reason)
  CHECK: sh prstate.sh 493
  EXPECT: /PR_493_(MERGED|OPEN_WITH_REASON)/
  EVIDENCE: pending

- [x] P495: #495 reviewed and resolved (merged, or open with a written RETURNED reason)
  CHECK: sh prstate.sh 495
  EXPECT: /PR_495_(MERGED|OPEN_WITH_REASON)/
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/gllvm-backlog; path=8b01951b15dd/30 entries; output=PR_495_MERGED

- [x] F484: a PR that closes #484 exists (open or merged)
  CHECK: gh pr list -R itchyshin/GLLVModels.jl --state all --limit 60 --json number,closingIssuesReferences -q '[.[]|select(any(.closingIssuesReferences[]; .number==484))]|if length>0 then "F484_PR_EXISTS" else "none" end'
  EXPECT: F484_PR_EXISTS
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/gllvm-backlog; path=8b01951b15dd/30 entries; output=F484_PR_EXISTS

- [ ] F485: a PR that closes #485 exists (open or merged)
  CHECK: gh pr list -R itchyshin/GLLVModels.jl --state all --limit 60 --json number,closingIssuesReferences -q '[.[]|select(any(.closingIssuesReferences[]; .number==485))]|if length>0 then "F485_PR_EXISTS" else "none" end'
  EXPECT: F485_PR_EXISTS
  EVIDENCE: pending

- [x] K1: decision packet on main lists #129 and #131
  CHECK: git show origin/main:docs/dev-log/owed/2026-09-25-true-parity-decision-packet.md | grep -c "#129\|#131" | awk '{print ($1>=2)?"K1_MET":"K1_NO"}'
  EXPECT: K1_MET
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/z3437171/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/gllvm-backlog; path=8b01951b15dd/30 entries; output=K1_MET

- [ ] H1: a handover for this lane is on main
  CHECK: git ls-tree -r --name-only origin/main docs/dev-log/handover | grep -q "2026-09-2[67].*backlog" && echo H1_MET
  EXPECT: H1_MET
  EVIDENCE: pending

- [x] R1: Rose audit of the packet and scoreboard recorded (manual; file under LOOP/lanes/gllvm-backlog-20260926/reviews/)
  EVIDENCE: reviews/rose-audit-scoreboard-packet.md (1038 words, 2026-09-26 08:18 MDT, Sonnet high): 32 rows = 19 PARTIAL + 13 OPEN recounted; 11 rows spot-checked incl. every PARTIAL citing a PASS receipt; D8 FORWARD=62 REVERSE=91 reproduced on origin/main; #497 verdict MERGE (merged ef0488df8); #496 MERGE_AFTER_FIXES (#1192 citation). Reviewed by orchestrator (Opus) 2026-09-26.

- [ ] D43: D-43 panel verdict recorded for milestone "backlog landed" (manual)
  EVIDENCE: pending
