# GOAL — true-parity-20260924 (IMMUTABLE — re-read at the top of EVERY arc)
Read this first, every cycle. Auto-compact eats messages, not this file. Unsure after a compaction?
Re-read THIS, then checkpoint.md, then continue.

## Mission
Execute all four maintainer-pasted gates of GLLVModels.jl docs/dev-log/handover/2026-09-24-claude-handover.md,
each to its runbook's stop line: #409 S4 probe receipt; #410 Totoro #323 Track A receipt; #399 Delta
dispersion A merged + D1 remeasured; #411 Stage 1 harness merged + Stage 1 slice PR. Programme docs state
the live truth. After-task report + handover.

## Headline
The four DRAFTs branch from 7a6fe4962, BEFORE the GLLVM -> GLLVModels rename (#423, 69a69b0a0). Each needs
rebase + conflict fix + rename port + fresh CI (~75 min) before it can merge. Everything hangs on that.

## Pastes received (Shinichi, 2026-09-24, in chat, verbatim)
`S4 probe yes` · `G0 Stage 1` · `ack Totoro D-139 #323 Track A` · `accept delta dispersion A`
Plus: "Yes, all four as DRAFT" (pre-rebase now) · D-280 trial "Defer to next clean plan" ·
Totoro executor "Claude runs it here".

## Invariants
- One Claude lane in GLLVM.jl. Never touch src/grouped_nongaussian_fit.jl (Cursor lease), #444/#433/#437/#439/#463.
- gllvmTMB is READ-ONLY (D-220). No Project.toml bump, tag, or release.
- Never widen a tolerance. Never `git add -A`. Never write in the Dropbox checkout.
- runtests.jl is position-sharded: follow ci-shard-suite on every rebase (parse, count includes).
- ≤2 Julia processes at once on the Mac, JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1.
- Merge order: #409 -> #410 -> #399 -> #411 harness -> Stage 1 slice PR. After each merge: merge-tree the rest, watch main CI.
- Fences: S4 probe != parity · Track A != programme complete · Stage 1 != full R grid · Delta A accepted != D1 pass.
- No agent persona as @handle on GitHub (run agent_mention_check.py on PR bodies).

## Authoritative WHAT
LOOP/lanes/true-parity-20260924/ultra-plan.md (detail wins there; this file wins on what must never be lost).
Ledger: .unlazy/true-parity-20260924/ (git-ignored) in this worktree.

## Definition of done
gate-check --reverify exits 0 on every leaf (or ABANDON with reason); #409 #410 #399 #411 + Stage 1 slice
MERGED; main CI green at final tip; S4 / Track A / D1 receipts on main; D-43 panel no BLOCKING; after-task,
handover, Melissa plan-actual written; vault AGENT_LOG + D-280 deferral recorded.

## Pre-authorisation
Worktrees under ~/local-scratch/lanes/; leases; scoped edits on the 4 DRAFT branches, docs branch, Stage 1
slice branch; local Julia tests; scratch R lib with frozen gllvmTMB b4d5fee64; local commits; ledger checks;
push --force-with-lease the 4 DRAFTs; push new branches; open PRs; ready+merge the six PRs when Julia CI green;
ssh Totoro via existing socket for the #410 pre-run.

## Must stop for
Totoro FULL run (after pre-run shown, D-139) · D1 still FAIL after alignment · any tolerance change · red main
CI after a merge · an API choice the runbook does not fix · any gllvmTMB write · Project.toml/tag/release ·
a foreign-lane file · compute beyond estimates · S4 probe estimate > 30 min.
