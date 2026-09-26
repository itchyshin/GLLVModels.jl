# OVERNIGHT GOAL (2026-09-26 22:15Z to 2026-09-27 11:00Z = 05:00 MDT). Re-read every wave.
Authority: DECISIONS D-290. Merge a silent-failure fix after an independent review (nothing blocking) and green CI (advisory frozen-R cell allowed at 277/9-278/8). WAIT for Shinichi (drafted reply) if a PR changes healthy-fit results, adds public API, touches frozen contract / required-cell / ledger signature, or changes the shared generic `_laplace_mode`.
Rules: <=3 agents live; Sonnet builders and reviewers; Opus only for a load-bearing diagnosis; one family per PR; tests assert relations on literal/hash-verified data; each builder its own LANE_ID and exact-file claims; "Part of #N" for tracker issues, never "Fixes" a tracker; squash merges via merge_train.sh; re-check mergeability after each merge (CHANGELOG conflicts); no writes to gllvmTMB.
Queue (dependency order):
 A1 #503 BetaBinomial kernel      A2 #503 COM-Poisson kernel (+ its FD-gradient kink anomaly)   A3 #501 ordered beta diagnosis (then fix if clear)
 B1 #505 NB1 grouped cov verdict (after #507 merges; same file)   B2 #503 mixed bridge   B3 #503 Tweedie/NB2 grouped: confirm rate, fix if live
 C1 #504 bootstrap adapters, highest-use families first (after #508 merges; confint_family.jl, one family per PR)
 D1 generic `_laplace_mode` (GP1 + Student-t shared): build + Opus review, then WAIT for Shinichi (shared kernel)
Close by 10:30Z: handover + after-task on main, vault log, Mission Control.

## Progress log
- 22:15Z #509 merged 8f0bc97f6 (maintainer's word). 22:4xZ #508 merged 2847b5dbf.
- ~22:35Z usage limit hit again (window already spent before the night); wave A builders killed; reset 22:50Z. NEW CAP: <=2 live.
- 22:55Z #507 rebased (CHANGELOG only; src/test identical; 39/39) -> 9d78698eb, train restarted. Wave A relaunched (wf_29e21376-8a1): betabinom (from commit b7f0886c9), orderedbeta (diagnose first), compoisson (reuse WIP edit).
